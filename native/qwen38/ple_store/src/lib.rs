//! Native read-only PLE row storage C ABI.

#![cfg(target_os = "linux")]

mod io;

use std::cell::RefCell;
use std::collections::HashMap;
use std::ffi::{c_char, c_int, c_void, CStr, CString};
use std::fs::{File, OpenOptions};
use std::mem::{align_of, size_of};
use std::os::fd::AsRawFd;
use std::os::unix::ffi::OsStrExt;
use std::os::unix::fs::OpenOptionsExt;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::Path;
use std::ptr;
use std::sync::Mutex;

use crate::io::{PageRead, Reader, PAGE_SIZE};

const STATS_LEN: usize = 8;

thread_local! {
    static LAST_ERROR: RefCell<CString> = RefCell::new(CString::default());
}

fn set_last_error(message: impl AsRef<str>) {
    let sanitized = message.as_ref().replace('\0', "\\0");
    let value = CString::new(sanitized).unwrap_or_else(|_| CString::default());
    let _ = LAST_ERROR.try_with(|slot| {
        if let Ok(mut slot) = slot.try_borrow_mut() {
            *slot = value;
        }
    });
}

fn clear_last_error() {
    let _ = LAST_ERROR.try_with(|slot| {
        if let Ok(mut slot) = slot.try_borrow_mut() {
            *slot = CString::default();
        }
    });
}

fn panic_message(payload: Box<dyn std::any::Any + Send>) -> String {
    if let Some(message) = payload.downcast_ref::<&str>() {
        format!("panic caught at C ABI boundary: {message}")
    } else if let Some(message) = payload.downcast_ref::<String>() {
        format!("panic caught at C ABI boundary: {message}")
    } else {
        "panic caught at C ABI boundary".to_owned()
    }
}

fn validate_array<T>(ptr: *const T, count: usize, name: &str) -> Result<(), String> {
    if count == 0 {
        return Ok(());
    }
    if ptr.is_null() {
        return Err(format!("{name} must not be null when count is nonzero"));
    }
    if !(ptr as usize).is_multiple_of(align_of::<T>()) {
        return Err(format!("{name} is not aligned for its element type"));
    }
    let bytes = count
        .checked_mul(size_of::<T>())
        .ok_or_else(|| format!("{name} byte length overflows usize"))?;
    if bytes > isize::MAX as usize {
        return Err(format!("{name} byte length exceeds isize::MAX"));
    }
    (ptr as usize)
        .checked_add(bytes)
        .ok_or_else(|| format!("{name} address range overflows"))?;
    Ok(())
}

fn validate_output(ptr: *mut u8, bytes: usize, name: &str) -> Result<(), String> {
    validate_array(ptr.cast_const(), bytes, name)
}

fn ranges_overlap(a: *const u8, a_bytes: usize, b: *const u8, b_bytes: usize) -> bool {
    if a_bytes == 0 || b_bytes == 0 {
        return false;
    }
    let a_start = a as usize;
    let b_start = b as usize;
    // Both address additions were already checked by validate_array.
    let a_end = a_start + a_bytes;
    let b_end = b_start + b_bytes;
    a_start < b_end && b_start < a_end
}

#[derive(Debug)]
struct Shard {
    file: File,
    data_offset: u64,
    row_count: u64,
    start_row: u64,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct PageKey {
    shard_index: usize,
    offset: u64,
}

struct CacheEntry {
    data: Box<[u8]>,
    older: Option<u64>,
    newer: Option<u64>,
}

struct RowCache {
    capacity_rows: usize,
    row_bytes: usize,
    entries: HashMap<u64, CacheEntry>,
    oldest: Option<u64>,
    newest: Option<u64>,
}

impl RowCache {
    fn new(cache_bytes: usize, row_bytes: usize) -> Self {
        Self {
            capacity_rows: cache_bytes / row_bytes,
            row_bytes,
            entries: HashMap::new(),
            oldest: None,
            newest: None,
        }
    }

    fn contains(&self, row_id: u64) -> bool {
        self.entries.contains_key(&row_id)
    }

    fn get(&self, row_id: u64) -> Option<&[u8]> {
        self.entries.get(&row_id).map(|entry| entry.data.as_ref())
    }

    fn get_with_newer(&self, row_id: u64) -> Option<(&[u8], Option<u64>)> {
        self.entries
            .get(&row_id)
            .map(|entry| (entry.data.as_ref(), entry.newer))
    }

    fn remove(&mut self, row_id: u64) -> Option<Box<[u8]>> {
        let entry = self.entries.remove(&row_id)?;
        if let Some(older) = entry.older {
            self.entries
                .get_mut(&older)
                .expect("LRU older link must reference a resident row")
                .newer = entry.newer;
        } else {
            self.oldest = entry.newer;
        }
        if let Some(newer) = entry.newer {
            self.entries
                .get_mut(&newer)
                .expect("LRU newer link must reference a resident row")
                .older = entry.older;
        } else {
            self.newest = entry.older;
        }
        Some(entry.data)
    }

    fn touch(&mut self, row_id: u64) -> bool {
        let Some(entry) = self.entries.get(&row_id) else {
            return false;
        };
        if self.newest == Some(row_id) {
            return true;
        }
        let older = entry.older;
        let newer = entry
            .newer
            .expect("a non-newest LRU entry must have a newer link");
        if let Some(older) = older {
            self.entries
                .get_mut(&older)
                .expect("LRU older link must reference a resident row")
                .newer = Some(newer);
        } else {
            self.oldest = Some(newer);
        }
        self.entries
            .get_mut(&newer)
            .expect("LRU newer link must reference a resident row")
            .older = older;

        let previous_newest = self.newest.expect("a nonempty LRU must have a newest row");
        self.entries
            .get_mut(&previous_newest)
            .expect("LRU tail must reference a resident row")
            .newer = Some(row_id);
        let entry = self
            .entries
            .get_mut(&row_id)
            .expect("touched LRU row must remain resident");
        entry.older = Some(previous_newest);
        entry.newer = None;
        self.newest = Some(row_id);
        true
    }

    fn insert_copy(&mut self, row_id: u64, source: &[u8]) {
        debug_assert_eq!(source.len(), self.row_bytes);
        if self.capacity_rows == 0 {
            return;
        }
        debug_assert!(!self.entries.contains_key(&row_id));
        while self.entries.len() >= self.capacity_rows {
            if self.pop_oldest().is_none() {
                break;
            }
        }
        // Evict before allocating so cache-owned payload never exceeds the
        // configured whole-row budget, even transiently.
        let data = source.to_vec().into_boxed_slice();
        let older = self.newest;
        self.entries.insert(
            row_id,
            CacheEntry {
                data,
                older,
                newer: None,
            },
        );
        if let Some(older) = older {
            self.entries
                .get_mut(&older)
                .expect("LRU tail must reference a resident row")
                .newer = Some(row_id);
        } else {
            self.oldest = Some(row_id);
        }
        self.newest = Some(row_id);
    }

    fn pop_oldest(&mut self) -> Option<Box<[u8]>> {
        self.remove(self.oldest?)
    }

    fn resident_rows(&self) -> usize {
        self.entries.len()
    }

    fn resident_bytes(&self) -> usize {
        self.entries.len() * self.row_bytes
    }
}

#[derive(Default)]
struct Counters {
    read_calls: u64,
    requested_rows: u64,
    cache_hits: u64,
    cache_misses: u64,
    io_pages_read: u64,
    io_bytes_read: u64,
}

struct Store {
    shards: Vec<Shard>,
    total_rows: u64,
    row_bytes: usize,
    queue_depth: usize,
    max_batch_pages: usize,
    reader: Reader,
    cache: RowCache,
    counters: Counters,
}

struct Handle {
    store: Mutex<Store>,
}

#[derive(Clone, Copy)]
struct RowLocation {
    shard_index: usize,
    byte_start: u64,
}

#[derive(Clone, Copy)]
struct PageCopy {
    missing_slot: usize,
    page_offset: usize,
    row_offset: usize,
    len: usize,
}

impl Store {
    fn locate_row(&self, row_id: u64) -> Result<RowLocation, String> {
        if row_id >= self.total_rows {
            return Err(format!(
                "row id {row_id} is out of bounds for {} rows",
                self.total_rows
            ));
        }
        let shard_index = self
            .shards
            .partition_point(|shard| shard.start_row + shard.row_count <= row_id);
        let shard = self
            .shards
            .get(shard_index)
            .ok_or_else(|| format!("row id {row_id} has no backing shard"))?;
        let local_row = row_id - shard.start_row;
        let byte_start = local_row
            .checked_mul(self.row_bytes as u64)
            .and_then(|value| shard.data_offset.checked_add(value))
            .ok_or_else(|| format!("byte offset for row id {row_id} overflows u64"))?;
        Ok(RowLocation {
            shard_index,
            byte_start,
        })
    }

    fn read_rows(&mut self, row_ids: &[i64], output: *mut u8) -> Result<(), String> {
        // Validate every ID before any caller output is touched. Cached rows
        // already completed location/range validation on the read that loaded
        // them; misses below validate their current backing location again.
        let mut validated_ids = Vec::with_capacity(row_ids.len());
        for &row_id in row_ids {
            let row_id = u64::try_from(row_id)
                .map_err(|_| format!("row id {row_id} must be nonnegative"))?;
            if row_id >= self.total_rows {
                return Err(format!(
                    "row id {row_id} is out of bounds for {} rows",
                    self.total_rows
                ));
            }
            validated_ids.push(row_id);
        }

        if validated_ids
            .iter()
            .all(|row_id| self.cache.contains(*row_id))
        {
            let mut order_matches = validated_ids.len() == self.cache.resident_rows();
            let mut order_cursor = self.cache.oldest;
            if !validated_ids.is_empty() {
                // SAFETY: q38_ple_read validated output for the full writable
                // range, and all row IDs were validated before this point.
                let output = unsafe {
                    std::slice::from_raw_parts_mut(output, validated_ids.len() * self.row_bytes)
                };
                for (index, &row_id) in validated_ids.iter().enumerate() {
                    let (source, newer) = self
                        .cache
                        .get_with_newer(row_id)
                        .expect("all-hit cache membership changed under store mutex");
                    output[index * self.row_bytes..(index + 1) * self.row_bytes]
                        .copy_from_slice(source);
                    if order_matches && order_cursor == Some(row_id) {
                        order_cursor = newer;
                    } else {
                        order_matches = false;
                    }
                }
            }
            order_matches &= order_cursor.is_none();
            if !order_matches {
                for row_id in validated_ids.iter().copied() {
                    let touched = self.cache.touch(row_id);
                    debug_assert!(touched);
                }
            }
            self.counters.read_calls = self.counters.read_calls.saturating_add(1);
            self.counters.requested_rows = self
                .counters
                .requested_rows
                .saturating_add(validated_ids.len() as u64);
            self.counters.cache_hits = self
                .counters
                .cache_hits
                .saturating_add(validated_ids.len() as u64);
            return Ok(());
        }

        // Validate every miss byte range before output is touched.
        let mut locations = Vec::with_capacity(row_ids.len());
        for &row_id in &validated_ids {
            let location = self.locate_row(row_id)?;
            location
                .byte_start
                .checked_add(self.row_bytes as u64)
                .ok_or_else(|| format!("byte range for row id {row_id} overflows u64"))?;
            locations.push(location);
        }

        let cache_hits: Vec<bool> = row_ids
            .iter()
            .map(|row_id| self.cache.contains(*row_id as u64))
            .collect();
        let hit_count = cache_hits.iter().filter(|hit| **hit).count();

        let mut missing_index = HashMap::<u64, usize>::new();
        let mut missing_rows = Vec::<(u64, RowLocation)>::new();
        for (index, &row_id) in row_ids.iter().enumerate() {
            if cache_hits[index] {
                continue;
            }
            let row_id = row_id as u64;
            if let std::collections::hash_map::Entry::Vacant(entry) = missing_index.entry(row_id) {
                entry.insert(missing_rows.len());
                missing_rows.push((row_id, locations[index]));
            }
        }

        let mut page_index = HashMap::<PageKey, usize>::new();
        let mut page_keys = Vec::<PageKey>::new();
        let mut page_copies = Vec::<Vec<PageCopy>>::new();
        for (missing_slot, &(row_id, location)) in missing_rows.iter().enumerate() {
            let mut copied = 0_usize;
            while copied < self.row_bytes {
                let absolute = location
                    .byte_start
                    .checked_add(copied as u64)
                    .ok_or_else(|| format!("byte range for row id {row_id} overflows u64"))?;
                let page_offset = absolute / PAGE_SIZE as u64 * PAGE_SIZE as u64;
                let key = PageKey {
                    shard_index: location.shard_index,
                    offset: page_offset,
                };
                let page_slot = match page_index.entry(key) {
                    std::collections::hash_map::Entry::Occupied(entry) => *entry.get(),
                    std::collections::hash_map::Entry::Vacant(entry) => {
                        let page_slot = page_keys.len();
                        entry.insert(page_slot);
                        page_keys.push(key);
                        page_copies.push(Vec::new());
                        page_slot
                    }
                };
                let in_page = (absolute - page_offset) as usize;
                let take = (self.row_bytes - copied).min(PAGE_SIZE - in_page);
                page_copies[page_slot].push(PageCopy {
                    missing_slot,
                    page_offset: in_page,
                    row_offset: copied,
                    len: take,
                });
                copied += take;
            }
        }

        let staged_bytes = missing_rows
            .len()
            .checked_mul(self.row_bytes)
            .ok_or_else(|| "missing-row staging size overflows usize".to_owned())?;
        let mut staged_rows = vec![0_u8; staged_bytes];
        let mut io_bytes = 0_u64;
        for chunk_start in (0..page_keys.len()).step_by(self.max_batch_pages) {
            let chunk_end = chunk_start
                .saturating_add(self.max_batch_pages)
                .min(page_keys.len());
            let page_reads: Vec<PageRead> = page_keys[chunk_start..chunk_end]
                .iter()
                .map(|key| PageRead {
                    fd: self.shards[key.shard_index].file.as_raw_fd(),
                    offset: key.offset,
                })
                .collect();
            let (read_sizes, chunk_io_bytes) = self
                .reader
                .read_pages(&page_reads, self.queue_depth)
                .map_err(|error| format!("io_uring page read failed: {error}"))?;
            io_bytes = io_bytes.saturating_add(chunk_io_bytes);

            for (local_page, &read_size) in read_sizes.iter().enumerate() {
                let global_page = chunk_start + local_page;
                let page = self.reader.page(local_page, read_size);
                for copy in &page_copies[global_page] {
                    let available_end =
                        copy.page_offset.checked_add(copy.len).ok_or_else(|| {
                            "page copy range overflows usize while staging rows".to_owned()
                        })?;
                    if available_end > read_size {
                        let row_id = missing_rows[copy.missing_slot].0;
                        return Err(format!(
                            "short read for row id {row_id}: page at offset {} returned \
                             {read_size} bytes but byte {available_end} is required",
                            page_keys[global_page].offset
                        ));
                    }
                    let destination_start = copy.missing_slot * self.row_bytes + copy.row_offset;
                    staged_rows[destination_start..destination_start + copy.len]
                        .copy_from_slice(&page[copy.page_offset..available_end]);
                }
            }
        }

        // All fallible validation and I/O is complete. Commit the staged misses
        // and resident hits to caller output in exact order, including duplicates.
        if !row_ids.is_empty() {
            // SAFETY: q38_ple_read validated output for row_ids.len()*row_bytes
            // writable bytes, and no Rust reference aliases it.
            let output =
                unsafe { std::slice::from_raw_parts_mut(output, row_ids.len() * self.row_bytes) };
            for (index, &row_id) in row_ids.iter().enumerate() {
                let destination = &mut output[index * self.row_bytes..(index + 1) * self.row_bytes];
                if cache_hits[index] {
                    let source = self
                        .cache
                        .get(row_id as u64)
                        .ok_or_else(|| format!("cached row id {row_id} disappeared"))?;
                    destination.copy_from_slice(source);
                } else {
                    let slot = missing_index[&(row_id as u64)];
                    let source_start = slot * self.row_bytes;
                    destination
                        .copy_from_slice(&staged_rows[source_start..source_start + self.row_bytes]);
                }
            }
        }

        // Apply requests in order: an O(1) touch makes each occurrence newest,
        // and a missing resident is copied from the now-committed output after
        // evicting enough old payload. This is exact LRU by last occurrence,
        // including duplicates and rows evicted earlier in the same batch.
        if !row_ids.is_empty() {
            // SAFETY: output was validated for the whole result above.
            let output = unsafe {
                std::slice::from_raw_parts(output.cast_const(), row_ids.len() * self.row_bytes)
            };
            for (index, &row_id) in row_ids.iter().enumerate() {
                let row_id = row_id as u64;
                if !self.cache.touch(row_id) {
                    let offset = index * self.row_bytes;
                    self.cache
                        .insert_copy(row_id, &output[offset..offset + self.row_bytes]);
                }
            }
        }

        self.counters.read_calls = self.counters.read_calls.saturating_add(1);
        self.counters.requested_rows = self
            .counters
            .requested_rows
            .saturating_add(row_ids.len() as u64);
        self.counters.cache_hits = self.counters.cache_hits.saturating_add(hit_count as u64);
        self.counters.cache_misses = self
            .counters
            .cache_misses
            .saturating_add((row_ids.len() - hit_count) as u64);
        self.counters.io_pages_read = self
            .counters
            .io_pages_read
            .saturating_add(page_keys.len() as u64);
        self.counters.io_bytes_read = self.counters.io_bytes_read.saturating_add(io_bytes);
        Ok(())
    }

    fn stats(&self) -> [u64; STATS_LEN] {
        [
            self.counters.read_calls,
            self.counters.requested_rows,
            self.counters.cache_hits,
            self.counters.cache_misses,
            self.counters.io_pages_read,
            self.counters.io_bytes_read,
            self.cache.resident_rows() as u64,
            self.cache.resident_bytes() as u64,
        ]
    }
}

unsafe fn handle_from_ptr<'a>(handle: *mut c_void) -> Result<&'a Handle, String> {
    if handle.is_null() {
        return Err("handle must not be null".to_owned());
    }
    if !(handle as usize).is_multiple_of(align_of::<Handle>()) {
        return Err("handle is not properly aligned".to_owned());
    }
    // SAFETY: a non-null, aligned handle returned by q38_ple_open remains valid
    // until q38_ple_close. This lifetime requirement is part of the C ABI.
    Ok(unsafe { &*handle.cast::<Handle>() })
}

#[allow(clippy::too_many_arguments)] // Mirrors the fixed external C ABI.
unsafe fn open_impl(
    paths: *const *const c_char,
    data_offsets: *const u64,
    row_counts: *const u64,
    shard_count: usize,
    row_bytes: usize,
    cache_bytes: usize,
    queue_depth: u32,
    max_batch_pages: usize,
) -> Result<*mut c_void, String> {
    if shard_count == 0 {
        return Err("shard_count must be positive".to_owned());
    }
    if row_bytes == 0 {
        return Err("row_bytes must be positive".to_owned());
    }
    let row_bytes_u64 =
        u64::try_from(row_bytes).map_err(|_| "row_bytes does not fit in u64".to_owned())?;
    if queue_depth == 0 {
        return Err("queue_depth must be positive".to_owned());
    }
    if max_batch_pages == 0 {
        return Err("max_batch_pages must be positive".to_owned());
    }
    validate_array(paths, shard_count, "paths")?;
    validate_array(data_offsets, shard_count, "data_offsets")?;
    validate_array(row_counts, shard_count, "row_counts")?;

    // SAFETY: all arrays were validated for non-null, alignment, and size.
    let paths = unsafe { std::slice::from_raw_parts(paths, shard_count) };
    let data_offsets = unsafe { std::slice::from_raw_parts(data_offsets, shard_count) };
    let row_counts = unsafe { std::slice::from_raw_parts(row_counts, shard_count) };

    let mut shards = Vec::with_capacity(shard_count);
    let mut total_rows = 0_u64;
    for index in 0..shard_count {
        let path_ptr = paths[index];
        if path_ptr.is_null() {
            return Err(format!("paths[{index}] must not be null"));
        }
        // SAFETY: each path is required by the ABI to point to a valid,
        // NUL-terminated byte string for the duration of this call.
        let path_bytes = unsafe { CStr::from_ptr(path_ptr) }.to_bytes();
        if path_bytes.is_empty() {
            return Err(format!("paths[{index}] must not be empty"));
        }
        let row_count = row_counts[index];
        let data_bytes = row_count
            .checked_mul(row_bytes_u64)
            .ok_or_else(|| format!("row byte size for shard {index} overflows u64"))?;
        let data_end = data_offsets[index]
            .checked_add(data_bytes)
            .ok_or_else(|| format!("data byte range for shard {index} overflows u64"))?;
        if data_end > i64::MAX as u64 {
            return Err(format!(
                "data byte range for shard {index} exceeds Linux signed file offsets"
            ));
        }
        let next_total = total_rows
            .checked_add(row_count)
            .ok_or_else(|| "total row count overflows u64".to_owned())?;
        let path = Path::new(std::ffi::OsStr::from_bytes(path_bytes));
        let file = OpenOptions::new()
            .read(true)
            .custom_flags(libc::O_DIRECT | libc::O_CLOEXEC)
            .open(path)
            .map_err(|error| format!("failed to open shard {index} {:?}: {error}", path))?;
        shards.push(Shard {
            file,
            data_offset: data_offsets[index],
            row_count,
            start_row: total_rows,
        });
        total_rows = next_total;
    }

    let reader = Reader::new(queue_depth, max_batch_pages)
        .map_err(|error| format!("failed to initialize io_uring reader: {error}"))?;
    let handle = Box::new(Handle {
        store: Mutex::new(Store {
            shards,
            total_rows,
            row_bytes,
            queue_depth: queue_depth as usize,
            max_batch_pages,
            reader,
            cache: RowCache::new(cache_bytes, row_bytes),
            counters: Counters::default(),
        }),
    });
    Ok(Box::into_raw(handle).cast::<c_void>())
}

/// Opens a read-only, multi-shard PLE row store.
///
/// `paths`, `data_offsets`, and `row_counts` each contain `shard_count`
/// elements. Logical rows are the concatenation of shards in array order.
///
/// # Safety
///
/// All non-null input pointers must be valid and aligned for their documented
/// element counts. Each path pointer must reference a NUL-terminated string for
/// this call. The returned handle must eventually be closed exactly once.
#[no_mangle]
pub unsafe extern "C" fn q38_ple_open(
    paths: *const *const c_char,
    data_offsets: *const u64,
    row_counts: *const u64,
    shard_count: usize,
    row_bytes: usize,
    cache_bytes: usize,
    queue_depth: u32,
    max_batch_pages: usize,
) -> *mut c_void {
    clear_last_error();
    match catch_unwind(AssertUnwindSafe(|| unsafe {
        open_impl(
            paths,
            data_offsets,
            row_counts,
            shard_count,
            row_bytes,
            cache_bytes,
            queue_depth,
            max_batch_pages,
        )
    })) {
        Ok(Ok(handle)) => handle,
        Ok(Err(error)) => {
            set_last_error(error);
            ptr::null_mut()
        }
        Err(payload) => {
            set_last_error(panic_message(payload));
            ptr::null_mut()
        }
    }
}

unsafe fn read_impl(
    handle: *mut c_void,
    row_ids: *const i64,
    count: usize,
    output: *mut u8,
    output_bytes: usize,
) -> Result<(), String> {
    let handle = unsafe { handle_from_ptr(handle) }?;
    validate_array(row_ids, count, "row_ids")?;
    let mut store = handle
        .store
        .lock()
        .map_err(|_| "store mutex is poisoned".to_owned())?;
    let required = count
        .checked_mul(store.row_bytes)
        .ok_or_else(|| "required output size overflows usize".to_owned())?;
    if output_bytes < required {
        return Err(format!(
            "output_bytes={output_bytes} is smaller than required {required}"
        ));
    }
    validate_output(output, required, "output")?;
    let row_id_bytes = count * size_of::<i64>();
    if ranges_overlap(
        row_ids.cast::<u8>(),
        row_id_bytes,
        output.cast_const(),
        required,
    ) {
        return Err("row_ids and output byte ranges must not overlap".to_owned());
    }
    // SAFETY: row_ids was validated for count readable, aligned elements.
    let row_ids = if count == 0 {
        &[]
    } else {
        unsafe { std::slice::from_raw_parts(row_ids, count) }
    };
    store.read_rows(row_ids, output)
}

/// Reads `count` logical rows into caller-owned output in the requested order.
/// Returns 0 on success and -1 on error. No output bytes are written when row
/// validation, page planning, or I/O fails.
///
/// # Safety
///
/// `handle` must be live, `row_ids` must contain `count` readable elements, and
/// `output` must provide at least `count * row_bytes` writable, non-overlapping
/// bytes. No concurrent call may close the handle.
#[no_mangle]
pub unsafe extern "C" fn q38_ple_read(
    handle: *mut c_void,
    row_ids: *const i64,
    count: usize,
    output: *mut u8,
    output_bytes: usize,
) -> c_int {
    clear_last_error();
    match catch_unwind(AssertUnwindSafe(|| unsafe {
        read_impl(handle, row_ids, count, output, output_bytes)
    })) {
        Ok(Ok(())) => 0,
        Ok(Err(error)) => {
            set_last_error(error);
            -1
        }
        Err(payload) => {
            set_last_error(panic_message(payload));
            -1
        }
    }
}

unsafe fn stats_impl(handle: *mut c_void, output: *mut u64, count: usize) -> Result<(), String> {
    let handle = unsafe { handle_from_ptr(handle) }?;
    if count < STATS_LEN {
        return Err(format!(
            "stats output count {count} is smaller than required {STATS_LEN}"
        ));
    }
    validate_array(output.cast_const(), STATS_LEN, "stats output")?;
    let store = handle
        .store
        .lock()
        .map_err(|_| "store mutex is poisoned".to_owned())?;
    let stats = store.stats();
    // SAFETY: output was validated for STATS_LEN writable u64 elements.
    unsafe { ptr::copy_nonoverlapping(stats.as_ptr(), output, STATS_LEN) };
    Ok(())
}

/// Writes eight u64 fields: successful read calls, requested rows, cache hits,
/// cache misses, submitted pages, returned I/O bytes, resident cache rows, and
/// resident cache payload bytes. Returns 0 on success and -1 on error.
///
/// # Safety
///
/// `handle` must be live and `output` must provide at least eight writable u64
/// elements. No concurrent call may close the handle.
#[no_mangle]
pub unsafe extern "C" fn q38_ple_stats(
    handle: *mut c_void,
    output: *mut u64,
    count: usize,
) -> c_int {
    clear_last_error();
    match catch_unwind(AssertUnwindSafe(|| unsafe {
        stats_impl(handle, output, count)
    })) {
        Ok(Ok(())) => 0,
        Ok(Err(error)) => {
            set_last_error(error);
            -1
        }
        Err(payload) => {
            set_last_error(panic_message(payload));
            -1
        }
    }
}

/// Releases a handle returned by q38_ple_open. The caller must ensure no other
/// thread is using the handle and must close it exactly once.
///
/// # Safety
///
/// A non-null `handle` must have been returned by q38_ple_open, must still be
/// live, and must have no concurrent users. Passing null is a no-op.
#[no_mangle]
pub unsafe extern "C" fn q38_ple_close(handle: *mut c_void) {
    clear_last_error();
    if handle.is_null() {
        return;
    }
    if !(handle as usize).is_multiple_of(align_of::<Handle>()) {
        set_last_error("handle is not properly aligned");
        return;
    }
    match catch_unwind(AssertUnwindSafe(|| {
        // SAFETY: the pointer must be a live handle returned by q38_ple_open,
        // and the ABI requires exactly one close with no concurrent users.
        drop(unsafe { Box::from_raw(handle.cast::<Handle>()) });
    })) {
        Ok(()) => {}
        Err(payload) => set_last_error(panic_message(payload)),
    }
}

/// Returns a thread-local error string valid until the next ABI call on the
/// same thread. The empty string means that the last call succeeded.
#[no_mangle]
pub extern "C" fn q38_ple_last_error() -> *const c_char {
    static EMPTY: [c_char; 1] = [0];
    LAST_ERROR
        .try_with(|slot| {
            slot.try_borrow()
                .map(|message| message.as_ptr())
                .unwrap_or(EMPTY.as_ptr())
        })
        .unwrap_or(EMPTY.as_ptr())
}

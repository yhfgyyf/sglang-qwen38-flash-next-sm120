use std::ffi::{c_char, c_void, CStr, CString};
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::mem::size_of;
use std::path::PathBuf;
use std::ptr;
use std::time::{SystemTime, UNIX_EPOCH};

use q38_ple_store::{q38_ple_close, q38_ple_last_error, q38_ple_open, q38_ple_read, q38_ple_stats};

const ROW_BYTES: usize = 160;
const STATS_LEN: usize = 8;

struct TempShard {
    path: PathBuf,
    rows: Vec<Vec<u8>>,
}

impl TempShard {
    fn new(tag: &str, data_offset: usize, row_count: usize, seed: u8) -> Self {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "q38-ple-store-{tag}-{}-{nonce}.bin",
            std::process::id()
        ));
        let rows: Vec<Vec<u8>> = (0..row_count)
            .map(|row| {
                (0..ROW_BYTES)
                    .map(|column| {
                        seed.wrapping_add((row as u8).wrapping_mul(17))
                            .wrapping_add((column as u8).wrapping_mul(29))
                    })
                    .collect()
            })
            .collect();
        let mut file = OpenOptions::new()
            .create_new(true)
            .read(true)
            .write(true)
            .open(&path)
            .unwrap();
        file.write_all(&vec![0xa5; data_offset]).unwrap();
        for row in &rows {
            file.write_all(row).unwrap();
        }
        file.sync_all().unwrap();
        Self { path, rows }
    }
}

impl Drop for TempShard {
    fn drop(&mut self) {
        fs::remove_file(&self.path).unwrap();
    }
}

fn last_error() -> String {
    unsafe {
        CStr::from_ptr(q38_ple_last_error())
            .to_string_lossy()
            .into_owned()
    }
}

struct Store(*mut c_void);

impl Store {
    fn open(
        shards: &[&TempShard],
        data_offsets: &[u64],
        row_counts: &[u64],
        cache_bytes: usize,
        queue_depth: u32,
        max_batch_pages: usize,
    ) -> Self {
        let paths: Vec<CString> = shards
            .iter()
            .map(|shard| CString::new(shard.path.as_os_str().as_encoded_bytes()).unwrap())
            .collect();
        let path_ptrs: Vec<*const c_char> = paths.iter().map(|path| path.as_ptr()).collect();
        let handle = unsafe {
            q38_ple_open(
                path_ptrs.as_ptr(),
                data_offsets.as_ptr(),
                row_counts.as_ptr(),
                shards.len(),
                ROW_BYTES,
                cache_bytes,
                queue_depth,
                max_batch_pages,
            )
        };
        assert!(!handle.is_null(), "q38_ple_open failed: {}", last_error());
        Self(handle)
    }

    fn read(&self, row_ids: &[i64]) -> Result<Vec<u8>, String> {
        let mut output = vec![0xcd; row_ids.len() * ROW_BYTES];
        let result = unsafe {
            q38_ple_read(
                self.0,
                row_ids.as_ptr(),
                row_ids.len(),
                output.as_mut_ptr(),
                output.len(),
            )
        };
        if result == 0 {
            Ok(output)
        } else {
            Err(last_error())
        }
    }

    fn stats(&self) -> [u64; STATS_LEN] {
        let mut stats = [0; STATS_LEN];
        let result = unsafe { q38_ple_stats(self.0, stats.as_mut_ptr(), stats.len()) };
        assert_eq!(result, 0, "q38_ple_stats failed: {}", last_error());
        stats
    }
}

impl Drop for Store {
    fn drop(&mut self) {
        unsafe { q38_ple_close(self.0) };
    }
}

#[test]
fn reads_exact_unaligned_cross_page_rows_across_shards_in_order() {
    let first = TempShard::new("exact-a", 4037, 30, 3);
    let second = TempShard::new("exact-b", 17, 4, 101);
    let store = Store::open(
        &[&first, &second],
        &[4037, 17],
        &[30, 4],
        ROW_BYTES * 16,
        2,
        8,
    );

    let row_ids = [0, 29, 0, 30, 33];
    let output = store.read(&row_ids).unwrap();
    let expected: Vec<u8> = [
        &first.rows[0],
        &first.rows[29],
        &first.rows[0],
        &second.rows[0],
        &second.rows[3],
    ]
    .into_iter()
    .flat_map(|row| row.iter().copied())
    .collect();
    assert_eq!(output, expected);

    // All three rows are resident, including a duplicate from the first call.
    assert_eq!(store.read(&[0, 0, 30]).unwrap().len(), ROW_BYTES * 3);
    let stats = store.stats();
    assert_eq!(stats[0], 2); // successful read calls
    assert_eq!(stats[1], 8); // requested rows
    assert_eq!(stats[2], 3); // cache hits
    assert_eq!(stats[3], 5); // cache misses (per requested row)
    assert_eq!(stats[4], 4); // unique 4 KiB pages submitted
    assert_eq!(stats[6], 4); // unique resident rows
    assert_eq!(stats[7], (4 * ROW_BYTES) as u64);
}

#[test]
fn enforces_true_lru_payload_budget() {
    let shard = TempShard::new("lru", 0, 4, 19);
    let store = Store::open(&[&shard], &[0], &[4], ROW_BYTES * 2 + (ROW_BYTES - 1), 1, 4);

    store.read(&[0, 1, 2]).unwrap();
    let stats = store.stats();
    assert_eq!(stats[6], 2);
    assert_eq!(stats[7], (2 * ROW_BYTES) as u64);

    store.read(&[1, 2]).unwrap();
    store.read(&[0]).unwrap();
    let stats = store.stats();
    assert_eq!(stats[2], 2);
    assert_eq!(stats[3], 4);
    assert_eq!(stats[4], 2); // each miss batch deduplicates the same page
    assert_eq!(stats[6], 2);
    assert!(stats[7] <= (ROW_BYTES * 2 + ROW_BYTES - 1) as u64);
}

#[test]
fn duplicate_last_occurrence_controls_exact_lru_eviction() {
    let shard = TempShard::new("duplicate-lru", 0, 4, 31);
    let store = Store::open(&[&shard], &[0], &[4], ROW_BYTES * 3, 1, 4);

    store.read(&[0, 1, 2]).unwrap();
    // Untouched row 2 becomes oldest; duplicate row 0 is newest by its last
    // occurrence, leaving the exact order [2, 1, 0].
    store.read(&[0, 1, 0]).unwrap();
    store.read(&[3]).unwrap();
    let before = store.stats();

    let output = store.read(&[0, 1, 2]).unwrap();
    let expected: Vec<u8> = [&shard.rows[0], &shard.rows[1], &shard.rows[2]]
        .into_iter()
        .flat_map(|row| row.iter().copied())
        .collect();
    assert_eq!(output, expected);
    let after = store.stats();
    assert_eq!(after[2] - before[2], 2); // 0 and 1 survived.
    assert_eq!(after[3] - before[3], 1); // 2 was the exact eviction.
    assert_eq!(after[6], 3);
    assert_eq!(after[7], (3 * ROW_BYTES) as u64);
}

#[test]
fn full_warm_read_in_current_lru_order_preserves_bytes_and_stats() {
    let shard = TempShard::new("warm-current-order", 0, 4, 37);
    let store = Store::open(&[&shard], &[0], &[4], ROW_BYTES * 4, 1, 4);
    let row_ids = [0_i64, 1, 2, 3];

    let expected = store.read(&row_ids).unwrap();
    let before = store.stats();
    assert_eq!(store.read(&row_ids).unwrap(), expected);
    let after = store.stats();

    assert_eq!(after[0] - before[0], 1);
    assert_eq!(after[1] - before[1], row_ids.len() as u64);
    assert_eq!(after[2] - before[2], row_ids.len() as u64);
    assert_eq!(after[3] - before[3], 0);
    assert_eq!(after[4] - before[4], 0);
    assert_eq!(after[6], row_ids.len() as u64);
    assert_eq!(after[7], (row_ids.len() * ROW_BYTES) as u64);
}

#[test]
fn rejects_bad_ids_and_small_outputs_before_writing() {
    let shard = TempShard::new("bad-id", 13, 2, 43);
    let store = Store::open(&[&shard], &[13], &[2], ROW_BYTES * 2, 1, 4);
    for row_ids in [&[-1_i64][..], &[2_i64][..]] {
        let mut output = vec![0x7b; ROW_BYTES];
        let result = unsafe {
            q38_ple_read(
                store.0,
                row_ids.as_ptr(),
                row_ids.len(),
                output.as_mut_ptr(),
                output.len(),
            )
        };
        assert_eq!(result, -1);
        assert_eq!(output, vec![0x7b; ROW_BYTES]);
        assert!(last_error().contains("row id"));
    }

    let mut output = vec![0x35; ROW_BYTES - 1];
    let row_ids = [0_i64];
    let output_len = output.len();
    let result = unsafe {
        q38_ple_read(
            store.0,
            row_ids.as_ptr(),
            row_ids.len(),
            output.as_mut_ptr(),
            output_len,
        )
    };
    assert_eq!(result, -1);
    assert_eq!(output, vec![0x35; ROW_BYTES - 1]);
    assert!(last_error().contains("output_bytes"));

    let mut overlapping = [0_i64; ROW_BYTES / size_of::<i64>()];
    let before = overlapping;
    let result = unsafe {
        q38_ple_read(
            store.0,
            overlapping.as_ptr(),
            1,
            overlapping.as_mut_ptr().cast::<u8>(),
            ROW_BYTES,
        )
    };
    assert_eq!(result, -1);
    assert_eq!(overlapping, before);
    assert!(last_error().contains("must not overlap"));
    assert_eq!(store.stats()[0], 0);

    // Even with an earlier cached hit, a late invalid ID must leave both the
    // caller output and successful-read counters unchanged.
    store.read(&[0]).unwrap();
    let before = store.stats();
    let row_ids = [0_i64, 2];
    let mut output = vec![0x91; ROW_BYTES * row_ids.len()];
    let result = unsafe {
        q38_ple_read(
            store.0,
            row_ids.as_ptr(),
            row_ids.len(),
            output.as_mut_ptr(),
            output.len(),
        )
    };
    assert_eq!(result, -1);
    assert_eq!(output, vec![0x91; ROW_BYTES * row_ids.len()]);
    assert!(last_error().contains("row id"));
    assert_eq!(store.stats(), before);
}

#[test]
fn allows_available_partial_final_page_and_reports_short_rows() {
    let shard = TempShard::new("short", 31, 1, 71);
    let store = Store::open(&[&shard], &[31], &[2], 0, 1, 2);

    assert_eq!(store.read(&[0]).unwrap(), shard.rows[0]);

    let before_failed_read = store.stats();
    let mut output = vec![0xe1; ROW_BYTES];
    let row_ids = [1_i64];
    let result = unsafe {
        q38_ple_read(
            store.0,
            row_ids.as_ptr(),
            row_ids.len(),
            output.as_mut_ptr(),
            output.len(),
        )
    };
    assert_eq!(result, -1);
    assert_eq!(output, vec![0xe1; ROW_BYTES]);
    assert!(last_error().contains("short read"), "{}", last_error());
    assert_eq!(store.stats(), before_failed_read);

    // A later scratch chunk failing must not expose an earlier staged row.
    let transactional = Store::open(&[&shard], &[31], &[27], 0, 1, 1);
    let row_ids = [0_i64, 26];
    let mut output = vec![0x6d; ROW_BYTES * row_ids.len()];
    let result = unsafe {
        q38_ple_read(
            transactional.0,
            row_ids.as_ptr(),
            row_ids.len(),
            output.as_mut_ptr(),
            output.len(),
        )
    };
    assert_eq!(result, -1);
    assert_eq!(output, vec![0x6d; ROW_BYTES * row_ids.len()]);
    assert!(last_error().contains("short read"), "{}", last_error());
    assert_eq!(transactional.stats(), [0; STATS_LEN]);
}

#[test]
fn reads_more_unique_pages_than_fixed_page_scratch_in_chunks() {
    let shard = TempShard::new("multi-chunk", 0, 209, 89);
    let store = Store::open(&[&shard], &[0], &[209], 0, 2, 2);
    let row_ids = [0_i64, 26, 52, 78, 104, 130, 156, 182, 208];

    let output = store.read(&row_ids).unwrap();
    let expected: Vec<u8> = row_ids
        .iter()
        .flat_map(|row_id| shard.rows[*row_id as usize].iter().copied())
        .collect();
    assert_eq!(output, expected);
    assert_eq!(store.stats()[4], 9);

    let crossing = TempShard::new("chunked-cross-page", 4090, 1, 117);
    let crossing_store = Store::open(&[&crossing], &[4090], &[1], 0, 1, 1);
    assert_eq!(crossing_store.read(&[0]).unwrap(), crossing.rows[0]);
    assert_eq!(crossing_store.stats()[4], 2);
}

#[test]
fn validates_open_and_accepts_an_empty_read() {
    let handle =
        unsafe { q38_ple_open(ptr::null(), ptr::null(), ptr::null(), 0, ROW_BYTES, 0, 1, 1) };
    assert!(handle.is_null());
    assert!(last_error().contains("shard_count"));

    let shard = TempShard::new("empty", 0, 1, 7);
    let store = Store::open(&[&shard], &[0], &[1], 0, 1, 1);
    let result = unsafe { q38_ple_read(store.0, ptr::null(), 0, ptr::null_mut(), 0) };
    assert_eq!(result, 0, "{}", last_error());
    assert_eq!(store.stats()[0], 1);
}

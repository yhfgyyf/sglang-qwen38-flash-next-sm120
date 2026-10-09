use std::io;
use std::os::fd::RawFd;
use std::ptr::NonNull;

use io_uring::{opcode, types, IoUring};

pub(crate) const PAGE_SIZE: usize = 4096;

#[derive(Clone, Copy, Debug)]
pub(crate) struct PageRead {
    pub(crate) fd: RawFd,
    pub(crate) offset: u64,
}

struct AlignedBuffer {
    ptr: NonNull<u8>,
    len: usize,
}

// The allocation is accessed only while the owning Store mutex is held.
unsafe impl Send for AlignedBuffer {}

impl AlignedBuffer {
    fn new(len: usize) -> io::Result<Self> {
        if len == 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "aligned buffer length must be positive",
            ));
        }
        let mut raw = std::ptr::null_mut();
        // SAFETY: posix_memalign initializes raw on success. This type owns the
        // allocation and releases it exactly once in Drop.
        let result = unsafe { libc::posix_memalign(&mut raw, PAGE_SIZE, len) };
        if result != 0 {
            return Err(io::Error::from_raw_os_error(result));
        }
        let ptr = NonNull::new(raw.cast::<u8>())
            .ok_or_else(|| io::Error::other("posix_memalign returned null"))?;
        Ok(Self { ptr, len })
    }

    fn page_ptr(&self, index: usize) -> *mut u8 {
        debug_assert!((index + 1) * PAGE_SIZE <= self.len);
        // SAFETY: callers bound index by max_batch_pages, which sized this
        // allocation, and serialize all access through the Store mutex.
        unsafe { self.ptr.as_ptr().add(index * PAGE_SIZE) }
    }

    fn page(&self, index: usize, read_size: usize) -> &[u8] {
        debug_assert!(read_size <= PAGE_SIZE);
        // SAFETY: the corresponding CQE has completed and the Store mutex
        // prevents the buffer from being reused while this view exists.
        unsafe { std::slice::from_raw_parts(self.page_ptr(index), read_size) }
    }
}

impl Drop for AlignedBuffer {
    fn drop(&mut self) {
        // SAFETY: ptr was allocated by posix_memalign and has not been freed.
        unsafe { libc::free(self.ptr.as_ptr().cast()) };
    }
}

/// Persistent synchronous io_uring state.
///
/// The aligned-buffer and submission/completion structure is adapted from
/// `rust/sglang-storage/src/io_uring_reader.rs` in SGLang (Apache-2.0). This
/// version removes PyO3, stages exact row slices in the parent module, accepts
/// partial final pages, and permanently retires a ring after an
/// indeterminate submission/completion failure.
pub(crate) struct Reader {
    // Field order matters: closing the ring cancels/drains kernel ownership
    // before the aligned buffer is freed during drop.
    ring: IoUring,
    buffer: AlignedBuffer,
    max_batch_pages: usize,
    broken: bool,
}

// Reader is accessed only behind Store's mutex. io-uring and the owned buffer
// are safe to move between caller threads while no operation is in progress.
unsafe impl Send for Reader {}

impl Reader {
    pub(crate) fn new(queue_depth: u32, max_batch_pages: usize) -> io::Result<Self> {
        let buffer_len = max_batch_pages.checked_mul(PAGE_SIZE).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                "page scratch size overflows usize",
            )
        })?;
        Ok(Self {
            ring: IoUring::new(queue_depth)?,
            buffer: AlignedBuffer::new(buffer_len)?,
            max_batch_pages,
            broken: false,
        })
    }

    pub(crate) fn read_pages(
        &mut self,
        requests: &[PageRead],
        queue_depth: usize,
    ) -> io::Result<(Vec<usize>, u64)> {
        if self.broken {
            return Err(io::Error::other(
                "io_uring reader is retired after an earlier incomplete operation",
            ));
        }
        if requests.len() > self.max_batch_pages {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "read needs {} pages but max_batch_pages is {}",
                    requests.len(),
                    self.max_batch_pages
                ),
            ));
        }
        if requests.is_empty() {
            return Ok((Vec::new(), 0));
        }

        let mut read_sizes = vec![usize::MAX; requests.len()];
        let mut total_bytes = 0_u64;
        for base in (0..requests.len()).step_by(queue_depth) {
            let batch = (requests.len() - base).min(queue_depth);
            let push_result = {
                let mut submission = self.ring.submission();
                let mut result = Ok(());
                for local in 0..batch {
                    let index = base + local;
                    let request = requests[index];
                    let entry = opcode::Read::new(
                        types::Fd(request.fd),
                        self.buffer.page_ptr(index),
                        PAGE_SIZE as u32,
                    )
                    .offset(request.offset)
                    .build()
                    .user_data((index + 1) as u64);
                    // SAFETY: every entry points into the persistent aligned
                    // allocation. The allocation is not reused until every CQE
                    // in the successfully submitted batch is consumed.
                    if unsafe { submission.push(&entry) }.is_err() {
                        result = Err(io::Error::new(
                            io::ErrorKind::WouldBlock,
                            "io_uring submission queue filled unexpectedly",
                        ));
                        break;
                    }
                }
                result
            };
            if let Err(error) = push_result {
                // Some SQEs may be staged in the ring. Never submit or reuse
                // their buffers after this point; ring drop precedes buffer drop.
                self.broken = true;
                return Err(error);
            }

            if let Err(error) = self.ring.submit_and_wait(batch) {
                // Submission failure can have indeterminate partial progress.
                // Retiring the ring prevents buffer reuse; closing the ring on
                // handle drop resolves kernel ownership before buffer release.
                self.broken = true;
                return Err(error);
            }

            let mut first_error = None;
            let mut missing_completion = false;
            {
                let mut completion = self.ring.completion();
                for _ in 0..batch {
                    let Some(entry) = completion.next() else {
                        missing_completion = true;
                        break;
                    };
                    let user_data = entry.user_data();
                    let index = match user_data
                        .checked_sub(1)
                        .and_then(|value| usize::try_from(value).ok())
                    {
                        Some(index) if (base..base + batch).contains(&index) => index,
                        _ => {
                            first_error.get_or_insert_with(|| {
                                io::Error::other("invalid io_uring completion user_data")
                            });
                            continue;
                        }
                    };
                    if read_sizes[index] != usize::MAX {
                        first_error.get_or_insert_with(|| {
                            io::Error::other("duplicate io_uring completion user_data")
                        });
                        continue;
                    }
                    let result = entry.result();
                    if result < 0 {
                        first_error.get_or_insert_with(|| io::Error::from_raw_os_error(-result));
                        read_sizes[index] = 0;
                    } else {
                        let size = result as usize;
                        read_sizes[index] = size;
                        total_bytes = total_bytes.saturating_add(size as u64);
                    }
                }
            }

            if missing_completion {
                self.broken = true;
                return Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    "missing io_uring completion; reader retired",
                ));
            }
            if let Some(error) = first_error {
                // All CQEs in this batch were consumed before returning.
                return Err(error);
            }
        }

        if read_sizes.contains(&usize::MAX) {
            self.broken = true;
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "missing io_uring read size; reader retired",
            ));
        }
        Ok((read_sizes, total_bytes))
    }

    pub(crate) fn page(&self, index: usize, read_size: usize) -> &[u8] {
        self.buffer.page(index, read_size)
    }
}

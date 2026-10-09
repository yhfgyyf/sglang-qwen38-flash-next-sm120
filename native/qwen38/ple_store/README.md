# q38 PLE store

This standalone Linux Rust crate exposes a synchronous C ABI for reading the
original PLE row bytes from one or more safetensors shards. Shards are logically
concatenated in the order passed to `q38_ple_open`; each `data_offsets[i]` is the
absolute file byte offset of that shard's first row.

The implementation opens every shard read-only with `O_DIRECT`, reuses one
`io_uring`, and allocates a fixed 4 KiB-aligned scratch area of
`max_batch_pages * 4096` bytes. A request deduplicates missing logical rows and
physical pages, reads arbitrarily large valid requests in scratch-sized chunks,
and stages only the unique missing row bytes before committing caller output.
No page payloads accumulate across chunks and no format conversion occurs.
Unaligned row starts, rows crossing 4 KiB pages, duplicate IDs, and partial final
file pages are supported. A partial page succeeds only when every requested byte
is present.

## Cache policy

The cache is an exact row-level LRU. Its resident payload never exceeds
`floor(cache_bytes / row_bytes) * row_bytes`; unused remainder bytes are not
allocated. Cache metadata contains entries only for resident rows, and no shard
is memory-mapped or copied wholesale. Batched accesses update LRU order by each
row's last occurrence, which is equivalent to processing the requested row IDs
sequentially.

## ABI contract

See `include/q38_ple_store.h` for declarations. Calls return `0` on success and
`-1` on error except `open`, `close`, and `last_error`. `q38_ple_last_error()` is
thread-local and remains valid until the next ABI call on that thread. Paths
must be valid NUL-terminated strings during `open`; arrays and output buffers
must be valid, aligned, and non-overlapping for the documented lengths. A handle
may be read concurrently because operations are serialized internally, but the
caller must call `close` exactly once with no concurrent users.

`q38_ple_stats(handle, output, count)` requires `count >= 8` and writes:

0. successful read calls
1. requested rows across successful calls
2. requested rows found in cache at batch start
3. requested rows absent from cache at batch start
4. unique 4 KiB pages submitted to `io_uring`
5. bytes returned by successful page reads
6. currently resident cache rows
7. currently resident cache payload bytes

Counters saturate at `UINT64_MAX`. Failed reads do not change counters or cache
state and do not write caller output. After an indeterminate submission failure,
the ring is retired so its aligned buffers cannot be reused; ring destruction
precedes buffer release.

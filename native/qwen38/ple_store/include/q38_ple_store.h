#ifndef Q38_PLE_STORE_H
#define Q38_PLE_STORE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

enum q38_ple_stat_index {
  Q38_PLE_STAT_READ_CALLS = 0,
  Q38_PLE_STAT_REQUESTED_ROWS = 1,
  Q38_PLE_STAT_CACHE_HITS = 2,
  Q38_PLE_STAT_CACHE_MISSES = 3,
  Q38_PLE_STAT_IO_PAGES_READ = 4,
  Q38_PLE_STAT_IO_BYTES_READ = 5,
  Q38_PLE_STAT_RESIDENT_ROWS = 6,
  Q38_PLE_STAT_RESIDENT_PAYLOAD_BYTES = 7,
  Q38_PLE_STAT_COUNT = 8,
};

void *q38_ple_open(const char *const *paths, const uint64_t *data_offsets,
                   const uint64_t *row_counts, size_t shard_count,
                   size_t row_bytes, size_t cache_bytes, uint32_t queue_depth,
                   size_t max_batch_pages);

int q38_ple_read(void *handle, const int64_t *row_ids, size_t count,
                 uint8_t *output, size_t output_bytes);

int q38_ple_stats(void *handle, uint64_t *output, size_t count);

void q38_ple_close(void *handle);

const char *q38_ple_last_error(void);

#ifdef __cplusplus
}
#endif

#endif

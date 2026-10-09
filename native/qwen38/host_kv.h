// Copyright 2026 Qwen38 native contributors. SPDX-License-Identifier: Apache-2.0
#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// One arena stores byte-exact K and V rows as
//   [layer][physical_slot][K then V][heads * head_dim].
// element_bytes is exactly 1 (FP8 E4M3FN storage) or 2 (BF16 compatibility).
// The backing allocation is mapped pinned host memory.  The byte budget is
// checked before CUDA allocation.  All tensor pointers and IDs are device
// pointers on `device`; id_bytes is exactly 4 (int32) or 8 (int64).
// Reserved physical slot 0 is zero-initialized for every layer.  Other slots
// must be scattered before they are gathered.
//
// Calls enqueue ordinary kernels and are CUDA graph capture/replay compatible.
// The caller must keep the arena alive until every graph that captured one of
// these calls is destroyed and its last replay is complete.  Captured graphs
// using one arena must be replayed serially on one stream.  Eager calls from
// different streams are serialized by an event maintained by the arena.
// Captures intentionally do not record that shared event: one event generation
// cannot represent the latest replay across independently captured graphs.
// Graph owners must quiesce replay before releasing their lifetime reference.
//
// Negative gather IDs are padding and produce all-zero K/V rows.  Negative
// scatter IDs and nonnegative IDs >= slot_count never access the arena; they
// set a deferred device error.  q38_host_kv_check synchronizes the arena's
// last eager-use event and reports/clears that error.  A C caller checking a
// captured replay must quiesce its replay stream first; the Python owner does
// this at its explicit debug boundary.  This fails loudly on invalid
// device-resident IDs without a CPU ID round trip in the hot path.
const char* q38_host_kv_last_error(void);

#define Q38_HOST_KV_ABI_VERSION 2U
uint32_t q38_host_kv_abi_version(void);

int q38_host_kv_required_bytes(uint64_t layer_count, uint64_t slot_count,
                               uint64_t heads, uint64_t head_dim,
                               uint64_t element_bytes,
                               uint64_t* result);

void* q38_host_kv_create(uint64_t layer_count, uint64_t slot_count,
                         uint64_t heads, uint64_t head_dim,
                         uint64_t element_bytes, uint64_t byte_budget,
                         int device);

uint64_t q38_host_kv_allocated_bytes(void* arena);

int q38_host_kv_scatter(void* arena, uint64_t layer, const void* keys,
                        uint64_t key_row_stride_bytes, const void* values,
                        uint64_t value_row_stride_bytes, const void* ids,
                        uint64_t count, int id_bytes, uintptr_t stream);

// key_output and value_output are packed contiguous storage-dtype
// [count, heads, head_dim] buffers. Their row stride is the arena row size.
int q38_host_kv_gather(void* arena, uint64_t layer, const void* ids,
                       uint64_t count, int id_bytes, void* key_output,
                       void* value_output, uintptr_t stream);

// Optional per-call duplicate elimination. row_map is caller-owned CUDA int32
// scratch with at least row_map_elements >= slot_count. The map, IDs, key
// output, and value output must be mutually non-overlapping. The map must
// remain alive until the stream completes and, independently of the arena's
// graph lease, across graph replay when captured. The call resets the first
// slot_count map entries, gathers each distinct valid physical slot once, then
// broadcasts exact bytes to duplicate output rows. The map is scratch and has
// no persistent meaning.
// Padding, OOB handling, output layout, and deferred errors match gather().
int q38_host_kv_gather_dedup(void* arena, uint64_t layer, const void* ids,
                             uint64_t count, int id_bytes, int32_t* row_map,
                             uint64_t row_map_elements, void* key_output,
                             void* value_output, uintptr_t stream);

// Debug/validation boundary: wait for this arena's last recorded eager use,
// surface launch/runtime or invalid-ID errors, and clear the invalid-ID flag.
// See the captured-replay quiescence contract above.
int q38_host_kv_check(void* arena);

// Externally serialize handle lifetime.  This waits for the last recorded
// eager use; captured uses follow the graph-owner quiescence contract above.
int q38_host_kv_close(void* arena);

#ifdef __cplusplus
}
#endif

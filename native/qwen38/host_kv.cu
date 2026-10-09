// Copyright 2026 Qwen38 native contributors. SPDX-License-Identifier: Apache-2.0
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace {

constexpr uint32_t kScatterOutOfBounds = 1;
constexpr uint32_t kGatherOutOfBounds = 2;

struct DeviceError {
  uint32_t code;
  uint32_t reserved;
  uint64_t row;
  int64_t id;
};
static_assert(sizeof(DeviceError) == 24);

__device__ void report_error(DeviceError* error, uint32_t code, uint64_t row,
                             int64_t id) {
  if (atomicCAS(&error->code, 0U, code) == 0U) {
    error->row = row;
    error->id = id;
  }
}

template <typename Id>
__device__ int64_t load_id(const Id* ids, uint64_t row) {
  return static_cast<int64_t>(ids[row]);
}

template <typename Id, typename Unit>
__global__ void scatter_rows(uint8_t* layer_base, uint64_t slots,
                             uint64_t row_bytes, const uint8_t* keys,
                             uint64_t key_stride, const uint8_t* values,
                             uint64_t value_stride, const Id* ids,
                             uint64_t count, DeviceError* error) {
  const uint64_t row = blockIdx.x;
  if (row >= count) {
    return;
  }
  const int64_t id = load_id(ids, row);
  if (id < 0 || static_cast<uint64_t>(id) >= slots) {
    if (threadIdx.x == 0) {
      report_error(error, kScatterOutOfBounds, row, id);
    }
    return;
  }
  const uint64_t units = row_bytes / sizeof(Unit);
  Unit* destination = reinterpret_cast<Unit*>(
      layer_base + static_cast<uint64_t>(id) * 2 * row_bytes);
  const Unit* key_source =
      reinterpret_cast<const Unit*>(keys + row * key_stride);
  const Unit* value_source =
      reinterpret_cast<const Unit*>(values + row * value_stride);
  Unit* key_destination = destination;
  Unit* value_destination = reinterpret_cast<Unit*>(
      reinterpret_cast<uint8_t*>(destination) + row_bytes);
  for (uint64_t column = threadIdx.x; column < units;
       column += blockDim.x) {
    key_destination[column] = key_source[column];
    value_destination[column] = value_source[column];
  }
}

template <typename Id, typename Unit>
__global__ void gather_rows(const uint8_t* layer_base, uint64_t slots,
                            uint64_t row_bytes, const Id* ids, uint64_t count,
                            uint8_t* key_output, uint8_t* value_output,
                            DeviceError* error) {
  const uint64_t row = blockIdx.x;
  if (row >= count) {
    return;
  }
  const int64_t id = load_id(ids, row);
  const bool padding = id < 0;
  const bool out_of_bounds = !padding && static_cast<uint64_t>(id) >= slots;
  if (out_of_bounds && threadIdx.x == 0) {
    report_error(error, kGatherOutOfBounds, row, id);
  }
  const uint64_t units = row_bytes / sizeof(Unit);
  Unit* key_destination =
      reinterpret_cast<Unit*>(key_output + row * row_bytes);
  Unit* value_destination =
      reinterpret_cast<Unit*>(value_output + row * row_bytes);
  if (padding || out_of_bounds) {
    const Unit zero{};
    for (uint64_t column = threadIdx.x; column < units;
         column += blockDim.x) {
      key_destination[column] = zero;
      value_destination[column] = zero;
    }
    return;
  }
  const Unit* source = reinterpret_cast<const Unit*>(
      layer_base + static_cast<uint64_t>(id) * 2 * row_bytes);
  const Unit* key_source = source;
  const Unit* value_source = reinterpret_cast<const Unit*>(
      reinterpret_cast<const uint8_t*>(source) + row_bytes);
  for (uint64_t column = threadIdx.x; column < units;
       column += blockDim.x) {
    key_destination[column] = key_source[column];
    value_destination[column] = value_source[column];
  }
}

template <typename Id>
__global__ void mark_canonical_rows(const Id* ids, uint64_t count,
                                    uint64_t slots, uint32_t* row_map) {
  const uint64_t row =
      static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (row >= count) {
    return;
  }
  const int64_t id = load_id(ids, row);
  if (id >= 0 && static_cast<uint64_t>(id) < slots) {
    atomicMin(row_map + static_cast<uint64_t>(id),
              static_cast<uint32_t>(row));
  }
}

template <typename Id, typename Unit>
__global__ void gather_canonical_rows(
    const uint8_t* layer_base, uint64_t slots, uint64_t row_bytes,
    const Id* ids, uint64_t count, const uint32_t* row_map,
    uint8_t* key_output, uint8_t* value_output, DeviceError* error) {
  const uint64_t row = blockIdx.x;
  if (row >= count) {
    return;
  }
  const int64_t id = load_id(ids, row);
  const bool padding = id < 0;
  const bool out_of_bounds = !padding && static_cast<uint64_t>(id) >= slots;
  if (out_of_bounds && threadIdx.x == 0) {
    report_error(error, kGatherOutOfBounds, row, id);
  }
  const uint64_t units = row_bytes / sizeof(Unit);
  Unit* key_destination =
      reinterpret_cast<Unit*>(key_output + row * row_bytes);
  Unit* value_destination =
      reinterpret_cast<Unit*>(value_output + row * row_bytes);
  if (padding || out_of_bounds) {
    const Unit zero{};
    for (uint64_t column = threadIdx.x; column < units;
         column += blockDim.x) {
      key_destination[column] = zero;
      value_destination[column] = zero;
    }
    return;
  }
  if (row_map[static_cast<uint64_t>(id)] != static_cast<uint32_t>(row)) {
    return;
  }
  const Unit* source = reinterpret_cast<const Unit*>(
      layer_base + static_cast<uint64_t>(id) * 2 * row_bytes);
  const Unit* key_source = source;
  const Unit* value_source = reinterpret_cast<const Unit*>(
      reinterpret_cast<const uint8_t*>(source) + row_bytes);
  for (uint64_t column = threadIdx.x; column < units;
       column += blockDim.x) {
    key_destination[column] = key_source[column];
    value_destination[column] = value_source[column];
  }
}

template <typename Id, typename Unit>
__global__ void broadcast_duplicate_rows(
    uint64_t slots, uint64_t row_bytes, const Id* ids, uint64_t count,
    const uint32_t* row_map, uint8_t* key_output, uint8_t* value_output) {
  const uint64_t row = blockIdx.x;
  if (row >= count) {
    return;
  }
  const int64_t id = load_id(ids, row);
  if (id < 0 || static_cast<uint64_t>(id) >= slots) {
    return;
  }
  const uint32_t canonical = row_map[static_cast<uint64_t>(id)];
  if (canonical == static_cast<uint32_t>(row)) {
    return;
  }
  const uint64_t units = row_bytes / sizeof(Unit);
  Unit* key_destination =
      reinterpret_cast<Unit*>(key_output + row * row_bytes);
  Unit* value_destination =
      reinterpret_cast<Unit*>(value_output + row * row_bytes);
  const Unit* key_source = reinterpret_cast<const Unit*>(
      key_output + static_cast<uint64_t>(canonical) * row_bytes);
  const Unit* value_source = reinterpret_cast<const Unit*>(
      value_output + static_cast<uint64_t>(canonical) * row_bytes);
  for (uint64_t column = threadIdx.x; column < units;
       column += blockDim.x) {
    key_destination[column] = key_source[column];
    value_destination[column] = value_source[column];
  }
}

template <typename Id>
cudaError_t launch_scatter(uint8_t* layer_base, uint64_t slots,
                           uint64_t row_bytes, const void* keys,
                           uint64_t key_stride, const void* values,
                           uint64_t value_stride, const void* ids,
                           uint64_t count, DeviceError* error,
                           cudaStream_t stream) {
  const bool vectorized =
      row_bytes % sizeof(uint4) == 0 && key_stride % sizeof(uint4) == 0 &&
      value_stride % sizeof(uint4) == 0 &&
      reinterpret_cast<uintptr_t>(layer_base) % alignof(uint4) == 0 &&
      reinterpret_cast<uintptr_t>(keys) % alignof(uint4) == 0 &&
      reinterpret_cast<uintptr_t>(values) % alignof(uint4) == 0;
  if (vectorized) {
    scatter_rows<Id, uint4><<<static_cast<unsigned>(count), 128, 0, stream>>>(
        layer_base, slots, row_bytes, static_cast<const uint8_t*>(keys),
        key_stride, static_cast<const uint8_t*>(values), value_stride,
        static_cast<const Id*>(ids), count, error);
  } else {
    // Byte fallback is required for FP8 rows and arbitrary legal leading
    // strides, including odd row widths and unaligned row starts.
    scatter_rows<Id, uint8_t>
        <<<static_cast<unsigned>(count), 128, 0, stream>>>(
            layer_base, slots, row_bytes, static_cast<const uint8_t*>(keys),
            key_stride, static_cast<const uint8_t*>(values), value_stride,
            static_cast<const Id*>(ids), count, error);
  }
  return cudaGetLastError();
}

template <typename Id>
cudaError_t launch_gather(const uint8_t* layer_base, uint64_t slots,
                          uint64_t row_bytes, const void* ids, uint64_t count,
                          void* key_output, void* value_output,
                          DeviceError* error, cudaStream_t stream) {
  const bool vectorized =
      row_bytes % sizeof(uint4) == 0 &&
      reinterpret_cast<uintptr_t>(layer_base) % alignof(uint4) == 0 &&
      reinterpret_cast<uintptr_t>(key_output) % alignof(uint4) == 0 &&
      reinterpret_cast<uintptr_t>(value_output) % alignof(uint4) == 0;
  if (vectorized) {
    gather_rows<Id, uint4><<<static_cast<unsigned>(count), 128, 0, stream>>>(
        layer_base, slots, row_bytes, static_cast<const Id*>(ids), count,
        static_cast<uint8_t*>(key_output), static_cast<uint8_t*>(value_output),
        error);
  } else {
    gather_rows<Id, uint8_t>
        <<<static_cast<unsigned>(count), 128, 0, stream>>>(
            layer_base, slots, row_bytes, static_cast<const Id*>(ids), count,
            static_cast<uint8_t*>(key_output),
            static_cast<uint8_t*>(value_output), error);
  }
  return cudaGetLastError();
}

template <typename Id>
cudaError_t launch_gather_dedup(
    const uint8_t* layer_base, uint64_t slots, uint64_t row_bytes,
    const void* ids, uint64_t count, void* row_map, void* key_output,
    void* value_output, DeviceError* error, cudaStream_t stream) {
  auto* typed_map = static_cast<uint32_t*>(row_map);
  // UINT32_MAX is the unelected sentinel. The unsigned atomic minimum makes
  // row zero (and every later row) strictly smaller than the reset value.
  cudaError_t status = cudaMemsetAsync(
      typed_map, 0xFF, static_cast<size_t>(slots) * sizeof(uint32_t), stream);
  if (status != cudaSuccess) {
    return status;
  }
  constexpr unsigned kMapThreads = 256;
  const auto map_blocks = static_cast<unsigned>(
      (count + static_cast<uint64_t>(kMapThreads) - 1) / kMapThreads);
  mark_canonical_rows<Id><<<map_blocks, kMapThreads, 0, stream>>>(
      static_cast<const Id*>(ids), count, slots, typed_map);
  status = cudaGetLastError();
  if (status != cudaSuccess) {
    return status;
  }

  const bool vectorized =
      row_bytes % sizeof(uint4) == 0 &&
      reinterpret_cast<uintptr_t>(layer_base) % alignof(uint4) == 0 &&
      reinterpret_cast<uintptr_t>(key_output) % alignof(uint4) == 0 &&
      reinterpret_cast<uintptr_t>(value_output) % alignof(uint4) == 0;
  if (vectorized) {
    gather_canonical_rows<Id, uint4>
        <<<static_cast<unsigned>(count), 128, 0, stream>>>(
            layer_base, slots, row_bytes, static_cast<const Id*>(ids), count,
            typed_map, static_cast<uint8_t*>(key_output),
            static_cast<uint8_t*>(value_output), error);
  } else {
    gather_canonical_rows<Id, uint8_t>
        <<<static_cast<unsigned>(count), 128, 0, stream>>>(
            layer_base, slots, row_bytes, static_cast<const Id*>(ids), count,
            typed_map, static_cast<uint8_t*>(key_output),
            static_cast<uint8_t*>(value_output), error);
  }
  status = cudaGetLastError();
  if (status != cudaSuccess) {
    return status;
  }
  if (vectorized) {
    broadcast_duplicate_rows<Id, uint4>
        <<<static_cast<unsigned>(count), 128, 0, stream>>>(
            slots, row_bytes, static_cast<const Id*>(ids), count, typed_map,
            static_cast<uint8_t*>(key_output),
            static_cast<uint8_t*>(value_output));
  } else {
    broadcast_duplicate_rows<Id, uint8_t>
        <<<static_cast<unsigned>(count), 128, 0, stream>>>(
            slots, row_bytes, static_cast<const Id*>(ids), count, typed_map,
            static_cast<uint8_t*>(key_output),
            static_cast<uint8_t*>(value_output));
  }
  return cudaGetLastError();
}

}  // namespace

__attribute__((visibility("hidden"))) cudaError_t q38_launch_host_kv_scatter(
    uint8_t* layer_base, uint64_t slots, uint64_t row_bytes,
    const void* keys, uint64_t key_stride, const void* values,
    uint64_t value_stride, const void* ids, uint64_t count, int id_bytes,
    void* error_pointer, cudaStream_t stream) {
  if (count == 0) {
    return cudaSuccess;
  }
  auto* error = static_cast<DeviceError*>(error_pointer);
  return id_bytes == 4
             ? launch_scatter<int32_t>(layer_base, slots, row_bytes, keys,
                                       key_stride, values, value_stride, ids,
                                       count, error, stream)
             : launch_scatter<int64_t>(layer_base, slots, row_bytes, keys,
                                       key_stride, values, value_stride, ids,
                                       count, error, stream);
}

__attribute__((visibility("hidden"))) cudaError_t q38_launch_host_kv_gather(
    const uint8_t* layer_base, uint64_t slots, uint64_t row_bytes,
    const void* ids, uint64_t count, int id_bytes, void* key_output,
    void* value_output, void* error_pointer, cudaStream_t stream) {
  if (count == 0) {
    return cudaSuccess;
  }
  auto* error = static_cast<DeviceError*>(error_pointer);
  return id_bytes == 4
             ? launch_gather<int32_t>(layer_base, slots, row_bytes, ids, count,
                                      key_output, value_output, error, stream)
             : launch_gather<int64_t>(layer_base, slots, row_bytes, ids, count,
                                      key_output, value_output, error, stream);
}

__attribute__((visibility("hidden"))) cudaError_t
q38_launch_host_kv_gather_dedup(
    const uint8_t* layer_base, uint64_t slots, uint64_t row_bytes,
    const void* ids, uint64_t count, int id_bytes, void* row_map,
    void* key_output, void* value_output, void* error_pointer,
    cudaStream_t stream) {
  if (count == 0) {
    return cudaSuccess;
  }
  auto* error = static_cast<DeviceError*>(error_pointer);
  return id_bytes == 4
             ? launch_gather_dedup<int32_t>(
                   layer_base, slots, row_bytes, ids, count, row_map,
                   key_output, value_output, error, stream)
             : launch_gather_dedup<int64_t>(
                   layer_base, slots, row_bytes, ids, count, row_map,
                   key_output, value_output, error, stream);
}

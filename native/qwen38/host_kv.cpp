// Copyright 2026 Qwen38 native contributors. SPDX-License-Identifier: Apache-2.0
#include "host_kv.h"

#include <cuda_runtime.h>

#include <cstring>
#include <limits>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>

__attribute__((visibility("hidden"))) cudaError_t q38_launch_host_kv_scatter(
    uint8_t* layer_base, uint64_t slots, uint64_t row_bytes,
    const void* keys, uint64_t key_stride, const void* values,
    uint64_t value_stride, const void* ids, uint64_t count, int id_bytes,
    void* error, cudaStream_t stream);

__attribute__((visibility("hidden"))) cudaError_t q38_launch_host_kv_gather(
    const uint8_t* layer_base, uint64_t slots, uint64_t row_bytes,
    const void* ids, uint64_t count, int id_bytes, void* key_output,
    void* value_output, void* error, cudaStream_t stream);

__attribute__((visibility("hidden"))) cudaError_t
q38_launch_host_kv_gather_dedup(const uint8_t* layer_base, uint64_t slots,
                                uint64_t row_bytes, const void* ids,
                                uint64_t count, int id_bytes, void* row_map,
                                void* key_output, void* value_output,
                                void* error, cudaStream_t stream);

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

thread_local std::string last_error;

void check(cudaError_t status) {
  if (status != cudaSuccess) {
    throw std::runtime_error(cudaGetErrorString(status));
  }
}

void require(bool condition, const char* message) {
  if (!condition) {
    throw std::invalid_argument(message);
  }
}

struct PointerRange {
  uintptr_t begin;
  uintptr_t end;
};

PointerRange checked_pointer_range(const void* pointer, uint64_t bytes,
                                   size_t alignment, const char* null_message,
                                   const char* alignment_message,
                                   const char* overflow_message) {
  require(pointer, null_message);
  const uintptr_t begin = reinterpret_cast<uintptr_t>(pointer);
  require(begin % alignment == 0, alignment_message);
  require(bytes <= std::numeric_limits<uintptr_t>::max() - begin,
          overflow_message);
  return {begin, begin + static_cast<uintptr_t>(bytes)};
}

bool disjoint(PointerRange left, PointerRange right) {
  return left.end <= right.begin || right.end <= left.begin;
}

uint64_t checked_multiply(uint64_t left, uint64_t right) {
  require(left != 0 && right != 0, "host KV geometry values must be positive");
  require(left <= std::numeric_limits<uint64_t>::max() / right,
          "host KV byte size overflows uint64");
  return left * right;
}

uint64_t required_bytes(uint64_t layers, uint64_t slots, uint64_t heads,
                        uint64_t head_dim, uint64_t element_bytes) {
  require(element_bytes == 1 || element_bytes == 2,
          "host KV element size must be 1 or 2 bytes");
  uint64_t result = checked_multiply(layers, slots);
  result = checked_multiply(result, 2);  // K and V.
  result = checked_multiply(result, heads);
  result = checked_multiply(result, head_dim);
  result = checked_multiply(result, element_bytes);
  require(result <= std::numeric_limits<size_t>::max(),
          "host KV byte size overflows size_t");
  return result;
}

template <class F>
int protect(F&& fn) noexcept {
  try {
    last_error.clear();
    fn();
    return 0;
  } catch (const std::exception& error) {
    last_error = error.what();
    return -1;
  } catch (...) {
    last_error = "unknown host KV exception";
    return -1;
  }
}

void validate_device_pointer(const void* pointer, int device,
                             size_t alignment, const char* name) {
  require(pointer, name);
  require(reinterpret_cast<uintptr_t>(pointer) % alignment == 0,
          "host KV device pointer alignment is invalid");
  cudaPointerAttributes attributes{};
  const cudaError_t status = cudaPointerGetAttributes(&attributes, pointer);
  if (status != cudaSuccess) {
    // Clear CUDA's per-thread error state before returning an argument error.
    cudaGetLastError();
    throw std::invalid_argument("host KV pointer is not CUDA-addressable");
  }
  require(attributes.type == cudaMemoryTypeDevice ||
              attributes.type == cudaMemoryTypeManaged,
          "host KV tensor/ID pointers must reference CUDA device memory");
  require(attributes.type == cudaMemoryTypeManaged || attributes.device == device,
          "host KV pointer is on the wrong CUDA device");
}

class Arena {
 public:
  const uint64_t layers;
  const uint64_t slots;
  const uint64_t heads;
  const uint64_t head_dim;
  const uint64_t element_bytes;
  const uint64_t row_bytes;
  const uint64_t layer_bytes;
  const uint64_t allocation_bytes;
  const int device;

  uint8_t* host_base = nullptr;
  uint8_t* device_base = nullptr;
  DeviceError* device_error = nullptr;
  cudaEvent_t last_use = nullptr;
  cudaStream_t last_stream = nullptr;
  bool use_recorded = false;
  bool poisoned = false;
  std::mutex mutex;

  Arena(uint64_t layer_count, uint64_t slot_count, uint64_t head_count,
        uint64_t dimension, uint64_t storage_element_bytes, uint64_t budget,
        int cuda_device)
      : layers(layer_count),
        slots(slot_count),
        heads(head_count),
        head_dim(dimension),
        element_bytes(storage_element_bytes),
        row_bytes(checked_multiply(checked_multiply(head_count, dimension),
                                   storage_element_bytes)),
        layer_bytes(checked_multiply(checked_multiply(slot_count, 2), row_bytes)),
        allocation_bytes(required_bytes(layer_count, slot_count, head_count,
                                        dimension, storage_element_bytes)),
        device(cuda_device) {
    require(budget > 0, "host KV byte budget must be explicit and positive");
    require(allocation_bytes <= budget,
            "host KV arena exceeds the explicit byte budget");
    check(cudaSetDevice(device));
    cudaDeviceProp properties{};
    check(cudaGetDeviceProperties(&properties, device));
    require(properties.major == 12 && properties.minor == 0,
            "host KV arena is validated only for SM120");
    require(properties.canMapHostMemory,
            "CUDA device cannot map pinned host memory");
    try {
      check(cudaHostAlloc(reinterpret_cast<void**>(&host_base),
                          static_cast<size_t>(allocation_bytes),
                          cudaHostAllocMapped));
      // Slot 0 is SGLang's reserved padding slot.  Initialize only its K/V
      // rows for every layer; touching an entire multi-GiB arena here would
      // add needless startup bandwidth and NUMA placement side effects.
      for (uint64_t layer = 0; layer < layers; ++layer) {
        std::memset(host_base + layer * layer_bytes, 0,
                    static_cast<size_t>(2 * row_bytes));
      }
      check(cudaHostGetDevicePointer(reinterpret_cast<void**>(&device_base),
                                     host_base, 0));
      check(cudaMalloc(reinterpret_cast<void**>(&device_error),
                       sizeof(DeviceError)));
      check(cudaMemset(device_error, 0, sizeof(DeviceError)));
      check(cudaEventCreateWithFlags(&last_use, cudaEventDisableTiming));
    } catch (...) {
      release();
      throw;
    }
  }

  ~Arena() {
    check_noexcept(cudaSetDevice(device));
    if (use_recorded && last_use) {
      check_noexcept(cudaEventSynchronize(last_use));
    }
    release();
  }

  void close() {
    std::lock_guard<std::mutex> lock(mutex);
    check(cudaSetDevice(device));
    if (use_recorded && last_use) {
      check(cudaEventSynchronize(last_use));
      use_recorded = false;
    }
    release();
  }

  static void check_noexcept(cudaError_t status) noexcept {
    // Destructors cannot report a second error.  cudaFree/cudaFreeHost still
    // provide the final context synchronization required before unmapping.
    (void)status;
  }

  void release() noexcept {
    if (last_use) {
      check_noexcept(cudaEventDestroy(last_use));
    }
    if (device_error) {
      check_noexcept(cudaFree(device_error));
    }
    if (host_base) {
      check_noexcept(cudaFreeHost(host_base));
    }
    last_use = nullptr;
    device_error = nullptr;
    device_base = nullptr;
    host_base = nullptr;
  }

  uint8_t* layer_base(uint64_t layer) const {
    require(layer < layers, "host KV layer index is out of bounds");
    return device_base + layer * layer_bytes;
  }

  bool validate_common(uint64_t layer, const void* ids, uint64_t count,
                       int id_bytes, cudaStream_t stream) {
    require(!poisoned, "host KV arena is poisoned by an enqueue failure");
    require(layer < layers, "host KV layer index is out of bounds");
    require(id_bytes == 4 || id_bytes == 8,
            "host KV IDs must be signed int32 or int64");
    if (count == 0) {
      return false;
    }
    require(count <= static_cast<uint64_t>(std::numeric_limits<unsigned>::max()),
            "host KV row count exceeds the CUDA launch bound");
    cudaStreamCaptureStatus capture_status;
    check(cudaStreamIsCapturing(stream, &capture_status));
    const bool capturing = capture_status != cudaStreamCaptureStatusNone;
    if (capturing) {
      // cudaSetDevice, pointer-attribute queries, and event queries are not
      // capture-safe.  The Python owner has already checked tensor type,
      // placement, shape, and lifetime before entering this C ABI call.
      return true;
    }
    check(cudaSetDevice(device));
    validate_device_pointer(ids, device, static_cast<size_t>(id_bytes),
                            "host KV IDs pointer is null");
    if (use_recorded && stream != last_stream) {
      // This creates a chain between eager streams.  A graph containing this
      // arena must itself be replayed serially on one stream, as documented.
      const cudaError_t state = cudaEventQuery(last_use);
      if (state == cudaErrorNotReady) {
        check(cudaStreamWaitEvent(stream, last_use, 0));
      } else {
        check(state);
      }
    }
    return false;
  }

  void record_use(cudaStream_t stream) {
    const cudaError_t status = cudaEventRecord(last_use, stream);
    if (status != cudaSuccess) {
      poisoned = true;
      // If capture has already been invalidated this may fail too, but an
      // eager enqueue must be drained before mapped backing can be released.
      cudaStreamSynchronize(stream);
      check(status);
    }
    last_stream = stream;
    use_recorded = true;
  }

  void scatter(uint64_t layer, const void* keys, uint64_t key_stride,
               const void* values, uint64_t value_stride, const void* ids,
               uint64_t count, int id_bytes, cudaStream_t stream) {
    std::lock_guard<std::mutex> lock(mutex);
    const bool capturing =
        validate_common(layer, ids, count, id_bytes, stream);
    require(key_stride >= row_bytes && value_stride >= row_bytes,
            "host KV input row stride is smaller than one storage row");
    require(key_stride % element_bytes == 0 &&
                value_stride % element_bytes == 0,
            "host KV input row stride is misaligned for its storage dtype");
    if (count == 0) {
      return;
    }
    if (!capturing) {
      validate_device_pointer(keys, device, static_cast<size_t>(element_bytes),
                              "host KV key input pointer is null");
      validate_device_pointer(values, device,
                              static_cast<size_t>(element_bytes),
                              "host KV value input pointer is null");
    }
    check(q38_launch_host_kv_scatter(
        layer_base(layer), slots, row_bytes, keys, key_stride, values,
        value_stride, ids, count, id_bytes, device_error, stream));
    if (!capturing) {
      record_use(stream);
    }
  }

  void gather(uint64_t layer, const void* ids, uint64_t count, int id_bytes,
              void* key_output, void* value_output, cudaStream_t stream) {
    std::lock_guard<std::mutex> lock(mutex);
    const bool capturing =
        validate_common(layer, ids, count, id_bytes, stream);
    if (count == 0) {
      return;
    }
    if (!capturing) {
      validate_device_pointer(key_output, device,
                              static_cast<size_t>(element_bytes),
                              "host KV key output pointer is null");
      validate_device_pointer(value_output, device,
                              static_cast<size_t>(element_bytes),
                              "host KV value output pointer is null");
    }
    check(q38_launch_host_kv_gather(
        layer_base(layer), slots, row_bytes, ids, count, id_bytes, key_output,
        value_output, device_error, stream));
    if (!capturing) {
      record_use(stream);
    }
  }

  void gather_dedup(uint64_t layer, const void* ids, uint64_t count,
                    int id_bytes, void* row_map, uint64_t row_map_elements,
                    void* key_output, void* value_output,
                    cudaStream_t stream) {
    std::lock_guard<std::mutex> lock(mutex);
    const bool capturing =
        validate_common(layer, ids, count, id_bytes, stream);
    require(count <= static_cast<uint64_t>(std::numeric_limits<int>::max()),
            "host KV dedup row count exceeds the CUDA grid bound");
    if (count == 0) {
      return;
    }
    require(count <= std::numeric_limits<size_t>::max() /
                         static_cast<uint64_t>(id_bytes),
            "host KV dedup ID byte size overflows size_t");
    require(count <= std::numeric_limits<size_t>::max() / row_bytes,
            "host KV dedup output byte size overflows size_t");
    require(slots <= std::numeric_limits<size_t>::max() / sizeof(uint32_t),
            "host KV dedup row-map byte size overflows size_t");
    require(row_map_elements >= slots,
            "host KV dedup row map is smaller than slot_count");
    const uint64_t id_bytes_total = count * static_cast<uint64_t>(id_bytes);
    const uint64_t output_bytes = count * row_bytes;
    const uint64_t row_map_bytes = slots * sizeof(uint32_t);
    const PointerRange id_range = checked_pointer_range(
        ids, id_bytes_total, static_cast<size_t>(id_bytes),
        "host KV IDs pointer is null", "host KV ID pointer alignment is invalid",
        "host KV dedup ID pointer range overflows uintptr");
    const PointerRange map_range = checked_pointer_range(
        row_map, row_map_bytes, alignof(uint32_t),
        "host KV dedup row map pointer is null",
        "host KV dedup row map pointer alignment is invalid",
        "host KV dedup row map pointer range overflows uintptr");
    const PointerRange key_range = checked_pointer_range(
        key_output, output_bytes, static_cast<size_t>(element_bytes),
        "host KV key output pointer is null",
        "host KV key output pointer alignment is invalid",
        "host KV dedup key output pointer range overflows uintptr");
    const PointerRange value_range = checked_pointer_range(
        value_output, output_bytes, static_cast<size_t>(element_bytes),
        "host KV value output pointer is null",
        "host KV value output pointer alignment is invalid",
        "host KV dedup value output pointer range overflows uintptr");
    require(disjoint(id_range, map_range) && disjoint(id_range, key_range) &&
                disjoint(id_range, value_range) &&
                disjoint(map_range, key_range) &&
                disjoint(map_range, value_range) &&
                disjoint(key_range, value_range),
            "host KV dedup IDs, row map, and outputs must not overlap");
    if (!capturing) {
      validate_device_pointer(row_map, device, alignof(uint32_t),
                              "host KV dedup row map pointer is null");
      validate_device_pointer(key_output, device,
                              static_cast<size_t>(element_bytes),
                              "host KV key output pointer is null");
      validate_device_pointer(value_output, device,
                              static_cast<size_t>(element_bytes),
                              "host KV value output pointer is null");
    }
    const cudaError_t status = q38_launch_host_kv_gather_dedup(
        layer_base(layer), slots, row_bytes, ids, count, id_bytes, row_map,
        key_output, value_output, device_error, stream);
    if (!capturing) {
      // Record after every attempted multi-kernel enqueue, including failure.
      // Earlier kernels may already be reading mapped host memory.
      record_use(stream);
    }
    if (status != cudaSuccess) {
      poisoned = true;
    }
    check(status);
  }

  void check_errors() {
    std::lock_guard<std::mutex> lock(mutex);
    check(cudaSetDevice(device));
    if (use_recorded) {
      check(cudaEventSynchronize(last_use));
    }
    DeviceError error{};
    check(cudaMemcpy(&error, device_error, sizeof(error),
                     cudaMemcpyDeviceToHost));
    check(cudaMemset(device_error, 0, sizeof(DeviceError)));
    if (error.code != 0) {
      std::ostringstream message;
      message << (error.code == kScatterOutOfBounds ? "scatter" : "gather")
              << " physical slot ID out of bounds at input row " << error.row
              << ": " << error.id << " not in ";
      if (error.code == kGatherOutOfBounds) {
        message << "negative-padding or ";
      }
      message << "[0, " << slots << ')';
      throw std::out_of_range(message.str());
    }
  }
};

}  // namespace

extern "C" const char* q38_host_kv_last_error(void) {
  return last_error.c_str();
}

extern "C" uint32_t q38_host_kv_abi_version(void) {
  return Q38_HOST_KV_ABI_VERSION;
}

extern "C" int q38_host_kv_required_bytes(uint64_t layers, uint64_t slots,
                                            uint64_t heads, uint64_t head_dim,
                                            uint64_t element_bytes,
                                            uint64_t* result) {
  return protect([&] {
    require(result, "host KV required-bytes output pointer is null");
    *result = required_bytes(layers, slots, heads, head_dim, element_bytes);
  });
}

extern "C" void* q38_host_kv_create(uint64_t layers, uint64_t slots,
                                      uint64_t heads, uint64_t head_dim,
                                      uint64_t element_bytes, uint64_t budget,
                                      int device) {
  Arena* arena = nullptr;
  if (protect([&] {
        arena = new Arena(layers, slots, heads, head_dim, element_bytes,
                          budget, device);
      }) != 0) {
    return nullptr;
  }
  return arena;
}

extern "C" uint64_t q38_host_kv_allocated_bytes(void* handle) {
  uint64_t result = 0;
  protect([&] {
    require(handle, "host KV arena is closed");
    result = static_cast<Arena*>(handle)->allocation_bytes;
  });
  return result;
}

extern "C" int q38_host_kv_scatter(
    void* handle, uint64_t layer, const void* keys, uint64_t key_stride,
    const void* values, uint64_t value_stride, const void* ids, uint64_t count,
    int id_bytes, uintptr_t stream) {
  return protect([&] {
    require(handle, "host KV arena is closed");
    static_cast<Arena*>(handle)->scatter(
        layer, keys, key_stride, values, value_stride, ids, count, id_bytes,
        reinterpret_cast<cudaStream_t>(stream));
  });
}

extern "C" int q38_host_kv_gather(void* handle, uint64_t layer,
                                    const void* ids, uint64_t count,
                                    int id_bytes, void* key_output,
                                    void* value_output, uintptr_t stream) {
  return protect([&] {
    require(handle, "host KV arena is closed");
    static_cast<Arena*>(handle)->gather(
        layer, ids, count, id_bytes, key_output, value_output,
        reinterpret_cast<cudaStream_t>(stream));
  });
}

extern "C" int q38_host_kv_gather_dedup(
    void* handle, uint64_t layer, const void* ids, uint64_t count,
    int id_bytes, int32_t* row_map, uint64_t row_map_elements,
    void* key_output, void* value_output, uintptr_t stream) {
  return protect([&] {
    require(handle, "host KV arena is closed");
    static_cast<Arena*>(handle)->gather_dedup(
        layer, ids, count, id_bytes, row_map, row_map_elements, key_output,
        value_output, reinterpret_cast<cudaStream_t>(stream));
  });
}

extern "C" int q38_host_kv_check(void* handle) {
  return protect([&] {
    require(handle, "host KV arena is closed");
    static_cast<Arena*>(handle)->check_errors();
  });
}

extern "C" int q38_host_kv_close(void* handle) {
  return protect([&] {
    auto* arena = static_cast<Arena*>(handle);
    if (arena) {
      arena->close();
      delete arena;
    }
  });
}

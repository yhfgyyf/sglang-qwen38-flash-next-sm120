// Copyright 2026 Qwen38 native contributors. SPDX-License-Identifier: Apache-2.0
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>

namespace {
__global__ void decode_e4m3(const uint8_t* input, __nv_bfloat16* output,
                           size_t count) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count;
       i += static_cast<size_t>(gridDim.x) * blockDim.x) {
    __nv_fp8_e4m3 value;
    value.__x = input[i];
    output[i] = __float2bfloat16_rn(static_cast<float>(value));
  }
}
}

cudaError_t q38_launch_decode(const uint8_t* input, void* output, size_t count,
                              cudaStream_t stream) {
  if (!count) return cudaSuccess;
  const size_t blocks = (count + 255) / 256;
  decode_e4m3<<<static_cast<unsigned>(blocks > 4096 ? 4096 : blocks), 256, 0, stream>>>(
      input, static_cast<__nv_bfloat16*>(output), count);
  return cudaGetLastError();
}

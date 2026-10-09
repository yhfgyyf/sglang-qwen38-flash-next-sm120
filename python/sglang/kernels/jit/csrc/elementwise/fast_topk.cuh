// Radix-select fast top-k, adapted from sgl-kernel's AOT topk.cu (itself
// adapted from tilelang's topk_selector). Ported to the JIT layer so that
// kTopK = 512 support ships with the sglang python package instead of
// requiring an sgl-kernel wheel release.
//
// Default semantics match the AOT fast_topk_v2 op: for each row b, select
// the kTopK largest scores in [row_starts[b], row_starts[b] + lengths[b])
// and write their indices relative to row_starts[b]. Output order within a
// row is unspecified (atomic collection order). The opt-in stable-tie
// specialization chooses the lowest relative indices at an equal cutoff.
#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>
#include <sgl_kernel/utils.cuh>

#include <tvm/ffi/container/tensor.h>

namespace sglang {

namespace fast_topk_detail {

constexpr uint32_t kThreadsPerBlock = 1024;
// Stage up to 4K threshold-bin indices per round. Rows whose coarse threshold
// bin is larger use an exact full-row radix rescan instead.
constexpr size_t kSmemBytes = 8 * 1024 * sizeof(uint32_t);  // 32KB

struct FastTopKParams {
  const float* __restrict__ input;         // [B, input_stride]
  const int32_t* __restrict__ row_starts;  // [B]
  int32_t* __restrict__ indices;           // [B, kTopK]
  const int32_t* __restrict__ lengths;     // [B]
  int64_t input_stride;
};

template <bool kStableTies>
SGL_DEVICE auto normalize_zero(float x) -> float {
  if constexpr (kStableTies) {
    // IEEE -0.0f and +0.0f compare equal, but their radix keys differ.
    // Stable tie selection treats them as one numeric cutoff group.
    if (x == 0.0f) {
      return 0.0f;
    }
  }
  return x;
}

template <bool kStableTies>
SGL_DEVICE auto convert_to_uint8(float x) -> uint8_t {
  x = normalize_zero<kStableTies>(x);
  const __half h = __float2half_rn(x);
  const uint16_t bits = __half_as_ushort(h);
  const uint16_t key = (bits & 0x8000) ? static_cast<uint16_t>(~bits)
                                       : static_cast<uint16_t>(bits | 0x8000);
  return static_cast<uint8_t>(key >> 8);
}

template <bool kStableTies>
SGL_DEVICE auto convert_to_uint32(float x) -> uint32_t {
  x = normalize_zero<kStableTies>(x);
  const uint32_t bits = __float_as_uint(x);
  return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
}

// When length <= kTopK, write the indices directly.
template <int kTopK>
SGL_DEVICE void naive_topk(
    const float* __restrict__ score, int32_t* __restrict__ indice, int32_t length) {
  const auto tid = threadIdx.x;
  for (int i = tid; i < kTopK; i += kThreadsPerBlock) {
    indice[i] = (i < length) ? i : -1;
  }
}

template <int kTopK, bool kStableTies>
SGL_DEVICE void collect_stable_threshold_ties(
    const float* __restrict__ input,
    int* __restrict__ index,
    int row_start,
    int length,
    uint32_t threshold_key,
    int num_ties,
    int* warp_offsets,
    int* scan_state) {
  constexpr auto BLOCK_SIZE = kThreadsPerBlock;
  constexpr auto WARP_SIZE = 32;
  constexpr auto NUM_WARPS = BLOCK_SIZE / WARP_SIZE;
  const auto tx = threadIdx.x;
  const auto lane = tx % WARP_SIZE;
  const auto warp = tx / WARP_SIZE;

  if (tx == 0) {
    scan_state[0] = 0;
  }
  __syncthreads();

  // Visit chunks in increasing relative-index order. Warp ballots and a
  // block-wide prefix assign each matching score its deterministic rank in
  // that order without an O(length * num_ties) per-item rank computation.
  for (int chunk_start = 0; chunk_start < length; chunk_start += BLOCK_SIZE) {
    const auto idx = chunk_start + tx;
    const auto matches =
        idx < length &&
        convert_to_uint32<kStableTies>(input[idx + row_start]) == threshold_key;
    const auto matches_in_warp = __ballot_sync(0xFFFFFFFFu, matches);
    if (lane == 0) {
      warp_offsets[warp] = __popc(matches_in_warp);
    }
    __syncthreads();

    if (tx == 0) {
      const auto selected_before = scan_state[0];
      int chunk_count = 0;
#pragma unroll
      for (int i = 0; i < NUM_WARPS; ++i) {
        const auto warp_count = warp_offsets[i];
        warp_offsets[i] = chunk_count;
        chunk_count += warp_count;
      }
      scan_state[1] = selected_before;
      scan_state[0] = selected_before + chunk_count;
    }
    __syncthreads();

    const auto lower_lanes =
        lane == 0 ? 0u : ((uint32_t{1} << lane) - uint32_t{1});
    const auto rank_in_chunk =
        warp_offsets[warp] + __popc(matches_in_warp & lower_lanes);
    const auto selected_rank = scan_state[1] + rank_in_chunk;
    if (matches && selected_rank < num_ties) {
      index[kTopK - num_ties + selected_rank] = idx;
    }
    __syncthreads();

    if (scan_state[0] >= num_ties) {
      break;
    }
  }
}

// Radix-select top-k. Assumes length > kTopK (checked by the caller).
template <int kTopK, bool kStableTies>
SGL_DEVICE void radix_select_topk(
    const float* __restrict__ input, int* __restrict__ index, int row_start, int length) {
  int topk = kTopK;
  constexpr auto BLOCK_SIZE = kThreadsPerBlock;
  constexpr auto RADIX = 256;
  constexpr auto SMEM_INPUT_SIZE = kSmemBytes / (2 * sizeof(int));

  alignas(128) __shared__ int s_histogram_buf[2][RADIX + 128];
  alignas(128) __shared__ int s_counter;
  alignas(128) __shared__ int s_threshold_bin_id;
  alignas(128) __shared__ int s_num_input[2];

  auto& s_histogram = s_histogram_buf[0];
  // allocate for two rounds
  extern __shared__ int s_input_idx[][SMEM_INPUT_SIZE];

  const int tx = threadIdx.x;

  // stage 1: 8bit coarse histogram
  if (tx < RADIX + 1) s_histogram[tx] = 0;
  __syncthreads();

  for (int idx = tx; idx < length; idx += BLOCK_SIZE) {
    const auto bin = convert_to_uint8<kStableTies>(input[idx + row_start]);
    ::atomicAdd(&s_histogram[bin], 1);
  }
  __syncthreads();

  const auto run_cumsum = [&] {
#pragma unroll 8
    for (int i = 0; i < 8; ++i) {
      static_assert(1 << 8 == RADIX);
      if (tx < RADIX) {
        const auto j = 1 << i;
        const auto k = i & 1;
        auto value = s_histogram_buf[k][tx];
        if (tx < RADIX - j) {
          value += s_histogram_buf[k][tx + j];
        }
        s_histogram_buf[k ^ 1][tx] = value;
      }
      __syncthreads();
    }
  };

  run_cumsum();
  if (tx < RADIX && s_histogram[tx] > topk && s_histogram[tx + 1] <= topk) {
    s_threshold_bin_id = tx;
    s_num_input[0] = 0;
    s_counter = 0;
  }
  __syncthreads();

  const auto threshold_bin = s_threshold_bin_id;
  topk -= s_histogram[threshold_bin + 1];

  if (topk == 0) {
    for (int idx = tx; idx < length; idx += BLOCK_SIZE) {
      const auto bin =
          static_cast<int>(convert_to_uint8<kStableTies>(input[idx + row_start]));
      if (bin > threshold_bin) {
        const auto pos = ::atomicAdd(&s_counter, 1);
        index[pos] = idx;
      }
    }
    __syncthreads();
    return;
  } else {
    __syncthreads();
    if (tx < RADIX + 1) {
      s_histogram[tx] = 0;
    }
    __syncthreads();

    for (int idx = tx; idx < length; idx += BLOCK_SIZE) {
      const auto raw_input = input[idx + row_start];
      const auto bin = static_cast<int>(convert_to_uint8<kStableTies>(raw_input));
      if (bin > threshold_bin) {
        const auto pos = ::atomicAdd(&s_counter, 1);
        index[pos] = idx;
      } else if (bin == threshold_bin) {
        const auto pos = ::atomicAdd(&s_num_input[0], 1);
        // fuse the histogram computation here
        if (pos < int(SMEM_INPUT_SIZE)) {
          s_input_idx[0][pos] = idx;
          const auto bin = convert_to_uint32<kStableTies>(raw_input);
          const auto sub_bin = (bin >> 24) & 0xFF;
          ::atomicAdd(&s_histogram[sub_bin], 1);
        }
      }
    }
    __syncthreads();
  }

  // The staged refinement below is exact only while every candidate fits in
  // s_input_idx. A coarse FP16 bin can be arbitrarily large (for example,
  // many distinct FP32 values in [1.0, 1.1]). On overflow, identify the exact
  // FP32 threshold with four full-row radix rescans. This keeps the common
  // staged path unchanged and requires no global workspace or host decision.
  if (s_num_input[0] > int(SMEM_INPUT_SIZE)) {
    int remaining = kTopK;
    int threshold_count = 0;
    uint32_t threshold_key = 0;

#pragma unroll 4
    for (int round = 0; round < 4; ++round) {
      if (tx < RADIX + 1) {
        s_histogram[tx] = 0;
      }
      __syncthreads();

      for (int idx = tx; idx < length; idx += BLOCK_SIZE) {
        const auto key = convert_to_uint32<kStableTies>(input[idx + row_start]);
        bool matches_prefix = true;
        if (round != 0) {
          matches_prefix = (key >> (32 - round * 8)) == threshold_key;
        }
        if (matches_prefix) {
          const auto bin = (key >> (24 - round * 8)) & 0xFF;
          ::atomicAdd(&s_histogram[bin], 1);
        }
      }
      __syncthreads();

      run_cumsum();
      if (tx < RADIX && s_histogram[tx] > remaining &&
          s_histogram[tx + 1] <= remaining) {
        s_threshold_bin_id = tx;
      }
      __syncthreads();

      const auto threshold_bin = s_threshold_bin_id;
      if constexpr (kStableTies) {
        threshold_count =
            s_histogram[threshold_bin] - s_histogram[threshold_bin + 1];
      }
      remaining -= s_histogram[threshold_bin + 1];
      threshold_key = (threshold_key << 8) | threshold_bin;

      // If the requested count ends exactly at a radix-bin boundary, all
      // keys in the threshold bin must be excluded. Filling the unresolved
      // suffix with ones turns the final comparison into that boundary.
      if (remaining == 0) {
        const auto trailing_bits = 24 - round * 8;
        if (trailing_bits != 0) {
          threshold_key = (threshold_key << trailing_bits) |
                          ((uint32_t{1} << trailing_bits) - 1);
        }
        __syncthreads();
        break;
      }
      __syncthreads();
    }

    if (tx == 0) {
      s_counter = 0;
      s_num_input[0] = 0;
    }
    __syncthreads();

    const auto stable_tie_scan =
        kStableTies && remaining != 0 && threshold_count > remaining;
    for (int idx = tx; idx < length; idx += BLOCK_SIZE) {
      const auto key = convert_to_uint32<kStableTies>(input[idx + row_start]);
      if (key > threshold_key) {
        const auto pos = ::atomicAdd(&s_counter, 1);
        index[pos] = idx;
      } else if (
          key == threshold_key && remaining != 0 && !stable_tie_scan) {
        const auto equal_pos = ::atomicAdd(&s_num_input[0], 1);
        if (equal_pos < remaining) {
          index[kTopK - remaining + equal_pos] = idx;
        }
      }
    }
    __syncthreads();
    if constexpr (kStableTies) {
      if (stable_tie_scan) {
        collect_stable_threshold_ties<kTopK, kStableTies>(
            input,
            index,
            row_start,
            length,
            threshold_key,
            remaining,
            s_histogram,
            s_num_input);
      }
    }
    return;
  }

  // stage 2: refine with 8bit radix passes
  uint32_t threshold_key = 0;
  int threshold_count = 0;
#pragma unroll 4
  for (int round = 0; round < 4; ++round) {
    __shared__ int s_last_remain;
    const auto r_idx = round % 2;

    // clip here to prevent overflow
    const auto _raw_num_input = s_num_input[r_idx];
    const auto num_input = (_raw_num_input < int(SMEM_INPUT_SIZE))
                               ? _raw_num_input
                               : int(SMEM_INPUT_SIZE);

    run_cumsum();
    if (tx < RADIX && s_histogram[tx] > topk && s_histogram[tx + 1] <= topk) {
      s_threshold_bin_id = tx;
      s_num_input[r_idx ^ 1] = 0;
      s_last_remain = topk - s_histogram[tx + 1];
    }
    __syncthreads();

    const auto threshold_bin = s_threshold_bin_id;
    if constexpr (kStableTies) {
      threshold_key = (threshold_key << 8) | threshold_bin;
      if (round == 3) {
        threshold_count =
            s_histogram[threshold_bin] - s_histogram[threshold_bin + 1];
      }
    }
    topk -= s_histogram[threshold_bin + 1];

    if (topk == 0) {
      for (int i = tx; i < num_input; i += BLOCK_SIZE) {
        const auto idx = s_input_idx[r_idx][i];
        const auto offset = 24 - round * 8;
        const auto bin =
            (convert_to_uint32<kStableTies>(input[idx + row_start]) >> offset) &
            0xFF;
        if (bin > threshold_bin) {
          const auto pos = ::atomicAdd(&s_counter, 1);
          index[pos] = idx;
        }
      }
      __syncthreads();
      break;
    } else {
      __syncthreads();
      if (tx < RADIX + 1) {
        s_histogram[tx] = 0;
      }
      __syncthreads();
      const auto stable_tie_scan =
          kStableTies && round == 3 && threshold_count > topk;
      for (int i = tx; i < num_input; i += BLOCK_SIZE) {
        const auto idx = s_input_idx[r_idx][i];
        const auto raw_input = input[idx + row_start];
        const auto offset = 24 - round * 8;
        const auto bin =
            (convert_to_uint32<kStableTies>(raw_input) >> offset) & 0xFF;
        if (bin > threshold_bin) {
          const auto pos = ::atomicAdd(&s_counter, 1);
          index[pos] = idx;
        } else if (bin == threshold_bin) {
          if (round == 3) {
            if (!stable_tie_scan) {
              const auto pos = ::atomicAdd(&s_last_remain, -1);
              if (pos > 0) {
                index[kTopK - pos] = idx;
              }
            }
          } else {
            const auto pos = ::atomicAdd(&s_num_input[r_idx ^ 1], 1);
            if (pos < int(SMEM_INPUT_SIZE)) {
              // fuse the histogram computation here
              s_input_idx[r_idx ^ 1][pos] = idx;
              const auto bin = convert_to_uint32<kStableTies>(raw_input);
              const auto sub_bin = (bin >> (offset - 8)) & 0xFF;
              ::atomicAdd(&s_histogram[sub_bin], 1);
            }
          }
        }
      }
      __syncthreads();
      if constexpr (kStableTies) {
        if (stable_tie_scan) {
          collect_stable_threshold_ties<kTopK, kStableTies>(
              input,
              index,
              row_start,
              length,
              threshold_key,
              topk,
              s_histogram,
              s_num_input);
        }
      }
    }
  }
}

template <int kTopK, bool kUsePDL, bool kStableTies>
__global__ __launch_bounds__(fast_topk_detail::kThreadsPerBlock) void fast_topk_kernel(
    const fast_topk_detail::FastTopKParams __grid_constant__ params) {
  using namespace fast_topk_detail;
  device::PDLWaitPrimary<kUsePDL>();

  const auto bid = static_cast<uint64_t>(blockIdx.x);
  const auto row_start = params.row_starts == nullptr ? 0 : params.row_starts[bid];
  const auto length = params.lengths[bid];
  const auto indice = params.indices + bid * kTopK;
  const auto score = params.input + bid * params.input_stride;
  if (length <= kTopK) {
    naive_topk<kTopK>(score, indice, length);
  } else {
    radix_select_topk<kTopK, kStableTies>(score, indice, row_start, length);
  }

  device::PDLTriggerSecondary<kUsePDL>();
}

}  // namespace fast_topk_detail

/**
 * \brief Per-row top-k selection over ragged rows of a fp32 score matrix.
 *
 * Row b selects the kTopK largest values in
 * score[b, row_starts[b] : row_starts[b] + lengths[b]) and writes their
 * indices (relative to row_starts[b]) into indices[b]. Unfilled slots are
 * -1 when lengths[b] < kTopK.
 */
template <int kTopK, bool kUsePDL, bool kStableTies>
struct FastTopKKernel {
  static constexpr auto kernel =
      fast_topk_detail::fast_topk_kernel<kTopK, kUsePDL, kStableTies>;

  static void
  run(const tvm::ffi::TensorView score,
      const tvm::ffi::TensorView row_starts,
      const tvm::ffi::TensorView indices,
      const tvm::ffi::TensorView lengths) {
    using namespace host;
    auto B = SymbolicSize{"batch"};
    auto L = SymbolicSize{"length"};
    auto S = SymbolicSize{"input_stride"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();

    TensorMatcher({B, L})  // score
        .with_strides({S, 1})
        .with_dtype<fp32_t>()
        .with_device(device)
        .verify(score);
    TensorMatcher({B})  // row_starts
        .with_dtype<int32_t>()
        .with_device(device)
        .verify(row_starts);
    TensorMatcher({B, kTopK})  // indices
        .with_dtype<int32_t>()
        .with_device(device)
        .verify(indices);
    TensorMatcher({B})  // lengths
        .with_dtype<int32_t>()
        .with_device(device)
        .verify(lengths);

    const auto params = fast_topk_detail::FastTopKParams{
        .input = static_cast<const float*>(score.data_ptr()),
        .row_starts = static_cast<const int32_t*>(row_starts.data_ptr()),
        .indices = static_cast<int32_t*>(indices.data_ptr()),
        .lengths = static_cast<const int32_t*>(lengths.data_ptr()),
        .input_stride = S.unwrap(),
    };

    const auto num_rows = static_cast<uint32_t>(B.unwrap());
    LaunchKernel(
        num_rows,
        fast_topk_detail::kThreadsPerBlock,
        device.unwrap(),
        fast_topk_detail::kSmemBytes)
        .enable_pdl(kUsePDL)(kernel, params);
  }
};

}  // namespace sglang

#!/usr/bin/env bash
set -euo pipefail

model_path="${MODEL_PATH:?Set MODEL_PATH to the Qwen3.8-Flash-Next-NVFP4 checkpoint directory}"
server_host="${SERVER_HOST:-127.0.0.1}"
server_port="${SERVER_PORT:-30000}"
if [[ -n "${CUDA_TOOLKIT:-}" ]]; then
  cuda_toolkit="$CUDA_TOOLKIT"
else
  nvcc_path="$(command -v nvcc || true)"
  if [[ -z "$nvcc_path" ]]; then
    echo "nvcc was not found; set CUDA_TOOLKIT to the CUDA 13.2 toolkit directory" >&2
    exit 2
  fi
  cuda_toolkit="$(cd "$(dirname "$nvcc_path")/.." && pwd -P)"
fi
python_bin="${SGLANG_PYTHON:-python}"
context_length="${CONTEXT_LENGTH:-262144}"
max_total_tokens="${MAX_TOTAL_TOKENS:-$context_length}"
mem_fraction_static="${MEM_FRACTION_STATIC:-0.98}"

if [[ ! -x "$cuda_toolkit/bin/nvcc" ]]; then
  echo "CUDA Toolkit is invalid: bin/nvcc is missing or not executable" >&2
  exit 2
fi

export CUDA_HOME="$cuda_toolkit"
export CUDA_PATH="$cuda_toolkit"
export CUDACXX="$cuda_toolkit/bin/nvcc"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export FLASHINFER_CUDA_ARCH_LIST="${FLASHINFER_CUDA_ARCH_LIST:-12.0f}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export SGLANG_QWEN4_PLE_NVME_PATH="$model_path"
export SGLANG_QWEN4_PLE_NVME_BACKEND="${SGLANG_QWEN4_PLE_NVME_BACKEND:-io_uring}"
export SGLANG_QWEN4_PLE_NVME_QUEUE_DEPTH="${SGLANG_QWEN4_PLE_NVME_QUEUE_DEPTH:-512}"
export SGLANG_QWEN4_PLE_NVME_MAX_BATCH_PAGES="${SGLANG_QWEN4_PLE_NVME_MAX_BATCH_PAGES:-4096}"
export SGLANG_QWEN4_PLE_NVME_CACHE_PAGES="${SGLANG_QWEN4_PLE_NVME_CACHE_PAGES:-5242880}"
export SGLANG_QWEN4_PLE_NVME_LOG_INTERVAL="${SGLANG_QWEN4_PLE_NVME_LOG_INTERVAL:-10}"

exec "$python_bin" -m sglang.launch_server \
  --model-path "$model_path" \
  --quantization modelopt_fp4 \
  --fp4-gemm-backend flashinfer_cutlass \
  --tp-size 1 \
  --dtype bfloat16 \
  --page-size 64 \
  --mamba-radix-cache-strategy extra_buffer \
  --mamba-track-interval 64 \
  --chunked-prefill-size 512 \
  --max-running-requests 1 \
  --context-length "$context_length" \
  --max-total-tokens "$max_total_tokens" \
  --mem-fraction-static "$mem_fraction_static" \
  --speculative-algorithm NEXTN \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4 \
  --cuda-graph-backend-decode breakable \
  --cuda-graph-backend-prefill tc_piecewise \
  --cuda-graph-max-bs-decode 1 \
  --cuda-graph-bs-decode 1 \
  --cuda-graph-max-bs-prefill 512 \
  --cuda-graph-bs-prefill 512 \
  --host "$server_host" \
  --port "$server_port" \
  "$@"

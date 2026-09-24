# sglang-qwen38-flash-next-sm120

[English](#sglang-qwen38-flash-next-sm120) | [中文版本](#中文版本)

This is an experimental SGLang fork for running `Qwen3.8-Flash-Next-NVFP4` on a single **NVIDIA RTX PRO 6000 Blackwell Workstation GPU (SM120, 96GB)**. It combines pinned upstream Qwen3.8, NVMe-backed PLE, and SM120 QSA sparse-decode implementations with Breakable CUDA Graph fixes, GDN projection/Conv1D fusion, hot-PLE row gathering, and QSA prefill tuning.

Updated **2026-09-24**. The latest published-source measurements are approximately **14.2K / 13.9K / 13.0K prefill tok/s** and **223 / 228 / 223 aggregate decode tok/s** for 8K / 32K / 128K input, respectively: concurrency 1, 512 output tokens, and warm PLE. See [Performance results](#performance-results) for the exact workload and statistics. The newer native sparse-GQA prototype is **not published or enabled in this repository**; its experimental results are reported separately.

> The current NVMe-backed PLE implementation supports one GPU with `TP=1` only. Multi-GPU PLE execution is not supported.

## What this project solves

Qwen3.8-Flash-Next contains approximately **51.2B** additional PLE (Per-Layer Embedding) parameters. In the tested checkpoint, the PLE tables use FP8 and occupy approximately **47.68GiB**.

The validated deployment uses the following configuration:

- The main model uses NVFP4 and runs on one SM120 GPU.
- PLE weights remain in the original safetensors files and are fetched from local NVMe with `io_uring + O_DIRECT`.
- A 4KiB-page application LRU keeps the active PLE working set in host memory.
- PLE reads are submitted ahead of use and overlap with preceding GPU computation when possible.
- CUDA Graph remains enabled for decode and speculative paths.
- NEXTN/EAGLE speculative decoding uses 3 steps, `topk=1`, and up to 4 draft tokens.

## SGLang pull requests used

| Upstream change | Pinned snapshot | Role in this repository |
| --- | --- | --- |
| [PR #36497 — Introduce Qwen 3.8 Flash Next](https://github.com/sgl-project/sglang/pull/36497) | `73a255206f` | Qwen3.8 model, GDN/QSA, GR, PLE, MTP, quantization, and multimodal support |
| [PR #36567 — Stream PLE embeddings from NVMe](https://github.com/sgl-project/sglang/pull/36567) | `e14d1c3cb6`, `405e494199`, `2179ad9bcb`, `d4477bd298` | Rust io_uring reader, NVMe PLE, SM121 sparse-decode kernel, and documentation |
| [PR #36556 — Support SM120/SM121 sparse decode](https://github.com/sgl-project/sglang/pull/36556) | Validated repository snapshot `93fd20af6f` | Compatible QSA sparse-decode path for SM120 without an incompatible fallback |

These are the pinned snapshots used for model validation, not a claim about the current status of the upstream pull requests. This repository does not automatically follow later upstream head changes.

## Additional changes in this repository

### 1. SM120 QSA sparse decode

- Updated `python/sglang/srt/layers/attention/qwen_sparse_attn_backend.py`.
- Extended the sparse GQA decode path from SM121 to SM120.
- Added backend-selection and fallback-order coverage in `test/registered/kernels/test_qsa.py`.

### 2. Breakable CUDA Graph dataclass outputs

- Updated `python/sglang/srt/model_executor/runner_backend/breakable_cuda_graph_backend.py`.
- Added recursive allocation, copy, and slicing for dataclass outputs.
- Allowed a speculative target-verify graph key for one request to safely hold four rows of verify logits and hidden states.
- Preserved dynamically attached top-k fields on `LogitsProcessorOutput`.
- Added CPU unit coverage for multi-row output and leading-dimension mismatch rejection.

### 3. Hot-PLE row gathering and profiling

- Updated `python/sglang/srt/models/qwen4_ple_nvme.py`.
- [Commit `7df4aea02b`](https://github.com/yhfgyyf/sglang-qwen38-flash-next-sm120/commit/7df4aea02ba2f3413f64071e3f2c6b129dc48ba9) adds an all-hit LRU fast path that avoids building page plans and per-row location objects, plus duplicate-row reuse for large, repetitive batches. Cache misses retain the original `io_uring` path.
- Set `SGLANG_QWEN4_PLE_NVME_NVTX=1` to enable `disk_read_pages`, `read_rows`, `future_wait`, and `stage_and_copy` NVTX ranges.
- `future_wait` measures exposed waiting on the background task, which can include CPU work as well as I/O. Overlapping CPU/GPU/I/O intervals must not be summed or labelled entirely as NVMe latency.

### 4. GDN projection and Conv1D fusion

- [Commit `b2f0b02763`](https://github.com/yhfgyyf/sglang-qwen38-flash-next-sm120/commit/b2f0b02763b98d150cc47b081def8ca892d288a4) changes `triton_gdn_fused_proj.py`, `linear/gdn_backend.py`, `qwen3_5.py`, and gated RMSNorm handling.
- Ordinary prefill uses projection views and stride-aware gated normalization to avoid unnecessary copies. Eligible decode calls fuse projection unpacking with the indexed causal Conv1D state update; unsupported tensor contracts keep a fallback.
- `SGLANG_ENABLE_GDN_DECODE_FUSED_PROJ_CONV=1` is the default. This does not quantize `lm_head` to FP8 or change the checkpoint.

### 5. QSA query tiles and text RoPE bounds

- [Commit `2b56a39e70`](https://github.com/yhfgyyf/sglang-qwen38-flash-next-sm120/commit/2b56a39e7045d6ec7c2e426b8d4b9df57030b61b) tunes the TileLang QSA indexer query tile to 8 rows for SM120 with 4 index heads and dimension 128.
- Plain-text `EXTEND` can use known CPU sequence lengths for the RoPE cache bound, avoiding the device-side maximum readback. Multimodal and speculative paths retain their original checks.
- This is QSA **indexer** tuning, not the unpublished native sparse-GQA final-attention prototype. The combined service measurements below do not isolate a speedup for each individual change; the RoPE-bound change alone previously showed no clear service gain.

## Validated environment

| Component | Validated configuration |
| --- | --- |
| GPU | NVIDIA RTX PRO 6000 Blackwell Workstation, SM120, 96GB |
| OS | Ubuntu 22.04, Linux 6.8 |
| NVIDIA driver | 595.45.04 |
| CUDA toolkit | 13.2, nvcc 13.2.51 |
| Python | 3.12 |
| PyTorch | 2.13.0+cu130 |
| SGLang | 0.5.19.dev series from this repository |
| FlashInfer | 0.6.17 |
| FlashAttention-4 | 4.0.0b19 |
| Transformers | 5.12.1 |

The system driver and toolkit use CUDA 13.2, while the tested PyTorch wheel is built for cu130. The driver-compatible combination was validated with the real model.

## Installation

### Release wheel

The release wheel targets **Linux x86_64 with glibc 2.34 or newer and CPython 3.12**. It includes the Rust storage extension required by the NVMe PLE backend.

```bash
conda create -n qwen38-sm120 python=3.12 -y
conda activate qwen38-sm120
python -m pip install --upgrade pip
python -m pip install "<downloaded-wheel>"
```

The wheel installs the package as `sglang` and replaces any existing SGLang installation in the active environment.

**The existing `v0.1.0` wheel predates the GDN/PLE/QSA optimization commits above.** Use the source installation below to reproduce the latest performance table. This documentation update does not rebuild or replace the release wheel.

### Source installation

```bash
git clone https://github.com/yhfgyyf/sglang-qwen38-flash-next-sm120.git
cd sglang-qwen38-flash-next-sm120

conda create -n qwen38-sm120 python=3.12 -y
conda activate qwen38-sm120

export CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"
export CUDA_PATH="$CUDA_HOME"
export CUDACXX="$CUDA_HOME/bin/nvcc"
export FLASHINFER_CUDA_ARCH_LIST=12.0f

python -m pip install -e ./python
```

If multiple CUDA toolkits are installed, place the CUDA 13.2 `nvcc` first on `PATH` before installation.

## Launch: default 256K context

Place the checkpoint on a local NVMe filesystem that supports `O_DIRECT` and `io_uring`, then run the complete command below. The command uses one GPU, a 262,144-token context, and a 20GiB PLE LRU payload limit.

```bash
export MODEL_PATH="<Qwen3.8-Flash-Next-NVFP4 model directory>"
export CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"
export CUDA_PATH="$CUDA_HOME"
export CUDACXX="$CUDA_HOME/bin/nvcc"
export CUDA_VISIBLE_DEVICES=0
export FLASHINFER_CUDA_ARCH_LIST=12.0f
export PYTORCH_ALLOC_CONF=expandable_segments:True

export SGLANG_QWEN4_PLE_NVME_PATH="$MODEL_PATH"
export SGLANG_QWEN4_PLE_NVME_BACKEND=io_uring
export SGLANG_QWEN4_PLE_NVME_QUEUE_DEPTH=512
export SGLANG_QWEN4_PLE_NVME_MAX_BATCH_PAGES=4096
export SGLANG_QWEN4_PLE_NVME_CACHE_PAGES=5242880
export SGLANG_QWEN4_PLE_NVME_LOG_INTERVAL=0
export SGLANG_QWEN4_PLE_NVME_NVTX=1
export SGLANG_ENABLE_GDN_DECODE_FUSED_PROJ_CONV=1

python -m sglang.launch_server \
  --model-path "$MODEL_PATH" \
  --quantization modelopt_fp4 \
  --fp4-gemm-backend flashinfer_cutlass \
  --tp-size 1 \
  --dtype bfloat16 \
  --page-size 64 \
  --mamba-radix-cache-strategy extra_buffer \
  --mamba-track-interval 64 \
  --linear-attn-backend triton \
  --chunked-prefill-size 8192 \
  --max-prefill-tokens 16384 \
  --max-running-requests 1 \
  --context-length 262144 \
  --max-total-tokens 262144 \
  --mem-fraction-static 0.98 \
  --speculative-algorithm NEXTN \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4 \
  --cuda-graph-backend-decode breakable \
  --cuda-graph-backend-prefill tc_piecewise \
  --cuda-graph-max-bs-decode 1 \
  --cuda-graph-bs-decode 1 \
  --cuda-graph-max-bs-prefill 8192 \
  --cuda-graph-bs-prefill 8192 \
  --host 127.0.0.1 \
  --port 30000
```

The total input and output token count must not exceed `--max-total-tokens`. The current PLE implementation requires `--tp-size 1`.

The 8K chunk and the 16K scheduling budget are different settings: the latest measured runs used `chunked_prefill_size=8192` and `max_prefill_tokens=16384`. The command retains Breakable decode/speculative graphs. With this EAGLE + FP4/MoE combination, the existing correctness safeguard automatically disables **target prefill** `tc_piecewise` replay; actual ordinary prefill was eager in all reported A/B arms. Do not bypass that safeguard or interpret the configured prefill graph size as proof that model prefill replay was active.

### Request example

```bash
curl http://127.0.0.1:30000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "default",
    "messages": [{"role": "user", "content": "Briefly explain PLE."}],
    "max_tokens": 128
  }'
```

## Performance results

### Published-source baseline: 2026-09-23

Measured at [`2b56a39e70`](https://github.com/yhfgyyf/sglang-qwen38-flash-next-sm120/commit/2b56a39e7045d6ec7c2e426b8d4b9df57030b61b), including the GDN/PLE/QSA changes above and the original Triton sparse-GQA attention. One GPU, concurrency 1, 262,144-token context, 8,192-token chunks, 512 output tokens, NEXTN/EAGLE 3/1/4, and NVMe `io_uring` PLE with a 20GiB host LRU; no profiler attached.

| Input tokens | Runs | Output tokens | Median TTFT (ms) | Median prefill (tok/s) | Median decode (tok/s) | Aggregate decode (tok/s) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8,192 | 5 | 512 | 575.478 | 14,235.125 | 222.632 | 223.173 |
| 32,768 | 5 | 512 | 2,349.649 | 13,945.911 | 227.779 | 227.814 |
| 131,072 | 5 | 512 | 10,101.447 | 12,975.567 | 224.134 | 222.922 |

Each length receives a full 512-output conditioning request. Before every measured request, the same prompt with one output token warms PLE, then KV/Radix is flushed. All 15 samples have `cached_tokens=0`, exactly 512 output tokens, and no Unicode replacement characters. Requests explicitly use `temperature=0` and `ignore_eos=true`.

- Prefill TPS = input tokens / client TTFT; this includes scheduling, PLE and first-token delivery, not just GPU compute.
- Per-request decode TPS = 511 / (E2E − TTFT). Aggregate decode = 5 × 511 / the sum of the five decode-phase durations; it is not the median and not multi-concurrency throughput.
- These are five fixed prompts per length; the 128K inputs repeat their corresponding 32K token sequences four times. Repetitive inputs benefit from PLE row reuse, so the results are workload-dependent and are not an isolated A/B against the older random-token benchmark.

### Native sparse-GQA research: not published or adopted

A local FlashInfer-derived CuTe prototype combines N16 K/V prefetch with shorter Q register lifetimes, per-lane address/mask reuse, and shared-memory swizzling. It is **not a stock FlashInfer API switch** and is not included in `main` or the release wheel. The launch command above continues to use the published Triton path.

Across first-8K, 32K-context, 128K-context and unequal-multi-request cases, full GPU-path latency fell by **12.96% / 10.49% / 11.73% / 13.77%**, including KV packing where required. Each operator call has 8,192 query tokens; these percentages are not whole-model speedups. Matched NCU/SASS checks caught spill in the first prefetch version; the final combination removed the observed spill and reduced dynamic shared memory from 27KiB to 24KiB per CTA.

| Input tokens | Experimental median prefill (tok/s) | Experimental aggregate decode (tok/s) |
| ---: | ---: | ---: |
| 8,192 | 14,482.873 | 205.336 |
| 32,768 | 14,174.335 | 227.048 |
| 131,072 | 13,208.366 | 222.483 |

Service prefill improved **1.6–2.4%** versus the initial and reverse Triton controls. However, two of five 8K greedy outputs diverged, and aggregate 8K decode fell **7.5–8.0%**; those samples are retained rather than hidden by the median. The outputs were readable, but not identical. The initial and reverse controls matched on all 15 paired outputs; the prototype matched 13/15.

Independent synthetic-tensor checks found nonempty outputs within `atol=rtol=0.01`, but **not bitwise equivalence**: large-shape differences were 1 BF16 ULP, small-query differences were more frequent, and all-masked rows produced NaN in the old kernel versus zero in the prototype. Twenty short-answer checks passed in both service arms, but that is not full-model accuracy validation or proof of the cause of the two divergences. The prototype remains unadopted; no FP8 `lm_head` is enabled.

### Historical 260K measurements: older code and workload

These retained results predate the optimization baseline above and use random valid token IDs. **260K was not rerun for the latest optimization**, so do not mix this mean-based table with the current median-based 8K/32K/128K table or extrapolate a new 260K speed.

| Input tokens | PLE state | Runs | Output tokens | Mean TTFT (s) | Prefill (tok/s) | Decode (tok/s) |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 260,000 | Cold PLE/NVMe | 1 | 512 | 60.643 | 4,287 | 201.3 |
| 260,000 | Warm host LRU | 4 | 512 | 35.166 | 7,393 | 161.5 |

The measured cold-versus-warm TTFT difference was 25.476s. This is an exposed end-to-end difference after overlap, not a sum of I/O durations. Decode also depends on speculative acceptance; the cold run's higher decode figure does not mean storage sped up decoding.

### Correctness and earlier multimodal checks

- The current published-source baseline passed 20/20 short-answer smoke checks. The performance workload and Unicode checks do not replace broad model-quality evaluation.

The following long-context and image checks are historical deployment checks, not reruns of the native sparse-GQA prototype:

- A separate readable-prompt 260K-by-512 test returned valid Chinese text without Unicode replacement characters.
- A 261,120-input and 32-output boundary request returned HTTP 200, and the server remained healthy afterward.
- A text-and-image request correctly recognized the test image and returned valid Chinese text with `image_tokens=651` and no corrupted characters.

## Tests

```bash
python -m pytest -q \
  test/registered/kernels/test_qsa.py \
  test/registered/kernels/ops/attention/test_gdn_fused_proj_conv.py \
  test/registered/unit/layers/attention/test_qsa_rope_host_bound.py \
  test/registered/unit/models/test_qwen4_ple_nvme.py \
  test/registered/unit/storage/test_io_uring_reader.py \
  test/registered/unit/model_executor/runner_backend/test_breakable_cuda_graph_backend.py
```

## Upstream and license

This project is derived from [SGLang](https://github.com/sgl-project/sglang) and retains its Apache-2.0 license. Model weights are not included in the repository or release; users must obtain the checkpoint separately and comply with its license.

---

# 中文版本

这是一个面向单张 **NVIDIA RTX PRO 6000 Blackwell Workstation（SM120，96GB）** 运行 `Qwen3.8-Flash-Next-NVFP4` 的 SGLang 实验性分支。项目基于固定的上游 Qwen3.8、NVMe PLE 和 SM120 QSA sparse decode 快照，补充了 Breakable CUDA Graph 修复、GDN 投影/Conv1D 融合、PLE 热缓存行读取和 QSA prefill 调整。

更新于 **2026-09-24**。已发布源码的最新实测在 8K / 32K / 128K 输入下，prefill 分别约 **1.42 万 / 1.39 万 / 1.30 万 tok/s**，汇总 decode 约 **223 / 228 / 223 tok/s**；条件为并发 1、输出 512、PLE 热缓存。完整工作负载和统计口径见[性能结果](#性能结果)。后续 native sparse-GQA 原型**尚未发布到本仓库，也未启用为默认实现**，实验结果单独列出。

> 当前 NVMe PLE 实现只支持单卡 `TP=1`，尚不支持多卡 PLE 运行。

## 项目解决什么问题

Qwen3.8-Flash-Next 包含约 **51.2B** 个额外 PLE（Per-Layer Embedding）参数。已测试 checkpoint 的 PLE 表使用 FP8，大小约为 **47.68GiB**。

本项目验证的运行参数如下：

- 主模型使用 NVFP4，在单张 SM120 GPU 上运行。
- PLE 权重保留在原始 safetensors 文件中，通过 `io_uring + O_DIRECT` 从本地 NVMe 按需读取。
- 使用 4KiB 页粒度的应用层 LRU，把当前 PLE 工作集保留在主机内存。
- PLE 读取会提前提交，并尽可能与前序 GPU 计算重叠。
- decode 和推测解码路径保留 CUDA Graph。
- NEXTN/EAGLE 推测解码使用 3 steps、`topk=1`，最多生成 4 个 draft tokens。

## 使用的 SGLang PR

| 上游变更 | 固定快照 | 本项目中的作用 |
| --- | --- | --- |
| [PR #36497 — Introduce Qwen 3.8 Flash Next](https://github.com/sgl-project/sglang/pull/36497) | `73a255206f` | Qwen3.8 模型、GDN/QSA、GR、PLE、MTP、量化和多模态支持 |
| [PR #36567 — Stream PLE embeddings from NVMe](https://github.com/sgl-project/sglang/pull/36567) | `e14d1c3cb6`、`405e494199`、`2179ad9bcb`、`d4477bd298` | Rust io_uring reader、NVMe PLE、SM121 sparse decode kernel 和文档 |
| [PR #36556 — Support SM120/SM121 sparse decode](https://github.com/sgl-project/sglang/pull/36556) | 本项目验证快照 `93fd20af6f` | 为 SM120 提供兼容的 QSA sparse decode 路径，避免不兼容 fallback |

表中是完成模型验证时使用的固定快照，不代表这些上游 PR 当前的开放或合并状态。本仓库不会自动跟随上游 head 的后续变化。

## 本项目额外修改

### 1. SM120 QSA sparse decode

- 修改 `python/sglang/srt/layers/attention/qwen_sparse_attn_backend.py`。
- 把 sparse GQA decode 路径从 SM121 扩展到 SM120。
- 在 `test/registered/kernels/test_qsa.py` 中补充 backend 选择和 fallback 顺序测试。

### 2. Breakable CUDA Graph dataclass 输出

- 修改 `python/sglang/srt/model_executor/runner_backend/breakable_cuda_graph_backend.py`。
- 支持递归分配、复制和切片 dataclass 输出。
- 允许一个请求对应的 speculative target-verify graph key 安全容纳四行 verify logits 和 hidden states。
- 保留 `LogitsProcessorOutput` 动态附加的 top-k 字段。
- 新增 CPU 单元测试，覆盖多行输出和 leading dimension 不一致的拒绝路径。

### 3. PLE 热缓存行读取与性能标记

- 修改 `python/sglang/srt/models/qwen4_ple_nvme.py`。
- [提交 `7df4aea02b`](https://github.com/yhfgyyf/sglang-qwen38-flash-next-sm120/commit/7df4aea02ba2f3413f64071e3f2c6b129dc48ba9) 增加 LRU 全命中快路径，省去 page plan 和逐行位置对象构造；大批量重复输入还会复用重复行的读取结果。缺页仍回退原有 `io_uring` 路径。
- 设置 `SGLANG_QWEN4_PLE_NVME_NVTX=1` 可以启用 `disk_read_pages`、`read_rows`、`future_wait` 和 `stage_and_copy` NVTX range。
- `future_wait` 表示对后台任务暴露出来的等待，可能同时包含 CPU 工作与 I/O；不能把重叠的 CPU/GPU/I/O 区间直接相加，或全部认定为 NVMe 延时。

### 4. GDN 投影与 Conv1D 融合

- [提交 `b2f0b02763`](https://github.com/yhfgyyf/sglang-qwen38-flash-next-sm120/commit/b2f0b02763b98d150cc47b081def8ca892d288a4) 修改 `triton_gdn_fused_proj.py`、`linear/gdn_backend.py`、`qwen3_5.py` 和 gated RMSNorm 处理。
- 普通 prefill 使用投影视图和支持 stride 的 gated normalization，减少不必要的复制；满足条件的 decode 将投影解包与带索引的 causal Conv1D 状态更新融合，不支持的张量条件保留回退。
- `SGLANG_ENABLE_GDN_DECODE_FUSED_PROJ_CONV=1` 为默认值。该优化不把 `lm_head` 改成 FP8，也不修改 checkpoint。

### 5. QSA query tile 与纯文本 RoPE 上界

- [提交 `2b56a39e70`](https://github.com/yhfgyyf/sglang-qwen38-flash-next-sm120/commit/2b56a39e7045d6ec7c2e426b8d4b9df57030b61b) 在 SM120、4 个 index head、维度 128 的条件下，将 TileLang QSA indexer 的 query tile 调整为 8 行。
- 纯文本 `EXTEND` 可以使用已知的 CPU 序列长度确定 RoPE cache 上界，省去设备端最大位置值的回读；多模态和 speculative 路径保留原检查。
- 这是 QSA **indexer** 调整，不是尚未发布的 native sparse-GQA 最终 attention 原型。下方整模型数据不用于拆分各项优化的独立收益；此前单独测试 RoPE 上界修改时，没有测出明确的服务提速。

## 已验证环境

| 组件 | 验证配置 |
| --- | --- |
| GPU | NVIDIA RTX PRO 6000 Blackwell Workstation，SM120，96GB |
| OS | Ubuntu 22.04，Linux 6.8 |
| NVIDIA Driver | 595.45.04 |
| CUDA Toolkit | 13.2，nvcc 13.2.51 |
| Python | 3.12 |
| PyTorch | 2.13.0+cu130 |
| SGLang | 0.5.19.dev 系列，本仓库源码版本 |
| FlashInfer | 0.6.17 |
| FlashAttention-4 | 4.0.0b19 |
| Transformers | 5.12.1 |

系统驱动和 Toolkit 使用 CUDA 13.2，测试使用的 PyTorch wheel 为 cu130。该驱动兼容组合已经通过真实模型验证。

## 安装

### 使用 Release wheel

Release wheel 面向 **Linux x86_64、glibc 2.34 或更高版本以及 CPython 3.12**，并包含 NVMe PLE backend 所需的 Rust storage extension。

```bash
conda create -n qwen38-sm120 python=3.12 -y
conda activate qwen38-sm120
python -m pip install --upgrade pip
python -m pip install "<downloaded-wheel>"
```

该 wheel 的包名为 `sglang`，会替换当前环境中已有的 SGLang 安装。

**现有 `v0.1.0` wheel 早于上述 GDN/PLE/QSA 优化提交。** 要复现最新性能表，请使用下面的源码安装方式。本次文档更新没有重打或替换 Release wheel。

### 从源码安装

```bash
git clone https://github.com/yhfgyyf/sglang-qwen38-flash-next-sm120.git
cd sglang-qwen38-flash-next-sm120

conda create -n qwen38-sm120 python=3.12 -y
conda activate qwen38-sm120

export CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"
export CUDA_PATH="$CUDA_HOME"
export CUDACXX="$CUDA_HOME/bin/nvcc"
export FLASHINFER_CUDA_ARCH_LIST=12.0f

python -m pip install -e ./python
```

如果系统中安装了多套 CUDA，请在安装前先把 CUDA 13.2 的 `nvcc` 放到 `PATH` 最前面。

## 启动：默认 256K 上下文

将 checkpoint 放到支持 `O_DIRECT` 和 `io_uring` 的本地 NVMe 文件系统，然后直接执行以下完整命令。该配置使用单张 GPU、262,144 token 上下文，以及 payload 上限为 20GiB 的 PLE LRU。

```bash
export MODEL_PATH="<Qwen3.8-Flash-Next-NVFP4 model directory>"
export CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"
export CUDA_PATH="$CUDA_HOME"
export CUDACXX="$CUDA_HOME/bin/nvcc"
export CUDA_VISIBLE_DEVICES=0
export FLASHINFER_CUDA_ARCH_LIST=12.0f
export PYTORCH_ALLOC_CONF=expandable_segments:True

export SGLANG_QWEN4_PLE_NVME_PATH="$MODEL_PATH"
export SGLANG_QWEN4_PLE_NVME_BACKEND=io_uring
export SGLANG_QWEN4_PLE_NVME_QUEUE_DEPTH=512
export SGLANG_QWEN4_PLE_NVME_MAX_BATCH_PAGES=4096
export SGLANG_QWEN4_PLE_NVME_CACHE_PAGES=5242880
export SGLANG_QWEN4_PLE_NVME_LOG_INTERVAL=0
export SGLANG_QWEN4_PLE_NVME_NVTX=1
export SGLANG_ENABLE_GDN_DECODE_FUSED_PROJ_CONV=1

python -m sglang.launch_server \
  --model-path "$MODEL_PATH" \
  --quantization modelopt_fp4 \
  --fp4-gemm-backend flashinfer_cutlass \
  --tp-size 1 \
  --dtype bfloat16 \
  --page-size 64 \
  --mamba-radix-cache-strategy extra_buffer \
  --mamba-track-interval 64 \
  --linear-attn-backend triton \
  --chunked-prefill-size 8192 \
  --max-prefill-tokens 16384 \
  --max-running-requests 1 \
  --context-length 262144 \
  --max-total-tokens 262144 \
  --mem-fraction-static 0.98 \
  --speculative-algorithm NEXTN \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4 \
  --cuda-graph-backend-decode breakable \
  --cuda-graph-backend-prefill tc_piecewise \
  --cuda-graph-max-bs-decode 1 \
  --cuda-graph-bs-decode 1 \
  --cuda-graph-max-bs-prefill 8192 \
  --cuda-graph-bs-prefill 8192 \
  --host 127.0.0.1 \
  --port 30000
```

输入和输出 token 总数不能超过 `--max-total-tokens`。当前 PLE 实现要求使用 `--tp-size 1`。

8K chunk 与 16K 调度预算是不同参数：最新实测使用 `chunked_prefill_size=8192`、`max_prefill_tokens=16384`。命令保留 Breakable decode/speculative Graph。该 EAGLE + FP4/MoE 组合会触发现有正确性保护，自动禁用 **target prefill** 的 `tc_piecewise` 回放，因此所有已报告 A/B 组的普通 prefill 实际为 eager。不要绕过该保护，也不要仅根据配置中的 prefill graph size 宣称整模型 prefill Graph 已生效。

### 请求示例

```bash
curl http://127.0.0.1:30000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "default",
    "messages": [{"role": "user", "content": "请简要介绍 PLE。"}],
    "max_tokens": 128
  }'
```

## 性能结果

### 已发布源码基线：2026-09-23

测试源码为 [`2b56a39e70`](https://github.com/yhfgyyf/sglang-qwen38-flash-next-sm120/commit/2b56a39e7045d6ec7c2e426b8d4b9df57030b61b)，包含上述 GDN/PLE/QSA 优化，最终 sparse-GQA attention 仍为原 Triton 实现。单卡、并发 1、262,144 上下文、8,192 chunk、512 输出、NEXTN/EAGLE 3/1/4、NVMe `io_uring` PLE 和 20GiB 主机 LRU；测吞吐时不挂 profiler。

| 输入 token | 次数 | 输出 token | TTFT 中位数（ms） | Prefill 中位数（tok/s） | Decode 中位数（tok/s） | 汇总 Decode（tok/s） |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8,192 | 5 | 512 | 575.478 | 14,235.125 | 222.632 | 223.173 |
| 32,768 | 5 | 512 | 2,349.649 | 13,945.911 | 227.779 | 227.814 |
| 131,072 | 5 | 512 | 10,101.447 | 12,975.567 | 224.134 | 222.922 |

每种长度先执行一次完整的 512 输出预热。每次正式测量前，再用同一 prompt + 1 输出预热 PLE，然后清空 KV/Radix。15 个样本均为 `cached_tokens=0`、恰好 512 输出、无 Unicode replacement character；请求显式设置 `temperature=0`、`ignore_eos=true`。

- Prefill TPS = 输入 token / 客户端 TTFT，包含调度、PLE 与首 token 交付，不是单纯 GPU 计算速度。
- 单次 decode TPS = 511 / (E2E − TTFT)；五次汇总 decode = 5 × 511 / 五次 decode 阶段时长之和。汇总值不是中位数，也不是多并发总吞吐。
- 每种长度使用 5 个固定 prompt；128K 由对应的 32K token 序列重复 4 遍构成。重复输入能从 PLE 行复用中获益，结果依赖工作负载，不能与旧随机 token 压测直接作单变量 A/B。

### Native sparse-GQA 研究：尚未发布或采用

本地 FlashInfer 衍生的 CuTe 原型结合了 N16 K/V 预取与 Q 寄存器存活期缩短、逐 lane 地址/掩码复用、共享内存 swizzle。它**不是原版 FlashInfer API 的开关切换**，尚未包含在 `main` 或 Release wheel 中。上面的启动命令仍使用已发布的 Triton 路径。

首块 8K、32K 上下文、128K 上下文和不等长多请求的完整 GPU 路径延时，分别降低 **12.96% / 10.49% / 11.73% / 13.77%**，需要 KV 打包的路径已计入打包成本。每次算子调用均为 8,192 query，这些百分比不是整模型提速。匹配形状的 NCU/SASS 检查发现首版预取出现 spill；最终组合消除了已观测 spill，并将每 CTA 动态共享内存从 27KiB 降至 24KiB。

| 输入 token | 实验版 Prefill 中位数（tok/s） | 实验版汇总 Decode（tok/s） |
| ---: | ---: | ---: |
| 8,192 | 14,482.873 | 205.336 |
| 32,768 | 14,174.335 | 227.048 |
| 131,072 | 13,208.366 | 222.483 |

相对前后两次 Triton 对照，服务 prefill 提升 **1.6–2.4%**。但 8K 的 5 条贪心输出中有 2 条分岔，汇总 decode 下降 **7.5–8.0%**；这些样本保留在统计中，没有被中位数掩盖。文本可读、没有乱码，但输出并不相同。前后 Triton 对照的 15/15 配对输出一致，原型为 13/15 一致。

独立合成张量检查中，所测非空选择输出满足 `atol=rtol=0.01`，但**不逐位等价**：大形状差异为 1 BF16 ULP，短 query 的差异更频繁，全空选择旧算子输出 NaN、原型输出 0。两组服务的 20 道短题均通过，但不能据此证明完整模型准确率不变，也不能确定两条生成分岔的原因。因此原型尚未采用，未启用 FP8 `lm_head`。

### 历史 260K 数据：旧代码与旧工作负载

保留的数据早于上述优化基线，输入为随机合法 token ID。**最新优化没有重测 260K**，不能把下方均值表与当前 8K/32K/128K 中位数表混作同一轮测试，也不能据此推算新版 260K 速度。

| 输入 token | PLE 状态 | 次数 | 输出 token | 平均 TTFT（s） | Prefill（tok/s） | Decode（tok/s） |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 260,000 | PLE/NVMe 冷缓存 | 1 | 512 | 60.643 | 4,287 | 201.3 |
| 260,000 | 主机 LRU 热缓存 | 4 | 512 | 35.166 | 7,393 | 161.5 |

冷、热请求 TTFT 的实测差为 25.476s，这是并行重叠后仍暴露的端到端差额，不是 I/O 区间累加。Decode 还受推测接受率影响，冷轮次 decode 更高不表示硬盘加快了解码。

### 正确性与此前多模态验证

- 当前已发布源码基线的短题烟测为 20/20；吞吐工作负载与 Unicode 检查不能代替全面的模型质量评测。

以下长上下文与图像结果属于历史部署验证，不是对新 native sparse-GQA 原型的复测：

- 另一次使用可读提示词的 260K×512 正确性测试返回了正常中文，没有 Unicode replacement character。
- 261,120 输入、32 输出的边界请求返回 HTTP 200，之后服务健康检查正常。
- 图文请求成功识别测试图片，返回正常中文；`image_tokens=651`，没有乱码。

## 测试

```bash
python -m pytest -q \
  test/registered/kernels/test_qsa.py \
  test/registered/kernels/ops/attention/test_gdn_fused_proj_conv.py \
  test/registered/unit/layers/attention/test_qsa_rope_host_bound.py \
  test/registered/unit/models/test_qwen4_ple_nvme.py \
  test/registered/unit/storage/test_io_uring_reader.py \
  test/registered/unit/model_executor/runner_backend/test_breakable_cuda_graph_backend.py
```

## 上游与许可证

本项目源自 [SGLang](https://github.com/sgl-project/sglang)，保留其 Apache-2.0 许可证。模型权重不包含在仓库或 Release 中，用户需要单独获取 checkpoint，并遵守对应许可证。

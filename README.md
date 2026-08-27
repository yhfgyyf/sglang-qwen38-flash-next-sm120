# sglang-qwen38-flash-next-sm120

[中文版本](#中文版本)

This is an experimental SGLang fork for running `Qwen3.8-Flash-Next-NVFP4` on a single **NVIDIA RTX PRO 6000 Blackwell Workstation GPU (SM120, 96GB)**. It combines the Qwen3.8 implementation, NVMe-backed PLE streaming, and the SM120 QSA sparse-decode fix from currently open SGLang pull requests. The fork also fixes speculative dataclass outputs in Breakable CUDA Graphs and adds optional NVTX ranges for PLE I/O profiling.

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

These pull requests were still open when this repository was published. This repository pins the commits used for the validated model runs instead of following later upstream head changes automatically.

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

### 3. PLE NVMe profiling ranges

- Updated `python/sglang/srt/models/qwen4_ple_nvme.py`.
- Set `SGLANG_QWEN4_PLE_NVME_NVTX=1` to enable `disk_read_pages`, `read_rows`, `future_wait`, and `stage_and_copy` NVTX ranges.
- `future_wait` is the closest measure of storage wait exposed on the critical path after GPU overlap.

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
python -m pip install <downloaded-wheel>
```

The wheel installs the package as `sglang` and replaces any existing SGLang installation in the active environment.

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
export SGLANG_QWEN4_PLE_NVME_LOG_INTERVAL=10

python -m sglang.launch_server \
  --model-path "$MODEL_PATH" \
  --quantization modelopt_fp4 \
  --fp4-gemm-backend flashinfer_cutlass \
  --tp-size 1 \
  --dtype bfloat16 \
  --page-size 64 \
  --mamba-radix-cache-strategy extra_buffer \
  --mamba-track-interval 64 \
  --chunked-prefill-size 512 \
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
  --cuda-graph-max-bs-prefill 512 \
  --cuda-graph-bs-prefill 512 \
  --host 127.0.0.1 \
  --port 30000
```

The total input and output token count must not exceed `--max-total-tokens`. The current PLE implementation requires `--tp-size 1`.

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

The measurements use one GPU, concurrency 1, a chunked-prefill size of 512, 512 output tokens, and NEXTN/EAGLE speculative decoding.

### 8K and 32K input, five runs each

| Input | Output | Runs | Mean TTFT | Prefill TPS | Decode TPS | Mean accept length | Unicode replacement characters |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8K | 512 | 5 | 2.224s | 3,683 | 167.4 | 2.924 | 0 |
| 32K | 512 | 5 | 10.076s | 3,252 | 141.6 | 2.462 | 0 |

### 260,000 input and 512 output, five runs

| PLE state | Sample | TTFT / prefill | Prefill TPS | Decode TPS | End to end | PLE O_DIRECT reads |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Cold start with NVMe misses | Run 1 | 91.726s | 2,834.5 | 136.49 | 95.470s | 13.942GiB |
| Warm host LRU | Mean of runs 2–5 | 69.004s | 3,768.1 | 141.51 | 72.656s | 0.030GiB |

For the same request and model computation, the cold run exposed an additional **22.722s** of TTFT, or approximately **32.93%**, relative to the warm LRU runs. This end-to-end difference is the visible latency caused by PLE pages coming from storage instead of the host cache. Background reads must not be added directly to model execution time because PLE I/O and GPU computation overlap.

No stable, significant NVMe penalty was observed during decode. Decode TPS also varies with the speculative accept length and generated branch, so a small cold-versus-warm mean difference cannot be attributed to storage alone.

### Correctness and multimodal input

- The five 260K-by-512 requests all returned HTTP 200 without OOM or Unicode replacement characters.
- A 261,120-input and 32-output boundary request returned HTTP 200, and the server remained healthy afterward.
- A text-and-image request correctly recognized the test image and returned valid Chinese text with `image_tokens=651` and no corrupted characters.

## Tests

```bash
python -m pytest -q \
  test/registered/kernels/test_qsa.py \
  test/registered/unit/models/test_qwen4_ple_nvme.py \
  test/registered/unit/storage/test_io_uring_reader.py \
  test/registered/unit/model_executor/runner_backend/test_breakable_cuda_graph_backend.py
```

## Upstream and license

This project is derived from [SGLang](https://github.com/sgl-project/sglang) and retains its Apache-2.0 license. Model weights are not included in the repository or release; users must obtain the checkpoint separately and comply with its license.

---

# 中文版本

这是一个面向单张 **NVIDIA RTX PRO 6000 Blackwell Workstation（SM120，96GB）** 运行 `Qwen3.8-Flash-Next-NVFP4` 的 SGLang 实验性分支。项目组合了仍处于开放状态的 SGLang PR 中的 Qwen3.8 实现、NVMe PLE 流式读取和 SM120 QSA sparse decode 修复，并补充了 Breakable CUDA Graph speculative dataclass 输出修复以及可选的 PLE I/O NVTX 性能标记。

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

这些 PR 在本仓库发布时仍未合并。本仓库固定使用已经完成真实模型验证的 commit，不会自动跟随上游 head 的后续变化。

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

### 3. PLE NVMe 性能标记

- 修改 `python/sglang/srt/models/qwen4_ple_nvme.py`。
- 设置 `SGLANG_QWEN4_PLE_NVME_NVTX=1` 可以启用 `disk_read_pages`、`read_rows`、`future_wait` 和 `stage_and_copy` NVTX range。
- `future_wait` 最接近 GPU 并行重叠之后真正暴露在关键路径上的存储等待。

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
python -m pip install <downloaded-wheel>
```

该 wheel 的包名为 `sglang`，会替换当前环境中已有的 SGLang 安装。

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
export SGLANG_QWEN4_PLE_NVME_LOG_INTERVAL=10

python -m sglang.launch_server \
  --model-path "$MODEL_PATH" \
  --quantization modelopt_fp4 \
  --fp4-gemm-backend flashinfer_cutlass \
  --tp-size 1 \
  --dtype bfloat16 \
  --page-size 64 \
  --mamba-radix-cache-strategy extra_buffer \
  --mamba-track-interval 64 \
  --chunked-prefill-size 512 \
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
  --cuda-graph-max-bs-prefill 512 \
  --cuda-graph-bs-prefill 512 \
  --host 127.0.0.1 \
  --port 30000
```

输入和输出 token 总数不能超过 `--max-total-tokens`。当前 PLE 实现要求使用 `--tp-size 1`。

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

测试使用单卡、并发 1、chunked prefill 512、输出 512，并开启 NEXTN/EAGLE 推测解码。

### 8K 与 32K 输入，每种 5 次

| 输入 | 输出 | 次数 | 平均 TTFT | Prefill TPS | Decode TPS | 平均 accept length | Unicode 替换字符 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8K | 512 | 5 | 2.224s | 3,683 | 167.4 | 2.924 | 0 |
| 32K | 512 | 5 | 10.076s | 3,252 | 141.6 | 2.462 | 0 |

### 260,000 输入、512 输出，共 5 次

| PLE 状态 | 样本 | TTFT / Prefill | Prefill TPS | Decode TPS | E2E | PLE O_DIRECT 实读 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 冷启动、NVMe 缺页 | 第 1 次 | 91.726s | 2,834.5 | 136.49 | 95.470s | 13.942GiB |
| 主机 LRU 热缓存 | 第 2–5 次平均 | 69.004s | 3,768.1 | 141.51 | 72.656s | 0.030GiB |

在相同请求和模型计算下，冷盘相对热 LRU 多暴露 **22.722s** TTFT，即约 **32.93%**。这个端到端差值是 PLE 页面来自硬盘而不是主机缓存时产生的可见额外延迟。PLE I/O 与 GPU 计算存在重叠，因此不能把后台读取时间直接与模型执行时间相加。

Decode 阶段没有观察到稳定、显著的 NVMe 惩罚。Decode TPS 还会受到推测解码 accept length 和生成分支影响，因此不能把小幅冷/热均值差直接归因于存储。

### 正确性与多模态输入

- 260K×512 的五次请求全部返回 HTTP 200，没有 OOM，也没有 Unicode replacement character。
- 261,120 输入、32 输出的边界请求返回 HTTP 200，之后服务健康检查正常。
- 图文请求成功识别测试图片，返回正常中文；`image_tokens=651`，没有乱码。

## 测试

```bash
python -m pytest -q \
  test/registered/kernels/test_qsa.py \
  test/registered/unit/models/test_qwen4_ple_nvme.py \
  test/registered/unit/storage/test_io_uring_reader.py \
  test/registered/unit/model_executor/runner_backend/test_breakable_cuda_graph_backend.py
```

## 上游与许可证

本项目源自 [SGLang](https://github.com/sgl-project/sglang)，保留其 Apache-2.0 许可证。模型权重不包含在仓库或 Release 中，用户需要单独获取 checkpoint，并遵守对应许可证。

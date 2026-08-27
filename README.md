# sglang-qwen38-flash-next-sm120

[English summary](#english-summary)

这是一个面向 **NVIDIA RTX PRO 6000 Blackwell Workstation（SM120，96GB）** 的 SGLang 实验性分支，用于在单卡上运行 `Qwen3.8-Flash-Next-NVFP4`。项目把尚未合并的 Qwen3.8 支持、NVMe PLE 流式读取和 SM120 QSA 修复组合在一起，并补充了 Breakable CUDA Graph speculative verify 输出修复、PLE NVTX 标记以及可复用启动脚本。

> 本项目不是 SGLang 官方发行版。代码基于 SGLang 的开放 PR 快照，主要服务于 SM120 单卡验证和复现；其他 GPU、模型或多卡配置没有在本项目中验证。

## 项目解决什么问题

Qwen3.8-Flash-Next 的 PLE（Per-Layer Embedding）包含约 **51.2B** 个额外参数。当前 checkpoint 的 PLE 表为 FP8，约 **47.68GiB**，无法和主模型权重一起常驻 96GB GPU；在 62GiB 主机内存上完整展开也会挤压 SGLang、文件缓存和 staging 空间。

本项目采用以下路径：

- 主模型使用 NVFP4，在单张 SM120 GPU 上运行。
- PLE 权重保留在原始 safetensors 文件中，按需通过 `io_uring + O_DIRECT` 从本地 NVMe 读取。
- 以 4KiB 页为单位维护应用层 LRU，只把当前工作集留在主机内存。
- PLE 读取提前提交，并与 PLE 前一层的 GPU 计算重叠。
- 保留 CUDA Graph；decode、target verify、draft decode 和 draft extend 使用 graph。
- 使用 NEXTN/EAGLE MTP 推测解码，配置为 3 steps、topk=1、最多 4 个 draft tokens。

这里有两个容易混淆的概念：

- **PLE N-gram**：使用 bigram/trigram 哈希访问模型固定的 embedding 权重，是模型结构的一部分。
- **NGRAM speculative decoding**：从历史文本匹配候选 token，是另一种可选推测解码算法。本模型的 PLE speculative 路径不使用它；本项目启用的是 **NEXTN/EAGLE**。

## 代码来源与使用的 SGLang PR

| 上游变更 | 固定快照 | 本项目中的作用 |
| --- | --- | --- |
| [PR #36497 — Introduce Qwen 3.8 Flash Next](https://github.com/sgl-project/sglang/pull/36497) | `73a255206f` | Qwen3.8 模型、GDN/QSA、GR、PLE、MTP、量化和多模态基础支持 |
| [PR #36567 — Stream PLE embeddings from NVMe](https://github.com/sgl-project/sglang/pull/36567) | `e14d1c3cb6`、`405e494199`、`2179ad9bcb`、`d4477bd298` | Rust io_uring reader、NVMe PLE、SM121 sparse decode kernel 和文档 |
| [PR #36556 — Support SM120/SM121 sparse decode](https://github.com/sgl-project/sglang/pull/36556) | 本项目验证快照 `93fd20af6f` | 让 SM120 使用兼容的 QSA sparse decode 路径，避免不兼容 fallback |

这些 PR 在本项目发布时仍未合并，上游 head 可能继续变化。本仓库以已完成真实模型测试的 commit 为准。

## 本项目额外修改

### 1. SM120 QSA sparse decode

- 修改 `python/sglang/srt/layers/attention/qwen_sparse_attn_backend.py`。
- 把原本面向 SM121 的 sparse GQA decode 路径扩展到 SM120。
- 为 backend 选择和 fallback 顺序补充 `test/registered/kernels/test_qsa.py` 覆盖。

### 2. Breakable CUDA Graph dataclass 输出

- 修改 `python/sglang/srt/model_executor/runner_backend/breakable_cuda_graph_backend.py`。
- 支持递归分配、复制和切片 dataclass 输出。
- speculative target verify 的 graph key 按请求数计算，但 logits/hidden states 会按 verify token 数输出；修复后 batch size 1 可以安全容纳 4 行 verify 输出。
- 保留 `LogitsProcessorOutput` 动态附加的 top-k 字段。
- 新增 CPU 单元测试，覆盖多行输出和 leading dimension 不一致的拒绝路径。

### 3. PLE NVMe 性能标记

- 修改 `python/sglang/srt/models/qwen4_ple_nvme.py`。
- 可通过 `SGLANG_QWEN4_PLE_NVME_NVTX=1` 打开四类 NVTX range：`disk_read_pages`、`read_rows`、`future_wait`、`stage_and_copy`。
- `future_wait` 最接近没有被 GPU 并行计算隐藏、真正暴露在关键路径上的存储等待。

### 4. 可复用启动脚本

- 新增 `scripts/run_qwen38_flash_next_sm120_nvme.sh`。
- 不包含机器相关路径；模型位置、CUDA Toolkit、Python、上下文和 LRU 容量都通过环境变量传入。
- 默认使用较保守的 64K 上下文和 4GiB PLE LRU；260K 配置需要显式覆盖。

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

“CUDA 13.2 环境”指系统驱动和 Toolkit 使用 CUDA 13.2；PyTorch wheel 自身为 cu130。驱动向后兼容路径已经通过真实模型验证。

## 安装

### 使用 Release wheel

Release wheel 为 **Linux x86_64 + CPython 3.12** 构建，并包含 PLE 所需的 Rust storage extension。建议使用独立环境：

```bash
conda create -n qwen38-sm120 python=3.12 -y
conda activate qwen38-sm120
python -m pip install --upgrade pip
python -m pip install <downloaded-wheel>
```

该 wheel 的包名仍为 `sglang`，会替换环境中已有的同名版本，因此不要安装到需要保留其他 SGLang 版本的环境。

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

如果系统中有多套 CUDA，请先把 CUDA 13.2 的 `nvcc` 放到 `PATH` 最前面，再执行以上命令。

## 启动

### 默认 64K 配置

先把 checkpoint 放在本地 NVMe 文件系统，并设置模型目录：

```bash
export MODEL_PATH="<Qwen3.8-Flash-Next-NVFP4 model directory>"
export CUDA_TOOLKIT="$(dirname "$(dirname "$(command -v nvcc)")")"

./scripts/run_qwen38_flash_next_sm120_nvme.sh
```

默认值：

- context length：65,536
- max running requests：1
- chunked prefill：512
- PLE LRU：1,048,576 个 4KiB 页，payload 上限约 4GiB
- PLE backend：io_uring，queue depth 512
- speculative decoding：NEXTN/EAGLE，3 steps，topk=1，4 draft tokens
- CUDA Graph：没有传入任何全局关闭参数

### 260K 输入、512 输出配置

以下配置是本项目完成五轮测试的长上下文设置：

```bash
export MODEL_PATH="<Qwen3.8-Flash-Next-NVFP4 model directory>"
export CUDA_TOOLKIT="$(dirname "$(dirname "$(command -v nvcc)")")"
export CONTEXT_LENGTH=262144
export MAX_TOTAL_TOKENS=262144
export MEM_FRACTION_STATIC=0.98
export SGLANG_QWEN4_PLE_NVME_CACHE_PAGES=5242880

./scripts/run_qwen38_flash_next_sm120_nvme.sh
```

5,242,880 个页对应 20GiB PLE payload；Python `OrderedDict`、key 和 bytes 对象还有额外开销。请在启动前确认主机可用内存、Swap、GPU 剩余显存和 NVMe 剩余空间。输入与输出 token 总数不能超过 `MAX_TOTAL_TOKENS`。

### 请求示例

服务默认监听 `127.0.0.1:30000`，兼容 OpenAI API：

```bash
curl http://127.0.0.1:30000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "default",
    "messages": [{"role": "user", "content": "请简要介绍 PLE。"}],
    "max_tokens": 128
  }'
```

## CUDA Graph 说明

本项目不会通过启动参数关闭 CUDA Graph。已验证日志中：

- decode graph：启用；
- target verify graph：启用；
- draft decode / draft extend graph：启用；
- target prefill graph：SGLang 的 FP4/MoE+EAGLE 正确性保护会自动关闭该阶段 graph。

因此，“保留 CUDA Graph”不等于所有阶段都强行 capture。绕过 target prefill 的 correctness guard 可能造成 decode replay 输出损坏，本项目没有这样做。PLE 主机 I/O 使用 breakable graph 的 eager 区域执行，不会把磁盘读取冻结进 graph。

## PLE LRU 与存储行为

io_uring reader 使用 `(safetensors file, aligned offset)` 作为 LRU key：

1. 根据 bigram/trigram hash 得到需要的 PLE 行。
2. 把行映射到 safetensors 的字节范围和 4KiB 页。
3. LRU 命中时直接复用内存中的页；缺页时用 O_DIRECT 批量读取。
4. 新页移动到 MRU 端；超过容量时从 LRU 端淘汰。
5. 拼接所需 FP8 行，异步拷贝到 GPU 并转换为 BF16。

SGLang 的 `/flush_cache` 会清理 KV/Radix 请求缓存，但不会清理 reader 内部的 PLE LRU。要得到真正的 PLE 冷缓存，最可靠的方法是重启 server。

## 性能结果

测试配置：单卡、并发 1、chunked prefill 512、输出 512、NEXTN/EAGLE 已开启。TPS 均为单请求结果。

### 8K 与 32K，每种 5 次

| 输入 | 输出 | 次数 | 平均 TTFT | Prefill TPS | Decode TPS | 平均 accept length | Unicode 替换字符 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8K | 512 | 5 | 2.224s | 3,683 | 167.4 | 2.924 | 0 |
| 32K | 512 | 5 | 10.076s | 3,252 | 141.6 | 2.462 | 0 |

### 260,000 输入、512 输出，共 5 次

| PLE 状态 | 样本 | TTFT / Prefill | Prefill TPS | Decode TPS | E2E | PLE O_DIRECT 实读 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 冷启动、NVMe 缺页 | 第 1 次 | 91.726s | 2,834.5 | 136.49 | 95.470s | 13.942GiB |
| 主机 LRU 热缓存 | 第 2–5 次平均 | 69.004s | 3,768.1 | 141.51 | 72.656s | 0.030GiB |

相同模型计算和请求下，冷盘相对热 LRU 多暴露 **22.722s** TTFT，即约 **32.93%**。这个端到端差值才是“PLE 在硬盘而不是内存”造成的可见额外延迟；不能把后台 `read_rows`、O_DIRECT 活动和 GPU 计算时间直接相加，因为它们存在嵌套与并行。

Decode 阶段没有观察到稳定、显著的 NVMe 惩罚。推测解码 accept length 和生成分支会让 Decode TPS 波动，因此不能把小幅冷/热均值差直接归因于硬盘。

### 正确性与多模态

- 260K×512：5/5 HTTP 200，无 OOM，输出中没有 Unicode replacement character。
- 261,120 输入、32 输出边界测试：HTTP 200，服务结束后健康检查正常。
- 图文输入：成功识别测试图片，返回正常中文；`image_tokens=651`，无乱码。

## 限制

- 只验证了单张 RTX PRO 6000 SM120；NVMe PLE 当前使用 TP1。
- Release wheel 只面向 Linux x86_64 和 CPython 3.12。
- 建议模型位于支持 O_DIRECT 的本地 NVMe 文件系统；受限容器需要允许 io_uring 系统调用。
- 260K 配置的 GPU 最低空闲显存约 1.3GiB，余量很小，不建议提高并发。
- PLE 完整表不会进入 wheel；用户需要单独准备模型 checkpoint。

## 测试

```bash
python -m pytest -q \
  test/registered/kernels/test_qsa.py \
  test/registered/unit/models/test_qwen4_ple_nvme.py \
  test/registered/unit/storage/test_io_uring_reader.py \
  test/registered/unit/model_executor/runner_backend/test_breakable_cuda_graph_backend.py
```

## 上游与许可证

本项目源自 [SGLang](https://github.com/sgl-project/sglang)，保留其 Apache-2.0 许可证。模型权重不包含在本仓库或 Release 中，使用时请遵守对应 checkpoint 的许可证。

## English summary

This experimental SGLang fork runs `Qwen3.8-Flash-Next-NVFP4` on one NVIDIA RTX PRO 6000 Blackwell Workstation GPU (SM120, 96GB). It combines the Qwen3.8 implementation from upstream PR #36497, NVMe-backed PLE streaming from PR #36567, and the SM120 sparse-decode fix from PR #36556. Additional local changes fix speculative dataclass outputs in Breakable CUDA Graphs, add optional NVTX ranges for PLE I/O, and provide a portable launch script.

The 47.68GiB FP8 PLE table stays in the original safetensors files and is fetched through io_uring/O_DIRECT into a host-memory LRU. CUDA Graph remains enabled for decode and speculative paths, while the target-prefill graph follows SGLang's correctness guard. The tested 260K-input workload reached 2,834.5 prefill tok/s with a cold NVMe cache and 3,768.1 tok/s with a warm host LRU; all five 512-token generations completed without invalid Unicode output.

See the Chinese sections above for installation, launch settings, performance methodology, and limitations.

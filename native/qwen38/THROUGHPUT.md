# Qwen3.8 native 吞吐量报告（2026-10-08）

> **状态：实验性，整体验收尚未完成。** 本文报告一次 RTX PRO 6000 Blackwell（SM120）上的小样本测量；它不是生产性能承诺，也不证明语义质量等价、C8/C10 容量，或完整原生调度器已经完成。

可复算的脱敏数据见 [`benchmarks/2026-10-08.json`](benchmarks/2026-10-08.json)，对应的 R56 启动方式见 [`README.md` 的 measured GPU-resident profile](README.md#measured-gpu-resident-autotuned-candidate-r56)。JSON 保留逐 wave 汇总、逐请求数值指标、输入/输出哈希、推测解码接受计数，以及模型、库和关键源码哈希；不包含 prompt、生成文本、原始请求 ID、完整日志、本地绝对路径或凭据。

## 结论

在已测的四个工作负载上，R56 的 E2E 输出吞吐量中位数相对 R52 分别为 **+2.82%、+6.65%、+22.26%、+35.91%**：

| 工作负载 | R52 output TPS | R56 output TPS | 变化 |
| --- | ---: | ---: | ---: |
| C1，32K 输入 | 95.458780 | 98.151043 | +2.82% |
| C1，64K 输入 | 62.911137 | 67.092567 | +6.65% |
| C1，128K 输入 | 32.960288 | 40.297674 | +22.26% |
| C4，32K/请求 | 113.191334 | 153.841927 | +35.91% |

这是**配置级对比**，不是 FP8、native executor、ReplaySSM 或某个 kernel 的单变量因果证明。R52 使用原始 SGLang 和实际 BF16 ordinary KV；R56 同时使用 FP8 ordinary KV、native 执行路径、ReplaySSM、canonical QSA、不同 PDL 策略和重新 autotune。所有测量输出哈希均不同，整段运行期间的硬件状态也没有被严格控制。

四组数据每组只有 5 个 measured wave。C1 32K 的五个配对 wave 中有一个 R56 比 R52 **慢 0.75%**；+2.82% 是两个 workload median 的比值，不能解读为每个 wave 都更快。当前性能目标因此仍未完整验收。

## 测量对象与固定条件

两组都使用相同的冻结输入、原始 BF16/NVFP4 checkpoint 权重、原始 BF16 QSA 辅助/索引状态、FP32 SSM state、TP=1，以及 NEXTN `steps/top-k/draft-tokens = 3/1/4`。服务资源均为 5 个 request slots、25 个 Mamba slots、147,456 token pool 和 8,192-token chunk。

| 项目 | R52：original SGLang | R56：native candidate |
| --- | --- | --- |
| Ordinary target/draft KV | 实际 BF16 | FP8 E4M3 |
| 执行路径 | 原始 SGLang | typed C++ executor + Rust PLE，native BF16/context-tail bundle |
| QSA / SSM | 原始顺序；无 ReplaySSM | canonical QSA；ReplaySSM 开启 |
| PDL | 原始策略，三个 PDL 环境变量未强制关闭 | 三个 PDL selector 全部关闭 |
| FlashInfer autotune | 原始策略 | fresh autotune 开启 |
| Fused finalize / file load / cache reuse | 未强制设置 | `0 / 0 / 0` |
| Host KV | 关闭 | 关闭 |

源码基线是 `125f783eb63c3e02314e94ee463f479dd24db1d2`，其 Git tree 与已发布的 `main` 提交 `4cacd4e586743600c35ae6748aa7b2b4fc5b151a` 相同。候选测量来自以该基线为基础的实验性 dirty worktree；**这里不把基线提交冒充为候选的已测试提交版本**。候选的实际关键源码、模型配置/索引和动态库 SHA256 已写入脱敏 JSON。测量后有文档、诊断工具及测试整理，本次发布还清理了冗余类型注解引号、未使用 import 和文件末尾空行，未更改吞吐路径的执行逻辑；JSON 刻意保留测量当时的哈希，三个 native `.so` 未改变，本次发布没有重新运行 GPU benchmark。

## 指标定义与方法

- **E2E output TPS**：先按 wave 计算全部已发出 output token 的整体速率，再取 5 个 measured wave 的中位数。
- **TTFT**：所有 measured request 的 client-observed time-to-first-token 中位数。
- **TPOT**：所有 measured request 的 client-observed time-per-output-token 中位数。
- **Derived prefill**：`input tokens / TTFT`，按**每个请求**解释；包含调度、PLE、GPU 工作和首 token 交付，不是纯 GPU prefill。
- **Derived decode**：`1 / TPOT`，按**每个请求**解释；包含推测解码，不能当作 MTP-off 的基础 decode kernel 速率。

每个 workload 使用 1 个排除在统计外的 warmup wave，加 5 个 measured wave；cold cache、每请求固定 512 output tokens、`ignore_eos=true`。所有机械测量都观察到目标实际并发度，没有请求错误、空输出、全零输出或 retraction。C4 的 TTFT、TPOT、derived prefill 和 derived decode 都是 request median/每请求值，不是四请求合计值。

### 延迟与派生速率

| 工作负载 | R52 TTFT (s) | R56 TTFT (s) | R52 TPOT (ms) | R56 TPOT (ms) |
| --- | ---: | ---: | ---: | ---: |
| C1，32K | 2.426366 | 2.246742 | 5.863903 | 5.806373 |
| C1，64K | 4.988445 | 4.595978 | 6.155177 | 5.697682 |
| C1，128K | 12.507332 | 9.491594 | 6.144767 | 5.673153 |
| C4，32K/请求 | 8.132068 | 6.217190 | 16.068323 | 13.603717 |

| 工作负载 | R52 input/TTFT (tok/s/请求) | R56 input/TTFT (tok/s/请求) | R52 1/TPOT (tok/s/请求) | R56 1/TPOT (tok/s/请求) |
| --- | ---: | ---: | ---: | ---: |
| C1，32K | 13,504.969 | 14,584.671 | 170.535 | 172.225 |
| C1，64K | 13,137.561 | 14,259.423 | 162.465 | 175.510 |
| C1，128K | 10,479.613 | 13,809.271 | 162.740 | 176.269 |
| C4，32K/请求 | 4,029.479 | 5,270.548 | 62.234 | 73.509 |

## 正确性、输出轨迹与接受率

R52 和 R56 各自的 42 个机械请求（包括 warmup）都完成了 fixed-512、cold-cache、actual-C、zero-error、zero-retraction 检查。但机械完成只说明传输和计数条件满足，**不等于语义质量 parity**。

- R52：7 个 finite probe、36 个 calibration request 通过；一次 normal-EOS round 为 20/20 measured semantic pass。
- R56：7 个 finite probe、36 个 calibration request 通过；两次 normal-EOS round 都是 19/20 measured semantic pass。repeat round 还存在偏离目标的 edge-case 检查结果。
- R52 与 R56 的 35 个 measured request 输出哈希全部不同。因此性能差异同时受到生成轨迹和推测接受行为影响。
- C1 32K 的 pooled 指标：R52 接受率 63.1599%、每次 verify 发出 2.89593 token、平均 176.8 次 verify；R56 为 59.5861%、2.78867 token、183.6 次 verify。

这些差异不支持“FP8 单独导致加速”或“executor 单独导致加速”的结论，也不能据此推断关闭 MTP 会更快；本轮没有 MTP-off 对照。

## R53 → R56 autotune 诊断

R53 与 R56 都是相同源码、模型、native binary 和 FP8 GPU-resident 配置；记录到的配置差异是 R53 关闭 autotune，而 R56 fresh autotune，且两者都保持 fused finalize、autotuner file loading、autotune cache reuse 为 `0`。

在相同的 20 个 target-verify phase 上，实际执行的 MoE GEMM kernel 发生了变化，GEMM interval union 从 **66.928 ms 降到 55.260 ms**。这是 profiler 诊断结果，不是 E2E 服务耗时的可加分解，也不能单独证明 R53→R56 的 E2E 改善完全由 autotune 引起：输出轨迹不同，运行历史和整段硬件状态也未严格匹配。

## Host-KV C6 补充实验

R50 与 R51 是 **native-to-native** 的 host FP8 KV 对比，不是 original SGLang 对比。两者都是 C6、7 request slots、35 Mamba slots、819,200 token pool、12 GiB host budget、4,096-token chunk、ReplaySSM 开启。R51 仅进一步启用 selected-gather dedup；该功能默认仍为关闭。

| 工作负载 | R50 TPS | R51 TPS | TPS 变化 | R50 / R51 TTFT (s) | R50 / R51 TPOT (ms) |
| --- | ---: | ---: | ---: | ---: | ---: |
| C6，32K/请求 | 109.579096 | 124.240864 | +13.38% | 8.805781 / 9.360941 | 36.378594 / 29.213089 |
| C6，64K/请求 | 69.133457 | 72.168383 | +4.39% | 18.368181 / 19.582031 | 50.289301 / 44.413757 |
| C6，128K/请求 | 37.267527 | 36.751736 | -1.38% | 40.500526 / 43.359799 | 81.718346 / 78.680666 |

结果是混合的：R51 的 TPOT 改善，但 TTFT 三组都更慢，而且 128K E2E TPS 略降。因此 dedup 没有被提升为默认开启。

R50 和 R51 都完成了 `261632 input + 512 output = 262144` logical-token 的 exact-boundary 请求，并随后完成 32K+32-token recovery。该结果只限定为一次有边界、有健康检查的容量资格验证，**不是 256K 生产就绪声明**。

## 与已发布 GitHub 表格的关系

[旧仓库的已发布性能表](https://github.com/yhfgyyf/sglang-qwen38-flash-next-sm120/blob/4cacd4e586743600c35ae6748aa7b2b4fc5b151a/README.md#performance-results) 报告约 223–228 aggregate decode tok/s；它与本报告当前约 170 tok/s 的 32K C1 数值不是同一 workload：

- 已发布表使用 warm PLE 和重复的技术文本；当前 R52/R56 使用 cold-cache 的冻结 public-code prompts，新的 code sampling 不再是旧的重复技术文本。
- 已发布 aggregate decode 的公式是 `5 × 511 / sum(五个 decode duration)`，不是 `1 / median(TPOT)`。
- 对旧流程的独立 32K reproduction audit 有 5 个 measured record：接受率 100%、每次 verify 发出 4 token、平均 128 次 verify，aggregate decode 为 **220.7977523 tok/s**。
- 上述接受率来自独立 reproduction audit；已发布 README 本身没有提供原始 acceptance metadata，不能把它写成原发布测得的 acceptance。
- 当前 32K C1 按同类 aggregate 公式计算：R52 为 **169.502120 tok/s**，R56 为 **171.308972 tok/s**。对应 pooled 接受/verify 数据见上一节。

因此，旧表和新表之间的 decode 差异主要说明 workload、cache condition 和生成轨迹不同，不能用来判断新 executor 退化，也不能推出“禁用推测解码会更快”。

## 未完成项与排除项

- C8/C10 的容量与性能尚未验证。
- 当前候选仍使用 SGLang 的加载、调度、HTTP、capture 和大量算子；不是完全原生控制面或 Python-free server。
- R52→R56 是多配置维度对比，输出全部不同，样本数仅为 5，且没有全程硬件状态控制。
- R56 的 normal-EOS 语义检查尚未达到 R52 的单轮结果，质量 parity 未建立。
- R57 在机械性能 wave 开始前收到来源不明的 SIGTERM；它没有任何可用性能结果，本文完全排除 R57。
- exact 256K 仅为 bounded qualification，不代表高并发、长期稳定性或生产 readiness。

在补齐同数值/同策略控制、更多重复、C8/C10、质量 parity 和稳定的独占硬件窗口之前，最准确的表述仍是：**R56 在本次四个已测 workload 的配置级比较中取得更高 E2E median output TPS，但总体产品与性能目标尚未验收完成。**

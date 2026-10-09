# Qwen3.8 SM120 native candidate

This directory contains an experimental, transitional execution path for Qwen3.8
Flash-Next NVFP4 on one RTX PRO 6000 Blackwell (SM120). SGLang still owns loading,
scheduling, HTTP, capture, and most operators; the candidate adds a typed C++ graph
executor, Rust direct-I/O PLE, and optional original-BF16 kernels and host ordinary KV.
This is **not** a fully native scheduler, a Python-free server, or a completed 256K
product path. Features are opt-in and overall performance acceptance is incomplete.

See the separate [throughput report](THROUGHPUT.md) for measured results,
metric definitions, configuration differences, and quality limitations.

## Fixed model contract

Do not change these when comparing arms:

- Checkpoint precision is preserved: original BF16 tensors remain BF16 and the checkpoint's NVFP4 tensors remain NVFP4.
- GDN/linear-attention recurrent SSM state remains FP32.
- QSA auxiliary/index state remains at its original BF16 precision.
- Candidate and matched-control ordinary target/draft K/V use FP8 E4M3,
  on GPU or in host KV. Historical original-SGLang BF16 controls are labelled
  separately; they do not change this candidate precision contract.
- Speculation remains NEXTN with steps/top-k/draft-tokens `3/1/4`.
- Tensor parallelism is one; the tested hardware target is SM120.

`native-bf16-*` means kernels over original BF16 weights with FP32
accumulation. It does not mean converting the checkpoint or KV cache to BF16.

## What each launch arm means

- `baseline` runs the frozen tree from `--baseline`; it lacks the newer native,
  ReplaySSM, canonical-order, and automatic-PDL-disable options.
- `python-matched` runs this worktree with the native executor disabled and is the valid current-source control.
- `native` runs the C++ executor and Rust PLE; positive `--host-kv-bytes` also opts into host ordinary KV.

The launcher prints the selected source, complete command, source hashes,
binary hashes, model hashes, launch options, and effective PDL/FlashInfer environment.
Save that output with every result.

The baseline defaults to FP8 too. Use `--baseline-kv-cache-dtype auto` or
`bfloat16` explicitly to reproduce the original working BF16-KV baseline;
this option is rejected for `native` and `python-matched`. Check the resolved
KV allocation in the server log. A BF16-baseline/FP8-candidate comparison is
a configuration-level result, not an equal-KV-precision kernel comparison.
The frozen baseline's unsupported FP8 prefill failure is not a speedup result.

## Prerequisites and build

The validated stack is Linux x86_64, Python 3.12, PyTorch 2.13.0+cu130, CUDA 13.2
(`nvcc` 13.2.51), C++17, CMake 3.24+, and Rust/Cargo. The checkpoint filesystem must
support `O_DIRECT`; PLE uses `io_uring`. Run commands from the repository root.
Use this feature branch, not the repository's default `main` branch:

```bash
git clone --branch feat/qwen38-native-pro6000 \
  https://github.com/yhfgyyf/sglang-qwen38-flash-next-sm120.git
cd sglang-qwen38-flash-next-sm120
```

Activate the Python environment installed using the repository's source-install
instructions first; `QWEN38_PYTHON` below selects that environment's interpreter.

From the repository root, build the Rust library first because CMake requires
its release artifact:

```bash
export CUDA_HOME=/usr/local/cuda-13.2
export CUDA_PATH=/usr/local/cuda-13.2
export CUDACXX=/usr/local/cuda-13.2/bin/nvcc
export PATH=/usr/local/cuda-13.2/bin:"$PATH"
export PYTHONPATH="$PWD/python${PYTHONPATH:+:$PYTHONPATH}"
export QWEN38_PYTHON="$(command -v python)"
cargo build --release --locked \
  --manifest-path native/qwen38/ple_store/Cargo.toml
cmake -S native/qwen38 -B native/qwen38/build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.2/bin/nvcc
cmake --build native/qwen38/build --parallel
```

The expected defaults are:

```text
native/qwen38/build/libqwen38_native.so
native/qwen38/build/libq38_host_kv.so
native/qwen38/ple_store/target/release/libq38_ple_store.so
```

Override a library only with an explicit, durable candidate path. For example:

```bash
export QWEN38_HOST_KV_LIBRARY=/absolute/path/to/libq38_host_kv.so
```

Do not copy temporary build paths into launch instructions. R50 loaded its backed-up
pre-prototype host library, not the newest `build/` file.

## Reproducible launch policy

R45/R46 and R48/R50 used all three automatic PDL selectors, FlashInfer autotuning,
fused MoE finalize, autotuner file loading, and autotune cache reuse all off:

```bash
export SGLANG_FLASHINFER_MOE_FUSED_FINALIZE=0
export FLASHINFER_AUTOTUNER_LOAD_FROM_FILE=0
export SGLANG_FLASHINFER_AUTOTUNE_CACHE=0
```

Pass both `--disable-auto-pdl` and `--disable-flashinfer-autotune`. The former
sets and records these exact values:

```text
SGLANG_JIT_DISABLE_PDL=1
SGLANG_FLASHINFER_CUTLASS_DISABLE_PDL=1
SGLANG_QSA_TRTLLM_DISABLE_PDL=1
```

ReplaySSM is default-off. Add `--enable-linear-replayssm-spec` only for an
explicit ReplaySSM arm; its absence is part of the control configuration.

All launcher profiles bind to `127.0.0.1`, select GPU 0, and use port 30001 by
default. Set `MODEL_PATH` and, if needed, `BASELINE_SOURCE` before launching.

### Recorded GPU-resident control

R45 used this C4-capable, 4K-chunk current-source control:

```bash
export MODEL_PATH=/absolute/path/to/Qwen3.8-Flash-Next-NVFP4
"$QWEN38_PYTHON" native/qwen38/launch_sm120.py \
  --arm python-matched --model "$MODEL_PATH" \
  --concurrency 4 --scheduler-slots 5 --mamba-slots 25 \
  --max-total-tokens 294912 --chunked-prefill-size 4096 \
  --canonical-qsa-order \
  --disable-auto-pdl --disable-flashinfer-autotune
```

R45 has native BF16 GDN, native BF16 linear, and native context-tail all off. C1
ran against this same five-request/25-Mamba-slot service; one/five is unverified
derived sizing. R46 switched to `native` and enabled all three native flags.

These are current-source GPU controls, not frozen `baseline`. Use `--arm baseline --baseline "$BASELINE_SOURCE"`
only after removing unsupported flags and recording the policy difference.

### Measured GPU-resident autotuned candidate (R56)

R56 retains the three PDL disables but enables fresh FlashInfer tuning. Fused
finalize, file loading, and tuning-cache reuse remain off. It uses ordinary FP8
KV on GPU, the native BF16/context-tail bundle, canonical QSA, and ReplaySSM:

```bash
export SGLANG_FLASHINFER_MOE_FUSED_FINALIZE=0
export FLASHINFER_AUTOTUNER_LOAD_FROM_FILE=0
export SGLANG_FLASHINFER_AUTOTUNE_CACHE=0
"$QWEN38_PYTHON" native/qwen38/launch_sm120.py \
  --arm native --model "$MODEL_PATH" \
  --concurrency 4 --scheduler-slots 5 --mamba-slots 25 \
  --max-total-tokens 147456 --chunked-prefill-size 8192 \
  --native-bf16-gdn --native-bf16-linear --native-context-tail \
  --canonical-qsa-order --disable-auto-pdl \
  --enable-linear-replayssm-spec
```

Do not add `--disable-flashinfer-autotune` to reproduce R56. Host KV and dedup
are off. This measured profile is still experimental: its small C1 32K margin
and incomplete semantic acceptance do not establish the overall performance
goal. Do not extrapolate this GPU token budget to C4 64K/128K or C8/C10.

### Verified host-KV C6 profile

R48 used this native host profile: seven scheduler slots, 35 Mamba slots,
819,200 logical KV tokens, a 12 GiB pinned-host budget, and 4,096-token chunks.
The native BF16 and canonical-order flags match the paired 4K native profile:

```bash
"$QWEN38_PYTHON" native/qwen38/launch_sm120.py \
  --arm native --model "$MODEL_PATH" \
  --concurrency 6 --scheduler-slots 7 --mamba-slots 35 \
  --max-total-tokens 819200 --host-kv-bytes 12884901888 \
  --chunked-prefill-size 4096 \
  --native-bf16-gdn --native-bf16-linear --native-context-tail \
  --canonical-qsa-order \
  --disable-auto-pdl --disable-flashinfer-autotune
```

R50 adds `--enable-linear-replayssm-spec`. R48 used the repository host library and
R50 an identical-SHA256 backup; compare bytes, not path names. The newer optional
`--host-kv-dedup` requires a rebuilt host library and a positive host budget. It
deduplicates only selected gathers, owns fixed graph-lifetime scratch, and is
default-off; it does not change prefill/move behavior or KV bytes.

## Validation

Run CPU/static checks before taking the GPU. These commands exercise the Rust
crate and launch/benchmark contracts without loading the model:

```bash
cargo test --release --locked \
  --manifest-path native/qwen38/ple_store/Cargo.toml
cargo clippy --all-targets --locked \
  --manifest-path native/qwen38/ple_store/Cargo.toml -- -D warnings
"$QWEN38_PYTHON" -m pytest -q \
  native/qwen38/tests/test_launcher.py \
  native/qwen38/tests/test_profile_request.py \
  native/qwen38/tests/test_bench_acceptance.py \
  native/qwen38/tests/test_bench_host_kv_gather.py
```

Native CUDA/storage integration requires the built libraries and the exclusive
SM120 GPU:

```bash
"$QWEN38_PYTHON" -m pytest -q native/qwen38/tests/test_native.py
"$QWEN38_PYTHON" -m pytest -q native/qwen38/tests/test_host_kv.py
"$QWEN38_PYTHON" -m pytest -q native/qwen38/tests/test_host_kv_pool.py
"$QWEN38_PYTHON" -m pytest -q native/qwen38/tests/test_host_kv_attention.py
"$QWEN38_PYTHON" -m pytest -q native/qwen38/tests/test_replayssm_qwen38.py
"$QWEN38_PYTHON" -m pytest -q native/qwen38/tests/test_replayssm_ple_commit.py
```

The full 128K host gather is deliberately manual because it needs about
256 MiB pinned host memory and about 512 MiB GPU memory:

```bash
QWEN38_HOST_KV_LARGE_TEST=1 "$QWEN38_PYTHON" -m pytest -q \
  native/qwen38/tests/test_host_kv.py -k full_128k_context_gather
```

The original-checkpoint PLE byte oracle is also explicit:

```bash
"$QWEN38_PYTHON" native/qwen38/tests/check_checkpoint_ple.py "$MODEL_PATH"
```

`bench_bf16.py`, `bench_paged_prefill.py`, `bench_host_kv_gather.py`, and
`bench_host_kv_dedup.py` are bounded kernel/component measurements. They are
not service-performance results. `profile_request.py` is a diagnostic trace
client and is explicitly not a throughput benchmark.

## Current evidence and limits

The most recent complete GPU-resident comparison is original SGLang R52
(ordinary BF16 KV, original PDL/autotune policy) versus native R56 (ordinary
FP8 KV, policy above). This is a **configuration-level** comparison. Both used
the same frozen prompt inputs, five measured waves plus one warmup, cold cache,
512 output tokens per request, and verified actual concurrency. All mechanical
waves had zero errors/retractions and nonempty, non-all-zero output.

| Workload | R52 median output TPS | R56 median output TPS | Change |
| --- | ---: | ---: | ---: |
| C1, 32K input | 95.459 | 98.151 | +2.82% |
| C1, 64K input | 62.911 | 67.093 | +6.65% |
| C1, 128K input | 32.960 | 40.298 | +22.26% |
| C4, 32K input | 113.191 | 153.842 | +35.91% |

These are ratios of workload medians, not a promise that every wave improved.
Generated texts and speculative acceptance differ; hardware state was not
controlled across whole runs. In particular, the C1 32K margin needs a repeated
original-baseline comparison before claiming a reliable win: one of its five
paired waves was 0.75% slower than R52, while the other four were faster.

R56 completed seven finite probes, 36 calibration requests, 42 mechanical
requests, and two 24-request normal-EOS rounds. Calibration passed the bounded
manual checks. Each EOS round had 19/20 measured semantic passes, versus 20/20
in R52's single round; a negative-frame-budget explanation was wrong. R56's
repeat also included off-target edge-case checks. All requests ended normally,
but semantic parity is not established. C8/C10 capacity and performance remain
unverified, and the serving control plane still depends on SGLang.

The diagnostic trace confirms that R56 tuning selected different MoE kernels;
the 20 target-verify phases' MoE GEMM interval union decreased from 66.928 ms
in untuned R53 to 55.260 ms. Separate finalize kernels remained. Merely enabling
fused-finalize in untuned R55 did **not** change the executed kernel signatures,
so R55's speed variation is not evidence of fusion. Profile timings are not
unprofiled service throughput.

Artifacts are retained under `/tmp/q38-native-pro6000.29v46y0X`, with prefixes
`baseline_gpu_bf16_c4_original_r52` and
`native_gpu_fp8_c4_autotune_unfused_r56`. Each workload directory contains its
request records and summary; the corresponding `_app.log` contains the launch
fingerprint. R57's autotuned/fused trial was interrupted by SIGTERM before any
measured EOS or mechanical wave; it is not a valid performance result.

Earlier host/control evidence remains scoped as follows:

- R45 versus R46 is a 4K configuration-level comparison. R46 also enables the
  native BF16/context-tail bundle, so this is not a one-variable executor result.
- R48 versus R50 is native-host versus native-host only. Complete C6 32K, 64K,
  and 128K median output TPS improved 12.29%, 12.33%, and 12.75%; this is not
  a comparison against untouched SGLang.
- R50 completed 18 finite probes, 36 calibration requests, 36 normal-EOS requests,
  108 mechanical requests, and two capacity/recovery requests. The exact boundary
  check passed 261,632 input + 512 output tokens, followed by a 32K + 32 recovery.
  This is a bounded profile result, not general 256K production readiness.
- R48 and R50 passed the bounded 36-request calibration (12 natural, 12 typed-tool,
  12 final-marker); this is not broad quality validation.
- Both normal-EOS reviews were partial: 24/30 measured strict semantic passes with
  the same defects. Mechanical completion and finite logprobs are not a quality pass.
- R51 adds selected host-KV dedup to R50. All 18 finite probes, 36 calibration,
  36 normal-EOS, 108 mechanical, and two boundary/recovery requests completed.
  EOS retained the same 24/30 measured semantic passes. C6 32K/64K/128K median
  TPS changed by +13.38%/+4.39%/-1.38%; TTFT worsened at all three lengths,
  while TPOT improved. It is not a universal win and remains default-off.
- Dedup raw-byte, graph-lifetime, and real-attention regressions passed. Its
  roughly 3x high-overlap gather microbenchmark speedup does not describe
  end-to-end performance. Partial-enqueue/capture-failure injection is untested.
- Host payload accounting excludes weights, activations, graphs, PLE, SSM/QSA
  state, allocator fragmentation, and capture peaks. A logical token count is
  not proof that a service profile fits safely.

## Shared-machine safety and acceptance

Serialize model serving, CUDA compilation/tests, large host allocations, and
profiling. CUDA-graph capture and first-use JIT/autotune can consume peak memory
well above the steady-state number. Confirm the prior service is idle, stop only
the process you own, and verify the GPU is released before the next arm.
R50 stopped idle and released the GPU at 12:07 UTC before dedup testing.

For concurrency claims, require server-observed actual C, not only client
in-flight requests. A mechanical acceptance wave must use cold/no-prefix-hit
requests, produce the exact requested output-token count, report zero
retractions, avoid empty/all-zero/nonfinite output, and reach the intended
actual scheduler concurrency. Retain per-wave inputs, hashes, configuration,
and server metrics; do not hide slow or failed waves.

No benchmark, byte test, trace, or passing calibration by itself establishes
end-to-end speedup, exact output equivalence, 128K/256K capacity, or production
readiness.

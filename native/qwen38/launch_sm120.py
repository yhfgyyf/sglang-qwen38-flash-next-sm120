"""Launch one isolated, loopback-only control or native-candidate service."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


PDL_DISABLE_ENV_NAMES = (
    "SGLANG_JIT_DISABLE_PDL",
    "SGLANG_FLASHINFER_CUTLASS_DISABLE_PDL",
    "SGLANG_QSA_TRTLLM_DISABLE_PDL",
)
FLASHINFER_POLICY_ENV_NAMES = (
    "SGLANG_FLASHINFER_MOE_FUSED_FINALIZE",
    "FLASHINFER_AUTOTUNER_LOAD_FROM_FILE",
    "SGLANG_FLASHINFER_AUTOTUNE_CACHE",
)


def fingerprint_targets(classes, layers):
    """Resolve known Qwen boundaries at registration, not on every module call."""
    if layers is None:
        return ["*"]
    classes = set(
        classes
        or (
            "QSAIndexer",
            "RadixAttention",
            "Qwen3_5GatedDeltaNet",
            "Qwen4ExpLinearDecoderLayer",
            "Qwen4ExpAttentionDecoderLayer",
        )
    )
    targets = set()
    for layer in layers:
        base = f"*.layers.{layer}"
        if classes & {"Qwen4ExpLinearDecoderLayer", "Qwen4ExpAttentionDecoderLayer"}:
            targets.add(base)
        if "QSAIndexer" in classes:
            targets.add(base + ".indexer")
        if "RadixAttention" in classes:
            targets.add(base + ".attn")
        if "Qwen3_5GatedDeltaNet" in classes:
            targets.add(base + ".linear_attn")
    return sorted(targets)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--arm", choices=("baseline", "python-matched", "native"), required=True
    )
    parser.add_argument(
        "--model", default="/home/yyf/data2/models/RadixArk/Qwen3___8-Flash-Next-NVFP4"
    )
    parser.add_argument(
        "--baseline", default="/home/yyf/sglang-qwen38-row512-pr-20261006"
    )
    parser.add_argument(
        "--baseline-kv-cache-dtype",
        choices=("auto", "bfloat16", "fp8_e4m3"),
        help="Explicit frozen-baseline KV control; candidate and matched arms remain FP8",
    )
    parser.add_argument("--port", type=int, default=30001)
    parser.add_argument("--concurrency", type=int, choices=(1, 4, 6, 8, 10), default=4)
    parser.add_argument(
        "--scheduler-slots",
        type=int,
        help="Allocated request slots; may exceed offered concurrency for chunked-prefill headroom",
    )
    parser.add_argument("--max-total-tokens", type=int, default=147456)
    parser.add_argument(
        "--chunked-prefill-size",
        type=int,
        choices=(4096, 8192),
        default=8192,
        help="Bound eager prefill temporaries without reducing KV capacity or context length",
    )
    parser.add_argument("--mamba-slots", type=int)
    parser.add_argument(
        "--mamba-strategy",
        choices=("extra_buffer", "extra_buffer_lazy"),
        default="extra_buffer",
    )
    parser.add_argument("--skip-mamba-decode-lock", action="store_true")
    parser.add_argument("--cache-bytes", type=int, default=536870912)
    parser.add_argument(
        "--disable-flashinfer-autotune",
        action="store_true",
        help="Use FlashInfer fallback tactics; independent of PDL and fused-finalize policy",
    )
    parser.add_argument(
        "--disable-auto-pdl",
        action="store_true",
        help="Disable automatic PDL only at the current profile's JIT, FlashInfer CUTLASS, and QSA TRT-LLM boundaries",
    )
    parser.add_argument(
        "--enable-linear-replayssm-spec",
        action="store_true",
        help="Opt in to ReplaySSM target-verify folding (candidate or matched control)",
    )
    parser.add_argument(
        "--native-bf16-gdn",
        action="store_true",
        help="Opt in to original-BF16 GDN small-M GEMM (candidate or matched control)",
    )
    parser.add_argument(
        "--native-bf16-linear",
        action="store_true",
        help="Opt in to measured original-BF16 QSA/down projection shapes",
    )
    parser.add_argument(
        "--native-paged-prefill",
        action="store_true",
        help="Opt in to direct paged FP8 chunk prefill (native only)",
    )
    parser.add_argument(
        "--native-context-tail",
        action="store_true",
        help="Admit the exact 256K greedy Python-HTTP boundary with a one-token MTP tail",
    )
    parser.add_argument(
        "--canonical-qsa-order",
        action="store_true",
        help="Use stable QSA cutoff ties and canonical block order (candidate or matched control)",
    )
    parser.add_argument(
        "--diagnostic-context-tail",
        action="store_true",
        help="Check actual near-boundary device positions; requires native context tail",
    )
    parser.add_argument(
        "--diagnostic-finite",
        action="store_true",
        help="Add eager-prefill finite/index assertions; not a performance run",
    )
    parser.add_argument(
        "--diagnostic-samples",
        help="Save bounded eager layer samples to a fresh diagnostic directory",
    )
    parser.add_argument(
        "--diagnostic-fingerprints",
        help="Save asynchronous GPU boundary fingerprints without forward-thread readback",
    )
    parser.add_argument(
        "--diagnostic-fingerprint-classes",
        nargs="+",
        choices=(
            "QSAIndexer",
            "RadixAttention",
            "Qwen3_5GatedDeltaNet",
            "Qwen4ExpLinearDecoderLayer",
            "Qwen4ExpAttentionDecoderLayer",
        ),
        help="Restrict fingerprint collection to these module classes",
    )
    parser.add_argument(
        "--diagnostic-fingerprint-layers",
        nargs="+",
        type=int,
        help="Restrict fingerprint collection to these zero-based layer IDs",
    )
    parser.add_argument(
        "--diagnostic-fingerprint-max-elements",
        type=int,
        help="Sample at most this many evenly strided elements per tensor",
    )
    parser.add_argument(
        "--diagnostic-fingerprint-before",
        action="store_true",
        help="Also sample selected QSA/attention inputs before their forward call",
    )
    parser.add_argument(
        "--diagnostic-fingerprint-defer-records",
        type=int,
        default=0,
        help="Delay background copies/writes until this many records are queued (up to128)",
    )
    parser.add_argument(
        "--diagnostic-fingerprint-control",
        choices=("record-stream", "noop", "allocator"),
        help="Diagnostic control: stream bookkeeping, empty callback, or one-time allocator metadata",
    )
    parser.add_argument(
        "--host-kv-bytes",
        type=int,
        default=0,
        help="Explicit bounded pinned-host ordinary FP8 KV budget (native only)",
    )
    parser.add_argument(
        "--host-kv-dedup",
        action="store_true",
        help="Deduplicate selected host-KV gather rows (native host-KV only)",
    )
    args = parser.parse_args()
    if args.baseline_kv_cache_dtype is not None and args.arm != "baseline":
        parser.error("baseline KV override requires --arm baseline")
    kv_cache_dtype = args.baseline_kv_cache_dtype or "fp8_e4m3"
    if args.disable_auto_pdl and args.arm == "baseline":
        parser.error("the frozen baseline does not implement --disable-auto-pdl")
    if args.enable_linear_replayssm_spec and args.arm == "baseline":
        parser.error(
            "the frozen baseline does not implement --enable-linear-replayssm-spec"
        )
    if args.host_kv_bytes < 0 or (args.host_kv_bytes and args.arm != "native"):
        parser.error("a positive host-KV budget requires the native arm")
    if args.host_kv_dedup and (args.arm != "native" or args.host_kv_bytes <= 0):
        parser.error(
            "host-KV dedup requires the native arm and a positive host-KV budget"
        )
    if args.native_bf16_gdn and args.arm == "baseline":
        parser.error("the frozen baseline does not implement native BF16 GDN GEMM")
    if args.native_bf16_linear and args.arm == "baseline":
        parser.error("the frozen baseline does not implement native BF16 linear GEMM")
    if args.native_paged_prefill and args.arm != "native":
        parser.error("native paged prefill requires the native arm")
    if args.native_context_tail and args.arm != "native":
        parser.error("native context tail requires the native arm")
    if args.canonical_qsa_order and args.arm == "baseline":
        parser.error("the frozen baseline does not implement canonical QSA order")
    if args.diagnostic_context_tail and not args.native_context_tail:
        parser.error("context-tail diagnostics require --native-context-tail")
    if args.diagnostic_fingerprints and (
        args.diagnostic_finite or args.diagnostic_samples
    ):
        parser.error(
            "asynchronous fingerprints cannot be combined with finite/sample hooks"
        )
    if args.diagnostic_fingerprint_classes and not args.diagnostic_fingerprints:
        parser.error("fingerprint classes require --diagnostic-fingerprints")
    if args.diagnostic_fingerprint_control and not args.diagnostic_fingerprints:
        parser.error("fingerprint control requires --diagnostic-fingerprints")
    if args.diagnostic_fingerprint_layers is not None and (
        not args.diagnostic_fingerprints or min(args.diagnostic_fingerprint_layers) < 0
    ):
        parser.error(
            "non-negative fingerprint layers require --diagnostic-fingerprints"
        )
    if args.diagnostic_fingerprint_max_elements is not None and (
        not args.diagnostic_fingerprints or args.diagnostic_fingerprint_max_elements < 1
    ):
        parser.error(
            "positive fingerprint max-elements requires --diagnostic-fingerprints"
        )
    if not 0 <= args.diagnostic_fingerprint_defer_records <= 128 or (
        args.diagnostic_fingerprint_defer_records and not args.diagnostic_fingerprints
    ):
        parser.error(
            "fingerprint defer-records requires a fingerprint run and a0..128 queue threshold"
        )
    before_classes = sorted(
        {"QSAIndexer", "RadixAttention"}
        & set(args.diagnostic_fingerprint_classes or ("QSAIndexer", "RadixAttention"))
    )
    if args.diagnostic_fingerprint_before and (
        not args.diagnostic_fingerprints or args.arm == "baseline" or not before_classes
    ):
        parser.error(
            "before fingerprints require supported boundaries and a non-baseline fingerprint run"
        )
    # NEXTN + extra_buffer needs five state slots per active request. Keep the
    # paired runs honest: do not let SGLang silently clamp actual concurrency.
    if args.scheduler_slots is None:
        args.scheduler_slots = args.concurrency
    if args.scheduler_slots < args.concurrency:
        parser.error("scheduler slots must cover the offered concurrency")
    state_ratio = 4 if args.mamba_strategy == "extra_buffer_lazy" else 5
    state_ratio -= int(args.skip_mamba_decode_lock)
    required_slots = state_ratio * args.scheduler_slots
    if args.mamba_slots is None:
        args.mamba_slots = required_slots
    if args.mamba_slots < required_slots:
        parser.error(f"this launch profile needs at least {required_slots} Mamba slots")
    root = Path(__file__).resolve().parents[2]
    source = Path(args.baseline).resolve() if args.arm == "baseline" else root
    env = dict(os.environ)
    env.update(
        {
            "PYTHONPATH": str(source / "python"),
            "CUDA_HOME": "/usr/local/cuda-13.2",
            "CUDA_PATH": "/usr/local/cuda-13.2",
            "CUDACXX": "/usr/local/cuda-13.2/bin/nvcc",
            "CUDA_VISIBLE_DEVICES": "0",
            "PATH": "/usr/local/cuda-13.2/bin:" + env.get("PATH", ""),
            "FLASHINFER_CUDA_ARCH_LIST": "12.0f",
            "PYTORCH_ALLOC_CONF": "expandable_segments:True",
            "SGLANG_QWEN4_PLE_NVME_PATH": args.model,
            "SGLANG_QWEN4_PLE_NVME_BACKEND": "io_uring",
            "SGLANG_QWEN4_PLE_NVME_CACHE_MODE": "row",
            "SGLANG_QWEN4_PLE_NVME_CACHE_BYTES": str(args.cache_bytes),
            "SGLANG_QWEN4_PLE_NVME_CACHE_PAGES": "0",
            "SGLANG_QWEN4_PLE_NVME_QUEUE_DEPTH": "512",
            "SGLANG_QWEN4_PLE_NVME_MAX_BATCH_PAGES": "4096",
            "SGLANG_QWEN4_PLE_NVME_LOG_INTERVAL": "0",
            "SGLANG_QWEN4_PLE_NVME_NVTX": "0",
            "SGLANG_ENABLE_GDN_DECODE_FUSED_PROJ_CONV": "1",
            "SGLANG_QWEN38_NATIVE_SPARSE_PREFILL": "0",
            "SGLANG_ENABLE_CUDA_GRAPH_DEDUP": "0",
            "QWEN38_NATIVE_EXECUTOR": "1" if args.arm == "native" else "0",
            "QWEN38_HOST_KV_BYTES": str(args.host_kv_bytes),
            "QWEN38_HOST_KV_DEDUP": "1" if args.host_kv_dedup else "0",
            "QWEN38_BF16_GDN": "1" if args.native_bf16_gdn else "0",
            "QWEN38_BF16_LINEAR": "1" if args.native_bf16_linear else "0",
            "QWEN38_PAGED_FP8_PREFILL": "1" if args.native_paged_prefill else "0",
            "QWEN38_CONTEXT_TAIL": "1" if args.native_context_tail else "0",
            "QWEN38_QSA_CANONICAL_ORDER": "1" if args.canonical_qsa_order else "0",
            "QWEN38_CONTEXT_TAIL_TRACE": "1" if args.diagnostic_context_tail else "0",
            "SGLANG_OPT_MAMBA_SKIP_DECODE_LOCK": (
                "1" if args.skip_mamba_decode_lock else "0"
            ),
        }
    )
    if args.disable_auto_pdl:
        env.update(dict.fromkeys(PDL_DISABLE_ENV_NAMES, "1"))
    cmd = [
        sys.executable,
        "-m",
        "sglang.launch_server",
        "--model-path",
        args.model,
        "--quantization",
        "modelopt_fp4",
        "--fp4-gemm-backend",
        "flashinfer_cutlass",
        "--tp-size",
        "1",
        "--dtype",
        "bfloat16",
        "--kv-cache-dtype",
        kv_cache_dtype,
        "--page-size",
        "64",
        "--mamba-radix-cache-strategy",
        args.mamba_strategy,
        "--mamba-track-interval",
        "64",
        "--linear-attn-backend",
        "triton",
        "--chunked-prefill-size",
        str(args.chunked_prefill_size),
        "--max-prefill-tokens",
        "16384",
        "--max-running-requests",
        str(args.scheduler_slots),
        "--context-length",
        "262144",
        "--max-total-tokens",
        str(args.max_total_tokens),
        "--mem-fraction-static",
        "0.98",
        "--max-mamba-cache-size",
        str(args.mamba_slots),
        "--mamba-ssm-dtype",
        "float32",
        "--speculative-algorithm",
        "NEXTN",
        "--speculative-num-steps",
        "3",
        "--speculative-eagle-topk",
        "1",
        "--speculative-num-draft-tokens",
        "4",
        "--cuda-graph-backend-decode",
        "breakable",
        "--cuda-graph-backend-prefill",
        "tc_piecewise",
        "--cuda-graph-max-bs-decode",
        str(args.scheduler_slots),
        "--cuda-graph-bs-decode",
        *[str(i) for i in range(1, args.scheduler_slots + 1)],
        "--cuda-graph-max-bs-prefill",
        str(args.chunked_prefill_size),
        "--cuda-graph-bs-prefill",
        str(args.chunked_prefill_size),
        "--reasoning-parser",
        "qwen3",
        "--tool-call-parser",
        "qwen3_coder",
        "--enable-metrics",
        "--random-seed",
        "314159",
        "--host",
        "127.0.0.1",
        "--port",
        str(args.port),
    ]
    if args.disable_flashinfer_autotune:
        cmd.append("--disable-flashinfer-autotune")
    if args.enable_linear_replayssm_spec:
        cmd.append("--enable-linear-replayssm-spec")
    if args.diagnostic_fingerprints:
        config = {
            "min_rows": 8192,
            "sample_dir": args.diagnostic_fingerprints,
            "module_classes": args.diagnostic_fingerprint_classes,
            "layer_ids": args.diagnostic_fingerprint_layers,
            "max_elements": args.diagnostic_fingerprint_max_elements,
            "defer_records": args.diagnostic_fingerprint_defer_records,
            "control": args.diagnostic_fingerprint_control,
        }
        hooks = [
            {
                "name": "qwen38-prefill-fingerprints",
                "target_modules": fingerprint_targets(
                    args.diagnostic_fingerprint_classes,
                    args.diagnostic_fingerprint_layers,
                ),
                "hook_factory": "native.qwen38.fingerprint_probe:make_fingerprint_hook",
                "config": config,
            }
        ]
        if args.diagnostic_fingerprint_before:
            hooks.append(
                {
                    "name": "qwen38-prefill-input-before-fingerprints",
                    "target_modules": fingerprint_targets(
                        before_classes, args.diagnostic_fingerprint_layers
                    ),
                    "hook_type": "forward_pre",
                    "hook_factory": "native.qwen38.fingerprint_probe:make_fingerprint_pre_hook",
                    "config": {**config, "module_classes": before_classes},
                }
            )
        cmd.extend(
            [
                "--forward-hooks",
                json.dumps(hooks),
            ]
        )
    elif args.diagnostic_finite or args.diagnostic_samples:
        cmd.extend(
            [
                "--forward-hooks",
                json.dumps(
                    [
                        {
                            "name": "qwen38-prefill-finite-index-probe",
                            "target_modules": ["*"],
                            "hook_factory": "native.qwen38.finite_probe:make_finite_hook",
                            "config": {
                                "min_rows": 8192,
                                "sample_dir": args.diagnostic_samples,
                            },
                        }
                    ]
                ),
            ]
        )
    print(f"arm={args.arm} source={source} interpreter={sys.executable}", flush=True)
    print("command=" + repr(cmd), flush=True)

    def git(*command):
        return subprocess.check_output(["git", "-C", str(source), *command])

    paths = set(git("diff", "--name-only", "HEAD").decode().splitlines())
    paths.update(
        git("ls-files", "--others", "--exclude-standard").decode().splitlines()
    )
    source_hashes = {
        path: hashlib.sha256((source / path).read_bytes()).hexdigest()
        for path in sorted(paths)
        if (source / path).is_file()
    }
    binary_hashes = {}
    if args.arm == "native":
        for variable, relative, required in (
            ("QWEN38_NATIVE_LIBRARY", "build/libqwen38_native.so", True),
            (
                "QWEN38_HOST_KV_LIBRARY",
                "build/libq38_host_kv.so",
                bool(args.host_kv_bytes),
            ),
            (
                "QWEN38_PLE_LIBRARY",
                "ple_store/target/release/libq38_ple_store.so",
                True,
            ),
        ):
            binary = Path(
                env.get(variable, root / "native/qwen38" / relative)
            ).resolve()
            if required and not binary.is_file():
                raise FileNotFoundError(f"required native runtime library: {binary}")
            if binary.is_file():
                binary_hashes[str(binary)] = hashlib.sha256(
                    binary.read_bytes()
                ).hexdigest()
    print(
        "runtime_fingerprint="
        + json.dumps(
            {
                "revision": git("rev-parse", "HEAD").decode().strip(),
                "source_sha256": source_hashes,
                "native_binary_sha256": binary_hashes,
                "model_config_sha256": hashlib.sha256(
                    (Path(args.model) / "config.json").read_bytes()
                ).hexdigest(),
                "model_index_sha256": hashlib.sha256(
                    (Path(args.model) / "model.safetensors.index.json").read_bytes()
                ).hexdigest(),
                "launch_options": vars(args),
                "resolved_kv_cache_dtype": kv_cache_dtype,
                "flashinfer_policy": {
                    "disable_autotune": args.disable_flashinfer_autotune,
                    "effective_env": {
                        name: env.get(name) for name in FLASHINFER_POLICY_ENV_NAMES
                    },
                },
                "pdl_policy": {
                    "disable_auto_pdl": args.disable_auto_pdl,
                    "effective_env": {
                        name: env.get(name) for name in PDL_DISABLE_ENV_NAMES
                    },
                },
            },
            sort_keys=True,
        ),
        flush=True,
    )
    os.execve(sys.executable, cmd, env)


if __name__ == "__main__":
    main()

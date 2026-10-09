#!/usr/bin/env python3
"""Microbenchmark the experimental SM120 original-BF16 small-M GEMM.

This is a kernel microbenchmark, not an end-to-end serving claim.  ``warm``
leaves cache state uncontrolled after warmup; ``flush`` writes a 256 MiB
buffer on the same stream immediately before each timed launch to evict L2
approximately.  Both eager and CUDA-graph measurements use CUDA events on
that stream and report the median of at least five repetitions.
"""

from __future__ import annotations

import argparse
import statistics
from dataclasses import replace

import torch
import torch.nn.functional as F

from sglang.srt.model_executor.qwen38_bf16 import (
    BF16SmallMGemmConfig,
    bf16_small_m_linear,
)


SHAPES = {
    "gdn_fused": (16480, 2560),
    "gdn_qkvz": (16384, 2560),
    "qsa_qkv": (13312, 2560),
    "out": (2560, 6144),
    "shared_gate_up": (1280, 2560),
    "down": (2560, 640),
    "indexer": (640, 2560),
    "router": (512, 2560),
    "hc_down": (320, 10240),
    "hc_up": (10240, 320),
}
MS = (1, 4, 5, 16, 20, 40)

BASE_CONFIG = BF16SmallMGemmConfig(
    block_m=16,
    block_n=64,
    block_k=64,
    num_warps=4,
    num_stages=3,
)
VARIANTS = {
    "direct_n32_k64_w4": replace(BASE_CONFIG, block_n=32),
    "direct_n64_k64_w4": BASE_CONFIG,
    "direct_n64_k128_w4": replace(BASE_CONFIG, block_k=128),
    "direct_n128_k64_w8": replace(BASE_CONFIG, block_n=128, num_warps=8),
    "split4_n32_k64_w4": replace(BASE_CONFIG, block_n=32, split_k=4),
    "split4_n64_k64_w4": replace(BASE_CONFIG, split_k=4),
}


def _parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _capture(call):
    call()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    capture_stream = torch.cuda.Stream()
    capture_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.graph(graph, stream=capture_stream):
        result = call()
    torch.cuda.current_stream().wait_stream(capture_stream)
    torch.cuda.synchronize()
    return graph.replay, result, graph


def _measure_us(call, *, reps: int, warmup: int, flush: torch.Tensor | None) -> float:
    for _ in range(warmup):
        call()
    torch.cuda.synchronize()
    samples = []
    for _ in range(reps):
        if flush is not None:
            flush.zero_()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        call()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000.0)
    return statistics.median(samples)


def _make_custom_call(x, weight, bias, config):
    out = torch.empty(
        (x.shape[0], weight.shape[0]), dtype=torch.bfloat16, device=x.device
    )
    workspace = None
    if config.split_k > 1:
        workspace = torch.empty(
            (config.split_k, x.shape[0], weight.shape[0]),
            dtype=torch.float32,
            device=x.device,
        )

    def call():
        return bf16_small_m_linear(
            x,
            weight,
            bias,
            out=out,
            workspace=workspace,
            config=config,
        )

    return call, out, workspace


def _check_result(label, result, x, weight, bias):
    reference = F.linear(
        x.float(), weight.float(), None if bias is None else bias.float()
    )
    torch_result = F.linear(x, weight, bias)
    custom_max = (result.float() - reference).abs().max().item()
    torch_max = (torch_result.float() - reference).abs().max().item()
    print(
        f"# numeric {label}: custom_max_fp32_error={custom_max:.8g} "
        f"torch_bf16_max_fp32_error={torch_max:.8g}"
    )
    if not torch.isfinite(result).all():
        raise RuntimeError(f"{label} produced non-finite output")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shapes", default="all")
    parser.add_argument("--ms", default=",".join(map(str, MS)))
    parser.add_argument(
        "--variants",
        default="torch,direct_n64_k64_w4,split4_n32_k64_w4",
    )
    parser.add_argument("--modes", default="eager,graph")
    parser.add_argument("--cache", choices=("warm", "flush"), default="flush")
    parser.add_argument("--reps", type=int, default=7)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--bias", action="store_true")
    args = parser.parse_args()
    if args.reps < 5:
        parser.error("--reps must be at least 5")
    if torch.cuda.get_device_capability() != (12, 0):
        raise RuntimeError("this prototype benchmark requires SM120")

    shape_names = list(SHAPES) if args.shapes == "all" else _parse_csv(args.shapes)
    unknown_shapes = set(shape_names) - SHAPES.keys()
    if unknown_shapes:
        parser.error(f"unknown shapes: {sorted(unknown_shapes)}")
    ms = [int(value) for value in _parse_csv(args.ms)]
    variants = _parse_csv(args.variants)
    unknown_variants = set(variants) - ({"torch"} | VARIANTS.keys())
    if unknown_variants:
        parser.error(f"unknown variants: {sorted(unknown_variants)}")
    modes = _parse_csv(args.modes)
    if set(modes) - {"eager", "graph"}:
        parser.error("--modes accepts eager and/or graph")

    device = torch.device("cuda")
    props = torch.cuda.get_device_properties(device)
    flush = None
    cache_label = "warm_l2_uncontrolled"
    if args.cache == "flush":
        flush = torch.empty(256 * 1024**2, dtype=torch.uint8, device=device)
        cache_label = "same_stream_l2_flush_256mib"
    print(
        f"# device={props.name} capability={props.major}.{props.minor} "
        f"torch={torch.__version__} cuda={torch.version.cuda} cache={cache_label}"
    )
    print("# kernel microbenchmark only; no end-to-end critical-path claim")
    print("shape,m,n,k,variant,mode,cache,reps,p50_us,tflops,speedup_vs_torch")

    for shape_name in shape_names:
        n, k = SHAPES[shape_name]
        weight = torch.randn((n, k), device=device, dtype=torch.bfloat16) * 0.02
        bias = (
            torch.randn(n, device=device, dtype=torch.bfloat16) * 0.01
            if args.bias
            else None
        )
        for m in ms:
            x = torch.randn((m, k), device=device, dtype=torch.bfloat16) * 0.1
            for mode in modes:
                baseline_us = None
                ordered_variants = ["torch"] + [v for v in variants if v != "torch"]
                if "torch" not in variants:
                    ordered_variants = variants
                for variant in ordered_variants:
                    keepalive = []
                    if variant == "torch":

                        def call():
                            return F.linear(x, weight, bias)

                        result = call()
                    else:
                        call, result, workspace = _make_custom_call(
                            x, weight, bias, VARIANTS[variant]
                        )
                        keepalive.extend((result, workspace))
                        call()
                        _check_result(
                            f"{shape_name}/M{m}/{variant}",
                            result,
                            x,
                            weight,
                            bias,
                        )
                    if mode == "graph":
                        call, result, graph = _capture(call)
                        keepalive.extend((result, graph))
                    p50_us = _measure_us(
                        call,
                        reps=args.reps,
                        warmup=args.warmup,
                        flush=flush,
                    )
                    if variant == "torch":
                        baseline_us = p50_us
                    speedup = (
                        baseline_us / p50_us
                        if baseline_us is not None
                        else float("nan")
                    )
                    tflops = 2 * m * n * k / (p50_us * 1e-6) / 1e12
                    print(
                        f"{shape_name},{m},{n},{k},{variant},{mode},"
                        f"{cache_label},{args.reps},{p50_us:.3f},{tflops:.5f},"
                        f"{speedup:.4f}"
                    )
                    del keepalive


if __name__ == "__main__":
    main()

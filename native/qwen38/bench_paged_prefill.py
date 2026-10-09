#!/usr/bin/env python3
"""Microbenchmark Qwen3.8 packed versus direct-paged FP8 chunk prefill.

This is a kernel-route experiment, not an end-to-end serving benchmark.  It
times the existing full-history gather/dequantize/packed-kernel route against
the direct paged kernel and validates every candidate before measuring it.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import statistics
import subprocess
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import torch
import triton

from sglang.srt.layers.attention.qsa import sparse_attn


TILE_CONFIGS = {
    "default": None,
    "n16w1s1": (16, 1, 1),
    "n16w2": (16, 2, 2),
    "n16w4": (16, 4, 2),
    "n32w2": (32, 2, 2),
    "n32w4": (32, 4, 2),
    "n64w4": (64, 4, 2),
    "n128w4": (128, 4, 2),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seq-lens", type=int, nargs="+", default=[32768, 131072])
    parser.add_argument("--query-len", type=int, default=8192)
    parser.add_argument("--topk", type=int, default=2048)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--reps", type=int, default=7)
    parser.add_argument("--seed", type=int, default=20261007)
    parser.add_argument(
        "--layout-profile",
        choices=("adversarial_token", "page64_block4"),
        default="adversarial_token",
        help="Physical slot mapping and logical sparse-index structure.",
    )
    parser.add_argument(
        "--scale-form", choices=("python", "device0d", "vector"), default="python"
    )
    parser.add_argument(
        "--scale-profiles",
        nargs="+",
        choices=("unit", "nonunit"),
        default=["unit", "nonunit"],
    )
    parser.add_argument(
        "--tiles", nargs="+", choices=tuple(TILE_CONFIGS), default=list(TILE_CONFIGS)
    )
    parser.add_argument(
        "--flush-mib",
        type=int,
        default=0,
        help="Optional same-stream uint8 cache sweep before every measured call.",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=None,
        help="Output path without extension; default is a fresh /tmp timestamp.",
    )
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sampled_tensor_sha256(tensor: torch.Tensor, samples: int = 4096) -> str:
    raw = tensor.detach().contiguous().view(torch.uint8).flatten()
    if raw.numel() > samples:
        positions = torch.arange(samples, dtype=torch.int64, device=raw.device)
        positions.mul_(raw.numel() - 1).div_(samples - 1, rounding_mode="floor")
        raw = raw.index_select(0, positions)
    digest = hashlib.sha256()
    digest.update(
        str((tuple(tensor.shape), tuple(tensor.stride()), tensor.dtype)).encode()
    )
    digest.update(raw.cpu().numpy().tobytes())
    return digest.hexdigest()


def git_output(*args: str) -> str:
    return subprocess.run(
        ["git", *args], check=True, text=True, capture_output=True
    ).stdout.strip()


def make_scale(value: float, form: str):
    if form == "python":
        return value
    scale = torch.tensor(value, dtype=torch.float32, device="cuda")
    return scale if form == "device0d" else scale.reshape(1)


def make_case(
    seq_len: int, query_len: int, topk: int, seed: int, layout_profile: str
) -> dict:
    if not 1 <= query_len <= seq_len or not 1 <= topk <= seq_len - query_len + 1:
        raise ValueError("require 1 <= query_len <= seq_len and topk <= prefix+1")
    torch.manual_seed(seed + seq_len + query_len)
    capacity = seq_len + 64
    table = torch.empty((1, seq_len), dtype=torch.int32, device="cuda")
    if layout_profile == "adversarial_token":
        table[0] = (
            torch.randperm(seq_len, dtype=torch.int64, device="cuda").to(torch.int32)
            + 1
        )
    elif layout_profile == "page64_block4":
        if seq_len % 64 or topk % 4:
            raise ValueError("page64_block4 requires seq_len % 64 == topk % 4 == 0")
        pages = torch.randperm(seq_len // 64, dtype=torch.int32, device="cuda") + 1
        offsets = torch.arange(64, dtype=torch.int32, device="cuda")
        table[0] = (pages[:, None] * 64 + offsets[None, :]).reshape(-1)
    else:
        raise ValueError(f"unknown layout profile: {layout_profile}")
    k = (torch.randn((capacity, 2, 256), dtype=torch.bfloat16, device="cuda") * 0.3).to(
        torch.float8_e4m3fn
    )
    v = (torch.randn((capacity, 2, 256), dtype=torch.bfloat16, device="cuda") * 0.3).to(
        torch.float8_e4m3fn
    )
    q = (
        torch.randn((query_len, 24, 256), dtype=torch.bfloat16, device="cuda") * 0.1
    ).to(torch.bfloat16)
    visible = torch.arange(
        seq_len - query_len + 1,
        seq_len + 1,
        dtype=torch.int32,
        device="cuda",
    )
    if layout_profile == "adversarial_token":
        columns = torch.arange(topk, dtype=torch.int32, device="cuda")
        indices = torch.div(
            visible[:, None] * columns[None, :], topk, rounding_mode="floor"
        ).contiguous()
    else:
        group_count = torch.div(visible + 3, 4, rounding_mode="floor")
        group_columns = torch.arange(topk // 4, dtype=torch.int32, device="cuda")
        selected_groups = torch.div(
            group_columns[None, :] * (group_count[:, None] - 1),
            topk // 4 - 1,
            rounding_mode="floor",
        )
        indices = (
            selected_groups[:, :, None] * 4
            + torch.arange(4, dtype=torch.int32, device="cuda")[None, None, :]
        ).reshape(query_len, topk)
        indices.masked_fill_(indices >= visible[:, None], -1)
    cu_q = torch.tensor([0, query_len], dtype=torch.int32, device="cuda")
    cu_k = torch.tensor([0, seq_len], dtype=torch.int32, device="cuda")
    kv_lens = torch.tensor([seq_len], dtype=torch.int32, device="cuda")
    requests = torch.zeros(1, dtype=torch.int32, device="cuda")
    slots = table[0].long()
    originals = {
        "q": q.clone(),
        "k": k.view(torch.uint8).clone(),
        "v": v.view(torch.uint8).clone(),
    }
    return {
        "q": q,
        "k": k,
        "v": v,
        "indices": indices,
        "cu_q": cu_q,
        "cu_k": cu_k,
        "kv_lens": kv_lens,
        "table": table,
        "requests": requests,
        "slots": slots,
        "originals": originals,
        "fingerprints": {
            name: sampled_tensor_sha256(value)
            for name, value in (
                ("q", q),
                ("k", k),
                ("v", v),
                ("indices", indices),
                ("table", table),
            )
        },
    }


def packed_route(case: dict, k_scale, v_scale) -> torch.Tensor:
    packed_k = (
        case["k"].index_select(0, case["slots"]).to(torch.bfloat16) * k_scale
    ).to(torch.bfloat16)
    packed_v = (
        case["v"].index_select(0, case["slots"]).to(torch.bfloat16) * v_scale
    ).to(torch.bfloat16)
    return sparse_attn.sparse_gqa_fwd_interface_triton_ck(
        case["q"],
        packed_k,
        packed_v,
        case["indices"],
        case["cu_q"],
        case["cu_k"],
        case["kv_lens"],
        256**-0.5,
        max_query_len=case["q"].shape[0],
    )


def paged_route(case: dict, k_scale, v_scale) -> torch.Tensor:
    return sparse_attn.sparse_gqa_fwd_interface_triton_paged_ck(
        case["q"],
        case["k"],
        case["v"],
        case["indices"],
        case["cu_q"],
        case["kv_lens"],
        case["table"],
        case["requests"],
        256**-0.5,
        max_query_len=case["q"].shape[0],
        k_scale=k_scale,
        v_scale=v_scale,
    )


@contextmanager
def tile_override(config):
    if config is None:
        yield
    else:
        with patch.object(sparse_attn, "_get_best_config", return_value=config):
            yield


def sweep_cache(flush: torch.Tensor | None) -> None:
    if flush is not None:
        flush.add_(1)


def stats(values: list[float]) -> dict:
    ordered = sorted(values)
    return {
        "min": min(values),
        "median": statistics.median(values),
        "mean": statistics.fmean(values),
        "max": max(values),
        "p20": ordered[max(0, round(0.2 * (len(ordered) - 1)))],
        "p80": ordered[min(len(ordered) - 1, round(0.8 * (len(ordered) - 1)))],
        "samples": values,
    }


def measure(call, warmup: int, reps: int, flush: torch.Tensor | None) -> dict:
    for _ in range(warmup):
        sweep_cache(flush)
        call()
    torch.cuda.synchronize()
    event_ms = []
    for _ in range(reps):
        sweep_cache(flush)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        call()
        end.record()
        end.synchronize()
        event_ms.append(start.elapsed_time(end))
    wall_ms = []
    for _ in range(reps):
        torch.cuda.synchronize()
        begin = time.perf_counter_ns()
        sweep_cache(flush)
        call()
        torch.cuda.synchronize()
        wall_ms.append((time.perf_counter_ns() - begin) / 1e6)
    return {"event_ms": stats(event_ms), "wall_ms": stats(wall_ms)}


def unchanged(case: dict) -> bool:
    return (
        torch.equal(case["q"], case["originals"]["q"])
        and torch.equal(case["k"].view(torch.uint8), case["originals"]["k"])
        and torch.equal(case["v"].view(torch.uint8), case["originals"]["v"])
    )


def main() -> int:
    args = parse_args()
    if args.reps < 5:
        raise ValueError("--reps must be at least 5")
    if torch.cuda.get_device_capability(0) != (12, 0):
        raise RuntimeError("benchmark requires the authorized SM120 GPU")
    now = datetime.now(timezone.utc)
    prefix = args.output_prefix or Path(
        f"/tmp/q38-paged-prefill-{now.strftime('%Y%m%dT%H%M%SZ')}"
    )
    prefix.parent.mkdir(parents=True, exist_ok=True)
    script = Path(__file__).resolve()
    source = Path(sparse_attn.__file__).resolve()
    flush = (
        torch.zeros(args.flush_mib * 1024 * 1024, dtype=torch.uint8, device="cuda")
        if args.flush_mib
        else None
    )
    report = {
        "schema_version": 1,
        "created_utc": now.isoformat(),
        "claim_scope": "isolated kernel-route microbenchmark; not an end-to-end critical-path claim",
        "command": sys.argv,
        "config": vars(args) | {"output_prefix": str(prefix)},
        "fingerprint": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "triton": triton.__version__,
            "device": torch.cuda.get_device_name(0),
            "capability": list(torch.cuda.get_device_capability(0)),
            "git_head": git_output("rev-parse", "HEAD"),
            "git_status_sha256": hashlib.sha256(
                git_output("status", "--short").encode()
            ).hexdigest(),
            "script_path": str(script),
            "script_sha256": file_sha256(script),
            "sparse_attn_path": str(source),
            "sparse_attn_sha256": file_sha256(source),
        },
        "cache_policy": (
            f"same_stream_{args.flush_mib}MiB_uint8_sweep"
            if flush is not None
            else "warm_no_flush"
        ),
        "results": [],
    }
    for seq_len in args.seq_lens:
        workload = make_case(
            seq_len,
            args.query_len,
            args.topk,
            args.seed,
            args.layout_profile,
        )
        shape = {
            name: list(workload[name].shape)
            for name in ("q", "k", "v", "indices", "table")
        }
        default_config = sparse_attn._get_best_config(args.query_len)
        for profile in args.scale_profiles:
            k_value, v_value = (1.0, 1.0) if profile == "unit" else (0.31, 0.27)
            k_scale = make_scale(k_value, args.scale_form)
            v_scale = make_scale(v_value, args.scale_form)
            baseline = packed_route(workload, k_scale, v_scale)
            torch.cuda.synchronize()
            baseline_measurement = measure(
                lambda: packed_route(workload, k_scale, v_scale),
                args.warmup,
                args.reps,
                flush,
            )
            report["results"].append(
                {
                    "seq_len": seq_len,
                    "query_len": args.query_len,
                    "topk": args.topk,
                    "scale_profile": profile,
                    "scale_form": args.scale_form,
                    "route": "packed_gather_dequant_ck",
                    "tile": "default",
                    "tile_config": list(default_config),
                    "shape": shape,
                    "input_sampled_sha256": workload["fingerprints"],
                    "correctness": {
                        "reference": True,
                        "finite": bool(torch.isfinite(baseline).all().item()),
                    },
                    "timing": baseline_measurement,
                }
            )
            for tile in args.tiles:
                config = TILE_CONFIGS[tile]
                try:
                    with tile_override(config):
                        candidate = paged_route(workload, k_scale, v_scale)
                        torch.cuda.synchronize()
                        difference = (candidate.float() - baseline.float()).abs()
                        denominator = baseline.float().abs().clamp_min(1e-12)
                        max_abs = float(difference.max().item())
                        max_rel = float((difference / denominator).max().item())
                        torch.testing.assert_close(
                            candidate, baseline, atol=0.006, rtol=0.02
                        )
                        candidate_measurement = measure(
                            lambda: paged_route(workload, k_scale, v_scale),
                            args.warmup,
                            args.reps,
                            flush,
                        )
                except Exception as exc:
                    failed = {
                        "seq_len": seq_len,
                        "query_len": args.query_len,
                        "topk": args.topk,
                        "scale_profile": profile,
                        "scale_form": args.scale_form,
                        "route": "paged_fp8_ck",
                        "tile": tile,
                        "tile_config": list(
                            default_config if config is None else config
                        ),
                        "shape": shape,
                        "input_sampled_sha256": workload["fingerprints"],
                        "status": "error",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                    report["results"].append(failed)
                    print(
                        json.dumps(
                            {
                                "seq_len": seq_len,
                                "scale": profile,
                                "tile": tile,
                                "status": "error",
                                "error": str(exc),
                            }
                        ),
                        flush=True,
                    )
                    continue
                passed = {
                    "seq_len": seq_len,
                    "query_len": args.query_len,
                    "topk": args.topk,
                    "scale_profile": profile,
                    "scale_form": args.scale_form,
                    "route": "paged_fp8_ck",
                    "tile": tile,
                    "tile_config": list(default_config if config is None else config),
                    "shape": shape,
                    "input_sampled_sha256": workload["fingerprints"],
                    "status": "ok",
                    "correctness": {
                        "atol": 0.006,
                        "rtol": 0.02,
                        "max_abs_error": max_abs,
                        "max_relative_error": max_rel,
                        "finite": bool(torch.isfinite(candidate).all().item()),
                    },
                    "timing": candidate_measurement,
                }
                report["results"].append(passed)
                print(
                    json.dumps(
                        {
                            "seq_len": seq_len,
                            "scale": profile,
                            "tile": tile,
                            "status": "ok",
                            "event_median_ms": candidate_measurement["event_ms"][
                                "median"
                            ],
                            "max_abs_error": max_abs,
                        }
                    ),
                    flush=True,
                )
            if not unchanged(workload):
                raise AssertionError("benchmark route mutated original Q/K/V tensors")
            del baseline
            torch.cuda.empty_cache()
        torch.cuda.empty_cache()
    json_path = prefix.with_suffix(".json")
    csv_path = prefix.with_suffix(".csv")
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    with csv_path.open("w", newline="") as handle:
        fieldnames = [
            "seq_len",
            "query_len",
            "topk",
            "scale_profile",
            "scale_form",
            "route",
            "tile",
            "block_n",
            "warps",
            "stages",
            "event_median_ms",
            "wall_median_ms",
            "max_abs_error",
            "max_relative_error",
            "status",
            "error",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in report["results"]:
            writer.writerow(
                {
                    "seq_len": row["seq_len"],
                    "query_len": row["query_len"],
                    "topk": row["topk"],
                    "scale_profile": row["scale_profile"],
                    "scale_form": row["scale_form"],
                    "route": row["route"],
                    "tile": row["tile"],
                    "block_n": row["tile_config"][0],
                    "warps": row["tile_config"][1],
                    "stages": row["tile_config"][2],
                    "event_median_ms": row.get("timing", {})
                    .get("event_ms", {})
                    .get("median", ""),
                    "wall_median_ms": row.get("timing", {})
                    .get("wall_ms", {})
                    .get("median", ""),
                    "max_abs_error": row.get("correctness", {}).get(
                        "max_abs_error", ""
                    ),
                    "max_relative_error": row.get("correctness", {}).get(
                        "max_relative_error", 0.0
                    ),
                    "status": row.get("status", "ok"),
                    "error": row.get("error", ""),
                }
            )
    print(
        json.dumps(
            {
                "json": str(json_path),
                "csv": str(csv_path),
                "rows": len(report["results"]),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

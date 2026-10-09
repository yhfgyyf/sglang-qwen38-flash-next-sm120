"""Microbenchmark the normal and coarse-bin-overflow fast_topk paths."""

from __future__ import annotations

import argparse
import json

import torch

from sglang.kernels.ops.elementwise.fast_topk import fast_topk


def _score(batch: int, length: int, topk: int, distribution: str) -> torch.Tensor:
    if distribution == "random":
        return torch.randn(batch, length, dtype=torch.float32, device="cuda")
    if distribution == "cutoff_ties":
        score = torch.full((batch, length), -1.0, dtype=torch.float32, device="cuda")
        if length > topk:
            selected_ties = min(32, topk)
            greater_count = topk - selected_ties
            tie_count = min(max(128, length // 2), length - greater_count)
            score[:, :tie_count] = 0.0
            score[:, tie_count : tie_count + greater_count] = 1.0
        return score
    row = torch.linspace(1.0, 1.1, length, dtype=torch.float32, device="cuda")
    return row.repeat(batch, 1)


def _measure(
    batch: int,
    length: int,
    topk: int,
    distribution: str,
    stable_ties: bool,
    warmup: int,
    reps: int,
) -> float:
    score = _score(batch, length, topk, distribution)
    lengths = torch.full((batch,), length, dtype=torch.int32, device="cuda")
    starts = torch.zeros(batch, dtype=torch.int32, device="cuda")
    for _ in range(warmup):
        fast_topk(
            score,
            lengths,
            topk,
            row_starts=starts,
            stable_ties=stable_ties,
        )
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(reps):
        fast_topk(
            score,
            lengths,
            topk,
            row_starts=starts,
            stable_ties=stable_ties,
        )
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000.0 / reps


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--topk", type=int, choices=(512, 2048), default=512)
    parser.add_argument("--stable-ties", action="store_true")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--reps", type=int, default=20)
    args = parser.parse_args()

    torch.manual_seed(0)
    for distribution in ("random", "same_coarse_bin", "cutoff_ties"):
        for batch in (1, 4, 128):
            for length in (2048, 8192, 32768):
                latency_us = _measure(
                    batch,
                    length,
                    args.topk,
                    distribution,
                    args.stable_ties,
                    args.warmup,
                    args.reps,
                )
                print(
                    json.dumps(
                        {
                            "distribution": distribution,
                            "batch": batch,
                            "length": length,
                            "topk": args.topk,
                            "stable_ties": args.stable_ties,
                            "latency_us": round(latency_us, 3),
                            "warmup": args.warmup,
                            "reps": args.reps,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )


if __name__ == "__main__":
    main()

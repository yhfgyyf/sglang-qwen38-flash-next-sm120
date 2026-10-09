"""Microbenchmark opt-in canonical QSA block expansion."""

from __future__ import annotations

import argparse
import json

import torch

from sglang.srt.layers.attention.qsa.kernel import expand_qsa_block_indices


def _measure(
    batch: int,
    block_topk: int,
    canonical_order: bool,
    warmup: int,
    reps: int,
) -> float:
    ratio = 4
    token_topk = block_topk * ratio
    blocks = torch.rand(batch, block_topk, device="cuda").argsort(dim=1).int()
    query_positions = torch.full(
        (batch,), token_topk + 2, dtype=torch.int32, device="cuda"
    )
    sequence_lengths = query_positions + 1

    for _ in range(warmup):
        expand_qsa_block_indices(
            blocks,
            query_positions,
            sequence_lengths,
            ratio,
            token_topk,
            canonical_order=canonical_order,
        )
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(reps):
        expand_qsa_block_indices(
            blocks,
            query_positions,
            sequence_lengths,
            ratio,
            token_topk,
            canonical_order=canonical_order,
        )
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000.0 / reps


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--reps", type=int, default=20)
    args = parser.parse_args()

    torch.manual_seed(0)
    for block_topk in (512, 2048):
        for batch in (1, 4, 128):
            for canonical_order in (False, True):
                latency_us = _measure(
                    batch,
                    block_topk,
                    canonical_order,
                    args.warmup,
                    args.reps,
                )
                print(
                    json.dumps(
                        {
                            "batch": batch,
                            "block_topk": block_topk,
                            "canonical_order": canonical_order,
                            "latency_us": round(latency_us, 3),
                            "reps": args.reps,
                            "warmup": args.warmup,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )


if __name__ == "__main__":
    main()

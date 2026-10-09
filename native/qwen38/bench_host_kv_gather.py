#!/usr/bin/env python3
"""Benchmark the native mapped-host FP8 KV gather over 12 target layers.

This is a bounded microbenchmark of the native gather, not an end-to-end
serving result or a measurement of physical PCIe bandwidth.  The requested
library is loaded explicitly so old/new binaries can be compared without
overwriting the repository build.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch


LAYERS = 12
HEADS = 2
HEAD_DIM = 256
ROW_BYTES = HEADS * HEAD_DIM
CONTEXT_SLOTS = 32 * 1024
VERIFY_QUERIES = 4
BLOCKS_PER_QUERY = 512
BLOCK_SIZE = 4
PENDING_PER_QUERY = 3
VALID_ROWS = BLOCKS_PER_QUERY * BLOCK_SIZE + PENDING_PER_QUERY
ROW_STRIDE = 2112
PADDING_ROWS = ROW_STRIDE - VALID_ROWS
MAX_ARENA_BYTES = 5 * 1024**3
HOST_RESERVE_BYTES = 8 * 1024**3
SLOT_ROUNDING = 4096
INITIALIZE_CHUNK_ROWS = 4096


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--library",
        type=Path,
        required=True,
        help="Explicit libq38_host_kv.so candidate; the repository binary is untouched.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Fresh JSON path. No file is written when this option is omitted.",
    )
    parser.add_argument(
        "--concurrencies",
        type=int,
        nargs="+",
        choices=(1, 4, 6, 8, 10),
        default=[1, 4, 6, 8, 10],
    )
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--samples", type=int, default=11)
    parser.add_argument("--seed", type=int, default=20261008)
    parser.add_argument("--id-width", type=int, choices=(32, 64), default=32)
    parser.add_argument(
        "--locality",
        choices=("shared-neighbor", "independent"),
        default="shared-neighbor",
        help="Whether a request's four neighboring queries share most blocks.",
    )
    parser.add_argument(
        "--cache-condition",
        choices=("warm", "sweep", "both"),
        default="both",
        help="No-flush layer cycling, best-effort same-stream sweep, or both.",
    )
    parser.add_argument(
        "--sweep-mib",
        type=int,
        default=192,
        help="Best-effort cache-displacement tensor size; excluded from CUDA timing.",
    )
    args = parser.parse_args(argv)
    if args.warmup < 1:
        parser.error("--warmup must be at least 1")
    if args.samples < 11:
        parser.error("--samples must be at least 11")
    if args.sweep_mib < 1:
        parser.error("--sweep-mib must be positive")
    return args


def round_up(value: int, multiple: int) -> int:
    return (value + multiple - 1) // multiple * multiple


def slot_count_for(concurrency: int) -> int:
    return round_up(1 + concurrency * CONTEXT_SLOTS, SLOT_ROUNDING)


def build_gather_ids(concurrency: int, seed: int, locality: str) -> torch.Tensor:
    """Create flattened request/query/row IDs on CPU for deterministic reuse."""
    if concurrency not in (1, 4, 6, 8, 10):
        raise ValueError("concurrency must be one of 1, 4, 6, 8, 10")
    if locality not in ("shared-neighbor", "independent"):
        raise ValueError(f"unknown locality: {locality}")
    requests = []
    offsets = torch.arange(BLOCK_SIZE, dtype=torch.int64)
    padding = torch.full((PADDING_ROWS,), -1, dtype=torch.int64)
    for request in range(concurrency):
        generator = torch.Generator().manual_seed(seed + request * 1_000_003)
        context_start = 1 + request * CONTEXT_SLOTS
        query_rows = []
        if locality == "shared-neighbor":
            permutation = torch.randperm(
                CONTEXT_SLOTS // BLOCK_SIZE, generator=generator
            )
        for query in range(VERIFY_QUERIES):
            if locality == "shared-neighbor":
                common = permutation[:448]
                query_specific = permutation[448 + 64 * query : 512 + 64 * query]
                blocks = torch.cat((common, query_specific))
                pending_block = permutation[704 + query]
            else:
                permutation = torch.randperm(
                    CONTEXT_SLOTS // BLOCK_SIZE, generator=generator
                )
                blocks = permutation[:BLOCKS_PER_QUERY]
                pending_block = permutation[BLOCKS_PER_QUERY]
            selected = blocks[:, None].mul(BLOCK_SIZE).add(offsets[None, :]).reshape(-1)
            pending = pending_block.mul(BLOCK_SIZE).add(offsets[:PENDING_PER_QUERY])
            positive = torch.cat((selected, pending)).add(context_start)
            query_rows.append(torch.cat((positive, padding)))
        requests.append(torch.stack(query_rows))
    return torch.stack(requests).reshape(-1)


def locality_summary(ids: torch.Tensor, concurrency: int) -> dict:
    shaped = ids.reshape(concurrency, VERIFY_QUERIES, ROW_STRIDE)
    intersections = []
    unions = []
    for request in range(concurrency):
        sets = [
            set(shaped[request, query, :VALID_ROWS].tolist())
            for query in range(VERIFY_QUERIES)
        ]
        for left in range(VERIFY_QUERIES):
            for right in range(left + 1, VERIFY_QUERIES):
                intersections.append(len(sets[left] & sets[right]))
                unions.append(len(sets[left] | sets[right]))
    return {
        "pairwise_positive_intersection": {
            "min": min(intersections),
            "max": max(intersections),
            "mean": statistics.fmean(intersections),
        },
        "pairwise_positive_jaccard_mean": statistics.fmean(
            intersection / union for intersection, union in zip(intersections, unions)
        ),
    }


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_output(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *args],
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()


def mem_available_bytes() -> int:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    raise RuntimeError("/proc/meminfo does not report MemAvailable")


def raw_rows(ids: torch.Tensor, layer: int, value: bool) -> torch.Tensor:
    """Return deterministic raw FP8 bytes keyed only by physical ID and layer."""
    columns = torch.arange(256, dtype=torch.uint8, device=ids.device).repeat(2)
    id_bytes = torch.stack(
        [
            torch.div(ids, 256**byte, rounding_mode="floor")
            .remainder(256)
            .to(torch.uint8)
            for byte in range(4)
        ],
        dim=1,
    ).repeat(1, ROW_BYTES // 4)
    if value:
        return id_bytes.mul(31).add(columns.mul(7)).add(layer * 13 + 5)
    return id_bytes.mul(17).add(columns.mul(3)).add(layer * 29 + 11)


def expected_rows(ids: torch.Tensor, layer: int, value: bool) -> torch.Tensor:
    padding = ids.lt(0)
    result = raw_rows(ids.clamp_min(0), layer, value)
    result.masked_fill_(padding[:, None], 0)
    return result


def stats(samples: list[float]) -> dict:
    ordered = sorted(samples)
    return {
        "min": min(samples),
        "median": statistics.median(samples),
        "mean": statistics.fmean(samples),
        "max": max(samples),
        "p20": ordered[round(0.2 * (len(ordered) - 1))],
        "p80": ordered[round(0.8 * (len(ordered) - 1))],
        "samples": samples,
    }


def cache_conditions(name: str) -> list[str]:
    return ["warm", "sweep"] if name == "both" else [name]


def initialize_and_validate(
    arena,
    ids: torch.Tensor,
    unique_ids: torch.Tensor,
    output_k: torch.Tensor,
    output_v: torch.Tensor,
    stream: torch.cuda.Stream,
) -> None:
    with torch.cuda.stream(stream):
        for layer in range(LAYERS):
            for begin in range(0, unique_ids.numel(), INITIALIZE_CHUNK_ROWS):
                chunk_ids = unique_ids[begin : begin + INITIALIZE_CHUNK_ROWS]
                key_bytes = raw_rows(chunk_ids, layer, value=False)
                value_bytes = raw_rows(chunk_ids, layer, value=True)
                arena.scatter(
                    layer,
                    chunk_ids,
                    key_bytes.view(torch.float8_e4m3fn).reshape(-1, HEADS, HEAD_DIM),
                    value_bytes.view(torch.float8_e4m3fn).reshape(-1, HEADS, HEAD_DIM),
                    stream=stream,
                )
            arena.gather(layer, ids, output_k, output_v, stream=stream)
            arena.check_errors()
            expected_k = expected_rows(ids, layer, value=False)
            expected_v = expected_rows(ids, layer, value=True)
            actual_k = output_k.view(torch.uint8).reshape(-1, ROW_BYTES)
            actual_v = output_v.view(torch.uint8).reshape(-1, ROW_BYTES)
            if not torch.equal(actual_k, expected_k):
                mismatches = torch.count_nonzero(actual_k != expected_k).item()
                raise AssertionError(
                    f"layer {layer} key gather differs in {mismatches} raw bytes"
                )
            if not torch.equal(actual_v, expected_v):
                mismatches = torch.count_nonzero(actual_v != expected_v).item()
                raise AssertionError(
                    f"layer {layer} value gather differs in {mismatches} raw bytes"
                )


def measure(
    arena,
    ids: torch.Tensor,
    output_k: torch.Tensor,
    output_v: torch.Tensor,
    stream: torch.cuda.Stream,
    warmup: int,
    sample_count: int,
    cache_condition: str,
    sweep: torch.Tensor | None,
) -> dict:
    def run_layers() -> None:
        for layer in range(LAYERS):
            arena.gather(layer, ids, output_k, output_v, stream=stream)

    with torch.cuda.stream(stream):
        for _ in range(warmup):
            if cache_condition == "sweep":
                sweep.add_(1)
            run_layers()
    stream.synchronize()

    events = [
        (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
        for _ in range(sample_count)
    ]
    samples = []
    for start, end in events:
        with torch.cuda.stream(stream):
            if cache_condition == "sweep":
                sweep.add_(1)
            start.record(stream)
            run_layers()
            end.record(stream)
        end.synchronize()
        samples.append(start.elapsed_time(end))
    arena.check_errors()
    return stats(samples)


def run_case(args, host_kv, concurrency: int) -> dict:
    cpu_ids = build_gather_ids(concurrency, args.seed, args.locality)
    positive_ids = torch.unique(cpu_ids[cpu_ids >= 0], sorted=True)
    slots = slot_count_for(concurrency)
    arena_bytes = host_kv.required_bytes(
        LAYERS,
        slots,
        HEADS,
        HEAD_DIM,
        dtype=torch.float8_e4m3fn,
    )
    if arena_bytes > MAX_ARENA_BYTES:
        raise MemoryError(
            f"C={concurrency} arena needs {arena_bytes} bytes, above the 5 GiB bound"
        )
    available_before = mem_available_bytes()
    if available_before - arena_bytes < HOST_RESERVE_BYTES:
        raise MemoryError(
            f"C={concurrency} would leave less than 8 GiB MemAvailable: "
            f"available={available_before}, arena={arena_bytes}"
        )

    id_dtype = torch.int32 if args.id_width == 32 else torch.int64
    stream = torch.cuda.Stream()
    arena = host_kv.HostKVArena(
        LAYERS,
        slots,
        heads=HEADS,
        head_dim=HEAD_DIM,
        dtype=torch.float8_e4m3fn,
        byte_budget=arena_bytes,
        label=f"host-kv-gather-C{concurrency}",
    )
    try:
        with torch.cuda.stream(stream):
            ids = cpu_ids.to(device="cuda", dtype=id_dtype)
            unique_ids = positive_ids.to(device="cuda", dtype=id_dtype)
            output_k = torch.empty(
                (ids.numel(), HEADS, HEAD_DIM),
                dtype=torch.float8_e4m3fn,
                device="cuda",
            )
            output_v = torch.empty_like(output_k)
            sweep = (
                torch.zeros(
                    args.sweep_mib * 1024 * 1024,
                    dtype=torch.uint8,
                    device="cuda",
                )
                if "sweep" in cache_conditions(args.cache_condition)
                else None
            )
        initialize_and_validate(arena, ids, unique_ids, output_k, output_v, stream)
        logical_output_bytes = ids.numel() * 2 * ROW_BYTES * LAYERS
        positive_host_bytes = int(ids.numel() - PADDING_ROWS * concurrency * 4)
        positive_host_bytes *= 2 * ROW_BYTES * LAYERS
        measurements = []
        for condition in cache_conditions(args.cache_condition):
            timing = measure(
                arena,
                ids,
                output_k,
                output_v,
                stream,
                args.warmup,
                args.samples,
                condition,
                sweep,
            )
            median_seconds = timing["median"] / 1000
            measurements.append(
                {
                    "cache_condition": condition,
                    "timing_sum_12_layers_ms": timing,
                    "timing_per_layer_ms": {
                        key: value / LAYERS
                        for key, value in timing.items()
                        if key != "samples"
                    }
                    | {"samples": [value / LAYERS for value in timing["samples"]]},
                    "calculated_payload_bandwidth": {
                        "positive_host_payload_gib_s": positive_host_bytes
                        / median_seconds
                        / 1024**3,
                        "logical_output_payload_gib_s": logical_output_bytes
                        / median_seconds
                        / 1024**3,
                        "physical_pcie_bandwidth_claim": False,
                    },
                }
            )
        return {
            "concurrency": concurrency,
            "status": "ok",
            "correctness": "all 12 layers exact by raw FP8 byte before timing",
            "shape": {
                "requests": concurrency,
                "verify_queries_per_request": VERIFY_QUERIES,
                "rows_per_query": ROW_STRIDE,
                "valid_rows_per_query": VALID_ROWS,
                "negative_padding_rows_per_query": PADDING_ROWS,
                "gather_rows": ids.numel(),
                "heads": HEADS,
                "head_dim": HEAD_DIM,
                "dtype": "torch.float8_e4m3fn",
                "row_bytes": ROW_BYTES,
                "layers_per_timing_iteration": LAYERS,
            },
            "arena": {
                "context_slots_per_request": CONTEXT_SLOTS,
                "slot_count_rounded": slots,
                "bytes": arena_bytes,
                "max_bytes": MAX_ARENA_BYTES,
                "mem_available_before_bytes": available_before,
                "required_mem_available_reserve_bytes": HOST_RESERVE_BYTES,
            },
            "selection": {
                "locality": args.locality,
                "seed": args.seed,
                "unique_positive_slots": positive_ids.numel(),
                **locality_summary(cpu_ids, concurrency),
            },
            "payload": {
                "positive_host_payload_bytes_per_12_layers": positive_host_bytes,
                "unique_host_payload_bytes_per_12_layers": positive_ids.numel()
                * 2
                * ROW_BYTES
                * LAYERS,
                "logical_output_payload_bytes_per_12_layers": logical_output_bytes,
            },
            "measurements": measurements,
        }
    finally:
        arena.close()


def gpu_properties() -> dict:
    properties = torch.cuda.get_device_properties(0)
    return {
        "name": properties.name,
        "capability": list(torch.cuda.get_device_capability(0)),
        "total_memory": properties.total_memory,
        "multi_processor_count": properties.multi_processor_count,
        "l2_cache_size": getattr(properties, "L2_cache_size", None),
        "pci_bus_id": getattr(properties, "pci_bus_id", None),
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    library_path = args.library.expanduser().resolve(strict=True)
    if args.output is not None and args.output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {args.output}")
    script = Path(__file__).resolve()
    source = script.with_name("host_kv.cu")
    repo = script.parents[2]
    # This must precede the first call to the module's cached library loader.
    os.environ["QWEN38_HOST_KV_LIBRARY"] = str(library_path)
    module_path = repo / "python/sglang/srt/model_executor/qwen38_host_kv.py"
    spec = importlib.util.spec_from_file_location(
        "q38_bench_host_kv_module", module_path
    )
    host_kv = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = host_kv
    spec.loader.exec_module(host_kv)

    host_kv.library.cache_clear()
    host_kv.library()
    if not torch.cuda.is_available() or torch.cuda.get_device_capability(0) != (12, 0):
        raise RuntimeError("benchmark requires the authorized SM120 GPU")

    created = datetime.now(timezone.utc)
    report = {
        "schema_version": 1,
        "created_utc": created.isoformat(),
        "claim_scope": (
            "native mapped-host gather microbenchmark; not end-to-end and "
            "not a physical PCIe bandwidth measurement"
        ),
        "command": sys.argv if argv is None else [sys.argv[0], *argv],
        "config": {
            "library": str(library_path),
            "output": str(args.output) if args.output is not None else None,
            "concurrencies": args.concurrencies,
            "warmup": args.warmup,
            "samples": args.samples,
            "seed": args.seed,
            "id_width": args.id_width,
            "locality": args.locality,
            "cache_condition": args.cache_condition,
            "sweep_mib": args.sweep_mib,
            "cuda_graph": False,
            "cuda_graph_reason": "direct caller-stream path isolates the native gather",
        },
        "cache_conditions": {
            "warm": (
                "no explicit flush; each sample cycles 12 disjoint layers; "
                "repeated neighboring-query IDs reduce the unique host working "
                "set below logical payload and C=1 may remain L2-resident"
            ),
            "sweep": (
                f"best-effort same-stream {args.sweep_mib} MiB uint8 add before "
                "each sample, ordered before and excluded from CUDA event timing; "
                "not asserted to make every host line cold"
            ),
        },
        "fingerprint": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "gpu": gpu_properties(),
            "git_head": git_output(repo, "rev-parse", "HEAD"),
            "git_status_sha256": hashlib.sha256(
                git_output(repo, "status", "--short").encode()
            ).hexdigest(),
            "library_path": str(library_path),
            "library_sha256": file_sha256(library_path),
            "source_path": str(source),
            "source_sha256": file_sha256(source),
            "script_path": str(script),
            "script_sha256": file_sha256(script),
        },
        "results": [],
    }
    for concurrency in args.concurrencies:
        result = run_case(args, host_kv, concurrency)
        report["results"].append(result)
        medians = {
            item["cache_condition"]: item["timing_sum_12_layers_ms"]["median"]
            for item in result["measurements"]
        }
        print(
            json.dumps({"concurrency": concurrency, "median_ms": medians}), flush=True
        )
        torch.cuda.empty_cache()

    if args.output is not None:
        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x") as handle:
            json.dump(report, handle, indent=2)
            handle.write("\n")
        print(f"wrote {output}")
    else:
        print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

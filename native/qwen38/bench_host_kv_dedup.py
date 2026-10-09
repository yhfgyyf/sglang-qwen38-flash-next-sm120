#!/usr/bin/env python3
"""Compare ordinary and ephemeral-deduplicated mapped-host KV gathers."""

from __future__ import annotations

import argparse
import ctypes as C
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
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        help="Fresh JSON path; no file is written when omitted.",
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
        "--cache-condition",
        choices=("warm", "sweep", "both"),
        default="both",
    )
    parser.add_argument("--sweep-mib", type=int, default=192)
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


def build_gather_ids(concurrency: int, seed: int) -> torch.Tensor:
    """Build four substantially overlapping block-aligned queries/request."""
    if concurrency not in (1, 4, 6, 8, 10):
        raise ValueError("concurrency must be one of 1, 4, 6, 8, 10")
    requests = []
    offsets = torch.arange(BLOCK_SIZE, dtype=torch.int64)
    padding = torch.full((PADDING_ROWS,), -1, dtype=torch.int64)
    for request in range(concurrency):
        generator = torch.Generator().manual_seed(seed + request * 1_000_003)
        permutation = torch.randperm(CONTEXT_SLOTS // BLOCK_SIZE, generator=generator)
        context_start = 1 + request * CONTEXT_SLOTS
        queries = []
        for query in range(VERIFY_QUERIES):
            common = permutation[:448]
            query_specific = permutation[448 + query * 64 : 512 + query * 64]
            blocks = torch.cat((common, query_specific))
            selected = blocks[:, None].mul(BLOCK_SIZE).add(offsets[None, :]).reshape(-1)
            pending = (
                permutation[704 + query]
                .mul(BLOCK_SIZE)
                .add(offsets[:PENDING_PER_QUERY])
            )
            positive = torch.cat((selected, pending)).add(context_start)
            queries.append(torch.cat((positive, padding)))
        requests.append(torch.stack(queries))
    return torch.stack(requests).reshape(-1)


def raw_rows(ids: torch.Tensor, layer: int, value: bool) -> torch.Tensor:
    columns = torch.arange(256, dtype=torch.uint8, device=ids.device).repeat(2)
    physical_bytes = torch.stack(
        [
            torch.div(ids, 256**byte, rounding_mode="floor")
            .remainder(256)
            .to(torch.uint8)
            for byte in range(4)
        ],
        dim=1,
    ).repeat(1, ROW_BYTES // 4)
    if value:
        return physical_bytes.mul(31).add(columns.mul(7)).add(layer * 13 + 5)
    return physical_bytes.mul(17).add(columns.mul(3)).add(layer * 29 + 11)


def expected_rows(ids: torch.Tensor, layer: int, value: bool) -> torch.Tensor:
    padding = ids.lt(0)
    result = raw_rows(ids.clamp_min(0), layer, value)
    result.masked_fill_(padding[:, None], 0)
    return result


def load_host_kv_module(repo: Path, library_path: Path):
    """Load the unchanged wrapper only after selecting an explicit library."""
    resolved = library_path.expanduser().resolve(strict=True)
    os.environ["QWEN38_HOST_KV_LIBRARY"] = str(resolved)
    module_path = repo / "python/sglang/srt/model_executor/qwen38_host_kv.py"
    name = f"q38_host_kv_dedup_{file_sha256(resolved)[:12]}"
    spec = importlib.util.spec_from_file_location(name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load host KV wrapper from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    module.library.cache_clear()
    module.library()
    return module


def declare_dedup(lib):
    try:
        function = lib.q38_host_kv_gather_dedup
    except AttributeError as exc:
        raise RuntimeError(
            "selected host KV library does not provide optional dedup gather"
        ) from exc
    function.restype = C.c_int
    function.argtypes = [
        C.c_void_p,
        C.c_uint64,
        C.c_void_p,
        C.c_uint64,
        C.c_int,
        C.c_void_p,
        C.c_uint64,
        C.c_void_p,
        C.c_void_p,
        C.c_size_t,
    ]
    return function


def call_dedup(
    arena,
    function,
    layer: int,
    ids: torch.Tensor,
    row_map: torch.Tensor,
    output_k: torch.Tensor,
    output_v: torch.Tensor,
    stream: torch.cuda.Stream,
) -> None:
    status = function(
        arena._handle,
        layer,
        ids.data_ptr(),
        ids.numel(),
        ids.element_size(),
        row_map.data_ptr(),
        row_map.numel(),
        output_k.data_ptr(),
        output_v.data_ptr(),
        stream.cuda_stream,
    )
    if status != 0:
        message = arena._lib.q38_host_kv_last_error()
        raise RuntimeError(
            message.decode() if message else "native host KV dedup error"
        )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def mem_available_bytes() -> int:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    raise RuntimeError("/proc/meminfo does not report MemAvailable")


def git_output(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *args],
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()


def cache_conditions(name: str) -> list[str]:
    return ["warm", "sweep"] if name == "both" else [name]


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


def initialize_and_validate(
    arena,
    dedup_function,
    ids: torch.Tensor,
    unique_ids: torch.Tensor,
    row_map: torch.Tensor,
    ordinary_outputs: tuple[torch.Tensor, torch.Tensor],
    dedup_outputs: tuple[torch.Tensor, torch.Tensor],
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
            for output in (*ordinary_outputs, *dedup_outputs):
                output.view(torch.uint8).fill_(0xA5)
            oracle_k = expected_rows(ids, layer, value=False)
            oracle_v = expected_rows(ids, layer, value=True)
            arena.gather(layer, ids, *ordinary_outputs, stream=stream)
            call_dedup(
                arena,
                dedup_function,
                layer,
                ids,
                row_map,
                *dedup_outputs,
                stream,
            )
            arena.check_errors()
            for route, outputs in (
                ("ordinary", ordinary_outputs),
                ("dedup", dedup_outputs),
            ):
                actual_k = outputs[0].view(torch.uint8).reshape(-1, ROW_BYTES)
                actual_v = outputs[1].view(torch.uint8).reshape(-1, ROW_BYTES)
                if not torch.equal(actual_k, oracle_k):
                    mismatch = torch.count_nonzero(actual_k != oracle_k).item()
                    raise AssertionError(
                        f"{route} layer {layer} key differs in {mismatch} bytes"
                    )
                if not torch.equal(actual_v, oracle_v):
                    mismatch = torch.count_nonzero(actual_v != oracle_v).item()
                    raise AssertionError(
                        f"{route} layer {layer} value differs in {mismatch} bytes"
                    )


def measure_routes(
    arena,
    dedup_function,
    ids: torch.Tensor,
    row_map: torch.Tensor,
    ordinary_outputs: tuple[torch.Tensor, torch.Tensor],
    dedup_outputs: tuple[torch.Tensor, torch.Tensor],
    stream: torch.cuda.Stream,
    warmup: int,
    sample_count: int,
    condition: str,
    sweep: torch.Tensor | None,
) -> dict:
    def run_ordinary() -> None:
        for layer in range(LAYERS):
            arena.gather(layer, ids, *ordinary_outputs, stream=stream)

    def run_dedup() -> None:
        for layer in range(LAYERS):
            call_dedup(
                arena,
                dedup_function,
                layer,
                ids,
                row_map,
                *dedup_outputs,
                stream,
            )

    routes = {"ordinary": run_ordinary, "dedup": run_dedup}
    with torch.cuda.stream(stream):
        for _ in range(warmup):
            for route in ("ordinary", "dedup"):
                if condition == "sweep":
                    sweep.add_(1)
                routes[route]()
    stream.synchronize()

    events = {
        route: [
            (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            for _ in range(sample_count)
        ]
        for route in routes
    }
    samples = {route: [] for route in routes}
    for sample in range(sample_count):
        order = ("ordinary", "dedup") if sample % 2 == 0 else ("dedup", "ordinary")
        for route in order:
            start, end = events[route][sample]
            with torch.cuda.stream(stream):
                if condition == "sweep":
                    sweep.add_(1)
                start.record(stream)
                routes[route]()
                end.record(stream)
            end.synchronize()
            samples[route].append(start.elapsed_time(end))
    arena.check_errors()
    return {route: stats(values) for route, values in samples.items()}


def run_case(args, host_kv, dedup_function, concurrency: int) -> dict:
    cpu_ids = build_gather_ids(concurrency, args.seed)
    positive_ids = torch.unique(cpu_ids[cpu_ids >= 0], sorted=True)
    slots = slot_count_for(concurrency)
    arena_bytes = host_kv.required_bytes(
        LAYERS, slots, HEADS, HEAD_DIM, dtype=torch.float8_e4m3fn
    )
    if arena_bytes > MAX_ARENA_BYTES:
        raise MemoryError(f"C={concurrency} arena exceeds the 5 GiB cap")
    available = mem_available_bytes()
    if available - arena_bytes < HOST_RESERVE_BYTES:
        raise MemoryError(f"C={concurrency} would violate the 8 GiB host reserve")

    id_dtype = torch.int32 if args.id_width == 32 else torch.int64
    stream = torch.cuda.Stream()
    arena = host_kv.HostKVArena(
        LAYERS,
        slots,
        heads=HEADS,
        head_dim=HEAD_DIM,
        dtype=torch.float8_e4m3fn,
        byte_budget=arena_bytes,
        label=f"host-kv-dedup-C{concurrency}",
    )
    try:
        with torch.cuda.stream(stream):
            ids = cpu_ids.to(device="cuda", dtype=id_dtype)
            unique_ids = positive_ids.to(device="cuda", dtype=id_dtype)
            row_map = torch.empty(slots, dtype=torch.int32, device="cuda")
            ordinary_k = torch.empty(
                (ids.numel(), HEADS, HEAD_DIM),
                dtype=torch.float8_e4m3fn,
                device="cuda",
            )
            ordinary_v = torch.empty_like(ordinary_k)
            dedup_k = torch.empty_like(ordinary_k)
            dedup_v = torch.empty_like(ordinary_k)
            sweep = (
                torch.zeros(
                    args.sweep_mib * 1024 * 1024,
                    dtype=torch.uint8,
                    device="cuda",
                )
                if "sweep" in cache_conditions(args.cache_condition)
                else None
            )
        ordinary_outputs = (ordinary_k, ordinary_v)
        dedup_outputs = (dedup_k, dedup_v)
        initialize_and_validate(
            arena,
            dedup_function,
            ids,
            unique_ids,
            row_map,
            ordinary_outputs,
            dedup_outputs,
            stream,
        )
        positive_rows = concurrency * VERIFY_QUERIES * VALID_ROWS
        logical_output_bytes = ids.numel() * 2 * ROW_BYTES * LAYERS
        ordinary_host_bytes = positive_rows * 2 * ROW_BYTES * LAYERS
        dedup_host_bytes = positive_ids.numel() * 2 * ROW_BYTES * LAYERS
        measurements = []
        for condition in cache_conditions(args.cache_condition):
            timings = measure_routes(
                arena,
                dedup_function,
                ids,
                row_map,
                ordinary_outputs,
                dedup_outputs,
                stream,
                args.warmup,
                args.samples,
                condition,
                sweep,
            )
            routes = {}
            for route, timing in timings.items():
                host_bytes = (
                    ordinary_host_bytes if route == "ordinary" else dedup_host_bytes
                )
                median_seconds = timing["median"] / 1000
                routes[route] = {
                    "sum_12_layers_ms": timing,
                    "per_layer_ms": {
                        key: value / LAYERS
                        for key, value in timing.items()
                        if key != "samples"
                    }
                    | {"samples": [value / LAYERS for value in timing["samples"]]},
                    "calculated_payload": {
                        "algorithmic_host_read_bytes": host_bytes,
                        "algorithmic_host_read_gib_s": host_bytes
                        / median_seconds
                        / 1024**3,
                        "logical_output_bytes": logical_output_bytes,
                        "physical_pcie_bandwidth_claim": False,
                    },
                }
            measurements.append(
                {
                    "cache_condition": condition,
                    "routes": routes,
                    "dedup_over_ordinary_median_ratio": routes["dedup"][
                        "sum_12_layers_ms"
                    ]["median"]
                    / routes["ordinary"]["sum_12_layers_ms"]["median"],
                }
            )
        return {
            "concurrency": concurrency,
            "status": "ok",
            "correctness": "ordinary and dedup match a frozen raw-byte oracle for all 12 layers",
            "shape": {
                "requests": concurrency,
                "verify_queries_per_request": VERIFY_QUERIES,
                "rows_per_query": ROW_STRIDE,
                "valid_rows_per_query": VALID_ROWS,
                "negative_padding_rows_per_query": PADDING_ROWS,
                "gather_rows": ids.numel(),
                "unique_positive_slots": positive_ids.numel(),
                "heads": HEADS,
                "head_dim": HEAD_DIM,
                "dtype": "torch.float8_e4m3fn",
                "layers_per_iteration": LAYERS,
            },
            "arena": {
                "slots": slots,
                "bytes": arena_bytes,
                "max_bytes": MAX_ARENA_BYTES,
                "mem_available_before_bytes": available,
                "host_reserve_bytes": HOST_RESERVE_BYTES,
                "row_map_bytes": slots * 4,
            },
            "seed": args.seed,
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
    }


def write_json_fresh(path: Path, payload: dict) -> Path:
    output = path.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    return output


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.output is not None and args.output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {args.output}")
    script = Path(__file__).resolve()
    repo = script.parents[2]
    library_path = args.library.expanduser().resolve(strict=True)
    host_kv = load_host_kv_module(repo, library_path)
    dedup_function = declare_dedup(host_kv.library())
    if not torch.cuda.is_available() or torch.cuda.get_device_capability(0) != (12, 0):
        raise RuntimeError("benchmark requires the authorized SM120 GPU")

    created = datetime.now(timezone.utc)
    source_paths = [
        script.with_name(name) for name in ("host_kv.cu", "host_kv.cpp", "host_kv.h")
    ]
    report = {
        "schema_version": 1,
        "created_utc": created.isoformat(),
        "claim_scope": (
            "same-library native ordinary-versus-ephemeral-dedup gather microbenchmark; "
            "not end-to-end and not a physical PCIe bandwidth measurement"
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
            "cache_condition": args.cache_condition,
            "sweep_mib": args.sweep_mib,
        },
        "cache_conditions": {
            "warm": "12-layer cycling without an explicit flush",
            "sweep": (
                f"best-effort same-stream {args.sweep_mib} MiB uint8 add before "
                "each timed route, excluded from CUDA event timing"
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
            "script_sha256": file_sha256(script),
            "source_sha256": {path.name: file_sha256(path) for path in source_paths},
        },
        "results": [],
    }
    for concurrency in args.concurrencies:
        result = run_case(args, host_kv, dedup_function, concurrency)
        report["results"].append(result)
        summary = {
            measurement["cache_condition"]: {
                route: details["sum_12_layers_ms"]["median"]
                for route, details in measurement["routes"].items()
            }
            for measurement in result["measurements"]
        }
        print(
            json.dumps({"concurrency": concurrency, "median_ms": summary}), flush=True
        )
        torch.cuda.empty_cache()

    if args.output is None:
        print(json.dumps(report, indent=2))
    else:
        print(f"wrote {write_json_fresh(args.output, report)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

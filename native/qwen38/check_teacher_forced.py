"""Collect aligned teacher-forced tail logprobs, not a speed benchmark.

Uses frozen public-code prompts and literal-loopback services only. Comparing
the same known token sequence avoids confusing divergent sampled completions
with a numerical regression. The default NLL allowance is fixed before runs;
this bounded diagnostic supplements, rather than replaces, task-quality tests.
"""

import argparse
import asyncio
import json
import math
from pathlib import Path
import statistics
import sys

import aiohttp

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "benchmark/qwen38_ple"))
from bench_native_acceptance import digest, flush_cache_when_idle, loopback, make_ids  # noqa: E402


def aligned_tail(response, ids, count):
    meta = response["meta_info"]
    if meta["prompt_tokens"] != len(ids) or meta.get("cached_tokens", 0) != 0:
        raise ValueError("teacher-forced run used wrong prompt length or cached prefix")
    rows = meta["input_token_logprobs"]
    if not count <= len(rows) <= count + 1:
        raise ValueError(f"expected {count} tail logprobs, received {len(rows)}")
    tail = rows[-count:]
    if [row[1] for row in tail] != ids[-count:]:
        raise ValueError("server logprob token IDs do not match the frozen suffix")
    missing = sum(row[0] is None for row in tail)
    if missing:
        raise ValueError(f"server returned {missing}/{count} null tail logprobs")
    values = [float(row[0]) for row in tail]
    if not all(math.isfinite(value) and value <= 1e-5 for value in values):
        raise ValueError("non-finite or positive teacher-forced log probability")
    return values


def paired_logprob_delta(actual, prior):
    if actual["input_sha256"] != prior["input_sha256"] or len(
        actual["logprobs"]
    ) != len(prior["logprobs"]):
        raise ValueError("candidate/reference workload mismatch")
    return max(abs(x - y) for x, y in zip(actual["logprobs"], prior["logprobs"]))


def selected_cases(manifest, count, offset=0):
    cases = manifest["cases"]
    if count < 1 or offset < 0 or offset + count > len(cases):
        raise ValueError("requested case range is outside the frozen manifest")
    return enumerate(cases[offset : offset + count], start=offset)


async def run(args):
    manifest = json.loads(args.manifest.read_text())
    expected = manifest.pop("content_sha256")
    if digest(manifest) != expected:
        raise ValueError("frozen prompt hash mismatch")
    if args.output.exists():
        raise FileExistsError(args.output)
    result = {
        "manifest_sha256": expected,
        "input_tokens": args.input_tokens,
        "tail_tokens": args.tail_tokens,
        "case_offset": args.case_offset,
        "max_nll_delta": args.max_nll_delta,
        "max_logprob_delta": args.max_logprob_delta,
        "records": [],
        "ok": False,
    }
    try:
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=300)
        ) as session:
            for index, case in selected_cases(manifest, args.cases, args.case_offset):
                ids = make_ids(case, args.input_tokens)
                await flush_cache_when_idle(session, args.endpoint, 30)
                payload = {
                    "input_ids": ids,
                    "stream": False,
                    "return_logprob": True,
                    # This server returns a leading None at the requested
                    # start, even for a suffix. Request its predecessor so all
                    # scored tail positions have a conditional probability.
                    "logprob_start_len": len(ids) - args.tail_tokens - 1,
                    "sampling_params": {
                        "temperature": 0,
                        "max_new_tokens": 1,
                        "ignore_eos": True,
                    },
                }
                async with session.post(
                    args.endpoint + "/generate", json=payload
                ) as reply:
                    reply.raise_for_status()
                    response = await reply.json()
                result["last_response_meta"] = response["meta_info"]
                values = aligned_tail(response, ids, args.tail_tokens)
                record = {
                    "case": index,
                    "input_sha256": digest(ids),
                    "logprobs": values,
                    "nll": -statistics.mean(values),
                    "meta_info": response["meta_info"],
                }
                result["records"].append(record)
                print(json.dumps({"case": index, "nll": record["nll"]}), flush=True)
        if args.reference:
            reference = json.loads(args.reference.read_text())
            if not reference["ok"] or len(reference["records"]) != args.cases:
                raise ValueError("reference run is incomplete")
            deltas = []
            token_deltas = []
            for actual, prior in zip(result["records"], reference["records"]):
                token_deltas.append(paired_logprob_delta(actual, prior))
                deltas.append(actual["nll"] - prior["nll"])
            result["paired_nll_deltas"] = deltas
            result["paired_max_logprob_deltas"] = token_deltas
            result["mean_nll_delta"] = statistics.mean(deltas)
            if (
                args.max_logprob_delta is not None
                and max(token_deltas) > args.max_logprob_delta
            ):
                raise ValueError("repeated-input logprob difference exceeds allowance")
            if result["mean_nll_delta"] > args.max_nll_delta:
                raise ValueError(
                    "teacher-forced mean NLL regression exceeds frozen allowance"
                )
        result["ok"] = len(result["records"]) == args.cases
    except Exception as error:
        result["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        with args.output.open("x") as handle:
            json.dump(result, handle, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--endpoint", type=loopback, default="http://127.0.0.1:30001")
    parser.add_argument("--input-tokens", type=int, default=32768)
    parser.add_argument("--tail-tokens", type=int, default=256)
    parser.add_argument("--cases", type=int, default=4)
    parser.add_argument("--case-offset", type=int, default=0)
    parser.add_argument("--max-nll-delta", type=float, default=0.05)
    parser.add_argument(
        "--max-logprob-delta",
        type=float,
        help="Optional symmetric per-token repeatability gate; requires --reference",
    )
    args = parser.parse_args()
    if not (
        1 <= args.cases <= 10
        and 0 <= args.case_offset <= 10 - args.cases
        and 1 <= args.tail_tokens < args.input_tokens
    ):
        parser.error("require cases within 0..9 and 1 <= tail tokens < input tokens")
    if args.max_logprob_delta is not None and (
        not args.reference or args.max_logprob_delta < 0
    ):
        parser.error("a non-negative max-logprob-delta requires a reference")
    asyncio.run(run(args))

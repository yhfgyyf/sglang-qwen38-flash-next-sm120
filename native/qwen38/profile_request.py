"""Collect a bounded server-side CPU/CUDA trace of frozen public-code prompts.

This diagnostic run is not a throughput measurement. The target service must
already be isolated and idle; only literal loopback endpoints are accepted.
"""

import argparse
import asyncio
import json
from pathlib import Path
import sys
import time

import aiohttp

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "benchmark/qwen38_ple"))
from bench_native_acceptance import (  # noqa: E402
    digest,
    flush_cache_when_idle,
    generate,
    loopback,
    make_ids,
    validate_manifest,
)


OUTPUT_TOKENS = 128
CONCURRENCY_CHOICES = (1, 4, 6, 8, 10)


async def run(args):
    manifest = json.loads(args.manifest.read_text())
    expected_hash = manifest.pop("content_sha256")
    if digest(manifest) != expected_hash:
        raise ValueError("frozen prompt manifest hash mismatch")
    cases = validate_manifest(manifest)
    if len(cases) < args.concurrency:
        raise ValueError("not enough independent frozen cases")
    inputs = [make_ids(case, args.input_tokens) for case in cases[: args.concurrency]]
    if len({tuple(input_ids) for input_ids in inputs}) != len(inputs):
        raise ValueError("selected frozen prompts are not distinct")
    args.output.mkdir(parents=True, exist_ok=False)
    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=300)
    ) as session:
        await flush_cache_when_idle(session, args.endpoint, 30)
        profile = {
            "output_dir": str(args.output.resolve()),
            "profile_id": args.profile_id,
            "activities": ["CPU", "GPU"],
            "num_steps": args.steps,
            "with_stack": False,
            "record_shapes": False,
        }
        async with session.post(
            args.endpoint + "/start_profile", json=profile
        ) as response:
            response.raise_for_status()
            print("profile_start=" + await response.text(), flush=True)
        base = time.perf_counter()
        records = await asyncio.gather(
            *(
                generate(
                    session,
                    args.endpoint,
                    input_ids,
                    OUTPUT_TOKENS,
                    (
                        "profile-request"
                        if args.concurrency == 1
                        else f"profile-request-{index}"
                    ),
                    base,
                )
                for index, input_ids in enumerate(inputs)
            )
        )
        if args.concurrency == 1:
            artifact = {"config": profile, "request": records[0]}
        else:
            artifact = {
                "config": {
                    **profile,
                    "workload": {
                        "concurrency": args.concurrency,
                        "input_tokens": args.input_tokens,
                        "output_tokens": OUTPUT_TOKENS,
                    },
                },
                "requests": records,
            }
        with (args.output / "request.json").open("x") as handle:
            json.dump(artifact, handle)
        summaries = [
            {
                key: record.get(key)
                for key in (
                    "id",
                    "ok",
                    "error",
                    "input_sha256",
                    "output_sha256",
                )
            }
            for record in records
        ]
        if args.concurrency == 1:
            summaries[0].pop("id")
            print(json.dumps(summaries[0]))
        else:
            print(json.dumps({"requests": summaries}))
        if not all(record["ok"] for record in records):
            raise RuntimeError("profile workload failed; inspect request.json")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile-id", required=True)
    parser.add_argument("--endpoint", type=loopback, default="http://127.0.0.1:30001")
    parser.add_argument("--input-tokens", type=int, default=32768)
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument(
        "--concurrency", type=int, choices=CONCURRENCY_CHOICES, default=1
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    asyncio.run(run(parse_args()))

"""Frozen-public-corpus, loopback-only mechanical throughput acceptance.

This does not establish model quality. Output is fixed by ignore_eos and exact
server token counts; normal-EOS coding/tool checks are a separate gate.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import ipaddress
import json
from pathlib import Path
import random
import re
import statistics
import time
from urllib.parse import urlparse

import aiohttp


MIN_MEASURED_WAVES = 3
FIXED_OUTPUT_TOKENS = 512
OBSERVED_METRICS = {
    "sglang:num_running_reqs",
    "sglang:num_queue_reqs",
    "sglang:num_retracted_reqs_total",
    "sglang:num_retracted_reqs",
    "sglang:token_usage",
    "sglang:cache_hit_rate",
    "sglang:spec_accept_length",
    "sglang:spec_accept_rate",
}
PRIORITY_LABEL = re.compile(r'(?:^|,)priority="([^"]*)"(?:,|$)')
PROMETHEUS_LABEL = re.compile(r'([a-zA-Z_][a-zA-Z0-9_]*)="((?:\\.|[^"])*)"')


def digest(value):
    return hashlib.sha256(json.dumps(value, separators=(",", ":")).encode()).hexdigest()


def loopback(endpoint):
    parsed = urlparse(endpoint)
    if (
        parsed.scheme != "http"
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
        or parsed.path not in ("", "/")
        or not parsed.hostname
        or not ipaddress.ip_address(parsed.hostname).is_loopback
    ):
        raise ValueError("endpoint must be a literal loopback http:// address")
    return endpoint.rstrip("/")


def prepare(args):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    root = Path(args.source).resolve()
    files = sorted((root / "python/sglang/srt").rglob("*.py"))
    rng = random.Random(args.seed)
    rng.shuffle(files)
    texts, sources, total_bytes = [], [], 0
    for path in files:
        raw = path.read_bytes()
        if len(raw) > 256000:
            continue
        text = raw.decode("utf-8")
        texts.append(f"\n# file: {path.relative_to(root)}\n{text}")
        sources.append(
            {
                "path": str(path.relative_to(root)),
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
        )
        total_bytes += len(raw)
        if total_bytes >= max(2500000, args.max_input * 10):
            break
    corpus = tokenizer.encode("\n".join(texts), add_special_tokens=False)
    if len(corpus) < args.max_input * 2:
        raise ValueError(
            "public code corpus is too small; do not repeat text to fill context"
        )
    marker = "UNIQUE_Q38_BODY_MARKER_47917"
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": marker}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    if rendered.count(marker) != 1:
        raise ValueError("cannot establish exact template boundary")
    before, after = rendered.split(marker)
    suffix = tokenizer.encode(
        "\nExplain the main control flow and likely edge cases in the supplied code.\n"
        + after,
        add_special_tokens=False,
    )
    cases = []
    for index in range(args.cases):
        tag = f"Case {rng.getrandbits(128):032x}. Independent code review.\n"
        prefix = tokenizer.encode(before + tag, add_special_tokens=False)
        start = rng.randrange(len(corpus) - args.max_input)
        cases.append(
            {
                "id": index,
                "prefix": prefix,
                "body": corpus[start : start + args.max_input],
                "suffix": suffix,
            }
        )
    manifest = {
        "version": 1,
        "kind": "public-repository-code-mechanical",
        "model": args.model,
        "seed": args.seed,
        "max_input": args.max_input,
        "sources": sources,
        "cases": cases,
    }
    manifest["content_sha256"] = digest(manifest)
    target = Path(args.manifest)
    with target.open("x") as file:
        json.dump(manifest, file, separators=(",", ":"))
    print(
        json.dumps(
            {
                "manifest": str(target),
                "sha256": manifest["content_sha256"],
                "cases": len(cases),
                "corpus_tokens": len(corpus),
            }
        )
    )


def make_ids(case, length):
    body_count = length - len(case["prefix"]) - len(case["suffix"])
    if not 0 < body_count <= len(case["body"]):
        raise ValueError("requested input length is outside the frozen manifest")
    return case["prefix"] + case["body"][:body_count] + case["suffix"]


def metric_values(text):
    values = {}
    for line in text.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        fields = line.split()
        if len(fields) < 2:
            raise ValueError(f"malformed Prometheus metric line: {line[:200]!r}")
        label, value = fields[0], fields[1]
        name = label.split("{", 1)[0]
        if name in OBSERVED_METRICS:
            values[label] = float(value)
    return values


def metric_main_total(values, metric_name):
    """Sum DP totals without priority or duplicated TP/PP breakdowns."""
    per_dp = {}
    for label, value in values.items():
        if label.split("{", 1)[0] != metric_name:
            continue
        labels = label.split("{", 1)[1].rsplit("}", 1)[0] if "{" in label else ""
        priority = PRIORITY_LABEL.search(labels)
        if priority is not None and priority.group(1) != "":
            continue
        parsed_labels = dict(PROMETHEUS_LABEL.findall(labels))
        dp_rank = parsed_labels.get("dp_rank", "single")
        per_dp[dp_rank] = max(per_dp.get(dp_rank, value), value)
    return sum(per_dp.values()) if per_dp else None


def safe_median(values):
    return statistics.median(values) if values else None


def validate_run_args(args):
    if args.waves < MIN_MEASURED_WAVES:
        raise ValueError(
            f"acceptance requires at least {MIN_MEASURED_WAVES} measured waves"
        )
    if args.warmup_waves < 1:
        raise ValueError("acceptance requires at least one warmup wave")
    normal_eos = getattr(args, "normal_eos", False)
    if normal_eos and args.waves < 5:
        raise ValueError("normal-EOS acceptance requires at least five measured waves")
    if normal_eos and not 1 <= args.output_tokens <= 4096:
        raise ValueError("normal-EOS output cap must be in 1..4096")
    if not normal_eos and args.output_tokens != FIXED_OUTPUT_TOKENS:
        raise ValueError(
            f"mechanical acceptance requires exactly {FIXED_OUTPUT_TOKENS} output tokens"
        )
    if args.input_tokens <= 0 or args.concurrency <= 0:
        raise ValueError("input_tokens and concurrency must be positive")


def validate_manifest(manifest):
    if manifest.get("version") != 1:
        raise ValueError("unsupported manifest version")
    if manifest.get("kind") != "public-repository-code-mechanical":
        raise ValueError("manifest is not the frozen public-code workload")
    cases = manifest.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("manifest has no frozen cases")
    case_ids = []
    for case in cases:
        if not isinstance(case, dict):
            raise ValueError("manifest case is not an object")
        case_ids.append(case.get("id"))
        for field in ("prefix", "body", "suffix"):
            token_ids = case.get(field)
            if not isinstance(token_ids, list) or not all(
                isinstance(token, int) and not isinstance(token, bool) and token >= 0
                for token in token_ids
            ):
                raise ValueError(f"manifest case {field} is not frozen token IDs")
    if len(set(case_ids)) != len(case_ids):
        raise ValueError("manifest case IDs are not unique")
    return cases


def metric_samples_for_wave(metric_samples, wave):
    return [
        sample
        for sample in metric_samples
        if wave["start_s"] <= sample["time_s"] <= wave["end_s"]
    ]


def build_summary(args, waves, metric_samples, postrun_health, run_error=None):
    normal_eos = getattr(args, "normal_eos", False)
    measured = [wave for wave in waves if not wave["warmup"]]
    requests = [request for wave in measured for request in wave["requests"]]
    successful_waves = [wave for wave in measured if wave["ok"]]
    successful_requests = [request for request in requests if request.get("ok")]
    expected_requests = args.waves * args.concurrency

    peaks = []
    measured_metric_samples = []
    for wave in measured:
        samples = metric_samples_for_wave(metric_samples, wave)
        measured_metric_samples.extend(samples)
        running = [
            value
            for sample in samples
            if "values" in sample
            for value in [
                metric_main_total(sample["values"], "sglang:num_running_reqs")
            ]
            if value is not None
        ]
        peaks.append(max(running) if running else None)
    observed_peaks = [peak for peak in peaks if peak is not None]
    active_concurrency_observed = len(peaks) == args.waves and all(
        peak is not None and peak >= args.concurrency for peak in peaks
    )
    exact_server_completions = len(requests) == expected_requests and all(
        request.get("ok")
        and (
            (request.get("meta", {}).get("finish_reason") or {}).get("type") == "stop"
            if normal_eos
            else request.get("output_tokens") == FIXED_OUTPUT_TOKENS
        )
        for request in requests
    )
    measured_waves_complete = (
        len(measured) == args.waves
        and len(successful_waves) == args.waves
        and len(requests) == expected_requests
    )
    health_ok = bool(postrun_health.get("ok"))
    observed_seconds = measured[-1]["end_s"] - measured[0]["start_s"] if measured else 0
    summary = {
        "ok": bool(
            measured_waves_complete
            and exact_server_completions
            and active_concurrency_observed
            and health_ok
            and run_error is None
        ),
        "acceptance_scope": "normal_eos_complete"
        if normal_eos
        else "mechanical_ignore_eos",
        "normal_eos_quality_checked": False,
        "fixed_output_tokens": None if normal_eos else FIXED_OUTPUT_TOKENS,
        "max_output_tokens": args.output_tokens,
        "waves": len(measured),
        "required_waves": args.waves,
        "warmup_waves_excluded": sum(wave["warmup"] for wave in waves),
        "measured_waves_complete": measured_waves_complete,
        "requests": len(requests),
        "expected_requests": expected_requests,
        "server_completed_requests": len(successful_requests),
        "exact_server_completions": exact_server_completions,
        "failed_requests": len(requests) - len(successful_requests),
        # Includes failure time and between-wave control overhead. Never
        # discard a failed interval from the overall completed-token rate.
        "observed_seconds": observed_seconds,
        "completed_output_tps": (
            sum(request["output_tokens"] for request in successful_requests)
            / observed_seconds
            if observed_seconds > 0
            else None
        ),
        "completed_request_tps": (
            len(successful_requests) / observed_seconds
            if observed_seconds > 0
            else None
        ),
        "concurrency": args.concurrency,
        "scheduler_peak_source": "sglang:num_running_reqs main totals in measured waves",
        "scheduler_peak_running": max(observed_peaks) if observed_peaks else None,
        "scheduler_peak_running_by_wave": peaks,
        "active_concurrency_observed": active_concurrency_observed,
        "output_tps_median": safe_median(
            [wave["output_tps"] for wave in successful_waves]
        ),
        "ttft_median_s": safe_median(
            [
                request["ttft_s"]
                for request in successful_requests
                if "ttft_s" in request
            ]
        ),
        "tpot_median_s": safe_median(
            [
                request["tpot_s"]
                for request in successful_requests
                if "tpot_s" in request
            ]
        ),
        "metric_errors": sum("error" in sample for sample in metric_samples),
        "measured_metric_errors": sum(
            "error" in sample for sample in measured_metric_samples
        ),
        "mtp_meta_records": sum(bool(request.get("mtp_meta")) for request in requests),
        "postrun_health": postrun_health,
        "run_error": run_error,
    }
    return summary


async def generate(
    session, endpoint, ids, output_tokens, rid, base, *, normal_eos=False
):
    payload = json.dumps(
        {
            "input_ids": ids,
            "sampling_params": {
                "temperature": 0,
                "max_new_tokens": output_tokens,
                "ignore_eos": not normal_eos,
            },
            "stream": True,
        },
        separators=(",", ":"),
    ).encode()
    record = {
        "id": rid,
        "input_sha256": digest(ids),
        "input_tokens": len(ids),
        "events": [],
        "ok": False,
    }
    previous = 0
    merged_meta = {}
    emitted_ids = []
    started = time.perf_counter()
    record["offered_s"] = started - base
    try:
        async with session.post(
            endpoint + "/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
            allow_redirects=False,
        ) as response:
            record["headers_s"] = time.perf_counter() - base
            if response.status != 200:
                raise RuntimeError(
                    f"HTTP {response.status}: {(await response.text())[:1500]}"
                )
            async for raw in response.content:
                line = raw.strip()
                if not line or line == b"data: [DONE]":
                    continue
                if not line.startswith(b"data: "):
                    raise ValueError("unexpected SSE framing")
                event = json.loads(line[6:])
                meta = event.get("meta_info") or {}
                if not isinstance(meta, dict):
                    raise ValueError("meta_info is not an object")
                merged_meta.update(meta)
                count = int(meta.get("completion_tokens", previous))
                if count < previous:
                    raise ValueError("nonmonotonic server token count")
                if count > output_tokens:
                    raise ValueError("server token count exceeds requested output")
                now = time.perf_counter()
                if count > previous:
                    output_ids = event.get("output_ids")
                    if not isinstance(output_ids, list) or not output_ids:
                        raise ValueError(
                            "server token count advanced without emitted output_ids"
                        )
                    new_tokens = count - previous
                    if len(output_ids) not in (new_tokens, count):
                        raise ValueError(
                            "emitted output_ids do not match incremental or cumulative "
                            "server token count"
                        )
                    if len(output_ids) == count:
                        if output_ids[:previous] != emitted_ids:
                            raise ValueError(
                                "server changed previously emitted output_ids"
                            )
                        emitted_ids = output_ids.copy()
                    else:
                        emitted_ids.extend(output_ids)
                    if previous == 0:
                        record["ttft_s"] = now - started
                        record["first_token_count"] = new_tokens
                    record["events"].append(
                        {
                            "time_s": now - base,
                            "tokens": new_tokens,
                            "completion_tokens": count,
                        }
                    )
                previous = count
                if "text" in event:
                    record["text"] = event["text"]
            record["meta"] = merged_meta
            record["output_ids"] = emitted_ids
            record["output_sha256"] = digest(emitted_ids)
            record["mtp_meta"] = {
                key: value
                for key, value in merged_meta.items()
                if key.startswith("spec_") or "mtp" in key.lower()
            }
            meta = merged_meta
            if previous == 0 or (not normal_eos and previous != output_tokens):
                raise ValueError(
                    f"output clipped/missing: {previous} != {output_tokens}"
                )
            if meta.get("cached_tokens", 0) != 0:
                raise ValueError("unexpected prefix-cache hit")
            if meta.get("num_retractions", 0) != 0:
                raise ValueError("request was retracted during capacity acceptance")
            if meta.get("prompt_tokens") != len(ids):
                raise ValueError("prompt token count mismatch")
            finish = (meta.get("finish_reason") or {}).get("type")
            if normal_eos and finish != "stop":
                raise ValueError(
                    "normal-EOS request did not reach EOS/stop before its cap"
                )
            if not normal_eos and finish != "length":
                raise ValueError("fixed-output probe did not finish by length")
            record["ok"] = True
    except Exception as error:
        record["error"] = repr(error)
    record["completed_s"] = time.perf_counter() - base
    record["e2e_s"] = time.perf_counter() - started
    record["output_tokens"] = previous
    if "ttft_s" in record and previous > 1:
        record["tpot_s"] = (record["e2e_s"] - record["ttft_s"]) / (previous - 1)
    return record


async def fetch_metrics(session, endpoint):
    async with session.get(endpoint + "/metrics", allow_redirects=False) as response:
        text = await response.text()
        if response.status != 200:
            raise RuntimeError(f"metrics HTTP {response.status}: {text[:500]}")
        return metric_values(text)


async def wait_for_idle(session, endpoint, timeout):
    deadline = time.perf_counter() + min(timeout, 30.0)
    while True:
        values = await fetch_metrics(session, endpoint)
        running = metric_main_total(values, "sglang:num_running_reqs")
        queued = metric_main_total(values, "sglang:num_queue_reqs")
        if running is None or queued is None:
            raise RuntimeError("scheduler idle metrics are missing main total series")
        if running == 0 and queued == 0:
            return
        if time.perf_counter() >= deadline:
            raise TimeoutError(
                f"scheduler did not become idle: running={running}, queued={queued}"
            )
        await asyncio.sleep(0.05)


async def flush_cache_when_idle(session, endpoint, timeout):
    await wait_for_idle(session, endpoint, timeout)
    async with session.get(
        endpoint + "/flush_cache", allow_redirects=False
    ) as response:
        body = await response.read()
        if response.status != 200:
            raise RuntimeError(f"flush_cache HTTP {response.status}: {body[:500]!r}")


async def postrun_health_check(session, endpoint):
    try:
        async with session.get(endpoint + "/health", allow_redirects=False) as response:
            body = await response.read()
            if response.status != 200:
                raise RuntimeError(f"health HTTP {response.status}: {body[:500]!r}")
            return {"ok": True, "status": response.status}
    except Exception as error:
        return {"ok": False, "error": repr(error)}


async def run(args):
    validate_run_args(args)
    endpoint = loopback(args.endpoint)
    manifest = json.loads(Path(args.manifest).read_text())
    recorded_hash = manifest.pop("content_sha256", None)
    if not isinstance(recorded_hash, str):
        raise ValueError("manifest content_sha256 is missing")
    if digest(manifest) != recorded_hash:
        raise ValueError("manifest digest mismatch")
    cases = validate_manifest(manifest)
    if args.concurrency > len(cases):
        raise ValueError("not enough independent frozen cases")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    base = time.perf_counter()
    metric_samples = []
    stop = asyncio.Event()
    timeout = aiohttp.ClientTimeout(total=args.timeout)
    async with aiohttp.ClientSession(
        timeout=timeout, trust_env=False, connector=aiohttp.TCPConnector(limit=0)
    ) as session:
        async with session.get(
            endpoint + "/get_server_info", allow_redirects=False
        ) as response:
            if response.status != 200:
                raise RuntimeError(f"get_server_info HTTP {response.status}")
            info = await response.json()
        planned_hashes = []
        for wave in range(-args.warmup_waves, args.waves):
            selected = [
                cases[(max(0, wave) * args.concurrency + index) % len(cases)]
                for index in range(args.concurrency)
            ]
            planned_hashes.append(
                {
                    "wave": wave,
                    "warmup": wave < 0,
                    "input_sha256": [
                        digest(make_ids(case, args.input_tokens)) for case in selected
                    ],
                }
            )
        run_manifest = {
            "args": vars(args),
            "manifest_sha256": recorded_hash,
            "workload_sha256": digest(planned_hashes),
            "wave_input_sha256": planned_hashes,
            "server_info": info,
            "acceptance_scope": "normal_eos_complete"
            if args.normal_eos
            else "mechanical_ignore_eos",
            "normal_eos_quality_checked": False,
        }
        (output / "run.json").write_text(json.dumps(run_manifest, indent=2))

        async def sample_metrics():
            while not stop.is_set():
                try:
                    values = await fetch_metrics(session, endpoint)
                    metric_samples.append(
                        {"time_s": time.perf_counter() - base, "values": values}
                    )
                except Exception as error:
                    metric_samples.append(
                        {"time_s": time.perf_counter() - base, "error": repr(error)}
                    )
                try:
                    await asyncio.wait_for(stop.wait(), timeout=0.1)
                except TimeoutError:
                    pass

        sampler = asyncio.create_task(sample_metrics())
        waves = []
        run_error = None
        try:
            for wave in range(-args.warmup_waves, args.waves):
                await flush_cache_when_idle(session, endpoint, args.timeout)
                selected = [
                    cases[(max(0, wave) * args.concurrency + i) % len(cases)]
                    for i in range(args.concurrency)
                ]
                ids = [make_ids(case, args.input_tokens) for case in selected]
                if len({tuple(value[:64]) for value in ids}) != len(ids):
                    raise ValueError("requests share a complete first cache page")
                input_hashes = [digest(value) for value in ids]
                expected_hashes = planned_hashes[wave + args.warmup_waves][
                    "input_sha256"
                ]
                if input_hashes != expected_hashes:
                    raise ValueError("frozen input hashes changed during the run")
                wave_start = time.perf_counter() - base
                records = await asyncio.gather(
                    *[
                        generate(
                            session,
                            endpoint,
                            value,
                            args.output_tokens,
                            f"{wave}-{index}",
                            base,
                            normal_eos=args.normal_eos,
                        )
                        for index, value in enumerate(ids)
                    ]
                )
                wave_end = time.perf_counter() - base
                result = {
                    "wave": wave,
                    "warmup": wave < 0,
                    "start_s": wave_start,
                    "end_s": wave_end,
                    "ok": all(record["ok"] for record in records),
                    "requests": records,
                    "idle_before_flush": True,
                    "input_sha256": input_hashes,
                    "output_tps": sum(
                        record["output_tokens"] for record in records if record["ok"]
                    )
                    / (wave_end - wave_start),
                }
                waves.append(result)
                with (output / "waves.jsonl").open("a") as file:
                    file.write(json.dumps(result, ensure_ascii=False) + "\n")
                print(
                    json.dumps(
                        {
                            key: value
                            for key, value in result.items()
                            if key != "requests"
                        }
                    ),
                    flush=True,
                )
                if not result["ok"]:
                    break
        except Exception as error:
            run_error = repr(error)
        finally:
            stop.set()
            await sampler
            (output / "metrics.json").write_text(json.dumps(metric_samples))
        postrun_health = await postrun_health_check(session, endpoint)
        summary = build_summary(
            args, waves, metric_samples, postrun_health, run_error=run_error
        )
        (output / "summary.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary), flush=True)
        if not summary["ok"]:
            raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--model", required=True)
    prep.add_argument("--source", required=True)
    prep.add_argument("--manifest", required=True)
    prep.add_argument("--max-input", type=int, default=131072)
    prep.add_argument("--cases", type=int, default=10)
    prep.add_argument("--seed", type=int, default=20261007)
    bench = sub.add_parser("run")
    bench.add_argument("--endpoint", default="http://127.0.0.1:30001")
    bench.add_argument("--manifest", required=True)
    bench.add_argument("--output", required=True)
    bench.add_argument("--input-tokens", type=int, default=32768)
    bench.add_argument("--output-tokens", type=int, default=512)
    bench.add_argument(
        "--normal-eos",
        action="store_true",
        help="Measure natural EOS/stop; output-tokens is a cap, not a forced length",
    )
    bench.add_argument("--concurrency", type=int, choices=(1, 4, 6, 8, 10), default=1)
    bench.add_argument("--waves", type=int, default=3)
    bench.add_argument("--warmup-waves", type=int, default=1)
    bench.add_argument("--timeout", type=float, default=900)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args)
    else:
        asyncio.run(run(args))


if __name__ == "__main__":
    main()

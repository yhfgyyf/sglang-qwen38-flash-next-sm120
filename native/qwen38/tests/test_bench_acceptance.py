import argparse
import importlib.util
import json
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[3]
CLIENT_PATH = ROOT / "benchmark/qwen38_ple/bench_native_acceptance.py"
SPEC = importlib.util.spec_from_file_location("qwen38_bench_acceptance", CLIENT_PATH)
CLIENT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = CLIENT
SPEC.loader.exec_module(CLIENT)


class FakeContent:
    def __init__(self, lines):
        self.lines = lines

    def __aiter__(self):
        return self._iterate()

    async def _iterate(self):
        for line in self.lines:
            yield line


class FakeResponse:
    def __init__(self, events, status=200):
        self.status = status
        self.content = FakeContent(
            [b"data: " + json.dumps(event).encode() + b"\n" for event in events]
            + [b"data: [DONE]\n"]
        )

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return False

    async def text(self):
        return "fake error"


class FakeSession:
    def __init__(self, events):
        self.events = events
        self.post_calls = []

    def post(self, *args, **kwargs):
        self.post_calls.append((args, kwargs))
        return FakeResponse(self.events)


class FakeGetResponse:
    def __init__(self, text="", status=200):
        self._text = text
        self.status = status

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return False

    def raise_for_status(self):
        if self.status >= 400:
            raise RuntimeError(f"HTTP {self.status}")

    async def text(self):
        return self._text

    async def read(self):
        return self._text.encode()


class IdleSession:
    def __init__(self):
        self.calls = []
        self.metric_count = 0

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        if url.endswith("/metrics"):
            running = 1 if self.metric_count == 0 else 0
            self.metric_count += 1
            return FakeGetResponse(
                f'sglang:num_running_reqs{{priority=""}} {running}\n'
                'sglang:num_queue_reqs{priority=""} 0\n'
            )
        if url.endswith("/flush_cache"):
            return FakeGetResponse("flushed")
        raise AssertionError(url)


class HealthSession:
    def __init__(self, status):
        self.status = status

    def get(self, url, **kwargs):
        assert url.endswith("/health")
        assert kwargs["allow_redirects"] is False
        return FakeGetResponse("redirect" if self.status == 302 else "ok", self.status)


def request_record(ok=True, ttft=0.1, tpot=0.01):
    record = {"ok": ok, "output_tokens": 512 if ok else 0}
    if ttft is not None:
        record["ttft_s"] = ttft
    if tpot is not None:
        record["tpot_s"] = tpot
    return record


def args(**overrides):
    values = {
        "concurrency": 4,
        "waves": 3,
        "warmup_waves": 1,
        "output_tokens": 512,
        "input_tokens": 32768,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


class BenchmarkAcceptanceTest(unittest.IsolatedAsyncioTestCase):
    def test_loopback_requires_literal_http_address(self):
        self.assertEqual(
            CLIENT.loopback("http://127.0.0.1:30001/"),
            "http://127.0.0.1:30001",
        )
        self.assertEqual(CLIENT.loopback("http://[::1]:30001"), "http://[::1]:30001")
        for endpoint in (
            "http://localhost:30001",
            "https://127.0.0.1:30001",
            "http://user@127.0.0.1:30001",
            "http://127.0.0.1:30001/generate",
            "http://127.0.0.1:30001/?redirect=http://example.com",
        ):
            with self.subTest(endpoint=endpoint), self.assertRaises(ValueError):
                CLIENT.loopback(endpoint)

    def test_metrics_use_only_main_total_priority_series(self):
        text = """
# HELP ignored help
sglang:num_running_reqs{dp_rank="0",priority=""} 3
sglang:num_running_reqs{dp_rank="0",tp_rank="1",priority=""} 3
sglang:num_running_reqs{dp_rank="0",priority="0"} 2
sglang:num_running_reqs{dp_rank="0",priority="1"} 1
sglang:num_running_reqs{dp_rank="1",priority=""} 4
sglang:num_queue_reqs{dp_rank="0",priority=""} 0
sglang:num_queue_reqs{dp_rank="0",priority="0"} 5
"""
        values = CLIENT.metric_values(text)
        self.assertEqual(CLIENT.metric_main_total(values, "sglang:num_running_reqs"), 7)
        self.assertEqual(CLIENT.metric_main_total(values, "sglang:num_queue_reqs"), 0)
        self.assertIsNone(CLIENT.metric_main_total(values, "sglang:missing"))

    def test_summary_excludes_warmup_and_handles_empty_latency_samples(self):
        waves = [
            {
                "wave": -1,
                "warmup": True,
                "start_s": 0.0,
                "end_s": 1.0,
                "ok": True,
                "output_tps": 9999,
                "requests": [request_record()],
            },
            {
                "wave": 0,
                "warmup": False,
                "start_s": 2.0,
                "end_s": 3.0,
                "ok": False,
                "output_tps": 0,
                "requests": [request_record(False, None, None)],
            },
        ]
        metrics = [
            {
                "time_s": 0.5,
                "values": {
                    'sglang:num_running_reqs{priority=""}': 99,
                },
            },
            {
                "time_s": 2.5,
                "values": {
                    'sglang:num_running_reqs{priority=""}': 1,
                    'sglang:num_running_reqs{priority="0"}': 1,
                },
            },
        ]
        summary = CLIENT.build_summary(args(), waves, metrics, {"ok": True})
        self.assertFalse(summary["ok"])
        self.assertEqual(summary["scheduler_peak_running"], 1)
        self.assertEqual(summary["scheduler_peak_running_by_wave"], [1])
        self.assertIsNone(summary["output_tps_median"])
        self.assertIsNone(summary["ttft_median_s"])
        self.assertIsNone(summary["tpot_median_s"])
        self.assertEqual(summary["acceptance_scope"], "mechanical_ignore_eos")
        self.assertFalse(summary["normal_eos_quality_checked"])

    def test_summary_requires_complete_measured_waves_and_server_peak(self):
        waves = []
        metrics = [
            {
                "time_s": 0.5,
                "values": {'sglang:num_running_reqs{priority=""}': 100},
            }
        ]
        for wave in range(3):
            start = 2.0 + wave * 2
            waves.append(
                {
                    "wave": wave,
                    "warmup": False,
                    "start_s": start,
                    "end_s": start + 1,
                    "ok": True,
                    "output_tps": 10 + wave * 10,
                    "requests": [request_record() for _ in range(4)],
                }
            )
            metrics.append(
                {
                    "time_s": start + 0.5,
                    "values": {
                        'sglang:num_running_reqs{priority=""}': 4,
                        'sglang:num_running_reqs{priority="0"}': 4,
                    },
                }
            )
        summary = CLIENT.build_summary(args(), waves, metrics, {"ok": True})
        self.assertTrue(summary["ok"])
        self.assertEqual(summary["output_tps_median"], 20)
        self.assertEqual(summary["scheduler_peak_running"], 4)
        self.assertEqual(summary["scheduler_peak_running_by_wave"], [4, 4, 4])
        self.assertTrue(summary["active_concurrency_observed"])
        self.assertEqual(summary["server_completed_requests"], 12)
        self.assertTrue(summary["exact_server_completions"])

        summary = CLIENT.build_summary(args(), waves, metrics, {"ok": False})
        self.assertFalse(summary["ok"])

    def test_acceptance_requires_three_measured_waves_and_exact_512(self):
        for invalid in (args(waves=2), args(output_tokens=511), args(warmup_waves=0)):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                CLIENT.validate_run_args(invalid)
        CLIENT.validate_run_args(args())

    def test_normal_eos_requires_five_waves_and_bounded_output_cap(self):
        CLIENT.validate_run_args(args(normal_eos=True, waves=5, output_tokens=2048))
        for invalid in (
            args(normal_eos=True, waves=4),
            args(normal_eos=True, waves=5, output_tokens=0),
            args(normal_eos=True, waves=5, output_tokens=4097),
        ):
            with self.assertRaises(ValueError):
                CLIENT.validate_run_args(invalid)

    async def test_normal_eos_accepts_early_stop_and_rejects_length_cap(self):
        event = {
            "output_ids": [101, 102],
            "meta_info": {
                "completion_tokens": 2,
                "prompt_tokens": 1,
                "cached_tokens": 0,
                "finish_reason": {"type": "stop", "matched": 102},
            },
        }
        session = FakeSession([event])
        record = await CLIENT.generate(
            session,
            "http://127.0.0.1:30001",
            [11],
            2048,
            "normal",
            0,
            normal_eos=True,
        )
        self.assertTrue(record["ok"], record.get("error"))
        self.assertEqual(record["output_tokens"], 2)
        sent = json.loads(session.post_calls[0][1]["data"])
        self.assertFalse(sent["sampling_params"]["ignore_eos"])
        event["meta_info"]["finish_reason"] = {"type": "length"}
        record = await CLIENT.generate(
            FakeSession([event]),
            "http://127.0.0.1:30001",
            [11],
            2048,
            "capped",
            0,
            normal_eos=True,
        )
        self.assertFalse(record["ok"])
        self.assertIn("did not reach EOS/stop", record["error"])

    def test_overall_completed_rate_retains_failed_interval(self):
        good = request_record()
        good.update(output_tokens=100, meta={"finish_reason": {"type": "stop"}})
        waves = [
            {
                "wave": 0,
                "warmup": False,
                "start_s": 1.0,
                "end_s": 2.0,
                "ok": True,
                "output_tps": 100.0,
                "requests": [good],
            },
            {
                "wave": 1,
                "warmup": False,
                "start_s": 3.0,
                "end_s": 5.0,
                "ok": False,
                "output_tps": 0.0,
                "requests": [request_record(False)],
            },
        ]
        summary = CLIENT.build_summary(
            args(normal_eos=True, waves=5, concurrency=1), waves, [], {"ok": True}
        )
        self.assertFalse(summary["ok"])
        self.assertEqual(summary["observed_seconds"], 4.0)
        self.assertEqual(summary["completed_output_tps"], 25.0)
        self.assertEqual(summary["completed_request_tps"], 0.25)
        self.assertEqual(summary["acceptance_scope"], "normal_eos_complete")
        self.assertIsNone(summary["fixed_output_tokens"])

    def test_manifest_validation_keeps_frozen_token_ids_and_hashes(self):
        case = {"id": 0, "prefix": [1, 2], "body": [3, 4, 5], "suffix": [6]}
        manifest = {
            "version": 1,
            "kind": "public-repository-code-mechanical",
            "cases": [case],
        }
        self.assertEqual(CLIENT.validate_manifest(manifest), [case])
        first = CLIENT.make_ids(case, 5)
        second = CLIENT.make_ids(case, 5)
        self.assertEqual(first, [1, 2, 3, 4, 6])
        self.assertEqual(CLIENT.digest(first), CLIENT.digest(second))

        manifest["cases"] = [case, dict(case)]
        with self.assertRaises(ValueError):
            CLIENT.validate_manifest(manifest)

    async def test_flush_waits_for_main_scheduler_totals_to_be_idle(self):
        session = IdleSession()
        await CLIENT.flush_cache_when_idle(session, "http://127.0.0.1:30001", timeout=1)
        self.assertEqual(
            [url.rsplit("/", 1)[-1] for url, _kwargs in session.calls],
            ["metrics", "metrics", "flush_cache"],
        )
        self.assertTrue(
            all(kwargs["allow_redirects"] is False for _url, kwargs in session.calls)
        )

    async def test_postrun_health_rejects_redirects(self):
        health = await CLIENT.postrun_health_check(
            HealthSession(302), "http://127.0.0.1:30001"
        )
        self.assertFalse(health["ok"])
        self.assertIn("302", health["error"])

    async def test_generate_uses_server_token_delta_for_ttft_and_retains_mtp_meta(self):
        ids = [11, 22, 33]
        events = [
            {"meta_info": {"completion_tokens": 0, "spec_verify_ct": 7}},
            {
                "text": "first batch",
                "output_ids": [101, 102],
                "meta_info": {
                    "completion_tokens": 2,
                    "spec_num_correct_drafts": 5,
                },
            },
            {
                "text": "done",
                "output_ids": list(range(2, 512)),
                "meta_info": {
                    "completion_tokens": 512,
                    "cached_tokens": 0,
                    "prompt_tokens": len(ids),
                    "finish_reason": {"type": "length"},
                },
            },
        ]
        session = FakeSession(events)
        record = await CLIENT.generate(
            session, "http://127.0.0.1:30001", ids, 512, "r0", 0
        )
        self.assertTrue(record["ok"], record.get("error"))
        self.assertEqual(record["first_token_count"], 2)
        self.assertEqual(record["events"][0]["tokens"], 2)
        self.assertEqual(record["output_tokens"], 512)
        self.assertEqual(record["meta"]["spec_verify_ct"], 7)
        self.assertEqual(record["meta"]["spec_num_correct_drafts"], 5)
        self.assertEqual(record["mtp_meta"]["spec_verify_ct"], 7)
        self.assertEqual(record["output_ids"], [101, 102] + list(range(2, 512)))
        self.assertEqual(record["output_sha256"], CLIENT.digest(record["output_ids"]))
        self.assertIs(session.post_calls[0][1]["allow_redirects"], False)

    async def test_cumulative_ids_cannot_rewrite_emitted_tokens(self):
        record = await CLIENT.generate(
            FakeSession(
                [
                    {"output_ids": [101], "meta_info": {"completion_tokens": 1}},
                    {"output_ids": [999, 102], "meta_info": {"completion_tokens": 2}},
                ]
            ),
            "http://127.0.0.1:30001",
            [11],
            2,
            "r0",
            0,
        )
        self.assertFalse(record["ok"])
        self.assertIn("changed previously emitted", record["error"])

    async def test_retracted_request_cannot_pass_capacity_acceptance(self):
        record = await CLIENT.generate(
            FakeSession(
                [
                    {
                        "output_ids": [101, 102],
                        "meta_info": {
                            "completion_tokens": 2,
                            "prompt_tokens": 1,
                            "finish_reason": {"type": "length"},
                            "num_retractions": 1,
                        },
                    }
                ]
            ),
            "http://127.0.0.1:30001",
            [11],
            2,
            "r0",
            0,
        )
        self.assertFalse(record["ok"])
        self.assertIn("retracted", record["error"])

    async def test_generate_rejects_nonmonotonic_server_count(self):
        ids = [11]
        events = [
            {
                "text": "a",
                "output_ids": [101, 102],
                "meta_info": {"completion_tokens": 2},
            },
            {
                "text": "b",
                "output_ids": [103],
                "meta_info": {"completion_tokens": 1},
            },
        ]
        record = await CLIENT.generate(
            FakeSession(events), "http://127.0.0.1:30001", ids, 512, "r0", 0
        )
        self.assertFalse(record["ok"])
        self.assertIn("nonmonotonic", record["error"])

    async def test_generate_does_not_treat_metadata_as_a_first_token(self):
        ids = [11]
        events = [{"meta_info": {"completion_tokens": 1}}]
        record = await CLIENT.generate(
            FakeSession(events), "http://127.0.0.1:30001", ids, 512, "r0", 0
        )
        self.assertFalse(record["ok"])
        self.assertNotIn("ttft_s", record)
        self.assertIn("without emitted output_ids", record["error"])


if __name__ == "__main__":
    unittest.main()

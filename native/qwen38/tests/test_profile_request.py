import argparse
import asyncio
import importlib.util
import json
from pathlib import Path

import pytest


MODULE_PATH = Path(__file__).resolve().parents[1] / "profile_request.py"
SPEC = importlib.util.spec_from_file_location("q38_profile_request", MODULE_PATH)
profile_request = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(profile_request)


class MockContent:
    def __init__(self, event):
        self._lines = [
            b"data: " + json.dumps(event).encode() + b"\n",
            b"data: [DONE]\n",
        ]

    def __aiter__(self):
        return self._iterate()

    async def _iterate(self):
        for line in self._lines:
            yield line


class MockResponse:
    def __init__(self, *, status=200, text="", event=None):
        self.status = status
        self._text = text
        self.content = MockContent(event) if event is not None else None

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


class GenerateResponse(MockResponse):
    def __init__(self, session, index, input_ids, status):
        event = {
            "output_ids": [index * 1000 + token for token in range(128)],
            "meta_info": {
                "completion_tokens": 128,
                "prompt_tokens": len(input_ids),
                "cached_tokens": 0,
                "finish_reason": {"type": "length"},
            },
        }
        super().__init__(status=status, text="failed generate", event=event)
        self._session = session

    async def __aenter__(self):
        self._session.inflight += 1
        self._session.max_inflight = max(
            self._session.max_inflight, self._session.inflight
        )
        if self._session.inflight == self._session.expected:
            self._session.release.set()
        await self._session.release.wait()
        return self

    async def __aexit__(self, *_args):
        self._session.inflight -= 1
        return False


class MockSession:
    def __init__(self, expected, failed_index=None):
        self.expected = expected
        self.failed_index = failed_index
        self.calls = []
        self.generated_inputs = []
        self.generated_payloads = []
        self.inflight = 0
        self.max_inflight = 0
        self.release = None

    async def __aenter__(self):
        self.release = asyncio.Event()
        return self

    async def __aexit__(self, *_args):
        return False

    def get(self, url, **kwargs):
        self.calls.append(("get", url, kwargs))
        if url.endswith("/metrics"):
            return MockResponse(
                text=(
                    'sglang:num_running_reqs{priority=""} 0\n'
                    'sglang:num_queue_reqs{priority=""} 0\n'
                )
            )
        if url.endswith("/flush_cache"):
            return MockResponse(text="flushed")
        raise AssertionError(url)

    def post(self, url, **kwargs):
        self.calls.append(("post", url, kwargs))
        if url.endswith("/start_profile"):
            return MockResponse(text="started")
        if url.endswith("/generate"):
            payload = json.loads(kwargs["data"])
            input_ids = payload["input_ids"]
            index = len(self.generated_inputs)
            self.generated_inputs.append(input_ids)
            self.generated_payloads.append(payload)
            status = 500 if index == self.failed_index else 200
            return GenerateResponse(self, index, input_ids, status)
        raise AssertionError(url)


def write_manifest(path, case_count=10):
    manifest = {
        "version": 1,
        "kind": "public-repository-code-mechanical",
        "cases": [
            {
                "id": index,
                "prefix": [100 + index],
                "body": [200 + index] * 16,
                "suffix": [300 + index],
            }
            for index in range(case_count)
        ],
    }
    manifest["content_sha256"] = profile_request.digest(manifest)
    path.write_text(json.dumps(manifest))


def make_args(tmp_path, *, concurrency=1):
    manifest = tmp_path / "manifest.json"
    write_manifest(manifest)
    return argparse.Namespace(
        manifest=manifest,
        output=tmp_path / "output",
        profile_id="test-profile",
        endpoint="http://127.0.0.1:30001",
        input_tokens=8,
        steps=64,
        concurrency=concurrency,
    )


async def run_with_session(monkeypatch, args, session):
    monkeypatch.setattr(
        profile_request.aiohttp,
        "ClientSession",
        lambda **_kwargs: session,
    )
    await asyncio.wait_for(profile_request.run(args), timeout=1)


def test_cli_defaults_to_single_request_and_restricts_concurrency(tmp_path):
    common = [
        "--manifest",
        str(tmp_path / "manifest.json"),
        "--output",
        str(tmp_path / "output"),
        "--profile-id",
        "test-profile",
    ]
    assert profile_request.parse_args(common).concurrency == 1
    for concurrency in (1, 4, 6, 8, 10):
        assert (
            profile_request.parse_args(
                [*common, "--concurrency", str(concurrency)]
            ).concurrency
            == concurrency
        )
    with pytest.raises(SystemExit):
        profile_request.parse_args([*common, "--concurrency", "2"])


def test_default_single_request_artifact_remains_compatible(tmp_path, monkeypatch):
    args = make_args(tmp_path)
    session = MockSession(expected=1)

    asyncio.run(run_with_session(monkeypatch, args, session))

    artifact = json.loads((args.output / "request.json").read_text())
    assert set(artifact) == {"config", "request"}
    assert artifact["config"] == {
        "output_dir": str(args.output.resolve()),
        "profile_id": args.profile_id,
        "activities": ["CPU", "GPU"],
        "num_steps": args.steps,
        "with_stack": False,
        "record_shapes": False,
    }
    assert artifact["request"]["id"] == "profile-request"
    assert artifact["request"]["ok"] is True
    assert session.max_inflight == 1


@pytest.mark.parametrize("concurrency", [1, 4])
def test_stdout_summary_contains_the_recorded_output_hash(
    tmp_path, monkeypatch, capsys, concurrency
):
    args = make_args(tmp_path, concurrency=concurrency)
    session = MockSession(expected=concurrency)

    asyncio.run(run_with_session(monkeypatch, args, session))

    artifact = json.loads((args.output / "request.json").read_text())
    summary = json.loads(capsys.readouterr().out.splitlines()[-1])
    records = [artifact["request"]] if concurrency == 1 else artifact["requests"]
    summaries = [summary] if concurrency == 1 else summary["requests"]
    assert len(summaries) == len(records)
    for record, item in zip(records, summaries, strict=True):
        expected_hash = profile_request.digest(record["output_ids"])
        assert item["output_sha256"] == record["output_sha256"] == expected_hash


def test_concurrent_requests_enter_together_with_distinct_frozen_cases(
    tmp_path, monkeypatch
):
    args = make_args(tmp_path, concurrency=6)
    session = MockSession(expected=6)

    asyncio.run(run_with_session(monkeypatch, args, session))

    artifact = json.loads((args.output / "request.json").read_text())
    assert set(artifact) == {"config", "requests"}
    assert artifact["config"]["workload"] == {
        "concurrency": 6,
        "input_tokens": args.input_tokens,
        "output_tokens": 128,
    }
    assert session.max_inflight == 6
    assert len({tuple(ids) for ids in session.generated_inputs}) == 6
    assert [record["id"] for record in artifact["requests"]] == [
        f"profile-request-{index}" for index in range(6)
    ]
    assert len({record["input_sha256"] for record in artifact["requests"]}) == 6
    assert all(record["ok"] for record in artifact["requests"])
    assert all(record["output_tokens"] == 128 for record in artifact["requests"])
    assert all(record["output_sha256"] for record in artifact["requests"])
    assert all(
        payload["sampling_params"]["max_new_tokens"] == 128
        for payload in session.generated_payloads
    )
    call_paths = [url.rsplit("/", 1)[-1] for _method, url, _kwargs in session.calls]
    assert call_paths[:3] == ["metrics", "flush_cache", "start_profile"]
    assert call_paths[3:] == ["generate"] * 6
    start_profile = session.calls[2][2]["json"]
    assert "workload" not in start_profile


def test_concurrent_failure_is_written_before_run_fails(tmp_path, monkeypatch):
    args = make_args(tmp_path, concurrency=4)
    session = MockSession(expected=4, failed_index=2)

    with pytest.raises(RuntimeError, match="profile workload failed"):
        asyncio.run(run_with_session(monkeypatch, args, session))

    artifact = json.loads((args.output / "request.json").read_text())
    assert len(artifact["requests"]) == 4
    failed = artifact["requests"][2]
    assert failed["id"] == "profile-request-2"
    assert failed["input_sha256"]
    assert failed["ok"] is False
    assert "HTTP 500" in failed["error"]
    assert session.max_inflight == 4

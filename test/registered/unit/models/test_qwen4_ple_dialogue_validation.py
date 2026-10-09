import base64
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

SCRIPT = (
    Path(__file__).resolve().parents[4]
    / "benchmark/qwen38_ple/validate_coding_dialogues.py"
)
SPEC = importlib.util.spec_from_file_location("ple_dialogue_validation", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def event(delta=None, finish=None, response_id="completion-1"):
    return {
        "id": response_id,
        "choices": [{"index": 0, "delta": delta or {}, "finish_reason": finish}],
    }


class DialogueValidationTests(unittest.TestCase):
    def test_cli_accepts_supported_high_concurrency_profiles(self):
        for concurrency in (6, 8, 10):
            with self.subTest(concurrency=concurrency):
                argv = [
                    str(SCRIPT),
                    "--dataset",
                    "unused.jsonl",
                    "--output",
                    "unused-output",
                    "--concurrency",
                    str(concurrency),
                ]
                with (
                    patch("sys.argv", argv),
                    patch.object(
                        MODULE, "load_samples", side_effect=RuntimeError("parsed-only")
                    ),
                    self.assertRaisesRegex(RuntimeError, "parsed-only"),
                ):
                    MODULE.main()

    def test_sse_utf8_json_and_required_done(self):
        payload = event({"content": "中文路径/测试.py 🙂"}, "stop")
        lines = [
            b"data: " + json.dumps(payload, ensure_ascii=False).encode(),
            b"",
            b"data: [DONE]",
            b"",
        ]
        self.assertEqual(MODULE.parse_sse(lines), [payload])

    def test_sse_missing_done_or_bad_utf8_rejected(self):
        for lines in ([b"data: {}", b""], [b"data: \xff", b"", b"data: [DONE]"]):
            with self.subTest(lines=lines), self.assertRaises(MODULE.ProtocolError):
                MODULE.parse_sse(lines)

    def test_duplicate_done_or_data_after_done_rejected(self):
        for tail in (b"data: [DONE]", b"data: {}"):
            with self.subTest(tail=tail), self.assertRaises(MODULE.ProtocolError):
                MODULE.parse_sse([b"data: [DONE]", b"", tail, b""])

    def test_tool_json_fragments_assemble_without_early_parse(self):
        events = [
            event({"role": "assistant"}),
            event(
                {
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "call-1",
                            "type": "function",
                            "function": {
                                "name": "read_code_excerpt",
                                "arguments": '{"path":"中文',
                            },
                        }
                    ]
                }
            ),
            event(
                {
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "call-1",
                            "function": {
                                "name": "read_code_excerpt",
                                "arguments": '/测试.py","line":2}',
                            },
                        }
                    ]
                },
                "tool_calls",
            ),
            {"id": "completion-1", "choices": [], "usage": {"completion_tokens": 28}},
        ]
        result = MODULE.assemble_stream(events)
        self.assertEqual(result["finish_reason"], "tool_calls")
        self.assertEqual(
            result["message"]["tool_calls"][0]["function"]["name"], "read_code_excerpt"
        )
        self.assertEqual(
            json.loads(result["message"]["tool_calls"][0]["function"]["arguments"]),
            {"path": "中文/测试.py", "line": 2},
        )
        self.assertEqual(result["usage"]["completion_tokens"], 28)

    def test_conflicting_stream_identity_or_post_finish_delta_rejected(self):
        cases = [
            [event({"content": "a"}), event({"content": "b"}, "stop", "different")],
            [event({"content": "a"}, "stop"), event({"content": "b"})],
            [event({"content": "a"}, "stop"), event(finish="stop")],
            [event({"content": "a"})],
        ]
        for events in cases:
            with self.subTest(events=events), self.assertRaises(MODULE.ProtocolError):
                MODULE.assemble_stream(events)

    def test_stream_rejects_nonstring_id_and_multiple_choices(self):
        cases = [
            [event({"content": "normal"}, "stop", 123)],
            [event({"content": "normal"}, "stop", "")],
            [
                dict(
                    event(),
                    choices=[
                        event({"content": "a"})["choices"][0],
                        event({"content": "b"}, "stop")["choices"][0],
                    ],
                )
            ],
        ]
        for events in cases:
            with self.subTest(events=events), self.assertRaises(MODULE.ProtocolError):
                MODULE.assemble_stream(events)

    def test_conflicting_tool_identity_rejected(self):
        events = [
            event(
                {
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "a",
                            "function": {
                                "name": "read_code_excerpt",
                                "arguments": "{}",
                            },
                        }
                    ]
                }
            ),
            event(
                {
                    "tool_calls": [
                        {"index": 0, "id": "b", "function": {"arguments": ""}}
                    ]
                },
                "tool_calls",
            ),
        ]
        with self.assertRaises(MODULE.ProtocolError):
            MODULE.assemble_stream(events)

    def test_reasoning_and_visible_content_are_separate(self):
        result = MODULE.assemble_stream(
            [
                event({"reasoning_content": "reason"}),
                event({"content": "正常文本"}),
                event(finish="stop"),
            ]
        )
        self.assertEqual(result["message"]["content"], "正常文本")
        self.assertEqual(result["message"]["reasoning_content"], "reason")

    def test_unicode_hard_errors_and_suspicion_are_distinct(self):
        self.assertFalse(MODULE.scan_text('路径/测试.py 🙂\nprint("ok")')["hard"])
        for text in (
            "broken\ufffd",
            "nul\x00",
            "surrogate\ud800",
            "nonchar\uffff",
            "ctrl\x01",
        ):
            with self.subTest(text=repr(text)):
                self.assertTrue(MODULE.scan_text(text)["hard"])
        self.assertTrue(MODULE.scan_text("cafÃ©")["suspicions"])
        self.assertTrue(MODULE.scan_text("重复行\n" * 20)["suspicions"])
        self.assertIn(
            "parser_markup",
            MODULE.scan_text("<tool_call><function=read>")["suspicions"],
        )

    def test_tool_arguments_require_exact_types_keys_and_values(self):
        def message(arguments, name="read_code_excerpt"):
            return {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {"name": name, "arguments": json.dumps(arguments)},
                    }
                ],
            }

        expected = {"sample_id": "coding-0001", "line": 1, "path": "中文.py"}
        self.assertEqual(
            MODULE.validate_tool_call(message(expected), "read_code_excerpt", expected)[
                "id"
            ],
            "call-1",
        )
        invalid = [
            dict(expected, line=True),
            dict(expected, line="1"),
            dict(expected, extra=0),
            dict(expected, path="wrong"),
        ]
        for arguments in invalid:
            with (
                self.subTest(arguments=arguments),
                self.assertRaises(MODULE.ProtocolError),
            ):
                MODULE.validate_tool_call(
                    message(arguments), "read_code_excerpt", expected
                )
        with self.assertRaises(MODULE.ProtocolError):
            MODULE.validate_tool_call(
                message(expected, "run_shell"), "read_code_excerpt", expected
            )

    def test_recursive_types_and_nonfinite_json_are_rejected(self):
        self.assertFalse(MODULE.same_json_value({"a": [1]}, {"a": [True]}))
        self.assertFalse(MODULE.same_json_value({"a": 1}, {"a": 1.0}))
        self.assertTrue(
            MODULE.same_json_value({"a": ["中文", False]}, {"a": ["中文", False]})
        )
        for raw in ('{"a":NaN}', '{"a":1,"a":2}'):
            with self.subTest(raw=raw), self.assertRaises(MODULE.ProtocolError):
                MODULE.strict_json(raw)

    def test_fixture_result_is_not_disclosed_before_tool_call(self):
        sample = {
            "sample_id": "coding-0001",
            "messages": [
                {"role": "user", "content": "检查中文路径"},
                {"role": "tool", "content": "def add(a, b): return a + b"},
            ],
        }
        fixture = MODULE.make_fixture(sample)
        self.assertNotIn(fixture["result"]["verification_marker"], fixture["request"])
        self.assertIn("def add", fixture["result"]["excerpt"])
        self.assertEqual(fixture["arguments"]["path"], "fixtures/中文/coding-0001.py")
        self.assertIs(fixture["arguments"]["include_context"], True)

    def test_plan_has_100_unique_sources_and_choice_coverage(self):
        samples = [
            {"sample_id": f"coding-{i:04d}", "length_band": band}
            for i, band in enumerate(["short"] * 30 + ["medium"] * 40 + ["long"] * 30)
        ]
        plan = MODULE.make_plan(samples)
        self.assertEqual(len({p["sample_id"] for p in plan}), 100)
        self.assertEqual(sum(p["choice"] == "auto" for p in plan), 50)
        self.assertEqual(sum(p["choice"] == "named" for p in plan), 25)
        self.assertEqual(sum(p["choice"] == "required" for p in plan), 25)
        for band in ("short", "medium", "long"):
            selected = [p for p in plan if p["length_band"] == band]
            self.assertEqual({p["stream"] for p in selected}, {False, True})
            self.assertEqual(
                {p["choice"] for p in selected}, {"auto", "named", "required"}
            )

    def test_nonstream_contract_rejects_missing_id_and_extra_choices(self):
        good = {
            "id": "response-1",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "正常"},
                }
            ],
            "usage": {},
        }
        self.assertEqual(MODULE.assemble_nonstream(good)["message"]["content"], "正常")
        for invalid in (dict(good, id=None), dict(good, choices=good["choices"] * 2)):
            with self.assertRaises(MODULE.ProtocolError):
                MODULE.assemble_nonstream(invalid)

    def test_strict_json_rejects_float_overflow(self):
        for number in ("1e999", "-1e999", "NaN", "Infinity", "-Infinity"):
            with self.subTest(number=number), self.assertRaises(MODULE.ProtocolError):
                MODULE.strict_json('{"value":' + number + "}")
        self.assertEqual(MODULE.strict_json('{"value":1.25e2}'), {"value": 125.0})

    def test_invalid_nonstream_responses_preserve_failure_evidence(self):
        payload = (
            '{"id":"completion-1","choices":[{"index":0,"finish_reason":"stop",'
            '"message":{"role":"assistant","content":"normal"}}],'
            '"usage":{"total_tokens":NUMBER}}'
        )
        cases = [
            (b'{"content":"literal \\\\xff; malformed \xff"}', UnicodeDecodeError),
            (payload.replace("NUMBER", "1e999").encode(), MODULE.ProtocolError),
            (payload.replace("NUMBER", "-1e999").encode(), MODULE.ProtocolError),
        ]
        for raw, error_type in cases:
            with self.subTest(raw=raw), tempfile.TemporaryDirectory() as directory:
                response = MagicMock()
                response.status_code = 200
                response.content = raw
                response.__enter__.return_value = response
                session = MagicMock()
                session.__enter__.return_value = session
                session.post.return_value = response
                client = MODULE.LocalClient(
                    "http://127.0.0.1:30001", Path(directory), "default", 30
                )
                with patch("requests.Session", return_value=session):
                    with self.assertRaises(error_type):
                        client.request(
                            "coding-1",
                            "natural",
                            [{"role": "user", "content": "test"}],
                            "auto",
                            False,
                        )
                saved = json.loads(
                    (Path(directory) / "coding-1-natural.json").read_text()
                )
                self.assertEqual(saved["error"]["type"], error_type.__name__)
                if error_type is UnicodeDecodeError:
                    self.assertIn("raw_body_base64", saved)
                    self.assertEqual(base64.b64decode(saved["raw_body_base64"]), raw)
                else:
                    self.assertEqual(saved["raw_body"].encode(), raw)
                self.assertEqual(len(client.records), 1)
                session.post.assert_called_once()

    def test_local_endpoint_only_and_redirects_disabled(self):
        for endpoint in (
            "https://example.com",
            "http://localhost",
            "http://127.0.0.1@evil.test",
            "http://127.0.0.1:30001/elsewhere",
        ):
            with self.subTest(endpoint=endpoint), self.assertRaises(ValueError):
                MODULE.validate_endpoint(endpoint)
        self.assertEqual(
            MODULE.validate_endpoint("http://127.0.0.1:30001"), "http://127.0.0.1:30001"
        )

    def test_loop_chains_real_responses_and_exact_tool_id(self):
        sample = {
            "sample_id": "coding-0001",
            "length_band": "short",
            "messages": [
                {"role": "user", "content": "真实问题"},
                {"role": "tool", "content": "历史代码"},
            ],
        }
        fixture = MODULE.make_fixture(sample)
        natural = {"role": "assistant", "content": "已检查当前实现。"}
        call = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "actual-id",
                    "type": "function",
                    "function": {
                        "name": "read_code_excerpt",
                        "arguments": json.dumps(fixture["arguments"]),
                    },
                }
            ],
        }
        final = {
            "role": "assistant",
            "content": fixture["result"]["verification_marker"],
        }
        responses = [
            {"message": natural, "finish_reason": "stop"},
            {"message": call, "finish_reason": "tool_calls"},
            {"message": final, "finish_reason": "stop"},
        ]
        sender = Mock(side_effect=responses)
        result = MODULE.run_dialogue(sample, {"choice": "auto", "stream": True}, sender)
        self.assertFalse(result["errors"])
        self.assertEqual(sender.call_count, 3)
        tool_prompt = sender.call_args_list[1].args[1]
        final_prompt = sender.call_args_list[2].args[1]
        self.assertIn(natural, tool_prompt)
        self.assertEqual(final_prompt[-1]["tool_call_id"], "actual-id")
        self.assertEqual(final_prompt[-2], call)
        self.assertIn(
            fixture["result"]["verification_marker"], final_prompt[-1]["content"]
        )

    def test_zero_attempts_or_missing_stages_cannot_pass(self):
        plan = [{"sample_id": "coding-0001", "choice": "auto", "stream": True}]
        report = MODULE.summarize(plan, [], [], 1)
        self.assertFalse(report["automated_pass"])
        self.assertEqual(report["missing_requests"], 3)

    def test_loader_rejects_duplicate_content_from_distinct_sources(self):
        messages = [
            {"role": "user", "content": "problem"},
            {"role": "assistant", "content": "analysis"},
            {"role": "user", "content": "next question"},
        ]
        samples = [
            {
                "sample_id": f"coding-{i}",
                "source_sha256": str(i),
                "content_sha256": MODULE.digest(messages),
                "messages": messages,
                "length_band": "short",
            }
            for i in range(2)
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dataset.jsonl"
            path.write_text("\n".join(MODULE.canonical(row) for row in samples))
            with self.assertRaisesRegex(ValueError, "duplicate.*content"):
                MODULE.load_samples(path)

    def test_malformed_unicode_response_is_preserved_for_adjudication(self):
        payload = {
            "id": "completion-1",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "broken\ud800"},
                }
            ],
        }
        session = MagicMock()
        session.__enter__.return_value = session
        response = MagicMock(status_code=200, content=json.dumps(payload).encode())
        session.post.return_value.__enter__.return_value = response
        with (
            tempfile.TemporaryDirectory() as directory,
            patch("requests.Session", return_value=session),
        ):
            client = MODULE.LocalClient(
                "http://127.0.0.1:30001", Path(directory), "default", 10
            )
            result = client.request(
                "coding-1",
                "natural",
                [{"role": "user", "content": "test"}],
                "auto",
                False,
            )
            self.assertIn(
                "unpaired_surrogate",
                MODULE.scan_text(result["message"]["content"])["hard"],
            )
            saved = json.loads((Path(directory) / "coding-1-natural.json").read_text())
            self.assertEqual(saved["message"]["content"], "broken\ud800")
            self.assertEqual(saved["raw_body"], response.content.decode())
            self.assertFalse(session.trust_env)
            self.assertFalse(session.post.call_args.kwargs["allow_redirects"])


if __name__ == "__main__":
    unittest.main()

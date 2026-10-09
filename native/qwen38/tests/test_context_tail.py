"""CPU-only contracts for the opt-in Qwen3.8 exact-context tail."""

from __future__ import annotations

import contextlib
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from sglang.srt.managers.io_struct import GenerateReqInput
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.tokenizer_manager import TokenizerManager
from sglang.srt.managers.utils import (
    GenerationBatchResult,
    compute_num_reserved_tokens,
)
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.speculative.context_tail import (
    CONTEXT_TAIL_LENGTH,
    context_tail_trace_enabled,
    context_tail_trace_near_bound,
    select_context_tail_verify_width,
    validate_context_tail_model_config,
    validate_context_tail_server_args,
)
from sglang.srt.speculative.eagle_worker_v2 import (
    EagleDraftWorker,
    EAGLEWorkerV2,
    _trace_context_tail_positions,
)

_C = CONTEXT_TAIL_LENGTH
_INPUT = 261_632


def _server_args(**overrides):
    values = dict(
        speculative_algorithm="EAGLE",
        speculative_num_steps=3,
        speculative_num_draft_tokens=4,
        speculative_eagle_topk=1,
        max_speculative_num_draft_tokens=4,
        speculative_adaptive=False,
        speculative_use_rejection_sampling=False,
        enable_multi_layer_eagle=False,
        tp_size=1,
        pp_size=1,
        dp_size=1,
        enable_dp_attention=False,
        enable_two_batch_overlap=False,
        context_length=_C,
        kv_cache_dtype="fp8_e4m3",
        page_size=64,
        quantization="modelopt_fp4",
        model_path="qwen38-test",
        served_model_name="qwen38-test",
        enable_priority_scheduling=False,
        default_priority_value=0,
        return_hidden_states_mode=None,
        enable_return_hidden_states=False,
        enable_custom_logit_processor=False,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def _req(max_new_tokens, *, top_k=1):
    return SimpleNamespace(
        rid="tail-test",
        origin_input_ids=[0] * _INPUT,
        sampling_params=SimpleNamespace(
            max_new_tokens=max_new_tokens,
            min_new_tokens=0,
            top_k=top_k,
        ),
    )


def _tokenizer_manager(
    args, *, preferred_sampling_params=None, allow_auto_truncate=False
):
    manager = TokenizerManager.__new__(TokenizerManager)
    manager.server_args = args
    manager.model_config = SimpleNamespace(vocab_size=152_064)
    manager.context_len = _C
    manager.num_reserved_tokens = compute_num_reserved_tokens(args)
    manager.qwen38_context_tail_enabled = validate_context_tail_server_args(args)
    manager.allow_auto_truncate = allow_auto_truncate
    manager.validate_total_tokens = True
    manager.is_generation = True
    manager.preferred_sampling_params = preferred_sampling_params
    manager.sampling_params_class = SamplingParams
    manager.tokenizer = None
    return manager


def _generate_request(**sampling_params):
    return GenerateReqInput(input_ids=[0], sampling_params=sampling_params)


class ContextTailAdmissionTests(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(
            os.environ,
            {"QWEN38_CONTEXT_TAIL": "1", "QWEN38_NATIVE_EXECUTOR": "1"},
            clear=False,
        )
        self.env.start()

    def tearDown(self):
        self.env.stop()

    def test_shared_ingress_reservation_stays_conservative_for_rust(self):
        args = _server_args()
        self.assertTrue(validate_context_tail_server_args(args))
        # Rust ingress only receives this shared scalar and cannot inspect the
        # normalized per-request sampling mode, so exact-tail admission remains
        # a Python HTTP capability.
        self.assertEqual(compute_num_reserved_tokens(args), 4)

    def test_tokenizer_manager_init_validates_and_stores_tail_capability(self):
        model_config = SimpleNamespace(
            is_generation=True,
            context_len=_C,
            image_token_id=None,
            hf_text_config=SimpleNamespace(model_type="qwen4_exp_text"),
        )
        manager = TokenizerManager.__new__(TokenizerManager)
        manager.server_args = _server_args()
        manager.model_config_class = SimpleNamespace(
            from_server_args=lambda _: model_config
        )

        manager.init_model_config()

        self.assertTrue(manager.qwen38_context_tail_enabled)
        self.assertEqual(manager.num_reserved_tokens, 4)

    def test_explicit_temperature_zero_and_top_k_one_admit_exact_total(self):
        manager = _tokenizer_manager(_server_args())

        for sampling_params in (
            {"max_new_tokens": 512, "temperature": 0},
            {"max_new_tokens": 512, "temperature": 0.0000005},
            {"max_new_tokens": 512, "top_k": 1},
        ):
            with self.subTest(sampling_params=sampling_params):
                request = _generate_request(**sampling_params)
                manager._validate_one_request(request, [0] * _INPUT)
                self.assertEqual(request.sampling_params["max_new_tokens"], 512)

    def test_preferred_sampling_defaults_and_request_overrides_are_resolved(self):
        preferred_greedy = _tokenizer_manager(
            _server_args(), preferred_sampling_params={"temperature": 0}
        )
        request = _generate_request(max_new_tokens=512)
        preferred_greedy._validate_one_request(request, [0] * _INPUT)

        preferred_non_greedy = _tokenizer_manager(
            _server_args(), preferred_sampling_params={"top_k": 2}
        )
        with self.assertRaisesRegex(ValueError, "maximum context length"):
            preferred_non_greedy._validate_one_request(
                _generate_request(max_new_tokens=512), [0] * _INPUT
            )

        # Request parameters take precedence over the server preference.
        preferred_non_greedy._validate_one_request(
            _generate_request(max_new_tokens=512, top_k=1), [0] * _INPUT
        )
        with self.assertRaisesRegex(ValueError, "maximum context length"):
            preferred_greedy._validate_one_request(
                _generate_request(max_new_tokens=512, temperature=1, top_k=2),
                [0] * _INPUT,
            )

    def test_greedy_plus_one_rejects_and_auto_truncates(self):
        manager = _tokenizer_manager(_server_args())
        too_long = _generate_request(max_new_tokens=513, top_k=1)
        with self.assertRaisesRegex(ValueError, "maximum context length"):
            manager._validate_one_request(too_long, [0] * _INPUT)

        truncating = _tokenizer_manager(
            _server_args(), allow_auto_truncate=True
        )
        greedy = _generate_request(max_new_tokens=513, top_k=1)
        truncating._validate_one_request(greedy, [0] * _INPUT)
        self.assertEqual(greedy.sampling_params["max_new_tokens"], 512)

        non_greedy = _generate_request(max_new_tokens=512, top_k=2)
        truncating._validate_one_request(non_greedy, [0] * _INPUT)
        self.assertEqual(non_greedy.sampling_params["max_new_tokens"], 508)

    def test_scheduler_waives_only_logical_tail_guard_not_physical_pool_guard(self):
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.max_new_tokens_limit = None
        scheduler.max_req_len = _C - 1
        scheduler.max_total_num_tokens = 294_912
        scheduler.page_size = 64
        scheduler.qwen38_context_tail_enabled = True
        scheduler.model_config = SimpleNamespace(context_len=_C)

        with patch(
            "sglang.srt.managers.scheduler.get_parallel",
            return_value=SimpleNamespace(attn_dcp_size=1),
        ):
            exact = _req(512)
            scheduler.init_req_max_new_tokens(exact)
            self.assertEqual(exact.sampling_params.max_new_tokens, 512)

            scheduler.max_new_tokens_limit = 511
            env_limited = _req(512)
            scheduler.init_req_max_new_tokens(env_limited)
            self.assertEqual(env_limited.sampling_params.max_new_tokens, 511)
            scheduler.max_new_tokens_limit = None

            scheduler.max_total_num_tokens = 262_144
            constrained = _req(512)
            scheduler.init_req_max_new_tokens(constrained)
            self.assertEqual(constrained.sampling_params.max_new_tokens, 447)

            scheduler.max_total_num_tokens = 294_912
            unsupported_sampling = _req(512, top_k=2)
            scheduler.init_req_max_new_tokens(unsupported_sampling)
            self.assertEqual(unsupported_sampling.sampling_params.max_new_tokens, 510)

    def test_opt_in_rejects_unsupported_profiles_while_default_is_inert(self):
        cases = (
            ("tp_size", 2),
            ("dp_size", 2),
            ("enable_two_batch_overlap", True),
            ("speculative_adaptive", True),
            ("enable_multi_layer_eagle", True),
            ("speculative_use_rejection_sampling", True),
            ("speculative_eagle_topk", 2),
            ("speculative_num_steps", 2),
            ("speculative_num_draft_tokens", 3),
            ("context_length", _C + 1),
            ("kv_cache_dtype", "bf16"),
        )
        for field, value in cases:
            with (
                self.subTest(field=field),
                self.assertRaisesRegex(ValueError, "QWEN38_CONTEXT_TAIL"),
            ):
                validate_context_tail_server_args(_server_args(**{field: value}))

        model_config = SimpleNamespace(
            context_len=_C,
            hf_text_config=SimpleNamespace(model_type="qwen4_exp_text"),
        )
        self.assertTrue(validate_context_tail_model_config(model_config))
        model_config.hf_text_config.model_type = "other"
        with self.assertRaisesRegex(ValueError, "qwen4_exp_text"):
            validate_context_tail_model_config(model_config)

        with patch.dict(os.environ, {"QWEN38_CONTEXT_TAIL": "0"}, clear=False):
            self.assertFalse(validate_context_tail_server_args(_server_args(tp_size=8)))
            self.assertEqual(compute_num_reserved_tokens(_server_args()), 4)


class ContextTailRouteTests(unittest.TestCase):
    def test_thresholds_match_overlap_and_non_overlap_invariant(self):
        # Normal width N=4 is legal only when U+N <= C-1, with U=K+4
        # under overlap and U=K otherwise.
        self.assertEqual(select_context_tail_verify_width([_C - 9], _C, True, 4), 4)
        self.assertEqual(select_context_tail_verify_width([_C - 8], _C, True, 4), 1)
        self.assertEqual(select_context_tail_verify_width([_C - 5], _C, False, 4), 4)
        self.assertEqual(select_context_tail_verify_width([_C - 4], _C, False, 4), 1)

    def test_all_accept_lengths_keep_every_forward_position_below_context(self):
        # Model one scheduler-overlap pipeline slot. K is the CPU-committed
        # length and L is the device-visible next forward position. Processing
        # the prior result advances K to old L; accepting a tokens advances L.
        for overlap in (False, True):
            states = {(_C - 16, _C - 16)}
            visited = set()
            while states:
                k, device_pos = states.pop()
                if (k, device_pos) in visited:
                    continue
                visited.add((k, device_pos))
                width = select_context_tail_verify_width([k], _C, overlap, 4)
                self.assertLess(device_pos + width - 1, _C)
                if device_pos == _C - 1:
                    continue  # the one permitted surplus width-1 forward
                for accepted in range(1, width + 1):
                    if overlap:
                        states.add((device_pos, device_pos + accepted))
                    else:
                        states.add((device_pos + accepted, device_pos + accepted))

            self.assertTrue(any(device_pos == _C - 1 for _, device_pos in visited))

    def test_verify_uses_per_forward_width_and_next_batch_can_return_to_four(self):
        worker = EAGLEWorkerV2.__new__(EAGLEWorkerV2)
        worker._target_worker = object()
        worker.req_to_token_pool = object()
        worker.token_to_kv_pool_allocator = object()
        worker.plan_stream = None
        worker.plan_stream_ctx = contextlib.nullcontext()
        worker.topk = 1
        worker.speculative_num_draft_tokens = 4
        worker.device = "cpu"
        batch = SimpleNamespace(spec_info=SimpleNamespace(draft_token_num=1))

        with patch(
            "sglang.srt.speculative.eagle_worker_v2.run_eagle_verify",
            return_value="verified",
        ) as verify:
            self.assertEqual(worker.verify(batch), "verified")
        self.assertEqual(verify.call_args.kwargs["num_draft_tokens"], 1)

        self.assertEqual(select_context_tail_verify_width([_C - 8], _C, True, 4), 1)
        self.assertEqual(select_context_tail_verify_width([1024], _C, True, 4), 4)
        self.assertEqual(worker.speculative_num_draft_tokens, 4)

    def test_width_one_draft_extend_catches_up_via_eager_fallback(self):
        draft_worker = EagleDraftWorker.__new__(EagleDraftWorker)
        draft_worker.device = "cpu"
        draft_worker.topk = 1
        draft_worker.speculative_num_draft_tokens = 4
        draft_worker.plan_stream = None
        draft_worker.plan_stream_ctx = contextlib.nullcontext()
        draft_worker.seed_dsa_topk_from_draft_extend = False
        graph_checks = []
        graph = SimpleNamespace(
            can_run_graph=lambda forward_batch: graph_checks.append(forward_batch)
            or False
        )
        draft_worker.cuda_graph_runner_for_draft_extend = graph
        eager_forwards = []
        draft_worker.draft_runner = SimpleNamespace(
            canary_manager=None,
            forward=lambda forward_batch: eager_forwards.append(forward_batch)
            or SimpleNamespace(
                logits_output=SimpleNamespace(
                    next_token_logits=torch.tensor([[0.0, 2.0, 1.0]]),
                    hidden_states=torch.tensor([[3.0, 4.0]]),
                )
            ),
        )
        batch = SimpleNamespace(
            seq_lens=torch.tensor([_C - 4], dtype=torch.int64),
            sampling_info=None,
        )
        result = GenerationBatchResult(
            logits_output=SimpleNamespace(hidden_states=torch.zeros((1, 2))),
            next_token_ids=torch.tensor([7], dtype=torch.int32),
            accept_lens=torch.tensor([1], dtype=torch.int32),
            speculative_num_draft_tokens=1,
            next_draft_input=SimpleNamespace(),
        )
        prepared = SimpleNamespace(
            input_ids=torch.tensor([7], dtype=torch.int64),
            spec_info=SimpleNamespace(),
        )

        with (
            patch(
                "sglang.srt.speculative.eagle_worker_v2.prepare_for_draft_extend",
                return_value=prepared,
            ) as prepare,
            patch(
                "sglang.srt.speculative.eagle_worker_v2.get_spec",
                return_value=SimpleNamespace(speculative_use_rejection_sampling=False),
            ),
        ):
            draft_worker._draft_extend_for_decode(batch, result)

        self.assertTrue(graph_checks)
        self.assertEqual(eager_forwards, [prepared])
        draft_extend_input = prepare.call_args.args[0]
        self.assertEqual(draft_extend_input.num_tokens_per_req, 1)
        self.assertEqual(draft_extend_input.num_tokens_for_logprob_per_req, 1)
        self.assertEqual(prepare.call_args.args[3], 1)
        self.assertEqual(result.next_draft_input.topk_index.tolist(), [[1]])
        self.assertEqual(result.next_draft_input.topk_p.tolist(), [[1.0]])
        self.assertEqual(result.next_draft_input.hidden_states.tolist(), [[3.0, 4.0]])


class ContextTailTraceTests(unittest.TestCase):
    def test_trace_requires_tail_opt_in_and_near_boundary(self):
        with patch.dict(
            os.environ,
            {"QWEN38_CONTEXT_TAIL": "0", "QWEN38_CONTEXT_TAIL_TRACE": "1"},
            clear=False,
        ):
            self.assertFalse(context_tail_trace_enabled())

        with patch.dict(
            os.environ,
            {"QWEN38_CONTEXT_TAIL": "1", "QWEN38_CONTEXT_TAIL_TRACE": "1"},
            clear=False,
        ):
            self.assertTrue(context_tail_trace_enabled())
            self.assertFalse(context_tail_trace_near_bound([_C - 129], _C))
            self.assertTrue(context_tail_trace_near_bound([_C - 128], _C))

    def test_trace_off_does_not_read_positions(self):
        with (
            patch.dict(
                os.environ,
                {"QWEN38_CONTEXT_TAIL": "1", "QWEN38_CONTEXT_TAIL_TRACE": "0"},
                clear=False,
            ),
            patch(
                "sglang.srt.speculative.eagle_worker_v2.torch.aminmax"
            ) as aminmax,
        ):
            record = _trace_context_tail_positions(
                stage="verify",
                positions=torch.tensor([_C - 1]),
                reqs=[SimpleNamespace(kv_committed_len=_C - 1)],
                width=1,
                context_len=_C,
            )

        self.assertIsNone(record)
        aminmax.assert_not_called()

    def test_trace_below_threshold_does_not_read_positions(self):
        with (
            patch.dict(
                os.environ,
                {"QWEN38_CONTEXT_TAIL": "1", "QWEN38_CONTEXT_TAIL_TRACE": "1"},
                clear=False,
            ),
            patch(
                "sglang.srt.speculative.eagle_worker_v2.torch.aminmax"
            ) as aminmax,
        ):
            record = _trace_context_tail_positions(
                stage="verify",
                positions=torch.tensor([_C - 129]),
                reqs=[SimpleNamespace(kv_committed_len=_C - 129)],
                width=4,
                context_len=_C,
            )

        self.assertIsNone(record)
        aminmax.assert_not_called()

    def test_trace_records_normal_and_tail_width_position_bounds(self):
        with patch.dict(
            os.environ,
            {"QWEN38_CONTEXT_TAIL": "1", "QWEN38_CONTEXT_TAIL_TRACE": "1"},
            clear=False,
        ):
            normal = _trace_context_tail_positions(
                stage="verify",
                positions=torch.tensor([_C - 8, _C - 5]),
                reqs=[SimpleNamespace(kv_committed_len=_C - 8)],
                width=4,
                context_len=_C,
            )
            tail = _trace_context_tail_positions(
                stage="verify",
                positions=torch.tensor([_C - 1]),
                reqs=[SimpleNamespace(kv_committed_len=_C - 1)],
                width=1,
                context_len=_C,
            )

        self.assertEqual(normal["width"], 4)
        self.assertEqual(normal["position_min"], _C - 8)
        self.assertEqual(normal["position_max"], _C - 5)
        self.assertEqual(tail["width"], 1)
        self.assertEqual(tail["position_min"], _C - 1)
        self.assertEqual(tail["position_max"], _C - 1)

    def test_trace_rejects_out_of_bounds_actual_position(self):
        with patch.dict(
            os.environ,
            {"QWEN38_CONTEXT_TAIL": "1", "QWEN38_CONTEXT_TAIL_TRACE": "1"},
            clear=False,
        ):
            with self.assertRaisesRegex(RuntimeError, "position bounds"):
                _trace_context_tail_positions(
                    stage="draft_extend",
                    positions=torch.tensor([_C]),
                    reqs=[SimpleNamespace(kv_committed_len=_C - 1)],
                    width=1,
                    context_len=_C,
                )


if __name__ == "__main__":
    unittest.main()

"""Contract tests for the Qwen3.8 active-host KV pool integration.

``HostKVPoolCPUTests`` does not initialize CUDA and is safe while a model owns
GPU0.  Run ``HostKVPoolGPUTests`` only in an explicitly granted SM120 window.
"""

from __future__ import annotations

import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.arg_groups.speculative_hook import (
    _resolve_speculative_algorithm_alias,
)
from sglang.srt.mem_cache.qwen38_host_kv_pool import (
    Qwen38HostKVPool,
    host_kv_budget_bytes,
    host_kv_token_capacity,
    selected_host_slots,
    validate_host_kv_profile,
)
from sglang.srt.model_executor.qwen38_host_kv import required_bytes


_PAGE_SIZE = 64
_HEADS = 2
_HEAD_DIM = 256
_BYTES_PER_LAYER_TOKEN = 2 * _HEADS * _HEAD_DIM
_TOTAL_TARGET_DRAFT_LAYERS = 13


def _profile(*, draft=False, args_overrides=None, config_overrides=None, **overrides):
    args = dict(
        tp_size=1,
        pp_size=1,
        dp_size=1,
        speculative_algorithm="EAGLE",
        speculative_eagle_topk=1,
        speculative_num_draft_tokens=4,
        speculative_num_steps=3,
        max_total_tokens=147456,
        enable_hierarchical_cache=False,
        enable_hicache_storage=False,
        enable_memory_saver=False,
        enable_kv_cache_pool=False,
        disaggregation_mode="null",
    )
    args.update(args_overrides or {})
    config = dict(
        model_type="qwen4_exp_text",
        num_hidden_layers=1 if draft else 48,
        num_key_value_heads=_HEADS,
        head_dim=_HEAD_DIM,
        indexer_compress_ratio=4,
        indexer_kv_heads=1,
        indexer_head_dim=128,
    )
    config.update(config_overrides or {})
    values = dict(
        server_args=SimpleNamespace(**args),
        model_config=SimpleNamespace(hf_text_config=SimpleNamespace(**config)),
        is_draft_worker=draft,
        kv_cache_dtype=torch.float8_e4m3fn,
        use_mla_backend=False,
        page_size=_PAGE_SIZE,
        post_capture_kv_active=False,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


class HostKVPoolCPUTests(unittest.TestCase):
    def test_nextn_alias_resolves_before_exact_target_and_draft_profiles(self):
        with patch.dict(
            os.environ,
            {"QWEN38_HOST_KV_BYTES": "1048576", "QWEN38_NATIVE_EXECUTOR": "1"},
            clear=False,
        ):
            # Draft ModelConfig is observed in both forms: the private MTP
            # copy may expose its one layer, while the KVC call site may still
            # retain the target's original 48-layer text config.
            for draft, configured_layers in ((False, 48), (True, 1), (True, 48)):
                kvc = _profile(
                    draft=draft,
                    args_overrides={"speculative_algorithm": "NEXTN"},
                    config_overrides={"num_hidden_layers": configured_layers},
                )
                with self.assertRaisesRegex(ValueError, "NEXTN 3/1/4 profile"):
                    validate_host_kv_profile(kvc)
                kvc.server_args.speculative_algorithm = (
                    _resolve_speculative_algorithm_alias("NEXTN", None)
                )
                self.assertEqual(kvc.server_args.speculative_algorithm, "EAGLE")
                self.assertTrue(validate_host_kv_profile(kvc))
                self.assertEqual(
                    kvc.model_config.hf_text_config.num_hidden_layers,
                    configured_layers,
                )

    def test_profile_rejects_each_fixed_topology_and_nextn_contract_drift(self):
        cases = (
            ("tp2", {"args_overrides": {"tp_size": 2}}),
            ("pp2", {"args_overrides": {"pp_size": 2}}),
            ("dp2", {"args_overrides": {"dp_size": 2}}),
            ("steps2", {"args_overrides": {"speculative_num_steps": 2}}),
            ("topk2", {"args_overrides": {"speculative_eagle_topk": 2}}),
            (
                "draft_tokens3",
                {"args_overrides": {"speculative_num_draft_tokens": 3}},
            ),
            ("unbounded", {"args_overrides": {"max_total_tokens": None}}),
            ("target47", {"config_overrides": {"num_hidden_layers": 47}}),
            ("page32", {"page_size": 32}),
            ("bf16_kv", {"kv_cache_dtype": torch.bfloat16}),
            ("post_capture", {"post_capture_kv_active": True}),
        )
        with patch.dict(
            os.environ,
            {"QWEN38_HOST_KV_BYTES": "1048576", "QWEN38_NATIVE_EXECUTOR": "1"},
            clear=False,
        ):
            for name, changes in cases:
                with (
                    self.subTest(name=name),
                    self.assertRaisesRegex(ValueError, "native host KV requires"),
                ):
                    validate_host_kv_profile(_profile(**changes))
            with self.assertRaisesRegex(ValueError, "native host KV requires"):
                validate_host_kv_profile(
                    _profile(draft=True, config_overrides={"num_hidden_layers": 2})
                )
        with (
            patch.dict(
                os.environ,
                {"QWEN38_HOST_KV_BYTES": "1048576", "QWEN38_NATIVE_EXECUTOR": "0"},
                clear=False,
            ),
            self.assertRaisesRegex(ValueError, "native host KV requires"),
        ):
            validate_host_kv_profile(_profile())

    def test_profile_is_inert_without_explicit_budget(self):
        with patch.dict(
            os.environ,
            {"QWEN38_HOST_KV_BYTES": "0", "QWEN38_NATIVE_EXECUTOR": "0"},
            clear=False,
        ):
            self.assertFalse(
                validate_host_kv_profile(
                    _profile(args_overrides={"tp_size": 99}, page_size=1)
                )
            )

    def test_aggregate_budget_prices_target_draft_and_padding_exactly(self):
        capacity = 256
        slots = capacity + _PAGE_SIZE
        budget = slots * _TOTAL_TARGET_DRAFT_LAYERS * _BYTES_PER_LAYER_TOKEN
        with patch.dict(os.environ, {"QWEN38_HOST_KV_BYTES": str(budget)}, clear=False):
            self.assertEqual(host_kv_budget_bytes(), budget)
            self.assertEqual(host_kv_token_capacity(_PAGE_SIZE), capacity)
            target = required_bytes(
                12, slots, _HEADS, _HEAD_DIM, dtype=torch.float8_e4m3fn
            )
            draft = required_bytes(
                1, slots, _HEADS, _HEAD_DIM, dtype=torch.float8_e4m3fn
            )
            self.assertEqual(target + draft, budget)
            self.assertEqual(target, budget * 12 // _TOTAL_TARGET_DRAFT_LAYERS)
            self.assertEqual(draft, budget // _TOTAL_TARGET_DRAFT_LAYERS)

        # One aggregate layer-token less loses a complete 64-token capacity
        # page after the reserved padding page is priced.
        with patch.dict(
            os.environ,
            {"QWEN38_HOST_KV_BYTES": str(budget - _BYTES_PER_LAYER_TOKEN)},
            clear=False,
        ):
            self.assertEqual(host_kv_token_capacity(_PAGE_SIZE), capacity - _PAGE_SIZE)

    def test_budget_rejects_negative_and_padding_only_capacity(self):
        with (
            patch.dict(os.environ, {"QWEN38_HOST_KV_BYTES": "-1"}, clear=False),
            self.assertRaisesRegex(ValueError, "nonnegative"),
        ):
            host_kv_budget_bytes()
        padding_only = _PAGE_SIZE * _TOTAL_TARGET_DRAFT_LAYERS * _BYTES_PER_LAYER_TOKEN
        with (
            patch.dict(
                os.environ, {"QWEN38_HOST_KV_BYTES": str(padding_only)}, clear=False
            ),
            self.assertRaisesRegex(MemoryError, "reserved padding page"),
        ):
            host_kv_token_capacity(_PAGE_SIZE)


def _selected_reference(req_to_token, req_ids, positions, lengths, stride):
    result = []
    mapping = req_to_token.cpu().tolist()
    for request, candidates, length in zip(
        req_ids.cpu().tolist(), positions.cpu().tolist(), lengths.cpu().tolist()
    ):
        valid = [
            mapping[request][position]
            for position in candidates
            if 0 <= position < length
        ]
        result.extend(valid + [-1] * (stride - len(valid)))
    return result


class HostKVPoolGPUTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if torch.cuda.get_device_capability() != (12, 0):
            raise RuntimeError(
                "host KV pool GPU tests require the authorized SM120 GPU"
            )

    def test_selected_slots_multirequest_holes_padding_and_dynamic_capture(self):
        mapping = torch.stack(
            [torch.arange(16, dtype=torch.int32) + base for base in (100, 200, 300)]
        ).cuda()
        req_ids = torch.tensor([0, 2, 1], dtype=torch.int32, device="cuda")
        positions = torch.tensor(
            [[4, -1, 2, 12, 1], [0, 3, -2, 2, 8], [7, 1, 0, -1, 2]],
            dtype=torch.int64,
            device="cuda",
        )
        lengths = torch.tensor([6, 4, 3], dtype=torch.int32, device="cuda")
        # Integration pads each request's selected list into its own aligned
        # row; exercise the full padding write, not only a tight top-k buffer.
        stride = 64
        output = torch.full(
            (len(req_ids) * stride,), 777, dtype=torch.int32, device="cuda"
        )

        selected_host_slots(mapping, req_ids, positions, lengths, stride, output)
        torch.cuda.synchronize()
        self.assertEqual(
            output.cpu().tolist(),
            _selected_reference(mapping, req_ids, positions, lengths, stride),
        )

        graph = torch.cuda.CUDAGraph()
        stream = torch.cuda.Stream()
        torch.cuda.synchronize()
        with torch.cuda.graph(graph, stream=stream):
            selected_host_slots(mapping, req_ids, positions, lengths, stride, output)

        dynamic_cases = (
            (
                [2, 0, 1],
                [[2, 1, -1, 0, 7], [4, 0, 3, -1, 2], [1, 5, 0, -2, 1]],
                [3, 5, 2],
            ),
            (
                [1, 1, 2],
                [[-1, 0, 2, 1, 9], [3, 2, 1, 0, -1], [0, -1, 1, -2, 2]],
                [2, 4, 1],
            ),
        )
        for dynamic_req_ids, dynamic_positions, dynamic_lengths in dynamic_cases:
            req_ids.copy_(
                torch.tensor(dynamic_req_ids, dtype=torch.int32, device="cuda")
            )
            positions.copy_(
                torch.tensor(dynamic_positions, dtype=torch.int64, device="cuda")
            )
            lengths.copy_(
                torch.tensor(dynamic_lengths, dtype=torch.int32, device="cuda")
            )
            output.fill_(999)
            graph.replay()
            torch.cuda.synchronize()
            self.assertEqual(
                output.cpu().tolist(),
                _selected_reference(mapping, req_ids, positions, lengths, stride),
            )

    def test_target_pool_quantizes_gathers_and_moves_exact_fp8_bytes(self):
        self._check_target_pool_fp8_bytes(selected_dedup=False)

    def test_dedup_pool_preserves_ordinary_gather_and_move_bytes(self):
        self._check_target_pool_fp8_bytes(selected_dedup=True)

    def _check_target_pool_fp8_bytes(self, *, selected_dedup):
        size = 64
        slots = size + _PAGE_SIZE
        aggregate_budget = required_bytes(
            _TOTAL_TARGET_DRAFT_LAYERS,
            slots,
            _HEADS,
            _HEAD_DIM,
            dtype=torch.float8_e4m3fn,
        )
        with patch.dict(
            os.environ,
            {
                "QWEN38_HOST_KV_BYTES": str(aggregate_budget),
                "QWEN38_NATIVE_EXECUTOR": "1",
                "QWEN38_HOST_KV_DEDUP": "1" if selected_dedup else "0",
            },
            clear=False,
        ):
            pool = Qwen38HostKVPool(
                size=size,
                page_size=_PAGE_SIZE,
                dtype=torch.float8_e4m3fn,
                head_num=_HEADS,
                head_dim=_HEAD_DIM,
                layer_num=12,
                device="cuda:0",
            )
            locations = torch.tensor([1, 2, 3, 4], dtype=torch.int64, device="cuda")
            references = []
            try:
                self.assertEqual(pool.get_kv_size_bytes(), (0, 0))
                self.assertEqual(
                    pool.host_bytes,
                    required_bytes(
                        12,
                        slots,
                        _HEADS,
                        _HEAD_DIM,
                        dtype=torch.float8_e4m3fn,
                    ),
                )
                self.assertEqual(pool.arena.pinned_bytes, pool.host_bytes)
                self.assertEqual(
                    pool.arena.gpu_bytes, 24 + (slots * 4 if selected_dedup else 0)
                )
                self.assertEqual(
                    pool.get_kv_buffer_shape(),
                    (
                        torch.Size((slots, _HEADS, _HEAD_DIM)),
                        torch.Size((slots, _HEADS, _HEAD_DIM)),
                    ),
                )
                k_scale = 0.5
                v_scale = 0.25
                for layer_id in range(12):
                    base = (
                        torch.arange(
                            locations.numel() * _HEADS * _HEAD_DIM,
                            dtype=torch.float32,
                            device="cuda",
                        )
                        .add_(layer_id * 37)
                        .remainder_(193)
                        .sub_(96)
                        .view(locations.numel(), _HEADS, _HEAD_DIM)
                    )
                    keys = (base * k_scale).to(torch.bfloat16)
                    values = ((base + 11) * v_scale).to(torch.bfloat16)
                    original_keys = keys.clone()
                    original_values = values.clone()
                    expected_keys = (keys / k_scale).to(torch.float8_e4m3fn)
                    expected_values = (values / v_scale).to(torch.float8_e4m3fn)
                    references.append((expected_keys, expected_values))
                    pool.set_kv_buffer(
                        SimpleNamespace(layer_id=layer_id),
                        locations,
                        keys,
                        values,
                        k_scale=k_scale,
                        v_scale=v_scale,
                    )
                    self.assertTrue(torch.equal(keys, original_keys))
                    self.assertTrue(torch.equal(values, original_values))

                reverse = torch.flip(locations, dims=(0,)).to(torch.int32)
                out_k = torch.empty(
                    (locations.numel(), _HEADS, _HEAD_DIM),
                    dtype=torch.float8_e4m3fn,
                    device="cuda",
                )
                out_v = torch.empty_like(out_k)
                for layer_id in (0, 11):
                    pool.gather(layer_id, reverse, out_k, out_v)
                    pool.arena.check_errors()
                    self.assertTrue(
                        torch.equal(
                            out_k.view(torch.uint8),
                            torch.flip(references[layer_id][0], dims=(0,)).view(
                                torch.uint8
                            ),
                        )
                    )
                    self.assertTrue(
                        torch.equal(
                            out_v.view(torch.uint8),
                            torch.flip(references[layer_id][1], dims=(0,)).view(
                                torch.uint8
                            ),
                        )
                    )

                source = torch.tensor([1, 2, 3], dtype=torch.int32, device="cuda")
                target = torch.tensor([2, 3, 4], dtype=torch.int32, device="cuda")
                pool.move_kv_cache(target, source)
                all_ids = torch.tensor([1, 2, 3, 4], dtype=torch.int64, device="cuda")
                expected_rows = torch.tensor([0, 0, 1, 2], device="cuda")
                for layer_id in range(12):
                    pool.gather(layer_id, all_ids, out_k, out_v)
                    pool.arena.check_errors()
                    self.assertTrue(
                        torch.equal(
                            out_k.view(torch.uint8),
                            references[layer_id][0][expected_rows].view(torch.uint8),
                        )
                    )
                    self.assertTrue(
                        torch.equal(
                            out_v.view(torch.uint8),
                            references[layer_id][1][expected_rows].view(torch.uint8),
                        )
                    )
            finally:
                pool.close()


if __name__ == "__main__":
    unittest.main()

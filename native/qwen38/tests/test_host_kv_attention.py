"""Real-kernel parity tests for Qwen38's native host-KV attention path.

These tests require the explicitly authorized SM120 GPU window.  They compare
the mapped-host FP8 pool against SGLang's ordinary GPU FP8 pool byte-for-byte,
then compare the real TRT-LLM/Triton attention output against an independently
dequantized BF16-cache reference.  Attention math is never mocked.
"""

from __future__ import annotations

import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.attention.qwen_sparse_attn_backend import (
    QwenSparseAttnBackend,
    _resolve_trtllm_sparse_decode,
)
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
from sglang.srt.mem_cache.qwen38_host_kv_pool import Qwen38HostKVPool
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.model_executor.qwen38_host_kv import required_bytes


_SIZE = 128
_PAGE_SIZE = 64
_HEADS = 2
_Q_HEADS = 24
_HEAD_DIM = 256
_TOKEN_TOPK = 2048
_LAYER_ID = 7
_K_SCALE = 0.5
_V_SCALE = 0.25


class _FullPoolAdapter:
    """Minimal hybrid-pool surface with global-to-local layer translation."""

    def __init__(self, full_pool):
        self.full_kv_pool = full_pool
        self.full_attention_layer_id_mapping = {_LAYER_ID: 0}

    def set_kv_buffer(self, layer, loc, keys, values):
        local_layer = SimpleNamespace(
            layer_id=self.full_attention_layer_id_mapping[layer.layer_id]
        )
        self.full_kv_pool.set_kv_buffer(local_layer, loc, keys, values)

    def get_key_buffer(self, layer_id):
        return self.full_kv_pool.get_key_buffer(
            self.full_attention_layer_id_mapping[layer_id]
        )

    def get_value_buffer(self, layer_id):
        return self.full_kv_pool.get_value_buffer(
            self.full_attention_layer_id_mapping[layer_id]
        )


def _request_slot_table() -> torch.Tensor:
    """Two real requests share a prefix; request zero is graph padding."""

    table = torch.zeros((3, _SIZE), dtype=torch.int32, device="cuda")
    shared_prefix = torch.roll(
        torch.arange(1, _PAGE_SIZE + 1, dtype=torch.int32, device="cuda"), 11
    )
    table[1, :_PAGE_SIZE] = shared_prefix
    table[2, :_PAGE_SIZE] = shared_prefix
    table[1, _PAGE_SIZE:] = 65 + torch.arange(
        _SIZE - _PAGE_SIZE, dtype=torch.int32, device="cuda"
    ).remainder(31)
    table[2, _PAGE_SIZE:] = 96 + torch.arange(
        _SIZE - _PAGE_SIZE, dtype=torch.int32, device="cuda"
    ).remainder(32)
    return table


def _make_backend(pool, req_to_token):
    runner = SimpleNamespace(
        token_to_kv_pool=_FullPoolAdapter(pool),
        req_to_token_pool=SimpleNamespace(req_to_token=req_to_token),
        device=torch.device("cuda:0"),
        model_config=SimpleNamespace(context_len=req_to_token.shape[1]),
    )
    return QwenSparseAttnBackend(runner)


def _make_gpu_pool(dtype=torch.float8_e4m3fn):
    return MHATokenToKVPool(
        size=_SIZE,
        page_size=_PAGE_SIZE,
        dtype=dtype,
        head_num=_HEADS,
        head_dim=_HEAD_DIM,
        layer_num=1,
        device="cuda:0",
        enable_memory_saver=False,
        enable_alt_stream=False,
        kv_cache_layout="nhd",
    )


def _make_host_pool():
    # The production budget is aggregate target+draft storage.  Price all 13
    # full-attention layers so the one-layer test pool receives its exact share.
    aggregate_budget = required_bytes(
        13,
        _SIZE + _PAGE_SIZE,
        _HEADS,
        _HEAD_DIM,
        dtype=torch.float8_e4m3fn,
    )
    with patch.dict(
        os.environ,
        {
            "QWEN38_HOST_KV_BYTES": str(aggregate_budget),
            "QWEN38_NATIVE_EXECUTOR": "1",
        },
        clear=False,
    ):
        return Qwen38HostKVPool(
            size=_SIZE,
            page_size=_PAGE_SIZE,
            dtype=torch.float8_e4m3fn,
            head_num=_HEADS,
            head_dim=_HEAD_DIM,
            layer_num=1,
            device="cuda:0",
        )


def _seed_pool(pool, keys, values):
    zero = torch.zeros((1, _HEADS, _HEAD_DIM), dtype=keys.dtype, device=keys.device)
    _FullPoolAdapter(pool).set_kv_buffer(
        SimpleNamespace(layer_id=_LAYER_ID),
        torch.zeros(1, dtype=torch.int64, device="cuda"),
        zero,
        zero,
    )
    locations = torch.arange(1, _SIZE, dtype=torch.int64, device="cuda")
    _FullPoolAdapter(pool).set_kv_buffer(
        SimpleNamespace(layer_id=_LAYER_ID), locations, keys, values
    )


def _random_bf16(shape, scale=0.2):
    return (torch.randn(shape, dtype=torch.float32, device="cuda") * scale).to(
        torch.bfloat16
    )


def _fp8_and_dequantized(value, scale):
    fp8 = (value / scale).to(torch.float8_e4m3fn)
    dequantized = (fp8.to(torch.bfloat16) * scale).to(torch.bfloat16)
    return fp8, dequantized


def _decode_inputs():
    # The same request fans out to rows 0 and 3.  Row 1 is a safe graph-padding
    # request whose only physical slot is the pool's reserved zero slot.
    row_requests = torch.tensor([2, 0, 1, 2], dtype=torch.int32, device="cuda")
    sequence_lengths = torch.tensor([43, 1, 57, 61], dtype=torch.int32, device="cuda")
    selected = (
        [0, 2, 5, 10, 31, 42],
        [0],
        [1, 8, 13, 32, 56],
        [0, 3, 7, 23, 48, 60],
    )
    topk = torch.full(
        (len(selected), _TOKEN_TOPK), -1, dtype=torch.int32, device="cuda"
    )
    for row, positions in enumerate(selected):
        topk[row, : len(positions)] = torch.tensor(
            positions, dtype=torch.int32, device="cuda"
        )
    q = _random_bf16((len(selected), _Q_HEADS, _HEAD_DIM), scale=0.1)
    return q, topk, row_requests, sequence_lengths


def _extend_inputs(req_to_token):
    extend_lengths = [3, 5]
    sequence_lengths = [67, 69]
    q_rows = sum(extend_lengths)
    q = _random_bf16((q_rows, _Q_HEADS, _HEAD_DIM), scale=0.1)
    keys = _random_bf16((q_rows, _HEADS, _HEAD_DIM))
    values = _random_bf16((q_rows, _HEADS, _HEAD_DIM))

    # Every query selects a sparse subset of its visible full context.  The
    # tensor retains the model's 2048-token indexer width and uses -1 padding.
    topk = torch.full((q_rows, _TOKEN_TOPK), -1, dtype=torch.int32, device="cuda")
    row = 0
    for prefix, chunk in zip((64, 64), extend_lengths):
        for offset in range(chunk):
            visible = prefix + offset + 1
            positions = sorted({0, 1, 7, 31, 63, visible - 1})
            topk[row, : len(positions)] = torch.tensor(
                positions, dtype=torch.int32, device="cuda"
            )
            row += 1

    request_ids = torch.tensor([1, 2], dtype=torch.int32, device="cuda")
    current_slots = torch.cat(
        [
            req_to_token[request_ids[index], length - extend : length]
            for index, (length, extend) in enumerate(
                zip(sequence_lengths, extend_lengths)
            )
        ]
    ).to(torch.int64)
    forward_batch = SimpleNamespace(
        forward_mode=ForwardMode.EXTEND,
        out_cache_loc=current_slots,
        req_pool_indices=request_ids,
        extend_seq_lens=torch.tensor(extend_lengths, dtype=torch.int32, device="cuda"),
        extend_seq_lens_cpu=extend_lengths,
        seq_lens_cpu=sequence_lengths,
        seq_lens=torch.tensor(sequence_lengths, dtype=torch.int32, device="cuda"),
    )
    return q, keys, values, topk, forward_batch, sequence_lengths


class HostKVAttentionGPUTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if torch.cuda.get_device_capability() != (12, 0):
            raise RuntimeError(
                "host KV attention tests require the authorized SM120 GPU"
            )

    def setUp(self):
        torch.manual_seed(20261007)
        self.req_to_token = _request_slot_table()
        self.gpu_pool = _make_gpu_pool()
        self.host_pool = _make_host_pool()
        self.reference_pool = _make_gpu_pool(dtype=torch.bfloat16)
        keys = _random_bf16((_SIZE - 1, _HEADS, _HEAD_DIM))
        values = _random_bf16((_SIZE - 1, _HEADS, _HEAD_DIM))
        fp8_keys, reference_keys = _fp8_and_dequantized(keys, _K_SCALE)
        fp8_values, reference_values = _fp8_and_dequantized(values, _V_SCALE)
        _seed_pool(self.gpu_pool, fp8_keys, fp8_values)
        _seed_pool(self.host_pool, fp8_keys, fp8_values)
        _seed_pool(self.reference_pool, reference_keys, reference_values)
        self.host_pool.arena.check_errors()
        self.gpu_backend = _make_backend(self.gpu_pool, self.req_to_token)
        self.host_backend = _make_backend(self.host_pool, self.req_to_token)
        self.reference_backend = _make_backend(self.reference_pool, self.req_to_token)
        self.layer = SimpleNamespace(
            layer_id=_LAYER_ID,
            tp_q_head_num=_Q_HEADS,
            head_dim=_HEAD_DIM,
            scaling=_HEAD_DIM**-0.5,
            k_scale_float=_K_SCALE,
            v_scale_float=_V_SCALE,
        )

    def tearDown(self):
        self.host_pool.close()
        torch.cuda.synchronize()

    def test_trtllm_decode_fp8_placement_and_bf16_reference_with_graph_replay(self):
        trtllm_decode = _resolve_trtllm_sparse_decode()
        self.assertIsNotNone(trtllm_decode)
        q, topk, row_requests, sequence_lengths = _decode_inputs()
        forward_batch = SimpleNamespace(
            # Deliberately different: row_req_pool_indices must own the fanout.
            req_pool_indices=torch.full_like(row_requests, 1)
        )
        eager_metadata = SimpleNamespace(
            sequence_lengths=sequence_lengths,
            row_req_pool_indices=row_requests,
            is_cuda_graph=False,
            fa2_valid_counts=None,
        )
        gpu_output = self.gpu_backend._forward_trtllm_sparse(
            q,
            self.gpu_backend.token_to_kv_pool.get_key_buffer(_LAYER_ID),
            self.gpu_backend.token_to_kv_pool.get_value_buffer(_LAYER_ID),
            self.layer,
            forward_batch,
            eager_metadata,
            topk,
            trtllm_decode,
        )
        reference_output = self.reference_backend._forward_trtllm_sparse(
            q,
            self.reference_backend.token_to_kv_pool.get_key_buffer(_LAYER_ID),
            self.reference_backend.token_to_kv_pool.get_value_buffer(_LAYER_ID),
            self.layer,
            forward_batch,
            eager_metadata,
            topk,
            trtllm_decode,
        )

        valid_counts = torch.empty(q.shape[0], dtype=torch.int32, device="cuda")
        graph_metadata = SimpleNamespace(
            sequence_lengths=sequence_lengths,
            row_req_pool_indices=row_requests,
            is_cuda_graph=True,
            fa2_valid_counts=valid_counts,
        )
        self.host_backend._cuda_graph_max_tokens = q.shape[0]
        host_output = self.host_backend._forward_trtllm_sparse(
            q,
            None,
            None,
            self.layer,
            forward_batch,
            graph_metadata,
            topk,
            trtllm_decode,
            host_pool=self.host_pool,
        )
        torch.cuda.synchronize()
        self.assertEqual(valid_counts.cpu().tolist(), [6, 1, 5, 6])
        torch.testing.assert_close(host_output, gpu_output, atol=0, rtol=0)
        torch.testing.assert_close(host_output, reference_output, atol=0.006, rtol=0.02)
        self.assertEqual(int(torch.count_nonzero(host_output[1])), 0)
        gpu_packed = next(iter(self.gpu_backend._fa2_scratch.values()))
        host_packed = next(iter(self.host_backend._fa2_scratch.values()))
        for row, count in enumerate(valid_counts.cpu().tolist()):
            selected_slice = slice(row * _TOKEN_TOPK, row * _TOKEN_TOPK + count)
            self.assertTrue(
                torch.equal(
                    host_packed[0][selected_slice].view(torch.uint8),
                    gpu_packed[0][selected_slice].view(torch.uint8),
                )
            )
            self.assertTrue(
                torch.equal(
                    host_packed[1][selected_slice].view(torch.uint8),
                    gpu_packed[1][selected_slice].view(torch.uint8),
                )
            )

        # Capture the exact host gather + TRT-LLM kernel and retain the arena
        # until the graph has been destroyed and its last replay has drained.
        lease = self.host_pool.arena.retain_for_graph()
        graph = torch.cuda.CUDAGraph()
        graph_output = None
        try:
            torch.cuda.synchronize()
            with torch.cuda.graph(graph):
                graph_output = self.host_backend._forward_trtllm_sparse(
                    q,
                    None,
                    None,
                    self.layer,
                    forward_batch,
                    graph_metadata,
                    topk,
                    trtllm_decode,
                    host_pool=self.host_pool,
                )
            graph.replay()
            torch.cuda.synchronize()
            torch.testing.assert_close(graph_output, gpu_output, atol=0, rtol=0)

            graph_scratch_key = next(
                key for key in self.host_backend._fa2_scratch if key[0]
            )
            graph_slot_key = next(
                key for key in self.host_backend._host_slot_scratch if key[0]
            )
            decode_scratch_ptrs = tuple(
                tensor.data_ptr()
                for tensor in self.host_backend._fa2_scratch[graph_scratch_key]
            )
            decode_slot_ptr = self.host_backend._host_slot_scratch[
                graph_slot_key
            ].data_ptr()

            # Grow eager decode beyond the captured batch. Graph-owned K/V and
            # slot scratch must remain at their captured addresses.
            large_q = torch.cat((q, q[:3]), dim=0)
            large_topk = torch.cat((topk, topk[:3]), dim=0)
            large_requests = torch.cat((row_requests, row_requests[:3]), dim=0)
            large_lengths = torch.cat((sequence_lengths, sequence_lengths[:3]), dim=0)
            large_metadata = SimpleNamespace(
                sequence_lengths=large_lengths,
                row_req_pool_indices=large_requests,
                is_cuda_graph=False,
                fa2_valid_counts=None,
            )
            large_gpu_output = self.gpu_backend._forward_trtllm_sparse(
                large_q,
                self.gpu_backend.token_to_kv_pool.get_key_buffer(_LAYER_ID),
                self.gpu_backend.token_to_kv_pool.get_value_buffer(_LAYER_ID),
                self.layer,
                forward_batch,
                large_metadata,
                large_topk,
                trtllm_decode,
            )
            large_host_output = self.host_backend._forward_trtllm_sparse(
                large_q,
                None,
                None,
                self.layer,
                forward_batch,
                large_metadata,
                large_topk,
                trtllm_decode,
                host_pool=self.host_pool,
            )
            torch.testing.assert_close(
                large_host_output, large_gpu_output, atol=0, rtol=0
            )
            self.assertTrue(any(not key[0] for key in self.host_backend._fa2_scratch))
            self.assertTrue(
                any(not key[0] for key in self.host_backend._host_slot_scratch)
            )

            extend_q, extend_k, extend_v, extend_topk, extend_batch, _ = _extend_inputs(
                self.req_to_token
            )
            self.host_backend.forward_extend(
                extend_q,
                extend_k,
                extend_v,
                self.layer,
                extend_batch,
                topk_indices=extend_topk,
            )
            self.host_pool.arena.check_errors()
            self.assertIsNotNone(self.host_backend._host_prefill_scratch)
            self.assertEqual(
                decode_scratch_ptrs,
                tuple(
                    tensor.data_ptr()
                    for tensor in self.host_backend._fa2_scratch[graph_scratch_key]
                ),
            )
            self.assertEqual(
                decode_slot_ptr,
                self.host_backend._host_slot_scratch[graph_slot_key].data_ptr(),
            )

            graph.replay()
            torch.cuda.synchronize()
            torch.testing.assert_close(graph_output, gpu_output, atol=0, rtol=0)
        finally:
            del graph
            torch.cuda.synchronize()
            lease.close()

    def test_prefix_extend_fp8_packing_reference_and_input_immutability(self):
        q, keys, values, topk, forward_batch, sequence_lengths = _extend_inputs(
            self.req_to_token
        )
        original_keys = keys.clone()
        original_values = values.clone()
        gpu_output = self.gpu_backend.forward_extend(
            q,
            keys,
            values,
            self.layer,
            forward_batch,
            topk_indices=topk,
        )
        self.assertTrue(torch.equal(keys, original_keys))
        self.assertTrue(torch.equal(values, original_values))
        self.gpu_backend._paged_fp8_prefill = True
        with (
            patch.object(
                torch.Tensor, "item", side_effect=AssertionError("device item")
            ),
            patch.object(
                torch.Tensor, "tolist", side_effect=AssertionError("device tolist")
            ),
        ):
            paged_output = self.gpu_backend.forward_extend(
                q, keys, values, self.layer, forward_batch, topk_indices=topk
            )
        torch.testing.assert_close(paged_output, gpu_output, atol=0, rtol=0)
        self.assertTrue(torch.equal(keys, original_keys))
        self.assertTrue(torch.equal(values, original_values))
        host_output = self.host_backend.forward_extend(
            q,
            keys,
            values,
            self.layer,
            forward_batch,
            topk_indices=topk,
        )
        self.assertTrue(torch.equal(keys, original_keys))
        self.assertTrue(torch.equal(values, original_values))
        self.host_pool.arena.check_errors()

        _, reference_keys = _fp8_and_dequantized(keys, _K_SCALE)
        _, reference_values = _fp8_and_dequantized(values, _V_SCALE)
        reference_output = self.reference_backend.forward_extend(
            q,
            reference_keys,
            reference_values,
            self.layer,
            forward_batch,
            topk_indices=topk,
        )

        packed_keys, packed_values = self.host_backend._host_prefill_scratch
        packed_rows = sum(sequence_lengths)
        request_ids = forward_batch.req_pool_indices.tolist()
        expected_keys = torch.cat(
            [
                self.gpu_backend.token_to_kv_pool.get_key_buffer(
                    _LAYER_ID
                ).index_select(
                    0,
                    self.req_to_token[request_ids[index], :length].long(),
                )
                for index, length in enumerate(sequence_lengths)
            ]
        )
        expected_values = torch.cat(
            [
                self.gpu_backend.token_to_kv_pool.get_value_buffer(
                    _LAYER_ID
                ).index_select(
                    0,
                    self.req_to_token[request_ids[index], :length].long(),
                )
                for index, length in enumerate(sequence_lengths)
            ]
        )
        self.assertTrue(
            torch.equal(
                packed_keys[:packed_rows].view(torch.uint8),
                expected_keys.view(torch.uint8),
            )
        )
        self.assertTrue(
            torch.equal(
                packed_values[:packed_rows].view(torch.uint8),
                expected_values.view(torch.uint8),
            )
        )
        torch.testing.assert_close(host_output, gpu_output, atol=0, rtol=0)
        torch.testing.assert_close(host_output, reference_output, atol=0.006, rtol=0.02)


if __name__ == "__main__":
    unittest.main()

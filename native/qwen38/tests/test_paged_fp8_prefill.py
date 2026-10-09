"""Paged FP8 chunk-prefill contracts against packed BF16 and math references."""

import unittest
from unittest.mock import patch

import torch

from sglang.srt.layers.attention.qsa.sparse_attn import (
    sparse_gqa_fwd_interface_triton_ck,
    sparse_gqa_fwd_interface_triton_paged_ck,
)


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class PagedFP8PrefillTest(unittest.TestCase):
    def _case(self, seq_lens=(67, 69), chunks=(3, 5), per_head=False):
        torch.manual_seed(27)
        capacity = sum(seq_lens) + 64
        table_width = max(seq_lens)
        requests = torch.tensor([2, 1], dtype=torch.int32, device="cuda")
        # Non-contiguous table and K/V strides catch accidental layout
        # assumptions; physical slots are deliberately unrelated to positions.
        table = torch.zeros((3, table_width * 2), dtype=torch.int32, device="cuda")[
            :, ::2
        ]
        perm = torch.randperm(capacity - 1, device="cuda") + 1
        offset = 0
        for req, length in zip((2, 1), seq_lens):
            table[req, :length] = perm[offset : offset + length].to(torch.int32)
            offset += length
        k = (torch.randn(capacity, 2, 512, device="cuda") * 0.3).to(
            torch.float8_e4m3fn
        )[:, :, ::2]
        v = (torch.randn_like(k, dtype=torch.float32) * 0.3).to(torch.float8_e4m3fn)
        q = (torch.randn(sum(chunks), 24, 256, device="cuda") * 0.1).to(torch.bfloat16)
        shape = (sum(chunks), 2, 2048) if per_head else (sum(chunks), 2048)
        indices = torch.full(shape, -1, dtype=torch.int32, device="cuda")
        row = 0
        for length, chunk in zip(seq_lens, chunks):
            for position in range(chunk):
                visible = length - chunk + position + 1
                for head in range(2 if per_head else 1):
                    selected = sorted({0, 1, 3 + head, 11, visible // 2, visible - 1})
                    target = indices[row, head] if per_head else indices[row]
                    target[: len(selected)] = torch.tensor(selected, device="cuda")
                row += 1
        cu_q = torch.tensor(
            [0, chunks[0], sum(chunks)], dtype=torch.int32, device="cuda"
        )
        lens = torch.tensor(seq_lens, dtype=torch.int32, device="cuda")
        return q, k, v, indices, cu_q, lens, table, requests, chunks

    def _packed_reference(self, case, k_scale, v_scale):
        q, k, v, indices, cu_q, lens, table, requests, chunks = case
        lengths = lens.tolist()
        slots = torch.cat(
            [table[r, :n].long() for r, n in zip(requests.tolist(), lengths)]
        )
        packed_k = (k.index_select(0, slots).to(torch.bfloat16) * k_scale).to(
            torch.bfloat16
        )
        packed_v = (v.index_select(0, slots).to(torch.bfloat16) * v_scale).to(
            torch.bfloat16
        )
        cu_k = torch.tensor(
            [0, lengths[0], sum(lengths)], dtype=torch.int32, device="cuda"
        )
        return sparse_gqa_fwd_interface_triton_ck(
            q,
            packed_k,
            packed_v,
            indices,
            cu_q,
            cu_k,
            lens,
            256**-0.5,
            max_query_len=max(chunks),
        )

    def _paged(self, case, k_scale=1.0, v_scale=1.0):
        q, k, v, indices, cu_q, lens, table, requests, chunks = case
        return sparse_gqa_fwd_interface_triton_paged_ck(
            q,
            k,
            v,
            indices,
            cu_q,
            lens,
            table,
            requests,
            256**-0.5,
            max_query_len=max(chunks),
            k_scale=k_scale,
            v_scale=v_scale,
        )

    def test_packed_parity_scalar_and_device_scales_per_head_and_shared(self):
        for per_head in (False, True):
            case = self._case(per_head=per_head)
            for device_scale in ("python", "scalar", "vector"):
                ks, vs = 0.31, 0.27
                if device_scale != "python":
                    ks, vs = (torch.tensor(x, device="cuda") for x in (ks, vs))
                    if device_scale == "vector":
                        ks, vs = ks.reshape(1), vs.reshape(1)
                expected = self._packed_reference(case, ks, vs)
                actual = self._paged(case, ks, vs)
                torch.testing.assert_close(actual, expected, atol=0, rtol=0)
                self.assertTrue(torch.isfinite(actual).all().item())

    def test_independent_attention_math(self):
        case = self._case()
        q, k, v, indices, _, lens, table, requests, chunks = case
        actual = self._paged(case)
        oracle = torch.empty_like(q)
        row = 0
        for req, length, chunk in zip(requests.tolist(), lens.tolist(), chunks):
            for _ in range(chunk):
                selected = indices[row]
                slots = table[req, selected[selected >= 0].long()].long()
                keys = (
                    k.index_select(0, slots)
                    .to(torch.float32)
                    .repeat_interleave(12, dim=1)
                )
                values = (
                    v.index_select(0, slots)
                    .to(torch.float32)
                    .repeat_interleave(12, dim=1)
                )
                scores = torch.einsum("hd,nhd->hn", q[row].float(), keys) * (256**-0.5)
                oracle[row] = torch.einsum("hn,nhd->hd", scores.softmax(-1), values).to(
                    q.dtype
                )
                row += 1
        torch.testing.assert_close(actual, oracle, atol=0.006, rtol=0.02)

    def test_graph_replay_tracks_device_request_mapping_without_scalar_reads(self):
        case = self._case()
        self._paged(case)
        with (
            patch.object(
                torch.Tensor, "item", side_effect=AssertionError("device item")
            ),
            patch.object(
                torch.Tensor, "tolist", side_effect=AssertionError("device tolist")
            ),
        ):
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                actual = self._paged(case)
            graph.replay()
        torch.testing.assert_close(
            actual, self._packed_reference(case, 1.0, 1.0), atol=0, rtol=0
        )
        # Mutate a selected physical mapping, not a captured host list.
        table = case[6]
        previous = actual.clone()
        table[2, 0] = table[1, 0]
        graph.replay()
        self.assertFalse(torch.equal(actual, previous))
        torch.testing.assert_close(
            actual, self._packed_reference(case, 1.0, 1.0), atol=0, rtol=0
        )

    def test_long_context_no_full_history_workspace(self):
        case = self._case(seq_lens=(131075, 131077))
        self._paged(case)
        torch.cuda.synchronize()
        before = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        actual = self._paged(case)
        torch.cuda.synchronize()
        extra = torch.cuda.max_memory_allocated() - before
        self.assertLess(extra, 2 * 1024 * 1024)
        torch.testing.assert_close(
            actual, self._packed_reference(case, 1.0, 1.0), atol=0, rtol=0
        )


if __name__ == "__main__":
    unittest.main()

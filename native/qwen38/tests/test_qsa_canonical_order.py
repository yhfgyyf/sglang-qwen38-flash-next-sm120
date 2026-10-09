"""Regression coverage for deterministic QSA compressed-block expansion."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from sglang.srt.layers.attention.qsa import qsa_indexer as qsa_indexer_module
from sglang.srt.layers.attention.qsa.kernel import (
    expand_qsa_block_indices,
    torch_expand_qsa_block_indices,
)
from sglang.srt.layers.attention.qsa.qsa_indexer import QSAIndexer


RATIO = 4


def _block16_bf16p_online_attention(
    slots: torch.Tensor,
    scores: torch.Tensor,
    values: torch.Tensor,
) -> torch.Tensor:
    """Mirror sparse_attn's BLOCK_N=16 FP32 recurrence and BF16 P for PV."""

    max_score = torch.tensor(float("-inf"), dtype=torch.float32)
    normalizer = torch.tensor(0.0, dtype=torch.float32)
    accumulator = torch.zeros(values.shape[1], dtype=torch.float32)
    assert slots.numel() % 16 == 0
    for start in range(0, slots.numel(), 16):
        tile = slots[start : start + 16]
        tile_scores = scores[tile]
        next_max = torch.maximum(max_score, tile_scores.max())
        alpha = torch.exp2(max_score - next_max)
        probabilities = torch.exp2(tile_scores - next_max)
        pv = probabilities.to(torch.bfloat16).float() @ values[tile].float()
        accumulator = accumulator * alpha + pv
        normalizer = normalizer * alpha + probabilities.sum()
        max_score = next_max
    return (accumulator / normalizer).to(torch.bfloat16)


def test_cpu_canonical_order_stabilizes_permutations_padding_and_tail():
    token_topk = 12
    query_positions = torch.tensor([10], dtype=torch.int32)
    sequence_lengths = torch.tensor([11], dtype=torch.int32)
    blocks_a = torch.tensor([[1, 0, -1]], dtype=torch.int32)
    blocks_b = torch.tensor([[0, 1, -1]], dtype=torch.int32)
    original_a = blocks_a.clone()
    original_b = blocks_b.clone()

    legacy_a = expand_qsa_block_indices(
        blocks_a, query_positions, sequence_lengths, RATIO, token_topk
    )
    legacy_b = expand_qsa_block_indices(
        blocks_b, query_positions, sequence_lengths, RATIO, token_topk
    )
    canonical_a = expand_qsa_block_indices(
        blocks_a,
        query_positions,
        sequence_lengths,
        RATIO,
        token_topk,
        canonical_order=True,
    )
    canonical_b = expand_qsa_block_indices(
        blocks_b,
        query_positions,
        sequence_lengths,
        RATIO,
        token_topk,
        canonical_order=True,
    )

    assert legacy_a[0, :11].tolist() == [4, 5, 6, 7, 0, 1, 2, 3, 8, 9, 10]
    assert legacy_b[0, :11].tolist() == list(range(11))
    assert torch.equal(canonical_a, canonical_b)
    assert canonical_a[0, :11].tolist() == list(range(11))
    assert torch.all(canonical_a[0, 11:] == -1)
    assert torch.equal(blocks_a, original_a)
    assert torch.equal(blocks_b, original_b)


def test_block16_online_attention_repeats_after_canonical_order():
    token_topk = 64
    query_positions = torch.tensor([63], dtype=torch.int32)
    sequence_lengths = torch.tensor([64], dtype=torch.int32)
    blocks_a = torch.arange(16, dtype=torch.int32).flip(0)[None]
    blocks_b = torch.arange(16, dtype=torch.int32)[None]

    legacy_a = expand_qsa_block_indices(
        blocks_a, query_positions, sequence_lengths, RATIO, token_topk
    )
    legacy_b = expand_qsa_block_indices(
        blocks_b, query_positions, sequence_lengths, RATIO, token_topk
    )
    canonical_a = expand_qsa_block_indices(
        blocks_a,
        query_positions,
        sequence_lengths,
        RATIO,
        token_topk,
        canonical_order=True,
    )
    canonical_b = expand_qsa_block_indices(
        blocks_b,
        query_positions,
        sequence_lengths,
        RATIO,
        token_topk,
        canonical_order=True,
    )

    assert not torch.equal(legacy_a, legacy_b)
    assert torch.equal(canonical_a, canonical_b)
    torch.manual_seed(0)
    scores = torch.randn(64, dtype=torch.float32)
    values = torch.randn(64, 8, dtype=torch.bfloat16)
    legacy_output_a = _block16_bf16p_online_attention(
        legacy_a[0, :token_topk], scores, values
    )
    legacy_output_b = _block16_bf16p_online_attention(
        legacy_b[0, :token_topk], scores, values
    )
    assert not torch.equal(legacy_output_a, legacy_output_b)
    assert torch.equal(
        _block16_bf16p_online_attention(canonical_a[0, :token_topk], scores, values),
        _block16_bf16p_online_attention(canonical_b[0, :token_topk], scores, values),
    )


class _IdentityRotary:
    rotary_dim = 4
    is_neox_style = True
    mrope_section = None
    mrope_interleaved = False
    mrope_interleaved_glm = False

    def __init__(self):
        self.cos_sin_cache = torch.tensor([[1.0, 1.0, 0.0, 0.0]])


def _make_cpu_indexer() -> QSAIndexer:
    config = SimpleNamespace(
        indexer_n_heads=1,
        indexer_kv_heads=1,
        indexer_head_dim=4,
        indexer_budget=2048,
        indexer_compress_ratio=4,
        hidden_size=8,
        rms_norm_eps=1e-6,
    )
    return QSAIndexer(config, layer_id=0, rotary_emb=_IdentityRotary())


def test_indexer_canonical_order_is_strictly_opt_in(monkeypatch):
    monkeypatch.delenv("QWEN38_QSA_CANONICAL_ORDER", raising=False)
    assert _make_cpu_indexer().canonical_order is False

    monkeypatch.setenv("QWEN38_QSA_CANONICAL_ORDER", "1")
    assert _make_cpu_indexer().canonical_order is True

    monkeypatch.setenv("QWEN38_QSA_CANONICAL_ORDER", "true")
    assert _make_cpu_indexer().canonical_order is False


def test_indexer_passes_canonical_order_to_prefill_and_decode(monkeypatch):
    indexer = QSAIndexer.__new__(QSAIndexer)
    indexer.token_topk = 2048
    indexer.compress_ratio = RATIO
    indexer.block_topk = 512
    indexer.canonical_order = True
    canonical_flags = []
    stable_tie_flags = []

    def fake_expand(
        block_indices,
        query_positions,
        sequence_lengths,
        compress_ratio,
        token_topk,
        *,
        canonical_order=False,
    ):
        canonical_flags.append(canonical_order)
        return torch.full(
            (block_indices.shape[0], token_topk + compress_ratio - 1),
            -1,
            dtype=torch.int32,
            device=block_indices.device,
        )

    monkeypatch.setattr(qsa_indexer_module, "expand_qsa_block_indices", fake_expand)
    monkeypatch.setattr(
        qsa_indexer_module,
        "qsa_mqa_prefill",
        lambda *args, **kwargs: torch.zeros((1, 513), dtype=torch.float32),
    )
    monkeypatch.setattr(
        qsa_indexer_module,
        "qsa_mqa_decode",
        lambda *args, **kwargs: torch.zeros((1, 513), dtype=torch.float32),
    )

    def fake_topk(*args, stable_ties=False, **kwargs):
        stable_tie_flags.append(stable_ties)
        return torch.arange(512, dtype=torch.int32)[None]

    monkeypatch.setattr(qsa_indexer_module, "qsa_fast_topk", fake_topk)
    q = torch.zeros((1, 1, 4), dtype=torch.float32)
    indexer.select_prefill_tokens(
        q,
        compressed_keys=torch.zeros((1, 1, 4)),
        row_starts=torch.zeros(1, dtype=torch.int32),
        row_ends=torch.ones(1, dtype=torch.int32),
        query_positions=torch.zeros(1, dtype=torch.int32),
        sequence_lengths_for_rows=torch.ones(1, dtype=torch.int32),
    )

    indexer.select_decode_tokens(
        q,
        compressed_cache=torch.empty((0, 1, 4)),
        compressed_page_table=torch.empty((1, 0), dtype=torch.int32),
        compressed_lengths=torch.tensor([513], dtype=torch.int32),
        max_model_len=513,
        query_positions=torch.tensor([512], dtype=torch.int32),
        sequence_lengths=torch.tensor([513], dtype=torch.int32),
    )

    assert canonical_flags == [True, True]
    assert stable_tie_flags == [True, True]


def _gpu_block_permutations(block_topk: int):
    rows = 2
    counts = (block_topk, block_topk - 17)
    blocks_a = torch.full((rows, block_topk), -1, dtype=torch.int32, device="cuda")
    blocks_b = blocks_a.clone()
    for row, count in enumerate(counts):
        selected = torch.arange(count, dtype=torch.int32, device="cuda")
        blocks_a[row, :count] = selected.flip(0)
        blocks_b[row, :count] = selected.roll(count // 3)
    query_positions = torch.full(
        (rows,), block_topk * RATIO + 2, dtype=torch.int32, device="cuda"
    )
    sequence_lengths = query_positions + 1
    return blocks_a, blocks_b, query_positions, sequence_lengths


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton expansion needs CUDA")
@pytest.mark.parametrize("block_topk", [512, 2048])
def test_cuda_canonical_order_matches_cpu_and_preserves_inputs(block_topk):
    token_topk = block_topk * RATIO
    blocks_a, blocks_b, query_positions, sequence_lengths = _gpu_block_permutations(
        block_topk
    )
    original_a = blocks_a.clone()
    original_b = blocks_b.clone()

    canonical_a = expand_qsa_block_indices(
        blocks_a,
        query_positions,
        sequence_lengths,
        RATIO,
        token_topk,
        canonical_order=True,
    )
    canonical_b = expand_qsa_block_indices(
        blocks_b,
        query_positions,
        sequence_lengths,
        RATIO,
        token_topk,
        canonical_order=True,
    )
    reference = torch_expand_qsa_block_indices(
        blocks_a.cpu(),
        query_positions.cpu(),
        sequence_lengths.cpu(),
        RATIO,
        token_topk,
        canonical_order=True,
    )

    assert torch.equal(canonical_a, canonical_b)
    assert torch.equal(canonical_a.cpu(), reference)
    assert torch.equal(blocks_a, original_a)
    assert torch.equal(blocks_b, original_b)

    legacy_default = expand_qsa_block_indices(
        blocks_a, query_positions, sequence_lengths, RATIO, token_topk
    )
    legacy_explicit = expand_qsa_block_indices(
        blocks_a,
        query_positions,
        sequence_lengths,
        RATIO,
        token_topk,
        canonical_order=False,
    )
    assert torch.equal(legacy_default, legacy_explicit)
    assert not torch.equal(legacy_default, canonical_a)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph needs CUDA")
def test_cuda_graph_replay_canonicalizes_mutated_blocks():
    block_topk = 512
    token_topk = block_topk * RATIO
    blocks, _, query_positions, sequence_lengths = _gpu_block_permutations(block_topk)

    expand_qsa_block_indices(
        blocks,
        query_positions,
        sequence_lengths,
        RATIO,
        token_topk,
        canonical_order=True,
    )
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = expand_qsa_block_indices(
            blocks,
            query_positions,
            sequence_lengths,
            RATIO,
            token_topk,
            canonical_order=True,
        )

    blocks.fill_(-1)
    blocks[0, :400] = torch.arange(400, dtype=torch.int32, device="cuda").roll(137)
    blocks[1, :300] = torch.arange(300, dtype=torch.int32, device="cuda").flip(0)
    query_positions.copy_(torch.tensor([1702, 1401], dtype=torch.int32, device="cuda"))
    sequence_lengths.copy_(query_positions + 1)
    graph.replay()

    expected = torch_expand_qsa_block_indices(
        blocks.cpu(),
        query_positions.cpu(),
        sequence_lengths.cpu(),
        RATIO,
        token_topk,
        canonical_order=True,
    )
    assert torch.equal(captured.cpu(), expected)

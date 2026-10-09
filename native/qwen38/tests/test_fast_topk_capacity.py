"""Capacity and replay regressions for the JIT ``fast_topk`` kernel.

Run with the pinned Qwen3.8 environment and this worktree on ``PYTHONPATH``.
These tests intentionally exercise score distributions whose FP16 coarse radix
bin contains more entries than the kernel's 4,096-index shared-memory staging
area.
"""

from __future__ import annotations

import pytest
import torch

from sglang.kernels.ops.elementwise.fast_topk import fast_topk


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="fast_topk requires CUDA"
)


def _section(
    score: torch.Tensor,
    row: int,
    row_start: int,
    length: int,
) -> torch.Tensor:
    return score[row, row_start : row_start + length]


def _assert_topk_values(
    score: torch.Tensor,
    lengths: torch.Tensor,
    indices: torch.Tensor,
    topk: int,
    row_starts: torch.Tensor | None = None,
) -> None:
    """Check the value threshold while allowing arbitrary equal-value ties."""

    lengths_cpu = lengths.cpu()
    starts_cpu = row_starts.cpu() if row_starts is not None else None
    for row in range(score.shape[0]):
        length = int(lengths_cpu[row])
        start = int(starts_cpu[row]) if starts_cpu is not None else 0
        actual_indices = indices[row]
        if length <= topk:
            assert torch.equal(
                actual_indices[:length].cpu(), torch.arange(length, dtype=torch.int32)
            )
            assert (actual_indices[length:] == -1).all()
            continue

        assert (actual_indices >= 0).all()
        assert (actual_indices < length).all()
        assert torch.unique(actual_indices).numel() == topk
        values = _section(score, row, start, length)
        actual = values[actual_indices.long()].sort(descending=True).values
        expected = torch.topk(values, topk).values.sort(descending=True).values
        assert torch.equal(actual, expected), f"row {row}: selected values differ"


def _assert_exact_distinct_topk(
    score: torch.Tensor,
    length: int,
    indices: torch.Tensor,
    topk: int,
    row_start: int = 0,
) -> None:
    values = _section(score, 0, row_start, length)
    expected = torch.topk(values, topk).indices.to(torch.int32)
    actual = indices[0]
    assert torch.equal(actual.sort().values, expected.sort().values)
    assert torch.equal(
        values[actual.long()].sort(descending=True).values,
        torch.topk(values, topk).values.sort(descending=True).values,
    )


@pytest.mark.parametrize("topk", [512, 2048])
@pytest.mark.parametrize("length", [8192, 32768, 65536])
def test_monotone_values_in_one_coarse_bin_select_exact_set(topk: int, length: int):
    # [1.0, 1.1] has one FP16 high-byte key but distinct FP32 values. The true
    # top-k is the final `topk` positions, independent of atomic arrival order.
    score = torch.linspace(1.0, 1.1, length, dtype=torch.float32, device="cuda")[None]
    lengths = torch.tensor([length], dtype=torch.int32, device="cuda")

    indices = fast_topk(score, lengths, topk)

    _assert_exact_distinct_topk(score, length, indices, topk)


@pytest.mark.parametrize("topk", [512, 2048])
@pytest.mark.parametrize("length", [4096, 4097])
def test_coarse_bin_at_and_just_over_capacity(topk: int, length: int):
    score = torch.linspace(1.0, 1.1, length, dtype=torch.float32, device="cuda")[None]
    lengths = torch.tensor([length], dtype=torch.int32, device="cuda")

    indices = fast_topk(score, lengths, topk)

    _assert_exact_distinct_topk(score, length, indices, topk)


@pytest.mark.parametrize("topk", [512, 2048])
def test_more_than_capacity_exact_ties_are_valid(topk: int):
    length = 8192
    score = torch.full((1, length), 3.25, dtype=torch.float32, device="cuda")
    lengths = torch.tensor([length], dtype=torch.int32, device="cuda")

    indices = fast_topk(score, lengths, topk)

    _assert_topk_values(score, lengths, indices, topk)


@pytest.mark.parametrize("topk", [512, 2048])
def test_overflow_with_exact_radix_boundary(topk: int):
    # Both groups share the FP16 coarse key, so all 8192 candidates overflow
    # staging. The second full-row FP32 radix pass ends exactly at the high
    # group's boundary, exercising remaining==0 and suffix-bit filling.
    length = 8192
    score = torch.ones((1, length), dtype=torch.float32, device="cuda")
    score[:, -topk:] = 1.0625
    lengths = torch.tensor([length], dtype=torch.int32, device="cuda")

    indices = fast_topk(score, lengths, topk)

    expected = torch.arange(length - topk, length, dtype=torch.int32, device="cuda")
    assert torch.equal(indices[0].sort().values, expected)


@pytest.mark.parametrize("topk", [512, 2048])
def test_ragged_row_starts_short_rows_and_outer_stride(topk: int):
    width = 12288
    base = torch.full((4, 2 * width), -100.0, dtype=torch.float32, device="cuda")
    score = base[:, :width]
    assert score.stride() == (2 * width, 1)

    starts = torch.tensor([7, 13, 1024, 33], dtype=torch.int32, device="cuda")
    lengths = torch.tensor(
        [topk - 1, 8192, 4097, topk + 17], dtype=torch.int32, device="cuda"
    )
    for row, (start, length) in enumerate(zip(starts.cpu(), lengths.cpu())):
        score[row, int(start) : int(start + length)] = torch.linspace(
            1.0 + row,
            1.1 + row,
            int(length),
            dtype=torch.float32,
            device="cuda",
        )

    indices = fast_topk(score, lengths, topk, row_starts=starts)

    _assert_topk_values(score, lengths, indices, topk, starts)


@pytest.mark.parametrize("topk", [512, 2048])
def test_repeated_launches_agree_with_topk_threshold(topk: int):
    length = 8192
    score = torch.linspace(1.0, 1.1, length, dtype=torch.float32, device="cuda")[None]
    lengths = torch.tensor([length], dtype=torch.int32, device="cuda")

    for _ in range(10):
        indices = fast_topk(score, lengths, topk)
        _assert_topk_values(score, lengths, indices, topk)


@pytest.mark.parametrize("topk", [512, 2048])
def test_negative_zero_and_infinity_value_order(topk: int):
    length = 8192
    score = torch.linspace(-1.1, -1.0, length, dtype=torch.float32, device="cuda")
    score[0] = -torch.inf
    score[1] = -0.0
    score[2] = 0.0
    score[-2:] = torch.inf
    score = score[None]
    lengths = torch.tensor([length], dtype=torch.int32, device="cuda")

    indices = fast_topk(score, lengths, topk)

    _assert_topk_values(score, lengths, indices, topk)


@pytest.mark.parametrize("topk", [512, 2048])
def test_cuda_graph_replay_reads_mutated_inputs(topk: int):
    width = 8192
    initial_length = 8192
    score = torch.empty((1, width), dtype=torch.float32, device="cuda")
    lengths = torch.tensor([initial_length], dtype=torch.int32, device="cuda")
    starts = torch.zeros(1, dtype=torch.int32, device="cuda")
    score.copy_(torch.linspace(1.0, 1.1, initial_length, device="cuda"))

    # Compile before capture and ensure capture uses only replay-safe device work.
    fast_topk(score, lengths, topk, row_starts=starts)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured_indices = fast_topk(score, lengths, topk, row_starts=starts)

    replay_start = 1024
    replay_length = 4097
    score.fill_(-100.0)
    score[:, replay_start : replay_start + replay_length] = torch.linspace(
        2.1, 2.0, replay_length, device="cuda"
    )
    starts.fill_(replay_start)
    lengths.fill_(replay_length)
    graph.replay()

    _assert_exact_distinct_topk(
        score, replay_length, captured_indices, topk, row_start=replay_start
    )

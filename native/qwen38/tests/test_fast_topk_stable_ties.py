"""Stable cutoff-tie regressions for the JIT ``fast_topk`` kernel."""

from __future__ import annotations

import importlib

import pytest
import torch

from sglang.kernels.ops.elementwise.fast_topk import fast_topk
from sglang.srt.layers.attention.qsa.kernel import qsa_fast_topk


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="fast_topk requires CUDA"
)


def _stable_reference(section: torch.Tensor, topk: int) -> torch.Tensor:
    return torch.argsort(section.cpu(), dim=0, descending=True, stable=True)[:topk].to(
        torch.int32
    )


def _assert_stable_membership(
    score: torch.Tensor,
    starts: torch.Tensor,
    lengths: torch.Tensor,
    indices: torch.Tensor,
    topk: int,
) -> None:
    starts_cpu = starts.cpu()
    lengths_cpu = lengths.cpu()
    for row in range(score.shape[0]):
        start = int(starts_cpu[row])
        length = int(lengths_cpu[row])
        section = score[row, start : start + length]
        expected = _stable_reference(section, topk)
        actual = indices[row].cpu()
        assert torch.equal(actual.sort().values, expected.sort().values)


def _make_tied_rows(
    topk: int, *, overflow: bool
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    tie_count = 5000 if overflow else 137
    selected_ties = 37
    greater_count = topk - selected_ties
    lesser_count = 211
    length = tie_count + greater_count + lesser_count
    starts = torch.tensor([7, 19, 31], dtype=torch.int32, device="cuda")
    width = int(starts.max()) + length + 17
    base = torch.full((3, 2 * width), -100.0, dtype=torch.float32, device="cuda")
    score = base[:, :width]
    assert score.stride() == (2 * width, 1)

    levels = (
        (3.25, 4.0, 2.0),
        (-3.25, -2.0, -4.0),
        (0.0, 1.0, -1.0),
    )
    for row, (tie, greater, lesser) in enumerate(levels):
        start = int(starts[row])
        section = score[row, start : start + length]
        section.fill_(lesser)
        section[:tie_count] = tie
        if row == 2:
            # Numeric -0.0 and +0.0 are one cutoff-tie group. Alternating
            # signs makes a radix-key-only tie break observably incorrect.
            section[:tie_count:2] = -0.0
            section[1:tie_count:2] = 0.0
        section[tie_count : tie_count + greater_count] = greater

    lengths = torch.full((3,), length, dtype=torch.int32, device="cuda")
    return score, starts, lengths


@pytest.mark.parametrize("topk", [512, 2048])
@pytest.mark.parametrize("overflow", [False, True], ids=["staged", "overflow"])
def test_stable_ties_choose_lowest_relative_indices(topk: int, overflow: bool):
    score, starts, lengths = _make_tied_rows(topk, overflow=overflow)

    first = fast_topk(score, lengths, topk, row_starts=starts, stable_ties=True)
    _assert_stable_membership(score, starts, lengths, first, topk)

    first_sorted = first.sort(dim=1).values
    for _ in range(4):
        repeated = fast_topk(score, lengths, topk, row_starts=starts, stable_ties=True)
        assert torch.equal(repeated.sort(dim=1).values, first_sorted)


@pytest.mark.parametrize("topk", [512, 2048])
@pytest.mark.parametrize("overflow", [False, True], ids=["staged", "overflow"])
def test_stable_ties_scan_late_scattered_cutoff_indices(topk: int, overflow: bool):
    leading_less = 2300
    selected_ties = 1536 if topk == 2048 else 37
    greater_count = topk - selected_ties
    tie_count = 5000 if overflow else max(selected_ties + 137, 193)
    row_start = 17
    length = leading_less + 2 * tie_count + greater_count + 113
    width = row_start + length + 29
    base = torch.full((1, 2 * width), -1.0, dtype=torch.float32, device="cuda")
    score = base[:, :width]
    section = score[0, row_start : row_start + length]
    tie_indices = leading_less + 2 * torch.arange(tie_count, device="cuda")
    section[tie_indices] = 0.25
    greater_start = leading_less + 2 * tie_count
    section[greater_start : greater_start + greater_count] = 1.0
    starts = torch.tensor([row_start], dtype=torch.int32, device="cuda")
    lengths = torch.tensor([length], dtype=torch.int32, device="cuda")

    actual = fast_topk(score, lengths, topk, row_starts=starts, stable_ties=True)

    _assert_stable_membership(score, starts, lengths, actual, topk)
    chosen_ties = actual[0][actual[0] < greater_start]
    assert torch.equal(
        chosen_ties.sort().values.cpu(),
        tie_indices[:selected_ties].to(torch.int32).cpu(),
    )


@pytest.mark.parametrize("topk", [512, 2048])
@pytest.mark.parametrize("overflow", [False, True], ids=["staged", "overflow"])
def test_stable_ties_support_positive_and_negative_infinity(topk: int, overflow: bool):
    tie_count = 5000 if overflow else topk + 113
    negative_greater_count = topk - 37
    length = max(tie_count + 211, tie_count + negative_greater_count)
    starts = torch.tensor([11, 23], dtype=torch.int32, device="cuda")
    width = int(starts.max()) + length + 17
    score = torch.full((2, width), -1.0, dtype=torch.float32, device="cuda")

    positive = score[0, 11 : 11 + length]
    positive.fill_(0.0)
    positive[:tie_count] = torch.inf

    negative = score[1, 23 : 23 + length]
    negative.fill_(-torch.inf)
    negative[tie_count : tie_count + negative_greater_count] = 0.0

    lengths = torch.full((2,), length, dtype=torch.int32, device="cuda")
    actual = fast_topk(score, lengths, topk, row_starts=starts, stable_ties=True)

    _assert_stable_membership(score, starts, lengths, actual, topk)


@pytest.mark.parametrize("topk", [512, 2048])
def test_stable_ties_preserve_distinct_score_membership(topk: int):
    length = 8192
    score = torch.linspace(-1.1, -1.0, length, dtype=torch.float32, device="cuda")[None]
    lengths = torch.tensor([length], dtype=torch.int32, device="cuda")
    starts = torch.zeros(1, dtype=torch.int32, device="cuda")

    legacy = fast_topk(score, lengths, topk, row_starts=starts)
    explicit_legacy = fast_topk(
        score, lengths, topk, row_starts=starts, stable_ties=False
    )
    stable = fast_topk(score, lengths, topk, row_starts=starts, stable_ties=True)

    expected = torch.arange(length - topk, length, dtype=torch.int32, device="cuda")
    assert torch.equal(legacy[0].sort().values, expected)
    assert torch.equal(explicit_legacy[0].sort().values, expected)
    assert torch.equal(stable[0].sort().values, expected)


@pytest.mark.parametrize("topk", [512, 2048])
def test_stable_ties_short_and_empty_rows_keep_identity_padding(topk: int):
    score = torch.randn((3, topk + 17), dtype=torch.float32, device="cuda")
    starts = torch.tensor([7, 3, 0], dtype=torch.int32, device="cuda")
    lengths = torch.tensor([0, topk - 1, topk], dtype=torch.int32, device="cuda")

    actual = fast_topk(score, lengths, topk, row_starts=starts, stable_ties=True).cpu()

    assert torch.all(actual[0] == -1)
    assert torch.equal(actual[1, : topk - 1], torch.arange(topk - 1, dtype=torch.int32))
    assert actual[1, -1] == -1
    assert torch.equal(actual[2], torch.arange(topk, dtype=torch.int32))


@pytest.mark.parametrize("topk", [512, 2048])
def test_stable_ties_cuda_graph_replay_reads_mutated_data(topk: int):
    width = 12288
    score = torch.full((1, width), -100.0, dtype=torch.float32, device="cuda")
    starts = torch.zeros(1, dtype=torch.int32, device="cuda")
    lengths = torch.tensor([topk + 257], dtype=torch.int32, device="cuda")
    score[0, : int(lengths[0])] = torch.linspace(
        1.0, 1.1, int(lengths[0]), dtype=torch.float32, device="cuda"
    )

    fast_topk(score, lengths, topk, row_starts=starts, stable_ties=True)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = fast_topk(score, lengths, topk, row_starts=starts, stable_ties=True)

    replay_start = 211
    tie_count = 5000
    selected_ties = 31
    greater_count = topk - selected_ties
    replay_length = tie_count + greater_count + 127
    score.fill_(-100.0)
    section = score[0, replay_start : replay_start + replay_length]
    section.fill_(-1.0)
    section[:tie_count] = 0.0
    section[:tie_count:2] = -0.0
    section[tie_count : tie_count + greater_count] = 1.0
    starts.fill_(replay_start)
    lengths.fill_(replay_length)
    graph.replay()

    _assert_stable_membership(score, starts, lengths, captured, topk)


def test_qsa_cpu_stable_ties_choose_lowest_relative_indices():
    topk = 512
    start = 17
    length = 700
    logits = torch.full((1, 800), -1.0, dtype=torch.float32)
    section = logits[0, start : start + length]
    section[:100] = 0.0
    section[:100:2] = -0.0
    section[100 : 100 + topk - 23] = 1.0
    starts = torch.tensor([start], dtype=torch.int32)
    ends = torch.tensor([start + length], dtype=torch.int32)

    actual = qsa_fast_topk(logits, starts, ends, topk=topk, stable_ties=True)
    expected = _stable_reference(section, topk)

    assert torch.equal(actual[0].sort().values, expected.sort().values)


def test_qsa_cuda_stable_topk_2048_routes_to_jit(monkeypatch):
    fast_topk_module = importlib.import_module(
        "sglang.kernels.ops.elementwise.fast_topk"
    )
    calls = []

    def fake_fast_topk(score, lengths, topk, row_starts=None, *, stable_ties=False):
        calls.append((topk, stable_ties, lengths.clone(), row_starts.clone()))
        return torch.arange(topk, dtype=torch.int32, device=score.device)[None]

    monkeypatch.setattr(fast_topk_module, "fast_topk", fake_fast_topk)
    logits = torch.zeros((1, 4096), dtype=torch.float32, device="cuda")
    starts = torch.tensor([23], dtype=torch.int32, device="cuda")
    ends = torch.tensor([3023], dtype=torch.int32, device="cuda")

    actual = qsa_fast_topk(logits, starts, ends, topk=2048, stable_ties=True)

    assert actual.shape == (1, 2048)
    assert len(calls) == 1
    called_topk, called_stable, called_lengths, called_starts = calls[0]
    assert called_topk == 2048
    assert called_stable is True
    assert called_lengths.tolist() == [3000]
    assert called_starts.tolist() == [23]

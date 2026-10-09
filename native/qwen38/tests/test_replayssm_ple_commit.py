from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from sglang.srt.layers.attention.hybrid_linear_attn_backend import (
    HybridLinearAttnBackend,
)
from sglang.srt.mem_cache.ple_state_pool import NGramPool, ShortConvPool
from sglang.srt.speculative import spec_utils


class ForwardMode:
    def __init__(self, idle=False):
        self.idle = idle

    def is_idle(self):
        return self.idle


class StaticReqPool:
    def __init__(self, *, ple=True, fold=True):
        self.mamba_pool = SimpleNamespace(
            replayssm_spec_fold=fold,
            replayssm_is_kda=False,
            replayssm_cache_base=None,
        )
        self.mapping = torch.tensor([0, 2, 7, -1, 5, 4, 6, 3, 1, 8], dtype=torch.int32)
        self.translation_inputs = []
        self.short_conv_pool = ShortConvPool(
            size=12,
            spec_state_size=4,
            state_shape=(2, 2) if ple else None,
            layer_ids=[0] if ple else [],
            dtype=torch.float32,
            device="cpu",
            speculative_num_draft_tokens=4,
        )
        self.ngram_pool = NGramPool(
            size=12,
            spec_state_size=4,
            context_len=1 if ple else 0,
            eos_token_id=-99,
            device="cpu",
            speculative_num_draft_tokens=4,
        )
        self.spec_state = object()

    def get_mamba_indices(self, req_pool_indices):
        return self.mapping[req_pool_indices]

    def translate_mamba_indices(self, mamba_indices):
        self.translation_inputs.append(mamba_indices.clone())
        return mamba_indices

    def get_speculative_mamba2_params_all_layers(self):
        return self.spec_state


def fill_intermediate_states(pool):
    pool.short_conv_pool.conv_state.fill_(-1)
    pool.ngram_pool.context.fill_(-1)
    for row in range(4):
        for step in range(4):
            pool.short_conv_pool.intermediate_conv_state[:, row, step].fill_(
                100 * row + 10 * step + 1
            )
            pool.ngram_pool.intermediate_context[row, step].fill_(
                100 * row + 10 * step + 2
            )


def make_backend(pool, state_indices=None):
    backend = HybridLinearAttnBackend.__new__(HybridLinearAttnBackend)
    backend.linear_attn_backend = SimpleNamespace(
        req_to_token_pool=pool,
        forward_metadata=SimpleNamespace(
            mamba_cache_indices=(
                state_indices
                if state_indices is not None
                else torch.tensor([5, 2, -1, 7], dtype=torch.int32)
            )
        ),
        _translate_mamba_indices=pool.translate_mamba_indices,
    )
    return backend


def make_case(*, ple=True, fold=True, idle=False, empty=False):
    pool = StaticReqPool(ple=ple, fold=fold)
    if ple:
        fill_intermediate_states(pool)
    backend = make_backend(pool)
    worker = SimpleNamespace(
        model_runner=SimpleNamespace(
            model_config=object(),
            req_to_token_pool=pool,
            attn_backend=backend,
            model=object(),
        )
    )
    batch = SimpleNamespace(
        forward_mode=ForwardMode(idle),
        req_pool_indices=torch.tensor([4, 1, 3, 2], dtype=torch.int64),
        mamba_track_indices=torch.tensor([8, 9, 10, 11], dtype=torch.int64),
        seq_lens=torch.tensor([0, 2, 4, 7], dtype=torch.int32),
    )
    accept_lens = torch.tensor([1, 2, 3, 4], dtype=torch.int32)
    accept_index = torch.tensor(
        [
            [0, -1, -1, -1],
            [4, 5, -1, -1],
            [8, 9, 10, -1],
            [12, 13, 14, 15],
        ],
        dtype=torch.int32,
    )
    if empty:
        accept_lens = torch.empty(0, dtype=torch.int32)
        accept_index = torch.empty((0, 4), dtype=torch.int32)
        batch.req_pool_indices = torch.empty(0, dtype=torch.int64)
        batch.mamba_track_indices = torch.empty(0, dtype=torch.int64)
        batch.seq_lens = torch.empty(0, dtype=torch.int32)
    return pool, backend, worker, batch, accept_lens, accept_index


def assert_committed_sidecars(pool):
    main = ((5, 0, 0), (2, 1, 1), (7, 3, 3))
    for slot, row, step in main:
        assert torch.equal(
            pool.short_conv_pool.conv_state[:, slot],
            pool.short_conv_pool.intermediate_conv_state[:, row, step],
        )
        assert torch.equal(
            pool.ngram_pool.context[slot],
            pool.ngram_pool.intermediate_context[row, step],
        )

    tracked = ((9, 1, 1), (11, 3, 0))
    for slot, row, step in tracked:
        assert torch.equal(
            pool.short_conv_pool.conv_state[:, slot],
            pool.short_conv_pool.intermediate_conv_state[:, row, step],
        )
        assert torch.equal(
            pool.ngram_pool.context[slot],
            pool.ngram_pool.intermediate_context[row, step],
        )

    for untouched in (0, 1, 3, 4, 6, 8, 10, 12):
        assert torch.all(pool.short_conv_pool.conv_state[:, untouched] == -1)
        assert torch.all(pool.ngram_pool.context[untouched] == -1)


def run_commit(
    worker, batch, accept_lens, accept_index, draft_token_num=4, track_interval=4
):
    with (
        patch.object(spec_utils, "mambaish_config", return_value={}),
        patch.object(
            spec_utils,
            "get_exec",
            return_value=SimpleNamespace(
                mamba=SimpleNamespace(mamba_track_interval=track_interval)
            ),
        ),
    ):
        spec_utils.commit_mamba_states_after_verify(
            worker,
            batch,
            accept_lens,
            accept_index,
            draft_token_num=draft_token_num,
        )


def test_gdn_fold_commits_ple_sidecars_for_accept_1_through_4():
    pool, _backend, worker, batch, accept_lens, accept_index = make_case()
    fold_path = (
        "sglang.kernels.ops.attention.fla.gdn_replayssm_spec_fold."
        "commit_gdn_replayssm_fold_after_verify"
    )
    scatter_path = (
        "sglang.srt.layers.attention.hybrid_linear_attn_backend."
        "scatter_mamba_states_after_mtp_verify"
    )

    with patch(fold_path) as fold, patch(scatter_path) as legacy_scatter:
        run_commit(worker, batch, accept_lens, accept_index)

    fold.assert_called_once()
    legacy_scatter.assert_not_called()
    call = fold.call_args.kwargs
    assert call["spec_state"] is pool.spec_state
    assert torch.equal(
        call["state_batch_indices"], torch.tensor([5, 2, -1, 7], dtype=torch.int32)
    )
    assert torch.equal(
        call["last_correct_step_indices"],
        torch.tensor([0, 1, 2, 3], dtype=torch.int32),
    )
    assert torch.equal(
        call["mamba_steps_to_track"],
        torch.tensor([-1, 1, -1, 0], dtype=torch.int32),
    )
    assert len(pool.translation_inputs) == 2
    assert torch.equal(
        pool.translation_inputs[0], torch.tensor([5, 2, -1, 7], dtype=torch.int32)
    )
    assert torch.equal(
        pool.translation_inputs[1], torch.tensor([8, 9, 10, 11], dtype=torch.int64)
    )
    assert_committed_sidecars(pool)


def test_gdn_fold_without_ple_pools_remains_a_noop_for_sidecars():
    pool, _backend, worker, batch, accept_lens, accept_index = make_case(ple=False)
    fold_path = (
        "sglang.kernels.ops.attention.fla.gdn_replayssm_spec_fold."
        "commit_gdn_replayssm_fold_after_verify"
    )

    with patch(fold_path) as fold:
        run_commit(worker, batch, accept_lens, accept_index)

    fold.assert_called_once()
    assert pool.short_conv_pool.conv_state is None
    assert pool.ngram_pool.context is None


def test_gdn_fold_translates_main_and_track_once_at_real_64_token_boundary():
    pool, _backend, worker, batch, accept_lens, accept_index = make_case()
    batch.seq_lens = torch.tensor([61, 62, 65, 127], dtype=torch.int32)
    physical = torch.tensor([0, 1, 6, 2, 5, 4, 7, 3, 10, 11, 8, 9, 12])

    def translate(indices):
        pool.translation_inputs.append(indices.clone())
        return torch.where(indices < 0, indices, physical[indices.clamp_min(0)]).to(
            indices.dtype
        )

    pool.translate_mamba_indices = translate
    fold_path = (
        "sglang.kernels.ops.attention.fla.gdn_replayssm_spec_fold."
        "commit_gdn_replayssm_fold_after_verify"
    )
    with patch(fold_path) as fold:
        run_commit(worker, batch, accept_lens, accept_index, track_interval=64)
    assert len(pool.translation_inputs) == 2
    assert torch.equal(
        fold.call_args.kwargs["state_batch_indices"],
        torch.tensor([4, 6, -1, 3], dtype=torch.int32),
    )
    assert torch.equal(
        fold.call_args.kwargs["mamba_track_indices"],
        torch.tensor([10, 11, 8, 9], dtype=torch.int64),
    )
    assert torch.equal(
        fold.call_args.kwargs["mamba_steps_to_track"],
        torch.tensor([-1, 1, -1, 0], dtype=torch.int32),
    )
    for slot, row, step in ((4, 0, 0), (6, 1, 1), (3, 3, 3), (11, 1, 1), (9, 3, 0)):
        assert torch.equal(
            pool.short_conv_pool.conv_state[:, slot],
            pool.short_conv_pool.intermediate_conv_state[:, row, step],
        )
        assert torch.equal(
            pool.ngram_pool.context[slot],
            pool.ngram_pool.intermediate_context[row, step],
        )
    for slot in (0, 1, 2, 5, 7, 8, 10, 12):
        assert torch.all(pool.short_conv_pool.conv_state[:, slot] == -1)
        assert torch.all(pool.ngram_pool.context[slot] == -1)


def test_gdn_fold_width_one_tail_commits_step_zero_and_boundary_tracks():
    pool, _backend, worker, batch, _accept_lens, _accept_index = make_case()
    batch.seq_lens = torch.tensor([3, 4, 4, 7], dtype=torch.int32)
    accept_lens = torch.ones(4, dtype=torch.int32)
    accept_index = torch.arange(4, dtype=torch.int32).unsqueeze(1)
    fold_path = (
        "sglang.kernels.ops.attention.fla.gdn_replayssm_spec_fold."
        "commit_gdn_replayssm_fold_after_verify"
    )
    with patch(fold_path) as fold:
        run_commit(worker, batch, accept_lens, accept_index, draft_token_num=1)
    fold.assert_called_once()
    assert torch.equal(
        fold.call_args.kwargs["last_correct_step_indices"],
        torch.zeros(4, dtype=torch.int32),
    )
    assert torch.equal(
        fold.call_args.kwargs["mamba_steps_to_track"],
        torch.tensor([0, -1, -1, 0], dtype=torch.int32),
    )
    for slot, row in ((5, 0), (2, 1), (7, 3), (8, 0), (11, 3)):
        assert torch.equal(
            pool.short_conv_pool.conv_state[:, slot],
            pool.short_conv_pool.intermediate_conv_state[:, row, 0],
        )
        assert torch.equal(
            pool.ngram_pool.context[slot],
            pool.ngram_pool.intermediate_context[row, 0],
        )
    for slot in (0, 1, 3, 4, 6, 9, 10, 12):
        assert torch.all(pool.short_conv_pool.conv_state[:, slot] == -1)
        assert torch.all(pool.ngram_pool.context[slot] == -1)


@pytest.mark.parametrize("idle,empty", [(True, False), (False, True)])
def test_gdn_fold_idle_or_empty_does_not_commit(idle, empty):
    pool, _backend, worker, batch, accept_lens, accept_index = make_case(
        idle=idle, empty=empty
    )
    before_conv = pool.short_conv_pool.conv_state.clone()
    before_context = pool.ngram_pool.context.clone()
    fold_path = (
        "sglang.kernels.ops.attention.fla.gdn_replayssm_spec_fold."
        "commit_gdn_replayssm_fold_after_verify"
    )

    with patch(fold_path) as fold:
        run_commit(worker, batch, accept_lens, accept_index)

    fold.assert_not_called()
    assert torch.equal(pool.short_conv_pool.conv_state, before_conv)
    assert torch.equal(pool.ngram_pool.context, before_context)


def test_disabled_fold_uses_the_legacy_ssm_and_one_ple_commit():
    pool, backend, worker, batch, accept_lens, accept_index = make_case(fold=False)
    batch.mamba_track_indices = None
    scatter_path = (
        "sglang.srt.layers.attention.hybrid_linear_attn_backend."
        "scatter_mamba_states_after_mtp_verify"
    )

    with (
        patch(scatter_path) as legacy_scatter,
        patch.object(
            backend,
            "commit_ple_state_after_mtp_verify",
            wraps=backend.commit_ple_state_after_mtp_verify,
        ) as ple_commit,
    ):
        run_commit(worker, batch, accept_lens, accept_index)

    legacy_scatter.assert_called_once()
    ple_commit.assert_called_once()
    for slot, row, step in ((5, 0, 0), (2, 1, 1), (7, 3, 3)):
        assert torch.equal(
            pool.short_conv_pool.conv_state[:, slot],
            pool.short_conv_pool.intermediate_conv_state[:, row, step],
        )
        assert torch.equal(
            pool.ngram_pool.context[slot],
            pool.ngram_pool.intermediate_context[row, step],
        )

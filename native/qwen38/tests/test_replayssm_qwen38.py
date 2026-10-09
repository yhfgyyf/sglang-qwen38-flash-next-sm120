"""Qwen3.8-shaped FP32 ReplaySSM checks against recurrent snapshots.

These are GPU kernel tests, not a substitute for PLE/model/service validation.
The baseline stores the recurrent kernel's per-token state; the candidate
records raw inputs and folds only the accepted prefix. The tolerance is the
existing small-shape regression's bound, not a general accuracy guarantee.
"""

import pytest
import torch

from sglang.kernels.ops.attention.fla.fused_sigmoid_gating_recurrent import (
    fused_sigmoid_gating_delta_rule_update,
)
from sglang.kernels.ops.attention.fla.gdn_replayssm_spec_fold import (
    commit_gdn_replayssm_fold_all_layers,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

H, HV, K, V = 16, 48, 128, 128
WINDOW = 4
SLOTS = 12
ATOL = 2 * torch.finfo(torch.float32).eps


def _rings(layers):
    return {
        "rawv": torch.zeros(
            layers, SLOTS, HV, WINDOW, V, device="cuda", dtype=torch.bfloat16
        ),
        "rawk": torch.zeros(
            layers, SLOTS, H, WINDOW, K, device="cuda", dtype=torch.bfloat16
        ),
        "g": torch.zeros(layers, SLOTS, HV, WINDOW, device="cuda"),
        "beta": torch.zeros(layers, SLOTS, HV, WINDOW, device="cuda"),
    }


def _inputs(batch, width, seed):
    gen = torch.Generator(device="cuda").manual_seed(seed)

    def rand(*shape, dtype=torch.bfloat16):
        return torch.randn(*shape, device="cuda", dtype=dtype, generator=gen)

    return {
        "q": rand(1, batch * width, H, K),
        "k": rand(1, batch * width, H, K),
        "v": rand(1, batch * width, HV, V),
        "a": rand(batch * width, HV),
        "b": rand(batch * width, HV),
        "A_log": rand(HV, dtype=torch.float32) * 0.1,
        "dt_bias": rand(HV, dtype=torch.float32) * 0.1,
    }


def _verify(inputs, state, slots, width, *, snapshots=None, rings=None, layer=0):
    kwargs = {}
    if snapshots is not None:
        kwargs.update(
            intermediate_states_buffer=snapshots,
            intermediate_state_indices=slots,
            cache_steps=WINDOW,
        )
    if rings is not None:
        kwargs.update(
            cache_ring=True,
            replayssm_rawv=rings["rawv"][layer],
            replayssm_rawk=rings["rawk"][layer],
            replayssm_g=rings["g"][layer],
            replayssm_beta=rings["beta"][layer],
        )
    return fused_sigmoid_gating_delta_rule_update(
        **inputs,
        initial_state_source=state,
        initial_state_indices=slots,
        cu_seqlens=torch.arange(
            0, (slots.numel() + 1) * width, width, device="cuda", dtype=torch.int32
        ),
        softplus_beta=1.0,
        softplus_threshold=20.0,
        use_qk_l2norm_in_kernel=True,
        is_kda=False,
        disable_state_update=True,
        **kwargs,
    )


def _fold(state, rings, slots, lengths, track_slots=None, track_steps=None):
    commit_gdn_replayssm_fold_all_layers(
        checkpoint_state=state,
        rawv_cache=rings["rawv"],
        rawk_cache=rings["rawk"],
        g_cache=rings["g"],
        beta_cache=rings["beta"],
        ssm_state_indices=slots,
        accept_lens=lengths,
        max_cache_len=WINDOW,
        num_k_heads=H,
        mamba_track_indices=track_slots,
        mamba_steps_to_track=track_steps,
        null_block_id=-1,
    )


@pytest.mark.parametrize("capture", [False, True], ids=["eager", "graph"])
def test_qwen38_36_layer_fold_accept_track_and_null(capture):
    """One fused launch covers all 36 layers, remapped slots and accept 1..4."""
    gen = torch.Generator(device="cuda").manual_seed(2038)
    initial = torch.randn(36, SLOTS, HV, K, V, device="cuda", generator=gen)
    candidate = initial.clone()
    expected = initial.clone()
    rings = _rings(36)
    snapshots = torch.empty(SLOTS, WINDOW, HV, K, V, device="cuda")
    physical = [9, 2, 11, 5, 6, 7]
    lengths = [1, 2, 3, 4, 4, 0]
    tracks = [1, 3, 8, 10, 0, 4]
    steps = [0, -1, 1, 3, 2, 0]
    verify_slots = torch.tensor(physical, device="cuda", dtype=torch.int32)
    commit_slots = torch.tensor([9, 2, 11, 5, -1, 7], device="cuda", dtype=torch.int32)
    accept = torch.tensor(lengths, device="cuda", dtype=torch.int32)
    track_slots = torch.tensor(tracks, device="cuda", dtype=torch.int64)
    track_steps = torch.tensor(steps, device="cuda", dtype=torch.int64)
    for layer in range(36):
        inputs = _inputs(len(physical), WINDOW, 3000 + layer)
        baseline_out = _verify(
            inputs, initial[layer], verify_slots, WINDOW, snapshots=snapshots
        )
        replay_out = _verify(
            inputs, candidate[layer], verify_slots, WINDOW, rings=rings, layer=layer
        )
        assert torch.equal(baseline_out, replay_out), (
            f"ring write changed output at layer {layer}"
        )
        for row in range(4):
            expected[layer, physical[row]].copy_(
                snapshots[physical[row], lengths[row] - 1]
            )
            if steps[row] >= 0:
                expected[layer, tracks[row]].copy_(snapshots[physical[row], steps[row]])

    def commit():
        _fold(candidate, rings, commit_slots, accept, track_slots, track_steps)

    if capture:
        # Compile outside capture, then restore the same initial checkpoint.
        commit()
        torch.cuda.synchronize()
        candidate.copy_(initial)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            commit()
        candidate.copy_(initial)
        graph.replay()
    else:
        commit()

    changed = {9, 2, 11, 5, 1, 8, 10}
    for layer in range(36):
        torch.testing.assert_close(candidate[layer], expected[layer], rtol=0, atol=ATOL)
        for slot in set(range(SLOTS)) - changed:
            assert torch.equal(candidate[layer, slot], initial[layer, slot])
    assert candidate.dtype == torch.float32


def test_qwen38_chained_slot_reuse_and_width_four_to_one():
    """128 commits alternate normal/tail width and reuse noncontiguous slots."""
    gen = torch.Generator(device="cuda").manual_seed(4038)
    baseline = torch.randn(SLOTS, HV, K, V, device="cuda", generator=gen)
    candidate = baseline.clone().unsqueeze(0)
    rings = _rings(1)
    snapshots = torch.empty(SLOTS, WINDOW, HV, K, V, device="cuda")
    for iteration in range(128):
        width = 1 if iteration % 5 == 2 else WINDOW
        physical = [9, 2, 11, 5] if iteration % 2 else [5, 11, 2, 9]
        lengths = (
            [1] * 4
            if width == 1
            else [1 + (iteration + row) % WINDOW for row in range(4)]
        )
        slots = torch.tensor(physical, device="cuda", dtype=torch.int32)
        accept = torch.tensor(lengths, device="cuda", dtype=torch.int32)
        inputs = _inputs(4, width, 5000 + iteration)
        baseline_out = _verify(inputs, baseline, slots, width, snapshots=snapshots)
        replay_out = _verify(inputs, candidate[0], slots, width, rings=rings)
        torch.testing.assert_close(baseline_out, replay_out, rtol=0, atol=ATOL)
        for slot, length in zip(physical, lengths):
            baseline[slot].copy_(snapshots[slot, length - 1])
        _fold(candidate, rings, slots, accept)
        torch.testing.assert_close(candidate[0], baseline, rtol=0, atol=ATOL)

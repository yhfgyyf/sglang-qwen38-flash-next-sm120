"""Regression coverage for finite GDN decay-prefix rounding reversals.

The CPU tests derive the expected result from the token-by-token delta-rule
recurrence.  The CUDA test deliberately goes through the repository's current
Triton cumsum and chunk wrapper; it does not use the chunk algebra as its
reference.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

CHUNK_SIZE = 64
RAW_GATES = np.array(
    [-1.0, -3 * 2.0**-26, -3 * 2.0**-26, -(2.0**-27)] + [-1.0] * 60,
    dtype=np.float32,
)
BOUNDARY_LENGTHS = (4, 17, 67, 129)
BOUNDARY_IMPULSES = (2, 15, 62, 127)
BOUNDARY_STATE_INDICES = (7, 2, 8, 4)
BOUNDARY_CARRY_IN = (0.125, -0.25, 0.375, -0.5)


def _parallel_fp32_prefix(values: np.ndarray) -> np.ndarray:
    """Inclusive power-of-two scan with FP32 rounding after every addition."""
    prefix = np.asarray(values, dtype=np.float32).copy()
    offset = 1
    while offset < prefix.size:
        previous = prefix.copy()
        prefix[offset:] = previous[offset:] + previous[:-offset]
        offset *= 2
    return prefix


def _repeated_rounding_pattern(length: int, *, shift: int = 0) -> np.ndarray:
    pattern = np.roll(RAW_GATES[:4], shift)
    return np.resize(pattern, length).astype(np.float32, copy=False)


def _sequential_delta_rule_fp64(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    log_decay: np.ndarray,
    beta: np.ndarray,
    *,
    scale: float = 1.0,
    initial_state: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Token-by-token delta rule in the kernel's [value, key] state layout."""
    q = np.asarray(q, dtype=np.float64)
    k = np.asarray(k, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    log_decay = np.asarray(log_decay, dtype=np.float64)
    beta = np.asarray(beta, dtype=np.float64)

    tokens, key_heads, key_dim = k.shape
    value_heads, value_dim = v.shape[1:]
    assert q.shape == k.shape
    assert v.shape[0] == tokens
    assert log_decay.shape == beta.shape == (tokens, value_heads)
    assert value_heads % key_heads == 0

    if initial_state is None:
        state = np.zeros((value_heads, value_dim, key_dim), dtype=np.float64)
    else:
        state = np.asarray(initial_state, dtype=np.float64).copy()
        assert state.shape == (value_heads, value_dim, key_dim)

    output = np.empty_like(v, dtype=np.float64)
    value_heads_per_key_head = value_heads // key_heads
    for token in range(tokens):
        for value_head in range(value_heads):
            key_head = value_head // value_heads_per_key_head
            state[value_head] *= math.exp(log_decay[token, value_head])
            prediction = state[value_head] @ k[token, key_head]
            residual = beta[token, value_head] * (v[token, value_head] - prediction)
            state[value_head] += np.outer(residual, k[token, key_head])
            output[token, value_head] = (state[value_head] @ q[token, key_head]) * scale
    return output, state


def _one_head_witness(raw_gates: np.ndarray, *, scale: float = 1.0):
    q = np.zeros((CHUNK_SIZE, 1, 128), dtype=np.float64)
    k = np.zeros_like(q)
    q[..., 0] = 1.0
    k[..., 0] = 1.0
    v = np.zeros((CHUNK_SIZE, 1, 128), dtype=np.float64)
    v[2, 0, 0] = 1.0
    log_decay = np.asarray(raw_gates, dtype=np.float64).reshape(CHUNK_SIZE, 1)
    beta = np.full((CHUNK_SIZE, 1), 0.5, dtype=np.float64)
    return _sequential_delta_rule_fp64(q, k, v, log_decay, beta, scale=scale)


def _packed_one_head_oracle(
    gate_sequences: tuple[np.ndarray, ...], *, scale: float
) -> tuple[np.ndarray, np.ndarray]:
    outputs = []
    final_states = []
    for raw_gates, impulse, carry_in in zip(
        gate_sequences, BOUNDARY_IMPULSES, BOUNDARY_CARRY_IN, strict=True
    ):
        length = raw_gates.size
        q = np.zeros((length, 1, 128), dtype=np.float64)
        k = np.zeros_like(q)
        q[..., 0] = 1.0
        k[..., 0] = 1.0
        v = np.zeros((length, 1, 128), dtype=np.float64)
        v[impulse, 0, 0] = 1.0
        beta = np.full((length, 1), 0.5, dtype=np.float64)
        initial_state = np.zeros((1, 128, 128), dtype=np.float64)
        initial_state[0, 0, 0] = carry_in
        output, final_state = _sequential_delta_rule_fp64(
            q,
            k,
            v,
            raw_gates.reshape(length, 1),
            beta,
            scale=scale,
            initial_state=initial_state,
        )
        outputs.append(output)
        final_states.append(final_state)
    return np.concatenate(outputs), np.stack(final_states)


def test_parallel_fp32_prefix_has_finite_adjacent_rounding_reversal():
    prefix = _parallel_fp32_prefix(RAW_GATES)

    np.testing.assert_array_equal(
        prefix[:4],
        np.array([-1.0, -1.0, -1.0000001192092896, -1.0], dtype=np.float32),
    )
    assert np.isfinite(prefix).all()
    adjacent_difference = float(prefix[3] - prefix[2])
    assert adjacent_difference > 0

    # safe_exp(x) = exp(where(x <= 0, x, -inf)) discards this finite edge.
    safe_argument = adjacent_difference if adjacent_difference <= 0 else float("-inf")
    adjacent_weight = math.exp(safe_argument)
    assert adjacent_weight == 0.0


def test_sequential_fp64_oracle_preserves_token_three_signal():
    output, state = _one_head_witness(RAW_GATES)

    assert np.isfinite(output).all()
    assert np.isfinite(state).all()
    assert output[2, 0, 0] == 0.5
    assert math.isclose(
        output[3, 0, 0],
        0.24999999813735485,
        rel_tol=0.0,
        abs_tol=1e-15,
    )
    assert np.count_nonzero(output[3, 0, 1:]) == 0


def test_sequential_fp64_oracle_returns_carry_in_final_state():
    q = np.array([[[1.0, 0.0]], [[1.0, 0.0]]])
    k = q.copy()
    v = np.array([[[0.0, 0.0]], [[1.0, 0.0]]])
    log_decay = np.full((2, 1), math.log(0.5))
    beta = np.full((2, 1), 0.5)
    initial_state = np.array([[[1.0, 0.0], [0.0, 2.0]]])

    output, final_state = _sequential_delta_rule_fp64(
        q,
        k,
        v,
        log_decay,
        beta,
        initial_state=initial_state,
    )

    np.testing.assert_array_equal(
        output,
        np.array([[[0.25, 0.0]], [[0.5625, 0.0]]]),
    )
    np.testing.assert_array_equal(
        final_state,
        np.array([[[0.5625, 0.0], [0.0, 0.5]]]),
    )
    np.testing.assert_array_equal(
        initial_state,
        np.array([[[1.0, 0.0], [0.0, 2.0]]]),
    )


def test_boundary_patterns_supply_known_fp32_rounding_reversals():
    first_four = _parallel_fp32_prefix(_repeated_rounding_pattern(4))
    cross_16 = _parallel_fp32_prefix(_repeated_rounding_pattern(17, shift=1))

    assert first_four[3] > first_four[2]
    assert cross_16[16] > cross_16[15]
    assert np.isfinite(first_four).all()
    assert np.isfinite(cross_16).all()


def _require_cuda():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("the production Triton GDN chunk path requires CUDA")
    return torch


def _run_production_chunk(raw_gates: np.ndarray):
    torch = _require_cuda()
    from sglang.kernels.ops.attention.fla.chunk import chunk_gated_delta_rule
    from sglang.kernels.ops.attention.fla.cumsum import chunk_local_cumsum

    device = torch.device("cuda")
    key_heads, value_heads, key_dim, value_dim = 16, 48, 128, 128

    q = torch.zeros(
        (1, CHUNK_SIZE, key_heads, key_dim),
        dtype=torch.bfloat16,
        device=device,
    )
    k = torch.zeros_like(q)
    q[..., 0] = 1
    k[..., 0] = 1
    v = torch.zeros(
        (1, CHUNK_SIZE, value_heads, value_dim),
        dtype=torch.bfloat16,
        device=device,
    )
    v[:, 2, :, 0] = 1
    g = (
        torch.from_numpy(np.asarray(raw_gates, dtype=np.float32).copy())
        .to(device)
        .view(1, CHUNK_SIZE, 1)
        .expand(-1, -1, value_heads)
        .contiguous()
    )
    beta = torch.full(
        (1, CHUNK_SIZE, value_heads), 0.5, dtype=torch.float32, device=device
    )
    cu_seqlens = torch.tensor([0, CHUNK_SIZE], dtype=torch.long, device=device)
    state = torch.zeros(
        (2, value_heads, value_dim, key_dim),
        dtype=torch.float32,
        device=device,
    )
    state_indices = torch.tensor([1], dtype=torch.int32, device=device)

    immutable = {
        "q": (q, q.clone()),
        "k": (k, k.clone()),
        "v": (v, v.clone()),
        "g": (g, g.clone()),
        "beta": (beta, beta.clone()),
        "cu_seqlens": (cu_seqlens, cu_seqlens.clone()),
        "state_indices": (state_indices, state_indices.clone()),
        "unused_state_slot": (state[0], state[0].clone()),
    }

    prefix = chunk_local_cumsum(g, chunk_size=CHUNK_SIZE, cu_seqlens=cu_seqlens)
    output, _, chunk_states = chunk_gated_delta_rule(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        initial_state=state,
        initial_state_indices=state_indices,
        cu_seqlens=cu_seqlens,
        head_first=False,
        use_qk_l2norm_in_kernel=True,
    )

    for name, (actual, before) in immutable.items():
        assert torch.equal(actual, before), f"production chunk mutated {name}"
    assert torch.isfinite(prefix).all()
    assert torch.isfinite(output).all()
    assert torch.isfinite(chunk_states).all()
    assert torch.isfinite(state).all()
    return output, prefix


def _run_production_varlen(gate_sequences: tuple[np.ndarray, ...]):
    torch = _require_cuda()
    from sglang.kernels.ops.attention.fla.chunk import chunk_gated_delta_rule
    from sglang.kernels.ops.attention.fla.cumsum import chunk_local_cumsum

    assert tuple(gates.size for gates in gate_sequences) == BOUNDARY_LENGTHS
    device = torch.device("cuda")
    key_heads, value_heads, key_dim, value_dim = 16, 48, 128, 128
    total_tokens = sum(BOUNDARY_LENGTHS)

    q = torch.zeros(
        (1, total_tokens, key_heads, key_dim),
        dtype=torch.bfloat16,
        device=device,
    )
    k = torch.zeros_like(q)
    q[..., 0] = 1
    k[..., 0] = 1
    v = torch.zeros(
        (1, total_tokens, value_heads, value_dim),
        dtype=torch.bfloat16,
        device=device,
    )
    sequence_starts = np.cumsum((0,) + BOUNDARY_LENGTHS[:-1])
    for start, impulse in zip(sequence_starts, BOUNDARY_IMPULSES, strict=True):
        v[:, int(start) + impulse, :, 0] = 1

    packed_gates = np.concatenate(gate_sequences)
    g = (
        torch.from_numpy(packed_gates.copy())
        .to(device)
        .view(1, total_tokens, 1)
        .expand(-1, -1, value_heads)
        .contiguous()
    )
    beta = torch.full(
        (1, total_tokens, value_heads), 0.5, dtype=torch.float32, device=device
    )
    cu_seqlens = torch.tensor(
        np.cumsum((0,) + BOUNDARY_LENGTHS), dtype=torch.long, device=device
    )
    state_indices = torch.tensor(
        BOUNDARY_STATE_INDICES, dtype=torch.int32, device=device
    )
    state = torch.zeros(
        (10, value_heads, value_dim, key_dim),
        dtype=torch.float32,
        device=device,
    )
    for slot, carry_in in zip(BOUNDARY_STATE_INDICES, BOUNDARY_CARRY_IN, strict=True):
        state[slot, :, 0, 0] = carry_in

    state_before = state.clone()
    immutable = {
        "q": (q, q.clone()),
        "k": (k, k.clone()),
        "v": (v, v.clone()),
        "g": (g, g.clone()),
        "beta": (beta, beta.clone()),
        "cu_seqlens": (cu_seqlens, cu_seqlens.clone()),
        "state_indices": (state_indices, state_indices.clone()),
    }

    prefix = chunk_local_cumsum(g, chunk_size=CHUNK_SIZE, cu_seqlens=cu_seqlens)
    output, _, chunk_states = chunk_gated_delta_rule(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        initial_state=state,
        initial_state_indices=state_indices,
        cu_seqlens=cu_seqlens,
        head_first=False,
        use_qk_l2norm_in_kernel=True,
    )

    for name, (actual, before) in immutable.items():
        assert torch.equal(actual, before), f"production chunk mutated {name}"
    unused_slots = torch.ones(state.shape[0], dtype=torch.bool, device=device)
    unused_slots[state_indices.long()] = False
    assert torch.equal(state[unused_slots], state_before[unused_slots])
    assert not torch.equal(
        state[state_indices.long()], state_before[state_indices.long()]
    )
    assert torch.isfinite(prefix).all()
    assert torch.isfinite(output).all()
    assert torch.isfinite(chunk_states).all()
    assert torch.isfinite(state).all()
    return output, prefix, state, state_indices


def _assert_matches_fp64_oracle(output, raw_gates, *, prefix=None):
    torch = _require_cuda()
    scale = 128**-0.5
    expected_one_head, _ = _one_head_witness(raw_gates, scale=scale)
    expected = (
        torch.from_numpy(expected_one_head)
        .to(device=output.device, dtype=torch.float32)
        .unsqueeze(0)
        .expand(-1, -1, 48, -1)
    )
    message = ""
    if prefix is not None:
        message = f"; actual prefix[:4]={prefix[0, :4, 0].float().cpu().tolist()}"
    torch.testing.assert_close(
        output.float(),
        expected,
        atol=6e-4,
        rtol=2e-2,
        msg=lambda msg: msg + message,
    )


def _packed_match_result(output, state, state_indices, gate_sequences):
    torch = _require_cuda()
    expected_output, expected_final_state = _packed_one_head_oracle(
        gate_sequences, scale=128**-0.5
    )
    expected_output = (
        torch.from_numpy(expected_output)
        .to(device=output.device, dtype=torch.float32)
        .unsqueeze(0)
        .expand(-1, -1, 48, -1)
    )
    expected_final_state = (
        torch.from_numpy(expected_final_state)
        .to(device=state.device, dtype=torch.float32)
        .expand(-1, 48, -1, -1)
    )
    selected_state = state[state_indices.long()]
    output_close = torch.allclose(output.float(), expected_output, atol=6e-4, rtol=2e-2)
    state_close = torch.allclose(
        selected_state, expected_final_state, atol=6e-4, rtol=2e-2
    )
    return {
        "output_close": output_close,
        "state_close": state_close,
        "output_max_abs_error": (output.float() - expected_output).abs().max().item(),
        "state_max_abs_error": (selected_state - expected_final_state)
        .abs()
        .max()
        .item(),
    }


def _all_nonincreasing(values):
    return (values[1:] <= values[:-1]).all().item()


def test_cuda_varlen_boundaries_match_oracle_for_finite_monotone_control():
    gate_sequences = tuple(
        np.full(length, -0.25, dtype=np.float32) for length in BOUNDARY_LENGTHS
    )
    output, prefix, state, state_indices = _run_production_varlen(gate_sequences)

    start = 0
    for length in BOUNDARY_LENGTHS:
        for chunk_start in range(start, start + length, CHUNK_SIZE):
            chunk_end = min(chunk_start + CHUNK_SIZE, start + length)
            assert _all_nonincreasing(prefix[0, chunk_start:chunk_end, 0])
        start += length
    result = _packed_match_result(output, state, state_indices, gate_sequences)
    assert result["output_close"] and result["state_close"], result


def test_cuda_varlen_rounding_reversals_cross_subchunk_and_chunk_boundaries():
    gate_sequences = (
        _repeated_rounding_pattern(4),
        _repeated_rounding_pattern(17, shift=1),
        _repeated_rounding_pattern(67),
        _repeated_rounding_pattern(129),
    )
    output, prefix, state, state_indices = _run_production_varlen(gate_sequences)

    # These two reversals are observed from the production CUDA cumsum. The
    # output and final-state comparison below still covers every sequence,
    # including the 67- and 129-token cross-chunk cases, unconditionally.
    reversal_indices = (3, 20)
    observed_reversals = [
        prefix[0, index, 0] > prefix[0, index - 1, 0] for index in reversal_indices
    ]
    if not all(bool(value.item()) for value in observed_reversals):
        observed = prefix[0, :, 0].float().cpu()
        pytest.skip(
            "selected CUDA cumsum missed a required finite boundary reversal: "
            f"edges={[(i, observed[i - 1 : i + 1].tolist()) for i in reversal_indices]}"
        )

    result = _packed_match_result(output, state, state_indices, gate_sequences)
    assert result["output_close"] and result["state_close"], result


def test_cuda_production_chunk_matches_oracle_when_prefix_rounding_reverses():
    output, prefix = _run_production_chunk(RAW_GATES)
    observed = prefix[0, :4, 0].float().cpu().numpy()
    if not observed[3] > observed[2]:
        pytest.skip(
            "selected CUDA cumsum did not exhibit the finite rounding reversal: "
            f"prefix[:4]={observed.tolist()}"
        )

    _assert_matches_fp64_oracle(output, RAW_GATES, prefix=prefix)

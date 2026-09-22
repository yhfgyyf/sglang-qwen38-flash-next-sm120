"""Parity checks for Qwen3.5 GDN decode projection/Conv1D fusion."""

import pytest
import torch

from sglang.kernels.ops.attention.triton_gdn_fused_proj import (
    fused_qkvzba_causal_conv1d_update_contiguous,
    fused_qkvzba_split_reshape_cat_contiguous,
)
from sglang.kernels.ops.mamba.causal_conv1d_triton import causal_conv1d_update
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=15, stage="base-b", runner_config="1-gpu-large")


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is not None,
    reason="CUDA-only GDN fusion",
)
@pytest.mark.parametrize("cache_indices", [[0], [0, 2, 4, 6], [0, -1, 4, 6]])
def test_gdn_fused_decode_projection_conv_matches_reference(cache_indices):
    torch.manual_seed(1234)
    q_heads, v_heads, q_dim, v_dim, width = 16, 48, 128, 128, 4
    packed_qkv_dim = 2 * q_heads * q_dim + v_heads * v_dim
    packed_z_dim = v_heads * v_dim
    batch = len(cache_indices)
    qkvz = torch.randn(
        batch, packed_qkv_dim + packed_z_dim, device="cuda", dtype=torch.bfloat16
    )
    ba = torch.randn(batch, 2 * v_heads, device="cuda", dtype=torch.bfloat16)
    state = torch.randn(
        8, packed_qkv_dim, width - 1, device="cuda", dtype=torch.bfloat16
    )
    weight = torch.randn(packed_qkv_dim, width, device="cuda", dtype=torch.bfloat16)
    bias = torch.randn(packed_qkv_dim, device="cuda", dtype=torch.bfloat16)
    indices = torch.tensor(cache_indices, device="cuda", dtype=torch.int32)

    reference_state = state.clone()
    reference_qkv, reference_z, reference_b, reference_a = (
        fused_qkvzba_split_reshape_cat_contiguous(
            qkvz, ba, q_heads, v_heads, q_dim, v_dim
        )
    )
    reference_qkv = causal_conv1d_update(
        reference_qkv,
        reference_state,
        weight,
        bias,
        "silu",
        conv_state_indices=indices,
    )
    actual_qkv, actual_z, actual_b, actual_a = (
        fused_qkvzba_causal_conv1d_update_contiguous(
            qkvz,
            ba,
            state,
            weight,
            bias,
            indices,
            qkv_dim=packed_qkv_dim,
            v_dim=packed_z_dim,
            num_v_heads=v_heads,
            head_v_dim=v_dim,
            activation="silu",
        )
    )

    # The reference kernel leaves a padded row's output uninitialized; GDN
    # skips that row downstream. Only active rows have defined Conv1D output.
    active = indices >= 0
    torch.testing.assert_close(
        actual_qkv[active], reference_qkv[active], rtol=0, atol=0
    )
    for actual, reference in (
        (actual_z, reference_z),
        (actual_b, reference_b),
        (actual_a, reference_a),
        (state, reference_state),
    ):
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)

"""Experimental original-BF16 small-M linear kernels for SM120.

The weights and activations stay BF16.  ``tl.dot`` accumulates into FP32 and
the epilogue rounds once to BF16.  The split-K path writes private FP32
partials and reduces them in a fixed order; it deliberately uses no atomics.

This module is a bounded prototype.  Callers must benchmark their exact
shape before choosing it over the framework linear implementation.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch
import triton
import triton.language as tl


@dataclass(frozen=True)
class BF16SmallMGemmConfig:
    block_m: int = 16
    block_n: int = 64
    block_k: int = 64
    num_warps: int = 4
    num_stages: int = 3
    split_k: int = 1

    def validate(self) -> None:
        for name in ("block_m", "block_n", "block_k"):
            value = getattr(self, name)
            if value <= 0 or value & (value - 1):
                raise ValueError(f"{name} must be a positive power of two, got {value}")
        if self.block_m < 16 or self.block_n < 16 or self.block_k < 16:
            raise ValueError("BF16 tensor-core tiles must be at least 16 in every axis")
        if self.num_warps not in (1, 2, 4, 8):
            raise ValueError(f"num_warps must be 1, 2, 4, or 8, got {self.num_warps}")
        if self.num_stages <= 0:
            raise ValueError(f"num_stages must be positive, got {self.num_stages}")
        if self.split_k <= 0:
            raise ValueError(f"split_k must be positive, got {self.split_k}")


DEFAULT_BF16_SMALL_M_CONFIG = BF16SmallMGemmConfig()


@triton.jit
def _bf16_small_m_gemm_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    stride_xm,
    stride_wn,
    stride_om,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    PIPELINE_STAGES: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in tl.range(0, K, BLOCK_K, num_stages=PIPELINE_STAGES):
        k = k0 + offs_k
        x = tl.load(
            x_ptr + offs_m[:, None] * stride_xm + k[None, :],
            mask=(offs_m[:, None] < M) & (k[None, :] < K),
            other=0.0,
        )
        weight = tl.load(
            weight_ptr + offs_n[None, :] * stride_wn + k[:, None],
            mask=(offs_n[None, :] < N) & (k[:, None] < K),
            other=0.0,
        )
        acc = tl.dot(x, weight, acc)

    if HAS_BIAS:
        bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
        acc += bias[None, :].to(tl.float32)
    tl.store(
        out_ptr + offs_m[:, None] * stride_om + offs_n[None, :],
        acc,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


@triton.jit
def _bf16_small_m_splitk_kernel(
    x_ptr,
    weight_ptr,
    partial_ptr,
    stride_xm,
    stride_wn,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SPLIT_K: tl.constexpr,
    PIPELINE_STAGES: tl.constexpr,
):
    pid = tl.program_id(0)
    split_id = tl.program_id(1)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    num_k_blocks = tl.cdiv(K, BLOCK_K)
    blocks_per_split = tl.cdiv(num_k_blocks, SPLIT_K)
    first_block = split_id * blocks_per_split
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for local_block in tl.range(0, blocks_per_split, 1, num_stages=PIPELINE_STAGES):
        block = first_block + local_block
        k = block * BLOCK_K + offs_k
        valid_k = (block < num_k_blocks) & (k < K)
        x = tl.load(
            x_ptr + offs_m[:, None] * stride_xm + k[None, :],
            mask=(offs_m[:, None] < M) & valid_k[None, :],
            other=0.0,
        )
        weight = tl.load(
            weight_ptr + offs_n[None, :] * stride_wn + k[:, None],
            mask=(offs_n[None, :] < N) & valid_k[:, None],
            other=0.0,
        )
        acc = tl.dot(x, weight, acc)

    partial_base = split_id * M * N
    tl.store(
        partial_ptr + partial_base + offs_m[:, None] * N + offs_n[None, :],
        acc,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


@triton.jit
def _bf16_small_m_splitk_reduce_kernel(
    partial_ptr,
    bias_ptr,
    out_ptr,
    stride_om,
    M: tl.constexpr,
    N: tl.constexpr,
    SPLIT_K: tl.constexpr,
    BLOCK: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    total = M * N
    mask = idx < total
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for split_id in tl.static_range(0, SPLIT_K):
        acc += tl.load(
            partial_ptr + split_id * total + idx,
            mask=mask,
            other=0.0,
        )
    col = idx % N
    row = idx // N
    if HAS_BIAS:
        acc += tl.load(bias_ptr + col, mask=mask, other=0.0).to(tl.float32)
    tl.store(out_ptr + row * stride_om + col, acc, mask=mask)


def _validate_linear_inputs(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    out: torch.Tensor | None,
) -> tuple[int, int, int]:
    if not x.is_cuda or not weight.is_cuda:
        raise ValueError("BF16 small-M GEMM requires CUDA tensors")
    if x.device != weight.device:
        raise ValueError(
            f"x and weight must share a device: {x.device}, {weight.device}"
        )
    if x.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16:
        raise TypeError(f"x and weight must stay BF16, got {x.dtype=} {weight.dtype=}")
    if x.ndim != 2 or weight.ndim != 2:
        raise ValueError(
            f"x and weight must be matrices, got {x.shape=} {weight.shape=}"
        )
    m, k = x.shape
    n, weight_k = weight.shape
    if weight_k != k:
        raise ValueError(
            f"incompatible linear shapes: x={x.shape}, weight={weight.shape}"
        )
    if x.stride(1) != 1 or weight.stride(1) != 1:
        raise ValueError("x and weight must be contiguous along K")
    if not weight.is_contiguous():
        raise ValueError("weight must be contiguous [N, K]")
    if bias is not None:
        if bias.device != x.device or bias.dtype not in (torch.bfloat16, torch.float32):
            raise TypeError("bias must be BF16 or FP32 on the same CUDA device")
        if bias.ndim != 1 or bias.numel() != n or bias.stride(0) != 1:
            raise ValueError(f"bias must be contiguous [N], got {bias.shape=}")
    if out is not None:
        if out.device != x.device or out.dtype != torch.bfloat16:
            raise TypeError("out must be BF16 on the same CUDA device")
        if out.shape != (m, n) or out.stride(1) != 1:
            raise ValueError(
                f"out must have shape {(m, n)} and contiguous columns, got "
                f"shape={out.shape}, stride={out.stride()}"
            )
    return m, n, k


def bf16_small_m_workspace_shape(
    x: torch.Tensor, weight: torch.Tensor, split_k: int
) -> tuple[int, int, int]:
    if x.ndim != 2 or weight.ndim != 2:
        raise ValueError("x and weight must be matrices")
    if split_k <= 1:
        return (0, 0, 0)
    return (int(split_k), int(x.shape[0]), int(weight.shape[0]))


def bf16_small_m_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    out: torch.Tensor | None = None,
    workspace: torch.Tensor | None = None,
    config: BF16SmallMGemmConfig = DEFAULT_BF16_SMALL_M_CONFIG,
    split_k: int | None = None,
) -> torch.Tensor:
    """Compute ``x @ weight.T + bias`` without changing BF16 weights.

    ``x`` may have a padded row stride; columns must remain contiguous.
    ``out`` may likewise have a padded row stride.  For CUDA-graph capture,
    supply both ``out`` and the FP32 ``workspace`` required by split-K.
    """

    if split_k is not None:
        config = replace(config, split_k=int(split_k))
    config.validate()
    m, n, k = _validate_linear_inputs(x, weight, bias, out)
    if out is None:
        out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
    if m == 0 or n == 0:
        return out

    bias_arg = bias if bias is not None else out
    grid = (triton.cdiv(m, config.block_m) * triton.cdiv(n, config.block_n),)
    if config.split_k == 1:
        _bf16_small_m_gemm_kernel[grid](
            x,
            weight,
            bias_arg,
            out,
            x.stride(0),
            weight.stride(0),
            out.stride(0),
            M=m,
            N=n,
            K=k,
            BLOCK_M=config.block_m,
            BLOCK_N=config.block_n,
            BLOCK_K=config.block_k,
            PIPELINE_STAGES=config.num_stages,
            HAS_BIAS=bias is not None,
            num_warps=config.num_warps,
            num_stages=config.num_stages,
        )
        return out

    required = config.split_k * m * n
    if workspace is None:
        workspace = torch.empty(required, dtype=torch.float32, device=x.device)
    if (
        workspace.device != x.device
        or workspace.dtype != torch.float32
        or not workspace.is_contiguous()
        or workspace.numel() < required
    ):
        raise ValueError(
            "split-K workspace must be contiguous FP32 on x.device with at least "
            f"{required} elements"
        )
    partial = workspace.reshape(-1)[:required]
    split_grid = (grid[0], config.split_k)
    _bf16_small_m_splitk_kernel[split_grid](
        x,
        weight,
        partial,
        x.stride(0),
        weight.stride(0),
        M=m,
        N=n,
        K=k,
        BLOCK_M=config.block_m,
        BLOCK_N=config.block_n,
        BLOCK_K=config.block_k,
        SPLIT_K=config.split_k,
        PIPELINE_STAGES=config.num_stages,
        num_warps=config.num_warps,
        num_stages=config.num_stages,
    )
    reduce_block = 256
    _bf16_small_m_splitk_reduce_kernel[(triton.cdiv(m * n, reduce_block),)](
        partial,
        bias_arg,
        out,
        out.stride(0),
        M=m,
        N=n,
        SPLIT_K=config.split_k,
        BLOCK=reduce_block,
        HAS_BIAS=bias is not None,
        num_warps=4,
    )
    return out


__all__ = [
    "BF16SmallMGemmConfig",
    "DEFAULT_BF16_SMALL_M_CONFIG",
    "bf16_small_m_linear",
    "bf16_small_m_workspace_shape",
]

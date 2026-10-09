"""CPU-only behavioral tests for the opt-in current-profile PDL policy."""

from __future__ import annotations

from contextlib import nullcontext
import os
from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from sglang.kernels.jit.utils import arch
from sglang.srt.environ import envs
from sglang.srt.layers.attention import qwen_sparse_attn_backend as qsa
from sglang.srt.layers.moe.moe_runner import flashinfer_cutlass as cutlass
from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig


PDL_ENV_NAMES = (
    "SGLANG_JIT_DISABLE_PDL",
    "SGLANG_FLASHINFER_CUTLASS_DISABLE_PDL",
    "SGLANG_QSA_TRTLLM_DISABLE_PDL",
)


def test_pdl_policy_environment_defaults_are_off(monkeypatch):
    for name in PDL_ENV_NAMES:
        monkeypatch.delenv(name, raising=False)
        assert getattr(envs, name).get() is False


@pytest.mark.parametrize(
    "major,hip,musa,expected",
    [
        (8, False, False, False),
        (9, False, False, True),
        (12, False, False, True),
        (12, True, False, False),
        (12, False, True, False),
    ],
)
@pytest.mark.parametrize("disabled", [False, True])
def test_jit_pdl_selector_platform_matrix(
    monkeypatch, major, hip, musa, expected, disabled
):
    monkeypatch.setenv("SGLANG_JIT_DISABLE_PDL", str(int(disabled)))
    monkeypatch.setattr(arch, "is_hip_runtime", lambda: hip)
    monkeypatch.setattr(arch, "is_musa_runtime", lambda: musa)
    monkeypatch.setattr(
        arch,
        "get_jit_cuda_arch",
        lambda: arch.ArchInfo(major=major, minor=0, suffix=""),
    )

    selector = arch.is_arch_support_pdl.__wrapped__
    assert selector() is (expected and not disabled)


@pytest.mark.parametrize(
    "first_value,expected",
    [("0", "True\nTrue"), ("1", "False\nFalse")],
)
def test_jit_pdl_selector_is_fixed_at_first_use_in_fresh_process(first_value, expected):
    script = """
import os
from sglang.kernels.jit.utils import arch
arch._CUDA_ARCH = arch.ArchInfo(major=12, minor=0, suffix='f')
arch._init_jit_cuda_arch_once = lambda: None
print(arch.is_arch_support_pdl())
os.environ['SGLANG_JIT_DISABLE_PDL'] = '0' if os.environ['SGLANG_JIT_DISABLE_PDL'] == '1' else '1'
print(arch.is_arch_support_pdl())
"""
    environment = dict(os.environ)
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": "",
            "PYTHONDONTWRITEBYTECODE": "1",
            "SGLANG_JIT_DISABLE_PDL": first_value,
        }
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[3],
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.strip() == expected


def _common_cutlass_case(quant_type: str):
    hidden_states = torch.arange(8, dtype=torch.bfloat16).view(2, 4)
    dispatch_output = SimpleNamespace(
        hidden_states=hidden_states,
        hidden_states_scale=None,
        topk_output=SimpleNamespace(
            topk_ids=torch.tensor([[0], [0]], dtype=torch.int64),
            topk_weights=torch.tensor([[0.75], [0.25]], dtype=torch.float32),
        ),
    )
    if quant_type == "fp4":
        weights = torch.arange(8, dtype=torch.uint8).view(1, 8)
        quant_scales = [
            torch.tensor([1.0]),
            torch.arange(4, dtype=torch.uint8),
            torch.tensor([2.0]),
            torch.tensor([3.0]),
            torch.arange(4, dtype=torch.uint8),
            torch.tensor([4.0]),
        ]
    else:
        weights = torch.ones((1, 2, 2), dtype=torch.bfloat16)
        quant_scales = None
    quant_info = cutlass.FlashInferCutlassMoeQuantInfo(
        quant_type=quant_type,
        w13_weight=weights,
        w2_weight=weights,
        quant_scales=quant_scales,
        apply_routed_scaling_factor=False,
    )
    return (
        dispatch_output,
        quant_info,
        MoeRunnerConfig(),
        torch.zeros_like(hidden_states),
    )


def _assert_same_argument(left, right):
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor)
        assert torch.equal(left, right)
        return
    if isinstance(left, (list, tuple)):
        assert isinstance(right, type(left))
        assert len(left) == len(right)
        for left_item, right_item in zip(left, right):
            _assert_same_argument(left_item, right_item)
        return
    assert left == right


@pytest.mark.parametrize("quant_type", ["bf16", "fp4"])
def test_common_flashinfer_cutlass_boundary_preserves_arguments(
    monkeypatch, quant_type
):
    calls = []

    def fused_moe(**kwargs):
        calls.append(kwargs)
        return (kwargs["output"],)

    monkeypatch.setattr(
        cutlass, "_flashinfer_cutlass_fused_moe", lambda: (fused_moe, object())
    )
    activation = object()
    monkeypatch.setattr(cutlass, "_activation_type", lambda config: activation)
    case = _common_cutlass_case(quant_type)

    monkeypatch.setenv("SGLANG_FLASHINFER_CUTLASS_DISABLE_PDL", "0")
    cutlass._run_flashinfer_cutlass(
        dispatch_output=case[0],
        quant_info=case[1],
        runner_config=case[2],
        output=case[3],
    )
    monkeypatch.setenv("SGLANG_FLASHINFER_CUTLASS_DISABLE_PDL", "1")
    cutlass._run_flashinfer_cutlass(
        dispatch_output=case[0],
        quant_info=case[1],
        runner_config=case[2],
        output=case[3],
    )

    default_call, disabled_call = calls
    assert default_call["enable_pdl"] is None
    assert disabled_call["enable_pdl"] is False
    assert default_call.keys() == disabled_call.keys()
    for name in default_call.keys() - {"enable_pdl"}:
        _assert_same_argument(default_call[name], disabled_call[name])


def test_mxfp4_flashinfer_cutlass_boundary_uses_same_named_policy(monkeypatch):
    calls = []

    class ActivationType:
        Swiglu = object()

    def fused_moe(**kwargs):
        calls.append(kwargs)

    monkeypatch.setattr(
        cutlass,
        "_flashinfer_cutlass_fused_moe",
        lambda: (fused_moe, ActivationType),
    )
    monkeypatch.setattr(cutlass, "get_tp_group", lambda: object())
    monkeypatch.setattr(cutlass, "is_allocation_symmetric", lambda: False)
    monkeypatch.setattr(
        cutlass, "use_symmetric_memory", lambda *args, **kwargs: nullcontext()
    )
    weights = torch.arange(8, dtype=torch.uint8).view(1, 8)
    scales = torch.arange(4, dtype=torch.uint8)
    quant_info = cutlass.FlashInferCutlassMxfp4MoeQuantInfo(
        w13_weight=weights,
        w2_weight=weights,
        w13_weight_scale=scales,
        w2_weight_scale=scales,
    )
    dispatch_output = SimpleNamespace(
        hidden_states=torch.ones((2, 4), dtype=torch.bfloat16),
        topk_output=SimpleNamespace(
            topk_ids=torch.tensor([[0], [0]], dtype=torch.int64),
            topk_weights=torch.ones((2, 1), dtype=torch.float32),
        ),
    )

    monkeypatch.setenv("SGLANG_FLASHINFER_CUTLASS_DISABLE_PDL", "0")
    cutlass.fused_experts_none_to_flashinfer_mxfp4(
        dispatch_output, quant_info, MoeRunnerConfig()
    )
    monkeypatch.setenv("SGLANG_FLASHINFER_CUTLASS_DISABLE_PDL", "1")
    cutlass.fused_experts_none_to_flashinfer_mxfp4(
        dispatch_output, quant_info, MoeRunnerConfig()
    )

    default_call, disabled_call = calls
    assert default_call["enable_pdl"] is None
    assert disabled_call["enable_pdl"] is False
    assert default_call.keys() == disabled_call.keys()
    for name in default_call.keys() - {"enable_pdl", "output"}:
        _assert_same_argument(default_call[name], disabled_call[name])
    assert default_call["output"].shape == disabled_call["output"].shape
    assert default_call["output"].dtype == disabled_call["output"].dtype


def _run_qsa_trtllm_boundary(monkeypatch, *, host: bool, disabled: bool):
    monkeypatch.setenv("SGLANG_QSA_TRTLLM_DISABLE_PDL", str(int(disabled)))

    def fill_valid_counts(sequence_lens, topk_indices, output, batch, topk):
        del sequence_lens, topk_indices, batch
        output.fill_(topk)

    monkeypatch.setattr(qsa, "qwen_sparse_valid_counts_triton", fill_valid_counts)
    monkeypatch.setattr(
        qsa, "qwen_sparse_kv_extraction_compact_triton", lambda *args: None
    )
    fake_host_module = ModuleType("sglang.srt.mem_cache.qwen38_host_kv_pool")
    fake_host_module.selected_host_slots = lambda *args: torch.arange(
        64, dtype=torch.int32
    )
    monkeypatch.setitem(
        sys.modules, "sglang.srt.mem_cache.qwen38_host_kv_pool", fake_host_module
    )

    self = SimpleNamespace(
        _cuda_graph_max_tokens=0,
        _host_slot_scratch={},
        _host_gather_reported=True,
        _trtllm_workspace=None,
        req_to_token_pool=SimpleNamespace(
            req_to_token=torch.arange(128, dtype=torch.int32).view(1, 128)
        ),
        token_to_kv_pool=SimpleNamespace(full_attention_layer_id_mapping={0: 0}),
        _get_trtllm_sparse_tables=lambda batch, pages, page, device: (
            torch.arange(batch + 1, dtype=torch.int32) * pages * page,
            torch.zeros((batch, pages), dtype=torch.int32),
        ),
        _get_fa2_scratch=lambda capacity, heads, dim, dtype, device, is_graph=False: (
            torch.zeros((capacity, heads, dim), dtype=dtype),
            torch.zeros((capacity, heads, dim), dtype=dtype),
        ),
        _kv_scales=lambda layer: (1.0, 1.0),
    )
    host_pool = None
    k_buffer = torch.zeros((128, 1, 2), dtype=torch.bfloat16)
    v_buffer = torch.zeros_like(k_buffer)
    if host:
        host_pool = SimpleNamespace(
            head_num=1,
            head_dim=2,
            dtype=torch.bfloat16,
            gather_selected=lambda *args: None,
        )
        k_buffer = v_buffer = None
    call = {}

    def trtllm_decode(**kwargs):
        call.update(kwargs)
        return kwargs["query"]

    output = qsa.QwenSparseAttnBackend._forward_trtllm_sparse(
        self,
        torch.ones((1, 1, 2), dtype=torch.bfloat16),
        k_buffer,
        v_buffer,
        SimpleNamespace(layer_id=0, scaling=0.5),
        SimpleNamespace(req_pool_indices=torch.tensor([0], dtype=torch.int32)),
        SimpleNamespace(
            sequence_lengths=torch.tensor([2], dtype=torch.int32),
            row_req_pool_indices=None,
            is_cuda_graph=False,
        ),
        torch.tensor([[0, 1]], dtype=torch.int32),
        trtllm_decode,
        host_pool=host_pool,
    )
    assert output.shape == (1, 2)
    return call


@pytest.mark.parametrize("host", [False, True])
def test_qsa_trtllm_shared_boundary_preserves_host_and_gpu_arguments(monkeypatch, host):
    default_call = _run_qsa_trtllm_boundary(monkeypatch, host=host, disabled=False)
    disabled_call = _run_qsa_trtllm_boundary(monkeypatch, host=host, disabled=True)

    assert default_call["enable_pdl"] is None
    assert disabled_call["enable_pdl"] is False
    assert default_call.keys() == disabled_call.keys()
    for name in default_call.keys() - {"enable_pdl", "workspace_buffer"}:
        _assert_same_argument(default_call[name], disabled_call[name])
    assert (
        default_call["workspace_buffer"].shape
        == disabled_call["workspace_buffer"].shape
    )

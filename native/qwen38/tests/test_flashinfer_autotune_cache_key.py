"""CPU-only regression tests for FlashInfer autotune cache isolation."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


def _load_flashinfer_autotune_module():
    source = (
        Path(__file__).resolve().parents[3]
        / "python/sglang/srt/model_executor/runner/flashinfer_autotune.py"
    )
    spec = importlib.util.spec_from_file_location(
        "q38_test_flashinfer_autotune", source
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def cache_path(monkeypatch, tmp_path):
    module = _load_flashinfer_autotune_module()
    monkeypatch.setitem(
        sys.modules, "flashinfer", SimpleNamespace(__version__="test-version")
    )
    monkeypatch.setattr(
        module.torch.cuda, "get_device_capability", lambda device: (12, 0)
    )
    monkeypatch.setenv("SGLANG_CACHE_DIR", str(tmp_path))

    def call(
        *,
        fused_finalize: bool,
        model_path: str = "qwen38-test",
        skip_ops: tuple[str, ...] = (),
        is_draft_worker: bool = False,
    ) -> Path:
        monkeypatch.setenv(
            "SGLANG_FLASHINFER_MOE_FUSED_FINALIZE",
            "1" if fused_finalize else "0",
        )
        model = SimpleNamespace(
            model_path=model_path,
            quantization="modelopt_fp4",
        )
        execution = SimpleNamespace(
            moe=SimpleNamespace(moe_runner_backend="flashinfer_cutlass"),
            kernel=SimpleNamespace(flashinfer_autotune_skip_ops=skip_ops),
        )
        monkeypatch.setattr(module, "get_model", lambda: model)
        monkeypatch.setattr(module, "get_exec", lambda: execution)

        runner = SimpleNamespace(
            device="cuda",
            dtype=torch.bfloat16,
            is_draft_worker=is_draft_worker,
            model_config=SimpleNamespace(
                quantization="modelopt_fp4",
                hf_config=SimpleNamespace(),
            ),
            ps=SimpleNamespace(
                tp_size=2,
                pp_size=1,
                attn_dp_size=1,
                moe_ep_size=2,
                tp_rank=0,
                pp_rank=0,
                dp_rank=0,
            ),
        )
        return module.flashinfer_autotune_cache_path(runner)

    return call


def test_fused_finalize_modes_use_distinct_stable_cache_paths(cache_path):
    fused_path = cache_path(fused_finalize=True)
    unfused_path = cache_path(fused_finalize=False)

    assert fused_path == cache_path(fused_finalize=True)
    assert unfused_path == cache_path(fused_finalize=False)
    assert fused_path != unfused_path


@pytest.mark.parametrize(
    "changed_input",
    [
        {"model_path": "different-model"},
        {"skip_ops": ("fused_moe",)},
        {"is_draft_worker": True},
    ],
)
def test_existing_cache_key_inputs_remain_isolated(cache_path, changed_input):
    baseline = cache_path(fused_finalize=True)
    changed = cache_path(fused_finalize=True, **changed_input)

    assert changed != baseline

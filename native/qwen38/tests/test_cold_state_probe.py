"""Contracts for the pre-GDN cold SSM-state diagnostic hook."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


def _load_probe():
    path = Path(__file__).resolve().parents[1] / "cold_state_probe.py"
    spec = importlib.util.spec_from_file_location("cold_state_probe", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


probe = _load_probe()


def _module(name="Qwen3_5GatedDeltaNet", layer_id=0):
    return type(name, (torch.nn.Module,), {"layer_id": layer_id})()


def _mode(name):
    mode_type = type(
        "ForwardMode",
        (),
        {"__module__": "sglang.srt.model_executor.forward_batch_info"},
    )
    mode = mode_type()
    mode.name = name
    return mode


def _cpu_batch(
    *,
    batch_size=4,
    prefixes=(0, 7, 0, 0),
    lengths=(2, 3, 0, 5),
    original_batch_size=None,
):
    return SimpleNamespace(
        forward_mode=_mode("EXTEND"),
        batch_size=batch_size,
        _original_batch_size=original_batch_size,
        extend_prefix_lens_cpu=list(prefixes),
        extend_seq_lens_cpu=list(lengths),
    )


def _make_hook(tmp_path, **config):
    arm = tmp_path / "arm"
    sample_dir = tmp_path / "samples"
    hook = probe.make_cold_state_hook(
        {
            "arm_file": str(arm),
            "sample_dir": str(sample_dir),
            "min_rows": 8,
            "max_records": 3,
            **config,
        }
    )
    return hook, arm, hook.output_dir


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("min_rows", 0),
        ("min_rows", True),
        ("max_records", 0),
        ("max_records", 65),
        ("max_records", True),
        ("arm_file", ""),
        ("sample_dir", ""),
    ],
)
def test_config_is_strict_and_bounded(tmp_path, field, value):
    config = {
        "arm_file": str(tmp_path / "arm"),
        "sample_dir": str(tmp_path / "samples"),
        field: value,
    }
    with pytest.raises(ValueError, match=field):
        probe.make_cold_state_hook(config)


def test_factory_instances_get_unique_subdirs_under_one_base(tmp_path):
    sample_dir = tmp_path / "samples"
    config = {"arm_file": str(tmp_path / "arm"), "sample_dir": str(sample_dir)}
    first = probe.make_cold_state_hook(config)
    second = probe.make_cold_state_hook(config)
    try:
        assert first.output_dir.parent == sample_dir
        assert second.output_dir.parent == sample_dir
        assert first.output_dir != second.output_dir
        assert first.output_dir.is_dir()
        assert second.output_dir.is_dir()
    finally:
        first.close()
        second.close()


def test_arm_file_is_polled_only_until_observed(tmp_path):
    hook, arm, _ = _make_hook(tmp_path)
    try:
        assert hook._is_armed() is False
        arm.touch()
        assert hook._is_armed() is True
        arm.unlink()
        assert hook._is_armed() is True
    finally:
        hook.close()


def test_writer_stays_idle_until_explicit_partial_flush(tmp_path):
    hook, _, sample_dir = _make_hook(tmp_path)
    assert hook._writer_gate.is_set() is False
    assert (sample_dir / "cold-state.json").exists() is False

    hook.flush()

    payload = json.loads((sample_dir / "cold-state.json").read_text())
    assert payload["expected_records"] == 3
    assert payload["persisted_records"] == 0
    assert payload["coverage"] == {
        "status": "partial",
        "cap_reached": False,
        "bounded_cap": 3,
        "finalized_by": "flush",
        "all_records_valid": True,
    }
    hook.close()


@pytest.mark.parametrize(
    ("module", "hidden_rows", "mode", "selected"),
    [
        (_module(), 8, "EXTEND", True),
        (_module(layer_id=1), 8, "EXTEND", False),
        (_module("Other"), 8, "EXTEND", False),
        (_module(), 7, "EXTEND", False),
        (_module(), 8, "DECODE", False),
        (_module(), 8, "TARGET_VERIFY", False),
        (_module(), 8, "DRAFT_EXTEND_V2", False),
        (_module(), 8, "IDLE", False),
    ],
)
def test_selection_is_narrow(tmp_path, module, hidden_rows, mode, selected):
    hook, _, _ = _make_hook(tmp_path)
    hidden = torch.empty((hidden_rows, 1))
    batch = _cpu_batch()
    batch.forward_mode = _mode(mode)
    try:
        assert hook._selected(module, (hidden, batch)) is selected
    finally:
        hook.close()


def test_cold_rows_use_cpu_metadata_and_exclude_padding(tmp_path):
    hook, _, _ = _make_hook(tmp_path)
    batch = _cpu_batch(original_batch_size=3)
    try:
        assert hook._cold_rows(batch) == [(0, 0, 2)]
    finally:
        hook.close()


@pytest.mark.parametrize(
    "mutate,match",
    [
        (lambda fb: setattr(fb, "extend_prefix_lens_cpu", None), "prefix"),
        (lambda fb: setattr(fb, "extend_seq_lens_cpu", [1]), "extend"),
        (lambda fb: setattr(fb, "batch_size", 5), "coverage"),
        (lambda fb: setattr(fb, "_original_batch_size", 5), "original"),
        (lambda fb: fb.extend_prefix_lens_cpu.__setitem__(0, True), "integer"),
    ],
)
def test_cpu_metadata_must_have_explicit_coverage(tmp_path, mutate, match):
    hook, _, _ = _make_hook(tmp_path)
    batch = _cpu_batch()
    mutate(batch)
    try:
        with pytest.raises(RuntimeError, match=match):
            hook._cold_rows(batch)
    finally:
        hook.close()


def test_unarmed_call_never_resolves_state(tmp_path, monkeypatch):
    hook, _, _ = _make_hook(tmp_path, min_rows=1)
    hidden = torch.empty((1, 1))
    batch = _cpu_batch(batch_size=1, prefixes=(0,), lengths=(1,))
    monkeypatch.setattr(
        hook,
        "_resolve_runtime",
        lambda *_: pytest.fail("unarmed hook resolved the SSM cache"),
    )
    try:
        assert hook(_module(), (hidden, batch)) is None
    finally:
        hook.close()


def test_compiler_tracing_skips_before_runtime_resolution(tmp_path, monkeypatch):
    hook, arm, _ = _make_hook(tmp_path, min_rows=1)
    arm.touch()
    hidden = torch.empty((1, 1))
    batch = _cpu_batch(batch_size=1, prefixes=(0,), lengths=(1,))
    monkeypatch.setattr(probe, "_is_compiling_or_tracing", lambda: True)
    monkeypatch.setattr(
        hook,
        "_resolve_runtime",
        lambda *_: pytest.fail("compiler tracing resolved the SSM cache"),
    )
    try:
        assert hook(_module(), (hidden, batch)) is None
    finally:
        hook.close()


def test_hicache_transfer_gated_pool_is_explicitly_unsupported(tmp_path):
    hook, _, _ = _make_hook(tmp_path)
    pool = SimpleNamespace(layer_transfer_counter=object())
    linear = SimpleNamespace(req_to_token_pool=pool)
    try:
        with pytest.raises(RuntimeError, match="HiCache"):
            hook._bind_static_state(linear, 0)
    finally:
        hook.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="probe kernel needs CUDA")
def test_kernel_covers_last_slot_head_key_and_value_and_preserves_input(tmp_path):
    state = torch.zeros((11, 48, 128, 128), dtype=torch.float32, device="cuda")
    state[10, 47, 127, 127] = 9
    before = state.clone()
    summary = probe._run_scan_for_test(
        state,
        slot=10,
        request_index=5,
        gpu_prefix=0,
        gpu_extend=7,
        expected_prefix=0,
        expected_extend=7,
        request_pool_size=5,
    )
    assert summary["coverage_valid"] is True
    assert summary["nonzero_count"] == 1
    assert summary["nonfinite_count"] == 0
    assert torch.equal(state, before)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="probe kernel needs CUDA")
def test_kernel_full_scan_signed_zero_nan_and_both_infinities(tmp_path):
    state = torch.zeros((11, 48, 128, 128), dtype=torch.float32, device="cuda")
    zero = probe._run_scan_for_test(
        state,
        slot=1,
        request_index=1,
        gpu_prefix=0,
        gpu_extend=1,
        expected_prefix=0,
        expected_extend=1,
        request_pool_size=1,
    )
    assert zero["nonzero_count"] == 0
    assert zero["nonfinite_count"] == 0

    state[1, 47, 127, 124] = -0.0
    state[1, 47, 127, 125] = float("nan")
    state[1, 47, 127, 126] = float("inf")
    state[1, 47, 127, 127] = -float("inf")
    before = state.clone()
    summary = probe._run_scan_for_test(
        state,
        slot=1,
        request_index=1,
        gpu_prefix=0,
        gpu_extend=1,
        expected_prefix=0,
        expected_extend=1,
        request_pool_size=1,
    )
    assert summary["nonzero_count"] == 3
    assert summary["nonfinite_count"] == 3
    assert torch.equal(state.view(torch.int32), before.view(torch.int32))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="probe kernel needs CUDA")
def test_kernel_supports_production_shape_with_noncontiguous_slot_stride(tmp_path):
    backing = torch.zeros((11, 48, 128, 129), dtype=torch.float32, device="cuda")
    state = backing[..., :128]
    assert not state.is_contiguous()
    state[10, 47, 127, 127] = 1
    summary = probe._run_scan_for_test(
        state,
        slot=10,
        request_index=1,
        gpu_prefix=0,
        gpu_extend=1,
        expected_prefix=0,
        expected_extend=1,
        request_pool_size=1,
    )
    assert summary["coverage_valid"] is True
    assert summary["nonzero_count"] == 1
    assert summary["nonfinite_count"] == 0
    assert summary["state_strides"] == list(state.stride())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="probe kernel needs CUDA")
@pytest.mark.parametrize(
    ("slot", "request_index", "gpu_prefix", "gpu_extend", "reason"),
    [
        (0, 1, 0, 1, "slot_out_of_range"),
        (11, 1, 0, 1, "slot_out_of_range"),
        (1, 0, 0, 1, "request_index_out_of_range"),
        (1, 3, 0, 1, "request_index_out_of_range"),
        (1, 1, 4, 1, "prefix_mismatch"),
        (1, 1, 0, 2, "extend_length_mismatch"),
    ],
)
def test_kernel_guards_invalid_metadata_before_state_dereference(
    tmp_path, slot, request_index, gpu_prefix, gpu_extend, reason
):
    state = torch.ones((11, 48, 128, 128), dtype=torch.float32, device="cuda")
    summary = probe._run_scan_for_test(
        state,
        slot=slot,
        request_index=request_index,
        gpu_prefix=gpu_prefix,
        gpu_extend=gpu_extend,
        expected_prefix=0,
        expected_extend=1,
        request_pool_size=2,
    )
    assert summary["coverage_valid"] is False
    assert reason in summary["coverage_errors"]
    assert summary["nonzero_count"] == 0
    assert summary["nonfinite_count"] == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="probe lifecycle needs CUDA")
def test_writer_defers_until_cap_and_persists_cold_rows_only(tmp_path, monkeypatch):
    hook, arm, sample_dir = _make_hook(tmp_path, min_rows=1, max_records=2)
    arm.touch()
    state = torch.zeros((11, 48, 128, 128), dtype=torch.float32, device="cuda")
    state[1, 0, 0, 0] = 1
    temporal = state.unsqueeze(0)
    pool = SimpleNamespace(
        size=4,
        layer_transfer_counter=None,
        mamba_map={0: 0},
        mamba_pool=SimpleNamespace(mamba_cache=SimpleNamespace(temporal=temporal)),
    )
    metadata = SimpleNamespace(
        mamba_cache_indices=torch.tensor([1, 2, 3], device="cuda"),
    )
    linear = SimpleNamespace(req_to_token_pool=pool, forward_metadata=metadata)
    monkeypatch.setattr(
        probe,
        "_get_attn_backend",
        lambda: SimpleNamespace(linear_attn_backend=linear),
    )
    batch = SimpleNamespace(
        forward_mode=_mode("EXTEND"),
        batch_size=3,
        _original_batch_size=3,
        extend_prefix_lens_cpu=[0, 9, 0],
        extend_seq_lens_cpu=[2, 1, 3],
        req_pool_indices=torch.tensor([1, 2, 3], device="cuda"),
        extend_prefix_lens=torch.tensor([0, 9, 0], device="cuda"),
        extend_seq_lens=torch.tensor([2, 1, 3], device="cuda"),
    )
    hidden = torch.zeros((8, 1), device="cuda")
    before = state.clone()
    try:
        hook(_module(), (hidden, batch))
        hook.flush()
        payload = json.loads((sample_dir / "cold-state.json").read_text())
        assert payload["expected_records"] == 2
        assert payload["persisted_records"] == 2
        assert payload["coverage"]["cap_reached"] is True
        assert [record["batch_row"] for record in payload["records"]] == [0, 2]
        assert payload["records"][0]["nonzero_count"] == 1
        assert payload["records"][1]["nonzero_count"] == 0
        assert torch.equal(state, before)
    finally:
        hook.close()

"""CPU contracts for task-local diagnostic probes."""

import importlib.util
from pathlib import Path

import pytest
import torch


def _load(name):
    path = Path(__file__).resolve().parents[1] / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


teacher = _load("check_teacher_forced")
probe = _load("finite_probe")


def _response(values):
    return {
        "meta_info": {
            "prompt_tokens": 4,
            "cached_tokens": 0,
            "input_token_logprobs": [[x, i + 1, None] for i, x in enumerate(values)],
        }
    }


def test_teacher_ignores_only_leading_placeholder():
    assert teacher.aligned_tail(_response([None, -1, -2, -3]), [1, 2, 3, 4], 3) == [
        -1,
        -2,
        -3,
    ]


@pytest.mark.parametrize("invalid", [None, float("nan"), float("inf"), 0.1])
def test_teacher_rejects_invalid_tail(invalid):
    with pytest.raises(ValueError):
        teacher.aligned_tail(_response([None, -1, invalid, -3]), [1, 2, 3, 4], 3)


def test_repeat_delta_is_symmetric_and_checks_workload():
    actual = {"input_sha256": "same", "logprobs": [-1.0, -4.0]}
    prior = {"input_sha256": "same", "logprobs": [-2.0, -3.0]}
    assert teacher.paired_logprob_delta(actual, prior) == 1.0
    assert teacher.paired_logprob_delta(prior, actual) == 1.0
    assert teacher.paired_logprob_delta(actual, actual) == 0.0
    with pytest.raises(ValueError, match="workload mismatch"):
        teacher.paired_logprob_delta(actual, {**prior, "input_sha256": "other"})
    with pytest.raises(ValueError, match="workload mismatch"):
        teacher.paired_logprob_delta(actual, {**prior, "logprobs": [-2.0]})


def test_case_offsets_preserve_frozen_ids_and_reject_incomplete_ranges():
    manifest = {"cases": ["a", "b", "c", "d"]}
    assert list(teacher.selected_cases(manifest, 2, 1)) == [(1, "b"), (2, "c")]
    assert list(teacher.selected_cases(manifest, 1, 3)) == [(3, "d")]
    for count, offset in ((0, 0), (1, -1), (2, 3), (5, 0)):
        with pytest.raises(ValueError, match="outside the frozen manifest"):
            teacher.selected_cases(manifest, count, offset)


def test_finite_probe_does_not_modify_values():
    hook = probe.make_finite_hook({"min_rows": 1})
    module = torch.nn.Identity()
    value = torch.tensor([[1.0, -2.0]])
    before = value.clone()
    assert hook(module, (value,), (value, None)) is None
    assert torch.equal(value, before)
    with pytest.raises(RuntimeError, match="Q38 finite probe"):
        hook(module, (), torch.tensor([[float("nan")]]))


def test_qsa_index_probe_detects_holes_and_oob():
    module = type("QSAIndexer", (), {"compress_ratio": 4, "block_topk": 512})()
    hook = probe.make_finite_hook({"min_rows": 1})
    positions = torch.tensor([2])
    good = torch.tensor([[0, 1, 2, -1]])
    assert hook(module, (None, positions), good) is None
    for bad in (torch.tensor([[0, -1, 1, 2]]), torch.tensor([[0, 1, 3, -1]])):
        with pytest.raises(RuntimeError, match="Q38 index probe"):
            hook(module, (None, positions), bad)


def test_attention_samples_preserve_boundary_rows_and_inputs(tmp_path):
    module = type("RadixAttention", (), {"layer_id": 3})()
    hook = probe.make_finite_hook({"min_rows": 8192, "sample_dir": str(tmp_path)})
    value = torch.arange(8192 * 2, dtype=torch.float32).reshape(8192, 2)
    before = value.clone()
    assert hook(module, (value,), value) is None
    assert torch.equal(value, before)
    saved = torch.load(next(tmp_path.rglob("*.pt")), weights_only=True)
    assert {0, 1, 2, 3, 2050, 2051, 2052, 8191}.issubset(saved["rows"])
    assert len(saved["rows"]) < 80
    assert torch.equal(saved["value"], value[saved["rows"]])
    assert torch.equal(saved["inputs"][0]["value"], saved["value"])
    assert saved["full_input_sha256"][0]["sha256"] == saved["full_output_sha256"]


def test_layer_hashes_detect_unsampled_hidden_and_residual_changes(tmp_path):
    module = type("Qwen4ExpLinearDecoderLayer", (), {"layer_id": 0})()
    hook = probe.make_finite_hook({"min_rows": 8192, "sample_dir": str(tmp_path)})
    hidden = torch.ones(8192, 2)
    residual = hidden.clone()
    hook(module, (hidden, residual), (hidden, residual))
    hidden[1000, 0] = 2
    residual[2000, 1] = 3
    before = (hidden.clone(), residual.clone())
    hook(module, (hidden, residual), (hidden, residual))
    saved = [
        torch.load(path, weights_only=True) for path in sorted(tmp_path.rglob("*.pt"))
    ]
    assert torch.equal(saved[0]["value"], saved[1]["value"])
    assert saved[0]["full_output_sha256"] != saved[1]["full_output_sha256"]
    for index in (0, 1):
        first = saved[0]["full_output_sha256_all"][index]["sha256"]
        second = saved[1]["full_output_sha256_all"][index]["sha256"]
        assert first != second
        assert saved[1]["full_input_sha256"][index]["sha256"] == second
    assert torch.equal(hidden, before[0])
    assert torch.equal(residual, before[1])

"""Opt-in pre-forward hooks keep the existing post-forward default intact."""

import importlib.util
from pathlib import Path

import pytest
import torch


def _manager():
    path = (
        Path(__file__).resolve().parents[3]
        / "python/sglang/srt/model_executor/hook_manager.py"
    )
    spec = importlib.util.spec_from_file_location("q38_test_hook_manager", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "hook_type, expected", [(None, [2, 3]), ("forward", [2, 3]), ("forward_pre", [2])]
)
def test_registration_calls_correct_phase(monkeypatch, hook_type, expected):
    manager = _manager()
    model = torch.nn.Sequential(torch.nn.Identity())
    model[0].forward = lambda value: value + 1
    observed = []

    def hook(module, *values):
        observed.extend(
            int(value[0] if isinstance(value, tuple) else value) for value in values
        )

    monkeypatch.setattr(manager, "resolve_callable", lambda path: lambda config: hook)
    spec = {"target_modules": ["0"], "hook_factory": "test:factory"}
    if hook_type is not None:
        spec["hook_type"] = hook_type
    manager.register_forward_hooks(model, [spec])
    assert model(torch.tensor(2)).item() == 3
    assert observed == expected


def test_unknown_phase_rejected_before_factory_runs(monkeypatch):
    manager = _manager()
    monkeypatch.setattr(
        manager,
        "resolve_callable",
        lambda path: pytest.fail("unexpected factory lookup"),
    )
    with pytest.raises(ValueError, match="hook_type"):
        manager.register_forward_hooks(
            torch.nn.Identity(),
            [
                {
                    "target_modules": ["*"],
                    "hook_factory": "test:factory",
                    "hook_type": "typo",
                }
            ],
        )

"""CPU-only checks that isolated launch profiles cannot inherit native flags."""

import importlib.util
import json
from pathlib import Path
import sys

import pytest


def _launcher():
    path = Path(__file__).resolve().parents[1] / "launch_sm120.py"
    spec = importlib.util.spec_from_file_location("q38_test_launcher", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PDL_ENV_NAMES = (
    "SGLANG_JIT_DISABLE_PDL",
    "SGLANG_FLASHINFER_CUTLASS_DISABLE_PDL",
    "SGLANG_QSA_TRTLLM_DISABLE_PDL",
)


def _capture_profile_launch(
    monkeypatch, tmp_path, capsys, *extra_args, arm="python-matched"
):
    module = _launcher()
    for name in ("config.json", "model.safetensors.index.json"):
        (tmp_path / name).write_text("{}")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "launch_sm120.py",
            "--arm",
            arm,
            "--model",
            str(tmp_path),
            *extra_args,
        ],
    )
    monkeypatch.setattr(
        module.subprocess,
        "check_output",
        lambda command: b"test-revision\n" if "rev-parse" in command else b"",
    )
    launched = {}

    def capture_exec(executable, command, env):
        launched.update(env)
        launched["command"] = command

    monkeypatch.setattr(module.os, "execve", capture_exec)
    module.main()
    fingerprint_line = next(
        line
        for line in capsys.readouterr().out.splitlines()
        if line.startswith("runtime_fingerprint=")
    )
    return launched, json.loads(fingerprint_line.partition("=")[2])


def test_disable_auto_pdl_sets_exact_policy_and_records_provenance(
    monkeypatch, tmp_path, capsys
):
    for name in PDL_ENV_NAMES:
        monkeypatch.setenv(name, "0")

    launched, fingerprint = _capture_profile_launch(
        monkeypatch, tmp_path, capsys, "--disable-auto-pdl"
    )

    assert {name: launched[name] for name in PDL_ENV_NAMES} == dict.fromkeys(
        PDL_ENV_NAMES, "1"
    )
    assert fingerprint["pdl_policy"] == {
        "disable_auto_pdl": True,
        "effective_env": dict.fromkeys(PDL_ENV_NAMES, "1"),
    }


@pytest.mark.parametrize("inherited", [None, ("0", "true", "yes")])
def test_default_launch_preserves_existing_pdl_environment(
    monkeypatch, tmp_path, capsys, inherited
):
    for index, name in enumerate(PDL_ENV_NAMES):
        if inherited is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, inherited[index])

    launched, fingerprint = _capture_profile_launch(monkeypatch, tmp_path, capsys)

    expected = {
        name: None if inherited is None else inherited[index]
        for index, name in enumerate(PDL_ENV_NAMES)
    }
    assert {name: launched.get(name) for name in PDL_ENV_NAMES} == expected
    assert fingerprint["pdl_policy"] == {
        "disable_auto_pdl": False,
        "effective_env": expected,
    }


def test_disable_auto_pdl_rejects_frozen_baseline(monkeypatch, capsys):
    module = _launcher()
    monkeypatch.setattr(
        sys,
        "argv",
        ["launch_sm120.py", "--arm", "baseline", "--disable-auto-pdl"],
    )
    with pytest.raises(SystemExit) as error:
        module.main()
    assert error.value.code == 2
    assert "frozen baseline" in capsys.readouterr().err


@pytest.mark.parametrize("dtype", ["auto", "bfloat16", "fp8_e4m3"])
def test_baseline_kv_override_is_explicit_and_fingerprinted(
    monkeypatch, tmp_path, capsys, dtype
):
    launched, fingerprint = _capture_profile_launch(
        monkeypatch,
        tmp_path,
        capsys,
        "--baseline-kv-cache-dtype",
        dtype,
        arm="baseline",
    )
    command = launched["command"]
    assert command[command.index("--kv-cache-dtype") + 1] == dtype
    assert fingerprint["launch_options"]["baseline_kv_cache_dtype"] == dtype
    assert fingerprint["resolved_kv_cache_dtype"] == dtype


@pytest.mark.parametrize("arm", ["native", "python-matched"])
def test_baseline_kv_override_cannot_change_candidate_precision(
    monkeypatch, capsys, arm
):
    module = _launcher()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "launch_sm120.py",
            "--arm",
            arm,
            "--baseline-kv-cache-dtype",
            "bfloat16",
        ],
    )
    with pytest.raises(SystemExit) as error:
        module.main()
    assert error.value.code == 2
    assert "baseline KV override requires --arm baseline" in capsys.readouterr().err


def test_baseline_kv_default_remains_fp8(monkeypatch, tmp_path, capsys):
    launched, fingerprint = _capture_profile_launch(
        monkeypatch, tmp_path, capsys, arm="baseline"
    )
    command = launched["command"]
    assert command[command.index("--kv-cache-dtype") + 1] == "fp8_e4m3"
    assert fingerprint["launch_options"]["baseline_kv_cache_dtype"] is None
    assert fingerprint["resolved_kv_cache_dtype"] == "fp8_e4m3"


def test_flashinfer_policy_records_effective_environment(monkeypatch, tmp_path, capsys):
    policy = {
        "SGLANG_FLASHINFER_MOE_FUSED_FINALIZE": "0",
        "FLASHINFER_AUTOTUNER_LOAD_FROM_FILE": "0",
        "SGLANG_FLASHINFER_AUTOTUNE_CACHE": "0",
    }
    for name, value in policy.items():
        monkeypatch.setenv(name, value)
    _, fingerprint = _capture_profile_launch(
        monkeypatch, tmp_path, capsys, "--disable-flashinfer-autotune"
    )
    assert fingerprint["flashinfer_policy"] == {
        "disable_autotune": True,
        "effective_env": policy,
    }


def test_linear_replayssm_spec_is_default_off(monkeypatch, tmp_path, capsys):
    launched, fingerprint = _capture_profile_launch(monkeypatch, tmp_path, capsys)
    assert "--enable-linear-replayssm-spec" not in launched["command"]
    assert fingerprint["launch_options"]["enable_linear_replayssm_spec"] is False


def test_host_kv_dedup_defaults_off_and_clears_inherited_env(
    monkeypatch, tmp_path, capsys
):
    monkeypatch.setenv("QWEN38_HOST_KV_DEDUP", "1")
    launched, fingerprint = _capture_profile_launch(monkeypatch, tmp_path, capsys)
    assert launched["QWEN38_HOST_KV_DEDUP"] == "0"
    assert fingerprint["launch_options"]["host_kv_dedup"] is False


def test_host_kv_dedup_native_opt_in_sets_env_and_provenance(
    monkeypatch, tmp_path, capsys
):
    stub = tmp_path / "runtime.so"
    stub.write_bytes(b"test fingerprint only")
    for variable in (
        "QWEN38_NATIVE_LIBRARY",
        "QWEN38_PLE_LIBRARY",
        "QWEN38_HOST_KV_LIBRARY",
    ):
        monkeypatch.setenv(variable, str(stub))
    launched, fingerprint = _capture_profile_launch(
        monkeypatch,
        tmp_path,
        capsys,
        "--host-kv-bytes",
        "1048576",
        "--host-kv-dedup",
        arm="native",
    )
    assert launched["QWEN38_HOST_KV_DEDUP"] == "1"
    assert launched["QWEN38_HOST_KV_BYTES"] == "1048576"
    assert fingerprint["launch_options"]["host_kv_dedup"] is True


@pytest.mark.parametrize(
    "arguments",
    [
        ["--arm", "native", "--host-kv-dedup"],
        [
            "--arm",
            "baseline",
            "--host-kv-bytes",
            "1048576",
            "--host-kv-dedup",
        ],
        [
            "--arm",
            "python-matched",
            "--host-kv-bytes",
            "1048576",
            "--host-kv-dedup",
        ],
    ],
)
def test_host_kv_dedup_rejects_unsupported_launches(monkeypatch, capsys, arguments):
    module = _launcher()
    monkeypatch.setattr(sys, "argv", ["launch_sm120.py", *arguments])
    with pytest.raises(SystemExit) as error:
        module.main()
    assert error.value.code == 2
    assert "host-KV" in capsys.readouterr().err


@pytest.mark.parametrize("arm", ["python-matched", "native"])
def test_linear_replayssm_spec_is_explicitly_forwarded(
    monkeypatch, tmp_path, capsys, arm
):
    if arm == "native":
        stub = tmp_path / "runtime.so"
        stub.write_bytes(b"test fingerprint only")
        for variable in (
            "QWEN38_NATIVE_LIBRARY",
            "QWEN38_PLE_LIBRARY",
            "QWEN38_HOST_KV_LIBRARY",
        ):
            monkeypatch.setenv(variable, str(stub))
    default_launch, _default_fingerprint = _capture_profile_launch(
        monkeypatch, tmp_path, capsys, arm=arm
    )
    launched, fingerprint = _capture_profile_launch(
        monkeypatch,
        tmp_path,
        capsys,
        "--enable-linear-replayssm-spec",
        arm=arm,
    )
    assert launched["command"].count("--enable-linear-replayssm-spec") == 1
    assert [
        argument
        for argument in launched["command"]
        if argument != "--enable-linear-replayssm-spec"
    ] == default_launch["command"]
    assert fingerprint["launch_options"]["enable_linear_replayssm_spec"] is True


def test_linear_replayssm_spec_rejects_frozen_baseline(monkeypatch, capsys):
    module = _launcher()
    monkeypatch.setattr(
        sys,
        "argv",
        ["launch_sm120.py", "--arm", "baseline", "--enable-linear-replayssm-spec"],
    )
    with pytest.raises(SystemExit) as error:
        module.main()
    assert error.value.code == 2
    assert "frozen baseline" in capsys.readouterr().err


@pytest.mark.parametrize("disabled", [False, True])
def test_flashinfer_autotune_control_is_explicit_and_independent(
    monkeypatch, tmp_path, capsys, disabled
):
    for name in PDL_ENV_NAMES:
        monkeypatch.delenv(name, raising=False)
    launched, fingerprint = _capture_profile_launch(
        monkeypatch,
        tmp_path,
        capsys,
        *(["--disable-flashinfer-autotune"] if disabled else []),
    )
    assert launched["command"].count("--disable-flashinfer-autotune") == int(disabled)
    assert fingerprint["launch_options"]["disable_flashinfer_autotune"] is disabled
    assert fingerprint["pdl_policy"]["disable_auto_pdl"] is False
    assert all(launched.get(name) is None for name in PDL_ENV_NAMES)


@pytest.mark.parametrize("chunk", [4096, 8192])
def test_prefill_chunk_control_preserves_cache_and_model_limits(
    monkeypatch, tmp_path, capsys, chunk
):
    launched, fingerprint = _capture_profile_launch(
        monkeypatch, tmp_path, capsys, "--chunked-prefill-size", str(chunk)
    )
    command = launched["command"]
    for name in (
        "--chunked-prefill-size",
        "--cuda-graph-max-bs-prefill",
        "--cuda-graph-bs-prefill",
    ):
        assert command[command.index(name) + 1] == str(chunk)
    for name, expected in (
        ("--kv-cache-dtype", "fp8_e4m3"),
        ("--context-length", "262144"),
        ("--max-total-tokens", "147456"),
        ("--max-prefill-tokens", "16384"),
        ("--speculative-num-steps", "3"),
        ("--speculative-num-draft-tokens", "4"),
    ):
        assert command[command.index(name) + 1] == expected
    assert fingerprint["launch_options"]["chunked_prefill_size"] == chunk


def test_layer_filtered_fingerprints_register_only_at_known_qwen_boundaries():
    module = _launcher()
    assert module.fingerprint_targets(None, None) == ["*"]
    assert module.fingerprint_targets(["QSAIndexer"], [2, 3]) == [
        "*.layers.2.indexer",
        "*.layers.3.indexer",
    ]
    assert module.fingerprint_targets(
        ["Qwen4ExpLinearDecoderLayer", "QSAIndexer"], [2, 3]
    ) == ["*.layers.2", "*.layers.2.indexer", "*.layers.3", "*.layers.3.indexer"]
    assert module.fingerprint_targets(["RadixAttention"], [3]) == ["*.layers.3.attn"]
    assert module.fingerprint_targets(["Qwen3_5GatedDeltaNet"], [0]) == [
        "*.layers.0.linear_attn"
    ]


@pytest.mark.parametrize(
    "flags",
    [
        ["--arm", "baseline", "--native-context-tail"],
        ["--arm", "python-matched", "--native-context-tail"],
        ["--arm", "baseline", "--canonical-qsa-order"],
        ["--arm", "baseline", "--native-bf16-gdn"],
        ["--arm", "baseline", "--native-bf16-linear"],
        ["--arm", "native", "--diagnostic-context-tail"],
        ["--arm", "native", "--diagnostic-fingerprint-classes", "QSAIndexer"],
        ["--arm", "native", "--diagnostic-fingerprint-layers", "2"],
        ["--arm", "native", "--diagnostic-fingerprint-max-elements", "128"],
        ["--arm", "native", "--diagnostic-fingerprint-before"],
        ["--arm", "native", "--diagnostic-fingerprint-defer-records", "4"],
        ["--arm", "native", "--diagnostic-fingerprint-control", "noop"],
        [
            "--arm",
            "native",
            "--diagnostic-fingerprints",
            "unused",
            "--diagnostic-fingerprint-defer-records",
            "129",
        ],
        [
            "--arm",
            "native",
            "--diagnostic-fingerprints",
            "unused",
            "--diagnostic-fingerprint-max-elements",
            "0",
        ],
        [
            "--arm",
            "baseline",
            "--diagnostic-fingerprints",
            "unused",
            "--diagnostic-fingerprint-before",
        ],
        [
            "--arm",
            "native",
            "--diagnostic-fingerprints",
            "unused",
            "--diagnostic-fingerprint-before",
            "--diagnostic-fingerprint-classes",
            "Qwen4ExpLinearDecoderLayer",
        ],
        [
            "--arm",
            "native",
            "--diagnostic-fingerprints",
            "unused",
            "--diagnostic-fingerprint-layers",
            "-1",
        ],
        [
            "--arm",
            "native",
            "--diagnostic-fingerprints",
            "unused",
            "--diagnostic-finite",
        ],
        [
            "--arm",
            "native",
            "--diagnostic-fingerprints",
            "unused",
            "--diagnostic-samples",
            "unused",
        ],
    ],
)
def test_unsupported_flags_reject_before_launch(monkeypatch, flags):
    module = _launcher()
    monkeypatch.setattr(sys, "argv", ["launch_sm120.py", *flags])
    with pytest.raises(SystemExit) as error:
        module.main()
    assert error.value.code == 2


@pytest.mark.parametrize("arm", ["baseline", "python-matched", "native"])
def test_default_profile_clears_inherited_tail_and_order_flags(
    monkeypatch, tmp_path, arm
):
    module = _launcher()
    for name in ("config.json", "model.safetensors.index.json"):
        (tmp_path / name).write_text("{}")
    stub = tmp_path / "never-loaded.so"
    stub.write_bytes(b"test fingerprint only")
    for variable in (
        "QWEN38_NATIVE_LIBRARY",
        "QWEN38_PLE_LIBRARY",
        "QWEN38_HOST_KV_LIBRARY",
    ):
        monkeypatch.setenv(variable, str(stub))
    flags = (
        "QWEN38_CONTEXT_TAIL",
        "QWEN38_CONTEXT_TAIL_TRACE",
        "QWEN38_QSA_CANONICAL_ORDER",
    )
    for variable in flags:
        monkeypatch.setenv(variable, "1")
    monkeypatch.setattr(
        sys, "argv", ["launch_sm120.py", "--arm", arm, "--model", str(tmp_path)]
    )
    monkeypatch.setattr(
        module.subprocess,
        "check_output",
        lambda command: b"test-revision\n" if "rev-parse" in command else b"",
    )
    launched = {}

    def capture_exec(executable, command, env):
        launched.update(env)

    monkeypatch.setattr(module.os, "execve", capture_exec)
    module.main()
    assert {name: launched[name] for name in flags} == dict.fromkeys(flags, "0")


@pytest.mark.parametrize(
    "fingerprints, fingerprint_classes, fingerprint_layers, sparse_before, control",
    [
        (False, None, None, False, None),
        (True, None, None, False, None),
        (True, ["QSAIndexer"], [2, 3], False, None),
        (True, ["QSAIndexer", "Qwen4ExpLinearDecoderLayer"], [0, 1, 2, 3], True, None),
        (True, ["Qwen3_5GatedDeltaNet"], [0], False, "record-stream"),
        (True, ["Qwen3_5GatedDeltaNet"], [0], False, "noop"),
        (True, ["Qwen3_5GatedDeltaNet"], [0], False, "allocator"),
    ],
)
def test_python_control_can_match_bf16_math_without_native_executor(
    monkeypatch,
    tmp_path,
    fingerprints,
    fingerprint_classes,
    fingerprint_layers,
    sparse_before,
    control,
):
    module = _launcher()
    for name in ("config.json", "model.safetensors.index.json"):
        (tmp_path / name).write_text("{}")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "launch_sm120.py",
            "--arm",
            "python-matched",
            "--model",
            str(tmp_path),
            "--native-bf16-gdn",
            "--native-bf16-linear",
            "--canonical-qsa-order",
            *(["--diagnostic-fingerprint-control", control] if control else []),
            *(
                ["--diagnostic-fingerprints", str(tmp_path / "fingerprints")]
                if fingerprints
                else []
            ),
            *(
                ["--diagnostic-fingerprint-classes", *fingerprint_classes]
                if fingerprint_classes
                else []
            ),
            *(
                ["--diagnostic-fingerprint-layers", *map(str, fingerprint_layers)]
                if fingerprint_layers
                else []
            ),
            *(
                [
                    "--diagnostic-fingerprint-before",
                    "--diagnostic-fingerprint-max-elements",
                    "2048",
                    "--diagnostic-fingerprint-defer-records",
                    "64",
                ]
                if sparse_before
                else []
            ),
        ],
    )
    monkeypatch.setattr(
        module.subprocess,
        "check_output",
        lambda command: b"test-revision\n" if "rev-parse" in command else b"",
    )
    launched = {}

    def capture_exec(executable, command, env):
        launched.update(env)
        launched["command"] = command

    monkeypatch.setattr(module.os, "execve", capture_exec)
    module.main()
    assert launched["QWEN38_BF16_GDN"] == "1"
    assert launched["QWEN38_BF16_LINEAR"] == "1"
    assert launched["QWEN38_QSA_CANONICAL_ORDER"] == "1"
    assert launched["QWEN38_NATIVE_EXECUTOR"] == "0"
    command = launched["command"]
    if fingerprints:
        hooks = json.loads(command[command.index("--forward-hooks") + 1])
        assert len(hooks) == (2 if sparse_before else 1)
        assert hooks[0]["hook_factory"].endswith(
            "fingerprint_probe:make_fingerprint_hook"
        )
        assert hooks[0]["config"]["sample_dir"] == str(tmp_path / "fingerprints")
        assert hooks[0]["config"]["module_classes"] == fingerprint_classes
        assert hooks[0]["config"]["layer_ids"] == fingerprint_layers
        assert hooks[0]["target_modules"] == module.fingerprint_targets(
            fingerprint_classes, fingerprint_layers
        )
        assert hooks[0]["config"]["max_elements"] == (2048 if sparse_before else None)
        assert hooks[0]["config"]["defer_records"] == (64 if sparse_before else 0)
        assert hooks[0]["config"]["control"] == control
        if sparse_before:
            assert hooks[1]["hook_type"] == "forward_pre"
            assert hooks[1]["hook_factory"].endswith(
                "fingerprint_probe:make_fingerprint_pre_hook"
            )
            assert hooks[1]["config"]["module_classes"] == ["QSAIndexer"]
            assert hooks[1]["config"]["layer_ids"] == fingerprint_layers
            assert hooks[1]["target_modules"] == module.fingerprint_targets(
                ["QSAIndexer"], fingerprint_layers
            )
    else:
        assert "--forward-hooks" not in command

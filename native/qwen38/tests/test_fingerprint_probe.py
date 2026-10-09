"""Contracts for the asynchronous GPU fingerprint diagnostic hook."""

from __future__ import annotations

import gc
import importlib.util
import json
import os
import threading
from pathlib import Path

import pytest
import torch


def _load_probe():
    path = Path(__file__).resolve().parents[1] / "fingerprint_probe.py"
    spec = importlib.util.spec_from_file_location("fingerprint_probe", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


probe = _load_probe()

BLOCK = 4096
MASK = 0xFFFFFFFF
FIRST_HIGH = 0x9E3779B1
FIRST_POSITION = 0x85EBCA77
FIRST_SEED = 0xD1B54A35
FIRST_MUL1 = 0x7FEB352D
FIRST_MUL2 = 0x846CA68B
SECOND_LOW = 0x27D4EB2D
SECOND_POSITION = 0x165667B1
SECOND_SEED = 0x94D049BB
SECOND_MUL1 = 0x2C1B3C6D
SECOND_MUL2 = 0x297A2D39


def _module(name: str, layer_id: int = 0):
    return type(name, (torch.nn.Module,), {"layer_id": layer_id})()


def _raw_halves(tensor: torch.Tensor):
    value = tensor.detach().contiguous().cpu()
    if value.dtype == torch.bfloat16:
        raw = value.view(torch.int16).to(torch.int64) & 0xFFFF
        return raw.flatten(), torch.zeros(value.numel(), dtype=torch.int64)
    if value.dtype in (torch.float32, torch.int32):
        raw = value.view(torch.int32).to(torch.int64) & MASK
        return raw.flatten(), torch.zeros(value.numel(), dtype=torch.int64)
    if value.dtype == torch.int64:
        raw = value.flatten()
        return raw & MASK, (raw >> 32) & MASK
    raise AssertionError(f"unsupported reference dtype {value.dtype}")


def _avalanche32(value, shift1, multiplier1, shift2, multiplier2, shift3):
    value = (value ^ (value >> shift1)) & MASK
    value = (value * multiplier1) & MASK
    value = (value ^ (value >> shift2)) & MASK
    value = (value * multiplier2) & MASK
    return (value ^ (value >> shift3)) & MASK


def _cpu_fingerprint(tensor: torch.Tensor, sample_stride=1) -> torch.Tensor:
    low, high = _raw_halves(tensor)
    low, high = low[::sample_stride], high[::sample_stride]
    blocks = (low.numel() + BLOCK - 1) // BLOCK
    result = []
    for block_start in range(0, blocks, 128):
        elem_start = block_start * BLOCK
        elem_end = min(low.numel(), (block_start + 128) * BLOCK)
        count = elem_end - elem_start
        positions = (
            torch.arange(elem_start, elem_end, dtype=torch.int64) * sample_stride
        )
        lo = low[elem_start:elem_end]
        hi = high[elem_start:elem_end]
        one = (
            lo
            ^ ((hi * FIRST_HIGH) & MASK)
            ^ ((positions * FIRST_POSITION) & MASK)
            ^ FIRST_SEED
        )
        one = _avalanche32(one, 16, FIRST_MUL1, 15, FIRST_MUL2, 16)
        two = (
            ((lo * SECOND_LOW) & MASK)
            ^ hi
            ^ ((positions * SECOND_POSITION) & MASK)
            ^ SECOND_SEED
        )
        two = _avalanche32(two, 15, SECOND_MUL1, 12, SECOND_MUL2, 15)
        padded = ((count + BLOCK - 1) // BLOCK) * BLOCK
        if padded != count:
            one = torch.nn.functional.pad(one, (0, padded - count))
            two = torch.nn.functional.pad(two, (0, padded - count))
        result.append(
            torch.stack(
                (
                    one.reshape(-1, BLOCK).sum(1) & MASK,
                    two.reshape(-1, BLOCK).sum(1) & MASK,
                ),
                dim=1,
            ).to(torch.uint32)
        )
    return torch.cat(result) if result else torch.empty((0, 2), dtype=torch.uint32)


def _records(root: Path):
    paths = sorted(root.rglob("*.pt"))
    return [torch.load(path, weights_only=True) for path in paths]


def _output_checksums(record):
    return next(
        value["checksums"]
        for value in record["tensors"]
        if value["boundary"] == "output"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="control needs CUDA")
@pytest.mark.parametrize("control", ["record-stream", "noop"])
def test_control_does_not_hash_allocate_events_or_enqueue(
    tmp_path, monkeypatch, control
):
    hook = probe.make_fingerprint_hook(
        {
            "min_rows": 4,
            "sample_dir": str(tmp_path),
            "defer_records": 64,
            "control": control,
        }
    )
    value = torch.arange(32, device="cuda", dtype=torch.float32).reshape(4, 8)
    output = value + 1
    before = output.clone()
    recorded = []
    original_record_stream = torch.Tensor.record_stream

    def record_stream(tensor, stream):
        recorded.append(tensor)
        return original_record_stream(tensor, stream)

    def forbidden(*args, **kwargs):
        raise AssertionError("record-only control must not hash or record events")

    try:
        monkeypatch.setattr(torch.Tensor, "record_stream", record_stream)
        monkeypatch.setattr(probe, "_fingerprint", forbidden)
        monkeypatch.setattr(torch.cuda, "Event", forbidden)
        hook(_module("Qwen3_5GatedDeltaNet"), (value,), output)
        assert len(recorded) == (2 if control == "record-stream" else 0)
        if recorded:
            assert recorded[0] is value and recorded[1] is output
        assert hook._queue.empty()
        assert hook._submitted == 0
        assert hook._copy_stream is None
        assert not list(tmp_path.rglob("*.pt"))
        assert torch.equal(output, before)
    finally:
        hook.close()


@pytest.mark.parametrize("control", [1, True, "invalid"])
def test_unknown_control_rejects(tmp_path, control):
    with pytest.raises(ValueError, match="control"):
        probe.make_fingerprint_hook({"sample_dir": str(tmp_path), "control": control})


@pytest.mark.skipif(not torch.cuda.is_available(), reason="fingerprints need CUDA")
@pytest.mark.parametrize(
    "make_value",
    [
        lambda: torch.arange(5000, dtype=torch.float32, device="cuda").reshape(5, 1000),
        lambda: torch.arange(85, dtype=torch.float32, device="cuda")
        .to(torch.bfloat16)
        .reshape(17, 5)
        .T,
        lambda: torch.arange(10000, dtype=torch.int32, device="cuda").reshape(5, 2000)[
            :, ::2
        ],
        lambda: torch.arange(5000, dtype=torch.int64, device="cuda").reshape(5, 1000),
    ],
    ids=["float32", "strided_bfloat16", "strided_int32", "int64"],
)
def test_numerical_fingerprints_match_cpu_reference(tmp_path, make_value):
    hook = probe.make_fingerprint_hook({"min_rows": 1, "sample_dir": str(tmp_path)})
    value = make_value()
    before = value.clone()

    assert hook(_module("QSAIndexer", 7), (), value) is None
    hook.flush()

    [record] = _records(tmp_path)
    [saved] = record["tensors"]
    assert record["kind"] == "noncryptographic_gpu_block_fingerprint"
    assert record["equivalence_proof"] is False
    assert "does not prove tensor equivalence" in record["limitations"]
    assert record["module_class"] == "QSAIndexer"
    assert record["layer_id"] == 7
    assert saved["shape"] == tuple(value.shape)
    assert saved["strides"] == tuple(value.stride())
    assert saved["dtype"] == str(value.dtype)
    assert torch.equal(saved["checksums"], _cpu_fingerprint(value))
    assert torch.equal(value, before)
    hook.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="fingerprints need CUDA")
def test_signed_zero_nan_and_unsampled_bitflip_change_fingerprint(tmp_path):
    bits = torch.zeros((16, 600), dtype=torch.int16)
    bits[5, 11] = -32768  # -0.0 in BF16.
    bits[6, 13] = 0x7FC1  # Preserve one specific NaN payload.
    first_cpu = bits.view(torch.bfloat16)
    second_bits = bits.clone()
    second_bits[8, 17] ^= 1  # Outside the old first/last-four sampled rows.
    second_cpu = second_bits.view(torch.bfloat16)
    first = first_cpu.cuda()
    second = second_cpu.cuda()
    hook = probe.make_fingerprint_hook({"min_rows": 1, "sample_dir": str(tmp_path)})
    module = _module("Qwen4ExpAttentionDecoderLayer", 3)

    hook(module, (), (first, None))
    hook(module, (), (second, None))
    hook.flush()

    records = _records(tmp_path)
    assert len(records) == 2
    assert records[0]["call"] == 0 and records[1]["call"] == 1
    assert records[0]["module_id"] == records[1]["module_id"]
    assert torch.equal(_output_checksums(records[0]), _cpu_fingerprint(first_cpu))
    assert torch.equal(_output_checksums(records[1]), _cpu_fingerprint(second_cpu))
    assert not torch.equal(_output_checksums(records[0]), _output_checksums(records[1]))
    hook.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="fingerprints need CUDA")
def test_each_checksum_detects_within_block_permutation_and_balanced_bit_changes():
    original = torch.arange(BLOCK, dtype=torch.int32)
    permuted = original.clone()
    permuted[117], permuted[3091] = original[3091], original[117]
    balanced = original.clone()
    balanced[701] += 1
    balanced[2703] -= 1

    gpu_hashes = [
        probe._fingerprint(value.cuda()).cpu()
        for value in (original, permuted, balanced)
    ]

    assert all(
        torch.equal(actual, _cpu_fingerprint(value))
        for actual, value in zip(gpu_hashes, (original, permuted, balanced))
    )
    assert torch.all(gpu_hashes[0] != gpu_hashes[1])
    assert torch.all(gpu_hashes[0] != gpu_hashes[2])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="fingerprints need CUDA")
def test_each_checksum_detects_two_float32_sign_bit_flips_in_one_block():
    original = torch.ones(BLOCK, dtype=torch.float32)
    changed = original.clone()
    changed_bits = changed.view(torch.int32)
    changed_bits[317] ^= -(2**31)
    changed_bits[3001] ^= -(2**31)

    actual = [probe._fingerprint(value.cuda()).cpu() for value in (original, changed)]

    assert torch.equal(actual[0], _cpu_fingerprint(original))
    assert torch.equal(actual[1], _cpu_fingerprint(changed))
    assert torch.all(actual[0] != actual[1])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="fingerprints need CUDA")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.int32])
def test_each_checksum_detects_raw_zero_one_swap_at_old_collision_positions(dtype):
    raw_dtype = torch.int16 if dtype == torch.bfloat16 else torch.int32
    original_raw = torch.zeros(BLOCK, dtype=raw_dtype)
    original_raw[2050] = 1
    changed_raw = original_raw.clone()
    changed_raw[2048], changed_raw[2050] = original_raw[2050], original_raw[2048]
    original = original_raw.view(dtype)
    changed = changed_raw.view(dtype)

    actual = [probe._fingerprint(value.cuda()).cpu() for value in (original, changed)]

    assert torch.equal(actual[0], _cpu_fingerprint(original))
    assert torch.equal(actual[1], _cpu_fingerprint(changed))
    assert torch.all(actual[0] != actual[1])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="fingerprints need CUDA")
def test_back_to_back_model_shape_reuse_across_streams(tmp_path, monkeypatch):
    shape = (8192, 10240)
    source = torch.empty(shape, dtype=torch.bfloat16, device="cuda")
    main = torch.cuda.current_stream()
    side = torch.cuda.Stream()
    hook = probe.make_fingerprint_hook({"min_rows": 8192, "sample_dir": str(tmp_path)})
    module = _module("Qwen4ExpLinearDecoderLayer", 47)

    # A forward hook must not introduce a device-wide synchronization.
    original_sync = torch.cuda.synchronize
    monkeypatch.setattr(
        torch.cuda,
        "synchronize",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("forward hook synchronized")
        ),
    )
    source.zero_()
    hook(module, (), (source, None))
    side.wait_stream(main)
    with torch.cuda.stream(side):
        source.fill_(-0.0)
        hook(module, (), (source, None))
    monkeypatch.setattr(torch.cuda, "synchronize", original_sync)

    # This is the first host wait/readback in the test.
    hook.flush()
    records = _records(tmp_path)
    assert len(records) == 2
    zero = torch.zeros(shape, dtype=torch.bfloat16)
    negative_zero = torch.full(shape, -0.0, dtype=torch.bfloat16)
    assert torch.equal(_output_checksums(records[0]), _cpu_fingerprint(zero))
    assert torch.equal(_output_checksums(records[1]), _cpu_fingerprint(negative_zero))
    assert not torch.equal(_output_checksums(records[0]), _output_checksums(records[1]))
    hook.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="fingerprints need CUDA")
def test_source_storage_is_not_reused_before_cross_stream_hash_finishes(tmp_path):
    shape = (512, 4096)
    probe._fingerprint(torch.zeros(shape, dtype=torch.float32, device="cuda")).cpu()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    allocator_stream = torch.cuda.Stream()
    hash_stream = torch.cuda.Stream()
    blocker_stream = torch.cuda.Stream()
    source_ready = torch.cuda.Event()
    unblock_hash = torch.cuda.Event()
    hash_done = torch.cuda.Event()
    hook = probe.make_fingerprint_hook({"min_rows": 1, "sample_dir": str(tmp_path)})

    with torch.cuda.stream(allocator_stream):
        source = torch.full(shape, 1.25, dtype=torch.float32, device="cuda")
        source_ready.record()
    source_ptr = source.data_ptr()
    with torch.cuda.stream(blocker_stream):
        torch.cuda._sleep(1_000_000_000)
        unblock_hash.record()
    with torch.cuda.stream(hash_stream):
        hash_stream.wait_event(source_ready)
        hash_stream.wait_event(unblock_hash)
        hook(_module("QSAIndexer"), (), source)
        hash_done.record()

    del source
    gc.collect()
    with torch.cuda.stream(allocator_stream):
        replacement = torch.full(shape, -3.5, dtype=torch.float32, device="cuda")
    replacement_ptr = replacement.data_ptr()
    overlapped = not hash_done.query()

    hook.flush()
    [record] = _records(tmp_path)
    actual = _output_checksums(record)
    expected = _cpu_fingerprint(torch.full(shape, 1.25, dtype=torch.float32))
    hook.close()

    assert overlapped
    assert replacement_ptr != source_ptr
    assert torch.equal(actual, expected)


def test_cpu_and_selection_guards_write_nothing(tmp_path):
    hook = probe.make_fingerprint_hook({"min_rows": 8, "sample_dir": str(tmp_path)})
    cpu = torch.ones((8, 4), dtype=torch.float32)
    assert hook(_module("QSAIndexer"), (cpu,), cpu) is None
    assert hook(_module("UnrelatedLayer"), (), cpu) is None
    assert hook(_module("Qwen4ExpLinearDecoderLayer"), (), (cpu[:7], None)) is None
    hook.flush()
    assert list(tmp_path.rglob("*.pt")) == []
    [directory] = list(tmp_path.iterdir())
    assert directory.name.startswith(f"pid{os.getpid()}-factory")
    hook.close()


def test_module_class_filter_keeps_only_requested_boundary(tmp_path):
    hook = probe.make_fingerprint_hook(
        {"sample_dir": str(tmp_path), "module_classes": ["QSAIndexer"]}
    )
    value = torch.zeros((8, 4))
    assert hook._selected(_module("QSAIndexer"), (value,), value)
    assert not hook._selected(_module("RadixAttention"), (value,), value)
    assert not hook._selected(_module("Qwen4ExpLinearDecoderLayer"), (), value)
    hook.close()


@pytest.mark.parametrize("classes", [[], ["UnrelatedLayer"], "QSAIndexer"])
def test_module_class_filter_rejects_invalid_boundaries(tmp_path, classes):
    with pytest.raises(ValueError, match="module_classes"):
        probe.make_fingerprint_hook(
            {"sample_dir": str(tmp_path), "module_classes": classes}
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="allocator needs CUDA")
def test_allocator_control_reports_allocation_and_use_stream_without_tensor_reads(
    tmp_path, monkeypatch
):
    origin = torch.cuda.Stream()
    with torch.cuda.stream(origin):
        value = torch.empty((4, 8), device="cuda", dtype=torch.bfloat16)
        output = torch.empty_like(value)
    producer = torch.cuda.current_stream()
    assert producer.cuda_stream != origin.cuda_stream
    hook = probe.make_fingerprint_hook(
        {"min_rows": 1, "sample_dir": str(tmp_path), "control": "allocator"}
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("allocator audit must not inspect tensor values")

    try:
        monkeypatch.setattr(probe, "_fingerprint", forbidden)
        monkeypatch.setattr(torch.Tensor, "cpu", forbidden)
        monkeypatch.setattr(torch.Tensor, "item", forbidden)
        module = _module("Qwen3_5GatedDeltaNet")
        hook(module, (value,), output)
        hook(module, (value,), output)
        paths = list(tmp_path.rglob("*allocations.json"))
        assert len(paths) == 1
        record = json.loads(paths[0].read_text())
        assert record["kind"] == "allocator_stream_metadata"
        assert record["producer_stream"] == producer.cuda_stream
        assert [item["boundary"] for item in record["tensors"]] == ["input", "output"]
        for item, tensor in zip(record["tensors"], (value, output)):
            assert item["data_ptr"] == tensor.data_ptr()
            assert item["allocation"]["stream"] == origin.cuda_stream
            assert item["allocation"]["state"] == "active_allocated"
        assert hook._submitted == 0
    finally:
        hook.close()


def test_layer_filter_keeps_only_requested_ids(tmp_path):
    hook = probe.make_fingerprint_hook(
        {"sample_dir": str(tmp_path), "layer_ids": [2, 3]}
    )
    value = torch.zeros((8, 4))
    assert hook._selected(_module("QSAIndexer", 3), (value,), value)
    assert hook._selected(_module("Qwen4ExpLinearDecoderLayer", 2), (), value)
    assert not hook._selected(_module("QSAIndexer", 7), (value,), value)
    assert not hook._selected(_module("Qwen4ExpLinearDecoderLayer", 0), (), value)
    hook.close()


def test_gdn_boundary_keeps_positional_input_and_output(tmp_path):
    hook = probe.make_fingerprint_hook(
        {
            "sample_dir": str(tmp_path),
            "module_classes": ["Qwen3_5GatedDeltaNet"],
            "layer_ids": [0],
        }
    )
    value = torch.zeros((8, 16))
    selected, rows = hook._selected(
        _module("Qwen3_5GatedDeltaNet", 0), (value, object()), value
    )
    assert rows == 8
    assert [(side, index) for side, index, _ in selected] == [
        ("input", 0),
        ("output", 0),
    ]
    assert not hook._selected(_module("Qwen3_5GatedDeltaNet", 1), (value,), value)
    hook.close()


@pytest.mark.parametrize("layers", [[], [-1], ["2"]])
def test_layer_filter_rejects_invalid_ids(tmp_path, layers):
    with pytest.raises(ValueError, match="layer_ids"):
        probe.make_fingerprint_hook({"sample_dir": str(tmp_path), "layer_ids": layers})


def test_atexit_reports_close_error_without_raising(tmp_path, monkeypatch, capsys):
    hook = probe.make_fingerprint_hook({"min_rows": 1, "sample_dir": str(tmp_path)})
    hook.close()
    monkeypatch.setattr(
        hook,
        "close",
        lambda: (_ for _ in ()).throw(OSError("late writer failure")),
    )

    assert hook._close_at_exit() is None

    assert (
        "fingerprint writer close failed: late writer failure"
        in capsys.readouterr().err
    )


def test_atexit_does_not_repeat_an_error_already_raised(tmp_path, capsys):
    hook = probe.make_fingerprint_hook({"min_rows": 1, "sample_dir": str(tmp_path)})
    hook._error = OSError("already surfaced")

    with pytest.raises(RuntimeError, match="already surfaced"):
        hook.flush()
    assert hook._close_at_exit() is None

    assert capsys.readouterr().err == ""


@pytest.mark.skipif(not torch.cuda.is_available(), reason="fingerprints need CUDA")
def test_records_only_real_matching_inputs_and_unique_modules(tmp_path):
    hook = probe.make_fingerprint_hook({"min_rows": 1, "sample_dir": str(tmp_path)})
    first_module = _module("RadixAttention", 5)
    second_module = _module("RadixAttention", 5)
    output = torch.ones((4, 8), dtype=torch.bfloat16, device="cuda")
    matching = torch.arange(32, dtype=torch.int32, device="cuda").reshape(4, 8)
    wrong_rows = torch.ones((3, 8), dtype=torch.float32, device="cuda")

    hook(first_module, (matching, object(), wrong_rows), output)
    hook(second_module, (), output)
    hook.flush()

    records = _records(tmp_path)
    assert len(records) == 2
    assert records[0]["module_id"] != records[1]["module_id"]
    assert [item["boundary"] for item in records[0]["tensors"]] == [
        "input",
        "output",
    ]
    assert [item["boundary"] for item in records[1]["tensors"]] == ["output"]
    hook.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="fingerprints need CUDA")
def test_bounded_queue_fails_loud_without_blocking_forward(tmp_path, monkeypatch):
    hook = probe.make_fingerprint_hook(
        {"min_rows": 1, "sample_dir": str(tmp_path), "queue_size": 1}
    )
    module = _module("QSAIndexer")
    value = torch.ones((4, 8), dtype=torch.float32, device="cuda")
    save_started = threading.Event()
    release_save = threading.Event()
    original_save = probe.torch.save

    def blocked_save(*args, **kwargs):
        original_save(*args, **kwargs)
        save_started.set()
        assert release_save.wait(10)

    monkeypatch.setattr(probe.torch, "save", blocked_save)
    hook(module, (), value)
    assert save_started.wait(10)
    assert list(tmp_path.rglob("*.pt")) == []
    assert len(list(tmp_path.rglob("*.pt.tmp"))) == 1
    hook(module, (), value)
    with pytest.raises(RuntimeError, match="queue is full"):
        hook(module, (), value)
    release_save.set()
    hook.flush()
    assert len(_records(tmp_path)) == 2
    hook.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="fingerprints need CUDA")
def test_writer_errors_surface_on_flush_and_later_hook(tmp_path, monkeypatch):
    hook = probe.make_fingerprint_hook({"min_rows": 1, "sample_dir": str(tmp_path)})
    module = _module("QSAIndexer")
    value = torch.ones((4, 8), dtype=torch.float32, device="cuda")

    monkeypatch.setattr(
        probe.torch,
        "save",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("save failed")),
    )
    hook(module, (), value)
    with pytest.raises(RuntimeError, match="save failed"):
        hook.flush()
    with pytest.raises(RuntimeError, match="save failed"):
        hook(module, (), value)
    with pytest.raises(RuntimeError, match="save failed"):
        hook.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="fingerprints need CUDA")
def test_sparse_fingerprint_reports_coverage_and_matches_strided_cpu_reference(
    tmp_path,
):
    value = torch.arange(10000, dtype=torch.float32, device="cuda").reshape(100, 100).T
    hook = probe.make_fingerprint_hook(
        {"min_rows": 1, "sample_dir": str(tmp_path), "max_elements": 127}
    )
    hook(_module("QSAIndexer", 3), (), value)
    hook.flush()
    [record] = _records(tmp_path)
    [saved] = record["tensors"]
    assert record["hook_phase"] == "after_forward"
    assert saved["sample_stride"] == 79
    assert saved["sampled_elements"] == 127
    assert "unsampled" in record["limitations"]
    assert torch.equal(saved["checksums"], _cpu_fingerprint(value, 79))
    hook.close()


@pytest.mark.parametrize("max_elements", [0, -1, 1.5, True])
def test_invalid_sample_limit_rejected(tmp_path, max_elements):
    with pytest.raises(ValueError, match="max_elements"):
        probe.make_fingerprint_hook(
            {"sample_dir": str(tmp_path), "max_elements": max_elements}
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="fingerprints need CUDA")
def test_pre_and_post_hooks_distinguish_input_mutation_without_host_wait(tmp_path):
    before = probe.make_fingerprint_pre_hook(
        {"min_rows": 1, "sample_dir": str(tmp_path), "module_classes": ["QSAIndexer"]}
    )
    after = probe.make_fingerprint_hook(
        {"min_rows": 1, "sample_dir": str(tmp_path), "module_classes": ["QSAIndexer"]}
    )
    module = _module("QSAIndexer", 3)
    module.forward = lambda value: value.add_(1)
    module.register_forward_pre_hook(before)
    module.register_forward_hook(after)
    value = torch.zeros((4, 8), dtype=torch.float32, device="cuda")
    module(value)
    before.flush()
    after.flush()
    records = {record["hook_phase"]: record for record in _records(tmp_path)}
    assert set(records) == {"before_forward", "after_forward"}
    [saved] = records["before_forward"]["tensors"]
    assert saved["boundary"] == "input"
    assert torch.equal(saved["checksums"], _cpu_fingerprint(torch.zeros((4, 8))))
    assert all(
        torch.equal(tensor["checksums"], _cpu_fingerprint(torch.ones((4, 8))))
        for tensor in records["after_forward"]["tensors"]
    )
    before.close()
    after.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="fingerprints need CUDA")
def test_model_shape_with_small_sample_cap_has_no_int32_address_overflow(tmp_path):
    value = torch.zeros((8192, 10240), dtype=torch.bfloat16, device="cuda")
    hook = probe.make_fingerprint_hook(
        {"min_rows": 1, "sample_dir": str(tmp_path), "max_elements": 128}
    )
    hook(_module("Qwen4ExpLinearDecoderLayer", 2), (), value)
    hook.flush()
    [record] = _records(tmp_path)
    [saved] = record["tensors"]
    assert saved["sample_stride"] == 655360
    assert saved["sampled_elements"] == 128
    assert torch.equal(saved["checksums"], _cpu_fingerprint(value, 655360))
    hook.close()


@pytest.mark.parametrize("defer_records", [-1, 129, True, 1.5])
def test_invalid_deferred_writer_threshold_rejected(tmp_path, defer_records):
    with pytest.raises(ValueError, match="defer_records"):
        probe.make_fingerprint_hook(
            {"sample_dir": str(tmp_path), "defer_records": defer_records}
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="fingerprints need CUDA")
@pytest.mark.parametrize("release", ["threshold", "flush", "close"])
def test_deferred_writer_waits_without_blocking_forward(tmp_path, release):
    hook = probe.make_fingerprint_hook(
        {"min_rows": 1, "sample_dir": str(tmp_path), "defer_records": 2}
    )
    module = _module("Qwen3_5GatedDeltaNet", 0)
    value = torch.ones((4, 8), dtype=torch.bfloat16, device="cuda")
    hook(module, (value,), value)
    assert not hook._writer_gate.is_set()
    assert hook._copy_stream is None
    assert not list(tmp_path.rglob("*.pt"))
    if release == "threshold":
        hook(module, (value,), value)
        assert hook._writer_gate.is_set()
        hook.flush()
    else:
        getattr(hook, release)()
    assert len(_records(tmp_path)) == (2 if release == "threshold" else 1)
    hook.close()

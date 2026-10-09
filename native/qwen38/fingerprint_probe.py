"""Low-perturbation, noncryptographic GPU fingerprints for Qwen3.8 probes."""

from __future__ import annotations

import atexit
import itertools
import json
import os
import queue
import sys
import threading
from pathlib import Path

import torch
import triton
import triton.language as tl


_BLOCK_ELEMENTS = 4096
_SUPPORTED_DTYPES = {torch.bfloat16, torch.float32, torch.int32, torch.int64}
_DECODER_LAYERS = {
    "Qwen4ExpLinearDecoderLayer",
    "Qwen4ExpAttentionDecoderLayer",
}
_BOUNDARIES = {"QSAIndexer", "RadixAttention", "Qwen3_5GatedDeltaNet"}
_factory_ids = itertools.count()


@triton.jit
def _fingerprint_kernel(
    source,
    checksums,
    numel,
    sample_stride,
    size0,
    size1,
    size2,
    size3,
    stride0,
    stride1,
    stride2,
    stride3,
    DTYPE_BITS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    sample = (tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)).to(tl.int64)
    logical = sample * sample_stride
    remaining = logical
    index3 = remaining % size3
    remaining //= size3
    index2 = remaining % size2
    remaining //= size2
    index1 = remaining % size1
    index0 = remaining // size1
    physical = index0 * stride0 + index1 * stride1 + index2 * stride2 + index3 * stride3
    mask = logical < numel
    value = tl.load(source + physical, mask=mask, other=0)
    if DTYPE_BITS == 16:
        low = value.to(tl.uint16, bitcast=True).to(tl.uint32)
        high = tl.zeros((BLOCK,), tl.uint32)
    elif DTYPE_BITS == 32:
        low = value.to(tl.uint32, bitcast=True)
        high = tl.zeros((BLOCK,), tl.uint32)
    else:
        raw = value.to(tl.uint64, bitcast=True)
        low = raw.to(tl.uint32)
        high = (raw >> 32).to(tl.uint32)

    position = logical.to(tl.uint32)
    first = low ^ (high * 0x9E3779B1) ^ (position * 0x85EBCA77) ^ 0xD1B54A35
    first ^= first >> 16
    first *= 0x7FEB352D
    first ^= first >> 15
    first *= 0x846CA68B
    first ^= first >> 16
    second = low * 0x27D4EB2D ^ high ^ (position * 0x165667B1) ^ 0x94D049BB
    second ^= second >> 15
    second *= 0x2C1B3C6D
    second ^= second >> 12
    second *= 0x297A2D39
    second ^= second >> 15
    first = tl.where(mask, first, 0)
    second = tl.where(mask, second, 0)
    output = tl.program_id(0) * 2
    tl.store(checksums + output, tl.sum(first, axis=0).to(tl.uint32))
    tl.store(checksums + output + 1, tl.sum(second, axis=0).to(tl.uint32))


def _tensors(value):
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _tensors(item)


def _fingerprint(tensor: torch.Tensor, sample_stride: int = 1) -> torch.Tensor:
    if tensor.ndim > 4:
        raise RuntimeError(
            f"fingerprint probe supports at most four dimensions, got {tensor.ndim}"
        )
    shape = (1,) * (4 - tensor.ndim) + tuple(tensor.shape)
    strides = (0,) * (4 - tensor.ndim) + tuple(tensor.stride())
    sampled_elements = triton.cdiv(tensor.numel(), sample_stride)
    blocks = triton.cdiv(sampled_elements, _BLOCK_ELEMENTS)
    result = torch.empty((blocks, 2), dtype=torch.uint32, device=tensor.device)
    if blocks:
        _fingerprint_kernel[(blocks,)](
            tensor,
            result,
            tensor.numel(),
            sample_stride,
            *shape,
            *strides,
            DTYPE_BITS=tensor.element_size() * 8,
            BLOCK=_BLOCK_ELEMENTS,
            num_warps=8,
        )
    return result


class _FingerprintHook:
    _phase = "after_forward"

    def __init__(self, config):
        self._min_rows = int(config.get("min_rows", 8192))
        if self._min_rows < 1:
            raise ValueError("fingerprint min_rows must be positive")
        self._control = config.get("control")
        if self._control not in (None, "record-stream", "noop", "allocator"):
            raise ValueError(
                "fingerprint control must be record-stream, noop, or allocator"
            )
        self._allocations_logged = set()
        self._max_elements = config.get("max_elements")
        if self._max_elements is not None and (
            type(self._max_elements) is not int or self._max_elements < 1
        ):
            raise ValueError("fingerprint max_elements must be a positive integer")
        supported_classes = _DECODER_LAYERS | _BOUNDARIES
        classes = config.get("module_classes")
        self._module_classes = (
            supported_classes if classes is None else frozenset(classes)
        )
        if not self._module_classes or not self._module_classes <= supported_classes:
            raise ValueError(
                "fingerprint module_classes must name supported boundaries"
            )
        layers = config.get("layer_ids")
        self._layer_ids = None if layers is None else frozenset(layers)
        if self._layer_ids is not None and (
            not self._layer_ids
            or any(not isinstance(layer, int) or layer < 0 for layer in self._layer_ids)
        ):
            raise ValueError("fingerprint layer_ids must be non-negative integers")
        sample_dir = config.get("sample_dir")
        if not sample_dir:
            raise ValueError("fingerprint sample_dir is required")
        self.output_dir = Path(sample_dir) / (
            f"pid{os.getpid()}-factory{next(_factory_ids)}"
        )
        self.output_dir.mkdir(parents=True, exist_ok=False)
        queue_size = int(config.get("queue_size", 128))
        if queue_size < 1:
            raise ValueError("fingerprint queue_size must be positive")
        self._defer_records = config.get("defer_records", 0)
        if (
            type(self._defer_records) is not int
            or not 0 <= self._defer_records <= queue_size
        ):
            raise ValueError("fingerprint defer_records must fit the bounded queue")
        self._submitted = 0
        self._writer_gate = threading.Event()
        if self._defer_records == 0:
            self._writer_gate.set()
        self._queue = queue.Queue(maxsize=queue_size)
        self._error = None
        self._error_reported = False
        self._closed = False
        self._module_ids = {}
        self._calls = {}
        self._next_module_id = 0
        self._copy_stream = None
        self._copy_device = None
        self._writer = threading.Thread(
            target=self._writer_loop,
            name=f"qwen38-fingerprint-{self.output_dir.name}",
            daemon=True,
        )
        self._writer.start()
        atexit.register(self._close_at_exit)

    def _raise_error(self):
        if self._error is not None:
            self._error_reported = True
            raise RuntimeError(
                f"fingerprint writer failed: {self._error}"
            ) from self._error

    def _identity(self, module):
        key = id(module)
        if key not in self._module_ids:
            self._module_ids[key] = self._next_module_id
            self._next_module_id += 1
        module_id = self._module_ids[key]
        call = self._calls.get(key, 0)
        self._calls[key] = call + 1
        return module_id, call

    def _selected(self, module, inputs, output):
        name = type(module).__name__
        if name not in self._module_classes:
            return []
        if (
            self._layer_ids is not None
            and getattr(module, "layer_id", None) not in self._layer_ids
        ):
            return []
        if name in _DECODER_LAYERS:
            outputs = [value for value in _tensors(output) if value.ndim > 0]
            if not outputs:
                return []
            rows = outputs[0].shape[0]
            return [
                ("output", i, value)
                for i, value in enumerate(outputs)
                if value.shape[0] == rows
            ], rows
        if name in _BOUNDARIES:
            outputs = list(_tensors(output))
            if not outputs or outputs[0].ndim == 0:
                return []
            rows = outputs[0].shape[0]
            selected = [
                ("input", i, value)
                for i, value in enumerate(_tensors(inputs))
                if value.ndim > 0 and value.shape[0] == rows
            ]
            selected.extend(
                ("output", i, value)
                for i, value in enumerate(outputs)
                if value.ndim > 0 and value.shape[0] == rows
            )
            return selected, rows
        return []

    def _write_allocator_metadata(self, module, selected, producer):
        if id(module) in self._allocations_logged:
            return
        snapshots = torch.cuda.memory_snapshot(include_traces=False)
        tensors = []
        for boundary, index, tensor in selected:
            pointer = tensor.data_ptr()
            allocation = None
            for segment in snapshots:
                if segment["device"] != tensor.device.index:
                    continue
                for block in segment["blocks"]:
                    if block["address"] <= pointer < block["address"] + block["size"]:
                        allocation = {
                            "stream": segment["stream"],
                            "pool_id": segment.get("segment_pool_id"),
                            "address": block["address"],
                            "size": block["size"],
                            "state": block["state"],
                        }
                        break
                if allocation is not None:
                    break
            tensors.append(
                {
                    "boundary": boundary,
                    "index": index,
                    "shape": tuple(tensor.shape),
                    "strides": tuple(tensor.stride()),
                    "data_ptr": pointer,
                    "allocation": allocation,
                }
            )
        module_id, _ = self._identity(module)
        name = type(module).__name__
        layer = getattr(module, "layer_id", "unknown")
        record = {
            "kind": "allocator_stream_metadata",
            "module_class": name,
            "layer_id": layer,
            "producer_stream": int(producer.cuda_stream),
            "tensors": tensors,
        }
        path = (
            self.output_dir
            / f"{name}-layer{layer}-module{module_id:03d}-allocations.json"
        )
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(record, indent=2))
        os.replace(temporary, path)
        self._allocations_logged.add(id(module))

    def __call__(self, module, inputs, output):
        self._raise_error()
        if self._closed:
            raise RuntimeError("fingerprint hook is closed")
        selection = self._selected(module, inputs, output)
        if not selection:
            return None
        selected, rows = selection
        if rows < self._min_rows:
            return None
        selected = [
            item
            for item in selected
            if item[2].is_cuda and item[2].dtype in _SUPPORTED_DTYPES
        ]
        if not selected:
            return None
        device = selected[0][2].device
        selected = [item for item in selected if item[2].device == device]
        if torch.cuda.is_current_stream_capturing():
            return None

        producer = torch.cuda.current_stream(device)
        if self._control is not None:
            # Isolate allocator bookkeeping from the callback itself. Neither
            # control launches hashes, records events, or queues writer work.
            if self._control == "record-stream":
                for _, _, tensor in selected:
                    tensor.record_stream(producer)
            elif self._control == "allocator":
                self._write_allocator_metadata(module, selected, producer)
            return None
        module_id, call = self._identity(module)
        metadata = []
        gpu_hashes = []
        for boundary, index, tensor in selected:
            sample_stride = (
                max(1, triton.cdiv(tensor.numel(), self._max_elements))
                if self._max_elements is not None
                else 1
            )
            metadata.append(
                {
                    "boundary": boundary,
                    "index": index,
                    "shape": tuple(tensor.shape),
                    "dtype": str(tensor.dtype),
                    "strides": tuple(tensor.stride()),
                    "storage_offset": tensor.storage_offset(),
                    "block_elements": _BLOCK_ELEMENTS,
                    "sample_stride": sample_stride,
                    "sampled_elements": triton.cdiv(tensor.numel(), sample_stride),
                }
            )
            gpu_hashes.append(_fingerprint(tensor, sample_stride))
            tensor.record_stream(producer)

        ready = torch.cuda.Event(enable_timing=False)
        ready.record(producer)
        name = type(module).__name__
        layer = getattr(module, "layer_id", "unknown")
        safe_layer = str(layer).replace(os.sep, "_")
        record = {
            "version": 1,
            "kind": "noncryptographic_gpu_block_fingerprint",
            "equivalence_proof": False,
            "limitations": "collisions and unsampled changes are possible; equality does not prove tensor equivalence",
            "hook_phase": self._phase,
            "module_class": name,
            "layer_id": layer,
            "module_id": module_id,
            "call": call,
            "tensors": metadata,
        }
        path = self.output_dir / (
            f"{name}-layer{safe_layer}-module{module_id:03d}-call{call:04d}.pt"
        )
        try:
            self._queue.put_nowait((record, gpu_hashes, ready, device, path))
        except queue.Full as error:
            raise RuntimeError("fingerprint writer queue is full") from error
        self._submitted += 1
        if self._submitted >= self._defer_records:
            self._writer_gate.set()
        return None

    def _writer_loop(self):
        self._writer_gate.wait()
        while True:
            job = self._queue.get()
            try:
                if job is None:
                    return
                if self._error is not None:
                    continue
                record, gpu_hashes, ready, device, path = job
                if self._copy_stream is None:
                    self._copy_device = device
                    self._copy_stream = torch.cuda.Stream(device=device)
                if device != self._copy_device:
                    raise RuntimeError("one fingerprint hook cannot span CUDA devices")
                cpu_hashes = []
                with torch.cuda.device(device), torch.cuda.stream(self._copy_stream):
                    self._copy_stream.wait_event(ready)
                    for gpu_hash in gpu_hashes:
                        cpu_hash = torch.empty_like(
                            gpu_hash, device="cpu", pin_memory=True
                        )
                        cpu_hash.copy_(gpu_hash, non_blocking=True)
                        gpu_hash.record_stream(self._copy_stream)
                        cpu_hashes.append(cpu_hash)
                    copied = torch.cuda.Event(enable_timing=False)
                    copied.record(self._copy_stream)
                copied.synchronize()
                for metadata, cpu_hash in zip(record["tensors"], cpu_hashes):
                    metadata["checksums"] = cpu_hash
                temporary = path.with_suffix(path.suffix + ".tmp")
                try:
                    torch.save(record, temporary)
                    os.replace(temporary, path)
                finally:
                    temporary.unlink(missing_ok=True)
            except BaseException as error:
                if self._error is None:
                    self._error = error
            finally:
                self._queue.task_done()

    def flush(self):
        self._writer_gate.set()
        self._queue.join()
        self._raise_error()

    def close(self):
        if not self._closed:
            self._writer_gate.set()
            self._queue.join()
            self._closed = True
            self._queue.put_nowait(None)
            self._writer.join(timeout=10)
        self._raise_error()

    def _close_at_exit(self):
        already_reported = self._error_reported
        try:
            self.close()
        except BaseException as error:
            if already_reported:
                return
            try:
                print(
                    f"fingerprint writer close failed: {error}",
                    file=sys.stderr,
                    flush=True,
                )
            except BaseException:
                pass


def make_fingerprint_hook(config):
    return _FingerprintHook(config)


class _FingerprintPreHook(_FingerprintHook):
    _phase = "before_forward"

    def __init__(self, config):
        config = dict(config)
        classes = config.get("module_classes")
        config["module_classes"] = _BOUNDARIES if classes is None else classes
        if not set(config["module_classes"]) <= _BOUNDARIES:
            raise ValueError(
                "pre-forward fingerprints require positional-input boundaries"
            )
        super().__init__(config)

    def _selected(self, module, inputs, output):
        tensors = list(_tensors(inputs))
        if not tensors:
            return []
        selection = super()._selected(module, inputs, tensors[0])
        if not selection:
            return []
        selected, rows = selection
        return [item for item in selected if item[0] == "input"], rows

    def __call__(self, module, inputs):
        return super().__call__(module, inputs, None)


def make_fingerprint_pre_hook(config):
    return _FingerprintPreHook(config)

"""Bounded pre-GDN diagnostic for cold Qwen3.5 SSM cache rows.

The hook is intentionally model-specific.  It observes the layer-0 state on the
current producer stream, never changes model tensors, and defers all host reads
and persistence to one background writer release.
"""

from __future__ import annotations

import atexit
import itertools
import json
import os
import sys
import threading
from pathlib import Path

import torch
import triton
import triton.language as tl


_BLOCK_ELEMENTS = 4096
_SUMMARY_WIDTH = 8
_MAX_RECORDS = 64
_TARGET_CLASS = "Qwen3_5GatedDeltaNet"
_factory_ids = itertools.count()

_ROW = 0
_SLOT = 1
_REQUEST = 2
_GPU_PREFIX = 3
_GPU_EXTEND = 4
_INVALID_MASK = 5
_NONZERO = 6
_NONFINITE = 7

_INVALID_SLOT = 1
_INVALID_REQUEST = 2
_PREFIX_MISMATCH = 4
_EXTEND_MISMATCH = 8


@triton.jit
def _cold_state_scan_kernel(
    state,
    slots,
    request_indices,
    prefix_lens,
    extend_lens,
    partials,
    summaries,
    batch_row,
    record_index,
    slot_count,
    request_pool_size,
    expected_prefix,
    expected_extend,
    state_elements,
    size1,
    size2,
    size3,
    stride0,
    stride1,
    stride2,
    stride3,
    max_blocks,
    BLOCK: tl.constexpr,
    SUMMARY_WIDTH: tl.constexpr,
):
    block = tl.program_id(0)
    offsets = block * BLOCK + tl.arange(0, BLOCK)

    slot = tl.load(slots + batch_row).to(tl.int64)
    request_index = tl.load(request_indices + batch_row).to(tl.int64)
    gpu_prefix = tl.load(prefix_lens + batch_row).to(tl.int64)
    gpu_extend = tl.load(extend_lens + batch_row).to(tl.int64)

    slot_valid = (slot >= 1) & (slot < slot_count)
    request_valid = (request_index >= 1) & (request_index <= request_pool_size)
    prefix_matches = gpu_prefix == expected_prefix
    extend_matches = gpu_extend == expected_extend
    coverage_valid = slot_valid & request_valid & prefix_matches & extend_matches

    if block == 0:
        invalid_mask = (
            tl.where(slot_valid, 0, 1)
            | tl.where(request_valid, 0, 2)
            | tl.where(prefix_matches, 0, 4)
            | tl.where(extend_matches, 0, 8)
        )
        summary = summaries + record_index * SUMMARY_WIDTH
        tl.store(summary, batch_row)
        tl.store(summary + 1, slot)
        tl.store(summary + 2, request_index)
        tl.store(summary + 3, gpu_prefix)
        tl.store(summary + 4, gpu_extend)
        tl.store(summary + 5, invalid_mask)

    logical = offsets.to(tl.int64)
    index3 = logical % size3
    remaining = logical // size3
    index2 = remaining % size2
    remaining //= size2
    index1 = remaining % size1
    safe_slot = tl.where(slot_valid, slot, 0)
    physical = (
        safe_slot * stride0 + index1 * stride1 + index2 * stride2 + index3 * stride3
    )
    mask = (logical < state_elements) & coverage_valid
    value = tl.load(state + physical, mask=mask, other=0.0).to(tl.float32)
    nonzero = mask & (value != 0.0)
    finite = (value == value) & (value != float("inf")) & (value != -float("inf"))
    nonfinite = mask & ~finite
    partial = partials + (record_index * max_blocks + block) * 2
    tl.store(partial, tl.sum(nonzero.to(tl.int32), axis=0))
    tl.store(partial + 1, tl.sum(nonfinite.to(tl.int32), axis=0))


@triton.jit
def _cold_state_finalize_kernel(
    partials,
    summaries,
    record_index,
    block_count,
    max_blocks,
    REDUCE_BLOCK: tl.constexpr,
    SUMMARY_WIDTH: tl.constexpr,
):
    offsets = tl.arange(0, REDUCE_BLOCK)
    mask = offsets < block_count
    base = partials + record_index * max_blocks * 2
    nonzero = tl.load(base + offsets * 2, mask=mask, other=0)
    nonfinite = tl.load(base + offsets * 2 + 1, mask=mask, other=0)
    summary = summaries + record_index * SUMMARY_WIDTH
    tl.store(summary + 6, tl.sum(nonzero, axis=0).to(tl.int64))
    tl.store(summary + 7, tl.sum(nonfinite, axis=0).to(tl.int64))


def _get_attn_backend():
    # Kept behind a tiny boundary so importing this diagnostic does not import
    # the model executor or initialize any backend.
    from sglang.srt.model_executor.forward_context import get_attn_backend

    return get_attn_backend()


def _is_compiling_or_tracing() -> bool:
    compiler = getattr(torch, "compiler", None)
    if compiler is not None and compiler.is_compiling():
        return True
    return torch.jit.is_tracing()


def _is_exact_extend(forward_mode) -> bool:
    mode_type = type(forward_mode)
    return (
        mode_type.__name__ == "ForwardMode"
        and mode_type.__module__ == "sglang.srt.model_executor.forward_batch_info"
        and getattr(forward_mode, "name", None) == "EXTEND"
    )


def _require_positive_int(config, name, default, maximum=None):
    value = config.get(name, default)
    if type(value) is not int or value < 1 or (maximum is not None and value > maximum):
        suffix = f" at most {maximum}" if maximum is not None else ""
        raise ValueError(f"cold-state {name} must be a positive integer{suffix}")
    return value


def _decode_summary(values, state_shape, state_strides, state_elements):
    invalid_mask = int(values[_INVALID_MASK])
    errors = []
    if invalid_mask & _INVALID_SLOT:
        errors.append("slot_out_of_range")
    if invalid_mask & _INVALID_REQUEST:
        errors.append("request_index_out_of_range")
    if invalid_mask & _PREFIX_MISMATCH:
        errors.append("prefix_mismatch")
    if invalid_mask & _EXTEND_MISMATCH:
        errors.append("extend_length_mismatch")
    return {
        "batch_row": int(values[_ROW]),
        "physical_slot": int(values[_SLOT]),
        "request_index": int(values[_REQUEST]),
        "gpu_prefix_len": int(values[_GPU_PREFIX]),
        "gpu_extend_len": int(values[_GPU_EXTEND]),
        "coverage_valid": not errors,
        "coverage_errors": errors,
        "nonzero_count": int(values[_NONZERO]),
        "nonfinite_count": int(values[_NONFINITE]),
        "state_elements": int(state_elements),
        "state_shape": list(state_shape),
        "state_strides": list(state_strides),
    }


def _launch_scan(
    *,
    state,
    slots,
    request_indices,
    prefix_lens,
    extend_lens,
    batch_row,
    expected_prefix,
    expected_extend,
    request_pool_size,
    partials,
    summaries,
    record_index,
):
    state_elements = state.shape[1] * state.shape[2] * state.shape[3]
    block_count = triton.cdiv(state_elements, _BLOCK_ELEMENTS)
    max_blocks = partials.shape[1]
    _cold_state_scan_kernel[(block_count,)](
        state,
        slots,
        request_indices,
        prefix_lens,
        extend_lens,
        partials,
        summaries,
        batch_row,
        record_index,
        state.shape[0],
        request_pool_size,
        expected_prefix,
        expected_extend,
        state_elements,
        state.shape[1],
        state.shape[2],
        state.shape[3],
        state.stride(0),
        state.stride(1),
        state.stride(2),
        state.stride(3),
        max_blocks,
        BLOCK=_BLOCK_ELEMENTS,
        SUMMARY_WIDTH=_SUMMARY_WIDTH,
        num_warps=8,
    )
    reduce_block = triton.next_power_of_2(block_count)
    _cold_state_finalize_kernel[(1,)](
        partials,
        summaries,
        record_index,
        block_count,
        max_blocks,
        REDUCE_BLOCK=reduce_block,
        SUMMARY_WIDTH=_SUMMARY_WIDTH,
        num_warps=4,
    )


class _ColdStateHook:
    def __init__(self, config):
        self._min_rows = _require_positive_int(config, "min_rows", 8192)
        self._max_records = _require_positive_int(
            config, "max_records", 8, maximum=_MAX_RECORDS
        )
        arm_file = config.get("arm_file")
        if not isinstance(arm_file, (str, os.PathLike)) or not str(arm_file):
            raise ValueError("cold-state arm_file is required")
        sample_dir = config.get("sample_dir")
        if not isinstance(sample_dir, (str, os.PathLike)) or not str(sample_dir):
            raise ValueError("cold-state sample_dir is required")

        self._arm_file = Path(arm_file)
        self.output_dir = Path(sample_dir) / (
            f"pid{os.getpid()}-factory{next(_factory_ids)}"
        )
        self.output_dir.mkdir(parents=True, exist_ok=False)
        self.output_path = self.output_dir / "cold-state.json"

        self._armed = False
        self._closed = False
        self._finalized = False
        self._final_reason = None
        self._count = 0
        self._host_records = []
        self._retained_refs = []
        self._linear_backend = None
        self._pool = None
        self._state = None
        self._state_shape = None
        self._state_strides = None
        self._state_elements = None
        self._partials = None
        self._summaries = None
        self._producer_stream = None
        self._producer_stream_id = None
        self._device = None
        self._ready = None
        self._writer_error = None
        self._error_reported = False
        self._writer_gate = threading.Event()
        self._writer = threading.Thread(
            target=self._writer_loop,
            name=f"qwen38-cold-state-{os.getpid()}",
            daemon=True,
        )
        self._writer.start()
        atexit.register(self._close_at_exit)

    def _raise_error(self):
        if self._writer_error is not None:
            self._error_reported = True
            raise RuntimeError(
                f"cold-state writer failed: {self._writer_error}"
            ) from self._writer_error

    def _is_armed(self):
        if not self._armed and self._arm_file.exists():
            self._armed = True
        return self._armed

    def _selected(self, module, inputs):
        if (
            type(module).__name__ != _TARGET_CLASS
            or getattr(module, "layer_id", None) != 0
        ):
            return False
        if not isinstance(inputs, tuple) or len(inputs) < 2:
            return False
        hidden_states, forward_batch = inputs[:2]
        if not isinstance(hidden_states, torch.Tensor) or hidden_states.ndim == 0:
            return False
        if hidden_states.shape[0] < self._min_rows:
            return False
        return _is_exact_extend(getattr(forward_batch, "forward_mode", None))

    def _cold_rows(self, forward_batch):
        batch_size = getattr(forward_batch, "batch_size", None)
        if type(batch_size) is not int or batch_size < 0:
            raise RuntimeError("cold-state batch_size metadata is invalid")
        prefixes = getattr(forward_batch, "extend_prefix_lens_cpu", None)
        lengths = getattr(forward_batch, "extend_seq_lens_cpu", None)
        if not isinstance(prefixes, (list, tuple)):
            raise RuntimeError("cold-state CPU prefix metadata is unavailable")
        if not isinstance(lengths, (list, tuple)):
            raise RuntimeError("cold-state CPU extend metadata is unavailable")
        if len(prefixes) < batch_size:
            raise RuntimeError(
                "cold-state CPU prefix metadata coverage is shorter than batch_size"
            )
        if len(lengths) < batch_size:
            raise RuntimeError(
                "cold-state CPU extend metadata coverage is shorter than batch_size"
            )
        original = getattr(forward_batch, "_original_batch_size", None)
        real_batch_size = batch_size if original is None else original
        if type(real_batch_size) is not int or not 0 <= real_batch_size <= batch_size:
            raise RuntimeError("cold-state original batch size is invalid")
        rows = []
        for row in range(batch_size):
            prefix, extend = prefixes[row], lengths[row]
            if type(prefix) is not int or type(extend) is not int:
                raise RuntimeError("cold-state CPU lengths must contain plain integers")
            if prefix < 0 or extend < 0:
                raise RuntimeError("cold-state CPU lengths must be non-negative")
            if row < real_batch_size and prefix == 0 and extend > 0:
                rows.append((row, prefix, extend))
        return rows

    def _bind_static_state(self, linear_backend, layer_id):
        pool = getattr(linear_backend, "req_to_token_pool", None)
        if pool is None:
            raise RuntimeError("cold-state linear backend has no request pool")
        if getattr(pool, "layer_transfer_counter", None) is not None:
            raise RuntimeError(
                "cold-state probe does not support HiCache transfer-gated Mamba pools"
            )
        mamba_map = getattr(pool, "mamba_map", None)
        if not isinstance(mamba_map, dict) or layer_id not in mamba_map:
            raise RuntimeError("cold-state layer is absent from the static Mamba map")
        mamba_pool = getattr(pool, "mamba_pool", None)
        mamba_cache = getattr(mamba_pool, "mamba_cache", None)
        temporal = getattr(mamba_cache, "temporal", None)
        if not isinstance(temporal, torch.Tensor) or temporal.ndim != 5:
            raise RuntimeError(
                "cold-state temporal cache must be a five-dimensional tensor"
            )
        layer_index = mamba_map[layer_id]
        if type(layer_index) is not int or not 0 <= layer_index < temporal.shape[0]:
            raise RuntimeError("cold-state Mamba layer index is invalid")
        state = temporal[layer_index]
        if not state.is_cuda or state.dtype != torch.float32 or state.ndim != 4:
            raise RuntimeError(
                "cold-state SSM cache must be a CUDA FP32 [S,H,V,K] tensor"
            )
        if min(state.shape) < 1:
            raise RuntimeError("cold-state SSM cache dimensions must be non-empty")
        request_pool_size = getattr(pool, "size", None)
        if type(request_pool_size) is not int or request_pool_size < 1:
            raise RuntimeError("cold-state request pool size is invalid")

        self._linear_backend = linear_backend
        self._pool = pool
        self._state = state
        self._state_shape = tuple(state.shape)
        self._state_strides = tuple(state.stride())
        self._state_elements = state.shape[1] * state.shape[2] * state.shape[3]
        block_count = triton.cdiv(self._state_elements, _BLOCK_ELEMENTS)
        if triton.next_power_of_2(block_count) > 65536:
            raise RuntimeError(
                "cold-state SSM row exceeds the bounded reduction contract"
            )
        self._partials = torch.zeros(
            (self._max_records, block_count, 2),
            dtype=torch.int32,
            device=state.device,
        )
        self._summaries = torch.zeros(
            (self._max_records, _SUMMARY_WIDTH),
            dtype=torch.int64,
            device=state.device,
        )
        return request_pool_size

    def _validate_gpu_metadata(self, tensor, name, batch_size, device):
        if (
            not isinstance(tensor, torch.Tensor)
            or not tensor.is_cuda
            or tensor.ndim != 1
            or tensor.shape[0] < batch_size
            or tensor.device != device
            or tensor.stride(0) != 1
            or tensor.dtype not in (torch.int32, torch.int64)
        ):
            raise RuntimeError(
                f"cold-state GPU {name} must be a contiguous CUDA int vector covering the batch"
            )

    def _resolve_runtime(self, forward_batch, hidden_states):
        attn_backend = _get_attn_backend()
        linear_backend = getattr(attn_backend, "linear_attn_backend", None)
        if linear_backend is None:
            raise RuntimeError(
                "cold-state active attention backend has no linear backend"
            )
        if self._linear_backend is None:
            request_pool_size = self._bind_static_state(linear_backend, 0)
        else:
            if linear_backend is not self._linear_backend:
                raise RuntimeError(
                    "cold-state active linear backend changed during collection"
                )
            request_pool_size = self._pool.size
        if hidden_states.device != self._state.device:
            raise RuntimeError(
                "cold-state hidden states and SSM cache are on different devices"
            )

        metadata = getattr(linear_backend, "forward_metadata", None)
        slots = getattr(metadata, "mamba_cache_indices", None)
        batch_size = forward_batch.batch_size
        tensors = {
            "physical slots": slots,
            "request indices": getattr(forward_batch, "req_pool_indices", None),
            "prefix lengths": getattr(forward_batch, "extend_prefix_lens", None),
            "extend lengths": getattr(forward_batch, "extend_seq_lens", None),
        }
        for name, tensor in tensors.items():
            self._validate_gpu_metadata(tensor, name, batch_size, self._state.device)
        return (
            request_pool_size,
            tensors["physical slots"],
            tensors["request indices"],
            tensors["prefix lengths"],
            tensors["extend lengths"],
        )

    def _validate_producer(self, device):
        producer = torch.cuda.current_stream(device)
        producer_id = int(producer.cuda_stream)
        if self._producer_stream is None:
            self._producer_stream = producer
            self._producer_stream_id = producer_id
            self._device = device
        elif device != self._device or producer_id != self._producer_stream_id:
            raise RuntimeError("cold-state records must use one CUDA producer stream")
        return producer

    def __call__(self, module, inputs):
        self._raise_error()
        if self._closed:
            raise RuntimeError("cold-state hook is closed")
        if self._finalized or not self._selected(module, inputs):
            return None
        if not self._is_armed():
            return None
        if _is_compiling_or_tracing():
            return None
        hidden_states, forward_batch = inputs[:2]
        if not hidden_states.is_cuda or torch.cuda.is_current_stream_capturing():
            return None
        cold_rows = self._cold_rows(forward_batch)
        if not cold_rows:
            return None
        runtime = self._resolve_runtime(forward_batch, hidden_states)
        request_pool_size, slots, requests, prefixes, extends = runtime
        producer = self._validate_producer(hidden_states.device)

        remaining = self._max_records - self._count
        for batch_row, expected_prefix, expected_extend in cold_rows[:remaining]:
            record_index = self._count
            _launch_scan(
                state=self._state,
                slots=slots,
                request_indices=requests,
                prefix_lens=prefixes,
                extend_lens=extends,
                batch_row=batch_row,
                expected_prefix=expected_prefix,
                expected_extend=expected_extend,
                request_pool_size=request_pool_size,
                partials=self._partials,
                summaries=self._summaries,
                record_index=record_index,
            )
            self._host_records.append(
                {
                    "batch_row": batch_row,
                    "cpu_prefix_len": expected_prefix,
                    "cpu_extend_len": expected_extend,
                }
            )
            self._retained_refs.append((slots, requests, prefixes, extends))
            self._count += 1
        if self._count == self._max_records:
            self._dispatch("cap", producer)
        return None

    def _dispatch(self, reason, producer=None):
        if self._finalized:
            return
        self._finalized = True
        self._final_reason = reason
        if self._count:
            if producer is None:
                producer = self._producer_stream
            if producer is None:
                raise RuntimeError("cold-state producer stream is unavailable")
            ready = torch.cuda.Event(enable_timing=False)
            ready.record(producer)
            self._ready = ready
        self._writer_gate.set()

    def _writer_loop(self):
        self._writer_gate.wait()
        try:
            if self._ready is not None:
                self._ready.synchronize()
                values = self._summaries[: self._count].cpu().tolist()
            else:
                values = []
            records = []
            for host, summary in zip(self._host_records, values):
                record = _decode_summary(
                    summary,
                    self._state_shape,
                    self._state_strides,
                    self._state_elements,
                )
                if record["batch_row"] != host["batch_row"]:
                    raise RuntimeError("GPU summary row does not match queued cold row")
                record.update(host)
                records.append(record)
            cap_reached = self._count == self._max_records
            payload = {
                "version": 1,
                "kind": "pre_gdn_cold_ssm_state_observation",
                "hook_phase": "before_forward",
                "module_class": _TARGET_CLASS,
                "layer_id": 0,
                "expected_records": self._max_records,
                "persisted_records": len(records),
                "coverage": {
                    "status": "complete" if cap_reached else "partial",
                    "cap_reached": cap_reached,
                    "bounded_cap": self._max_records,
                    "finalized_by": self._final_reason,
                    "all_records_valid": all(
                        record["coverage_valid"] for record in records
                    ),
                },
                "state": (
                    None
                    if self._state is None
                    else {
                        "dtype": str(self._state.dtype),
                        "shape": list(self._state_shape),
                        "strides": list(self._state_strides),
                        "elements_per_slot": self._state_elements,
                    }
                ),
                "limitations": (
                    "Zero observations are not proof of initialization causality. "
                    "Collection stops at the configured cap; the final event is "
                    "recorded before GDN layer 0 but may overlap later layer work."
                ),
                "records": records,
            }
            temporary = self.output_path.with_suffix(".json.tmp")
            try:
                temporary.write_text(json.dumps(payload, indent=2) + "\n")
                os.replace(temporary, self.output_path)
            finally:
                temporary.unlink(missing_ok=True)
            self._retained_refs.clear()
        except BaseException as error:
            self._writer_error = error

    def flush(self):
        self._raise_error()
        self._dispatch("flush")
        self._writer.join(timeout=30)
        if self._writer.is_alive():
            raise RuntimeError("cold-state writer did not finish within 30 seconds")
        self._raise_error()

    def close(self):
        if not self._closed:
            self._dispatch("close")
            self._writer.join(timeout=30)
            self._closed = True
            if self._writer.is_alive():
                raise RuntimeError("cold-state writer did not finish within 30 seconds")
        self._raise_error()

    def _close_at_exit(self):
        already_reported = self._error_reported
        try:
            self.close()
        except BaseException as error:
            if not already_reported:
                try:
                    print(
                        f"cold-state writer close failed: {error}",
                        file=sys.stderr,
                        flush=True,
                    )
                except BaseException:
                    pass


def make_cold_state_hook(config):
    return _ColdStateHook(config)


def _run_scan_for_test(
    state,
    *,
    slot,
    request_index,
    gpu_prefix,
    gpu_extend,
    expected_prefix,
    expected_extend,
    request_pool_size,
):
    """Exercise the production kernel without constructing model runtime state."""
    if state.ndim != 4 or not state.is_cuda or state.dtype != torch.float32:
        raise ValueError("test state must be a CUDA FP32 [S,H,V,K] tensor")
    device = state.device
    slots = torch.tensor([slot], dtype=torch.int64, device=device)
    requests = torch.tensor([request_index], dtype=torch.int64, device=device)
    prefixes = torch.tensor([gpu_prefix], dtype=torch.int64, device=device)
    extends = torch.tensor([gpu_extend], dtype=torch.int64, device=device)
    state_elements = state.shape[1] * state.shape[2] * state.shape[3]
    blocks = triton.cdiv(state_elements, _BLOCK_ELEMENTS)
    partials = torch.zeros((1, blocks, 2), dtype=torch.int32, device=device)
    summaries = torch.zeros((1, _SUMMARY_WIDTH), dtype=torch.int64, device=device)
    _launch_scan(
        state=state,
        slots=slots,
        request_indices=requests,
        prefix_lens=prefixes,
        extend_lens=extends,
        batch_row=0,
        expected_prefix=expected_prefix,
        expected_extend=expected_extend,
        request_pool_size=request_pool_size,
        partials=partials,
        summaries=summaries,
        record_index=0,
    )
    values = summaries[0].cpu().tolist()
    return _decode_summary(values, state.shape, state.stride(), state_elements)

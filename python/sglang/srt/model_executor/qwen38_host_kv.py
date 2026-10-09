# Copyright 2026 Qwen38 native contributors. SPDX-License-Identifier: Apache-2.0
"""Mapped-host ordinary FP8/BF16 KV storage for the Qwen3.8 SM120 path.

Each :class:`HostKVArena` is independent.  The target model (12 ordinary
attention layers) and one-layer draft model must therefore own separate arena
instances; compressed QSA state is not stored here.

Scatter and gather enqueue native CUDA kernels and never copy IDs through the
CPU. Gather outputs are caller-owned packed ``[rows, heads, head_dim]`` tensors
in the arena's storage dtype. Optional selected-gather dedup owns one fixed
CUDA row-map for the arena's full lifetime. A negative gather ID is padding and
produces zeros. Positive OOB gather IDs and all negative/OOB scatter IDs are
made safe in the kernel and reported by :meth:`check_errors`; ``debug=True``
checks every eager call.
Reserved physical slot zero starts as zero for every layer; all other slots
must be scattered before gather.

Before capturing, obtain a :meth:`retain_for_graph` lease and keep both the
lease and arena through graph destruction and completion of its last replay.
Lease release quiesces the device before the lifetime pin is dropped.  Graphs
using one arena must be replayed serially on one stream.  These rules are
necessary because CUDA graphs retain raw mapped-host pointers.
"""

from __future__ import annotations

import ctypes as C
import os
import threading
from functools import lru_cache
from pathlib import Path

import torch

_UINT64_MAX = (1 << 64) - 1
_SIZE_T_MAX = (1 << (8 * C.sizeof(C.c_size_t))) - 1
_NATIVE_ABI_VERSION = 2
_STORAGE_ELEMENT_BYTES = {
    torch.float8_e4m3fn: 1,
    torch.bfloat16: 2,
}


def _positive_u64(name: str, value: int) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not 0 < value <= _UINT64_MAX
    ):
        raise ValueError(f"{name} must be a positive uint64")
    return value


def required_bytes(
    layer_count: int,
    slot_count: int,
    heads: int = 2,
    head_dim: int = 256,
    *,
    dtype: torch.dtype = torch.bfloat16,
) -> int:
    """Return exact mapped-host payload bytes, rejecting invalid/overflow geometry."""
    try:
        element_bytes = _STORAGE_ELEMENT_BYTES[dtype]
    except KeyError as exc:
        raise ValueError(
            "host KV dtype must be torch.float8_e4m3fn or torch.bfloat16"
        ) from exc
    values = (
        _positive_u64("layer_count", layer_count),
        _positive_u64("slot_count", slot_count),
        _positive_u64("heads", heads),
        _positive_u64("head_dim", head_dim),
    )
    result = 1
    for value in (*values[:2], 2, *values[2:], element_bytes):
        if result > _UINT64_MAX // value:
            raise OverflowError("host KV byte size overflows uint64")
        result *= value
    if result > _SIZE_T_MAX:
        raise OverflowError("host KV byte size overflows size_t")
    return result


def _declare(lib, name, result, *args):
    function = getattr(lib, name)
    function.restype = result
    function.argtypes = args
    return function


def _validate_library_abi(lib) -> None:
    try:
        version = getattr(lib, "q38_host_kv_abi_version")
    except AttributeError as exc:
        raise RuntimeError(
            "host KV native ABI mismatch: version symbol is missing; rebuild "
            "libq38_host_kv.so"
        ) from exc
    version.restype = C.c_uint32
    version.argtypes = []
    actual = int(version())
    if actual != _NATIVE_ABI_VERSION:
        raise RuntimeError(
            "host KV native ABI mismatch: "
            f"Python requires {_NATIVE_ABI_VERSION}, library reports {actual}"
        )


@lru_cache(maxsize=2)
def library(selected_dedup: bool = False):
    if not isinstance(selected_dedup, bool):
        raise ValueError("selected_dedup must be a bool")
    root = Path(__file__).resolve().parents[4] / "native" / "qwen38"
    path = Path(
        os.environ.get("QWEN38_HOST_KV_LIBRARY", root / "build" / "libq38_host_kv.so")
    )
    lib = C.CDLL(str(path))
    _validate_library_abi(lib)
    pointer, u64, stream = C.c_void_p, C.c_uint64, C.c_size_t
    _declare(lib, "q38_host_kv_last_error", C.c_char_p)
    _declare(
        lib,
        "q38_host_kv_required_bytes",
        C.c_int,
        u64,
        u64,
        u64,
        u64,
        u64,
        C.POINTER(u64),
    )
    _declare(
        lib,
        "q38_host_kv_create",
        pointer,
        u64,
        u64,
        u64,
        u64,
        u64,
        u64,
        C.c_int,
    )
    _declare(lib, "q38_host_kv_allocated_bytes", u64, pointer)
    _declare(
        lib,
        "q38_host_kv_scatter",
        C.c_int,
        pointer,
        u64,
        pointer,
        u64,
        pointer,
        u64,
        pointer,
        u64,
        C.c_int,
        stream,
    )
    _declare(
        lib,
        "q38_host_kv_gather",
        C.c_int,
        pointer,
        u64,
        pointer,
        u64,
        C.c_int,
        pointer,
        pointer,
        stream,
    )
    if selected_dedup:
        try:
            _declare(
                lib,
                "q38_host_kv_gather_dedup",
                C.c_int,
                pointer,
                u64,
                pointer,
                u64,
                C.c_int,
                pointer,
                u64,
                pointer,
                pointer,
                stream,
            )
        except AttributeError as exc:
            raise RuntimeError(
                "selected host KV library does not provide optional dedup gather"
            ) from exc
    _declare(lib, "q38_host_kv_check", C.c_int, pointer)
    _declare(lib, "q38_host_kv_close", C.c_int, pointer)
    return lib


def _check(status: int, lib) -> None:
    if status != 0:
        message = lib.q38_host_kv_last_error()
        raise RuntimeError(message.decode() if message else "native host KV error")


def _device_index(device) -> int:
    if device is None:
        return torch.cuda.current_device()
    if isinstance(device, int) and not isinstance(device, bool):
        return device
    resolved = torch.device(device)
    if resolved.type != "cuda":
        raise ValueError("host KV arena requires a CUDA device")
    return torch.cuda.current_device() if resolved.index is None else resolved.index


class HostKVGraphLease:
    """Explicit lifetime pin held until a captured graph is destroyed/quiescent."""

    def __init__(self, arena: HostKVArena):
        self._arena = arena

    def close(self) -> None:
        arena = self._arena
        if arena is None:
            return
        with arena._lock:
            # A shared event cannot identify the latest replay among several
            # independently captured graphs.  Lease release is the cold
            # lifetime boundary, so quiesce before allowing mapped backing to
            # be unmapped.
            torch.cuda.synchronize(arena.device)
            arena._graph_refs -= 1
        self._arena = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


class HostKVArena:
    """Own one byte-exact mapped-host FP8/BF16 arena and its native handle."""

    def __init__(
        self,
        layer_count: int,
        slot_count: int,
        *,
        byte_budget: int,
        heads: int = 2,
        head_dim: int = 256,
        dtype: torch.dtype = torch.bfloat16,
        device=None,
        debug: bool = False,
        label: str = "host-kv",
        selected_dedup: bool = False,
    ):
        self.layer_count = _positive_u64("layer_count", layer_count)
        self.slot_count = _positive_u64("slot_count", slot_count)
        self.heads = _positive_u64("heads", heads)
        self.head_dim = _positive_u64("head_dim", head_dim)
        try:
            self.element_bytes = _STORAGE_ELEMENT_BYTES[dtype]
        except KeyError as exc:
            raise ValueError(
                "host KV dtype must be torch.float8_e4m3fn or torch.bfloat16"
            ) from exc
        self.dtype = dtype
        self.nbytes = required_bytes(
            layer_count, slot_count, heads, head_dim, dtype=dtype
        )
        self.pinned_bytes = self.nbytes
        if not isinstance(selected_dedup, bool):
            raise ValueError("selected_dedup must be a bool")
        self.selected_dedup = selected_dedup
        self._selected_row_map = None
        self.gpu_bytes = 24  # Native deferred-error record; KV payload is host-only.
        if (
            isinstance(byte_budget, bool)
            or not isinstance(byte_budget, int)
            or not 0 < byte_budget <= _UINT64_MAX
        ):
            raise ValueError("byte_budget must be an explicit positive uint64")
        if self.nbytes > byte_budget:
            raise MemoryError(
                f"{label} needs {self.nbytes} bytes, exceeding budget {byte_budget}"
            )
        self.byte_budget = byte_budget
        self.row_elements = self.heads * self.head_dim
        self.row_bytes = self.row_elements * self.element_bytes
        self.device = _device_index(device)
        if torch.cuda.get_device_capability(self.device) != (12, 0):
            raise ValueError("Qwen38 host KV is validated only for SM120")
        self.debug = bool(debug)
        self.label = str(label)
        self._lock = threading.RLock()
        self._graph_refs = 0
        self._lib = library(selected_dedup=self.selected_dedup)
        self._handle = self._lib.q38_host_kv_create(
            self.layer_count,
            self.slot_count,
            self.heads,
            self.head_dim,
            self.element_bytes,
            self.byte_budget,
            self.device,
        )
        if not self._handle:
            _check(-1, self._lib)
        allocated = self._lib.q38_host_kv_allocated_bytes(self._handle)
        if allocated != self.nbytes:
            # This is an ABI mismatch, so do not leave an unknown allocation alive.
            self._lib.q38_host_kv_close(self._handle)
            self._handle = None
            raise RuntimeError(
                f"host KV ABI size mismatch: Python={self.nbytes}, native={allocated}"
            )
        if self.selected_dedup:
            try:
                self._selected_row_map = torch.empty(
                    self.slot_count,
                    dtype=torch.int32,
                    device=f"cuda:{self.device}",
                )
            except Exception:
                # The handle has not escaped and no graph can exist yet. Native
                # close is the required quiescence boundary before dropping any
                # arena-owned device state.
                self._lib.q38_host_kv_close(self._handle)
                self._handle = None
                raise
            self.gpu_bytes += self._selected_row_map.numel() * 4

    @classmethod
    def required_bytes(
        cls,
        layer_count: int,
        slot_count: int,
        heads: int = 2,
        head_dim: int = 256,
        *,
        dtype: torch.dtype = torch.bfloat16,
    ) -> int:
        return required_bytes(layer_count, slot_count, heads, head_dim, dtype=dtype)

    def _require_open(self) -> None:
        if not self._handle:
            raise RuntimeError(f"{self.label} arena is closed")

    @property
    def closed(self) -> bool:
        return not bool(self._handle)

    @property
    def graph_references(self) -> int:
        return self._graph_refs

    def _validate_layer(self, layer: int) -> int:
        if isinstance(layer, bool) or not isinstance(layer, int):
            raise ValueError("host KV layer must be an integer")
        if not 0 <= layer < self.layer_count:
            raise ValueError(
                f"host KV layer {layer} is outside [0, {self.layer_count})"
            )
        return layer

    def _validate_ids(self, ids: torch.Tensor) -> int:
        if ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("host KV IDs must be torch.int32 or torch.int64")
        if ids.ndim != 1 or not ids.is_contiguous():
            raise ValueError("host KV IDs must be a contiguous one-dimensional tensor")
        if not ids.is_cuda or ids.device.index != self.device:
            raise ValueError("host KV IDs must be on the arena CUDA device")
        return ids.element_size()

    def _validate_rows(
        self, tensor: torch.Tensor, count: int, *, packed: bool, name: str
    ) -> int:
        if tensor.dtype != self.dtype:
            raise ValueError(f"{name} must have dtype {self.dtype}")
        if not tensor.is_cuda or tensor.device.index != self.device:
            raise ValueError(f"{name} must be on the arena CUDA device")
        if tensor.ndim < 2 or tensor.shape[0] != count:
            raise ValueError(f"{name} must have leading row count {count}")
        trailing = 1
        expected_stride = 1
        for size, stride in zip(
            reversed(tensor.shape[1:]), reversed(tensor.stride()[1:])
        ):
            trailing *= size
            if size > 1 and stride != expected_stride:
                raise ValueError(f"{name} must be contiguous within each row")
            expected_stride *= size
        if trailing != self.row_elements:
            raise ValueError(
                f"{name} rows must contain exactly {self.row_elements} values"
            )
        row_stride = tensor.stride(0)
        if count > 1 and row_stride < self.row_elements:
            raise ValueError(f"{name} rows may not overlap")
        if packed and not tensor.is_contiguous():
            raise ValueError(f"{name} must be packed contiguous")
        return row_stride * tensor.element_size()

    def _validate_selected_workspace(self) -> torch.Tensor:
        workspace = self._selected_row_map
        if (
            workspace is None
            or workspace.dtype != torch.int32
            or workspace.ndim != 1
            or workspace.numel() != self.slot_count
            or not workspace.is_contiguous()
            or not workspace.is_cuda
            or workspace.device.index != self.device
        ):
            raise ValueError(
                "selected-gather workspace must be the arena-owned packed CUDA "
                f"torch.int32[{self.slot_count}] tensor on device {self.device}"
            )
        return workspace

    @staticmethod
    def _validate_selected_non_aliasing(**tensors: torch.Tensor) -> None:
        ranges = []
        for name, tensor in tensors.items():
            size = tensor.numel() * tensor.element_size()
            if size:
                start = tensor.data_ptr()
                ranges.append((name, start, start + size))
        for index, (left_name, left_start, left_end) in enumerate(ranges):
            for right_name, right_start, right_end in ranges[index + 1 :]:
                if left_start < right_end and right_start < left_end:
                    raise ValueError(
                        "selected-gather CUDA buffers must not overlap: "
                        f"{left_name} and {right_name}"
                    )

    def _stream(self, stream):
        stream = stream or torch.cuda.current_stream(self.device)
        if stream.device.index != self.device:
            raise ValueError("host KV stream is on the wrong CUDA device")
        return stream

    def _require_capture_lease(self, stream) -> None:
        if stream.is_capturing() and self._graph_refs == 0:
            raise RuntimeError(
                "retain_for_graph() is required before host KV graph capture"
            )

    def retain_for_graph(self) -> HostKVGraphLease:
        """Pin this arena; close the lease only after graph destruction/quiescence."""
        with self._lock:
            self._require_open()
            self._graph_refs += 1
            return HostKVGraphLease(self)

    def scatter(
        self,
        layer: int,
        ids: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        *,
        stream=None,
    ) -> None:
        """Scatter rows to physical slots; leading row strides may be noncontiguous."""
        with self._lock:
            self._require_open()
            layer = self._validate_layer(layer)
            id_bytes = self._validate_ids(ids)
            count = ids.numel()
            key_stride = self._validate_rows(keys, count, packed=False, name="keys")
            value_stride = self._validate_rows(
                values, count, packed=False, name="values"
            )
            stream = self._stream(stream)
            self._require_capture_lease(stream)
            if self.debug and stream.is_capturing():
                raise RuntimeError(
                    "debug ID checking cannot synchronize during capture"
                )
            _check(
                self._lib.q38_host_kv_scatter(
                    self._handle,
                    layer,
                    keys.data_ptr(),
                    key_stride,
                    values.data_ptr(),
                    value_stride,
                    ids.data_ptr(),
                    count,
                    id_bytes,
                    stream.cuda_stream,
                ),
                self._lib,
            )
            if self.debug:
                self.check_errors()

    def gather(
        self,
        layer: int,
        ids: torch.Tensor,
        key_output: torch.Tensor,
        value_output: torch.Tensor,
        *,
        stream=None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Gather IDs in order into caller-owned packed K and V outputs."""
        with self._lock:
            self._require_open()
            layer = self._validate_layer(layer)
            id_bytes = self._validate_ids(ids)
            count = ids.numel()
            self._validate_rows(key_output, count, packed=True, name="key_output")
            self._validate_rows(value_output, count, packed=True, name="value_output")
            stream = self._stream(stream)
            self._require_capture_lease(stream)
            if self.debug and stream.is_capturing():
                raise RuntimeError(
                    "debug ID checking cannot synchronize during capture"
                )
            _check(
                self._lib.q38_host_kv_gather(
                    self._handle,
                    layer,
                    ids.data_ptr(),
                    count,
                    id_bytes,
                    key_output.data_ptr(),
                    value_output.data_ptr(),
                    stream.cuda_stream,
                ),
                self._lib,
            )
            if self.debug:
                self.check_errors()
            return key_output, value_output

    def gather_selected(
        self,
        layer: int,
        ids: torch.Tensor,
        key_output: torch.Tensor,
        value_output: torch.Tensor,
        *,
        stream=None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Gather selected IDs, deduplicating repeated physical slots when enabled."""
        if not self.selected_dedup:
            return self.gather(layer, ids, key_output, value_output, stream=stream)
        with self._lock:
            self._require_open()
            layer = self._validate_layer(layer)
            id_bytes = self._validate_ids(ids)
            count = ids.numel()
            self._validate_rows(key_output, count, packed=True, name="key_output")
            self._validate_rows(value_output, count, packed=True, name="value_output")
            workspace = self._validate_selected_workspace()
            self._validate_selected_non_aliasing(
                ids=ids,
                workspace=workspace,
                key_output=key_output,
                value_output=value_output,
            )
            stream = self._stream(stream)
            self._require_capture_lease(stream)
            if self.debug and stream.is_capturing():
                raise RuntimeError(
                    "debug ID checking cannot synchronize during capture"
                )
            _check(
                self._lib.q38_host_kv_gather_dedup(
                    self._handle,
                    layer,
                    ids.data_ptr(),
                    count,
                    id_bytes,
                    workspace.data_ptr(),
                    workspace.numel(),
                    key_output.data_ptr(),
                    value_output.data_ptr(),
                    stream.cuda_stream,
                ),
                self._lib,
            )
            if self.debug:
                self.check_errors()
            return key_output, value_output

    def check_errors(self) -> None:
        """Synchronize this arena's last use and surface deferred device ID errors."""
        with self._lock:
            self._require_open()
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("host KV errors cannot be checked during capture")
            if self._graph_refs:
                # Captures omit the arena's shared eager event; synchronize at
                # this explicit debug boundary before reading the device flag.
                torch.cuda.synchronize(self.device)
            _check(self._lib.q38_host_kv_check(self._handle), self._lib)

    def close(self) -> None:
        with self._lock:
            if not self._handle:
                return
            if self._graph_refs:
                raise RuntimeError(
                    f"cannot close {self.label}: {self._graph_refs} graph lease(s) remain"
                )
            _check(self._lib.q38_host_kv_close(self._handle), self._lib)
            self._handle = None
            self._selected_row_map = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()

    def __del__(self):
        if getattr(self, "_handle", None) and not getattr(self, "_graph_refs", 0):
            try:
                self.close()
            except Exception:
                pass


# Concise integration-facing name; keep the explicit arena name for clarity in
# tests and ownership code.
NativeHostKV = HostKVArena

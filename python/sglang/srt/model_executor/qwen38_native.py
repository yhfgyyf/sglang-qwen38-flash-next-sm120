# Copyright 2026 Qwen38 native contributors. SPDX-License-Identifier: Apache-2.0
"""Qwen3.8/SM120 native graph and PLE bridge.

Loading/capture remains Python; replay of a compiled plan never calls a Python
graph-break closure. Unknown breaks and deduplicated mutable executables are
rejected, not silently interpreted. This module does not replace the scheduler.
"""

from __future__ import annotations

import ctypes as C
import logging
import os
import threading
import weakref
from dataclasses import dataclass
from functools import lru_cache, wraps
from pathlib import Path

import torch

logger = logging.getLogger(__name__)


def enabled() -> bool:
    return os.environ.get("QWEN38_NATIVE_EXECUTOR", "0") == "1"


def _declare(lib, name, result, *args):
    fn = getattr(lib, name)
    fn.restype = result
    fn.argtypes = args
    return fn


@lru_cache(maxsize=1)
def libraries():
    root = Path(__file__).resolve().parents[4] / "native" / "qwen38"
    native_path = Path(
        os.environ.get("QWEN38_NATIVE_LIBRARY", root / "build" / "libqwen38_native.so")
    )
    store_path = Path(
        os.environ.get(
            "QWEN38_PLE_LIBRARY",
            root / "ple_store" / "target" / "release" / "libq38_ple_store.so",
        )
    )
    # Load once, before any capture. ctypes CDLL releases the GIL during calls.
    store = C.CDLL(str(store_path), mode=C.RTLD_GLOBAL)
    native = C.CDLL(str(native_path))
    ptr, size, u64 = C.c_void_p, C.c_size_t, C.c_uint64
    _declare(store, "q38_ple_last_error", C.c_char_p)
    _declare(
        store,
        "q38_ple_open",
        ptr,
        C.POINTER(C.c_char_p),
        C.POINTER(u64),
        C.POINTER(u64),
        size,
        size,
        size,
        C.c_uint32,
        size,
    )
    _declare(store, "q38_ple_read", C.c_int, ptr, ptr, size, ptr, size)
    _declare(store, "q38_ple_stats", C.c_int, ptr, C.POINTER(u64), size)
    _declare(store, "q38_ple_close", None, ptr)
    _declare(native, "q38_native_last_error", C.c_char_p)
    _declare(native, "q38_pipe_create", ptr, ptr, size, size, C.c_int)
    _declare(native, "q38_pipe_issue", C.c_int64, ptr, ptr, size, size)
    _declare(native, "q38_pipe_collect", C.c_int, ptr, C.c_int64, ptr, size, size)
    _declare(native, "q38_pipe_pending", C.c_int, ptr)
    _declare(native, "q38_pipe_close", C.c_int, ptr)
    _declare(native, "q38_plan_create", ptr, C.c_int)
    _declare(native, "q38_plan_add_graph", C.c_int, ptr, size)
    _declare(native, "q38_plan_add_issue", C.c_int, ptr, ptr, ptr, size)
    _declare(native, "q38_plan_add_collect", C.c_int, ptr, ptr, ptr, size)
    _declare(native, "q38_plan_seal", C.c_int, ptr)
    _declare(native, "q38_plan_replay", C.c_int, ptr, size)
    _declare(native, "q38_plan_close", C.c_int, ptr)
    _declare(native, "q38_decode_fp8", C.c_int, ptr, ptr, size, size)
    return native, store


def _check(status, lib, name="q38_native_last_error"):
    if status != 0:
        message = getattr(lib, name)()
        raise RuntimeError(message.decode() if message else "native engine error")


def _locked(fn):
    @wraps(fn)
    def call(self, *args, **kwargs):
        with self._lock:
            return fn(self, *args, **kwargs)

    return call


@dataclass(frozen=True)
class NativePLEOperation:
    kind: str
    pipe: NativePLEPipe
    tensor: torch.Tensor
    count: int


class ReplayDomain:
    """All plans sharing an allocator pool must use one replay stream/thread.

    The SGLang adapter still owns metadata preparation outside this boundary;
    it is not safe to introduce concurrent replay/metadata mutation here.
    """

    def __init__(self):
        self.stream = None
        self.thread = None

    def bind(self, stream):
        identity = (stream.device.index, stream.cuda_stream)
        thread = threading.get_ident()
        if self.stream is None:
            self.stream, self.thread = identity, thread
        elif self.stream != identity or self.thread != thread:
            raise RuntimeError(
                "native plans sharing a graph pool require one replay stream/thread"
            )


_domains = weakref.WeakValueDictionary()


def replay_domain(device, pool):
    key = (device, tuple(pool))
    if key not in _domains:
        _domains[key] = domain = ReplayDomain()
        return domain
    return _domains[key]


class NativePLEPipe:
    """Own store and async pipe; one pending read, multiple captured plans.

    Plans retain this object and the borrowed tensors. Explicit close while a
    plan is alive is an error; close plans before closing the embedding.
    """

    def __init__(
        self,
        manifest,
        *,
        cache_bytes,
        queue_depth=512,
        max_batch=4096,
        max_rows=131072,
        device=None,
    ):
        self._lock = threading.RLock()
        if manifest.dtype != "F8_E4M3" or manifest.row_bytes != 160:
            raise ValueError("native Qwen38 requires original 160-byte FP8 E4M3 rows")
        if cache_bytes < 0:
            raise ValueError("native PLE requires an explicit nonnegative cache budget")
        self.device = torch.cuda.current_device() if device is None else device
        if torch.cuda.get_device_capability(self.device) != (12, 0):
            raise ValueError("this native executor is validated only for SM120")
        self.native, self.store_lib = libraries()
        self.handle = self.store = None
        self._plan_refs = 0
        self._pending = None
        self.max_rows = max_rows
        self.row_bytes = manifest.row_bytes
        n = len(manifest.shards)
        paths = (C.c_char_p * n)(*[os.fsencode(s.tensor.path) for s in manifest.shards])
        offsets = (C.c_uint64 * n)(*[s.tensor.offset for s in manifest.shards])
        counts = (C.c_uint64 * n)(*[s.row_end - s.row_start for s in manifest.shards])
        self.store = self.store_lib.q38_ple_open(
            paths,
            offsets,
            counts,
            n,
            self.row_bytes,
            cache_bytes,
            queue_depth,
            max_batch,
        )
        if not self.store:
            _check(-1, self.store_lib, "q38_ple_last_error")
        self.handle = self.native.q38_pipe_create(
            self.store, self.row_bytes, max_rows, self.device
        )
        if not self.handle:
            error = self.native.q38_native_last_error().decode()
            self.store_lib.q38_ple_close(self.store)
            self.store = None
            raise RuntimeError(error)

    def validate_ids(self, ids):
        if (
            ids.dtype != torch.int64
            or not ids.is_cuda
            or ids.device.index != self.device
            or not ids.is_contiguous()
            or not 0 < ids.numel() <= self.max_rows
        ):
            raise ValueError(
                "native PLE IDs must be contiguous CUDA int64 within capacity"
            )

    def validate_output(self, output, count):
        if (
            output.dtype != torch.bfloat16
            or not output.is_cuda
            or output.device.index != self.device
            or not output.is_contiguous()
            or output.numel() != count * self.row_bytes
        ):
            raise ValueError("native PLE output must be contiguous matching CUDA BF16")

    @_locked
    def issue(self, ids):
        self.validate_ids(ids)
        if self._pending is not None:
            raise RuntimeError("native PLE gather already pending")
        ticket = self.native.q38_pipe_issue(
            self.handle,
            ids.data_ptr(),
            ids.numel(),
            torch.cuda.current_stream(self.device).cuda_stream,
        )
        if ticket < 0:
            _check(-1, self.native)
        # Hold source storage until the worker has consumed its ID copy.
        self._pending = (ticket, ids)
        return ticket

    @_locked
    def collect(self, ticket, output, stream=None):
        if self._pending is None or self._pending[0] != ticket:
            raise RuntimeError("stale native PLE ticket")
        count = self._pending[1].numel()
        self.validate_output(output, count)
        stream = stream or torch.cuda.current_stream(self.device)
        status = self.native.q38_pipe_collect(
            self.handle, ticket, output.data_ptr(), count, stream.cuda_stream
        )
        if status != 0:
            error = self.native.q38_native_last_error().decode()
            if self.native.q38_pipe_pending(self.handle) == 0:
                self._pending = None
            raise RuntimeError(error)
        self._pending = None
        return output

    @_locked
    def snapshot_stats(self):
        values = (C.c_uint64 * 8)()
        _check(
            self.store_lib.q38_ple_stats(self.store, values, 8),
            self.store_lib,
            "q38_ple_last_error",
        )
        return list(values)

    @_locked
    def close(self):
        if self._plan_refs:
            raise RuntimeError("close native graph plans before their PLE pipe")
        if self.handle:
            _check(self.native.q38_pipe_close(self.handle), self.native)
            self.handle = None
            self._pending = None
        if self.store:
            self.store_lib.q38_ple_close(self.store)
            self.store = None

    def __del__(self):
        if getattr(self, "handle", None) and not getattr(self, "_plan_refs", 0):
            try:
                self.close()
            except Exception:
                pass


class NativeGraphPlan:
    def __init__(self, segments, operations, *, pool=(0, 0)):
        self._lock = threading.RLock()
        if len(segments) != len(operations) + 1:
            raise ValueError(
                "native plan requires one graph on each side of every break"
            )
        if any(not isinstance(op, NativePLEOperation) for op in operations):
            raise ValueError(
                "unknown graph break: native plan refuses Python callbacks"
            )
        if any(not hasattr(seg, "raw_cuda_graph_exec") for seg in segments):
            raise ValueError(
                "native plan requires immutable graph executables; disable graph dedup"
            )
        self.native, _ = libraries()
        self.handle = None
        self._reported_replay = False
        self.segments = tuple(segments)
        self.operations = tuple(operations)
        pipes = tuple({op.pipe for op in operations})
        self.pipes = ()
        self._host_kv_leases = ()
        self.device = torch.cuda.current_device()
        self.domain = replay_domain(self.device, pool)
        self.handle = self.native.q38_plan_create(self.device)
        if not self.handle:
            _check(-1, self.native)
        try:
            if os.environ.get("QWEN38_HOST_KV_BYTES", "0") != "0":
                from sglang.srt.mem_cache.qwen38_host_kv_pool import (
                    retain_active_host_arenas,
                )

                self._host_kv_leases = retain_active_host_arenas()
            for pipe in pipes:
                with pipe._lock:
                    if not pipe.handle:
                        raise RuntimeError("native PLE pipe is closed")
                    pipe._plan_refs += 1
                    self.pipes += (pipe,)
            for index, segment in enumerate(segments):
                _check(
                    self.native.q38_plan_add_graph(
                        self.handle, segment.raw_cuda_graph_exec()
                    ),
                    self.native,
                )
                if index == len(operations):
                    continue
                op = operations[index]
                if op.kind == "issue":
                    op.pipe.validate_ids(op.tensor)
                    if op.count != op.tensor.numel():
                        raise ValueError("native issue count must equal ID tensor size")
                    fn = self.native.q38_plan_add_issue
                elif op.kind == "collect":
                    op.pipe.validate_output(op.tensor, op.count)
                    fn = self.native.q38_plan_add_collect
                else:
                    raise ValueError(f"unknown native operation {op.kind!r}")
                _check(
                    fn(self.handle, op.pipe.handle, op.tensor.data_ptr(), op.count),
                    self.native,
                )
            _check(self.native.q38_plan_seal(self.handle), self.native)
        except Exception:
            self.close()
            raise

    @_locked
    def replay(self, stream):
        self.domain.bind(stream)
        _check(
            self.native.q38_plan_replay(self.handle, stream.cuda_stream), self.native
        )
        if not self._reported_replay:
            logger.info(
                "Qwen38 native C++ replay executed: graphs=%d typed_ple_ops=%d Python_break_callbacks=0",
                len(self.segments),
                len(self.operations),
            )
            self._reported_replay = True

    @_locked
    def close(self):
        if self.handle:
            _check(self.native.q38_plan_close(self.handle), self.native)
            self.handle = None
            for pipe in self.pipes:
                with pipe._lock:
                    pipe._plan_refs -= 1
            self.operations = ()
            self.segments = ()
            self.pipes = ()
            for lease in self._host_kv_leases:
                lease.close()
            self._host_kv_leases = ()

    def __del__(self):
        if getattr(self, "handle", None):
            try:
                self.close()
            except Exception:
                pass

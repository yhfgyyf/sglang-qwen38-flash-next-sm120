"""Integration contracts for the opt-in host-KV selected-gather path.

The CPU tests use CUDA/C-ABI fakes and are safe while GPU0 is occupied.  The
GPU tests are intentionally skipped without an explicitly visible CUDA device.
"""

from __future__ import annotations

import ast
import ctypes as C
import gc
import importlib.util
from pathlib import Path
import sys
from types import ModuleType
from types import SimpleNamespace
import weakref

import pytest
import torch


ROOT = Path(__file__).resolve().parents[3]


def _load_module(name, relative_path):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


host_kv = _load_module(
    "q38_host_kv_dedup_integration_host",
    "python/sglang/srt/model_executor/qwen38_host_kv.py",
)


class _KVCache:
    def __init__(self, size, page_size, dtype, layer_num, device, _memory_saver):
        self.size = size
        self.page_size = page_size
        self.dtype = dtype
        self.layer_num = layer_num
        self.device = device


class _UnquantizedKVCacheMethod:
    pass


_memory_pool_name = "sglang.srt.mem_cache.memory_pool"
_host_module_name = "sglang.srt.model_executor.qwen38_host_kv"
_saved_modules = {
    name: sys.modules.get(name) for name in (_memory_pool_name, _host_module_name)
}
_fake_memory_pool = ModuleType(_memory_pool_name)
_fake_memory_pool.KVCache = _KVCache
_fake_memory_pool.UnquantizedKVCacheMethod = _UnquantizedKVCacheMethod
sys.modules[_memory_pool_name] = _fake_memory_pool
sys.modules[_host_module_name] = host_kv
try:
    host_pool_module = _load_module(
        "q38_host_kv_dedup_integration_pool",
        "python/sglang/srt/mem_cache/qwen38_host_kv_pool.py",
    )
finally:
    for _name, _module in _saved_modules.items():
        if _module is None:
            sys.modules.pop(_name, None)
        else:
            sys.modules[_name] = _module


class _Function:
    def __init__(self, result=0):
        self.result = result
        self.calls = []

    def __call__(self, *args):
        self.calls.append(args)
        return self.result(*args) if callable(self.result) else self.result


class _Library:
    def __init__(self, allocated_bytes, *, dedup=True):
        self.q38_host_kv_last_error = _Function(b"fake host-KV error")
        self.q38_host_kv_required_bytes = _Function()
        self.q38_host_kv_create = _Function(0xA000)
        self.q38_host_kv_allocated_bytes = _Function(allocated_bytes)
        self.q38_host_kv_scatter = _Function()
        self.q38_host_kv_gather = _Function()
        self.q38_host_kv_check = _Function()
        self.q38_host_kv_close = _Function()
        if dedup:
            self.q38_host_kv_gather_dedup = _Function()


class _Tensor:
    def __init__(
        self,
        shape,
        dtype,
        address,
        *,
        device=0,
        strides=None,
        contiguous=True,
        is_cuda=True,
    ):
        self.shape = tuple(shape)
        self.ndim = len(self.shape)
        self.dtype = dtype
        self.device = SimpleNamespace(index=device)
        self.is_cuda = is_cuda
        self._address = address
        self._contiguous = contiguous
        if strides is None:
            values = []
            stride = 1
            for size in reversed(self.shape):
                values.append(stride)
                stride *= size
            strides = tuple(reversed(values))
        self._strides = tuple(strides)

    def data_ptr(self):
        return self._address

    def element_size(self):
        return {
            torch.int32: 4,
            torch.int64: 8,
            torch.bfloat16: 2,
            torch.float8_e4m3fn: 1,
        }[self.dtype]

    def is_contiguous(self):
        return self._contiguous

    def numel(self):
        result = 1
        for size in self.shape:
            result *= size
        return result

    def stride(self, dimension=None):
        return self._strides if dimension is None else self._strides[dimension]


class _Stream:
    def __init__(self, *, capturing=False):
        self.device = SimpleNamespace(index=0)
        self.cuda_stream = 0xB000
        self._capturing = capturing

    def is_capturing(self):
        return self._capturing


def _base_cdll(*, dedup):
    library = _Library(0, dedup=dedup)
    library.q38_host_kv_abi_version = _Function(2)
    return library


def test_optional_symbol_is_ignored_when_off_and_required_when_on(monkeypatch):
    libraries = []

    def load(_path):
        library = _base_cdll(dedup=False)
        libraries.append(library)
        return library

    monkeypatch.setattr(host_kv.C, "CDLL", load)
    host_kv.library.cache_clear()
    try:
        assert host_kv.library(selected_dedup=False) is libraries[0]
        with pytest.raises(RuntimeError, match="optional dedup gather"):
            host_kv.library(selected_dedup=True)
    finally:
        host_kv.library.cache_clear()


def test_optional_symbol_declaration_matches_additive_abi(monkeypatch):
    library = _base_cdll(dedup=True)
    monkeypatch.setattr(host_kv.C, "CDLL", lambda _path: library)
    host_kv.library.cache_clear()
    try:
        assert host_kv.library(selected_dedup=True) is library
        function = library.q38_host_kv_gather_dedup
        assert function.restype is C.c_int
        assert len(function.argtypes) == 10
        assert function.argtypes[5] is C.c_void_p
        assert function.argtypes[6] is C.c_uint64
    finally:
        host_kv.library.cache_clear()


def _fake_arena(monkeypatch, *, selected_dedup=True, slots=7):
    allocation_addresses = iter(range(0x2000, 0x3000, 0x100))
    allocations = []
    nbytes = host_kv.required_bytes(1, slots)
    library = _Library(nbytes)

    def allocate(shape, *, dtype, device):
        tensor = _Tensor((shape,), dtype, next(allocation_addresses), device=0)
        allocations.append((shape, dtype, device, tensor))
        return tensor

    monkeypatch.setattr(host_kv, "library", lambda selected_dedup=False: library)
    monkeypatch.setattr(host_kv.torch, "empty", allocate)
    monkeypatch.setattr(
        host_kv.torch.cuda, "get_device_capability", lambda _device: (12, 0)
    )
    monkeypatch.setattr(host_kv.torch.cuda, "current_stream", lambda _device: _Stream())
    arena = host_kv.HostKVArena(
        1,
        slots,
        byte_budget=nbytes,
        device="cuda:0",
        selected_dedup=selected_dedup,
    )
    return arena, library, allocations


def _gather_tensors(
    count, *, ids_address=0x4000, key_address=0x8000, value_address=0xC000
):
    ids = _Tensor((count,), torch.int32, ids_address)
    keys = _Tensor((count, 2, 256), torch.bfloat16, key_address)
    values = _Tensor((count, 2, 256), torch.bfloat16, value_address)
    return ids, keys, values


def test_selected_gather_owns_one_stable_slot_workspace(monkeypatch):
    arena, library, allocations = _fake_arena(monkeypatch, slots=7)
    try:
        assert [(shape, dtype) for shape, dtype, _device, _tensor in allocations] == [
            (7, torch.int32)
        ]
        workspace = allocations[0][3]
        assert arena.gpu_bytes == 24 + 7 * 4

        first = _gather_tensors(3)
        oversized = _gather_tensors(
            11, ids_address=0x10000, key_address=0x14000, value_address=0x18000
        )
        assert arena.gather_selected(0, *first) == first[1:]
        assert arena.gather_selected(0, *oversized) == oversized[1:]

        assert len(allocations) == 1
        calls = library.q38_host_kv_gather_dedup.calls
        assert len(calls) == 2
        assert [call[3] for call in calls] == [3, 11]
        assert all(call[5] == workspace.data_ptr() for call in calls)
        assert all(call[6] == 7 for call in calls)
    finally:
        arena.close()


def test_selected_gather_validates_workspace_and_all_pairwise_aliases(monkeypatch):
    arena, _library, _allocations = _fake_arena(monkeypatch, slots=7)
    ids, keys, values = _gather_tensors(3)
    try:
        bad_workspaces = (
            _Tensor((7,), torch.int64, 0x2000),
            _Tensor((7,), torch.int32, 0x2000, device=1),
            _Tensor((7,), torch.int32, 0x2000, strides=(2,), contiguous=False),
            _Tensor((6,), torch.int32, 0x2000),
        )
        for workspace in bad_workspaces:
            arena._selected_row_map = workspace
            with pytest.raises(ValueError, match="selected-gather workspace"):
                arena.gather_selected(
                    0, ids, keys, values, stream=_Stream(capturing=True)
                )

        arena._selected_row_map = _Tensor((7,), torch.int32, 0x2000)
        aliases = (
            (_Tensor((3,), torch.int32, 0x2000), keys, values),
            (ids, _Tensor((3, 2, 256), torch.bfloat16, 0x4000), values),
            (ids, keys, _Tensor((3, 2, 256), torch.bfloat16, 0x4000)),
            (ids, _Tensor((3, 2, 256), torch.bfloat16, 0x2000), values),
            (ids, keys, _Tensor((3, 2, 256), torch.bfloat16, 0x2000)),
            (ids, keys, _Tensor((3, 2, 256), torch.bfloat16, 0x8000)),
        )
        for aliased_ids, aliased_keys, aliased_values in aliases:
            with pytest.raises(ValueError, match="must not overlap"):
                arena.gather_selected(
                    0,
                    aliased_ids,
                    aliased_keys,
                    aliased_values,
                    stream=_Stream(capturing=True),
                )
    finally:
        arena._selected_row_map = _Tensor((7,), torch.int32, 0x2000)
        arena.close()


def test_selected_gather_validates_cuda_input_and_output_layout_during_capture(
    monkeypatch,
):
    arena, _library, _allocations = _fake_arena(monkeypatch, slots=7)
    ids, keys, values = _gather_tensors(3)
    cases = (
        (_Tensor((3,), torch.int32, 0x4000, is_cuda=False), keys, values),
        (
            _Tensor((3,), torch.int32, 0x4000, strides=(2,), contiguous=False),
            keys,
            values,
        ),
        (
            ids,
            _Tensor((3, 2, 256), torch.bfloat16, 0x8000, device=1),
            values,
        ),
        (
            ids,
            _Tensor(
                (3, 2, 256),
                torch.bfloat16,
                0x8000,
                strides=(513, 256, 1),
                contiguous=False,
            ),
            values,
        ),
        (
            ids,
            keys,
            _Tensor((3, 2, 256), torch.bfloat16, 0xC000, is_cuda=False),
        ),
    )
    try:
        for invalid_ids, invalid_keys, invalid_values in cases:
            with pytest.raises(ValueError):
                arena.gather_selected(
                    0,
                    invalid_ids,
                    invalid_keys,
                    invalid_values,
                    stream=_Stream(capturing=True),
                )
    finally:
        arena.close()


def test_selected_gather_off_delegates_to_ordinary_gather_without_workspace(
    monkeypatch,
):
    arena, library, allocations = _fake_arena(monkeypatch, selected_dedup=False)
    ordinary_calls = []
    tensors = _gather_tensors(2)
    arena.gather = (
        lambda *args, **kwargs: ordinary_calls.append((args, kwargs)) or tensors[1:]
    )
    try:
        assert allocations == []
        assert arena.gpu_bytes == 24
        assert arena.gather_selected(0, *tensors, stream="stream") == tensors[1:]
        assert ordinary_calls == [((0, *tensors), {"stream": "stream"})]
        assert library.q38_host_kv_gather_dedup.calls == []
    finally:
        arena.close()


def test_workspace_is_released_only_after_native_close(monkeypatch):
    arena, library, allocations = _fake_arena(monkeypatch)
    workspace = allocations[0][3]

    def close_result(_handle):
        assert arena._selected_row_map is workspace
        return 0

    library.q38_host_kv_close.result = close_result
    arena.close()
    assert arena._selected_row_map is None


def test_pool_passes_opt_in_and_routes_only_selected_gather(monkeypatch):
    created = []

    class Arena:
        closed = False
        graph_references = 1

        def __init__(self, *args, **kwargs):
            created.append((args, kwargs))
            self.selected_dedup = kwargs["selected_dedup"]
            self.gpu_bytes = 24 + (args[1] * 4 if self.selected_dedup else 0)

        def retain_for_graph(self):
            return SimpleNamespace(close=lambda: None)

        def gather(self, *args):
            return ("ordinary", args)

        def gather_selected(self, *args):
            return ("selected", args)

    def base_init(self, size, page_size, dtype, layer_num, device, _memory_saver):
        self.size = size
        self.page_size = page_size
        self.dtype = dtype
        self.layer_num = layer_num
        self.device = device

    monkeypatch.setattr(host_pool_module.KVCache, "__init__", base_init)
    monkeypatch.setattr(host_pool_module, "HostKVArena", Arena)
    monkeypatch.setattr(host_pool_module, "_check_host_headroom", lambda _size: None)
    budget = host_kv.required_bytes(13, 65, dtype=torch.float8_e4m3fn)
    with monkeypatch.context() as context:
        context.setenv("QWEN38_HOST_KV_BYTES", str(budget))
        context.setenv("QWEN38_HOST_KV_DEDUP", "1")
        pool = host_pool_module.Qwen38HostKVPool(
            size=1,
            page_size=64,
            dtype=torch.float8_e4m3fn,
            head_num=2,
            head_dim=256,
            layer_num=12,
            device="cuda:0",
        )
    assert created[0][1]["selected_dedup"] is True
    assert pool.gather(3, "ids", "k", "v")[0] == "ordinary"
    assert pool.gather_selected(3, "ids", "k", "v")[0] == "selected"


@pytest.mark.parametrize("value, expected", [(None, False), ("0", False), ("1", True)])
def test_pool_dedup_environment_is_explicit(monkeypatch, value, expected):
    if value is None:
        monkeypatch.delenv("QWEN38_HOST_KV_DEDUP", raising=False)
    else:
        monkeypatch.setenv("QWEN38_HOST_KV_DEDUP", value)
    assert host_pool_module.host_kv_dedup_enabled() is expected
    monkeypatch.setenv("QWEN38_HOST_KV_DEDUP", "true")
    with pytest.raises(ValueError, match="must be 0 or 1"):
        host_pool_module.host_kv_dedup_enabled()


def _method_calls(path, method_name):
    tree = ast.parse(path.read_text())
    method = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == method_name
    )
    return [
        (ast.unparse(call.func.value), call.func.attr)
        for call in ast.walk(method)
        if isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
    ]


def test_attention_selected_decode_routes_dedup_but_prefill_and_move_stay_ordinary():
    attention = ROOT / "python/sglang/srt/layers/attention/qwen_sparse_attn_backend.py"
    pool = ROOT / "python/sglang/srt/mem_cache/qwen38_host_kv_pool.py"
    decode_calls = _method_calls(attention, "_forward_trtllm_sparse")
    prefill_calls = _method_calls(attention, "forward_extend")
    move_calls = _method_calls(pool, "move_kv_cache")

    assert ("host_pool", "gather_selected") in decode_calls
    assert ("host_pool", "gather") not in decode_calls
    assert ("host_pool", "gather") in prefill_calls
    assert ("host_pool", "gather_selected") not in prefill_calls
    assert ("self.arena", "gather") in move_calls
    assert ("self.arena", "gather_selected") not in move_calls


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU tests are opt-in")
class TestHostKVDedupIntegrationGPU:
    @staticmethod
    def _arena(slots=32):
        if torch.cuda.get_device_capability() != (12, 0):
            pytest.skip("host KV dedup integration requires SM120")
        size = host_kv.required_bytes(1, slots, dtype=torch.float8_e4m3fn)
        return host_kv.HostKVArena(
            1,
            slots,
            byte_budget=size,
            dtype=torch.float8_e4m3fn,
            selected_dedup=True,
            label="dedup-integration",
        )

    @staticmethod
    def _source_rows(slots=32):
        # Every physical slot and K/V plane has a distinct raw-byte pattern;
        # repeating arange(512) per row cannot detect wrong-slot gathers.
        rows = torch.arange(slots, device="cuda", dtype=torch.int64)[:, None]
        columns = torch.arange(512, device="cuda", dtype=torch.int64)[None, :]
        keys = (rows * 17 + columns * 3).to(torch.uint8).reshape(slots, 2, 256)
        values = (rows * 29 + columns * 7 + 113).to(torch.uint8).reshape(slots, 2, 256)
        return keys.view(torch.float8_e4m3fn), values.view(torch.float8_e4m3fn)

    def test_selected_gather_matches_ordinary_fp8_bytes(self):
        arena = self._arena()
        try:
            keys, values = self._source_rows()
            slots = torch.arange(32, dtype=torch.int32, device="cuda")
            arena.scatter(0, slots, keys, values)
            ids = torch.tensor(
                [7, 7, -1, 3, 31, 3, 0], dtype=torch.int64, device="cuda"
            )
            ordinary_k = torch.empty(
                (ids.numel(), 2, 256), dtype=keys.dtype, device="cuda"
            )
            ordinary_v = torch.empty_like(ordinary_k)
            selected_k = torch.empty_like(ordinary_k)
            selected_v = torch.empty_like(ordinary_k)
            arena.gather(0, ids, ordinary_k, ordinary_v)
            arena.gather_selected(0, ids, selected_k, selected_v)
            arena.check_errors()
            assert torch.equal(
                ordinary_k.view(torch.uint8), selected_k.view(torch.uint8)
            )
            assert torch.equal(
                ordinary_v.view(torch.uint8), selected_v.view(torch.uint8)
            )
        finally:
            arena.close()

    def test_two_graph_shapes_accept_dynamic_ids_and_alternating_replay(self):
        arena = self._arena()
        leases = [arena.retain_for_graph(), arena.retain_for_graph()]
        graphs = []
        cases = []
        graph = None
        try:
            keys, values = self._source_rows()
            slots = torch.arange(32, dtype=torch.int32, device="cuda")
            arena.scatter(0, slots, keys, values)
            arena.check_errors()
            for count in (5, 9):
                ids = torch.zeros(count, dtype=torch.int32, device="cuda")
                out_k = torch.empty((count, 2, 256), dtype=keys.dtype, device="cuda")
                out_v = torch.empty_like(out_k)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    arena.gather_selected(0, ids, out_k, out_v)
                graphs.append(graph)
                cases.append((ids, out_k, out_v))
            for graph_index, requested in (
                (0, [3, 3, -1, 8, 1]),
                (1, [9, 2, 9, 4, -1, 4, 7, 2, 0]),
                (0, [6, 5, 6, 5, -1]),
            ):
                ids, out_k, out_v = cases[graph_index]
                ids.copy_(torch.tensor(requested, dtype=torch.int32, device="cuda"))
                graphs[graph_index].replay()
                torch.cuda.synchronize()
                valid = ids.clamp_min(0).long()
                expected_k = keys.view(torch.uint8)[valid].clone()
                expected_v = values.view(torch.uint8)[valid].clone()
                expected_k[ids < 0] = 0
                expected_v[ids < 0] = 0
                assert torch.equal(out_k.view(torch.uint8), expected_k)
                assert torch.equal(out_v.view(torch.uint8), expected_v)
        finally:
            graphs.clear()
            graph = None
            torch.cuda.synchronize()
            for lease in leases:
                lease.close()
            arena.close()

    def test_workspace_lives_through_graph_lease_and_dies_after_safe_close(self):
        arena = self._arena()
        workspace = weakref.ref(arena._selected_row_map)
        lease = arena.retain_for_graph()
        with pytest.raises(RuntimeError, match="graph lease"):
            arena.close()
        assert workspace() is not None
        lease.close()
        arena.close()
        gc.collect()
        assert workspace() is None

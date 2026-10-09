"""CPU contracts and SM120 GPU regressions for optional dedup gather."""

from __future__ import annotations

import ctypes as C
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import torch


ROOT = Path(__file__).resolve().parents[3]
BENCH_PATH = ROOT / "native/qwen38/bench_host_kv_dedup.py"
SPEC = importlib.util.spec_from_file_location("q38_bench_host_kv_dedup", BENCH_PATH)
BENCH = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = BENCH
SPEC.loader.exec_module(BENCH)


class FakeFunction:
    def __call__(self, *_args):
        return 0


class HostKVDedupCPUContractTests(unittest.TestCase):
    def test_neighbor_workload_is_deterministic_disjoint_and_overlapping(self):
        ids = BENCH.build_gather_ids(6, 1234)
        self.assertTrue(torch.equal(ids, BENCH.build_gather_ids(6, 1234)))
        shaped = ids.reshape(6, BENCH.VERIFY_QUERIES, BENCH.ROW_STRIDE)
        self.assertEqual(ids.numel(), 24 * 2112)
        for request in range(6):
            lower = 1 + request * BENCH.CONTEXT_SLOTS
            upper = lower + BENCH.CONTEXT_SLOTS
            sets = []
            for query in range(BENCH.VERIFY_QUERIES):
                row = shaped[request, query]
                positive = row[: BENCH.VALID_ROWS]
                self.assertTrue(torch.all(row[BENCH.VALID_ROWS :] == -1))
                self.assertGreaterEqual(int(positive.min()), lower)
                self.assertLess(int(positive.max()), upper)
                blocks = positive[: BENCH.BLOCKS_PER_QUERY * BENCH.BLOCK_SIZE]
                blocks = blocks.reshape(BENCH.BLOCKS_PER_QUERY, BENCH.BLOCK_SIZE)
                self.assertTrue(torch.all(blocks[:, 1:] - blocks[:, :-1] == 1))
                self.assertTrue(torch.all((blocks[:, 0] - lower) % 4 == 0))
                sets.append(set(positive.tolist()))
            for left in range(BENCH.VERIFY_QUERIES):
                for right in range(left + 1, BENCH.VERIFY_QUERIES):
                    self.assertEqual(len(sets[left] & sets[right]), 448 * 4)

    def test_optional_symbol_declaration_is_fail_closed(self):
        with self.assertRaisesRegex(RuntimeError, "does not provide optional"):
            BENCH.declare_dedup(SimpleNamespace())
        function = FakeFunction()
        self.assertIs(
            BENCH.declare_dedup(SimpleNamespace(q38_host_kv_gather_dedup=function)),
            function,
        )
        self.assertIs(function.restype, C.c_int)
        self.assertEqual(len(function.argtypes), 10)
        self.assertIs(function.argtypes[6], C.c_uint64)

    def test_fresh_json_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            BENCH.write_json_fresh(path, {"ok": True})
            self.assertEqual(json.loads(path.read_text()), {"ok": True})
            with self.assertRaises(FileExistsError):
                BENCH.write_json_fresh(path, {"ok": False})

    def test_parser_requires_explicit_library_and_11_samples(self):
        with self.assertRaises(SystemExit):
            BENCH.parse_args([])
        with self.assertRaises(SystemExit):
            BENCH.parse_args(["--library", "/tmp/candidate.so", "--samples", "10"])
        args = BENCH.parse_args(["--library", "/tmp/candidate.so"])
        self.assertEqual(args.samples, 11)
        self.assertEqual(args.cache_condition, "both")
        self.assertIsNone(args.output)

    def test_header_keeps_abi2_and_legacy_gather_source_mapping(self):
        header = (ROOT / "native/qwen38/host_kv.h").read_text()
        source = (ROOT / "native/qwen38/host_kv.cu").read_text()
        self.assertIn("#define Q38_HOST_KV_ABI_VERSION 2U", header)
        self.assertIn("q38_host_kv_gather_dedup", header)
        self.assertNotIn("gather_rows_vectorized", source)
        self.assertIn(
            "gather_rows<Id, uint4><<<static_cast<unsigned>(count), 128",
            source,
        )


@unittest.skipUnless(torch.cuda.is_available(), "CUDA GPU tests are opt-in")
class HostKVDedupGPUTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if torch.cuda.get_device_capability() != (12, 0):
            raise RuntimeError("host KV dedup GPU tests require SM120")
        library = Path(
            os.environ.get(
                "QWEN38_HOST_KV_LIBRARY",
                ROOT / "native/qwen38/build/libq38_host_kv.so",
            )
        )
        cls.host_kv = BENCH.load_host_kv_module(ROOT, library)
        cls.dedup = BENCH.declare_dedup(cls.host_kv.library())

    def arena(
        self,
        slots,
        *,
        layers=1,
        heads=2,
        head_dim=256,
        dtype=torch.float8_e4m3fn,
    ):
        size = self.host_kv.required_bytes(layers, slots, heads, head_dim, dtype=dtype)
        return self.host_kv.HostKVArena(
            layers,
            slots,
            heads=heads,
            head_dim=head_dim,
            dtype=dtype,
            byte_budget=size,
            label="dedup-test",
        )

    @staticmethod
    def source_bytes(slots, row_bytes, offset=0):
        columns = torch.arange(256, dtype=torch.uint8, device="cuda")
        columns = columns.repeat((row_bytes + 255) // 256)[:row_bytes]
        rows = torch.arange(slots, dtype=torch.int64, device="cuda")
        row_bytes_id = torch.stack(
            [
                torch.div(rows, 256**byte, rounding_mode="floor")
                .remainder(256)
                .to(torch.uint8)
                for byte in range(4)
            ],
            dim=1,
        ).repeat(1, (row_bytes + 3) // 4)[:, :row_bytes]
        return row_bytes_id.mul(37).add(columns).add(offset)

    def initialize(self, arena, dtype, heads, head_dim, *, layer=0, offset=0):
        row_bytes = heads * head_dim * dtype.itemsize
        key_bytes = self.source_bytes(arena.slot_count, row_bytes, 19 + offset)
        value_bytes = self.source_bytes(arena.slot_count, row_bytes, 113 + offset)
        ids = torch.arange(arena.slot_count, dtype=torch.int32, device="cuda")
        keys = key_bytes.view(dtype).reshape(-1, heads, head_dim)
        values = value_bytes.view(dtype).reshape(-1, heads, head_dim)
        arena.scatter(layer, ids, keys, values)
        arena.check_errors()
        return key_bytes, value_bytes

    def call(self, arena, ids, row_map, output_k, output_v, stream=None, *, layer=0):
        stream = stream or torch.cuda.current_stream()
        return self.dedup(
            arena._handle,
            layer,
            ids.data_ptr() if ids is not None else 0,
            ids.numel() if ids is not None else 0,
            ids.element_size() if ids is not None else 4,
            row_map.data_ptr() if row_map is not None else 0,
            row_map.numel() if row_map is not None else 0,
            output_k.data_ptr() if output_k is not None else 0,
            output_v.data_ptr() if output_v is not None else 0,
            stream.cuda_stream,
        )

    def assert_exact(
        self,
        arena,
        requested,
        id_dtype,
        dtype,
        heads,
        head_dim,
        source_k,
        source_v,
        *,
        expect_oob=False,
    ):
        ids = torch.tensor(requested, dtype=id_dtype, device="cuda")
        count = len(requested)
        row_bytes = heads * head_dim * dtype.itemsize
        row_map = torch.full(
            (arena.slot_count,), 0x12345678, dtype=torch.int32, device="cuda"
        )
        output_k = torch.empty((count, heads, head_dim), dtype=dtype, device="cuda")
        output_v = torch.empty_like(output_k)
        output_k.view(torch.uint8).fill_(0xA5)
        output_v.view(torch.uint8).fill_(0x5A)
        self.assertEqual(self.call(arena, ids, row_map, output_k, output_v), 0)
        if expect_oob:
            with self.assertRaisesRegex(RuntimeError, "gather.*out of bounds"):
                arena.check_errors()
        else:
            arena.check_errors()
        expected_k = torch.zeros((count, row_bytes), dtype=torch.uint8, device="cuda")
        expected_v = torch.zeros_like(expected_k)
        requested_gpu = ids.to(torch.int64)
        valid = requested_gpu.ge(0) & requested_gpu.lt(arena.slot_count)
        expected_k[valid] = source_k[requested_gpu[valid]]
        expected_v[valid] = source_v[requested_gpu[valid]]
        self.assertTrue(
            torch.equal(
                output_k.view(torch.uint8).reshape(count, row_bytes), expected_k
            )
        )
        self.assertTrue(
            torch.equal(
                output_v.view(torch.uint8).reshape(count, row_bytes), expected_v
            )
        )
        map_cpu = row_map.cpu()
        for slot in set(value for value in requested if 0 <= value < arena.slot_count):
            self.assertEqual(map_cpu[slot].item(), requested.index(slot))
        return row_map

    def test_counts_unique_identical_alternating_and_cross_block_duplicates(self):
        slots = 600
        arena = self.arena(slots)
        try:
            source_k, source_v = self.initialize(arena, torch.float8_e4m3fn, 2, 256)
            counts = (0, 1, 3, 4, 5, 31, 33, 511, 512, 513)
            for count in counts:
                patterns = (
                    list(range(count)),
                    [0] * count,
                    [3 if row % 2 else 5 for row in range(count)],
                    [17 if row in (1, 300) else row % slots for row in range(count)],
                )
                for name, pattern in zip(
                    ("unique", "identical", "alternating", "cross-block"), patterns
                ):
                    with self.subTest(count=count, pattern=name):
                        self.assert_exact(
                            arena,
                            pattern,
                            torch.int32,
                            torch.float8_e4m3fn,
                            2,
                            256,
                            source_k,
                            source_v,
                        )
        finally:
            arena.close()

    def test_id_width_raw_dtypes_padding_oob_and_odd_byte_fallback(self):
        cases = (
            (torch.float8_e4m3fn, 1, 256),
            (torch.float8_e4m3fn, 1, 255),
            (torch.bfloat16, 2, 256),
        )
        for dtype, heads, head_dim in cases:
            arena = self.arena(19, heads=heads, head_dim=head_dim, dtype=dtype)
            try:
                source_k, source_v = self.initialize(arena, dtype, heads, head_dim)
                for id_dtype in (torch.int32, torch.int64):
                    minimum = -(1 << (31 if id_dtype == torch.int32 else 63))
                    maximum = (1 << (31 if id_dtype == torch.int32 else 63)) - 1
                    requested = [
                        0,
                        7,
                        7,
                        -1,
                        minimum,
                        18,
                        0,
                        arena.slot_count,
                        maximum,
                        3,
                    ]
                    with self.subTest(
                        dtype=dtype, head_dim=head_dim, id_dtype=id_dtype
                    ):
                        row_map = self.assert_exact(
                            arena,
                            requested,
                            id_dtype,
                            dtype,
                            heads,
                            head_dim,
                            source_k,
                            source_v,
                            expect_oob=True,
                        )
                        self.assertEqual(row_map[0].item(), 0)
                        self.assertEqual(row_map[1].item(), -1)
            finally:
                arena.close()

    def test_real_24_by_2112_shape_matches_ordinary_gather(self):
        cpu_ids = BENCH.build_gather_ids(6, 20261008)
        positive = torch.unique(cpu_ids[cpu_ids >= 0], sorted=True)
        remapped = cpu_ids.clone()
        valid = remapped >= 0
        remapped[valid] = torch.searchsorted(positive, remapped[valid])
        slots = positive.numel()
        arena = self.arena(slots)
        try:
            self.initialize(arena, torch.float8_e4m3fn, 2, 256)
            ids = remapped.to(dtype=torch.int32, device="cuda")
            row_map = torch.empty(slots, dtype=torch.int32, device="cuda")
            ordinary_k = torch.empty(
                (ids.numel(), 2, 256), dtype=torch.float8_e4m3fn, device="cuda"
            )
            ordinary_v = torch.empty_like(ordinary_k)
            dedup_k = torch.empty_like(ordinary_k)
            dedup_v = torch.empty_like(ordinary_k)
            arena.gather(0, ids, ordinary_k, ordinary_v)
            self.assertEqual(self.call(arena, ids, row_map, dedup_k, dedup_v), 0)
            arena.check_errors()
            self.assertTrue(
                torch.equal(ordinary_k.view(torch.uint8), dedup_k.view(torch.uint8))
            )
            self.assertTrue(
                torch.equal(ordinary_v.view(torch.uint8), dedup_v.view(torch.uint8))
            )
        finally:
            arena.close()

    def test_workspace_output_canaries_and_invalid_arguments_fail_before_enqueue(self):
        slots = 23
        count = 7
        arena = self.arena(slots)
        ids = torch.tensor([0, 5, 5, -1, 22, 0, 9], dtype=torch.int32, device="cuda")
        try:
            self.initialize(arena, torch.float8_e4m3fn, 2, 256)
            map_storage = torch.full(
                (slots + 2,), 0x13572468, dtype=torch.int32, device="cuda"
            )
            row_map = map_storage[1:-1]
            output_storages = [
                torch.full(
                    (32 + count * 512 + 32,),
                    0xCC,
                    dtype=torch.uint8,
                    device="cuda",
                )
                for _ in range(2)
            ]
            outputs = [
                storage[32:-32].view(torch.float8_e4m3fn).reshape(count, 2, 256)
                for storage in output_storages
            ]
            self.assertEqual(self.call(arena, ids, row_map, *outputs), 0)
            arena.check_errors()
            self.assertEqual(map_storage[0].item(), 0x13572468)
            self.assertEqual(map_storage[-1].item(), 0x13572468)
            for storage in output_storages:
                self.assertTrue(torch.all(storage[:32] == 0xCC))
                self.assertTrue(torch.all(storage[-32:] == 0xCC))

            sentinel_k = torch.full_like(outputs[0].view(torch.uint8), 0xA5)
            sentinel_v = torch.full_like(outputs[1].view(torch.uint8), 0x5A)
            outputs[0].view(torch.uint8).copy_(sentinel_k)
            outputs[1].view(torch.uint8).copy_(sentinel_v)
            status = self.dedup(
                arena._handle,
                0,
                ids.data_ptr(),
                ids.numel(),
                ids.element_size(),
                row_map.data_ptr(),
                slots - 1,
                outputs[0].data_ptr(),
                outputs[1].data_ptr(),
                torch.cuda.current_stream().cuda_stream,
            )
            self.assertNotEqual(status, 0)
            self.assertIn(
                b"smaller than slot_count", arena._lib.q38_host_kv_last_error()
            )
            self.assertTrue(torch.equal(outputs[0].view(torch.uint8), sentinel_k))
            self.assertTrue(torch.equal(outputs[1].view(torch.uint8), sentinel_v))

            misaligned = torch.empty(slots * 4 + 1, dtype=torch.uint8, device="cuda")
            status = self.dedup(
                arena._handle,
                0,
                ids.data_ptr(),
                ids.numel(),
                ids.element_size(),
                misaligned.data_ptr() + 1,
                slots,
                outputs[0].data_ptr(),
                outputs[1].data_ptr(),
                torch.cuda.current_stream().cuda_stream,
            )
            self.assertNotEqual(status, 0)
            self.assertIn(b"alignment", arena._lib.q38_host_kv_last_error())

            status = self.dedup(
                arena._handle,
                0,
                ids.data_ptr(),
                ids.numel(),
                ids.element_size(),
                ids.data_ptr(),
                slots,
                outputs[0].data_ptr(),
                outputs[1].data_ptr(),
                torch.cuda.current_stream().cuda_stream,
            )
            self.assertNotEqual(status, 0)
            self.assertIn(b"must not overlap", arena._lib.q38_host_kv_last_error())

            status = self.dedup(
                arena._handle,
                0,
                ids.data_ptr(),
                ids.numel(),
                ids.element_size(),
                row_map.data_ptr(),
                slots,
                outputs[0].data_ptr(),
                outputs[0].data_ptr(),
                torch.cuda.current_stream().cuda_stream,
            )
            self.assertNotEqual(status, 0)
            self.assertIn(b"must not overlap", arena._lib.q38_host_kv_last_error())

            cpu_map = torch.empty(slots, dtype=torch.int32)
            status = self.dedup(
                arena._handle,
                0,
                ids.data_ptr(),
                ids.numel(),
                ids.element_size(),
                cpu_map.data_ptr(),
                slots,
                outputs[0].data_ptr(),
                outputs[1].data_ptr(),
                torch.cuda.current_stream().cuda_stream,
            )
            self.assertNotEqual(status, 0)
            # CUDA versions may reject an unregistered host pointer in the
            # attribute query or return a non-device memory classification.
            self.assertRegex(
                arena._lib.q38_host_kv_last_error().decode(),
                "CUDA-addressable|CUDA device memory",
            )
            self.assertTrue(torch.equal(outputs[0].view(torch.uint8), sentinel_k))
            self.assertTrue(torch.equal(outputs[1].view(torch.uint8), sentinel_v))

            self.assertEqual(self.dedup(arena._handle, 0, 0, 0, 4, 0, 0, 0, 0, 0), 0)
        finally:
            arena.close()

    def test_alternating_layers_and_new_scattered_content(self):
        slots = 17
        arena = self.arena(slots, layers=2)
        ids = torch.tensor([0, 7, 7, 3, -1, 0], dtype=torch.int32, device="cuda")
        row_map = torch.empty(slots, dtype=torch.int32, device="cuda")
        output_k = torch.empty(
            (ids.numel(), 2, 256), dtype=torch.float8_e4m3fn, device="cuda"
        )
        output_v = torch.empty_like(output_k)
        try:
            layers = {
                0: self.initialize(
                    arena, torch.float8_e4m3fn, 2, 256, layer=0, offset=0
                ),
                1: self.initialize(
                    arena, torch.float8_e4m3fn, 2, 256, layer=1, offset=47
                ),
            }
            for layer in (0, 1, 0):
                self.assertEqual(
                    self.call(
                        arena,
                        ids,
                        row_map,
                        output_k,
                        output_v,
                        layer=layer,
                    ),
                    0,
                )
                arena.check_errors()
                source_k, source_v = layers[layer]
                for output, source in ((output_k, source_k), (output_v, source_v)):
                    expected = torch.zeros(
                        (ids.numel(), 512), dtype=torch.uint8, device="cuda"
                    )
                    valid = ids >= 0
                    expected[valid] = source[ids[valid].long()]
                    self.assertTrue(
                        torch.equal(
                            output.view(torch.uint8).reshape(ids.numel(), 512),
                            expected,
                        )
                    )

            layers[0] = self.initialize(
                arena, torch.float8_e4m3fn, 2, 256, layer=0, offset=91
            )
            self.assertEqual(
                self.call(arena, ids, row_map, output_k, output_v, layer=0), 0
            )
            arena.check_errors()
            self.assertTrue(
                torch.equal(output_k[1].view(torch.uint8).reshape(-1), layers[0][0][7])
            )
            self.assertTrue(
                torch.equal(output_v[1].view(torch.uint8).reshape(-1), layers[0][1][7])
            )
        finally:
            arena.close()

    def test_graph_replay_dynamic_indices(self):
        slots = 13
        arena = self.arena(slots)
        lease = arena.retain_for_graph()
        stream = torch.cuda.Stream()
        graph = torch.cuda.CUDAGraph()
        ids = torch.tensor([0, 3, 3, -1, 7, 0], dtype=torch.int64, device="cuda")
        row_map = torch.empty(slots, dtype=torch.int32, device="cuda")
        output_k = torch.empty(
            (ids.numel(), 2, 256), dtype=torch.float8_e4m3fn, device="cuda"
        )
        output_v = torch.empty_like(output_k)
        try:
            source_k, source_v = self.initialize(arena, torch.float8_e4m3fn, 2, 256)
            self.assertEqual(
                self.call(arena, ids, row_map, output_k, output_v, stream), 0
            )
            stream.synchronize()
            arena.check_errors()
            with torch.cuda.graph(graph, stream=stream):
                self.assertEqual(
                    self.call(arena, ids, row_map, output_k, output_v, stream), 0
                )
            patterns = ([0, 3, 3, -1, 7, 0], [5, 5, 2, 2, 2, 8])
            for requested in patterns:
                with torch.cuda.stream(stream):
                    ids.copy_(torch.tensor(requested, dtype=torch.int64, device="cuda"))
                    graph.replay()
                stream.synchronize()
                arena.check_errors()
                expected_k = torch.zeros(
                    (len(requested), 512), dtype=torch.uint8, device="cuda"
                )
                expected_v = torch.zeros_like(expected_k)
                for row, slot in enumerate(requested):
                    if slot >= 0:
                        expected_k[row].copy_(source_k[slot])
                        expected_v[row].copy_(source_v[slot])
                self.assertTrue(
                    torch.equal(
                        output_k.view(torch.uint8).reshape(len(requested), 512),
                        expected_k,
                    )
                )
                self.assertTrue(
                    torch.equal(
                        output_v.view(torch.uint8).reshape(len(requested), 512),
                        expected_v,
                    )
                )
        finally:
            del graph
            stream.synchronize()
            lease.close()
            arena.close()


if __name__ == "__main__":
    unittest.main()

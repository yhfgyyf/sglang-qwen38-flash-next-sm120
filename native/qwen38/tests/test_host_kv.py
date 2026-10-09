"""Focused CPU-contract and SM120 GPU tests for mapped-host ordinary KV.

The CPU class is safe while another process owns the GPU.  Run the whole file
only during an explicitly exclusive SM120 window.
"""

import ctypes as C
import math
import os
import unittest
from types import SimpleNamespace

import torch

from sglang.srt.model_executor.qwen38_host_kv import (
    HostKVArena,
    _validate_library_abi,
    library,
    required_bytes,
)


class HostKVGeometryTests(unittest.TestCase):
    def test_default_geometry_and_native_size_agree(self):
        self.assertEqual(required_bytes(12, 131072), 12 * 131072 * 2 * 2 * 256 * 2)
        self.assertEqual(
            required_bytes(12, 131072, dtype=torch.float8_e4m3fn),
            required_bytes(12, 131072) // 2,
        )
        result = C.c_uint64()
        lib = library()
        self.assertEqual(lib.q38_host_kv_abi_version(), 2)
        self.assertEqual(
            lib.q38_host_kv_required_bytes(12, 131072, 2, 256, 2, C.byref(result)),
            0,
        )
        self.assertEqual(result.value, required_bytes(12, 131072))

    def test_fp8_budget_is_exactly_half_bf16_and_rejects_one_byte_short(self):
        fp8_bytes = required_bytes(1, 17, 1, 255, dtype=torch.float8_e4m3fn)
        bf16_bytes = required_bytes(1, 17, 1, 255)
        self.assertEqual(fp8_bytes * 2, bf16_bytes)
        with self.assertRaisesRegex(MemoryError, "exceeding budget"):
            HostKVArena(
                1,
                17,
                heads=1,
                head_dim=255,
                dtype=torch.float8_e4m3fn,
                byte_budget=fp8_bytes - 1,
            )

    def test_python_rejects_stale_or_wrong_native_abi(self):
        with self.assertRaisesRegex(RuntimeError, "ABI mismatch.*missing"):
            _validate_library_abi(SimpleNamespace())

        class Version:
            def __call__(self):
                return 1

        with self.assertRaisesRegex(RuntimeError, "requires 2.*reports 1"):
            _validate_library_abi(SimpleNamespace(q38_host_kv_abi_version=Version()))

    def test_empty_invalid_and_overflow_geometry(self):
        for values in ((0, 1, 2, 256), (1, 0, 2, 256), (1, 1, 0, 256)):
            with self.subTest(values=values), self.assertRaises(ValueError):
                required_bytes(*values)
        with self.assertRaises(OverflowError):
            required_bytes(1 << 63, 2, 2, 256)
        with self.assertRaisesRegex(ValueError, "byte_budget"):
            HostKVArena(1, 1, byte_budget=0)
        with self.assertRaisesRegex(MemoryError, "exceeding budget"):
            HostKVArena(1, 2, byte_budget=required_bytes(1, 1))

    def test_native_invalid_geometry_has_private_error(self):
        result = C.c_uint64(123)
        lib = library()
        self.assertNotEqual(
            lib.q38_host_kv_required_bytes(0, 1, 2, 256, 2, C.byref(result)), 0
        )
        self.assertIn(b"positive", lib.q38_host_kv_last_error())
        self.assertNotEqual(
            lib.q38_host_kv_required_bytes(1, 1, 2, 256, 3, C.byref(result)), 0
        )
        self.assertIn(b"element size", lib.q38_host_kv_last_error())


class HostKVGPUTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if torch.cuda.get_device_capability() != (12, 0):
            raise RuntimeError("host KV GPU tests require the authorized SM120 GPU")

    def arena(self, layers=3, slots=67, **kwargs):
        dtype = kwargs.get("dtype", torch.bfloat16)
        heads = kwargs.get("heads", 2)
        head_dim = kwargs.get("head_dim", 256)
        size = required_bytes(layers, slots, heads, head_dim, dtype=dtype)
        label = kwargs.pop("label", "test")
        return HostKVArena(
            layers,
            slots,
            byte_budget=size,
            label=label,
            **kwargs,
        )

    @staticmethod
    def fp8_codes(shape, offset=0):
        raw = (
            torch.arange(math.prod(shape), dtype=torch.int64, device="cuda")
            .add_(offset)
            .remainder_(256)
            .to(torch.uint8)
            .reshape(shape)
        )
        return raw.view(torch.float8_e4m3fn)

    @staticmethod
    def strided_rows(count, offset):
        row_elements = 2 * 256
        storage = torch.empty(
            (count, row_elements + 13), dtype=torch.bfloat16, device="cuda"
        )
        rows = storage[:, :row_elements].view(count, 2, 256)
        source = (
            torch.arange(count * row_elements, device="cuda", dtype=torch.float32)
            .add_(offset)
            .remainder_(997)
            .view(count, 2, 256)
            .to(torch.bfloat16)
        )
        rows.copy_(source)
        return rows, source

    def test_exact_scatter_gather_layers_id_width_stride_and_boundaries(self):
        arena = self.arena()
        try:
            physical = [0, 66, 5, 31, 1, 64]
            for layer in range(3):
                keys, expected_keys = self.strided_rows(len(physical), layer * 100)
                values, expected_values = self.strided_rows(
                    len(physical), layer * 100 + 37
                )
                self.assertFalse(keys.is_contiguous())
                for dtype in (torch.int32, torch.int64):
                    ids = torch.tensor(physical, dtype=dtype, device="cuda")
                    arena.scatter(layer, ids, keys, values)
                    order = torch.tensor(
                        [66, 0, 31, 66, 64, 5], dtype=dtype, device="cuda"
                    )
                    out_k = torch.empty(
                        (6, 2, 256), dtype=torch.bfloat16, device="cuda"
                    )
                    out_v = torch.empty_like(out_k)
                    arena.gather(layer, order, out_k, out_v)
                    arena.check_errors()
                    lookup = {slot: index for index, slot in enumerate(physical)}
                    expected_order = torch.tensor(
                        [lookup[int(slot)] for slot in order.cpu()], device="cuda"
                    )
                    self.assertTrue(
                        torch.equal(
                            out_k.view(torch.int16),
                            expected_keys[expected_order].view(torch.int16),
                        )
                    )
                    self.assertTrue(
                        torch.equal(
                            out_v.view(torch.int16),
                            expected_values[expected_order].view(torch.int16),
                        )
                    )
        finally:
            arena.close()

    def test_fp8_all_codes_are_byte_exact(self):
        arena = self.arena(
            layers=1,
            slots=5,
            heads=1,
            head_dim=256,
            dtype=torch.float8_e4m3fn,
        )
        ids = torch.tensor([4], dtype=torch.int64, device="cuda")
        keys = self.fp8_codes((1, 1, 256))
        values = self.fp8_codes((1, 1, 256), offset=73)
        out_k = torch.empty_like(keys)
        out_v = torch.empty_like(values)
        try:
            self.assertEqual(arena.dtype, torch.float8_e4m3fn)
            self.assertEqual(arena.nbytes * 2, required_bytes(1, 5, 1, 256))
            arena.scatter(0, ids, keys, values)
            arena.gather(0, ids, out_k, out_v)
            arena.check_errors()
            self.assertTrue(
                torch.equal(out_k.view(torch.uint8), keys.view(torch.uint8))
            )
            self.assertTrue(
                torch.equal(out_v.view(torch.uint8), values.view(torch.uint8))
            )
            self.assertEqual(
                sorted(out_k.view(torch.uint8).flatten().cpu().tolist()),
                list(range(256)),
            )
        finally:
            arena.close()

    def test_vector_gather_warp_rows_counts_ids_padding_and_oob_are_exact(self):
        counts = (1, 3, 4, 5, 31, 33)
        slots = 41
        pattern = [7, -1, 7, slots, slots - 1, -99]
        pattern.extend((index * 11) % (slots - 1) + 1 for index in range(27))
        # Keep the sole OOB row at index 4 so a tail warp must both report the
        # deferred error and zero its output for every count >= 5.
        pattern[3], pattern[4] = pattern[4], pattern[3]
        for dtype in (torch.float8_e4m3fn, torch.bfloat16):
            arena = self.arena(layers=1, slots=slots, dtype=dtype)
            row_bytes = 2 * 256 * dtype.itemsize
            source_bytes = (
                torch.arange(
                    (slots - 1) * row_bytes,
                    dtype=torch.int64,
                    device="cuda",
                )
                .add_(19)
                .remainder_(256)
                .to(torch.uint8)
                .reshape(slots - 1, row_bytes)
            )
            value_bytes = source_bytes.add(73)
            keys = source_bytes.view(dtype).reshape(slots - 1, 2, 256)
            values = value_bytes.view(dtype).reshape(slots - 1, 2, 256)
            try:
                for id_dtype in (torch.int32, torch.int64):
                    physical = torch.arange(1, slots, dtype=id_dtype, device="cuda")
                    arena.scatter(0, physical, keys, values)
                    arena.check_errors()
                    for count in counts:
                        with self.subTest(dtype=dtype, id_dtype=id_dtype, count=count):
                            requested = pattern[:count]
                            ids = torch.tensor(requested, dtype=id_dtype, device="cuda")
                            out_k = torch.empty(
                                (count, 2, 256), dtype=dtype, device="cuda"
                            )
                            out_v = torch.empty_like(out_k)
                            arena.gather(0, ids, out_k, out_v)
                            if count >= 5:
                                with self.assertRaisesRegex(
                                    RuntimeError, f"input row 4: {slots}"
                                ):
                                    arena.check_errors()
                            else:
                                arena.check_errors()
                            expected_k = torch.zeros(
                                (count, row_bytes),
                                dtype=torch.uint8,
                                device="cuda",
                            )
                            expected_v = torch.zeros_like(expected_k)
                            for row, slot in enumerate(requested):
                                if 0 < slot < slots:
                                    expected_k[row].copy_(source_bytes[slot - 1])
                                    expected_v[row].copy_(value_bytes[slot - 1])
                            self.assertTrue(
                                torch.equal(
                                    out_k.view(torch.uint8).reshape(count, row_bytes),
                                    expected_k,
                                )
                            )
                            self.assertTrue(
                                torch.equal(
                                    out_v.view(torch.uint8).reshape(count, row_bytes),
                                    expected_v,
                                )
                            )
            finally:
                arena.close()

    def test_fp8_odd_rows_and_unaligned_leading_strides_are_byte_exact(self):
        arena = self.arena(
            layers=1,
            slots=9,
            heads=1,
            head_dim=255,
            dtype=torch.float8_e4m3fn,
        )
        ids = torch.tensor([8, 2, 6], dtype=torch.int32, device="cuda")
        key_storage = self.fp8_codes((3, 257), offset=19)
        value_storage = self.fp8_codes((3, 257), offset=101)
        keys = key_storage[:, 1:256].view(3, 1, 255)
        values = value_storage[:, 1:256].view(3, 1, 255)
        out_k = torch.empty_like(keys, memory_format=torch.contiguous_format)
        out_v = torch.empty_like(values, memory_format=torch.contiguous_format)
        try:
            self.assertEqual(arena.row_bytes, 255)
            self.assertEqual(keys.stride(0), 257)
            self.assertEqual(keys.data_ptr() % 2, 1)
            self.assertFalse(keys.is_contiguous())
            arena.scatter(0, ids, keys, values)
            arena.gather(0, ids, out_k, out_v)
            arena.check_errors()
            self.assertTrue(
                torch.equal(out_k.view(torch.uint8), keys.view(torch.uint8))
            )
            self.assertTrue(
                torch.equal(out_v.view(torch.uint8), values.view(torch.uint8))
            )
        finally:
            arena.close()

    def test_duplicate_order_negative_padding_and_full_arena_chunk(self):
        slots = 1025
        arena = self.arena(layers=1, slots=slots)
        try:
            ids = torch.arange(slots, dtype=torch.int32, device="cuda")
            keys = (
                torch.arange(slots, dtype=torch.float32, device="cuda")
                .to(torch.bfloat16)[:, None, None]
                .expand(-1, 2, 256)
                .contiguous()
            )
            values = (keys + 7).contiguous()
            arena.scatter(0, ids, keys, values)
            gather_ids = torch.flip(ids, dims=(0,))
            out_k = torch.empty_like(keys)
            out_v = torch.empty_like(values)
            arena.gather(0, gather_ids, out_k, out_v)
            arena.check_errors()
            self.assertTrue(torch.equal(out_k, torch.flip(keys, dims=(0,))))
            self.assertTrue(torch.equal(out_v, torch.flip(values, dims=(0,))))

            padded_ids = torch.tensor([-1, 17, -99, 17], device="cuda")
            padded_k = torch.empty((4, 2, 256), dtype=torch.bfloat16, device="cuda")
            padded_v = torch.empty_like(padded_k)
            arena.gather(0, padded_ids, padded_k, padded_v)
            arena.check_errors()
            self.assertTrue(torch.count_nonzero(padded_k[[0, 2]]) == 0)
            self.assertTrue(torch.count_nonzero(padded_v[[0, 2]]) == 0)
            self.assertTrue(torch.equal(padded_k[1], padded_k[3]))
            self.assertTrue(torch.equal(padded_v[1], padded_v[3]))
        finally:
            arena.close()

    @unittest.skipUnless(
        os.environ.get("QWEN38_HOST_KV_LARGE_TEST") == "1",
        "set QWEN38_HOST_KV_LARGE_TEST=1 for the 128K-row allocation test",
    )
    def test_opt_in_full_128k_context_gather(self):
        # Kept opt-in because it needs ~256 MiB pinned host and ~512 MiB GPU,
        # above the bounded few-MiB shared-machine validation window.
        slots = 128 * 1024
        arena = self.arena(layers=1, slots=slots)
        try:
            ids = torch.arange(slots, dtype=torch.int64, device="cuda")
            row_values = (ids % 997).to(torch.bfloat16)
            keys = row_values[:, None, None].expand(-1, 2, 256).contiguous()
            values = (keys + 3).contiguous()
            arena.scatter(0, ids, keys, values)
            reverse_ids = torch.flip(ids, dims=(0,))
            out_k = torch.empty_like(keys)
            out_v = torch.empty_like(values)
            arena.gather(0, reverse_ids, out_k, out_v)
            arena.check_errors()
            expected_rows = torch.flip(row_values, dims=(0,))
            self.assertTrue(torch.equal(out_k[:, 0, 0], expected_rows))
            self.assertTrue(torch.equal(out_v[:, 1, 255], expected_rows + 3))
        finally:
            arena.close()

    def test_captured_replay_uses_changing_ids_and_content(self):
        arena = self.arena(layers=1, slots=11)
        lease = arena.retain_for_graph()
        stream = torch.cuda.Stream()
        graph = torch.cuda.CUDAGraph()
        ids = torch.tensor([0, 3, 10, 6], dtype=torch.int64, device="cuda")
        keys = torch.empty((4, 2, 256), dtype=torch.bfloat16, device="cuda")
        values = torch.empty_like(keys)
        out_k = torch.empty_like(keys)
        out_v = torch.empty_like(values)
        try:
            # Materialize modules/events before stream capture.
            keys.zero_()
            values.zero_()
            arena.scatter(0, ids, keys, values)
            arena.gather(0, ids, out_k, out_v)
            arena.check_errors()
            torch.cuda.synchronize()
            with torch.cuda.graph(graph, stream=stream):
                arena.scatter(0, ids, keys, values)
                arena.gather(0, ids, out_k, out_v)
            for iteration, requested in enumerate(([0, 3, 10, 6], [1, 9, 4, 2])):
                ids.copy_(torch.tensor(requested, device="cuda"))
                expected_k = torch.full_like(keys, iteration + 11)
                expected_v = torch.full_like(values, iteration + 29)
                keys.copy_(expected_k)
                values.copy_(expected_v)
                graph.replay()
                arena.check_errors()
                self.assertTrue(torch.equal(out_k, expected_k))
                self.assertTrue(torch.equal(out_v, expected_v))
        finally:
            del graph
            torch.cuda.synchronize()
            lease.close()
            arena.close()

    def test_fp8_captured_replay_uses_dynamic_ids_and_bytes(self):
        arena = self.arena(
            layers=1,
            slots=11,
            heads=1,
            head_dim=255,
            dtype=torch.float8_e4m3fn,
        )
        lease = arena.retain_for_graph()
        stream = torch.cuda.Stream()
        graph = torch.cuda.CUDAGraph()
        ids = torch.tensor([0, 3, 10, 6], dtype=torch.int64, device="cuda")
        keys = self.fp8_codes((4, 1, 255))
        values = self.fp8_codes((4, 1, 255), offset=37)
        out_k = torch.empty_like(keys)
        out_v = torch.empty_like(values)
        try:
            arena.scatter(0, ids, keys, values)
            arena.gather(0, ids, out_k, out_v)
            arena.check_errors()
            torch.cuda.synchronize()
            with torch.cuda.graph(graph, stream=stream):
                arena.scatter(0, ids, keys, values)
                arena.gather(0, ids, out_k, out_v)
            for iteration, requested in enumerate(([0, 3, 10, 6], [1, 9, 4, 2])):
                ids.copy_(torch.tensor(requested, device="cuda"))
                keys.view(torch.uint8).copy_(
                    self.fp8_codes((4, 1, 255), offset=iteration + 91).view(torch.uint8)
                )
                values.view(torch.uint8).copy_(
                    self.fp8_codes((4, 1, 255), offset=iteration + 173).view(
                        torch.uint8
                    )
                )
                graph.replay()
                arena.check_errors()
                self.assertTrue(
                    torch.equal(out_k.view(torch.uint8), keys.view(torch.uint8))
                )
                self.assertTrue(
                    torch.equal(out_v.view(torch.uint8), values.view(torch.uint8))
                )
        finally:
            del graph
            torch.cuda.synchronize()
            lease.close()
            arena.close()

    def test_explicit_capturing_stream_enforces_lease_and_debug_guards(self):
        ids = torch.tensor([1], dtype=torch.int32, device="cuda")
        row = torch.ones((1, 2, 256), dtype=torch.bfloat16, device="cuda")
        output = torch.empty_like(row)

        def capture_while_current_stream_is_eager(arena, operation, pattern):
            stream = torch.cuda.Stream()
            graph = torch.cuda.CUDAGraph()
            marker = torch.zeros(1, device="cuda")
            torch.cuda.synchronize()
            with torch.cuda.stream(stream):
                graph.capture_begin()
            try:
                self.assertFalse(torch.cuda.is_current_stream_capturing())
                self.assertTrue(stream.is_capturing())
                with self.assertRaisesRegex(RuntimeError, pattern):
                    operation(stream)
            finally:
                if stream.is_capturing():
                    with torch.cuda.stream(stream):
                        marker.add_(1)
                        graph.capture_end()
                del graph
                torch.cuda.synchronize()

        no_lease = self.arena(layers=1, slots=3)
        try:
            capture_while_current_stream_is_eager(
                no_lease,
                lambda stream: no_lease.scatter(0, ids, row, row, stream=stream),
                "retain_for_graph",
            )
        finally:
            no_lease.close()

        debug = self.arena(layers=1, slots=3, debug=True)
        lease = debug.retain_for_graph()
        try:
            capture_while_current_stream_is_eager(
                debug,
                lambda stream: debug.gather(0, ids, output, output, stream=stream),
                "debug ID checking",
            )
        finally:
            lease.close()
            debug.close()

    def test_separately_captured_graphs_share_arena_and_lifetime(self):
        arena = self.arena(layers=1, slots=13)
        leases = [arena.retain_for_graph(), arena.retain_for_graph()]
        streams = [torch.cuda.Stream(), torch.cuda.Stream()]
        graphs = [torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()]
        ids = [
            torch.tensor([0, 4], dtype=torch.int32, device="cuda"),
            torch.tensor([12, 7, 2], dtype=torch.int64, device="cuda"),
        ]
        keys = [
            torch.full(
                (len(item), 2, 256), index + 3, dtype=torch.bfloat16, device="cuda"
            )
            for index, item in enumerate(ids)
        ]
        values = [item + 20 for item in keys]
        outputs = [
            (torch.empty_like(k), torch.empty_like(v)) for k, v in zip(keys, values)
        ]
        try:
            # Warm up both ID widths before capture.
            for index in range(2):
                arena.scatter(0, ids[index], keys[index], values[index])
                arena.gather(0, ids[index], *outputs[index])
            arena.check_errors()
            torch.cuda.synchronize()
            for index in range(2):
                with torch.cuda.graph(graphs[index], stream=streams[index]):
                    arena.scatter(0, ids[index], keys[index], values[index])
                    arena.gather(0, ids[index], *outputs[index])
            with self.assertRaisesRegex(RuntimeError, "2 graph lease"):
                arena.close()
            for iteration in range(3):
                for index in (iteration % 2, (iteration + 1) % 2):
                    keys[index].fill_(iteration * 10 + index)
                    values[index].fill_(iteration * 10 + index + 5)
                    graphs[index].replay()
                    arena.check_errors()
                    self.assertTrue(torch.equal(outputs[index][0], keys[index]))
                    self.assertTrue(torch.equal(outputs[index][1], values[index]))
        finally:
            graphs.clear()
            torch.cuda.synchronize()
            for lease in leases:
                lease.close()
            arena.close()

    def test_invalid_ids_empty_calls_and_lifetime_failures(self):
        arena = self.arena(layers=1, slots=4)
        empty_ids = torch.empty(0, dtype=torch.int32, device="cuda")
        empty = torch.empty((0, 2, 256), dtype=torch.bfloat16, device="cuda")
        try:
            arena.scatter(0, empty_ids, empty, empty)
            arena.gather(0, empty_ids, empty, empty)

            bad_gather = torch.tensor([4], dtype=torch.int32, device="cuda")
            out_k = torch.empty((1, 2, 256), dtype=torch.bfloat16, device="cuda")
            out_v = torch.empty_like(out_k)
            arena.gather(0, bad_gather, out_k, out_v)
            with self.assertRaisesRegex(RuntimeError, "gather.*out of bounds"):
                arena.check_errors()
            self.assertEqual(torch.count_nonzero(out_k).item(), 0)
            self.assertEqual(torch.count_nonzero(out_v).item(), 0)

            bad_scatter = torch.tensor([-1], dtype=torch.int64, device="cuda")
            row = torch.zeros((1, 2, 256), dtype=torch.bfloat16, device="cuda")
            arena.scatter(0, bad_scatter, row, row)
            with self.assertRaisesRegex(RuntimeError, "scatter.*out of bounds"):
                arena.check_errors()

            lease = arena.retain_for_graph()
            with self.assertRaisesRegex(RuntimeError, "graph lease"):
                arena.close()
            lease.close()
        finally:
            arena.close()
        with self.assertRaisesRegex(RuntimeError, "closed"):
            arena.check_errors()

    def test_target_and_draft_are_isolated(self):
        target = self.arena(layers=12, slots=3, label="target")
        draft = self.arena(layers=1, slots=3, label="draft")
        ids = torch.tensor([1], dtype=torch.int64, device="cuda")
        target_k = torch.full((1, 2, 256), 12, dtype=torch.bfloat16, device="cuda")
        target_v = torch.full_like(target_k, 13)
        draft_k = torch.full_like(target_k, 1)
        draft_v = torch.full_like(target_k, 2)
        out_k = torch.empty_like(target_k)
        out_v = torch.empty_like(target_v)
        try:
            padding = torch.tensor([0], dtype=torch.int32, device="cuda")
            target.gather(11, padding, out_k, out_v)
            target.check_errors()
            self.assertEqual(torch.count_nonzero(out_k).item(), 0)
            self.assertEqual(torch.count_nonzero(out_v).item(), 0)
            draft.gather(0, padding, out_k, out_v)
            draft.check_errors()
            self.assertEqual(torch.count_nonzero(out_k).item(), 0)
            self.assertEqual(torch.count_nonzero(out_v).item(), 0)

            target.scatter(11, ids, target_k, target_v)
            draft.scatter(0, ids, draft_k, draft_v)
            target.gather(11, ids, out_k, out_v)
            target.check_errors()
            self.assertTrue(torch.equal(out_k, target_k))
            self.assertTrue(torch.equal(out_v, target_v))
            draft.gather(0, ids, out_k, out_v)
            draft.check_errors()
            self.assertTrue(torch.equal(out_k, draft_k))
            self.assertTrue(torch.equal(out_v, draft_v))
        finally:
            target.close()
            draft.close()


if __name__ == "__main__":
    unittest.main()

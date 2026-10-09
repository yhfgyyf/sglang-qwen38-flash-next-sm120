"""Real CUDA/storage tests, independent of a full model load.

Run with the pinned SGLang Python environment and this worktree on PYTHONPATH.
Missing native builds/CUDA are failures for this explicit GPU test entrypoint.
"""
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from sglang.srt.model_executor.qwen38_native import (
    NativeGraphPlan, NativePLEOperation, NativePLEPipe, _check, libraries,
)


class NativeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if torch.cuda.get_device_capability() != (12, 0):
            raise RuntimeError("these tests require the authorized SM120 GPU")
        cls.native, cls.store = libraries()

    def test_exhaustive_fp8_conversion(self):
        codes = torch.arange(256, dtype=torch.int32, device="cuda").to(torch.uint8)
        output = torch.empty(256, dtype=torch.bfloat16, device="cuda")
        expected = codes.view(torch.float8_e4m3fn).to(torch.bfloat16)
        _check(self.native.q38_decode_fp8(codes.data_ptr(), output.data_ptr(),
            256, torch.cuda.current_stream().cuda_stream), self.native)
        torch.testing.assert_close(output, expected, rtol=0, atol=0, equal_nan=True)
        finite = ~torch.isnan(expected)
        self.assertTrue(torch.equal(output.view(torch.int16)[finite], expected.view(torch.int16)[finite]))

    def fixture(self, directory):
        shards = []
        reference = []
        for index, (offset, count) in enumerate(((4079, 91), (113, 70))):
            path = Path(directory) / f"shard{index}.bin"
            # Non-NaN E4M3 bytes, including signed zeros and values.
            payload = bytes((i * 13 + index * 7) % 126 for i in range(count * 160))
            path.write_bytes(bytes(offset) + payload)
            tensor = SimpleNamespace(path=path, offset=offset)
            start = len(reference)
            reference.extend(payload[i:i + 160] for i in range(0, len(payload), 160))
            shards.append(SimpleNamespace(tensor=tensor, row_start=start, row_end=len(reference)))
        manifest = SimpleNamespace(dtype="F8_E4M3", row_bytes=160, shards=shards)
        return manifest, reference

    def test_async_gather_duplicates_boundaries_and_stale_ticket(self):
        with tempfile.TemporaryDirectory(prefix="q38-native-test-") as directory:
            manifest, rows = self.fixture(directory)
            pipe = NativePLEPipe(manifest, cache_bytes=160 * 4, queue_depth=2,
                                 max_batch=3, max_rows=32)
            try:
                ids = torch.tensor([0, 25, 90, 91, 160, 25, 0], device="cuda")
                output = torch.empty((len(ids), 160), dtype=torch.bfloat16, device="cuda")
                expected_codes = torch.tensor(list(b"".join(rows[i] for i in ids.tolist())), dtype=torch.uint8)
                expected = expected_codes.view(torch.float8_e4m3fn).to(torch.bfloat16).reshape_as(output)
                for _ in range(3):
                    ticket = pipe.issue(ids)
                    with self.assertRaisesRegex(RuntimeError, "already pending"):
                        pipe.issue(ids)
                    with self.assertRaisesRegex(RuntimeError, "stale"):
                        pipe.collect(ticket + 1, output)
                    pipe.collect(ticket, output)
                    torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)
                bad = torch.tensor([-1], device="cuda")
                ticket = pipe.issue(bad)
                with self.assertRaises(RuntimeError):
                    pipe.collect(ticket, output[:1])
                ticket = pipe.issue(ids)
                pipe.collect(ticket, output)
                torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)
            finally:
                pipe.close()

    def test_native_plan_replays_dynamic_ids_without_python_breaks(self):
        with tempfile.TemporaryDirectory(prefix="q38-native-plan-") as directory:
            manifest, rows = self.fixture(directory)
            pipe = NativePLEPipe(manifest, cache_bytes=160 * 16, max_rows=32)
            pool = torch.cuda.graph_pool_handle()
            stream = torch.cuda.Stream()
            source = torch.tensor([0, 25, 90, 91], device="cuda")
            ids = torch.empty_like(source)
            gather = torch.empty((4, 160), device="cuda", dtype=torch.bfloat16)
            independent = torch.zeros(1, device="cuda")
            output = torch.empty_like(gather)
            torch.cuda.synchronize()
            segments = [torch.cuda.CUDAGraph() for _ in range(3)]
            with torch.cuda.graph(segments[0], pool=pool, stream=stream):
                ids.copy_(source)
            with torch.cuda.graph(segments[1], pool=pool, stream=stream):
                independent.add_(1)
            with torch.cuda.graph(segments[2], pool=pool, stream=stream):
                output.copy_(gather)
            plan = NativeGraphPlan(segments, [
                NativePLEOperation("issue", pipe, ids, 4),
                NativePLEOperation("collect", pipe, gather, 4),
            ], pool=pool)
            try:
                with self.assertRaisesRegex(RuntimeError, "plans"):
                    pipe.close()
                for offset in (0, 1, 2, 0, 1):
                    requested = [offset, 25, 90 + offset, 91]
                    source.copy_(torch.tensor(requested, device="cuda"))
                    plan.replay(torch.cuda.current_stream())
                    expected = torch.tensor(list(b"".join(rows[i] for i in requested)), dtype=torch.uint8)
                    expected = expected.view(torch.float8_e4m3fn).to(torch.bfloat16).reshape_as(output)
                    torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)
                self.assertEqual(independent.item(), 5)
                # The pool is shared by batch shapes: a second stream must be
                # rejected before any graph can touch those same allocations.
                with self.assertRaisesRegex(RuntimeError, "one replay stream"):
                    plan.replay(torch.cuda.Stream())
                # A real storage error after earlier graph launches must drain
                # those launches and poison this plan before owners can close.
                source.fill_(-1)
                with self.assertRaises(RuntimeError):
                    plan.replay(torch.cuda.current_stream())
                self.assertEqual(independent.item(), 6)
                with self.assertRaisesRegex(RuntimeError, "poisoned"):
                    plan.replay(torch.cuda.current_stream())
            finally:
                plan.close()
                pipe.close()

    def test_capture_rejection_preserves_pending_gather(self):
        with tempfile.TemporaryDirectory(prefix="q38-native-capture-") as directory:
            manifest, rows = self.fixture(directory)
            pipe = NativePLEPipe(manifest, cache_bytes=0, max_rows=2)
            ids = torch.tensor([0], device="cuda")
            output = torch.empty((1, 160), dtype=torch.bfloat16, device="cuda")
            marker = torch.zeros(1, device="cuda")
            stream = torch.cuda.Stream()
            graph = torch.cuda.CUDAGraph()
            try:
                torch.cuda.synchronize()
                with torch.cuda.graph(graph, stream=stream):
                    with self.assertRaisesRegex(RuntimeError, "capturing"):
                        pipe.issue(ids)
                    marker.add_(1)
                self.assertIsNone(pipe._pending)
                ticket = pipe.issue(ids)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    with self.assertRaisesRegex(RuntimeError, "capturing"):
                        pipe.collect(ticket, output, stream)
                    marker.add_(1)
                self.assertIsNotNone(pipe._pending)
                pipe.collect(ticket, output)
                expected = torch.tensor(list(rows[0]), dtype=torch.uint8).view(torch.float8_e4m3fn).to(torch.bfloat16)
                torch.testing.assert_close(output.cpu()[0], expected, rtol=0, atol=0)
            finally:
                pipe.close()

    def test_issue_count_mismatch_is_rejected_before_device_read(self):
        with tempfile.TemporaryDirectory(prefix="q38-native-count-") as directory:
            manifest, _ = self.fixture(directory)
            pipe = NativePLEPipe(manifest, cache_bytes=0, max_rows=8)
            ids = torch.tensor([0], device="cuda")
            output = torch.empty((4, 160), device="cuda", dtype=torch.bfloat16)
            marker = torch.zeros(1, device="cuda")
            stream = torch.cuda.Stream()
            segments = []
            try:
                torch.cuda.synchronize()
                for _ in range(3):
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, stream=stream):
                        marker.add_(1)
                    segments.append(graph)
                with self.assertRaisesRegex(ValueError, "ID tensor size"):
                    NativeGraphPlan(segments, [NativePLEOperation("issue", pipe, ids, 4),
                                              NativePLEOperation("collect", pipe, output, 4)])
                self.assertEqual(pipe._plan_refs, 0)
            finally:
                pipe.close()

    def test_owner_lock_serializes_close_after_collect(self):
        with tempfile.TemporaryDirectory(prefix="q38-native-close-") as directory:
            manifest, _ = self.fixture(directory)
            pipe = NativePLEPipe(manifest, cache_bytes=0, max_rows=2)
            ids = torch.tensor([0], device="cuda")
            output = torch.empty((1, 160), device="cuda", dtype=torch.bfloat16)
            ticket = pipe.issue(ids)
            entered, proceed, closed = threading.Event(), threading.Event(), threading.Event()
            errors = []
            original = pipe.native.q38_pipe_collect

            def paused(*args):
                entered.set()
                if not proceed.wait(5):
                    raise RuntimeError("test synchronization timed out")
                return original(*args)

            def collect():
                try:
                    pipe.collect(ticket, output)
                except Exception as error:
                    errors.append(error)

            def close():
                try:
                    pipe.close()
                except Exception as error:
                    errors.append(error)
                closed.set()

            pipe.native.q38_pipe_collect = paused
            reader = threading.Thread(target=collect)
            closer = threading.Thread(target=close)
            try:
                reader.start()
                self.assertTrue(entered.wait(5))
                closer.start()
                self.assertFalse(closed.wait(0.02))
            finally:
                proceed.set()
                reader.join(5)
                if closer.ident is not None:
                    closer.join(5)
                pipe.native.q38_pipe_collect = original
                pipe.close()
            self.assertFalse(reader.is_alive())
            self.assertFalse(closer.is_alive())
            self.assertEqual(errors, [])

    def test_plan_rejects_unknown_break_and_unmatched_collect(self):
        with self.assertRaisesRegex(ValueError, "unknown graph break"):
            NativeGraphPlan([None, None], [None])
        plan = self.native.q38_plan_create(torch.cuda.current_device())
        try:
            self.assertNotEqual(self.native.q38_plan_seal(plan), 0)
            self.assertIn(b"empty", self.native.q38_native_last_error())
        finally:
            _check(self.native.q38_plan_close(plan), self.native)


if __name__ == "__main__":
    unittest.main()

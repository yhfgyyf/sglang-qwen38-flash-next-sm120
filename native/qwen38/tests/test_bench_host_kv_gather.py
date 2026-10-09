"""CPU-only contracts for the mapped-host gather microbenchmark."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest

import torch


SCRIPT = Path(__file__).resolve().parents[1] / "bench_host_kv_gather.py"
SPEC = importlib.util.spec_from_file_location("q38_bench_host_kv_gather", SCRIPT)
BENCH = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = BENCH
SPEC.loader.exec_module(BENCH)


class HostKVGatherBenchmarkContractTests(unittest.TestCase):
    def test_shared_neighbor_ids_have_exact_shape_locality_and_padding(self):
        concurrency = 4
        ids = BENCH.build_gather_ids(concurrency, 12345, "shared-neighbor")
        self.assertTrue(
            torch.equal(
                ids, BENCH.build_gather_ids(concurrency, 12345, "shared-neighbor")
            )
        )
        shaped = ids.reshape(concurrency, BENCH.VERIFY_QUERIES, BENCH.ROW_STRIDE)
        for request in range(concurrency):
            lower = 1 + request * BENCH.CONTEXT_SLOTS
            upper = lower + BENCH.CONTEXT_SLOTS
            positive_sets = []
            for query in range(BENCH.VERIFY_QUERIES):
                row = shaped[request, query]
                positive = row[: BENCH.VALID_ROWS]
                self.assertTrue(torch.all(row[BENCH.VALID_ROWS :] == -1))
                self.assertEqual(torch.unique(positive).numel(), BENCH.VALID_ROWS)
                self.assertGreaterEqual(int(positive.min()), lower)
                self.assertLess(int(positive.max()), upper)
                blocks = positive[: BENCH.BLOCKS_PER_QUERY * BENCH.BLOCK_SIZE]
                blocks = blocks.reshape(BENCH.BLOCKS_PER_QUERY, BENCH.BLOCK_SIZE)
                self.assertTrue(torch.all(blocks[:, 1:] - blocks[:, :-1] == 1))
                self.assertTrue(torch.all((blocks[:, 0] - lower) % 4 == 0))
                positive_sets.append(set(positive.tolist()))
            for left in range(BENCH.VERIFY_QUERIES):
                for right in range(left + 1, BENCH.VERIFY_QUERIES):
                    self.assertEqual(
                        len(positive_sets[left] & positive_sets[right]),
                        448 * BENCH.BLOCK_SIZE,
                    )

    def test_requests_use_disjoint_contexts_and_rounded_arena_is_bounded(self):
        ids = BENCH.build_gather_ids(10, 99, "independent").reshape(
            10, BENCH.VERIFY_QUERIES, BENCH.ROW_STRIDE
        )
        request_sets = [set(request[request >= 0].tolist()) for request in ids]
        for left in range(10):
            for right in range(left + 1, 10):
                self.assertFalse(request_sets[left] & request_sets[right])
        slots = BENCH.slot_count_for(10)
        self.assertEqual(slots % BENCH.SLOT_ROUNDING, 0)
        self.assertGreater(slots, 10 * BENCH.CONTEXT_SLOTS)
        arena_bytes = BENCH.LAYERS * slots * 2 * BENCH.ROW_BYTES
        self.assertLessEqual(arena_bytes, BENCH.MAX_ARENA_BYTES)

    def test_raw_oracle_is_byte_exact_and_zeroes_only_negative_padding(self):
        ids = torch.tensor([1, -1, 257], dtype=torch.int64)
        keys = BENCH.expected_rows(ids, layer=11, value=False)
        values = BENCH.expected_rows(ids, layer=11, value=True)
        self.assertEqual(tuple(keys.shape), (3, BENCH.ROW_BYTES))
        self.assertEqual(keys.dtype, torch.uint8)
        self.assertEqual(torch.count_nonzero(keys[1]).item(), 0)
        self.assertEqual(torch.count_nonzero(values[1]).item(), 0)
        self.assertFalse(torch.equal(keys[0], keys[2]))
        self.assertFalse(torch.equal(values[0], values[2]))
        self.assertFalse(torch.equal(keys[0], values[0]))

    def test_parser_requires_explicit_library_and_at_least_11_samples(self):
        with self.assertRaises(SystemExit):
            BENCH.parse_args([])
        with self.assertRaises(SystemExit):
            BENCH.parse_args(["--library", "/tmp/candidate.so", "--samples", "10"])
        args = BENCH.parse_args(["--library", "/tmp/candidate.so"])
        self.assertEqual(args.samples, 11)
        self.assertIsNone(args.output)
        self.assertEqual(args.cache_condition, "both")


if __name__ == "__main__":
    unittest.main()

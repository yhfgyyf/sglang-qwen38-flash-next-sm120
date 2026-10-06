import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

from sglang.srt.models.qwen4_ple_cache import BoundedByteLRU, FP8RowCacheReader
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _FakeBacking:
    def __init__(self, total_rows=70_000, row_bytes=4):
        self.manifest = SimpleNamespace(total_rows=total_rows, row_bytes=row_bytes)
        self.rows = {
            row_id: row_id.to_bytes(row_bytes, "little") for row_id in range(total_rows)
        }
        self.read_calls = []
        self.closed = False
        self.close_calls = 0
        self.close_error = None
        self.error = None

    def read_rows(self, row_ids):
        self.read_calls.append(tuple(row_ids))
        if self.error is not None:
            error, self.error = self.error, None
            raise error
        return [self.rows[row_id] for row_id in row_ids]

    def close(self):
        self.close_calls += 1
        self.closed = True
        if self.close_error is not None:
            raise self.close_error


def _budget_for_rows(count, row_bytes=4):
    probe = BoundedByteLRU(1 << 30, key_size=sys.getsizeof)
    for row_id in range(count):
        probe[row_id] = bytes([row_id]) * row_bytes
    return probe.accounted_bytes


class TestBoundedByteLRU(CustomTestCase):
    def test_accounts_owned_objects_and_evicts_in_lru_order(self):
        budget = _budget_for_rows(2)
        cache = BoundedByteLRU(budget, key_size=sys.getsizeof)
        cache[1] = b"aaaa"
        cache[2] = b"bbbb"
        cache.move_to_end(1)
        cache[3] = b"cccc"

        self.assertEqual(list(cache), [1, 3])
        self.assertEqual(cache.payload_bytes, 8)
        self.assertLessEqual(cache.accounted_bytes, budget)
        self.assertEqual(cache.popitem(last=False), (1, b"aaaa"))
        self.assertEqual(len(cache), 1)

    def test_page_key_sizer_can_exclude_shared_path(self):
        path = Path("/shared/model.safetensors")
        key = (path, 4096)
        value = b"page"
        cache = BoundedByteLRU(
            1 << 20,
            key_size=lambda page_key: sys.getsizeof(page_key)
            + sys.getsizeof(page_key[1]),
        )
        cache[key] = value

        empty = BoundedByteLRU(
            1 << 20,
            key_size=lambda page_key: sys.getsizeof(page_key)
            + sys.getsizeof(page_key[1]),
        )
        container_overhead = sys.getsizeof(cache) - sys.getsizeof(empty)
        self.assertEqual(
            cache.accounted_bytes,
            sys.getsizeof(key)
            + sys.getsizeof(key[1])
            + sys.getsizeof(value)
            + container_overhead,
        )

    def test_zero_tiny_and_oversized_budgets_skip_entries(self):
        with self.assertRaisesRegex(ValueError, "cannot be negative"):
            BoundedByteLRU(-1, key_size=sys.getsizeof)

        for budget in (0, 1, sys.getsizeof(b"row")):
            cache = BoundedByteLRU(budget, key_size=sys.getsizeof)
            cache[1] = b"row"
            self.assertEqual(len(cache), 0)
            self.assertEqual(cache.payload_bytes, 0)
            self.assertEqual(cache.accounted_bytes, 0)

        budget = _budget_for_rows(1)
        cache = BoundedByteLRU(budget, key_size=sys.getsizeof)
        cache[1] = b"row1"
        cache[2] = b"x" * budget
        self.assertEqual(list(cache.items()), [(1, b"row1")])

    def test_replacement_and_clear_keep_metadata_exact(self):
        cache = BoundedByteLRU(1 << 20, key_size=sys.getsizeof)
        cache[7] = b"a"
        first_accounted = cache.accounted_bytes
        cache[7] = b"replacement"

        self.assertEqual(cache[7], b"replacement")
        self.assertEqual(cache.payload_bytes, len(b"replacement"))
        self.assertGreater(cache.accounted_bytes, first_accounted)
        cache.clear()
        self.assertEqual(len(cache), 0)
        self.assertEqual(cache.payload_bytes, 0)
        self.assertEqual(cache.accounted_bytes, 0)

    def test_rejects_mutable_values_and_invalid_key_accounting(self):
        cache = BoundedByteLRU(1024, key_size=lambda _: -1)
        with self.assertRaisesRegex(ValueError, "nonnegative integer"):
            cache[1] = b"row"
        cache = BoundedByteLRU(1024, key_size=sys.getsizeof)
        with self.assertRaisesRegex(TypeError, "must be bytes"):
            cache[1] = bytearray(b"row")


class TestFP8RowCacheReader(CustomTestCase):
    def test_exact_values_order_duplicates_and_unique_miss_io(self):
        backing = _FakeBacking(total_rows=10)
        reader = FP8RowCacheReader(backing, _budget_for_rows(10))

        row_ids = [4, 0, 4, 2, 0]
        self.assertEqual(
            reader.read_rows(row_ids), [backing.rows[row_id] for row_id in row_ids]
        )
        self.assertEqual(backing.read_calls, [(4, 0, 2)])
        self.assertEqual(
            reader.read_rows([2, 4, 2]),
            [backing.rows[2], backing.rows[4], backing.rows[2]],
        )
        self.assertEqual(backing.read_calls, [(4, 0, 2)])
        self.assertEqual(
            reader.snapshot_stats(),
            {
                "calls": 2,
                "requested_rows": 8,
                "unique_hits": 2,
                "unique_misses": 3,
                "entries": 3,
                "payload_bytes": 12,
                "accounted_bytes": reader._cache.accounted_bytes,
                "budget_bytes": _budget_for_rows(10),
            },
        )

    def test_partial_hits_only_read_unique_misses(self):
        backing = _FakeBacking(total_rows=10)
        reader = FP8RowCacheReader(backing, _budget_for_rows(10))
        reader.read_rows([1, 2])
        reader.read_rows([2, 3, 3, 1, 4])

        self.assertEqual(backing.read_calls, [(1, 2), (3, 4)])
        stats = reader.snapshot_stats()
        self.assertEqual(stats["unique_hits"], 2)
        self.assertEqual(stats["unique_misses"], 4)
        self.assertEqual(stats["requested_rows"], 7)

    def test_row_cache_obeys_budget_and_zero_disables_retention(self):
        backing = _FakeBacking(total_rows=10)
        reader = FP8RowCacheReader(backing, _budget_for_rows(2))
        reader.read_rows([0, 1])
        reader.read_rows([0])  # Make row 0 most recently used.
        reader.read_rows([2])
        self.assertEqual(list(reader._cache), [0, 2])
        self.assertLessEqual(
            reader.snapshot_stats()["accounted_bytes"], _budget_for_rows(2)
        )

        disabled = FP8RowCacheReader(backing, 0)
        disabled.read_rows([5, 5])
        disabled.read_rows([5])
        self.assertEqual(backing.read_calls[-2:], [(5,), (5,)])
        self.assertEqual(disabled.snapshot_stats()["entries"], 0)

    def test_invalid_ids_fail_before_io_and_cross_shard_ids_work(self):
        backing = _FakeBacking()
        reader = FP8RowCacheReader(backing, _budget_for_rows(4))
        for row_ids in ([-1], [backing.manifest.total_rows]):
            with self.assertRaisesRegex(IndexError, "outside"):
                reader.read_rows(row_ids)
        self.assertEqual(backing.read_calls, [])

        row_ids = [65_536, 2, 65_535, 65_536]
        self.assertEqual(
            reader.read_rows(row_ids), [backing.rows[row_id] for row_id in row_ids]
        )
        self.assertEqual(backing.read_calls, [(65_536, 2, 65_535)])

    def test_backing_exception_does_not_poison_cache(self):
        backing = _FakeBacking(total_rows=10)
        reader = FP8RowCacheReader(backing, _budget_for_rows(2))
        backing.error = OSError("read failed")
        with self.assertRaisesRegex(OSError, "read failed"):
            reader.read_rows([3, 3])

        self.assertEqual(list(reader._cache), [])
        self.assertEqual(reader.read_rows([3]), [backing.rows[3]])
        self.assertEqual(backing.read_calls, [(3,), (3,)])
        self.assertEqual(reader.snapshot_stats()["unique_misses"], 2)

    def test_invalid_backing_payload_is_not_cached(self):
        backing = _FakeBacking(total_rows=10)
        reader = FP8RowCacheReader(backing, _budget_for_rows(2))
        backing.rows[3] = b"short"
        with self.assertRaisesRegex(ValueError, "expected 4"):
            reader.read_rows([3])
        self.assertEqual(list(reader._cache), [])

        backing.rows[3] = bytearray(b"xxxx")
        with self.assertRaisesRegex(TypeError, "immutable bytes"):
            reader.read_rows([3])
        self.assertEqual(list(reader._cache), [])

    def test_close_releases_cached_rows_and_is_repeatable(self):
        backing = _FakeBacking(total_rows=10)
        reader = FP8RowCacheReader(backing, _budget_for_rows(2))
        reader.read_rows([1, 2])

        reader.close()
        self.assertTrue(backing.closed)
        self.assertEqual(reader.snapshot_stats()["entries"], 0)
        self.assertEqual(reader.snapshot_stats()["payload_bytes"], 0)
        self.assertEqual(reader.snapshot_stats()["accounted_bytes"], 0)

        reader.close()
        self.assertEqual(backing.close_calls, 2)
        self.assertEqual(reader.snapshot_stats()["entries"], 0)

    def test_close_releases_cached_rows_when_backing_close_fails(self):
        backing = _FakeBacking(total_rows=10)
        reader = FP8RowCacheReader(backing, _budget_for_rows(2))
        reader.read_rows([1, 2])
        backing.close_error = OSError("close failed")

        with self.assertRaisesRegex(OSError, "close failed"):
            reader.close()
        self.assertEqual(reader.snapshot_stats()["entries"], 0)
        self.assertEqual(reader.snapshot_stats()["payload_bytes"], 0)
        self.assertEqual(reader.snapshot_stats()["accounted_bytes"], 0)


if __name__ == "__main__":
    unittest.main()

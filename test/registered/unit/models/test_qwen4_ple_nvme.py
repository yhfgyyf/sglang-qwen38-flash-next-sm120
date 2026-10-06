import json
import os
import struct
import sys
import tempfile
import unittest
from collections import OrderedDict
from pathlib import Path
from unittest.mock import Mock, patch

from sglang.srt.models.qwen4_ple_cache import BoundedByteLRU, FP8RowCacheReader
from sglang.srt.models.qwen4_ple_nvme import (
    IoUringPageRowReader,
    MMapRowReader,
    NVMePLEEmbedding,
    PLEManifest,
    RowLocation,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _write_safetensors(
    path: Path, tensors: dict[str, tuple[str, tuple[int, ...], bytes]]
):
    header = {}
    payload = bytearray()
    for name, (dtype, shape, data) in tensors.items():
        start = len(payload)
        payload.extend(data)
        header[name] = {
            "dtype": dtype,
            "shape": list(shape),
            "data_offsets": [start, len(payload)],
        }
    encoded = json.dumps(header, separators=(",", ":")).encode()
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload)


class TestQwen4PLENvmeManifest(CustomTestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.snapshot = Path(self.temp.name)
        prefix = "model.layers.2.ple.ple_embedding.ngram_embedding"
        self.names = [f"{prefix}.shard_{index}.weight" for index in range(2)]
        _write_safetensors(
            self.snapshot / "model-00001-of-00002.safetensors",
            {self.names[0]: ("F8_E4M3", (2, 4), bytes(range(8)))},
        )
        _write_safetensors(
            self.snapshot / "model-00002-of-00002.safetensors",
            {self.names[1]: ("F8_E4M3", (1, 4), bytes(range(8, 12)))},
        )
        index = {
            "metadata": {},
            "weight_map": {
                self.names[0]: "model-00001-of-00002.safetensors",
                self.names[1]: "model-00002-of-00002.safetensors",
            },
        }
        (self.snapshot / "model.safetensors.index.json").write_text(json.dumps(index))

    def tearDown(self):
        self.temp.cleanup()

    def test_maps_global_rows_to_exact_safetensors_ranges(self):
        manifest = PLEManifest.from_snapshot(self.snapshot, expected_shards=2)
        self.assertEqual(manifest.dtype, "F8_E4M3")
        self.assertEqual(manifest.embedding_dim, 4)
        self.assertEqual(manifest.total_rows, 3)
        self.assertEqual(manifest.row_bytes, 4)

        reader = MMapRowReader(manifest)
        self.addCleanup(reader.close)
        self.assertEqual(
            reader.read_rows([2, 0, 1]),
            [bytes(range(8, 12)), bytes(range(4)), bytes(range(4, 8))],
        )

    def test_rejects_noncontiguous_shards(self):
        index_path = self.snapshot / "model.safetensors.index.json"
        index = json.loads(index_path.read_text())
        second = index["weight_map"].pop(self.names[1])
        index["weight_map"][self.names[1].replace("shard_1", "shard_2")] = second
        index_path.write_text(json.dumps(index))
        with self.assertRaisesRegex(ValueError, "not contiguous"):
            PLEManifest.from_snapshot(self.snapshot)

    def test_enumerates_every_page_spanned_by_a_row(self):
        reader = object.__new__(IoUringPageRowReader)
        reader.page_size = 4096
        path = self.snapshot / "model-00001-of-00002.safetensors"
        self.assertEqual(
            reader._page_keys(RowLocation(path=path, offset=100, nbytes=9000)),
            ((path, 0), (path, 4096), (path, 8192)),
        )

    def test_hot_cache_assembles_rows_without_miss_planning(self):
        manifest = PLEManifest.from_snapshot(self.snapshot)
        reader = object.__new__(IoUringPageRowReader)
        reader.manifest = manifest
        reader.page_size = 5  # Some four-byte rows cross page boundaries.
        reader._cache = OrderedDict()
        reader._load_pages = Mock(side_effect=AssertionError("unexpected miss plan"))
        row_ids = [2, 0, 1, 2]
        for row_id in row_ids:
            for path, offset in reader._page_keys(manifest.locate(row_id)):
                contents = path.read_bytes()
                reader._cache[(path, offset)] = contents[offset : offset + 5]

        reference = MMapRowReader(manifest)
        self.addCleanup(reference.close)
        self.assertEqual(reader.read_rows(row_ids), reference.read_rows(row_ids))
        reader._load_pages.assert_not_called()

        # Long prefill chunks with repeated IDs should only slice each hot row
        # once, while preserving output order and first-use LRU behavior.
        reader._read_cached_rows = Mock(wraps=reader._read_cached_rows)
        repeated_ids = row_ids * 2048
        self.assertEqual(
            reader.read_rows(repeated_ids), reference.read_rows(repeated_ids)
        )
        self.assertEqual(reader._read_cached_rows.call_args.args[0], (2, 0, 1))
        reader._load_pages.assert_not_called()

    def test_cache_miss_keeps_batched_io_uring_fallback(self):
        manifest = PLEManifest.from_snapshot(self.snapshot)
        reader = object.__new__(IoUringPageRowReader)
        reader.manifest = manifest
        reader.page_size = 5
        reader.max_batch = 2
        reader.cache_pages = 100
        reader.cache_bytes = -1
        reader._disk_pages = 0
        reader._disk_bytes = 0
        reader._cache = OrderedDict()
        reader._nvtx_enabled = False
        reader._fd = lambda path: path
        reader._ring = Mock()
        reader._ring.read_pages.side_effect = lambda paths, offsets: [
            path.read_bytes()[offset : offset + 5]
            for path, offset in zip(paths, offsets)
        ]

        reference = MMapRowReader(manifest)
        self.addCleanup(reference.close)
        row_ids = [2, 0, 1, 2] * 2048
        self.assertEqual(reader.read_rows(row_ids), reference.read_rows(row_ids))
        self.assertTrue(reader._ring.read_pages.called)

    def test_byte_bounded_pages_match_legacy_reads_at_tiny_budgets(self):
        manifest = PLEManifest.from_snapshot(self.snapshot)
        reference = MMapRowReader(manifest)
        self.addCleanup(reference.close)
        for budget in (0, 64, 1024):
            with self.subTest(budget=budget):
                reader = object.__new__(IoUringPageRowReader)
                reader.manifest = manifest
                reader.page_size = 5
                reader.max_batch = 2
                reader.cache_pages = 0
                reader.cache_bytes = budget
                reader._disk_pages = reader._disk_bytes = 0
                reader._cache = BoundedByteLRU(
                    budget,
                    key_size=lambda key: sys.getsizeof(key) + sys.getsizeof(key[1]),
                )
                reader._nvtx_enabled = False
                reader._fd = lambda path: path
                reader._ring = Mock()
                reader._ring.read_pages.side_effect = lambda paths, offsets: [
                    path.read_bytes()[offset : offset + 5]
                    for path, offset in zip(paths, offsets)
                ]
                for ids in ([2, 0, 1, 2], [1, 0, 2], [2] * 8192):
                    self.assertEqual(reader.read_rows(ids), reference.read_rows(ids))
                    self.assertLessEqual(
                        reader.snapshot_stats()["accounted_bytes"], budget
                    )
                self.assertGreater(reader.snapshot_stats()["disk_bytes"], 0)

    def test_cache_mode_selection_and_invalid_combinations(self):
        manifest = PLEManifest.from_snapshot(self.snapshot)
        embedding = object.__new__(NVMePLEEmbedding)
        embedding.manifest = manifest
        reader_path = "sglang.srt.models.qwen4_ple_nvme.IoUringPageRowReader"

        base_env = {
            "SGLANG_QWEN4_PLE_NVME_BACKEND": "io_uring",
            "SGLANG_QWEN4_PLE_NVME_CACHE_MODE": "page",
            "SGLANG_QWEN4_PLE_NVME_CACHE_BYTES": "-1",
        }
        with patch.dict(os.environ, base_env), patch(reader_path) as reader_cls:
            embedding._create_reader()
            self.assertEqual(reader_cls.call_args.kwargs["cache_bytes"], -1)

        mmap_env = base_env | {"SGLANG_QWEN4_PLE_NVME_BACKEND": "mmap"}
        with (
            patch.dict(os.environ, mmap_env),
            patch("sglang.srt.models.qwen4_ple_nvme.MMapRowReader") as mmap_cls,
        ):
            self.assertIs(embedding._create_reader(), mmap_cls.return_value)

        unset_bytes_env = {
            "SGLANG_QWEN4_PLE_NVME_BACKEND": "mmap",
            "SGLANG_QWEN4_PLE_NVME_CACHE_MODE": "page",
        }
        with (
            patch.dict(os.environ, unset_bytes_env, clear=True),
            patch("sglang.srt.models.qwen4_ple_nvme.MMapRowReader") as mmap_cls,
        ):
            self.assertIs(embedding._create_reader(), mmap_cls.return_value)

        row_env = base_env | {
            "SGLANG_QWEN4_PLE_NVME_CACHE_MODE": "row",
            "SGLANG_QWEN4_PLE_NVME_CACHE_BYTES": "1024",
        }
        backing = Mock(manifest=manifest)
        with (
            patch.dict(os.environ, row_env),
            patch(reader_path, return_value=backing) as reader_cls,
        ):
            reader = embedding._create_reader()
            self.assertIsInstance(reader, FP8RowCacheReader)
            self.assertEqual(reader.snapshot_stats()["budget_bytes"], 1024)
            self.assertEqual(reader_cls.call_args.kwargs["cache_pages"], 0)

        invalid = (
            ({"SGLANG_QWEN4_PLE_NVME_CACHE_MODE": "row"}, "explicit CACHE_BYTES"),
            (
                {"SGLANG_QWEN4_PLE_NVME_CACHE_BYTES": "-2"},
                "must be -1.*nonnegative",
            ),
            ({"SGLANG_QWEN4_PLE_NVME_CACHE_MODE": "invalid"}, "cache mode"),
            (
                {
                    "SGLANG_QWEN4_PLE_NVME_BACKEND": "mmap",
                    "SGLANG_QWEN4_PLE_NVME_CACHE_BYTES": "0",
                },
                "requires the io_uring backend",
            ),
        )
        for overrides, message in invalid:
            with (
                self.subTest(overrides=overrides),
                patch.dict(os.environ, base_env | overrides),
                self.assertRaisesRegex(ValueError, message),
            ):
                embedding._create_reader()

    def test_malformed_cache_bytes_fail_before_backend_construction(self):
        manifest = PLEManifest.from_snapshot(self.snapshot)
        embedding = object.__new__(NVMePLEEmbedding)
        embedding.manifest = manifest
        reader_path = "sglang.srt.models.qwen4_ple_nvme.IoUringPageRowReader"
        mmap_path = "sglang.srt.models.qwen4_ple_nvme.MMapRowReader"
        configurations = (
            ("mmap", "page"),
            ("io_uring", "page"),
            ("io_uring", "row"),
        )

        for backend, mode in configurations:
            environment = {
                "SGLANG_QWEN4_PLE_NVME_BACKEND": backend,
                "SGLANG_QWEN4_PLE_NVME_CACHE_MODE": mode,
                "SGLANG_QWEN4_PLE_NVME_CACHE_BYTES": "512MiB",
            }
            with (
                self.subTest(backend=backend, mode=mode),
                patch.dict(os.environ, environment),
                patch(reader_path) as reader_cls,
                patch(mmap_path) as mmap_cls,
                self.assertRaisesRegex(ValueError, "CACHE_BYTES.*integer"),
            ):
                embedding._create_reader()
            reader_cls.assert_not_called()
            mmap_cls.assert_not_called()


if __name__ == "__main__":
    unittest.main()

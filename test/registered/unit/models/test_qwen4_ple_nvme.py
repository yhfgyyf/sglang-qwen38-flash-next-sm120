import json
import struct
import tempfile
import unittest
from collections import OrderedDict
from pathlib import Path
from unittest.mock import Mock

from sglang.srt.models.qwen4_ple_nvme import (
    IoUringPageRowReader,
    MMapRowReader,
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


if __name__ == "__main__":
    unittest.main()

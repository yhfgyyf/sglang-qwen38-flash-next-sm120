"""Read-only original-checkpoint byte oracle; no model execution or conversion."""
import argparse
import ctypes as C
import hashlib
import json
import os
import random

from sglang.srt.model_executor.qwen38_native import libraries, _check
from sglang.srt.models.qwen4_ple_nvme import PLEManifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("snapshot")
    args = parser.parse_args()
    manifest = PLEManifest.from_snapshot(args.snapshot, expected_shards=128)
    _, lib = libraries()
    n = len(manifest.shards)
    paths = (C.c_char_p * n)(*[os.fsencode(s.tensor.path) for s in manifest.shards])
    offsets = (C.c_uint64 * n)(*[s.tensor.offset for s in manifest.shards])
    counts = (C.c_uint64 * n)(*[s.row_end - s.row_start for s in manifest.shards])
    handle = lib.q38_ple_open(paths, offsets, counts, n, 160, 536870912, 512, 4096)
    if not handle:
        _check(-1, lib, "q38_ple_last_error")
    rng = random.Random(20261007)
    ids = []
    for shard in manifest.shards:
        ids.extend([shard.row_start, shard.row_end - 1])
        ids.extend(rng.sample(range(shard.row_start + 1, shard.row_end - 1), 64))
    rng.shuffle(ids)
    # >=8192 unique rows spans more than one fixed 4096-page scratch batch.
    descriptors = {}
    try:
        expected_rows = []
        for row in ids:
            location = manifest.locate(row)
            path = str(location.path)
            if path not in descriptors:
                descriptors[path] = os.open(path, os.O_RDONLY)
            payload = os.pread(descriptors[path], location.nbytes, location.offset)
            if len(payload) != 160:
                raise OSError("reference pread returned a short row")
            expected_rows.append(payload)
        # Exercise both diverse misses and a full 8K-token PLE gather of 131K
        # rows including duplicates, in original requested order.
        for count in (len(ids), 131072):
            requested = [ids[index % len(ids)] for index in range(count)]
            expected = b"".join(expected_rows[index % len(ids)] for index in range(count))
            native_ids = (C.c_int64 * count)(*requested)
            output = (C.c_uint8 * (160 * count))()
            _check(lib.q38_ple_read(handle, native_ids, count, output, len(output)), lib, "q38_ple_last_error")
            actual = bytes(output)
            if actual != expected:
                raise AssertionError("native PLE bytes differ from independent pread reference")
            print(json.dumps({"rows": count, "bytes": len(actual), "shards": 128,
                              "sha256": hashlib.sha256(actual).hexdigest(), "exact": True}), flush=True)
    finally:
        lib.q38_ple_close(handle)
        for fd in descriptors.values():
            os.close(fd)


if __name__ == "__main__":
    main()

"""NVMe-backed Qwen4 PLE embeddings.

Qwen4's PLE table is a large, row-sharded FP8 matrix. This module parses the
safetensors headers without loading the tensors, reads only selected rows, and
overlaps those reads with the decoder layer immediately before PLE.
"""

from __future__ import annotations

import ctypes
import errno
import json
import logging
import math
import mmap
import os
import re
import struct
import sys
import time
from collections import OrderedDict
from collections.abc import Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import nullcontext
from itertools import pairwise
from pathlib import Path
from typing import Any, Protocol

import msgspec
import torch
from torch import nn

from sglang.srt.environ import envs
from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph import (
    eager_on_graph,
    is_in_breakable_cuda_graph,
)
from sglang.srt.models.qwen4_ple_cache import BoundedByteLRU, FP8RowCacheReader

logger = logging.getLogger(__name__)

_PLE_SHARD_PATTERN = re.compile(
    r"^(?P<prefix>.+\.ngram_embedding)\.shard_(?P<index>\d+)\.weight$"
)
_DTYPE_BYTES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "I16": 2,
    "U16": 2,
    "F16": 2,
    "BF16": 2,
    "I32": 4,
    "U32": 4,
    "F32": 4,
    "I64": 8,
    "U64": 8,
    "F64": 8,
}
_MAX_HEADER_BYTES = 128 * 1024 * 1024


class TensorRecord(msgspec.Struct, frozen=True):
    path: Path
    name: str
    dtype: str
    shape: tuple[int, ...]
    offset: int
    nbytes: int

    @property
    def itemsize(self) -> int:
        return _DTYPE_BYTES[self.dtype]

    @property
    def row_bytes(self) -> int:
        if len(self.shape) != 2:
            raise ValueError(f"{self.name} is not a matrix: {self.shape}")
        return self.shape[1] * self.itemsize


class RowLocation(msgspec.Struct, frozen=True):
    path: Path
    offset: int
    nbytes: int


class PLEShard(msgspec.Struct, frozen=True):
    index: int
    row_start: int
    row_end: int
    tensor: TensorRecord


def _read_safetensors_header(path: Path) -> dict[str, TensorRecord]:
    with path.open("rb") as handle:
        length_bytes = handle.read(8)
        if len(length_bytes) != 8:
            raise ValueError(f"truncated safetensors length in {path}")
        (header_length,) = struct.unpack("<Q", length_bytes)
        if header_length <= 0 or header_length > _MAX_HEADER_BYTES:
            raise ValueError(
                f"invalid safetensors header length {header_length} in {path}"
            )
        header_bytes = handle.read(header_length)
        if len(header_bytes) != header_length:
            raise ValueError(f"truncated safetensors header in {path}")

    try:
        header = json.loads(header_bytes)
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid safetensors JSON in {path}") from error

    data_start = 8 + header_length
    records = {}
    for name, metadata in header.items():
        if name == "__metadata__":
            continue
        try:
            dtype = str(metadata["dtype"])
            shape = tuple(int(value) for value in metadata["shape"])
            relative_start, relative_end = (
                int(value) for value in metadata["data_offsets"]
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"invalid metadata for tensor {name!r} in {path}"
            ) from error
        if dtype not in _DTYPE_BYTES:
            raise ValueError(f"unsupported dtype {dtype!r} for {name!r}")
        if any(dimension < 0 for dimension in shape):
            raise ValueError(f"negative tensor shape for {name!r}: {shape}")
        if relative_start < 0 or relative_end < relative_start:
            raise ValueError(f"invalid data offsets for {name!r}")
        expected_bytes = math.prod(shape) * _DTYPE_BYTES[dtype]
        actual_bytes = relative_end - relative_start
        if actual_bytes != expected_bytes:
            raise ValueError(
                f"size mismatch for {name!r}: {actual_bytes} != {expected_bytes}"
            )
        records[name] = TensorRecord(
            path=path,
            name=name,
            dtype=dtype,
            shape=shape,
            offset=data_start + relative_start,
            nbytes=actual_bytes,
        )
    return records


class PLEManifest(msgspec.Struct, frozen=True):
    prefix: str
    dtype: str
    embedding_dim: int
    total_rows: int
    shard_size: int
    shards: tuple[PLEShard, ...]

    @classmethod
    def from_snapshot(
        cls, snapshot: str | Path, *, expected_shards: int | None = None
    ) -> PLEManifest:
        snapshot_path = Path(snapshot)
        index_path = snapshot_path / "model.safetensors.index.json"
        with index_path.open() as handle:
            weight_map = json.load(handle)["weight_map"]

        matched = []
        for name, filename in weight_map.items():
            match = _PLE_SHARD_PATTERN.match(name)
            if match:
                matched.append(
                    (name, match.group("prefix"), int(match.group("index")), filename)
                )
        if not matched:
            raise ValueError(f"no sharded PLE embedding found in {index_path}")

        prefixes = {prefix for _, prefix, _, _ in matched}
        if len(prefixes) != 1:
            raise ValueError(f"expected one PLE table, found {sorted(prefixes)}")
        prefix = prefixes.pop()
        matched.sort(key=lambda item: item[2])
        indices = [index for _, _, index, _ in matched]
        if indices != list(range(len(indices))):
            raise ValueError(f"PLE shard indices are not contiguous: {indices}")
        if expected_shards is not None and len(matched) != expected_shards:
            raise ValueError(
                f"expected {expected_shards} PLE shards, found {len(matched)}"
            )

        header_cache: dict[Path, dict[str, TensorRecord]] = {}
        records = []
        for name, _, shard_index, filename in matched:
            path = snapshot_path / filename
            header = header_cache.setdefault(path, _read_safetensors_header(path))
            try:
                record = header[name]
            except KeyError as error:
                raise ValueError(f"{name!r} is absent from {path}") from error
            if len(record.shape) != 2:
                raise ValueError(f"PLE shard {name!r} is not a matrix")
            records.append((shard_index, record))

        dtypes = {record.dtype for _, record in records}
        dimensions = {record.shape[1] for _, record in records}
        if len(dtypes) != 1 or len(dimensions) != 1:
            raise ValueError(
                f"inconsistent PLE shards: dtypes={dtypes}, dimensions={dimensions}"
            )
        shard_size = max(record.shape[0] for _, record in records)
        shards = tuple(
            PLEShard(
                index=shard_index,
                row_start=shard_index * shard_size,
                row_end=shard_index * shard_size + record.shape[0],
                tensor=record,
            )
            for shard_index, record in records
        )
        for previous, current in pairwise(shards):
            if previous.row_end != current.row_start:
                raise ValueError(
                    "only the final PLE shard may be short; "
                    f"shards {previous.index} and {current.index} are discontinuous"
                )
        return cls(
            prefix=prefix,
            dtype=records[0][1].dtype,
            embedding_dim=records[0][1].shape[1],
            total_rows=shards[-1].row_end,
            shard_size=shard_size,
            shards=shards,
        )

    @property
    def row_bytes(self) -> int:
        return self.shards[0].tensor.row_bytes

    def locate(self, row_id: int) -> RowLocation:
        if row_id < 0 or row_id >= self.total_rows:
            raise IndexError(f"PLE row {row_id} is outside [0, {self.total_rows})")
        shard = self.shards[row_id // self.shard_size]
        if row_id >= shard.row_end:
            raise IndexError(f"PLE row {row_id} is not materialized")
        local_row = row_id - shard.row_start
        return RowLocation(
            path=shard.tensor.path,
            offset=shard.tensor.offset + local_row * shard.tensor.row_bytes,
            nbytes=shard.tensor.row_bytes,
        )

    def summary(self) -> dict[str, int | str]:
        return {
            "prefix": self.prefix,
            "dtype": self.dtype,
            "embedding_dim": self.embedding_dim,
            "row_bytes": self.row_bytes,
            "total_rows": self.total_rows,
            "shards": len(self.shards),
            "tensor_bytes": sum(shard.tensor.nbytes for shard in self.shards),
            "files": len({shard.tensor.path for shard in self.shards}),
        }


class RowReader(Protocol):
    def read_rows(self, row_ids: Sequence[int]) -> list[bytes]: ...

    def close(self) -> None: ...


class MMapRowReader:
    """Portable correctness/debug backend backed by the page cache."""

    def __init__(self, manifest: PLEManifest) -> None:
        self.manifest = manifest
        self._files: dict[Path, Any] = {}
        self._maps: dict[Path, mmap.mmap] = {}

    def _mapping(self, path: Path) -> mmap.mmap:
        mapping = self._maps.get(path)
        if mapping is None:
            handle = path.open("rb")
            mapping = mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ)
            self._files[path] = handle
            self._maps[path] = mapping
        return mapping

    def read_rows(self, row_ids: Sequence[int]) -> list[bytes]:
        output = []
        for row_id in row_ids:
            location = self.manifest.locate(row_id)
            mapping = self._mapping(location.path)
            output.append(mapping[location.offset : location.offset + location.nbytes])
        return output

    def close(self) -> None:
        for mapping in self._maps.values():
            mapping.close()
        for handle in self._files.values():
            handle.close()
        self._maps.clear()
        self._files.clear()


class IoUringPageRowReader:
    """Read aligned pages through SGLang's persistent native io_uring reader."""

    def __init__(
        self,
        manifest: PLEManifest,
        *,
        queue_depth: int,
        max_batch: int,
        cache_pages: int,
        page_size: int = 4096,
        cache_bytes: int = -1,
    ) -> None:
        if not hasattr(os, "O_DIRECT"):
            raise OSError("O_DIRECT is unavailable on this platform")
        if cache_pages < 0:
            raise ValueError("cache_pages cannot be negative")
        if cache_bytes < -1:
            raise ValueError("cache_bytes must be -1 (legacy) or nonnegative")
        from sglang.srt.rust_extensions import load_rust_extension

        IoUringReader = load_rust_extension(
            "sglang.srt.rust_extensions._storage"
        ).IoUringReader

        self.manifest = manifest
        self.page_size = page_size
        self.max_batch = max_batch
        self.cache_pages = cache_pages
        self.cache_bytes = cache_bytes
        try:
            self._ring = IoUringReader(queue_depth, max_batch, page_size)
        except OSError as error:
            if error.errno == errno.EPERM:
                raise OSError(
                    errno.EPERM,
                    "io_uring is blocked; allow io_uring_setup, io_uring_enter, "
                    "and io_uring_register in the container seccomp profile",
                ) from error
            raise
        self._fds: dict[Path, int] = {}
        self._cache = (
            BoundedByteLRU(
                cache_bytes,
                key_size=lambda key: sys.getsizeof(key) + sys.getsizeof(key[1]),
            )
            if cache_bytes >= 0
            else OrderedDict()
        )
        self._disk_pages = 0
        self._disk_bytes = 0
        self._nvtx_enabled = os.getenv("SGLANG_QWEN4_PLE_NVME_NVTX", "0") == "1"

    def _fd(self, path: Path) -> int:
        descriptor = self._fds.get(path)
        if descriptor is None:
            descriptor = os.open(path, os.O_RDONLY | os.O_DIRECT)
            self._fds[path] = descriptor
        return descriptor

    def _page_keys(self, location: RowLocation) -> tuple[tuple[Path, int], ...]:
        first = location.offset // self.page_size * self.page_size
        last_byte = location.offset + location.nbytes - 1
        last = last_byte // self.page_size * self.page_size
        return tuple(
            (location.path, offset)
            for offset in range(first, last + self.page_size, self.page_size)
        )

    def _load_pages(
        self, keys: Sequence[tuple[Path, int]]
    ) -> dict[tuple[Path, int], bytes]:
        pages = {}
        misses = []
        for key in dict.fromkeys(keys):
            page = self._cache.get(key)
            if page is None:
                misses.append(key)
            else:
                self._cache.move_to_end(key)
                pages[key] = page
        for start in range(0, len(misses), self.max_batch):
            chunk = misses[start : start + self.max_batch]
            disk_read_context = (
                torch.cuda.nvtx.range("qwen4_ple_nvme.disk_read_pages")
                if self._nvtx_enabled
                else nullcontext()
            )
            with disk_read_context:
                loaded = self._ring.read_pages(
                    [self._fd(path) for path, _ in chunk],
                    [offset for _, offset in chunk],
                )
            self._disk_pages += len(chunk)
            self._disk_bytes += len(chunk) * self.page_size
            for key, page in zip(chunk, loaded, strict=True):
                pages[key] = page
                if self.cache_pages or self.cache_bytes >= 0:
                    self._cache[key] = page
                    if key in self._cache:
                        self._cache.move_to_end(key)
                    if self.cache_bytes < 0:
                        while len(self._cache) > self.cache_pages:
                            self._cache.popitem(last=False)
        return pages

    def snapshot_stats(self) -> dict[str, int]:
        cache = self._cache
        bounded = isinstance(cache, BoundedByteLRU)
        return {
            "entries": len(cache),
            "payload_bytes": (
                cache.payload_bytes if bounded else len(cache) * self.page_size
            ),
            "accounted_bytes": cache.accounted_bytes if bounded else -1,
            "budget_bytes": cache.max_bytes if bounded else -1,
            "disk_pages": self._disk_pages,
            "disk_bytes": self._disk_bytes,
        }

    def read_rows(self, row_ids: Sequence[int]) -> list[bytes]:
        # Repeated text can produce the same n-gram IDs thousands of times in
        # one prefill chunk. Build the unique IDs in C so hot-page lookup and
        # byte slicing run only once per distinct row. Keep near-unique inputs
        # on the original path to avoid an extra full-size dictionary.
        if self._cache and len(row_ids) >= 8192:
            sample = row_ids[:8192]
            if len(set(sample)) * 4 < len(sample) * 3:
                unique_ids = tuple(dict.fromkeys(row_ids))
                cached_rows = self._read_cached_rows(unique_ids)
                if cached_rows is not None:
                    by_id = dict(zip(unique_ids, cached_rows, strict=True))
                    return [by_id[row_id] for row_id in row_ids]
            else:
                cached_rows = self._read_cached_rows(row_ids)
        else:
            cached_rows = self._read_cached_rows(row_ids)
        if cached_rows is not None:
            return cached_rows

        locations = [self.manifest.locate(row_id) for row_id in row_ids]
        location_keys = [self._page_keys(location) for location in locations]
        pages = self._load_pages([key for keys in location_keys for key in keys])

        output = []
        for location, keys in zip(locations, location_keys, strict=True):
            remaining = location.nbytes
            cursor = location.offset
            parts = []
            for key in keys:
                page_offset = key[1]
                within_page = cursor - page_offset
                available = min(remaining, len(pages[key]) - within_page)
                if available <= 0:
                    raise OSError(
                        f"io_uring page does not cover {location.path}:{cursor}"
                    )
                parts.append(pages[key][within_page : within_page + available])
                cursor += available
                remaining -= available
            if remaining:
                raise OSError(f"incomplete row read: {remaining} bytes remain")
            output.append(b"".join(parts))
        return output

    def _read_cached_rows(self, row_ids: Sequence[int]) -> list[bytes] | None:
        """Avoid building a page plan and RowLocation objects on all-hit reads."""
        cache = self._cache
        manifest = self.manifest
        row_bytes = manifest.row_bytes
        page_size = self.page_size
        if not cache or row_bytes <= 0 or row_bytes > page_size:
            return None

        shards = manifest.shards
        shard_size = manifest.shard_size
        seen = set()
        output = []
        for row_id in row_ids:
            if row_id < 0 or row_id >= manifest.total_rows:
                manifest.locate(row_id)  # Preserve the public range error.
            shard = shards[row_id // shard_size]
            if row_id >= shard.row_end:
                manifest.locate(row_id)
            path = shard.tensor.path
            offset = shard.tensor.offset + (row_id - shard.row_start) * row_bytes
            page_start = offset // page_size * page_size
            key = (path, page_start)
            page = cache.get(key)
            if page is None:
                return None

            within_page = offset - page_start
            if within_page + row_bytes <= len(page):
                row = page[within_page : within_page + row_bytes]
                touched = (key,)
            else:
                first_length = page_size - within_page
                next_key = (path, page_start + page_size)
                next_page = cache.get(next_key)
                if (
                    first_length <= 0
                    or len(page) < page_size
                    or next_page is None
                    or len(next_page) < row_bytes - first_length
                ):
                    return None
                row = page[within_page:] + next_page[: row_bytes - first_length]
                touched = (key, next_key)

            for touched_key in touched:
                if touched_key not in seen:
                    cache.move_to_end(touched_key)
                    seen.add(touched_key)
            output.append(row)
        return output

    def close(self) -> None:
        for descriptor in self._fds.values():
            os.close(descriptor)
        self._fds.clear()
        self._cache.clear()


class PendingGather:
    """Mutable bridge updated by breakable CUDA-graph replay."""

    def __init__(
        self, future: Future[list[bytes]], input_shape: tuple[int, ...]
    ) -> None:
        self.future = future
        self.input_shape = input_shape


def _capture_start_gather(embedding: Any, input_ids: torch.Tensor) -> PendingGather:
    future: Future[list[bytes]] = Future()
    future.set_result([])
    return PendingGather(future, tuple(input_ids.shape))


def _capture_finish_gather(
    embedding: Any,
    pending: PendingGather,
    device: torch.device,
    out: torch.Tensor | None = None,
    stream: torch.cuda.Stream | None = None,
) -> torch.Tensor:
    expected_shape = (*pending.input_shape, embedding.embedding_dim)
    output = (
        out if out is not None else embedding.allocate_output(expected_shape, device)
    )
    output.zero_()
    return output


def _native_start_gather(embedding, input_ids, *, _native_output):
    from sglang.srt.model_executor.qwen38_native import NativePLEOperation

    if embedding._native is None:
        raise RuntimeError("native graph requires native PLE storage")
    return NativePLEOperation("issue", embedding._native, input_ids, input_ids.numel())


def _native_finish_gather(
    embedding, pending, device, out=None, stream=None, *, _native_output
):
    from sglang.srt.model_executor.qwen38_native import NativePLEOperation

    if embedding._native is None:
        raise RuntimeError("native graph requires native PLE storage")
    return NativePLEOperation(
        "collect", embedding._native, _native_output, math.prod(pending.input_shape)
    )


class NVMePLEEmbedding(nn.Module):
    """TP1 Qwen4 PLE embedding backed by sparse reads from a local snapshot."""

    def __init__(
        self,
        snapshot: str | Path,
        *,
        num_embeddings: int,
        embedding_dim: int,
        expected_shards: int | None = None,
    ) -> None:
        super().__init__()
        self.manifest = PLEManifest.from_snapshot(
            snapshot, expected_shards=expected_shards
        )
        if self.manifest.total_rows != num_embeddings:
            raise ValueError(
                "PLE snapshot row count does not match the model config: "
                f"{self.manifest.total_rows} != {num_embeddings}"
            )
        if self.manifest.embedding_dim != embedding_dim:
            raise ValueError(
                "PLE snapshot dimension does not match the model config: "
                f"{self.manifest.embedding_dim} != {embedding_dim}"
            )
        if self.manifest.dtype != "F8_E4M3":
            raise ValueError(
                f"the NVMe PLE path requires FP8 E4M3 rows, got {self.manifest.dtype}"
            )

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.tp_size = 1
        self.register_buffer(
            "weight_scale", torch.ones(1, dtype=torch.bfloat16), persistent=True
        )
        self._native = None
        if os.getenv("QWEN38_NATIVE_EXECUTOR", "0") == "1":
            from sglang.srt.model_executor.qwen38_native import NativePLEPipe

            if envs.SGLANG_QWEN4_PLE_NVME_CACHE_MODE.get() != "row":
                raise ValueError("native PLE currently requires explicit row cache mode")
            self._native = NativePLEPipe(
                self.manifest,
                cache_bytes=envs.SGLANG_QWEN4_PLE_NVME_CACHE_BYTES.get(),
                queue_depth=envs.SGLANG_QWEN4_PLE_NVME_QUEUE_DEPTH.get(),
                max_batch=envs.SGLANG_QWEN4_PLE_NVME_MAX_BATCH_PAGES.get(),
            )
            self._reader = self._native
            self._io_executor = None
        else:
            self._reader = self._create_reader()
            self._io_executor = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="ple-prefetch"
            )
        self._stage: torch.Tensor | None = None
        self._stage_event: torch.cuda.Event | None = None
        self._calls = 0
        self._rows = 0
        self._read_seconds = 0.0
        self._gathers = 0
        self._wait_seconds = 0.0
        self._nvtx_enabled = os.getenv("SGLANG_QWEN4_PLE_NVME_NVTX", "0") == "1"
        summary = self.manifest.summary()
        logger.info(
            "Qwen4 PLE NVMe table: %.2f GiB across %d files (%d rows)",
            int(summary["tensor_bytes"]) / (1024**3),
            summary["files"],
            self.manifest.total_rows,
        )

    def _create_reader(self) -> RowReader:
        backend = envs.SGLANG_QWEN4_PLE_NVME_BACKEND.get()
        cache_mode = envs.SGLANG_QWEN4_PLE_NVME_CACHE_MODE.get()
        cache_bytes_field = envs.SGLANG_QWEN4_PLE_NVME_CACHE_BYTES
        if cache_bytes_field.is_set():
            raw_cache_bytes = os.environ[cache_bytes_field.name]
            try:
                cache_bytes = cache_bytes_field.parse(raw_cache_bytes)
            except ValueError as error:
                raise ValueError(
                    f"PLE CACHE_BYTES must be an integer, got {raw_cache_bytes!r}"
                ) from error
        else:
            cache_bytes = cache_bytes_field.get()
        if cache_mode not in ("page", "row"):
            raise ValueError(f"unsupported PLE cache mode: {cache_mode!r}")
        if cache_bytes < -1:
            raise ValueError("PLE CACHE_BYTES must be -1 (legacy) or nonnegative")
        if backend == "mmap":
            if cache_mode != "page" or cache_bytes >= 0:
                raise ValueError("bounded PLE caching requires the io_uring backend")
            return MMapRowReader(self.manifest)
        if backend == "io_uring":
            if cache_mode == "row":
                if cache_bytes < 0:
                    raise ValueError(
                        "row caching requires an explicit CACHE_BYTES budget"
                    )
                backing = IoUringPageRowReader(
                    self.manifest,
                    queue_depth=envs.SGLANG_QWEN4_PLE_NVME_QUEUE_DEPTH.get(),
                    max_batch=envs.SGLANG_QWEN4_PLE_NVME_MAX_BATCH_PAGES.get(),
                    cache_pages=0,
                )
                return FP8RowCacheReader(backing, cache_bytes)
            return IoUringPageRowReader(
                self.manifest,
                queue_depth=envs.SGLANG_QWEN4_PLE_NVME_QUEUE_DEPTH.get(),
                max_batch=envs.SGLANG_QWEN4_PLE_NVME_MAX_BATCH_PAGES.get(),
                cache_pages=envs.SGLANG_QWEN4_PLE_NVME_CACHE_PAGES.get(),
                cache_bytes=cache_bytes,
            )
        raise ValueError(f"unsupported SGLANG_QWEN4_PLE_NVME_BACKEND={backend!r}")

    def _stage_buffer(self, nbytes: int) -> torch.Tensor:
        if self._stage_event is not None:
            self._stage_event.synchronize()
        if self._stage is None or self._stage.numel() < nbytes:
            self._stage = torch.empty(
                nbytes, dtype=torch.uint8, device="cpu", pin_memory=True
            )
        return self._stage[:nbytes]

    def allocate_output(
        self, shape: Sequence[int], device: torch.device
    ) -> torch.Tensor:
        return torch.empty(tuple(shape), dtype=torch.bfloat16, device=device)

    def gather(
        self, input_ids: torch.Tensor, out: torch.Tensor | None = None
    ) -> torch.Tensor:
        return self.finish_gather(
            self.start_gather(input_ids), input_ids.device, out=out
        )

    @eager_on_graph(
        True, capture_stub=_capture_start_gather, native_export=_native_start_gather
    )
    def start_gather(self, input_ids: torch.Tensor) -> PendingGather:
        if self._native is not None:
            pending = PendingGather(None, tuple(input_ids.shape))
            pending.native_ticket = self._native.issue(input_ids)
            return pending
        row_ids = (
            input_ids.detach().reshape(-1).to(device="cpu", dtype=torch.int64).tolist()
        )
        return PendingGather(
            self._io_executor.submit(self._timed_read_rows, row_ids),
            tuple(input_ids.shape),
        )

    def _timed_read_rows(self, row_ids: list[int]) -> list[bytes]:
        started = time.perf_counter()
        nvtx_context = (
            torch.cuda.nvtx.range("qwen4_ple_nvme.read_rows")
            if self._nvtx_enabled
            else nullcontext()
        )
        with nvtx_context:
            rows = self._reader.read_rows(row_ids)
        self._calls += 1
        self._rows += len(row_ids)
        self._read_seconds += time.perf_counter() - started
        return rows

    def _log_stats(self, input_shape: tuple[int, ...]) -> None:
        interval = envs.SGLANG_QWEN4_PLE_NVME_LOG_INTERVAL.get()
        if interval <= 0 or (input_shape[0] < 1024 and self._gathers % interval):
            return
        stats = {
            "gathers": self._gathers,
            "demand_calls": self._calls,
            "demand_rows": self._rows,
            "read_ms": round(self._read_seconds * 1000, 3),
            "wait_ms": round(self._wait_seconds * 1000, 3),
        }
        if hasattr(self._reader, "snapshot_stats"):
            stats["cache"] = self._reader.snapshot_stats()
        if isinstance(self._reader, FP8RowCacheReader):
            stats["backing"] = self._reader.backing.snapshot_stats()
        logger.info("Qwen4 PLE stats: %s", json.dumps(stats, sort_keys=True))

    @eager_on_graph(
        True, capture_stub=_capture_finish_gather, native_export=_native_finish_gather
    )
    def finish_gather(
        self,
        pending: PendingGather,
        device: torch.device,
        out: torch.Tensor | None = None,
        stream: torch.cuda.Stream | None = None,
    ) -> torch.Tensor:
        if self._native is not None:
            expected_shape = (*pending.input_shape, self.embedding_dim)
            output = out if out is not None else self.allocate_output(expected_shape, device)
            if tuple(output.shape) != expected_shape or output.device != device:
                raise ValueError("invalid native PLE output buffer")
            return self._native.collect(pending.native_ticket, output, stream)
        wait_context = (
            torch.cuda.nvtx.range("qwen4_ple_nvme.future_wait")
            if self._nvtx_enabled
            else nullcontext()
        )
        with wait_context:
            started = time.perf_counter()
            rows = pending.future.result()
            self._wait_seconds += time.perf_counter() - started
        self._gathers += 1
        self._log_stats(pending.input_shape)

        stage_context = (
            torch.cuda.nvtx.range("qwen4_ple_nvme.stage_and_copy")
            if self._nvtx_enabled
            else nullcontext()
        )
        with stage_context:
            expected_shape = (*pending.input_shape, self.embedding_dim)
            output = (
                out if out is not None else self.allocate_output(expected_shape, device)
            )
            if (
                tuple(output.shape) != expected_shape
                or output.dtype != torch.bfloat16
                or output.device != device
            ):
                raise ValueError("invalid NVMe PLE output buffer")

            raw = b"".join(rows)
            expected_bytes = math.prod(pending.input_shape) * self.embedding_dim
            if len(raw) != expected_bytes:
                raise OSError(
                    f"NVMe PLE read returned {len(raw)} bytes; expected {expected_bytes}"
                )
            if raw:
                stage = self._stage_buffer(len(raw))
                ctypes.memmove(stage.data_ptr(), raw, len(raw))
                stream_context = (
                    torch.cuda.stream(stream) if stream is not None else nullcontext()
                )
                with stream_context:
                    device_bytes = stage.to(device=device, non_blocking=True)
                    decoded = device_bytes.view(torch.float8_e4m3fn).to(torch.bfloat16)
                    output.copy_(decoded.view(expected_shape))
                    if self._stage_event is None:
                        self._stage_event = torch.cuda.Event()
                    self._stage_event.record()
                if is_in_breakable_cuda_graph():
                    self._stage_event.synchronize()
            return output

    def reduce(self, output: torch.Tensor) -> torch.Tensor:
        return output

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.gather(input_ids)

    def close(self) -> None:
        if self._io_executor is not None:
            self._io_executor.shutdown()
        self._reader.close()

    def extra_repr(self) -> str:
        return (
            f"num_embeddings={self.num_embeddings}, "
            f"embedding_dim={self.embedding_dim}, backend={type(self._reader).__name__}"
        )


def is_nvme_ple_embedding(module: Any) -> bool:
    return isinstance(module, NVMePLEEmbedding)

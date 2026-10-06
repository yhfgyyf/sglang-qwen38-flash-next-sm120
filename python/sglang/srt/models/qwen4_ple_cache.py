"""CPU byte-bounded row caching for NVMe-backed Qwen4 PLE tables.

The accounting here intentionally includes owned Python key/value objects and
the live ``OrderedDict`` storage, not just the FP8 payload. It still is not an
RSS guarantee: allocator arenas and fragmentation, interpreter-wide objects,
and transient in-flight I/O/output buffers are outside the cache's ownership.

This module deliberately does not import :mod:`qwen4_ple_nvme`; the reader is
wrapped by interface so it can be constructed without a circular import.
"""

from __future__ import annotations

import sys
from collections import OrderedDict
from collections.abc import Callable, Sequence
from typing import Generic, Protocol, TypeVar

_Key = TypeVar("_Key")
_EMPTY_ORDERED_DICT_BYTES = sys.getsizeof(OrderedDict())
_SINGLE_ENTRY_CONTAINER_BYTES = (
    sys.getsizeof(OrderedDict(((None, b""),))) - _EMPTY_ORDERED_DICT_BYTES
)
_MISSING = object()


class _Manifest(Protocol):
    row_bytes: int
    total_rows: int


class _BackingReader(Protocol):
    manifest: _Manifest

    def read_rows(self, row_ids: Sequence[int]) -> list[bytes]: ...

    def close(self) -> None: ...


class BoundedByteLRU(OrderedDict[_Key, bytes], Generic[_Key]):
    """An ``OrderedDict`` byte cache with ownership-based accounting.

    ``key_size`` must count only memory owned by a key. For row integer keys,
    use ``sys.getsizeof``. A page key ``(path, offset)`` should count the tuple
    and offset integer but not the shared ``Path`` object.

    Values must be immutable ``bytes``. ``accounted_bytes`` includes their
    Python object sizes, key sizes, and the current container overhead relative
    to an empty ``OrderedDict``. An entry that cannot fit by itself is skipped.
    """

    def __init__(self, max_bytes: int, *, key_size: Callable[[_Key], int]) -> None:
        if max_bytes < 0:
            raise ValueError("max_bytes cannot be negative")
        super().__init__()
        self.max_bytes = max_bytes
        self._key_size = key_size
        # Some Python builds give OrderedDict subclasses a different empty size.
        self._empty_container_bytes = sys.getsizeof(self)
        self._owned_bytes = 0
        self._payload_bytes = 0

    @property
    def payload_bytes(self) -> int:
        """Number of FP8/page payload bytes retained by the cache."""

        return self._payload_bytes

    @property
    def accounted_bytes(self) -> int:
        """Owned key/value bytes plus live container overhead."""

        container_bytes = max(0, sys.getsizeof(self) - self._empty_container_bytes)
        return self._owned_bytes + container_bytes

    def __setitem__(self, key: _Key, value: bytes) -> None:
        if not isinstance(value, bytes):
            raise TypeError("BoundedByteLRU values must be bytes")
        key_bytes = self._measure_key(key)
        value_bytes = sys.getsizeof(value)

        # A skipped oversized insertion must not evict useful existing entries.
        if key_bytes + value_bytes + _SINGLE_ENTRY_CONTAINER_BYTES > self.max_bytes:
            if key in self:
                self._delete(key)
            return

        previous = OrderedDict.get(self, key, _MISSING)
        is_new_key = previous is _MISSING
        if previous is not _MISSING:
            self._owned_bytes -= key_bytes + sys.getsizeof(previous)
            self._payload_bytes -= len(previous)

        OrderedDict.__setitem__(self, key, value)
        self._owned_bytes += key_bytes + value_bytes
        self._payload_bytes += len(value)

        while self and self.accounted_bytes > self.max_bytes:
            self.popitem(last=False)

        # Deletions can retain a large hash table. If that forced the newest
        # entry out, retry once after the empty container has been compacted.
        if is_new_key and not self:
            OrderedDict.__setitem__(self, key, value)
            self._owned_bytes = key_bytes + value_bytes
            self._payload_bytes = len(value)
            if self.accounted_bytes > self.max_bytes:
                self.popitem(last=False)

    def popitem(self, last: bool = True) -> tuple[_Key, bytes]:
        key, value = OrderedDict.popitem(self, last=last)
        self._owned_bytes -= self._measure_key(key) + sys.getsizeof(value)
        self._payload_bytes -= len(value)
        if not self:
            OrderedDict.clear(self)
        return key, value

    def clear(self) -> None:
        OrderedDict.clear(self)
        self._owned_bytes = 0
        self._payload_bytes = 0

    def _delete(self, key: _Key) -> None:
        value = OrderedDict.__getitem__(self, key)
        key_bytes = self._measure_key(key)
        OrderedDict.__delitem__(self, key)
        self._owned_bytes -= key_bytes + sys.getsizeof(value)
        self._payload_bytes -= len(value)
        if not self:
            OrderedDict.clear(self)

    def _measure_key(self, key: _Key) -> int:
        size = self._key_size(key)
        if not isinstance(size, int) or size < 0:
            raise ValueError("key_size must return a nonnegative integer")
        return size


class FP8RowCacheReader:
    """Cache exact FP8 rows returned by a cacheless backing reader.

    Hit and miss counters are per-call unique row outcomes: each first
    occurrence contributes once, while ``requested_rows`` includes duplicates.
    """

    def __init__(self, backing_reader: _BackingReader, cache_bytes: int) -> None:
        self.backing = backing_reader
        self.manifest = backing_reader.manifest
        self._row_bytes = self.manifest.row_bytes
        self._total_rows = self.manifest.total_rows
        if self._row_bytes <= 0:
            raise ValueError("manifest.row_bytes must be positive")
        if self._total_rows < 0:
            raise ValueError("manifest.total_rows cannot be negative")

        self._cache = BoundedByteLRU[int](cache_bytes, key_size=sys.getsizeof)
        self._calls = 0
        self._requested_rows = 0
        self._unique_hits = 0
        self._unique_misses = 0

    def read_rows(self, row_ids: Sequence[int]) -> list[bytes]:
        requested = tuple(row_ids)
        unique_ids = tuple(dict.fromkeys(requested))
        self._validate_ids(unique_ids)

        cached: dict[int, bytes] = {}
        misses = []
        for row_id in unique_ids:
            row = self._cache.get(row_id)
            if row is None:
                misses.append(row_id)
            else:
                self._cache.move_to_end(row_id)
                cached[row_id] = row

        self._calls += 1
        self._requested_rows += len(requested)
        self._unique_hits += len(cached)
        self._unique_misses += len(misses)

        if misses:
            loaded = self.backing.read_rows(misses)
            loaded_by_id = self._validated_rows(misses, loaded)
            for row_id, row in loaded_by_id.items():
                self._cache[row_id] = row
                if row_id in self._cache:
                    self._cache.move_to_end(row_id)
            # Return newly loaded rows even when the configured budget cannot
            # retain them.
            cached.update(loaded_by_id)

        return [cached[row_id] for row_id in requested]

    def snapshot_stats(self) -> dict[str, int]:
        return {
            "calls": self._calls,
            "requested_rows": self._requested_rows,
            "unique_hits": self._unique_hits,
            "unique_misses": self._unique_misses,
            "entries": len(self._cache),
            "payload_bytes": self._cache.payload_bytes,
            "accounted_bytes": self._cache.accounted_bytes,
            "budget_bytes": self._cache.max_bytes,
        }

    def close(self) -> None:
        try:
            self.backing.close()
        finally:
            self._cache.clear()

    def _validated_rows(
        self, row_ids: Sequence[int], rows: Sequence[bytes]
    ) -> dict[int, bytes]:
        if len(row_ids) != len(rows):
            raise ValueError(
                f"row ID/payload length mismatch: {len(row_ids)} != {len(rows)}"
            )
        loaded_by_id = {}
        for row_id, row in zip(row_ids, rows, strict=True):
            if not isinstance(row, bytes):
                raise TypeError(f"PLE row {row_id} is not immutable bytes")
            if len(row) != self._row_bytes:
                raise ValueError(
                    f"PLE row {row_id} has {len(row)} bytes; expected {self._row_bytes}"
                )
            loaded_by_id[row_id] = row
        return loaded_by_id

    def _validate_ids(self, row_ids: Sequence[int]) -> None:
        for row_id in row_ids:
            if row_id < 0 or row_id >= self._total_rows:
                raise IndexError(f"PLE row {row_id} is outside [0, {self._total_rows})")

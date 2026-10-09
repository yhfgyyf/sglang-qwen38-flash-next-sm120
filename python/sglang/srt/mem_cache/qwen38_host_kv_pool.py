# Copyright 2026 Qwen38 native contributors. SPDX-License-Identifier: Apache-2.0
"""Opt-in active-host FP8 KV for one Qwen3.8/SM120/NEXTN profile.

Only ordinary FP8 E4M3 K/V lives here. Logical token slots and compressed QSA keys
keep their original GPU addressing; this is not a prefix-cache/HiCache backend.
"""

from __future__ import annotations

import logging
import os
import weakref

import torch
import triton
import triton.language as tl

from sglang.srt.mem_cache.memory_pool import KVCache, UnquantizedKVCacheMethod
from sglang.srt.model_executor.qwen38_host_kv import HostKVArena, required_bytes

logger = logging.getLogger(__name__)
_ARENAS = weakref.WeakSet()
_TOTAL_FULL_LAYERS = 13  # 12 target QSA layers plus the original one-layer MTP.
_BYTES_PER_LAYER_TOKEN = 2 * 2 * 256


def host_kv_budget_bytes() -> int:
    budget = int(os.environ.get("QWEN38_HOST_KV_BYTES", "0"))
    if budget < 0:
        raise ValueError("QWEN38_HOST_KV_BYTES must be nonnegative")
    return budget


def host_kv_dedup_enabled() -> bool:
    value = os.environ.get("QWEN38_HOST_KV_DEDUP", "0")
    if value not in ("0", "1"):
        raise ValueError("QWEN38_HOST_KV_DEDUP must be 0 or 1")
    return value == "1"


def validate_host_kv_profile(kvc) -> bool:
    """Fail before sizing/allocation if an unsupported profile opts in."""
    if not host_kv_budget_bytes():
        return False
    args = kvc.server_args
    config = kvc.model_config.hf_text_config
    supported = (
        os.environ.get("QWEN38_NATIVE_EXECUTOR", "0") == "1"
        and getattr(config, "model_type", None) == "qwen4_exp_text"
        # MTP constructs a private one-layer model copy; depending on this
        # call site ModelConfig still carries the original 48-layer text config.
        and getattr(config, "num_hidden_layers", None)
        in ((1, 48) if kvc.is_draft_worker else (48,))
        and getattr(config, "num_key_value_heads", None) == 2
        and getattr(config, "head_dim", None) == 256
        and getattr(config, "indexer_compress_ratio", None) == 4
        and getattr(config, "indexer_kv_heads", None) == 1
        and getattr(config, "indexer_head_dim", None) == 128
        and kvc.kv_cache_dtype == torch.float8_e4m3fn
        and not kvc.use_mla_backend
        and getattr(args, "tp_size", 1) == 1
        and getattr(args, "pp_size", 1) == 1
        and getattr(args, "dp_size", 1) == 1
        # The server normalizes the CLI NEXTN alias to EAGLE before this point.
        and getattr(args, "speculative_algorithm", None) == "EAGLE"
        and getattr(args, "speculative_eagle_topk", None) == 1
        and getattr(args, "speculative_num_draft_tokens", None) == 4
        and getattr(args, "speculative_num_steps", None) == 3
        and getattr(args, "max_total_tokens", None) is not None
        and kvc.page_size == 64
        and not getattr(args, "enable_hierarchical_cache", False)
        and not getattr(args, "enable_hicache_storage", False)
        and not getattr(args, "enable_memory_saver", False)
        and not getattr(args, "enable_kv_cache_pool", False)
        and getattr(args, "disaggregation_mode", "null") == "null"
        and not getattr(kvc, "post_capture_kv_active", False)
    )
    if not supported:
        raise ValueError(
            "native host KV requires the explicit Qwen3.8 SM120 TP1/PP1/DP1 "
            "FP8-E4M3-KV NEXTN 3/1/4 profile, bounded --max-total-tokens, and no "
            "HiCache, disaggregation, memory-saver or post-capture resizing"
        )
    return True


def host_kv_token_capacity(page_size: int) -> int:
    """Aggregate target+draft budget includes the allocator's padding page."""
    capacity = (
        host_kv_budget_bytes() // (_TOTAL_FULL_LAYERS * _BYTES_PER_LAYER_TOKEN)
        - page_size
    )
    if capacity <= 0:
        raise MemoryError("host KV budget cannot hold the reserved padding page")
    return capacity // page_size * page_size


def retain_active_host_arenas():
    """Graph owners retain these leases until destruction and replay drain."""
    return tuple(arena.retain_for_graph() for arena in tuple(_ARENAS))


def check_active_host_arenas():
    """Cold-boundary deferred bounds validation; never a per-token GPU sync."""
    for arena in tuple(_ARENAS):
        arena.check_errors()


def _check_host_headroom(nbytes: int) -> None:
    # MemAvailable includes reclaimable file cache; MemFree alone would reject
    # a machine whose model mmap has populated the page cache.
    with open("/proc/meminfo") as handle:
        entries = dict(line.split(":", 1) for line in handle)
    available = int(entries["MemAvailable"].split()[0]) * 1024
    reserve = 8 * 1024**3
    if nbytes > available - reserve:
        raise MemoryError(
            f"host KV needs {nbytes} pinned bytes, MemAvailable={available}; "
            f"preserving {reserve} bytes for the host and PLE"
        )


class Qwen38HostKVPool(KVCache):
    """Full-attention sub-pool, indexed by dense local attention-layer ID."""

    native_host_kv = True

    def __init__(
        self,
        size,
        page_size,
        dtype,
        head_num,
        head_dim,
        layer_num,
        device,
        enable_memory_saver=False,
        enable_kv_cache_copy=False,
        quant_method=None,
        post_capture_active=False,
        **kwargs,
    ):
        if (
            dtype != torch.float8_e4m3fn
            or (head_num, head_dim) != (2, 256)
            or layer_num not in (1, 12)
            or page_size != 64
            or enable_memory_saver
            or quant_method is not None
            or post_capture_active
            or kwargs
        ):
            raise ValueError("unsupported native host-KV geometry or pool option")
        super().__init__(size, page_size, dtype, layer_num, device, False)
        self.head_num = head_num
        self.head_dim = self.v_head_dim = head_dim
        self.quant_method = UnquantizedKVCacheMethod()
        self.use_hnd = False
        self.kv_cache_layout = "nhd"
        self.post_capture_active = False
        self.post_capture_backed_bytes = 0
        self._capture_lease = None
        self.arena = None
        self.host_bytes = required_bytes(
            layer_num, size + page_size, head_num, head_dim, dtype=dtype
        )
        budget = host_kv_budget_bytes() * layer_num // _TOTAL_FULL_LAYERS
        if self.host_bytes > budget:
            raise MemoryError("target/draft host KV exceeds the aggregate byte budget")
        _check_host_headroom(self.host_bytes)
        self.arena = HostKVArena(
            layer_num,
            size + page_size,
            byte_budget=budget,
            heads=head_num,
            head_dim=head_dim,
            dtype=dtype,
            device=device,
            label="draft" if layer_num == 1 else "target",
            selected_dedup=host_kv_dedup_enabled(),
        )
        self._capture_lease = self.arena.retain_for_graph()
        _ARENAS.add(self.arena)
        logger.info(
            "Qwen38 native host KV allocated: layers=%d slots=%d FP8_E4M3_host_GiB=%.3f "
            "GPU_aux_bytes=%d aggregate_budget_GiB=%.3f selected_dedup=%s",
            layer_num,
            size + page_size,
            self.host_bytes / 1024**3,
            self.arena.gpu_bytes,
            host_kv_budget_bytes() / 1024**3,
            self.arena.selected_dedup,
        )

    @property
    def is_quantized_kv_cache(self):
        return False

    def get_kv_size_bytes(self):
        # This interface prices device payloads; host_bytes is reported above.
        return (0, 0)

    def get_kv_buffer_shape(self):
        shape = torch.Size((self.size + self.page_size, self.head_num, self.head_dim))
        return shape, shape

    def get_key_buffer(self, layer_id, *args):
        raise RuntimeError("host KV requires explicit gather; no GPU full-K buffer")

    def get_value_buffer(self, layer_id, *args):
        raise RuntimeError("host KV requires explicit gather; no GPU full-V buffer")

    def get_kv_buffer(self, layer_id):
        raise RuntimeError("host KV requires explicit gather; no GPU full-KV buffer")

    def get_contiguous_buf_infos(self):
        raise RuntimeError("native active-host KV does not support KV transfer/HiCache")

    def set_kv_buffer(
        self,
        layer,
        loc,
        cache_k,
        cache_v,
        k_scale=None,
        v_scale=None,
        layer_id_override=None,
        dcp_kv_mask=None,
    ):
        if dcp_kv_mask is not None:
            raise ValueError("native FP8 host KV does not support DCP")
        layer_id = layer.layer_id if layer_id_override is None else layer_id_override
        if cache_k.dtype != self.dtype:
            cache_k = (cache_k if k_scale is None else cache_k / k_scale).to(self.dtype)
        if cache_v.dtype != self.dtype:
            cache_v = (cache_v if v_scale is None else cache_v / v_scale).to(self.dtype)
        self.arena.scatter(layer_id, loc, cache_k, cache_v)

    def gather(self, layer_id, ids, out_k, out_v):
        return self.arena.gather(layer_id, ids, out_k, out_v)

    def gather_selected(self, layer_id, ids, out_k, out_v):
        return self.arena.gather_selected(layer_id, ids, out_k, out_v)

    def move_kv_cache(self, tgt_loc, src_loc):
        if tgt_loc.shape != src_loc.shape:
            raise ValueError("host KV relocation source/target shapes differ")
        count = src_loc.numel()
        if not count:
            return
        shape = (count, self.head_num, self.head_dim)
        k = torch.empty(shape, dtype=self.dtype, device=self.device)
        v = torch.empty_like(k)
        for layer in range(self.layer_num):
            # Stage all sources before writes, so overlapping moves are safe.
            self.arena.gather(layer, src_loc, k, v)
            self.arena.scatter(layer, tgt_loc, k, v)

    def close(self):
        if self.arena is None or self.arena.closed:
            return
        if self.arena.graph_references > 1:
            raise RuntimeError(
                "destroy native graph plans before closing the host KV pool"
            )
        self._capture_lease.close()
        self._capture_lease = None
        self.arena.close()

    def __del__(self):
        # Plan leases retain the arena even if the model pool is destroyed first.
        lease = getattr(self, "_capture_lease", None)
        if lease is not None:
            lease.close()


@triton.jit
def _selected_host_slots(
    req_to_token,
    req_ids,
    positions,
    lengths,
    output,
    REQ_STRIDE: tl.constexpr,
    IDX_STRIDE: tl.constexpr,
    TOPK: tl.constexpr,
    STRIDE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    length = tl.load(lengths + row)
    req = tl.load(req_ids + row)
    position = tl.load(positions + row * IDX_STRIDE + col, col < TOPK, other=-1)
    valid = (col < TOPK) & (position >= 0) & (position < length)
    rank = tl.cumsum(valid.to(tl.int32)) - 1
    count = tl.sum(valid.to(tl.int32))
    slot = tl.load(req_to_token + req * REQ_STRIDE + position, valid, other=-1)
    # Valid positions are compacted in source order. Trailing padding writes
    # are disjoint from compacted destinations, including non-prefix masks.
    tl.store(output + row * STRIDE + rank, slot, valid)
    tl.store(output + row * STRIDE + col, -1, (col >= count) & (col < STRIDE))


def selected_host_slots(req_to_token, req_ids, topk, lengths, stride, output):
    rows, width = topk.shape
    if stride < width or output.numel() < rows * stride:
        raise ValueError("host selected-slot scratch is too small")
    _selected_host_slots[(rows,)](
        req_to_token,
        req_ids,
        topk,
        lengths,
        output,
        req_to_token.stride(0),
        topk.stride(0),
        width,
        stride,
        triton.next_power_of_2(stride),
        num_warps=8,
    )
    return output[: rows * stride]

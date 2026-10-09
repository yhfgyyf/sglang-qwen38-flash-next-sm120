"""Narrow contracts for the opt-in Qwen3.8 exact-context decode tail.

This module is deliberately CPU-only. The route decision uses scheduler-owned
request lengths and never observes a device tensor, so enabling it cannot add a
per-token synchronization to decode. Exact-tail admission is supported by the
Python HTTP tokenizer path only: Rust ingress receives the shared conservative
EAGLE reservation and cannot resolve per-request normalized sampling parameters.
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from typing import Any

CONTEXT_TAIL_LENGTH = 262_144
CONTEXT_TAIL_ENV = "QWEN38_CONTEXT_TAIL"
CONTEXT_TAIL_TRACE_ENV = "QWEN38_CONTEXT_TAIL_TRACE"
CONTEXT_TAIL_TRACE_WINDOW = 128
_NATIVE_ENV = "QWEN38_NATIVE_EXECUTOR"


def context_tail_enabled() -> bool:
    """Return whether the exact-context tail was explicitly enabled."""

    return os.environ.get(CONTEXT_TAIL_ENV, "0") == "1"


def context_tail_trace_enabled() -> bool:
    """Return whether opt-in near-boundary device-position tracing is enabled."""

    return context_tail_enabled() and os.environ.get(CONTEXT_TAIL_TRACE_ENV, "0") == "1"


def context_tail_trace_near_bound(
    kv_committed_lens: Iterable[int], context_len: int
) -> bool:
    """Gate trace synchronization to requests near the logical context bound."""

    lengths = tuple(int(length) for length in kv_committed_lens)
    return bool(lengths) and max(lengths) >= context_len - CONTEXT_TAIL_TRACE_WINDOW


def _field(obj: Any, name: str, default: Any = None) -> Any:
    return getattr(obj, name, default)


def validate_context_tail_server_args(server_args: Any) -> bool:
    """Fail closed unless server_args is the measured 3/1/4 native profile.

    Returning False means the opt-in is disabled and all existing limits stay
    intact. The shared Python/Rust ingress reservation remains conservative;
    Python request validation selectively waives it for normalized greedy
    requests after this profile has been validated.
    """

    if not context_tail_enabled():
        return False

    algorithm = str(_field(server_args, "speculative_algorithm", "")).upper()
    kv_cache_dtype = str(_field(server_args, "kv_cache_dtype", "")).lower()
    checks = (
        (os.environ.get(_NATIVE_ENV, "0") == "1", "native executor"),
        (algorithm in {"EAGLE", "NEXTN"}, "EAGLE/NEXTN"),
        (_field(server_args, "speculative_num_steps") == 3, "3 draft steps"),
        (
            _field(server_args, "speculative_eagle_topk") == 1,
            "EAGLE top-k 1",
        ),
        (
            _field(server_args, "speculative_num_draft_tokens") == 4,
            "4 verify tokens",
        ),
        (
            not bool(_field(server_args, "speculative_adaptive", False)),
            "no adaptive speculation",
        ),
        (
            not bool(
                _field(server_args, "speculative_use_rejection_sampling", False)
            ),
            "no rejection sampling",
        ),
        (
            not bool(_field(server_args, "enable_multi_layer_eagle", False)),
            "single-layer EAGLE",
        ),
        (_field(server_args, "tp_size", 1) == 1, "TP=1"),
        (_field(server_args, "pp_size", 1) == 1, "PP=1"),
        (_field(server_args, "dp_size", 1) == 1, "DP=1"),
        (
            not bool(_field(server_args, "enable_dp_attention", False)),
            "no DP attention",
        ),
        (
            not bool(_field(server_args, "enable_two_batch_overlap", False)),
            "no two-batch overlap",
        ),
        (
            _field(server_args, "context_length") == CONTEXT_TAIL_LENGTH,
            f"context length {CONTEXT_TAIL_LENGTH}",
        ),
        (
            kv_cache_dtype in {"fp8_e4m3", "fp8_e4m3fn"},
            "ordinary KV FP8 E4M3",
        ),
        (_field(server_args, "page_size") == 64, "page size 64"),
        (
            str(_field(server_args, "quantization", "")).lower() == "modelopt_fp4",
            "ModelOpt FP4 weights",
        ),
    )
    failed = [description for ok, description in checks if not ok]
    if failed:
        raise ValueError(
            f"{CONTEXT_TAIL_ENV}=1 only supports the Qwen3.8 native "
            f"NEXTN 3/1/4 profile; invalid: {', '.join(failed)}"
        )
    return True


def validate_context_tail_model_config(model_config: Any) -> bool:
    """Validate the target model identity after its config has been loaded."""

    if not context_tail_enabled():
        return False
    if _field(model_config, "context_len") != CONTEXT_TAIL_LENGTH:
        raise ValueError(
            f"{CONTEXT_TAIL_ENV}=1 requires target context_len={CONTEXT_TAIL_LENGTH}"
        )
    text_config = _field(model_config, "hf_text_config")
    model_type = _field(text_config, "model_type")
    if model_type != "qwen4_exp_text":
        raise ValueError(
            f"{CONTEXT_TAIL_ENV}=1 requires the Qwen3.8 qwen4_exp_text model, "
            f"got {model_type!r}"
        )
    return True


def context_tail_request_is_greedy(sampling_params: Any) -> bool:
    """The admitted exact-tail lane is intentionally limited to greedy decode."""

    return _field(sampling_params, "top_k") == 1


def select_context_tail_verify_width(
    kv_committed_lens: Iterable[int],
    context_len: int,
    enable_overlap: bool,
    normal_width: int,
) -> int:
    """Select normal verify width or a one-token verify for this forward.

    kv_committed_len may lag the device-visible length by one normal verify
    during scheduler overlap. Conservatively use U = K + normal_width in that
    mode. A normal width is legal only when U + normal_width <= C - 1.
    """

    lengths = tuple(int(length) for length in kv_committed_lens)
    if not lengths:
        return normal_width
    if normal_width <= 1:
        return normal_width
    upper_committed = max(lengths) + (normal_width if enable_overlap else 0)
    if upper_committed + normal_width <= context_len - 1:
        return normal_width
    return 1

"""Opt-in eager-prefill assertions; never use these runs for speed evidence.

Loaded through SGLang's existing forward-hook interface after graph capture.
The assertions enqueue on the current CUDA stream without a host readback.
They do not sanitize, clamp, or otherwise modify model values.
"""

import hashlib
import itertools
import os
from pathlib import Path

import torch


_factory_ids = itertools.count()


def _tensors(value):
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _tensors(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from _tensors(item)
    elif type(value).__name__ == "LogitsProcessorOutput":
        yield from _tensors(vars(value))


def make_finite_hook(config):
    min_rows = int(config.get("min_rows", 8192))
    samples = config.get("sample_dir")
    if samples:
        samples = Path(samples) / f"pid{os.getpid()}-factory{next(_factory_ids)}"
        samples.mkdir(parents=True, exist_ok=False)
    layer_calls = {}

    def sample(tensor, rows):
        return {
            "value": tensor.index_select(
                0, torch.tensor(rows, dtype=torch.long, device=tensor.device)
            ).cpu(),
            "shape": tuple(tensor.shape),
            "rows": rows,
        }

    def full_digest(tensor):
        data = tensor.detach().cpu().contiguous().view(torch.uint8).numpy()
        return hashlib.sha256(memoryview(data)).hexdigest()

    def check(module, inputs, output):
        name = type(module).__name__
        layer = getattr(module, "layer_id", "unknown")
        check_small = name in ("LogitsProcessor", "Sampler")
        for boundary, values in (("input", inputs), ("output", output)):
            for index, tensor in enumerate(_tensors(values)):
                if (
                    not tensor.is_floating_point()
                    or tensor.ndim == 0
                    or (tensor.shape[0] < min_rows and not check_small)
                ):
                    continue
                torch._assert_async(
                    torch.isfinite(tensor).all(),
                    f"Q38 finite probe: {name} layer={layer} {boundary}[{index}] "
                    f"shape={tuple(tensor.shape)} dtype={tensor.dtype}",
                )
        if (
            samples is not None
            and name in ("Qwen4ExpLinearDecoderLayer", "Qwen4ExpAttentionDecoderLayer")
            and isinstance(output, tuple)
            and isinstance(output[0], torch.Tensor)
            and output[0].shape[0] >= min_rows
        ):
            tensor = output[0]
            call = layer_calls.get(id(module), 0)
            layer_calls[id(module)] = call + 1
            rows = [
                0,
                1,
                2,
                3,
                tensor.shape[0] - 4,
                tensor.shape[0] - 3,
                tensor.shape[0] - 2,
                tensor.shape[0] - 1,
            ]
            record = sample(tensor, rows)
            # A rare changed row may be absent from the bounded snapshots.
            # Include every full hidden/residual boundary for exact localization.
            record["full_input_sha256"] = [
                {"shape": tuple(value.shape), "sha256": full_digest(value)}
                for value in _tensors(inputs)
                if value.ndim > 0 and value.shape[0] == tensor.shape[0]
            ]
            record["full_output_sha256_all"] = [
                {"shape": tuple(value.shape), "sha256": full_digest(value)}
                for value in _tensors(output)
                if value.ndim > 0 and value.shape[0] == tensor.shape[0]
            ]
            record["full_output_sha256"] = record["full_output_sha256_all"][0][
                "sha256"
            ]
            torch.save(record, samples / f"{name}-layer{layer}-call{call:04d}.pt")
        if (
            samples is not None
            and name in ("QSAIndexer", "RadixAttention")
            and isinstance(output, torch.Tensor)
            and output.shape[0] >= min_rows
        ):
            call = layer_calls.get(id(module), 0)
            layer_calls[id(module)] = call + 1
            # Include the first atomic-top-k position (2051), both sides of
            # its boundary, and a bounded spread of later rows. Preserve the
            # original index order so CPU comparison can distinguish a
            # permutation from a changed selected set.
            rows = sorted(
                set(range(4))
                | set(range(2047, 2054))
                | set(range(0, output.shape[0], max(1, output.shape[0] // 64)))
                | set(range(output.shape[0] - 4, output.shape[0]))
            )
            rows = [row for row in rows if row < output.shape[0]]
            record = sample(output, rows)
            record["inputs"] = [
                sample(tensor, rows)
                for tensor in _tensors(inputs)
                if tensor.ndim > 0 and tensor.shape[0] == output.shape[0]
            ]
            # Hash complete Q/K/V (and indexer hidden input) rather than
            # inferring historical-key equality from a few sampled rows.
            # This synchronizing diagnostic is never a performance run.
            record["full_input_sha256"] = [
                {"shape": tuple(tensor.shape), "sha256": full_digest(tensor)}
                for tensor in _tensors(inputs)
                if tensor.ndim > 0 and tensor.shape[0] == output.shape[0]
            ]
            record["full_output_sha256"] = full_digest(output)
            torch.save(record, samples / f"{name}-layer{layer}-call{call:04d}.pt")
        if (
            name == "QSAIndexer"
            and isinstance(output, torch.Tensor)
            and output.ndim == 2
            and output.shape[0] >= min_rows
            and len(inputs) >= 2
        ):
            positions = inputs[1]
            if positions.ndim == 2:
                positions = positions[0]
            visible = positions[: output.shape[0]] + 1
            ratio = module.compress_ratio
            expected = torch.minimum(
                visible // ratio,
                torch.full_like(visible, module.block_topk),
            ) * ratio + visible.remainder(ratio)
            valid = output >= 0
            torch._assert_async(
                (valid.sum(1) == expected).all(),
                f"Q38 index probe: layer={layer} wrong valid count",
            )
            torch._assert_async(
                ((~valid) | (output < visible[:, None])).all(),
                f"Q38 index probe: layer={layer} non-causal or out-of-bounds index",
            )
            columns = torch.arange(output.shape[1], device=output.device)
            torch._assert_async(
                (valid == (columns[None, :] < expected[:, None])).all(),
                f"Q38 index probe: layer={layer} noncontiguous selected tokens",
            )

    return check

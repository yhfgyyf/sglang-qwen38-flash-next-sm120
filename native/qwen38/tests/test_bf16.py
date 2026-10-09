"""Numerical and CUDA-graph coverage for the SM120 BF16 GEMM prototype."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn.functional as F

from sglang.srt.model_executor.qwen38_bf16 import (
    BF16SmallMGemmConfig,
    bf16_small_m_linear,
)


CHECKPOINT_SHAPES = {
    "fused_gdn": (16480, 2560),
    "gdn_qkvz_core": (16384, 2560),
    "qsa_qkv": (13312, 2560),
    "out": (2560, 6144),
    "shared_gate_up": (1280, 2560),
    "down": (2560, 640),
    "indexer": (640, 2560),
    "router": (512, 2560),
    "hc_down": (320, 10240),
    "hc_up": (10240, 320),
}

DIRECT = BF16SmallMGemmConfig(
    block_m=16, block_n=64, block_k=64, num_warps=4, num_stages=3
)
SPLIT4 = BF16SmallMGemmConfig(
    block_m=16,
    block_n=32,
    block_k=64,
    num_warps=4,
    num_stages=3,
    split_k=4,
)


def _reference(x, weight, bias):
    return F.linear(
        x.float(),
        weight.float(),
        None if bias is None else bias.float(),
    )


def _assert_numeric(test, got, x, weight, bias, label, split_k):
    reference = _reference(x, weight, bias)
    torch_bias = None if bias is None else bias.to(torch.bfloat16)
    torch_bf16 = F.linear(x, weight, torch_bias)
    ideal_bf16 = reference.to(torch.bfloat16)
    custom_error = (got.float() - reference).abs()
    torch_error = (torch_bf16.float() - reference).abs()
    rounding_error = (ideal_bf16.float() - reference).abs()
    stats = {
        "custom_max": custom_error.max().item(),
        "custom_mean": custom_error.mean().item(),
        "torch_max": torch_error.max().item(),
        "torch_mean": torch_error.mean().item(),
        "ideal_round_max": rounding_error.max().item(),
        "ideal_round_mean": rounding_error.mean().item(),
    }
    print(
        "BF16_NUMERIC",
        label,
        f"split_k={split_k}",
        " ".join(f"{name}={value:.8g}" for name, value in stats.items()),
    )

    # Bound the prototype against the actual torch BF16 route plus a small,
    # reference-derived rounding allowance. This is intentionally not a
    # broad fixed rtol selected after seeing the result.
    max_allowance = stats["torch_max"] + 4 * stats["ideal_round_max"] + 1e-6
    mean_allowance = stats["torch_mean"] + 2 * stats["ideal_round_mean"] + 1e-7
    test.assertLessEqual(stats["custom_max"], max_allowance, label)
    test.assertLessEqual(stats["custom_mean"], mean_allowance, label)
    test.assertEqual(got.dtype, torch.bfloat16)


class BF16SmallMGemmTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if torch.cuda.get_device_capability() != (12, 0):
            raise RuntimeError("this test requires the authorized SM120 GPU")
        torch.manual_seed(20261007)

    def test_opt_in_gdn_dispatch_uses_original_bf16_and_protected_fallback(self):
        from sglang.srt.layers.quantization import unquant
        from sglang.srt.model_executor import qwen38_bf16

        weight = torch.randn((16480, 2560), device="cuda", dtype=torch.bfloat16) * 0.02
        with patch.object(unquant, "_enable_qwen38_bf16_gdn", True):
            for rows in (1, 4):
                x = torch.randn((rows, 2560), device="cuda", dtype=torch.bfloat16) * 0.1
                with patch.object(
                    qwen38_bf16, "bf16_small_m_linear", wraps=bf16_small_m_linear
                ) as selected:
                    output = unquant.bf16_gemm_dispatch(x, weight, None)
                    selected.assert_called_once()
                _assert_numeric(self, output, x, weight, None, f"gdn_{rows}", 1)
            # Untuned row counts must remain on the original framework path.
            x = torch.randn((5, 2560), device="cuda", dtype=torch.bfloat16)
            with patch.object(qwen38_bf16, "bf16_small_m_linear") as selected:
                output = unquant.bf16_gemm_dispatch(x, weight, None)
                selected.assert_not_called()
            torch.testing.assert_close(output, F.linear(x, weight), atol=0, rtol=0)

        with patch.object(unquant, "_enable_qwen38_bf16_gdn", False):
            x = torch.randn((1, 2560), device="cuda", dtype=torch.bfloat16)
            with patch.object(qwen38_bf16, "bf16_small_m_linear") as selected:
                output = unquant.bf16_gemm_dispatch(x, weight, None)
                selected.assert_not_called()
            torch.testing.assert_close(output, F.linear(x, weight), atol=0, rtol=0)

    def test_checkpoint_shapes_against_fp32_and_torch_bf16(self):
        device = torch.device("cuda")
        rows = [1, 4, 5, 16, 20, 40, 1, 4, 5, 16]
        for (name, (n, k)), m in zip(CHECKPOINT_SHAPES.items(), rows):
            with self.subTest(name=name, m=m, n=n, k=k):
                x = torch.randn((m, k), device=device, dtype=torch.bfloat16) * 0.1
                weight = torch.randn((n, k), device=device, dtype=torch.bfloat16) * 0.1
                bias = torch.randn(n, device=device, dtype=torch.bfloat16) * 0.01
                got = bf16_small_m_linear(x, weight, bias, config=DIRECT)
                _assert_numeric(self, got, x, weight, bias, name, 1)

    def test_opt_in_linear_allowlist_and_compiled_dynamic_fallback(self):
        from sglang.srt.layers.quantization import unquant
        from sglang.srt.model_executor import qwen38_bf16

        method = unquant.UnquantizedLinearMethod()
        with patch.object(unquant, "_enable_qwen38_bf16_linear", True):
            for n, k, rows in ((13312, 2560, (1, 4, 16)), (2560, 640, (1, 4, 16, 40))):
                weight = torch.randn((n, k), device="cuda", dtype=torch.bfloat16) * 0.02
                layer = SimpleNamespace(weight=weight)
                for m in rows:
                    x = torch.randn((m, k), device="cuda", dtype=torch.bfloat16) * 0.1
                    with patch.object(
                        qwen38_bf16, "bf16_small_m_linear", wraps=bf16_small_m_linear
                    ) as selected:
                        got = method.apply(layer, x)
                        selected.assert_called_once()
                    _assert_numeric(
                        self, got, x, weight, None, f"linear_{m}_{n}_{k}", 1
                    )
                for m in (5, 20):
                    x = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
                    with patch.object(qwen38_bf16, "bf16_small_m_linear") as selected:
                        got = method.apply(layer, x)
                        selected.assert_not_called()
                    torch.testing.assert_close(got, F.linear(x, weight), atol=0, rtol=0)

            # One compiled dynamic function must route both allowlisted and
            # untuned row counts without baking an m-dependent kernel choice.
            compiled = torch.compile(
                lambda x: method.apply(layer, x),
                backend="eager",
                dynamic=True,
                fullgraph=True,
            )
            for m in (4, 5):
                x = torch.randn((m, k), device="cuda", dtype=torch.bfloat16) * 0.1
                got = compiled(x)
                _assert_numeric(self, got, x, weight, None, f"compiled_{m}", 1)

            x = torch.randn((4, k), device="cuda", dtype=torch.bfloat16)
            method.apply(layer, x)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                got = method.apply(layer, x)
            x.mul_(0.5)
            graph.replay()
            _assert_numeric(self, got, x, weight, None, "direct_graph", 1)

        with patch.object(unquant, "_enable_qwen38_bf16_linear", False):
            with patch.object(qwen38_bf16, "bf16_small_m_linear") as selected:
                got = method.apply(layer, x)
                selected.assert_not_called()
            torch.testing.assert_close(got, F.linear(x, weight), atol=0, rtol=0)

    def test_tails_bias_and_explicit_row_strides(self):
        device = torch.device("cuda")
        m, n, k = 5, 70, 129
        x_storage = torch.randn((m, k + 11), device=device, dtype=torch.bfloat16)
        x = x_storage[:, :k]
        weight = torch.randn((n, k), device=device, dtype=torch.bfloat16)
        bias = torch.randn(n, device=device, dtype=torch.float32)
        out_storage = torch.empty((m, n + 7), device=device, dtype=torch.bfloat16)
        out = out_storage[:, :n]
        returned = bf16_small_m_linear(
            x,
            weight,
            bias,
            out=out,
            config=DIRECT,
        )
        self.assertIs(returned, out)
        self.assertGreater(x.stride(0), k)
        self.assertGreater(out.stride(0), n)
        _assert_numeric(self, out, x, weight, bias, "tails_row_stride", 1)

    def test_splitk_is_deterministic_and_handles_adverse_values(self):
        device = torch.device("cuda")
        m, n, k = 4, 97, 513
        columns = torch.arange(k, device=device)
        alternating = torch.where(columns % 2 == 0, 1.0, -1.0)
        x = alternating.expand(m, -1).to(torch.bfloat16).contiguous()
        weight = torch.ones((n, k), device=device, dtype=torch.bfloat16)
        weight[:, 1::4] *= 0.5
        weight[:, 3::4] *= 0.5
        weight[::3] *= 32
        bias = torch.linspace(-1, 1, n, device=device, dtype=torch.float32)
        workspace = torch.empty(
            (SPLIT4.split_k, m, n), device=device, dtype=torch.float32
        )
        first = bf16_small_m_linear(
            x,
            weight,
            bias,
            workspace=workspace,
            config=SPLIT4,
        )
        second = bf16_small_m_linear(
            x,
            weight,
            bias,
            workspace=workspace,
            config=SPLIT4,
        )
        self.assertTrue(torch.equal(first, second))
        _assert_numeric(self, first, x, weight, bias, "adverse_splitk", 4)

    def test_splitk_cuda_graph_replay_reads_changing_inputs(self):
        device = torch.device("cuda")
        m, n, k = 4, 512, 257
        static_x = torch.empty((m, k), device=device, dtype=torch.bfloat16)
        weight = torch.randn((n, k), device=device, dtype=torch.bfloat16) * 0.125
        bias = torch.randn(n, device=device, dtype=torch.bfloat16) * 0.01
        out = torch.empty((m, n), device=device, dtype=torch.bfloat16)
        workspace = torch.empty(
            (SPLIT4.split_k, m, n), device=device, dtype=torch.float32
        )
        x0 = torch.randn_like(static_x)
        x1 = torch.randn_like(static_x)
        static_x.copy_(x0)

        # Compile before capture; capture itself performs no allocation.
        bf16_small_m_linear(
            static_x,
            weight,
            bias,
            out=out,
            workspace=workspace,
            config=SPLIT4,
        )
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            bf16_small_m_linear(
                static_x,
                weight,
                bias,
                out=out,
                workspace=workspace,
                config=SPLIT4,
            )

        static_x.copy_(x0)
        graph.replay()
        torch.cuda.synchronize()
        got0 = out.clone()
        static_x.copy_(x1)
        graph.replay()
        torch.cuda.synchronize()
        got1 = out.clone()
        self.assertFalse(torch.equal(got0, got1))
        _assert_numeric(self, got0, x0, weight, bias, "graph_input0", 4)
        _assert_numeric(self, got1, x1, weight, bias, "graph_input1", 4)


if __name__ == "__main__":
    unittest.main()

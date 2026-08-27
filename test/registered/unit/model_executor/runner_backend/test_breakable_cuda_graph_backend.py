"""CPU-only output-buffer tests for the breakable CUDA graph backend."""

import unittest

import torch

from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.model_executor.runner_backend.breakable_cuda_graph_backend import (
    BreakableCudaGraphBackend,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestBreakableOutputBuffer(CustomTestCase):
    def setUp(self):
        self.backend = BreakableCudaGraphBackend.__new__(BreakableCudaGraphBackend)

    def test_logits_output_keeps_speculative_token_rows(self):
        output = LogitsProcessorOutput(
            next_token_logits=torch.arange(20).reshape(4, 5),
            hidden_states=torch.arange(12).reshape(4, 3),
        )
        output.topk_p = torch.arange(8).reshape(4, 2)
        output.topk_index = torch.arange(8).reshape(4, 2)

        rows = self.backend._output_rows(output, cap=1)
        self.assertEqual(rows, 4)

        buffer = self.backend._alloc_full_buffer(output, rows)
        self.backend._copy_output_to_buffer(output, buffer, rows)
        stored = self.backend._slice_output(buffer, rows)

        self.assertIsInstance(stored, LogitsProcessorOutput)
        self.assertTrue(torch.equal(stored.next_token_logits, output.next_token_logits))
        self.assertTrue(torch.equal(stored.hidden_states, output.hidden_states))
        self.assertTrue(torch.equal(stored.topk_p, output.topk_p))
        self.assertTrue(torch.equal(stored.topk_index, output.topk_index))

    def test_logits_output_rejects_mismatched_tensor_rows(self):
        output = LogitsProcessorOutput(
            next_token_logits=torch.arange(5).reshape(1, 5),
            hidden_states=torch.arange(12).reshape(4, 3),
        )

        with self.assertRaisesRegex(ValueError, "same leading dimension"):
            self.backend._output_rows(output, cap=1)


if __name__ == "__main__":
    unittest.main()

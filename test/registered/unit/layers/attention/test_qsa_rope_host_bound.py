"""Host-known RoPE bounds must be used only for plain-text prefill."""

from types import SimpleNamespace

import torch

from sglang.srt.layers.attention.qsa.qsa_indexer import (
    known_text_prefill_rope_max_position,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestQsaRopeHostBound(CustomTestCase):
    def test_text_extend_uses_largest_final_sequence_length(self):
        batch = SimpleNamespace(
            forward_mode=ForwardMode.EXTEND,
            seq_lens_cpu=torch.tensor([8192, 32000], dtype=torch.int32),
            mm_inputs=[None, None],
        )
        self.assertEqual(known_text_prefill_rope_max_position(batch), 31999)

    def test_multimodal_and_speculative_paths_keep_device_check(self):
        batch = SimpleNamespace(
            forward_mode=ForwardMode.EXTEND,
            seq_lens_cpu=torch.tensor([8192], dtype=torch.int32),
            mm_inputs=[SimpleNamespace()],
        )
        self.assertIsNone(known_text_prefill_rope_max_position(batch))
        batch.mm_inputs = [None]
        batch.forward_mode = ForwardMode.TARGET_VERIFY
        self.assertIsNone(known_text_prefill_rope_max_position(batch))
        batch.forward_mode = ForwardMode.EXTEND
        batch.seq_lens_cpu = None
        self.assertIsNone(known_text_prefill_rope_max_position(batch))

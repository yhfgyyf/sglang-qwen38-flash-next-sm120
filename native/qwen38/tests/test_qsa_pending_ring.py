"""Regression coverage for QSA speculative pending-ring compression.

Run with the pinned SGLang Python environment and this worktree on PYTHONPATH.
The test intentionally exercises the real fused QSA compression kernel.
"""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.layers.attention.qsa.qsa_indexer import QSAIndexer
from sglang.srt.layers.attention.qsa.graph_metadata import launch_graph_metadata
from sglang.srt.layers.attention.qsa.metadata import (
    build_group_ring_slots,
    build_pending_ring_slots,
)
from sglang.srt.mem_cache.qsa_kv_pool import qsa_pending_ring_size


HEAD_DIM = 128
COMPRESS_RATIO = 4
PENDING_RING_SIZE = 8
REQUEST_INDEX = 1
COMPRESSED_SLOT = 1


class _IdentityRotary:
    """Small RoPE surface with an identity FP32 cache for the fused kernel."""

    rotary_dim = HEAD_DIM
    is_neox_style = True
    mrope_section = None
    mrope_interleaved = False
    mrope_interleaved_glm = False

    def __init__(self, device):
        cos = torch.ones((32, HEAD_DIM // 2), dtype=torch.float32, device=device)
        sin = torch.zeros_like(cos)
        self.cos_sin_cache = torch.cat((cos, sin), dim=-1)


class _Pool:
    def __init__(self, device):
        self.index_state_dtype = torch.bfloat16
        self.qsa_pending_ring_size = PENDING_RING_SIZE
        self.key_state = torch.zeros(
            (24, 1, HEAD_DIM), dtype=torch.bfloat16, device=device
        )
        self.qsa_rope_position_buffer = torch.zeros(
            (24, 3), dtype=torch.int64, device=device
        )
        self.compressed = torch.zeros(
            (4, 1, HEAD_DIM), dtype=torch.bfloat16, device=device
        )

    def get_qsa_key_state_buffer(self, _layer_id):
        return self.key_state

    def set_qsa_key_state_buffer(self, _layer_id, loc, token_k):
        self.key_state[loc.long()] = token_k.to(self.key_state.dtype)

    def set_qsa_rope_position_buffer(self, loc, positions):
        if positions.ndim == 1:
            positions = positions.unsqueeze(0).expand(3, -1)
        self.qsa_rope_position_buffer[loc.long()] = positions.long().transpose(0, 1)

    def get_qsa_compressed_k_buffer(self, _layer_id):
        return self.compressed

    def set_qsa_compressed_k_buffer(self, _layer_id, loc, compressed_k):
        self.compressed[loc.long()] = compressed_k.to(self.compressed.dtype)


def _make_indexer(device):
    config = SimpleNamespace(
        indexer_n_heads=4,
        indexer_kv_heads=1,
        indexer_head_dim=HEAD_DIM,
        indexer_budget=2048,
        indexer_compress_ratio=COMPRESS_RATIO,
        hidden_size=256,
        rms_norm_eps=1e-6,
    )
    indexer = QSAIndexer(
        config,
        layer_id=0,
        quant_config=None,
        rotary_emb=_IdentityRotary(device),
    ).to(device=device, dtype=torch.bfloat16)
    with torch.no_grad():
        indexer.k_layernorm._weight_loader(
            indexer.k_layernorm.weight,
            torch.linspace(0.75, 1.25, HEAD_DIM, dtype=torch.bfloat16, device=device),
        )
    return indexer


def _metadata(pool, logical_positions):
    logical_positions = logical_positions.to(dtype=torch.int64)
    boundary_rows = torch.nonzero(
        (logical_positions + 1) % COMPRESS_RATIO == 0, as_tuple=False
    ).flatten()
    return SimpleNamespace(
        token_to_kv_pool=pool,
        compress_member_rows=None,
        is_cuda_graph=False,
        write_locs=torch.full(
            (boundary_rows.numel(),),
            COMPRESSED_SLOT,
            dtype=torch.int32,
            device=logical_positions.device,
        ),
        compress_group_positions=logical_positions.index_select(0, boundary_rows),
        compress_sequence_ids=boundary_rows,
        compress_group_ring_locs=None,
        token_to_batch_idx=torch.arange(
            logical_positions.numel(),
            dtype=torch.int32,
            device=logical_positions.device,
        ),
        req_pool_indices=torch.full(
            (logical_positions.numel(),),
            REQUEST_INDEX,
            dtype=torch.int32,
            device=logical_positions.device,
        ),
        sequence_lengths=(logical_positions + 1).to(torch.int32),
    )


def _seed_prefix_tail(pool, keys, prefix_length, device):
    tail_length = prefix_length % COMPRESS_RATIO
    if tail_length == 0:
        return
    prefix_positions = torch.arange(
        prefix_length - tail_length,
        prefix_length,
        dtype=torch.int64,
        device=device,
    )
    prefix_slots = (
        REQUEST_INDEX * PENDING_RING_SIZE + prefix_positions % PENDING_RING_SIZE
    )
    pool.set_qsa_key_state_buffer(
        0, prefix_slots, keys.index_select(0, prefix_positions)
    )
    pool.set_qsa_rope_position_buffer(prefix_slots, prefix_positions)


class QSAPendingRingArithmeticTests(unittest.TestCase):
    def test_ring_size_contract(self):
        self.assertEqual(qsa_pending_ring_size(4, None), 4)
        self.assertEqual(qsa_pending_ring_size(4, 1), 4)
        self.assertEqual(qsa_pending_ring_size(4, 4), 8)

    def test_partial_accept_keeps_every_possible_pending_tail(self):
        """An eight-slot ring preserves all accept-0..4 outcomes."""

        for prefix_length in range(8):
            request = torch.tensor([REQUEST_INDEX], dtype=torch.int32)
            prefix_positions = range(prefix_length - prefix_length % 4, prefix_length)
            candidates = range(prefix_length, prefix_length + 4)
            ring = {}
            for position in (*prefix_positions, *candidates):
                slot = int(
                    build_pending_ring_slots(
                        token_to_batch_idx=torch.tensor([0], dtype=torch.int32),
                        req_pool_indices=request,
                        sequence_lengths=torch.tensor(
                            [position + 1], dtype=torch.int32
                        ),
                        logical_positions=torch.tensor([position]),
                        compress_ratio=COMPRESS_RATIO,
                        pending_ring_size=PENDING_RING_SIZE,
                        is_extend=False,
                    )[0]
                )
                ring[slot] = position

            for accepted in range(5):
                accepted_length = prefix_length + accepted
                tail_start = accepted_length // COMPRESS_RATIO * COMPRESS_RATIO
                for position in range(tail_start, accepted_length):
                    slot = (
                        REQUEST_INDEX * PENDING_RING_SIZE + position % PENDING_RING_SIZE
                    )
                    self.assertEqual(
                        ring[slot],
                        position,
                        (prefix_length, accepted, position),
                    )

                # Replay the next real step(s), overwriting rejected candidates,
                # and prove the next boundary still gathers the exact group.
                replay_ring = dict(ring)
                next_boundary = (
                    (accepted_length + COMPRESS_RATIO - 1)
                    // COMPRESS_RATIO
                    * COMPRESS_RATIO
                )
                if next_boundary == accepted_length:
                    next_boundary += COMPRESS_RATIO
                for position in range(accepted_length, next_boundary):
                    slot = (
                        REQUEST_INDEX * PENDING_RING_SIZE + position % PENDING_RING_SIZE
                    )
                    replay_ring[slot] = position
                group_end = next_boundary - 1
                group_slots = build_group_ring_slots(
                    req_pool_indices=request,
                    group_end_positions=torch.tensor([group_end]),
                    sequence_ids=torch.tensor([0]),
                    compress_ratio=COMPRESS_RATIO,
                    pending_ring_size=PENDING_RING_SIZE,
                )[0].tolist()
                self.assertEqual(
                    [replay_ring[slot] for slot in group_slots],
                    list(range(next_boundary - COMPRESS_RATIO, next_boundary)),
                    (prefix_length, accepted),
                )

    def test_multi_request_and_padding_slots_do_not_alias(self):
        token_to_batch = torch.arange(6, dtype=torch.int32)
        requests = torch.tensor([1, 1, 3, 3, 0, 0], dtype=torch.int32)
        positions = torch.tensor([6, 7, 14, 15, 0, 1], dtype=torch.int64)
        slots = build_pending_ring_slots(
            token_to_batch_idx=token_to_batch,
            req_pool_indices=requests,
            sequence_lengths=(positions + 1).to(torch.int32),
            logical_positions=positions,
            compress_ratio=COMPRESS_RATIO,
            pending_ring_size=PENDING_RING_SIZE,
            is_extend=False,
        )
        self.assertEqual(slots.tolist(), [14, 15, 30, 31, 0, 1])
        self.assertEqual(len(set(slots[:4].tolist())), 4)
        self.assertTrue((slots[4:] < PENDING_RING_SIZE).all())

    def test_optimistic_compressed_group_is_masked_until_accepted(self):
        for prefix_length in range(8):
            candidate_positions = range(prefix_length, prefix_length + 4)
            written_blocks = {
                position // COMPRESS_RATIO
                for position in candidate_positions
                if (position + 1) % COMPRESS_RATIO == 0
            }
            self.assertTrue(
                all(
                    block >= prefix_length // COMPRESS_RATIO for block in written_blocks
                )
            )
            for accepted in range(5):
                visible_blocks = (prefix_length + accepted) // COMPRESS_RATIO
                accepted_end = prefix_length + accepted - 1
                for block in written_blocks:
                    group_end = (block + 1) * COMPRESS_RATIO - 1
                    if group_end > accepted_end:
                        self.assertGreaterEqual(block, visible_blocks)


class QSAPendingRingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if torch.cuda.get_device_capability() != (12, 0):
            raise RuntimeError("this test requires the authorized SM120 GPU")

    def test_batched_verify_matches_stepwise_compression_across_ring_wrap(self):
        """A prefix-6, four-token verify window must preserve group positions 4-7."""

        device = torch.device("cuda")
        torch.manual_seed(20261007)
        indexer = _make_indexer(device)
        keys = torch.randn((10, 1, HEAD_DIM), dtype=torch.bfloat16, device=device)

        stepwise_pool = _Pool(device)
        batched_pool = _Pool(device)
        _seed_prefix_tail(stepwise_pool, keys, 6, device)
        _seed_prefix_tail(batched_pool, keys, 6, device)
        self.assertTrue(indexer._use_fused_compress(stepwise_pool))

        candidate_positions = torch.arange(6, 10, dtype=torch.int64, device=device)
        for row, position in enumerate(candidate_positions):
            one_position = position.reshape(1)
            indexer.update_key_state_and_compress(
                keys[6 + row : 7 + row],
                one_position,
                one_position,
                _metadata(stepwise_pool, one_position),
            )

        indexer.update_key_state_and_compress(
            keys[6:],
            candidate_positions,
            candidate_positions,
            _metadata(batched_pool, candidate_positions),
        )

        torch.testing.assert_close(
            batched_pool.compressed[COMPRESSED_SLOT],
            stepwise_pool.compressed[COMPRESSED_SLOT],
            rtol=0,
            atol=0,
        )

    def test_prefix_and_candidate_length_matrix_matches_stepwise(self):
        device = torch.device("cuda")
        torch.manual_seed(20261008)
        indexer = _make_indexer(device)

        for prefix_length in range(8):
            for candidate_length in range(1, 5):
                keys = torch.randn(
                    (prefix_length + candidate_length, 1, HEAD_DIM),
                    dtype=torch.bfloat16,
                    device=device,
                )
                stepwise_pool = _Pool(device)
                batched_pool = _Pool(device)
                _seed_prefix_tail(stepwise_pool, keys, prefix_length, device)
                _seed_prefix_tail(batched_pool, keys, prefix_length, device)
                positions = torch.arange(
                    prefix_length,
                    prefix_length + candidate_length,
                    dtype=torch.int64,
                    device=device,
                )
                for row, position in enumerate(positions):
                    one_position = position.reshape(1)
                    indexer.update_key_state_and_compress(
                        keys[prefix_length + row : prefix_length + row + 1],
                        one_position,
                        one_position,
                        _metadata(stepwise_pool, one_position),
                    )
                indexer.update_key_state_and_compress(
                    keys[prefix_length:],
                    positions,
                    positions,
                    _metadata(batched_pool, positions),
                )
                torch.testing.assert_close(
                    batched_pool.compressed,
                    stepwise_pool.compressed,
                    rtol=0,
                    atol=0,
                    msg=(f"prefix={prefix_length}, candidates={candidate_length}"),
                )

    def test_graph_replay_multi_request_and_padding_use_extended_ring(self):
        device = torch.device("cuda")
        num_rows = 12
        pool = SimpleNamespace(
            qsa_pending_ring_size=PENDING_RING_SIZE,
            qsa_compressed_page_size=16,
        )
        indexer = SimpleNamespace(
            compress_ratio=COMPRESS_RATIO,
            graph_compressed_page_table=torch.zeros(
                (num_rows, 1), dtype=torch.int32, device=device
            ),
            graph_prefix_lengths=torch.zeros(
                num_rows, dtype=torch.int32, device=device
            ),
            graph_compressed_lengths=torch.zeros(
                num_rows, dtype=torch.int32, device=device
            ),
            graph_write_locs=torch.zeros(num_rows, dtype=torch.int32, device=device),
            decode_logical_positions=torch.zeros(
                num_rows, dtype=torch.int32, device=device
            ),
            pending_ring_slots=torch.zeros(num_rows, dtype=torch.int64, device=device),
            graph_ring_group_locs=torch.zeros(
                (num_rows, COMPRESS_RATIO), dtype=torch.int32, device=device
            ),
        )
        metadata = SimpleNamespace(
            indexer_metadata=indexer,
            sequence_lengths=torch.zeros(num_rows, dtype=torch.int32, device=device),
            row_req_pool_indices=torch.zeros(
                num_rows, dtype=torch.int32, device=device
            ),
        )
        req_to_token = torch.arange(4 * 64, dtype=torch.int32, device=device).reshape(
            4, 64
        )
        launch_graph_metadata(
            mode=1,
            bs=3,
            num_rows=num_rows,
            seq_lens=torch.tensor([6, 14, 1], dtype=torch.int32, device=device),
            req_pool_indices=torch.tensor([1, 3, 0], dtype=torch.int32, device=device),
            extend_lens=None,
            extend_len=4,
            num_padding=1,
            metadata=metadata,
            req_to_token=req_to_token,
            pool=pool,
        )
        torch.cuda.synchronize()

        self.assertEqual(
            metadata.sequence_lengths.tolist(),
            [7, 8, 9, 10, 15, 16, 17, 18, 1, 1, 1, 1],
        )
        self.assertEqual(
            metadata.row_req_pool_indices.tolist(), [1] * 4 + [3] * 4 + [0] * 4
        )
        rows = torch.arange(num_rows, dtype=torch.int32, device=device)
        expected_slots = build_pending_ring_slots(
            token_to_batch_idx=rows,
            req_pool_indices=metadata.row_req_pool_indices,
            sequence_lengths=metadata.sequence_lengths,
            logical_positions=indexer.decode_logical_positions,
            compress_ratio=COMPRESS_RATIO,
            pending_ring_size=PENDING_RING_SIZE,
            is_extend=False,
        )
        expected_groups = build_group_ring_slots(
            req_pool_indices=metadata.row_req_pool_indices,
            group_end_positions=indexer.decode_logical_positions,
            sequence_ids=rows.long(),
            compress_ratio=COMPRESS_RATIO,
            pending_ring_size=PENDING_RING_SIZE,
        ).to(torch.int32)
        torch.testing.assert_close(indexer.pending_ring_slots, expected_slots)
        torch.testing.assert_close(indexer.graph_ring_group_locs, expected_groups)
        self.assertEqual(
            indexer.pending_ring_slots.tolist(),
            [14, 15, 8, 9, 30, 31, 24, 25, 0, 0, 0, 0],
        )


if __name__ == "__main__":
    unittest.main()

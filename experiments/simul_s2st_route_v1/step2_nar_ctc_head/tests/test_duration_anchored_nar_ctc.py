#!/usr/bin/env python3
"""Unit tests for the duration-anchored causal NAR CTC head."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from experiments.simul_s2st_route_v1.step2_nar_ctc_head.duration_anchored_nar_ctc import (
    DurationAnchoredCausalNARCTC,
    adjacent_repeats,
    causal_mask,
    duration_frame_lengths,
    expand_text_to_frames,
)


class GeometryTest(unittest.TestCase):
    def test_causal_mask_blocks_the_future(self) -> None:
        mask = causal_mask(4, torch.device("cpu"))
        self.assertFalse(bool(mask[0, 0]))
        self.assertTrue(bool(mask[0, 1]))
        self.assertTrue(bool(mask[2, 3]))
        self.assertFalse(bool(mask[3, 2]))

    def test_duration_frames_cover_ctc_floor(self) -> None:
        duration = torch.tensor([4000, 4000])
        units = torch.tensor([100, 400])
        repeats = torch.tensor([0, 50])
        frames = duration_frame_lengths(
            duration,
            frames_per_second=75.0,
            unit_lengths=units,
            unit_repeats=repeats,
        )
        # 4s * 75 = 300; second row needs 450.
        self.assertEqual(frames.tolist(), [300, 450])

    def test_expand_interpolates_and_respects_lengths(self) -> None:
        encoded = torch.zeros(2, 3, 4)
        encoded[0, :, 0] = torch.tensor([0.0, 1.0, 2.0])
        encoded[1, 0, 0] = 5.0
        out = expand_text_to_frames(
            encoded,
            text_lengths=torch.tensor([3, 1]),
            frame_lengths=torch.tensor([5, 2]),
        )
        self.assertEqual(tuple(out.shape), (2, 5, 4))
        self.assertAlmostEqual(float(out[0, 0, 0]), 0.0, places=5)
        self.assertAlmostEqual(float(out[0, -1, 0]), 2.0, places=5)
        self.assertTrue(torch.allclose(out[1, :2, 0], torch.tensor([5.0, 5.0])))
        self.assertTrue(torch.allclose(out[1, 2:], torch.zeros(3, 4)))


class HeadTest(unittest.TestCase):
    def test_forward_shapes_and_blank_id(self) -> None:
        head = DurationAnchoredCausalNARCTC(
            qwen_hidden_size=16,
            model_size=32,
            semantic_vocab_size=8,
            frames_per_second=50.0,
            num_heads=4,
            t2u_layers=1,
            decoder_layers=1,
            max_frames=200,
        )
        hidden = torch.randn(2, 5, 16)
        lengths = torch.tensor([5, 3])
        duration = torch.tensor([2000, 1000])
        logits, frames = head(hidden, lengths, duration)
        self.assertEqual(head.blank_id, 8)
        self.assertEqual(logits.shape[-1], 9)
        self.assertEqual(frames.tolist(), [100, 50])
        self.assertEqual(logits.shape[1], 100)

    def test_t2u_and_decoder_are_both_causal(self) -> None:
        """Changing a future text token must not affect earlier frame logits."""

        head = DurationAnchoredCausalNARCTC(
            qwen_hidden_size=8,
            model_size=16,
            semantic_vocab_size=4,
            frames_per_second=50.0,
            num_heads=4,
            t2u_layers=2,
            decoder_layers=2,
            max_frames=80,
        )
        head.eval()
        base = torch.randn(1, 4, 8)
        lengths = torch.tensor([4])
        duration = torch.tensor([800])
        with torch.no_grad():
            left, frames = head(base, lengths, duration)
            mutated = base.clone()
            mutated[0, -1] += 10.0
            right, _ = head(mutated, lengths, duration)
        # First half of the timeline is driven by early text; it must be identical.
        cutoff = int(frames.item()) // 2
        self.assertTrue(torch.allclose(left[0, :cutoff], right[0, :cutoff], atol=1e-5, rtol=1e-5))
        # The last frames may move — otherwise the head is ignoring the mutation entirely.
        self.assertFalse(torch.allclose(left[0, -1], right[0, -1], atol=1e-5, rtol=1e-5))

    def test_padding_text_cannot_leak_into_frames(self) -> None:
        head = DurationAnchoredCausalNARCTC(
            qwen_hidden_size=8,
            model_size=16,
            semantic_vocab_size=4,
            frames_per_second=50.0,
            num_heads=4,
            t2u_layers=1,
            decoder_layers=1,
            max_frames=80,
        )
        head.eval()
        hidden = torch.randn(1, 4, 8)
        lengths = torch.tensor([2])
        duration = torch.tensor([600])
        with torch.no_grad():
            left, _ = head(hidden, lengths, duration)
            mutated = hidden.clone()
            mutated[0, 2:] += 50.0
            right, _ = head(mutated, lengths, duration)
        self.assertTrue(torch.allclose(left, right, atol=1e-5, rtol=1e-5))

    def test_greedy_decode_collapses_blanks(self) -> None:
        head = DurationAnchoredCausalNARCTC(semantic_vocab_size=3, model_size=16, num_heads=4)
        logits = torch.full((1, 5, 4), -5.0)
        logits[0, 0, 1] = 5
        logits[0, 1, 1] = 5
        logits[0, 2, 3] = 5  # blank
        logits[0, 3, 2] = 5
        logits[0, 4, 3] = 5
        decoded = head.greedy_decode(logits, torch.tensor([5]))
        self.assertEqual(decoded, [[1, 2]])


class RepeatTest(unittest.TestCase):
    def test_adjacent_repeats(self) -> None:
        self.assertEqual(int(adjacent_repeats(torch.tensor([1, 1, 2, 2, 2]))), 3)


if __name__ == "__main__":
    unittest.main(verbosity=2)

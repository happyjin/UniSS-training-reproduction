from __future__ import annotations

import unittest

import torch

from training.simul_uniss.train_bicodec_refinement import boundary_loss, multi_resolution_stft_loss


class BiCodecRefinementTests(unittest.TestCase):
    def test_losses_are_zero_for_identical_waveforms(self) -> None:
        waveform = torch.randn(2, 2048)
        self.assertAlmostEqual(float(boundary_loss(waveform, waveform)), 0.0, places=6)
        self.assertAlmostEqual(float(multi_resolution_stft_loss(waveform, waveform)), 0.0, places=6)

    def test_boundary_loss_detects_edge_change(self) -> None:
        reference = torch.zeros(1, 2000)
        prediction = reference.clone()
        prediction[:, :20] = 1.0
        self.assertGreater(float(boundary_loss(prediction, reference)), 0.0)


if __name__ == "__main__":
    unittest.main()

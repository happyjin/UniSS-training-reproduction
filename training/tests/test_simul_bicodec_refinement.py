from __future__ import annotations

import unittest

import torch

from training.simul_uniss.train_bicodec_refinement import (
    BiCodecRefinementModel,
    boundary_loss,
    multi_resolution_stft_loss,
    refinement_losses,
)


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

    def test_refinement_wrapper_is_differentiable(self) -> None:
        class FakeBiCodec(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.scale = torch.nn.Parameter(torch.tensor(1.0))

        model = BiCodecRefinementModel(FakeBiCodec())
        model.forward = lambda semantic, global_tokens: semantic.float() * model.bicodec.scale
        batch = {
            "semantic": torch.ones(1, 2048, dtype=torch.long),
            "global": torch.ones(1, 1, dtype=torch.long),
            "reference": torch.zeros(1, 2048),
        }
        losses = refinement_losses(model, batch)
        losses["total"].backward()
        self.assertIsNotNone(model.bicodec.scale.grad)


if __name__ == "__main__":
    unittest.main()

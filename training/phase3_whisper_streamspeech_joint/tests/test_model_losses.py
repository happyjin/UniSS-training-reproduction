from __future__ import annotations

import unittest
from collections import OrderedDict

import torch

from training.phase3_whisper_streamspeech_joint.config import JointLossWeights
from training.phase3_whisper_streamspeech_joint.losses import NormalizedLoss
from training.phase3_whisper_streamspeech_joint.model import (
    COMPONENTS,
    distributed_component_losses,
)


class ModelLossTest(unittest.TestCase):
    def test_component_loss_uses_streamspeech_and_replay_weights(self) -> None:
        parameter = torch.tensor(2.0, requires_grad=True)
        losses = OrderedDict(
            (
                ("bicodec_ctc", NormalizedLoss(parameter * 2, torch.tensor(2.0))),
                ("ar_s2tt", NormalizedLoss(parameter * 3, torch.tensor(3.0))),
                ("asr_ctc", NormalizedLoss(parameter * 4, torch.tensor(4.0))),
                ("nar_s2tt_ctc", NormalizedLoss(parameter * 5, torch.tensor(5.0))),
                ("phase3_replay", NormalizedLoss(parameter * 6, torch.tensor(6.0))),
            )
        )
        self.assertEqual(tuple(losses), COMPONENTS)
        total, means = distributed_component_losses(losses, JointLossWeights())
        # Every component mean equals parameter; weights sum to 17.5.
        self.assertEqual(float(total.detach()), 35.0)
        self.assertTrue(all(float(value) == 2.0 for value in means.values()))
        total.backward()
        self.assertEqual(float(parameter.grad), 17.5)

    def test_inactive_component_has_no_loss(self) -> None:
        parameter = torch.tensor(3.0, requires_grad=True)
        zero = parameter * 0.0
        losses = OrderedDict(
            (name, NormalizedLoss(zero, torch.tensor(0.0))) for name in COMPONENTS
        )
        total, means = distributed_component_losses(losses, JointLossWeights())
        self.assertEqual(float(total), 0.0)
        self.assertTrue(all(float(value) == 0.0 for value in means.values()))


if __name__ == "__main__":
    unittest.main()

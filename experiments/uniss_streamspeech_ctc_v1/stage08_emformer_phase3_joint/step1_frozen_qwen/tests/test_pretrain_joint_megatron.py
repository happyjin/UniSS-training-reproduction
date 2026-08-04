import sys
import unittest
from pathlib import Path

import torch


STEP = Path(__file__).resolve().parents[1]
TREE = STEP.parents[1]
for path in (
    TREE / "stage02_ctc_probe",
    TREE / "stage03_multitask_encoder",
    TREE / "stage03_multitask_encoder" / "ar_s2tt_v1",
    TREE / "stage04_b2_discrete_bridge",
    TREE / "stage07_end_to_end_eval",
    STEP,
):
    sys.path.insert(0, str(path))

from pretrain_joint_megatron import DirectionBalancedJointDataset, endpoint_multitask_losses


class JointLossTest(unittest.TestCase):
    def test_bilingual_micro_batch_has_finite_shared_loss(self) -> None:
        logits = {
            name: torch.zeros(2, 3, 3, requires_grad=True)
            for name in ("asr_eng", "asr_cmn", "nar_s2tt_eng", "nar_s2tt_cmn")
        }
        output = {
            "logits": logits,
            "output_lengths": torch.tensor([3, 3]),
            "ar_logits": {
                "cmn": (torch.zeros(1, 1, 2, requires_grad=True), torch.tensor([0])),
                "eng": (torch.zeros(1, 1, 2, requires_grad=True), torch.tensor([1])),
            },
            "ar_anchor": torch.tensor(0.0),
        }
        total, asr, nar, ar = endpoint_multitask_losses(
            output,
            source_targets=torch.tensor([0, 1]),
            source_lengths=torch.tensor([1, 1]),
            target_targets=torch.tensor([1, 0]),
            target_padded=torch.tensor([[1], [0]]),
            target_lengths=torch.tensor([1, 1]),
            direction_ids=torch.tensor([0, 1]),
            vocab={"eng": 2, "cmn": 2},
            asr_weight=4.0,
            nar_weight=4.0,
            ar_weight=8.0,
        )
        self.assertTrue(all(torch.isfinite(value) for value in (total, asr, nar, ar)))
        total.backward()
        self.assertIsNotNone(logits["asr_eng"].grad)
        self.assertIsNotNone(logits["nar_s2tt_eng"].grad)

    def test_rejects_unknown_direction(self) -> None:
        output = {
            "logits": {
                name: torch.zeros(1, 2, 3)
                for name in ("asr_eng", "asr_cmn", "nar_s2tt_eng", "nar_s2tt_cmn")
            },
            "output_lengths": torch.tensor([2]),
            "ar_logits": {},
            "ar_anchor": torch.tensor(0.0),
        }
        with self.assertRaises(ValueError):
            endpoint_multitask_losses(
                output,
                source_targets=torch.tensor([0]),
                source_lengths=torch.tensor([1]),
                target_targets=torch.tensor([0]),
                target_padded=torch.tensor([[0]]),
                target_lengths=torch.tensor([1]),
                direction_ids=torch.tensor([3]),
                vocab={"eng": 2, "cmn": 2},
                asr_weight=4.0,
                nar_weight=4.0,
                ar_weight=8.0,
            )

    def test_balanced_virtual_index_alternates_directions(self) -> None:
        dataset = DirectionBalancedJointDataset.__new__(DirectionBalancedJointDataset)
        dataset.dataset = [
            {
                "waveform": torch.zeros(4),
                "source_token_ids": torch.tensor([0]),
                "target_token_ids": torch.tensor([1]),
                "direction_id": direction,
                "phase3_record": {"id": str(index)},
            }
            for index, direction in enumerate((0, 0, 1))
        ]
        dataset.direction_indices = {
            0: torch.tensor([0, 1]).numpy(),
            1: torch.tensor([2]).numpy(),
        }
        dataset.pairs = 2
        self.assertEqual(len(dataset), 4)
        self.assertEqual(
            [int(dataset[index]["direction_id"]) for index in range(4)],
            [0, 1, 0, 1],
        )


if __name__ == "__main__":
    unittest.main()

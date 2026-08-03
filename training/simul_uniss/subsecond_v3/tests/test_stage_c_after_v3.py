from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch

from training.simul_uniss.subsecond_v1.stage_c import StageCGateConfig
from training.simul_uniss.subsecond_v3.stage_c_after_v3 import (
    extract_latent_gate_inputs,
)
from training.simul_uniss.subsecond_v3.stage_c_after_v3_data import (
    collate_stage_c_after_v3,
)


class FakeLatentStudent:
    def __init__(self) -> None:
        self.config = SimpleNamespace(segment_frames=4)
        self.codebook = torch.eye(4, dtype=torch.float32)


class StageCAfterV3Test(unittest.TestCase):
    def test_extracts_bounded_formal_evidence(self) -> None:
        student = FakeLatentStudent()
        output = {
            "glm_latent": torch.tensor(
                [
                    [[1.0, 0.0, 0.0, 0.0], [0.9, 0.1, 0.0, 0.0]],
                    [[0.0, 1.0, 0.0, 0.0], [0.0, 0.9, 0.1, 0.0]],
                ]
            ),
            "token_lengths": torch.tensor([2, 2]),
            "stability_logits": torch.tensor([[2.0, 3.0], [1.0, -1.0]]),
            "target_capacity_logits": torch.tensor([[0.0, 2.0], [0.0, -2.0]]),
            "source_ctc_logits": torch.randn(2, 4, 8),
            "output_lengths": torch.tensor([4, 4]),
        }
        batch = {
            "utterance_sample_lengths": torch.tensor([2_560, 3_200]),
            "full_samples": torch.tensor([8_000, 8_000]),
            "direction": torch.tensor([0, 1]),
            "support_count": torch.tensor([2, 0]),
            "safe_label": torch.tensor([1.0, 0.0]),
        }
        values = extract_latent_gate_inputs(
            student,  # type: ignore[arg-type]
            output,
            batch,
            StageCGateConfig(),
            tail_token_count=2,
            codebook_temperature=0.05,
            codebook_chunk_size=2,
        )
        self.assertEqual(tuple(values["context"].shape), (2, 4))
        self.assertEqual(tuple(values["evidence"].shape), (2, 8))
        self.assertTrue(bool((values["evidence"] >= 0.0).all()))
        self.assertTrue(bool((values["evidence"] <= 1.0).all()))
        self.assertEqual(values["labels"].tolist(), [1.0, 0.0])
        self.assertEqual(values["support_ready"].tolist(), [1.0, 0.0])

    def test_packed_collate_flattens_prefixes(self) -> None:
        def row(value: float) -> dict[str, torch.Tensor]:
            return {
                "waveform": torch.full((400,), value),
                "utterance_samples": torch.tensor(400),
                "full_samples": torch.tensor(800),
                "reference_glm": torch.tensor([1, 2]),
                "support_count": torch.tensor(0),
                "direction": torch.tensor(0),
                "record_index": torch.tensor(1),
                "safe_label": torch.tensor(value),
                "event_support_end_ms": torch.tensor(320),
            }

        batch = collate_stage_c_after_v3([[row(0.0), row(1.0)], [row(1.0)]])
        self.assertEqual(tuple(batch["waveform"].shape), (3, 400))
        self.assertEqual(batch["safe_label"].tolist(), [0.0, 1.0, 1.0])


if __name__ == "__main__":
    unittest.main()

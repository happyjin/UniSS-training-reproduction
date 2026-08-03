import sys
import unittest
from pathlib import Path

import torch
import numpy as np


STAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STAGE))

from endpoint_model import EndpointCTCStudent
from audio_data import DistributedLengthBucketBatchSampler
from training.simul_uniss.subsecond_v2.stage_b_latent_model import LatentStageBModelConfig


class EndpointModelTest(unittest.TestCase):
    def test_length_bucket_aligns_rank_duration_ranges(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory(dir="/opt/dlami/nvme/jasonleeeli") as directory:
            path = Path(directory) / "lengths.u32"
            np.arange(1, 33, dtype=np.uint32).tofile(path)
            samplers = [
                DistributedLengthBucketBatchSampler(path, 4, rank, 2, seed=1)
                for rank in range(2)
            ]
            left = list(samplers[0])
            right = list(samplers[1])
            self.assertEqual(len(left), len(right))
            for left_batch, right_batch in zip(left, right):
                combined = left_batch + right_batch
                values = np.arange(1, 33)[combined]
                self.assertLessEqual(int(values.max() - values.min()), 7)

    def test_small_causal_model_outputs_four_heads(self) -> None:
        config = LatentStageBModelConfig(
            policy_vocab_size=10,
            hidden_size=16,
            num_layers=2,
            num_heads=2,
            ffn_dim=32,
            n_mels=8,
            stack_factor=4,
            segment_frames=4,
            right_context_frames=2,
            left_context_frames=8,
        )
        model = EndpointCTCStudent(config, eng_vocab_size=11, cmn_vocab_size=13)
        waveform = torch.randn(2, 16_000)
        output = model(waveform, torch.tensor([16_000, 12_000]))
        self.assertEqual(set(output["logits"]), {"asr_eng", "asr_cmn", "nar_s2tt_eng", "nar_s2tt_cmn"})
        self.assertEqual(output["logits"]["asr_eng"].shape[-1], 12)
        self.assertEqual(output["logits"]["asr_cmn"].shape[-1], 14)


if __name__ == "__main__":
    unittest.main()

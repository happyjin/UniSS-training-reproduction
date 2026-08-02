from __future__ import annotations

import json
import tempfile
import unittest
from array import array
from pathlib import Path

import soundfile as sf
import torch

from training.simul_uniss.jsonl_index import write_index
from training.simul_uniss.subsecond_v2.stage_b_latent_data import (
    LatentStageBAudioDataset,
    collate_stage_b_latent,
)
from training.simul_uniss.subsecond_v2.stage_b_latent_model import (
    LatentCausalAudioStudent,
    LatentStageBModelConfig,
    nearest_codebook_tokens,
    pool_student_frames,
)


class _Tokenizer:
    ctc_vocab_size = 16

    def encode_ctc(self, text: str) -> list[int]:
        return [index + 1 for index, _ in enumerate(text)] or [1]


class LatentStageBTest(unittest.TestCase):
    def test_fixed_rate_pool_preserves_odd_final_frame(self) -> None:
        hidden = torch.tensor([[[1.0], [3.0], [5.0]]])
        pooled, lengths = pool_student_frames(hidden, torch.tensor([3]), factor=2)
        torch.testing.assert_close(pooled, torch.tensor([[[2.0], [5.0]]]))
        self.assertEqual(lengths.tolist(), [2])

    def test_nearest_codebook_uses_teacher_euclidean_geometry(self) -> None:
        codebook = torch.tensor([[0.0, 0.0], [2.0, 0.0], [0.0, 3.0]])
        latent = torch.tensor([[[1.8, 0.1], [0.1, 2.8]]])
        self.assertEqual(nearest_codebook_tokens(latent, codebook).tolist(), [[1, 2]])

    def test_dataset_keeps_repeated_glm_tokens_without_ctc_shift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audio = root / "audio.wav"
            sf.write(audio, [0.0] * 16_000, 16_000)
            manifest = root / "manifest.jsonl"
            record = {
                "source_audio": str(audio),
                "teacher_source_glm": [7, 7, 7, 9, 9, 10],
                "teacher_source_glm_end_ms": [80, 160, 240, 320, 400, 480],
                "transcription": "hello",
                "src_lang": "eng",
                "source_words": [{"text": "hello", "end_ms": 480}],
                "target_support": [{"support_end_ms": 320}],
            }
            manifest.write_text(json.dumps(record) + "\n", encoding="utf-8")
            write_index(manifest, array("Q", [0]))
            dataset = LatentStageBAudioDataset(
                manifest,
                _Tokenizer(),  # type: ignore[arg-type]
                max_audio_seconds=1.0,
                prefix_training=False,
                teacher_glm_field="teacher_source_glm",
                teacher_glm_end_field="teacher_source_glm_end_ms",
            )
            value = dataset[0]
            self.assertEqual(value["teacher_glm_ids"].tolist(), [7, 7, 7, 9, 9, 10])
            self.assertEqual(value["stability_target"].tolist(), [1.0, 1.0, 0.0, 0.0, 0.0, 0.0])
            batch = collate_stage_b_latent([value])
            self.assertEqual(batch["teacher_glm_ids"].tolist(), [[7, 7, 7, 9, 9, 10]])

    def test_latent_model_has_no_glm_classification_head(self) -> None:
        config = LatentStageBModelConfig(
            policy_vocab_size=16,
            codebook_size=32,
            codebook_dim=8,
            hidden_size=16,
            num_layers=1,
            num_heads=4,
            ffn_dim=32,
            n_mels=8,
            left_context_frames=4,
        )
        model = LatentCausalAudioStudent(config, torch.randn(32, 8))
        names = dict(model.named_parameters())
        self.assertIn("glm_latent_head.weight", names)
        self.assertNotIn("teacher_glm_head.weight", names)


if __name__ == "__main__":
    unittest.main()

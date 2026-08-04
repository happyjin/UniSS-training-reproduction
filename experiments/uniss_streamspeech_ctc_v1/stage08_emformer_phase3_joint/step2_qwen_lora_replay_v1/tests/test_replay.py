import sys
import unittest
from pathlib import Path

import torch
from torch import nn


STEP = Path(__file__).resolve().parents[1]
TREE = STEP.parents[1]
for path in (TREE / "stage03_multitask_encoder", TREE / "stage04_b2_discrete_bridge", STEP):
    sys.path.insert(0, str(path))

from phase3_batches import offline_replay_lm_batch
from replay_data import full_phase3_record


class FakeQwen(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embeddings = nn.Embedding(180500, 4)

    def get_input_embeddings(self):
        return self.embeddings


def encode(text: str) -> list[int]:
    return [100 + index for index, _ in enumerate(text.split())]


class ReplayTest(unittest.TestCase):
    def record(self):
        return {
            "id": "row",
            "src_lang": "eng",
            "tgt_lang": "cmn",
            "transcription": "hello",
            "translation": "ni hao",
            "source_glm": [1, 2, 3],
            "bicodec_global": list(range(32)),
            "target_bicodec": [4, 5, 6],
        }

    def test_full_record_keeps_teacher_source_tokens(self) -> None:
        value = full_phase3_record(self.record())
        self.assertEqual(value["source_glm"], [1, 2, 3])

    def test_offline_batch_masks_prompt_and_supervises_target(self) -> None:
        inputs, attention, labels, tokens = offline_replay_lm_batch(
            FakeQwen(), encode, [self.record()], torch.device("cpu")
        )
        self.assertEqual(inputs.shape[:2], attention.shape)
        self.assertGreater(tokens, 0)
        self.assertEqual(int((labels != -100).sum()), tokens)
        self.assertGreater(int((labels == -100).sum()), 0)

    def test_rejects_missing_source_glm(self) -> None:
        record = self.record()
        del record["source_glm"]
        with self.assertRaises(KeyError):
            full_phase3_record(record)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import json
import tempfile
import unittest
from array import array
from pathlib import Path

import soundfile as sf

from training.simul_uniss.jsonl_index import write_index
from training.simul_uniss.subsecond_v1.data import StageBAudioDataset


class _Tokenizer:
    def encode_ctc(self, text: str) -> list[int]:
        return [index + 1 for index, _ in enumerate(text)] or [1]


class FormalStageBDataTest(unittest.TestCase):
    def test_formal_teacher_and_timestamp_capacity_are_used(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audio = root / "audio.wav"
            sf.write(audio, [0.0] * 16000, 16000)
            manifest = root / "manifest.jsonl"
            record = {
                "source_audio": str(audio),
                "source_glm": [99, 98],
                "source_glm_end_ms": [500, 1000],
                "teacher_source_glm": [1, 2, 3],
                "teacher_source_glm_end_ms": [200, 600, 900],
                "transcription": "hello world",
                "src_lang": "eng",
                "source_words": [
                    {"text": "hello", "end_ms": 300},
                    {"text": "world", "end_ms": 900},
                ],
                "target_support": [
                    {"support_end_ms": 250},
                    {"support_end_ms": 800},
                ],
            }
            payload = (json.dumps(record) + "\n").encode()
            manifest.write_bytes(payload)
            write_index(manifest, array("Q", [0]))
            dataset = StageBAudioDataset(
                manifest,
                _Tokenizer(),  # type: ignore[arg-type]
                max_audio_seconds=1.0,
                prefix_training=False,
                teacher_glm_field="teacher_source_glm",
                teacher_glm_end_field="teacher_source_glm_end_ms",
            )
            value = dataset[0]
            self.assertEqual(value["teacher_glm"].tolist(), [2, 3, 4])
            self.assertEqual(float(value["target_capacity"]), 1.0)
            self.assertGreater(len(value["source_policy"]), 0)


if __name__ == "__main__":
    unittest.main()

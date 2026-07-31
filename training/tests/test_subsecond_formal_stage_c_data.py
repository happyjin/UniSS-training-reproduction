from __future__ import annotations

import json
import tempfile
import unittest
from array import array
from pathlib import Path

import soundfile as sf

from training.simul_uniss.jsonl_index import write_index
from training.simul_uniss.subsecond_v2.stage_c_data import StageCFormalCommitDataset


class FormalStageCDataTest(unittest.TestCase):
    def test_deterministic_slots_create_negative_and_positive_prefixes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audio = root / "audio.wav"
            sf.write(audio, [0.0] * (16000 * 2), 16000)
            manifest = root / "manifest.jsonl"
            value = {
                "id": "x",
                "formal_a68_pass": True,
                "source_audio": str(audio),
                "src_lang": "eng",
                "teacher_source_glm": [1, 2, 3],
                "micro_write_events": [
                    {
                        "support_end_ms": 560,
                        "safe_if_source_ms_gte": 640,
                        "uncertain_alignment": False,
                    }
                ],
            }
            manifest.write_bytes((json.dumps(value) + "\n").encode())
            write_index(manifest, array("Q", [0]))
            dataset = StageCFormalCommitDataset(
                manifest,
                max_audio_seconds=2.0,
                prefixes_per_record=2,
                random_prefix=False,
            )
            self.assertEqual(float(dataset[0]["safe_label"]), 0.0)
            self.assertEqual(float(dataset[1]["safe_label"]), 1.0)


if __name__ == "__main__":
    unittest.main()

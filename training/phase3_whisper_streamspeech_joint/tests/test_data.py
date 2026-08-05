from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import soundfile as sf
import torch

from training.phase3_whisper_streamspeech_joint.build_joint_manifests import build_manifests
from training.phase3_whisper_streamspeech_joint.dataset import (
    DeterministicReplaySchedule,
    JointAudioDataset,
    collate_joint,
)
from training.simul_uniss.jsonl_index import write_index


class DummyTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return [100 + ord(character) for character in text]


class KindDataset:
    def __init__(self, kind: str, size: int) -> None:
        self.kind = kind
        self.size = size

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int):
        return {"sample_kind": self.kind, "index": index}


class DataTest(unittest.TestCase):
    def _source(self, root: Path, records: int = 20) -> Path:
        audio = root / "audio.wav"
        sf.write(audio, torch.zeros(1600).numpy(), 16_000)
        path = root / "source.jsonl"
        offsets = []
        offset = 0
        with path.open("wb") as handle:
            for index in range(records):
                value = {
                    "id": f"item-{index}",
                    "src_lang": "eng" if index % 2 == 0 else "cmn",
                    "tgt_lang": "cmn" if index % 2 == 0 else "eng",
                    "transcription": "ab",
                    "translation": "cd",
                    "source_glm": [1, 2],
                    "target_bicodec": [3, 4, 5],
                    "bicodec_global": list(range(32)),
                    "source_audio": str(audio),
                    "source_duration_ms": 100,
                }
                encoded = (json.dumps(value) + "\n").encode()
                offsets.append(offset)
                handle.write(encoded)
                offset += len(encoded)
        write_index(path, offsets)
        return path

    def test_manifest_dataset_and_collation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            summary = build_manifests(
                train_sources=[self._source(root)],
                output_dir=root / "joint",
                tokenizer=DummyTokenizer(),
                validation_per_mille=500,
            )
            self.assertEqual(summary["status"], "complete")
            dataset = JointAudioDataset(
                root / "joint/joint_train.jsonl",
                root / "joint/tokenizer_maps",
            )
            batch = collate_joint([dataset[0], dataset[min(1, len(dataset) - 1)]])
            self.assertEqual(batch["sample_kind"], "joint")
            self.assertEqual(tuple(batch["waveform"].shape), (2, 1600))
            self.assertEqual(tuple(batch["bicodec_global"].shape), (2, 32))

    def test_replay_schedule_is_exact_and_restart_stable(self) -> None:
        schedule = DeterministicReplaySchedule(
            KindDataset("joint", 7), KindDataset("replay", 3), cycles=10
        )
        kinds = [schedule[index]["sample_kind"] for index in range(len(schedule))]
        self.assertEqual(kinds.count("joint"), 40)
        self.assertEqual(kinds.count("replay"), 10)
        self.assertEqual(schedule.replay_probability, 0.2)
        self.assertEqual(schedule.scheduled_index(17), schedule.scheduled_index(17))


if __name__ == "__main__":
    unittest.main()

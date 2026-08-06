from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import soundfile as sf
import torch

from training.phase3_whisper_streamspeech_joint.build_joint_manifests import build_manifests
from training.phase3_whisper_streamspeech_joint.build_replay_index import build_replay_index
from training.phase3_whisper_streamspeech_joint.dataset import (
    DeterministicReplaySchedule,
    DirectionBalancedJointDataset,
    JointAudioDataset,
    IndexedPhase3ReplayDataset,
    SynchronizedKindRandomSampler,
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
            balanced = DirectionBalancedJointDataset(
                dataset,
                root / "joint/direction_indices",
                "train",
            )
            self.assertEqual([balanced[index]["direction_id"] for index in range(4)], [0, 1, 0, 1])

    def test_replay_schedule_is_exact_and_restart_stable(self) -> None:
        schedule = DeterministicReplaySchedule(
            KindDataset("joint", 7), KindDataset("replay", 3), cycles=10
        )
        kinds = [schedule[index]["sample_kind"] for index in range(len(schedule))]
        self.assertEqual(kinds.count("joint"), 40)
        self.assertEqual(kinds.count("replay"), 10)
        self.assertEqual(schedule.replay_probability, 0.2)
        self.assertEqual(schedule.scheduled_index(17), schedule.scheduled_index(17))

    def test_replay_kind_is_synchronized_across_data_parallel_lanes(self) -> None:
        data_parallel_size = 4
        schedule = DeterministicReplaySchedule(
            KindDataset("joint", 31),
            KindDataset("replay", 11),
            data_parallel_group_size=data_parallel_size,
            cycles=3,
        )
        samplers = [
            SynchronizedKindRandomSampler(
                schedule,
                total_samples=len(schedule),
                consumed_samples=0,
                micro_batch_size=1,
                data_parallel_rank=rank,
                data_parallel_size=data_parallel_size,
                data_sharding=False,
            )
            for rank in range(data_parallel_size)
        ]
        per_rank = [list(iter(sampler)) for sampler in samplers]
        self.assertTrue(per_rank[0])
        for microbatch in zip(*per_rank):
            indices = [values[0] for values in microbatch]
            kinds = [schedule[index]["sample_kind"] for index in indices]
            self.assertEqual(len(set(kinds)), 1)
            self.assertEqual(len(set(index // data_parallel_size for index in indices)), 1)
            self.assertEqual(
                sorted(index % data_parallel_size for index in indices),
                list(range(data_parallel_size)),
            )

    def test_indexed_replay_does_not_scan_the_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            packed = root / "packed.jsonl"
            row = {
                "tokens": [1, 2, 0, 0],
                "labels": [2, 3, 0, 0],
                "loss_mask": [1, 1, 0, 0],
                "position_ids": [0, 1, 0, 0],
                "sample_boundaries": [[0, 2]],
            }
            packed.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n")
            offsets = root / "replay.u64"
            build_replay_index(packed, offsets, max_records=1, progress_interval=0)
            dataset = IndexedPhase3ReplayDataset(
                packed,
                offsets,
                seq_length=4,
                require_complete=False,
            )
            self.assertEqual(len(dataset), 1)
            self.assertEqual(dataset[0]["sample_kind"], "replay")
            self.assertEqual(dataset[0]["tokens"].tolist(), [1, 2, 0, 0])


if __name__ == "__main__":
    unittest.main()

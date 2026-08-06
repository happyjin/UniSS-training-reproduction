"""Indexed joint audio records and deterministic exact-Phase3 replay."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
import torchaudio
import numpy as np
from torch.utils.data import Dataset

from training.megatron_uniss_dataset import UniSSPackedJsonlDataset
from training.megatron_uniss_dataset import packed_json_to_megatron_item
from training.phase3_whisper_streamspeech_joint.build_joint_manifests import SCHEMA
from training.phase3_whisper_streamspeech_joint.tokenizer_maps import CompactCTCMap
from training.simul_uniss.jsonl_index import load_index


DIRECTION_ID = {("eng", "cmn"): 0, ("cmn", "eng"): 1}


class JointAudioDataset(Dataset[dict[str, object]]):
    def __init__(self, manifest: str | Path, tokenizer_maps: str | Path) -> None:
        self.path = Path(manifest)
        offsets = load_index(self.path)
        if offsets is None:
            raise ValueError(f"missing offset index for {self.path}")
        self.offsets = offsets
        root = Path(tokenizer_maps)
        self.maps = {
            language: CompactCTCMap.load(root / f"ctc_qwen_{language}.json")
            for language in ("eng", "cmn")
        }

    def __len__(self) -> int:
        return len(self.offsets)

    def _read(self, index: int) -> dict[str, object]:
        with self.path.open("rb") as handle:
            handle.seek(int(self.offsets[index]))
            value = json.loads(handle.readline())
        if value.get("schema_version") != SCHEMA:
            raise ValueError(f"unexpected record schema at {self.path}:{index}")
        return value

    def __getitem__(self, index: int) -> dict[str, object]:
        value = self._read(index)
        source_language = str(value["src_lang"])
        target_language = str(value["tgt_lang"])
        waveform, sample_rate = torchaudio.load(str(value["source_audio"]))
        waveform = waveform[:1]
        if sample_rate != 16_000:
            waveform = torchaudio.functional.resample(waveform, sample_rate, 16_000)
        source_qwen = [int(token) for token in value["source_qwen_ids"]]
        target_qwen = [int(token) for token in value["target_qwen_ids"]]
        return {
            "sample_kind": "joint",
            "id": str(value["id"]),
            "direction_id": DIRECTION_ID[(source_language, target_language)],
            "source_language": source_language,
            "target_language": target_language,
            "waveform": waveform.squeeze(0),
            "waveform_length": len(waveform[0]),
            "source_ctc_ids": torch.tensor(self.maps[source_language].encode(source_qwen), dtype=torch.long),
            "target_ctc_ids": torch.tensor(self.maps[target_language].encode(target_qwen), dtype=torch.long),
            "source_qwen_ids": torch.tensor(source_qwen, dtype=torch.long),
            "target_qwen_ids": torch.tensor(target_qwen, dtype=torch.long),
            "source_glm": torch.tensor(value["source_glm"], dtype=torch.long),
            "target_bicodec": torch.tensor(value["target_bicodec"], dtype=torch.long),
            "bicodec_global": torch.tensor(value["bicodec_global"], dtype=torch.long),
            "phase3_record_json": json.dumps(value, ensure_ascii=False, separators=(",", ":")),
        }


class Phase3ReplayDataset(UniSSPackedJsonlDataset):
    def __getitem__(self, index: int) -> dict[str, object]:
        value: dict[str, object] = dict(super().__getitem__(index))
        value["sample_kind"] = "replay"
        return value


class IndexedPhase3ReplayDataset(Dataset[dict[str, object]]):
    """Random access to a huge packed JSONL without rescanning it per rank."""

    def __init__(
        self,
        path: str | Path,
        offsets: str | Path,
        *,
        seq_length: int,
        require_complete: bool = True,
    ) -> None:
        self.path = Path(path).resolve()
        self.offset_path = Path(offsets).resolve()
        metadata_path = self.offset_path.with_suffix(self.offset_path.suffix + ".json")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("schema_version") != "uniss_phase3_replay_offsets_v1":
            raise ValueError("unexpected replay offset schema")
        stat = self.path.stat()
        if (
            Path(str(metadata["source"])).resolve() != self.path
            or int(metadata["source_size_bytes"]) != stat.st_size
            or int(metadata["source_mtime_ns"]) != stat.st_mtime_ns
        ):
            raise ValueError("packed replay source changed after indexing")
        if require_complete and not bool(metadata.get("complete")):
            raise ValueError("formal training requires a complete replay index")
        self.offsets = np.memmap(self.offset_path, mode="r", dtype=np.uint64)
        if len(self.offsets) != int(metadata["records"]):
            raise ValueError("replay offset count does not match metadata")
        self.seq_length = int(seq_length)

    def __len__(self) -> int:
        return len(self.offsets)

    def __getitem__(self, index: int) -> dict[str, object]:
        with self.path.open("rb") as handle:
            handle.seek(int(self.offsets[index]))
            value = json.loads(handle.readline())
        result: dict[str, object] = dict(
            packed_json_to_megatron_item(value, seq_length=self.seq_length)
        )
        result["sample_kind"] = "replay"
        return result


class DirectionBalancedJointDataset(Dataset[dict[str, object]]):
    """Alternating EN→ZH/ZH→EN view over one immutable joint manifest."""

    def __init__(self, dataset: JointAudioDataset, direction_index_dir: str | Path, split: str) -> None:
        self.dataset = dataset
        root = Path(direction_index_dir)
        self.indices = {
            direction: np.load(
                root / f"{split}_{direction.replace('->', '_to_')}.npy",
                mmap_mode="r",
                allow_pickle=False,
            )
            for direction in ("eng->cmn", "cmn->eng")
        }
        if any(not len(values) for values in self.indices.values()):
            raise ValueError(f"both directions must be non-empty for {split}")
        self.pairs = max(len(values) for values in self.indices.values())

    def __len__(self) -> int:
        return 2 * self.pairs

    def __getitem__(self, index: int) -> dict[str, object]:
        direction = "eng->cmn" if index % 2 == 0 else "cmn->eng"
        values = self.indices[direction]
        source_index = int(values[(index // 2) % len(values)])
        value = self.dataset[source_index]
        expected = 0 if direction == "eng->cmn" else 1
        if int(value["direction_id"]) != expected:
            raise ValueError("direction index points to the wrong record")
        return value


@dataclass(frozen=True)
class ScheduledIndex:
    sample_kind: str
    source_index: int


class DeterministicReplaySchedule(Dataset[dict[str, object]]):
    """A restart-stable 4 joint : 1 replay virtual dataset.

    The exact ratio is represented as integers so checkpoint resume cannot
    drift due to independent RNG state.  Megatron may still globally shuffle
    these virtual indices.
    """

    def __init__(
        self,
        joint: Dataset,
        replay: Dataset,
        *,
        joint_slots: int = 4,
        replay_slots: int = 1,
        cycles: int | None = None,
    ) -> None:
        if not len(joint) or not len(replay):
            raise ValueError("joint and replay datasets must be non-empty")
        if joint_slots <= 0 or replay_slots <= 0:
            raise ValueError("schedule slots must be positive")
        self.joint = joint
        self.replay = replay
        self.joint_slots = int(joint_slots)
        self.replay_slots = int(replay_slots)
        self.cycle_size = self.joint_slots + self.replay_slots
        self.cycles = int(cycles or max(len(joint), len(replay)))

    @property
    def replay_probability(self) -> float:
        return self.replay_slots / self.cycle_size

    def __len__(self) -> int:
        return self.cycles * self.cycle_size

    def scheduled_index(self, index: int) -> ScheduledIndex:
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        cycle, slot = divmod(index, self.cycle_size)
        if slot < self.joint_slots:
            return ScheduledIndex("joint", (cycle * self.joint_slots + slot) % len(self.joint))
        replay_slot = slot - self.joint_slots
        return ScheduledIndex("replay", (cycle * self.replay_slots + replay_slot) % len(self.replay))

    def __getitem__(self, index: int) -> dict[str, object]:
        scheduled = self.scheduled_index(index)
        dataset = self.joint if scheduled.sample_kind == "joint" else self.replay
        value = dict(dataset[scheduled.source_index])
        if value.get("sample_kind") != scheduled.sample_kind:
            raise ValueError("scheduled sample kind does not match source dataset")
        return value


def collate_joint(batch: list[dict[str, object]]) -> dict[str, object]:
    if not batch or any(value.get("sample_kind") != "joint" for value in batch):
        raise ValueError("collate_joint accepts only joint samples")
    waveform_lengths = torch.tensor([int(value["waveform_length"]) for value in batch], dtype=torch.long)
    waveform = torch.zeros(len(batch), int(waveform_lengths.max()), dtype=torch.float32)
    for row, value in enumerate(batch):
        samples = value["waveform"]
        waveform[row, : len(samples)] = samples  # type: ignore[arg-type]
    result: dict[str, object] = {
        "sample_kind": "joint",
        "ids": [str(value["id"]) for value in batch],
        "direction_ids": torch.tensor([int(value["direction_id"]) for value in batch]),
        "waveform": waveform,
        "waveform_lengths": waveform_lengths,
        "phase3_record_json": [str(value["phase3_record_json"]) for value in batch],
    }
    for name in (
        "source_ctc_ids",
        "target_ctc_ids",
        "source_qwen_ids",
        "target_qwen_ids",
        "source_glm",
        "target_bicodec",
    ):
        tensors = [value[name] for value in batch]
        lengths = torch.tensor([len(value) for value in tensors], dtype=torch.long)
        padded = torch.full((len(batch), int(lengths.max())), -1, dtype=torch.long)
        for row, value in enumerate(tensors):
            padded[row, : len(value)] = value  # type: ignore[arg-type]
        result[name] = padded
        result[f"{name}_lengths"] = lengths
    result["bicodec_global"] = torch.stack([value["bicodec_global"] for value in batch])
    return result

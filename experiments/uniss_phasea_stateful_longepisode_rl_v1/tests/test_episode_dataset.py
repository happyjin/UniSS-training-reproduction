import json
import struct

from experiments.uniss_phasea_stateful_longepisode_rl_v1.training.dataset import (
    EpisodeGRPOPackedDataset,
    collate_episode_grpo,
)


def test_dataset_reads_sidecars_and_repeats(tmp_path):
    path = tmp_path / "packs.jsonl"
    row = {
        "tokens": [1, 2, 3, 4], "labels": [2, 3, 4, 5],
        "position_ids": [0, 1, 0, 1], "family_ids": [1, 1, 3, 3],
        "loss_mask": [1.0] * 4, "response_mask": [1.0, 1.0, 0.0, 0.0],
        "old_log_probs": [-1.0, -1.0, 0.0, 0.0],
        "advantages": [1.0, 1.0, 0.0, 0.0],
        "replay_mask": [0.0, 0.0, 1.0, 1.0],
        "sample_boundaries": [[0, 2], [2, 4]],
    }
    raw = (json.dumps(row) + "\n").encode()
    path.write_bytes(raw)
    path.with_suffix(".jsonl.offsets.bin").write_bytes(struct.pack("<Q", 0))
    dataset = EpisodeGRPOPackedDataset(path, 4, target_length=2)
    batch = collate_episode_grpo([dataset[0], dataset[1]])
    assert tuple(batch["tokens"].shape) == (2, 4)
    assert batch["sample_boundaries"] == [[(0, 2), (2, 4)], [(0, 2), (2, 4)]]


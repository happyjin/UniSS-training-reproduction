import json
from pathlib import Path

from experiments.uniss_phasea_stateful_longepisode_rl_v1.training.merge_rollout_workers import (
    merge_workers,
)


def test_merge_workers(tmp_path: Path):
    workers = tmp_path / "workers"
    for index in range(2):
        root = workers / f"worker_{index}"
        root.mkdir(parents=True)
        trajectory = root / "trajectories.jsonl"
        trajectory.write_text("{}\n", encoding="utf-8")
        payload = {
            "status": "complete",
            "worker_index": index,
            "num_workers": 2,
            "group_size": 2,
            "trajectory_path": str(trajectory),
            "summaries": [
                {
                    "episode_id": f"episode_{index}",
                    "candidates": [
                        {
                            "reward": {"total": float(index + 1)},
                            "observation": {
                                "first_write_ms": 640.0,
                                "spoken_text_fraction": 1.0,
                            },
                        }
                    ],
                }
            ],
        }
        (root / "ROLLOUT.json").write_text(json.dumps(payload), encoding="utf-8")
    merged = merge_workers(workers, 2)
    assert merged["episodes"] == 2
    assert merged["candidates"] == 2
    assert merged["aggregate"]["mean_reward"] == 1.5

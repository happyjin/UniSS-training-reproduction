from __future__ import annotations

import os
import subprocess
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def run_script(path: str, *args: str) -> str:
    result = subprocess.run(
        [str(REPO_ROOT / path), *args],
        cwd=REPO_ROOT,
        env=os.environ.copy(),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    return result.stdout


class SimulScriptsTests(unittest.TestCase):
    def test_prepare_dry_run_uses_exactly_fifteen_shards(self) -> None:
        output = run_script("scripts/simul_uniss/prepare_bootstrap_15shard.sh", "--dry-run")
        self.assertEqual(output.count("train-000"), 15)
        self.assertIn("train-00000.parquet", output)
        self.assertIn("train-00014.parquet", output)
        self.assertNotIn("train-00015.parquet", output)
        self.assertIn("simul_uniss_v1/bootstrap_15shard", output)

    def test_qwen_stages_are_isolated_and_have_validation(self) -> None:
        action = run_script(
            "scripts/simul_uniss/train_qwen_stage.sh", "--stage", "action", "--dry-run", "--smoke"
        )
        interleaved = run_script(
            "scripts/simul_uniss/train_qwen_stage.sh",
            "--stage",
            "interleaved",
            "--dry-run",
            "--smoke",
        )
        joint = run_script(
            "scripts/simul_uniss/train_qwen_stage.sh", "--stage", "joint", "--dry-run", "--smoke"
        )
        for output in (action, interleaved, joint):
            self.assertIn("pretrain_simul_uniss_megatron.py", output)
            self.assertIn("--simul-packed-valid", output)
            self.assertIn("--eval-iters 1", output)
            self.assertIn("--log-validation-ppl-to-tensorboard", output)
            self.assertIn("checkpoints/simul_uniss_v1", output)
            self.assertNotIn("--save /opt/dlami/nvme/jasonleeeli/projects/UniSS/checkpoints/uniss_qwen0p5b_phase1", output)
        self.assertIn("packed_action_train.jsonl", action)
        self.assertIn("stage3_action", action)
        self.assertIn("packed_train.jsonl", interleaved)
        self.assertIn("stage4_interleaved", interleaved)
        self.assertIn("stage6_joint", joint)
        self.assertIn("--lr 3e-6", joint)

    def test_gpu_smoke_pipeline_covers_real_components(self) -> None:
        output = run_script("scripts/simul_uniss/run_gpu_smoke_pipeline.sh", "--dry-run")
        self.assertIn("run_stage0_prefix_baseline.sh", output)
        self.assertIn("train_stage1_audio_student.sh", output)
        self.assertIn("train_streaming_student", output)
        self.assertIn("--stage action", output)
        self.assertIn("--stage interleaved", output)
        self.assertIn("--stage joint", output)
        self.assertIn("--decoder bicodec", output)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import os
import subprocess
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


class StageBV2ScriptsTest(unittest.TestCase):
    def test_smoke_dry_run_uses_isolated_quantization_aware_trainer(self) -> None:
        result = subprocess.run(
            [
                "bash",
                "scripts/simul_uniss_subsecond_v2/train_stage_b_v2_causal.sh",
                "smoke",
                "--dry-run",
            ],
            cwd=REPO_ROOT,
            env=os.environ.copy(),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
        output = result.stdout
        self.assertIn("training.simul_uniss.subsecond_v2.train_stage_b_v2", output)
        self.assertIn("--sidecar-manifest", output)
        self.assertIn("--representation-only-steps 1", output)
        self.assertIn("--quantize-chunk-size 256", output)
        self.assertIn("--master-addr 127.0.0.1", output)
        self.assertNotIn("stage_b_latent_formal_15shard_v1", output)

    def test_config_separates_clone_and_prefix_outputs(self) -> None:
        source = (
            REPO_ROOT
            / "configs/experiments/simul_uniss_subsecond_v2/stage_b_v2_causal_15shard_v1.env"
        ).read_text(encoding="utf-8")
        self.assertIn("stage_b_v2_clone_pretrain_15shard_v1", source)
        self.assertIn("stage_b_v2_prefix80_finetune_100k_v1", source)
        self.assertIn('STAGE_B_V2_BATCH_SIZE="${STAGE_B_V2_BATCH_SIZE:-32}"', source)

    def test_phase3_evaluation_uses_explicit_v2_student_label(self) -> None:
        pipeline = (
            REPO_ROOT
            / "scripts/simul_uniss_subsecond_v2/run_stage_b_v2_repair_pipeline.sh"
        ).read_text(encoding="utf-8")
        launcher = (
            REPO_ROOT
            / "scripts/simul_uniss_subsecond_v2/run_phase3_token_stream_sensitivity.sh"
        ).read_text(encoding="utf-8")
        evaluator = (
            REPO_ROOT
            / "training/simul_uniss/subsecond_v2/evaluate_phase3_token_streams.py"
        ).read_text(encoding="utf-8")
        self.assertIn("STUDENT_STREAM_NAME=student_v2_prefix80", pipeline)
        self.assertIn('--student-stream-name "${STUDENT_STREAM_NAME}"', launcher)
        self.assertIn("args.student_stream_name", evaluator)


if __name__ == "__main__":
    unittest.main()

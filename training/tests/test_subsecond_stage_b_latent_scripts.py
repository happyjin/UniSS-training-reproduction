from __future__ import annotations

import os
import subprocess
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


class LatentStageBScriptsTest(unittest.TestCase):
    def test_smoke_dry_run_is_isolated_and_uses_fixed_localhost_rendezvous(self) -> None:
        env = os.environ.copy()
        result = subprocess.run(
            [
                "bash",
                "scripts/simul_uniss_subsecond_v2/train_stage_b_latent_formal.sh",
                "smoke",
                "--dry-run",
            ],
            cwd=REPO_ROOT,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
        output = result.stdout
        self.assertIn("--master-addr 127.0.0.1", output)
        self.assertIn("--master-port 29743", output)
        self.assertIn("training.simul_uniss.subsecond_v2.train_stage_b_latent", output)
        self.assertIn("--valid-manifest", output)
        self.assertIn("--codebook-key codebook.weight", output)
        self.assertIn("--capacity-weight", output)
        self.assertIn("--consistency-weight", output)
        self.assertIn("validate_stage_b_latent", output)
        self.assertIn("stage_b_latent_formal_launcher", output)
        self.assertNotIn("subsecond_v1.train_stage_b", output)

    def test_formal_config_keeps_historical_ctc_paths_isolated(self) -> None:
        source = (
            REPO_ROOT
            / "configs/experiments/simul_uniss_subsecond_v2/stage_b_latent_formal_15shard_v1.env"
        ).read_text(encoding="utf-8")
        self.assertIn("stage_b_latent_formal_15shard_v1", source)
        self.assertIn('STAGE_B_LATENT_BATCH_SIZE="${STAGE_B_LATENT_BATCH_SIZE:-64}"', source)
        self.assertIn('STAGE_B_CAPACITY_WEIGHT="${STAGE_B_CAPACITY_WEIGHT:-0.4}"', source)
        self.assertNotIn('STAGE_B_ROOT="', source)

    def test_trainer_exposes_non_mutating_throughput_scan(self) -> None:
        source = (
            REPO_ROOT
            / "training/simul_uniss/subsecond_v2/train_stage_b_latent.py"
        ).read_text(encoding="utf-8")
        self.assertIn('"--throughput-scan"', source)
        self.assertIn('"status": "throughput_scan_complete"', source)
        self.assertIn("not args.throughput_scan", source)


if __name__ == "__main__":
    unittest.main()

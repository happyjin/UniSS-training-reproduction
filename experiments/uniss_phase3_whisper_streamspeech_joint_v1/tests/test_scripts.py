from __future__ import annotations

import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT = ROOT / "experiments/uniss_phase3_whisper_streamspeech_joint_v1"


class ScriptTest(unittest.TestCase):
    def test_all_shell_scripts_parse(self) -> None:
        scripts = sorted((EXPERIMENT / "scripts").glob("*.sh"))
        self.assertGreaterEqual(len(scripts), 10)
        for script in scripts:
            subprocess.run(["bash", "-n", str(script)], check=True)

    def test_runner_is_megatron_and_non_overwriting(self) -> None:
        runner = (EXPERIMENT / "scripts/run_megatron_8gpu.sh").read_text()
        self.assertIn("pretrain_joint_megatron.py", runner)
        self.assertIn("torch.distributed.run", runner)
        self.assertIn("refuse_existing", runner)
        self.assertIn("--global-batch-size", runner)
        self.assertIn("--joint-phase3-replay-weight 0.5", runner)

    def test_historical_entrypoints_are_not_referenced_as_outputs(self) -> None:
        env = (EXPERIMENT / "experiment.env").read_text()
        self.assertIn("phase3_whisper_streamspeech_joint_v1", env)
        self.assertNotIn("checkpoints/uniss_qwen0p5b_phase3_unist198_after_phase2_v4", env)


if __name__ == "__main__":
    unittest.main()

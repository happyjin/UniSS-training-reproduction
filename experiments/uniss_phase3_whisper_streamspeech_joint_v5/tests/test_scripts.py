from __future__ import annotations

import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT = ROOT / "experiments/uniss_phase3_whisper_streamspeech_joint_v5"


class ScriptTest(unittest.TestCase):
    def test_all_shell_scripts_parse(self) -> None:
        scripts = sorted((EXPERIMENT / "scripts").glob("*.sh"))
        self.assertGreaterEqual(len(scripts), 8)
        for script in scripts:
            subprocess.run(["bash", "-n", str(script)], check=True)

    def test_v5_is_isolated_and_guarded(self) -> None:
        runner = (EXPERIMENT / "scripts/run_megatron_8gpu.sh").read_text()
        self.assertIn("uniss_phase3_whisper_streamspeech_joint_v5", runner)
        self.assertIn("--joint-freeze-whisper-codebook", runner)
        self.assertIn("--joint-bridge-surrogate topk_soft", runner)
        self.assertIn("--joint-bridge-commitment-weight 0.25", runner)
        self.assertIn("--joint-max-bridge-commitment 5.0", runner)
        self.assertIn("--lr \"${BASE_LR:-2e-5}\"", runner)
        self.assertIn("refuse_existing", runner)

    def test_pilot_uses_exactly_first_fifteen_stage_a_shards(self) -> None:
        prepare = (EXPERIMENT / "scripts/prepare_15shard_joint_manifest.sh").read_text()
        self.assertIn("seq 0 14", prepare)
        self.assertIn("train-%05d", prepare)
        self.assertIn("build_joint_manifests_parallel", prepare)
        self.assertIn("PILOT_REPLAY_RECORDS:-100000", prepare)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
EXPERIMENT = ROOT / "experiments/uniss_phase3_whisper_streamspeech_joint_v6/full198"


class Full198ScriptTest(unittest.TestCase):
    def test_shell_scripts_parse(self) -> None:
        scripts = sorted((EXPERIMENT / "scripts").glob("*.sh"))
        self.assertGreaterEqual(len(scripts), 8)
        for script in scripts:
            subprocess.run(["bash", "-n", str(script)], check=True)

    def test_full198_data_is_complete(self) -> None:
        data = ROOT / "data/processed/phase3_whisper_streamspeech_joint_v1/full198_joint"
        self.assertEqual((data / "joint_train.jsonl").stat().st_size, 66_077_912_812)
        self.assertTrue((data / "joint_valid.jsonl").is_file())
        self.assertTrue((data / "tokenizer_maps/ctc_qwen_cmn.json").is_file())
        self.assertTrue((data / "tokenizer_maps/ctc_qwen_eng.json").is_file())

    def test_formal_schedule_and_isolation(self) -> None:
        env = (EXPERIMENT / "experiment.env").read_text()
        stage_a = (EXPERIMENT / "scripts/run_stage_a.sh").read_text()
        stage_b = (EXPERIMENT / "scripts/run_stage_b.sh").read_text()
        pipeline = (EXPERIMENT / "scripts/run_pipeline.sh").read_text()
        self.assertIn('MICRO_BATCH_SIZE="${_FULL198_REQUESTED_MICRO_BATCH_SIZE:-2}"', env)
        self.assertIn("phase3_joint_v6_stage_a_heads_only_full198_v1", env)
        self.assertIn("phase3_joint_v6_stage_b_guarded_joint_full198_v1", env)
        self.assertIn('TRAIN_ITERS="${TRAIN_ITERS:-500}"', stage_a)
        self.assertIn('TRAIN_ITERS="${TRAIN_ITERS:-9075}"', stage_b)
        self.assertIn("stage_a_env.sh", stage_a)
        self.assertIn("stage_b_env.sh", stage_b)
        self.assertIn('bash "${SCRIPT_ROOT}/run_stage_a.sh"', pipeline)
        self.assertIn('bash "${SCRIPT_ROOT}/run_stage_b.sh"', pipeline)

    def test_stage_b_loads_only_full198_stage_a(self) -> None:
        stage_b = (EXPERIMENT / "scripts/run_stage_b.sh").read_text()
        self.assertIn("${STAGE_A_RUN_NAME}", stage_b)
        self.assertNotIn("15shard", stage_b)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT = ROOT / "experiments/uniss_phase3_whisper_streamspeech_joint_v6"


class ScriptTest(unittest.TestCase):
    def test_all_shell_scripts_parse(self) -> None:
        scripts = sorted((EXPERIMENT / "scripts").glob("*.sh"))
        self.assertGreaterEqual(len(scripts), 10)
        for script in scripts:
            subprocess.run(["bash", "-n", str(script)], check=True)

    def test_v6_is_isolated_and_teacher_guarded(self) -> None:
        runner = (EXPERIMENT / "scripts/run_stage_8gpu.sh").read_text()
        self.assertIn("uniss_phase3_whisper_streamspeech_joint_v6", runner)
        self.assertIn("--joint-teacher-glm-ce-weight", runner)
        self.assertIn("--joint-teacher-glm-commitment-weight", runner)
        self.assertIn("--joint-whisper-quantize-weight", runner)
        self.assertIn("--joint-max-bridge-commitment-ratio", runner)
        self.assertIn("refuse_existing", runner)

    def test_stages_freeze_then_unfreeze_one_layer(self) -> None:
        stage_a = (EXPERIMENT / "scripts/stage_a_env.sh").read_text()
        stage_b = (EXPERIMENT / "scripts/stage_b_env.sh").read_text()
        self.assertIn("FREEZE_WHISPER=1", stage_a)
        self.assertIn("FREEZE_QWEN=1", stage_a)
        self.assertIn("TRAINABLE_WHISPER_LAYERS=1", stage_b)
        self.assertIn("WHISPER_QUANTIZE_WEIGHT=1", stage_b)
        self.assertIn("BRIDGE_GRADIENT_SCALE=0.1", stage_b)

    def test_validation_balancing_is_disabled_only_for_single_direction_smoke(self) -> None:
        runner = (EXPERIMENT / "scripts/run_stage_8gpu.sh").read_text()
        self.assertIn('[[ "${BALANCE_VALIDATION}" == "1" ]]', runner)
        self.assertNotIn(
            "EXTRA_ARGS=(--joint-allow-partial-replay-index --joint-balance-validation)",
            runner,
        )
        for name in ("run_stage_a_15shard.sh", "run_stage_b_15shard.sh"):
            script = (EXPERIMENT / f"scripts/{name}").read_text()
            self.assertIn("BALANCE_VALIDATION=1", script)
        for name in ("run_stage_a_smoke.sh", "run_stage_b_smoke.sh"):
            script = (EXPERIMENT / f"scripts/{name}").read_text()
            self.assertIn("BALANCE_VALIDATION=0", script)

    def test_stage_a_measures_baseline_and_stage_b_enforces_guard(self) -> None:
        runner = (EXPERIMENT / "scripts/run_stage_8gpu.sh").read_text()
        stage_a = (EXPERIMENT / "scripts/stage_a_env.sh").read_text()
        stage_b = (EXPERIMENT / "scripts/stage_b_env.sh").read_text()
        self.assertIn('[[ -n "${MAX_COMMITMENT:-}" ]]', runner)
        self.assertIn("MAX_COMMITMENT=\n", stage_a)
        self.assertIn("MAX_COMMITMENT_RATIO=\n", stage_a)
        self.assertIn("MAX_COMMITMENT=0.10", stage_b)
        self.assertIn("MAX_COMMITMENT_RATIO=3.0", stage_b)
        self.assertIn("GUARD_CONSECUTIVE_VIOLATIONS=8", stage_b)
        self.assertIn("--joint-bridge-guard-relative-consecutive-violations", runner)


if __name__ == "__main__":
    unittest.main()

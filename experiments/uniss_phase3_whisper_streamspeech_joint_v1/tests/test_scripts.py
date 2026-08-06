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
        self.assertIn("--bf16", runner)
        self.assertIn("--joint-phase3-replay-weight 0.5", runner)

    def test_full198_pipeline_waits_for_all_parts_before_megatron(self) -> None:
        pipeline = (EXPERIMENT / "scripts/wait_and_train_full198.sh").read_text()
        self.assertIn("STAGE_A_SOURCE_PART_COMPLETE.json", pipeline)
        self.assertIn("prepare_full198_joint_manifest.sh", pipeline)
        self.assertIn("run_full198_8gpu.sh", pipeline)
        self.assertLess(
            pipeline.index("prepare_full198_joint_manifest.sh"),
            pipeline.index("run_full198_8gpu.sh"),
        )

    def test_high_throughput_stage_a_uses_disjoint_exact_decode_workers(self) -> None:
        launcher = (
            EXPERIMENT / "scripts/launch_full198_stage_a_high_throughput_tmux.sh"
        ).read_text()
        worker = (
            EXPERIMENT / "scripts/prepare_full198_stage_a_parallel_worker.sh"
        ).read_text()
        self.assertIn("WORKERS_PER_GPU", launcher)
        self.assertIn("TOTAL_WORKERS", launcher)
        self.assertIn("shard=WORKER_ID", worker)
        self.assertIn("shard+=TOTAL_WORKERS", worker)
        self.assertIn("stage_a prepare-part", worker)
        self.assertNotIn("decode_batch", worker)

    def test_historical_entrypoints_are_not_referenced_as_outputs(self) -> None:
        env = (EXPERIMENT / "experiment.env").read_text()
        self.assertIn("phase3_whisper_streamspeech_joint_v1", env)
        self.assertNotIn("checkpoints/uniss_qwen0p5b_phase3_unist198_after_phase2_v4", env)


if __name__ == "__main__":
    unittest.main()

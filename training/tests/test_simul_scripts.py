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
        output = run_script(
            "scripts/simul_uniss/prepare_bootstrap_15shard.sh", "--dry-run"
        )
        self.assertEqual(output.count("train-000"), 15)
        self.assertIn("train-00000.parquet", output)
        self.assertIn("train-00014.parquet", output)
        self.assertNotIn("train-00015.parquet", output)
        self.assertIn("simul_uniss_v1/bootstrap_15shard", output)

    def test_qwen_stages_are_isolated_and_have_validation(self) -> None:
        action = run_script(
            "scripts/simul_uniss/train_qwen_stage.sh",
            "--stage",
            "action",
            "--dry-run",
            "--smoke",
        )
        interleaved = run_script(
            "scripts/simul_uniss/train_qwen_stage.sh",
            "--stage",
            "interleaved",
            "--dry-run",
            "--smoke",
        )
        joint = run_script(
            "scripts/simul_uniss/train_qwen_stage.sh",
            "--stage",
            "joint",
            "--dry-run",
            "--smoke",
        )
        for output in (action, interleaved, joint):
            self.assertIn("pretrain_simul_uniss_megatron.py", output)
            self.assertIn("--simul-packed-valid", output)
            self.assertIn("--eval-iters 1", output)
            self.assertIn("--dataloader-type cyclic", output)
            self.assertIn("--seed 20260722", output)
            self.assertIn("--log-validation-ppl-to-tensorboard", output)
            self.assertIn("checkpoints/simul_uniss_v1", output)
            self.assertNotIn(
                "--save /opt/dlami/nvme/jasonleeeli/projects/UniSS/checkpoints/uniss_qwen0p5b_phase1",
                output,
            )
        self.assertIn("packed_action_train.jsonl", action)
        self.assertIn("stage3_action", action)
        self.assertIn("packed_train.jsonl", interleaved)
        self.assertIn("stage4_interleaved", interleaved)
        self.assertIn("stage6_joint", joint)
        self.assertIn("--lr 3e-6", joint)

    def test_qwen_stage_restores_pip_nvidia_library_paths(self) -> None:
        script = (REPO_ROOT / "scripts/simul_uniss/train_qwen_stage.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn("nvidia/*/lib", script)
        self.assertIn("libcudnn_graph.so.9", script)
        self.assertIn("transformer_engine.pytorch", script)

    def test_iterable_stages_use_bounded_shuffle(self) -> None:
        for relative_path in (
            "scripts/simul_uniss/train_stage7_grpo.sh",
            "scripts/simul_uniss/train_stage8_nar.sh",
        ):
            script = (REPO_ROOT / relative_path).read_text(encoding="utf-8")
            self.assertIn("--shuffle-buffer-size", script)
            self.assertIn("SIMUL_ITERABLE_SHUFFLE_BUFFER_SIZE", script)
            self.assertIn("--seed", script)

    def test_gpu_smoke_pipeline_covers_real_components(self) -> None:
        output = run_script(
            "scripts/simul_uniss/run_gpu_smoke_pipeline.sh", "--dry-run"
        )
        self.assertIn("run_stage0_prefix_baseline.sh", output)
        self.assertIn("train_stage1_audio_student.sh", output)
        self.assertIn("train_streaming_student", output)
        self.assertIn("--stage action", output)
        self.assertIn("--stage interleaved", output)
        self.assertIn("--stage joint", output)
        self.assertIn("train_stage5_bicodec_refinement.sh", output)
        self.assertIn("--decoder bicodec", output)

    def test_short_training_pipeline_covers_all_training_stages(self) -> None:
        output = run_script(
            "scripts/simul_uniss/run_short_training_pipeline.sh", "--dry-run"
        )
        for expected in (
            "token streaming student",
            "audio streaming student",
            "action Qwen",
            "interleaved Qwen",
            "low-LR joint Qwen",
            "BiCodec boundary refinement",
            "GRPO policy",
            "NAR semantic generator",
            "real BiCodec streaming replay",
        ):
            self.assertIn(expected, output)

    def test_action_preparation_publishes_a_completion_marker(self) -> None:
        output = run_script("scripts/simul_uniss/prepare_action_data.sh", "--dry-run")
        self.assertIn("packed_action_train.jsonl", output)
        self.assertIn("ACTION_PREPARE_COMPLETE", output)
        self.assertIn("atomically publish", output)

    def test_gpu_launcher_does_not_match_its_own_command_line(self) -> None:
        output = run_script(
            "scripts/simul_uniss/launch_gpu_smoke_when_ready.sh", "--dry-run"
        )
        self.assertIn("iteration >= 15465", output)
        self.assertIn("[p]ython", output)
        self.assertIn("run_gpu_smoke_pipeline.sh", output)

    def test_short_training_launcher_waits_for_durable_markers(self) -> None:
        output = run_script(
            "scripts/simul_uniss/launch_short_training_when_ready.sh", "--dry-run"
        )
        self.assertIn("GPU_SMOKE_COMPLETE", output)
        self.assertIn("ACTION_PREPARE_COMPLETE", output)
        self.assertIn("run_short_training_pipeline.sh", output)


if __name__ == "__main__":
    unittest.main()

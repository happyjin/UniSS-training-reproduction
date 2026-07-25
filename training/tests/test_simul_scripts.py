from __future__ import annotations

import os
import subprocess
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def run_script(path: str, *args: str, extra_env: dict[str, str] | None = None) -> str:
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    result = subprocess.run(
        [str(REPO_ROOT / path), *args],
        cwd=REPO_ROOT,
        env=env,
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

    def test_qwen_stage_supports_opt_in_global_shuffle_and_full_validation(self) -> None:
        output = run_script(
            "scripts/simul_uniss/train_qwen_stage.sh",
            "--stage",
            "action",
            "--dry-run",
            "--smoke",
            extra_env={
                "SIMUL_NO_DATA_SHARDING": "1",
                "SIMUL_FULL_VALIDATION": "1",
            },
        )
        self.assertIn("--dataloader-type cyclic", output)
        self.assertIn("--no-data-sharding", output)
        self.assertIn("--full-validation", output)

    def test_historical_qwen_config_keeps_original_data_sharding_default(self) -> None:
        output = run_script(
            "scripts/simul_uniss/train_qwen_stage.sh",
            "--stage",
            "action",
            "--dry-run",
            "--smoke",
        )
        self.assertNotIn("--no-data-sharding", output)

    def test_v2_qwen_stages_are_isolated_eight_gpu_global_shuffle_runs(self) -> None:
        experiment = "experiments/simul_uniss_v2_15shard"
        expected_loads = {
            "stage03_action_sft": "checkpoints/uniss_qwen0p5b_phase3_unist198_after_phase2_v4",
            "stage04_interleaved_s2st": "checkpoints/simul_uniss_v2_15shard/stage03_action_sft",
            "stage06_joint_refinement": "checkpoints/simul_uniss_v2_15shard/stage04_interleaved_s2st",
        }
        for stage, expected_load in expected_loads.items():
            output = run_script(f"{experiment}/{stage}/run.sh", "--dry-run")
            self.assertIn("--nproc_per_node 8", output)
            self.assertIn("--dataloader-type cyclic", output)
            self.assertIn("--no-data-sharding", output)
            self.assertIn("--full-validation", output)
            self.assertIn("--seed 20260725", output)
            self.assertIn("checkpoints/simul_uniss_v2_15shard", output)
            self.assertIn(expected_load, output)
            self.assertNotIn("checkpoints/simul_uniss_v1/stage", output)

    def test_v2_stage1_token_student_uses_eight_gpu_torchrun(self) -> None:
        output = run_script(
            "experiments/simul_uniss_v2_15shard/stage01_02_streaming_student/run_token_8gpu.sh",
            "--dry-run",
        )
        self.assertIn("--nproc_per_node 8", output)
        self.assertIn("--master_addr 127.0.0.1", output)
        self.assertIn("training.simul_uniss.train_streaming_student", output)
        self.assertIn("--device cuda", output)
        self.assertIn("stage01_02_streaming_token_student", output)
        self.assertIn("--seed 20260725", output)

    def test_v2_stage0_and_audio_student_are_isolated(self) -> None:
        prepare = run_script(
            "experiments/simul_uniss_v2_15shard/stage00_baselines/prepare_audio.sh",
            "--dry-run",
        )
        self.assertIn("--limit-records 1000", prepare)
        self.assertIn("simul_uniss_v2_15shard/stage00_audio_reconstruction", prepare)
        audio = run_script(
            "experiments/simul_uniss_v2_15shard/stage01_02_streaming_student/run_audio_8gpu.sh",
            "--dry-run",
        )
        self.assertIn("--nproc_per_node 8", audio)
        self.assertIn("--master_addr 127.0.0.1", audio)
        self.assertIn("training.simul_uniss.train_audio_student", audio)
        self.assertIn("--device cuda", audio)
        self.assertIn("stage01_02_streaming_audio_student", audio)
        self.assertIn("--seed 20260725", audio)

    def test_v2_stage5_refinement_uses_eight_gpu_torchrun(self) -> None:
        output = run_script(
            "experiments/simul_uniss_v2_15shard/stage05_streaming_bicodec/run_refinement_8gpu.sh",
            "--dry-run",
        )
        self.assertIn("--nproc_per_node 8", output)
        self.assertIn("--master_addr 127.0.0.1", output)
        self.assertIn("training.simul_uniss.train_bicodec_refinement", output)
        self.assertIn("--device cuda", output)
        self.assertIn("stage05_bicodec_refinement", output)
        self.assertIn("--seed 20260725", output)

    def test_v2_stage7_grpo_bootstrap_uses_eight_gpu_torchrun(self) -> None:
        output = run_script(
            "experiments/simul_uniss_v2_15shard/stage07_grpo/run_8gpu.sh",
            "--dry-run",
        )
        self.assertIn("--nproc_per_node 8", output)
        self.assertIn("--master_addr 127.0.0.1", output)
        self.assertIn("training.simul_uniss.policy_grpo", output)
        self.assertIn("--shuffle-buffer-size 8192", output)
        self.assertIn("stage07_grpo_policy_bootstrap", output)
        self.assertIn("--seed 20260725", output)

    def test_v2_stage8_nar_uses_eight_gpu_torchrun(self) -> None:
        output = run_script(
            "experiments/simul_uniss_v2_15shard/stage08_nar_optional/run_8gpu.sh",
            "--dry-run",
        )
        self.assertIn("--nproc_per_node 8", output)
        self.assertIn("--master_addr 127.0.0.1", output)
        self.assertIn("training.simul_uniss.nar_semantic", output)
        self.assertIn("--shuffle-buffer-size 8192", output)
        self.assertIn("stage08_nar_semantic_optional", output)
        self.assertIn("--seed 20260725", output)

    def test_v2_component_pipeline_covers_required_non_qwen_stages(self) -> None:
        output = run_script(
            "experiments/simul_uniss_v2_15shard/orchestration/run_component_pipeline_8gpu.sh",
            "--dry-run",
        )
        self.assertIn("reconstruct_unist_audio", output)
        self.assertIn("prefix_reencode_baseline", output)
        self.assertIn("train_streaming_student", output)
        self.assertIn("train_audio_student", output)
        self.assertIn("run_stage5_streaming_replay", output)
        self.assertIn("train_bicodec_refinement", output)
        self.assertIn("policy_grpo", output)
        self.assertIn("stage08_nar_optional=profiling-gated-not-auto-started", output)

    def test_v2_shuffle_smoke_dry_run_covers_all_qwen_stages(self) -> None:
        output = run_script(
            "experiments/simul_uniss_v2_15shard/orchestration/run_shuffle_smoke_8gpu.sh",
            "--dry-run",
        )
        self.assertEqual(output.count("--nproc_per_node 8"), 3)
        self.assertEqual(output.count("--no-data-sharding"), 3)
        self.assertEqual(output.count("--full-validation"), 3)
        self.assertEqual(output.count("--lr-warmup-iters 0"), 3)
        self.assertIn("shuffle_smoke_8gpu_v2/stage03_action_sft", output)
        self.assertIn("shuffle_smoke_8gpu_v2/stage04_interleaved_s2st", output)
        self.assertIn("shuffle_smoke_8gpu_v2/stage06_joint_refinement", output)

    def test_v2_shuffle_smoke_has_safe_existing_result_verification(self) -> None:
        script = (
            REPO_ROOT
            / "experiments/simul_uniss_v2_15shard/orchestration/run_shuffle_smoke_8gpu.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("--verify-existing", script)
        self.assertIn("verify_stage stage03", script)
        self.assertIn("verify_stage stage04", script)
        self.assertIn("verify_stage stage06", script)
        self.assertIn("write_complete_marker", script)

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

    def test_historical_non_megatron_stages_keep_validation_opt_in(self) -> None:
        audio = (REPO_ROOT / "training/simul_uniss/train_audio_student.py").read_text(
            encoding="utf-8"
        )
        bicodec = (
            REPO_ROOT / "training/simul_uniss/train_bicodec_refinement.py"
        ).read_text(encoding="utf-8")
        for source in (audio, bicodec):
            self.assertIn('default=0,', source)
            self.assertIn("preserves historical all-train behavior", source)

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

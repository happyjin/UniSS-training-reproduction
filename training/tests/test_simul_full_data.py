from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
import unittest
from array import array
from pathlib import Path
from unittest.mock import patch

from training.simul_uniss import PACKED_SCHEMA_VERSION
from training.simul_uniss.full_data_pipeline import (
    assemble,
    generate_schedule,
    mark_packed_part,
    mark_prepared_part,
    verify_packed_part,
    verify_prepared_part,
)
from training.simul_uniss.dataset import SimulPackedJsonlDataset
from training.simul_uniss.jsonl_index import load_index
from training.simul_uniss import prepare_data
from training.simul_uniss.schema import sha256_file

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


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


class FullDataPipelineTests(unittest.TestCase):
    def test_full_prepare_can_skip_invalid_rows_without_changing_strict_default(self) -> None:
        valid = {
            "id": "valid",
            "transcription": "hello",
            "translation": "world",
            "source_glm": list(range(16)),
            "source_bicodec": list(range(64)),
            "target_bicodec": list(range(48)),
            "bicodec_global": list(range(32)),
            "src_lang": "eng",
            "tgt_lang": "eng",
        }
        invalid = {**valid, "id": "invalid", "target_bicodec": []}
        with tempfile.TemporaryDirectory() as value:
            root = Path(value)
            source = root / "input.parquet"
            source.write_bytes(b"fixture")
            output = root / "output"
            argv = [
                "prepare_data",
                "--input",
                str(source),
                "--output-dir",
                str(output),
                "--tokenizer",
                str(root / "tokenizer"),
                "--skip-sha256",
                "--skip-invalid-records",
            ]
            with (
                patch.object(prepare_data, "iter_raw_records", return_value=iter((valid, invalid))),
                patch.object(prepare_data, "load_text_encoder", return_value=lambda text: [1, 2]),
                patch("sys.argv", argv),
            ):
                prepare_data.main()
            stats = json.loads((output / "stats.json").read_text(encoding="utf-8"))
            self.assertEqual(stats["records"], 1)
            self.assertEqual(stats["input_records"], 2)
            self.assertEqual(stats["skipped_invalid_records"], 1)
            self.assertIn("must be non-empty", next(iter(stats["invalid_reasons"])))

            strict_output = root / "strict"
            strict_argv = [
                "prepare_data",
                "--input",
                str(source),
                "--output-dir",
                str(strict_output),
                "--tokenizer",
                str(root / "tokenizer"),
                "--skip-sha256",
            ]
            with (
                patch.object(prepare_data, "iter_raw_records", return_value=iter((invalid,))),
                patch.object(prepare_data, "load_text_encoder", return_value=lambda text: [1, 2]),
                patch("sys.argv", strict_argv),
                self.assertRaisesRegex(ValueError, "must be non-empty"),
            ):
                prepare_data.main()

    def create_parts(self, root: Path, index: int) -> tuple[Path, Path, Path]:
        source = root / f"train-{index:05d}.parquet"
        source.write_bytes(f"source-{index}".encode())
        temporary_prepared = root / f".prepared-{index}"
        temporary_prepared.mkdir()
        (temporary_prepared / "schedules.jsonl").write_text('{"id":"a"}\n', encoding="utf-8")
        (temporary_prepared / "samples.jsonl").write_text('{"id":"a"}\n', encoding="utf-8")
        write_json(
            temporary_prepared / "manifest.json",
            {
                "shards": [
                    {
                        "path": str(source.resolve()),
                        "size_bytes": source.stat().st_size,
                        "sha256": sha256_file(source),
                    }
                ]
            },
        )
        write_json(
            temporary_prepared / "stats.json",
            {"records": 1, "events": 2, "wait_events": 1, "write_events": 1},
        )
        prepared = root / "prepared" / f"train-{index:05d}"
        mark_prepared_part(source, temporary_prepared, index, published_dir=prepared)
        prepared.parent.mkdir(exist_ok=True)
        temporary_prepared.rename(prepared)
        verify_prepared_part(source, prepared, index)
        stats = json.loads((prepared / "stats.json").read_text(encoding="utf-8"))
        self.assertEqual(stats["schedules"], str((prepared / "schedules.jsonl").resolve()))

        temporary_packed = root / f".packed-{index}"
        temporary_packed.mkdir()
        line = json.dumps({"schema_version": PACKED_SCHEMA_VERSION}) + "\n"
        (temporary_packed / "packed_interleaved.jsonl").write_text(line, encoding="utf-8")
        (temporary_packed / "packed_action.jsonl").write_text(line, encoding="utf-8")
        mark_packed_part(prepared, temporary_packed, index)
        packed = root / "packed" / f"train-{index:05d}"
        packed.parent.mkdir(exist_ok=True)
        temporary_packed.rename(packed)
        verify_packed_part(prepared, packed, index)
        return source, prepared, packed

    def test_atomic_part_markers_survive_directory_publish(self) -> None:
        with tempfile.TemporaryDirectory() as value:
            root = Path(value)
            self.create_parts(root, 0)

    def test_assembly_and_schedule_use_real_packed_counts(self) -> None:
        with tempfile.TemporaryDirectory() as value:
            root = Path(value)
            self.create_parts(root, 0)
            self.create_parts(root, 1)
            output = root / "output"
            marker = output / "DATA_ASSEMBLY_COMPLETE.json"
            assemble(
                argparse.Namespace(
                    prepared_parts=str(root / "prepared"),
                    packed_parts=str(root / "packed"),
                    shard_start=0,
                    shard_count=2,
                    schedules_output=str(output / "schedules.jsonl"),
                    interleaved_output=str(output / "interleaved.jsonl"),
                    action_output=str(output / "action.jsonl"),
                    manifest_output=str(output / "manifest.json"),
                    marker_output=str(marker),
                )
            )
            schedule = output / "training_schedule.env"
            values = generate_schedule(
                argparse.Namespace(
                    assembly_marker=str(marker),
                    output=str(schedule),
                    global_batch_size=2,
                    stage3_epochs="1.0",
                    stage4_epochs="1.0",
                    stage6_epochs="0.5",
                    warmup_fraction=0.05,
                )
            )
            self.assertEqual(values["STAGE3_TRAIN_ITERS"], 1)
            self.assertEqual(values["STAGE4_TRAIN_ITERS"], 1)
            self.assertEqual(values["STAGE6_TRAIN_ITERS"], 1)
            self.assertIn('STAGE3_TRAIN_ITERS="${STAGE3_TRAIN_ITERS:-1}"', schedule.read_text())
            dataset = SimulPackedJsonlDataset(output / "interleaved.jsonl", 4096)
            self.assertIsInstance(dataset.offsets, array)
            self.assertEqual(len(dataset), 2)
            self.assertEqual(len(load_index(output / "schedules.jsonl") or []), 2)


class FullDataScriptTests(unittest.TestCase):
    def test_full_preparation_dry_run_spans_all_198_shards(self) -> None:
        output = run_script(
            "experiments/simul_uniss_v3_full198/data_preparation/run_full_preparation.sh",
            "--dry-run",
        )
        self.assertIn("shard_count=198", output)
        self.assertIn("train-00000.parquet", output)
        self.assertIn("train-00197.parquet", output)
        self.assertIn("full_data_pipeline assemble", output)
        self.assertIn("training_schedule.env", output)
        self.assertIn("--skip-invalid-records", output)

    def test_full_qwen_stages_reuse_reader_with_isolated_paths(self) -> None:
        experiment = "experiments/simul_uniss_v3_full198"
        stage_values = {
            "stage03_action_sft": ("7", "packed_action_train.jsonl", "3"),
            "stage04_interleaved_s2st": ("8", "packed_interleaved_train.jsonl", "4"),
            "stage06_joint_refinement": ("9", "packed_interleaved_train.jsonl", "5"),
        }
        for stage, (iterations, data_name, warmup) in stage_values.items():
            number = {"stage03_action_sft": "3", "stage04_interleaved_s2st": "4", "stage06_joint_refinement": "6"}[stage]
            output = run_script(
                f"{experiment}/{stage}/run.sh",
                "--dry-run",
                extra_env={
                    f"STAGE{number}_TRAIN_ITERS": iterations,
                    f"STAGE{number}_QWEN_WARMUP_ITERS": warmup,
                },
            )
            self.assertIn("--nproc_per_node 8", output)
            self.assertIn(f"--train-iters {iterations}", output)
            self.assertIn(f"--lr-warmup-iters {warmup}", output)
            self.assertIn(data_name, output)
            self.assertIn("simul_uniss_v3_full198", output)
            self.assertIn("--no-data-sharding", output)
            self.assertIn("--full-validation", output)

    def test_full_stage0_is_stratified_across_all_shards(self) -> None:
        output = run_script(
            "experiments/simul_uniss_v3_full198/stage00_baselines/prepare_audio.sh",
            "--dry-run",
        )
        self.assertEqual(output.count("train-"), 198)
        self.assertIn("--records-per-shard 5", output)
        self.assertIn("--limit-records 990", output)

    def test_full_component_pipeline_keeps_stage8_gated(self) -> None:
        output = run_script(
            "experiments/simul_uniss_v3_full198/orchestration/run_component_pipeline_8gpu.sh",
            "--dry-run",
        )
        self.assertIn("--nproc_per_node 8", output)
        self.assertIn("stage08_nar_optional=profiling-gated-not-auto-started", output)
        self.assertIn("--shuffle-buffer-size 65536", output)

    def test_full_training_waiter_enforces_safe_launch_order(self) -> None:
        output = run_script(
            "experiments/simul_uniss_v3_full198/orchestration/launch_training_when_ready.sh",
            "--dry-run",
        )
        ready = output.index("FULL_DATA_READY.json")
        smoke = output.index("run_shuffle_smoke_8gpu.sh")
        tensorboard = output.index("start_tensorboard.sh")
        qwen = output.index("launch_qwen_pipeline_tmux.sh")
        components = output.index("launch_component_pipeline_when_ready.sh")
        self.assertLess(ready, smoke)
        self.assertLess(smoke, tensorboard)
        self.assertLess(tensorboard, qwen)
        self.assertLess(qwen, components)
        self.assertNotIn("stage08_nar_optional", output)
        source = (
            REPO_ROOT
            / "experiments/simul_uniss_v3_full198/orchestration/launch_training_when_ready.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("printf -v command '{\\n%s\\n} 2>&1 | tee -a %q'", source)

    def test_gbs128_restart_is_isolated_and_uses_phase3_batch_geometry(self) -> None:
        experiment = "experiments/simul_uniss_v4_full198_gbs128"
        schedule = run_script(f"{experiment}/data_preparation/generate_schedule.sh", "--dry-run")
        self.assertIn("--global-batch-size 128", schedule)
        self.assertIn("simul_uniss_v4_full198_gbs128/training_schedule.env", schedule)
        output = run_script(
            f"{experiment}/stage03_action_sft/run.sh",
            "--dry-run",
            extra_env={"STAGE3_TRAIN_ITERS": "22652", "STAGE3_QWEN_WARMUP_ITERS": "1133"},
        )
        self.assertIn("--nproc_per_node 8", output)
        self.assertIn("--micro-batch-size 2", output)
        self.assertIn("--global-batch-size 128", output)
        self.assertIn("--train-iters 22652", output)
        self.assertIn("--lr-warmup-iters 1133", output)
        self.assertIn("data/megatron/simul_uniss_v3_full198/packed_action_train.jsonl", output)
        self.assertIn("checkpoints/simul_uniss_v4_full198_gbs128/stage03_action_sft", output)
        self.assertNotIn("checkpoints/simul_uniss_v3_full198/stage03_action_sft", output)

    def test_gbs128_smoke_covers_all_qwen_stages_without_old_outputs(self) -> None:
        output = run_script(
            "experiments/simul_uniss_v4_full198_gbs128/orchestration/run_shuffle_smoke_8gpu.sh",
            "--dry-run",
            extra_env={
                "STAGE3_TRAIN_ITERS": "22652",
                "STAGE4_TRAIN_ITERS": "22652",
                "STAGE6_TRAIN_ITERS": "5663",
                "STAGE3_QWEN_WARMUP_ITERS": "1133",
                "STAGE4_QWEN_WARMUP_ITERS": "1133",
                "STAGE6_QWEN_WARMUP_ITERS": "284",
            },
        )
        self.assertEqual(output.count("--nproc_per_node 8"), 3)
        self.assertEqual(output.count("--micro-batch-size 2"), 3)
        self.assertEqual(output.count("--global-batch-size 128"), 3)
        self.assertIn("simul_uniss_v4_full198_gbs128", output)
        self.assertNotIn("checkpoints/simul_uniss_v3_full198/stage03_action_sft", output)
        pipeline = (
            REPO_ROOT
            / "experiments/simul_uniss_v4_full198_gbs128/orchestration/run_qwen_pipeline_8gpu.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("Skipping verified completed", pipeline)
        self.assertIn("Refusing existing stage output", pipeline)

    def test_mbs4_stage3_utilization_run_is_isolated_and_logs_less_often(self) -> None:
        experiment = "experiments/simul_uniss_v5_full198_mbs4_gbs128_stage3"
        output = run_script(
            f"{experiment}/stage03_action_sft/run.sh",
            "--dry-run",
            extra_env={"STAGE3_TRAIN_ITERS": "22652", "STAGE3_QWEN_WARMUP_ITERS": "1133"},
        )
        self.assertIn("--micro-batch-size 4", output)
        self.assertIn("--global-batch-size 128", output)
        self.assertIn("--log-interval 10", output)
        self.assertIn("--tensorboard-log-interval 10", output)
        self.assertIn("--log-memory-interval 10", output)
        self.assertIn("checkpoints/simul_uniss_v5_full198_mbs4_gbs128_stage3", output)
        self.assertNotIn("checkpoints/simul_uniss_v4_full198_gbs128/stage03_action_sft", output)

        historical = run_script(
            "experiments/simul_uniss_v3_full198/stage03_action_sft/run.sh",
            "--dry-run",
            extra_env={"STAGE3_TRAIN_ITERS": "2", "STAGE3_QWEN_WARMUP_ITERS": "0"},
        )
        self.assertIn("--log-interval 1", historical)
        self.assertIn("--tensorboard-log-interval 1", historical)


if __name__ == "__main__":
    unittest.main()

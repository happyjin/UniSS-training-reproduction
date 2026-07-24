import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

from training import validate_packed_jsonl
from training import pack_sequences_parallel


REPO_ROOT = Path(__file__).resolve().parents[2]


def run_script(script: str, *args: str, extra_env: dict[str, str] | None = None) -> str:
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    result = subprocess.run(
        [str(REPO_ROOT / script), *args],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    return result.stdout


class UniST198FullScriptsTest(unittest.TestCase):
    def _packed_record(self, length: int = 6) -> dict[str, object]:
        return {
            "tokens": list(range(length)),
            "labels": list(range(1, length + 1)),
            "loss_mask": [1] * length,
            "position_ids": list(range(length)),
            "sample_boundaries": [[0, length]],
            "tasks": ["quality"],
            "source_ids": ["sample"],
        }

    def test_packed_validator_checks_first_last_and_newline(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "packed.jsonl"
            record = self._packed_record()
            path.write_text(json.dumps(record) + "\n" + json.dumps(record) + "\n", encoding="utf-8")
            result = validate_packed_jsonl.validate_file(path, seq_length=6)
            self.assertEqual(result["status"], "ok")
            self.assertEqual(result["seq_length"], 6)

            path.write_text(json.dumps(record), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "does not end with a newline"):
                validate_packed_jsonl.validate_file(path, seq_length=6)

    def test_runner_dry_run_calculates_schedule_and_uses_eight_gpus(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = run_script(
                "scripts/run_qwen0p5b_unist198_all_phases.sh",
                "--dry-run",
                extra_env={
                    "PHASE1_PACKED_COUNT_OVERRIDE": "256",
                    "PHASE2_PACKED_COUNT_OVERRIDE": "129",
                    "PHASE3_PACKED_COUNT_OVERRIDE": "128",
                    "PHASE1_SAVE": str(Path(tmp) / "phase1"),
                    "PHASE2_SAVE": str(Path(tmp) / "phase2"),
                    "PHASE3_SAVE": str(Path(tmp) / "phase3"),
                },
            )
        self.assertIn("phase1=6/2", output)
        self.assertIn("phase2=2/1", output)
        self.assertIn("phase3=1/0", output)
        self.assertIn(r"CUDA_VISIBLE_DEVICES=0\,1\,2\,3\,4\,5\,6\,7", output)
        self.assertIn("NPROC_PER_NODE=8", output)
        self.assertIn("MICRO_BATCH_SIZE=2", output)
        self.assertEqual(output.count("--attention-backend fused"), 3)
        self.assertIn("TRAIN_ITERS=6", output)
        self.assertIn("TRAIN_ITERS=2", output)
        self.assertIn("TRAIN_ITERS=1", output)
        self.assertIn("tensorboard/phase1", output)
        self.assertIn("tensorboard/phase2", output)
        self.assertIn("tensorboard/phase3", output)
        self.assertIn("uniss_qwen0p5b_phase1_unist198_full_v1", output)
        self.assertIn("uniss_qwen0p5b_phase2_unist198_full_v1", output)
        self.assertIn("uniss_qwen0p5b_phase3_unist198_full_v1", output)
        self.assertNotIn("unist13_full", output)

    def test_runner_can_stop_after_phase1_without_future_packed_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = run_script(
                "scripts/run_qwen0p5b_unist198_all_phases.sh",
                "--dry-run",
                "--start-phase",
                "phase1",
                "--end-phase",
                "phase1",
                extra_env={
                    "PHASE1_PACKED_COUNT_OVERRIDE": "256",
                    "PHASE1_SAVE": str(Path(tmp) / "phase1"),
                },
            )
        self.assertIn("START=phase1, END=phase1", output)
        self.assertIn("phase1=6/2", output)
        self.assertIn("TRAIN_ITERS=6", output)
        self.assertIn("phase2 skipped because END_PHASE=phase1", output)
        self.assertIn("phase3 skipped because END_PHASE=phase1", output)
        self.assertNotIn("Missing packed count sidecar", output)

    def test_phase1_recovery_b_dry_run_is_isolated_shuffled_and_low_lr(self):
        output = run_script(
            "scripts/run_qwen0p5b_unist198_phase1_recovery_b.sh",
            "--dry-run",
        )
        self.assertIn("source_iteration=3300", output)
        self.assertIn("checkpoints/candidates/uniss_qwen0p5b_phase1_unist198_iter3300", output)
        self.assertIn("checkpoints/uniss_qwen0p5b_phase1_unist198_recovery_b1_v2", output)
        self.assertIn("runs/uniss_qwen0p5b_phase1_unist198_recovery_b1_v2/tensorboard", output)
        self.assertIn("TRAIN_ITERS=500", output)
        self.assertIn("--lr 1e-4", output)
        self.assertIn("--min-lr 1e-5", output)
        self.assertIn("--lr-warmup-iters 200", output)
        self.assertIn("--lr-decay-style cosine", output)
        self.assertIn("--lr-decay-iters 500", output)
        self.assertIn("--dataloader-type cyclic", output)
        self.assertIn("--eval-iters 10", output)
        self.assertNotIn("--full-validation", output)
        self.assertIn("--attention-backend fused", output)
        self.assertIn("--finetune", output)
        self.assertIn("--no-load-optim", output)
        self.assertIn("--no-load-rng", output)
        self.assertIn("FINETUNE=1 LOAD_OPTIM=0 LOAD_RNG=0", output)
        self.assertIn(r"CUDA_VISIBLE_DEVICES=0\,1\,2\,3\,4\,5\,6\,7", output)
        self.assertIn("NPROC_PER_NODE=8", output)
        self.assertNotIn("logs/uniss_qwen0p5b_phase1_unist198_full_v1.log", output)

    def test_phase2_recovery_dry_run_is_isolated_shuffled_and_stops_before_phase3(self):
        with tempfile.TemporaryDirectory() as tmp:
            isolated_dirs = {
                "PILOT_SAVE_DIR": str(Path(tmp) / "pilot"),
                "FULL_SAVE_DIR": str(Path(tmp) / "full"),
            }
            pilot = run_script(
                "scripts/run_qwen0p5b_unist198_phase2_recovery_v1.sh",
                "--dry-run",
                "--mode",
                "pilot",
                extra_env=isolated_dirs,
            )
            pipeline = run_script(
                "scripts/run_qwen0p5b_unist198_phase2_recovery_pipeline.sh",
                "--dry-run",
                extra_env=isolated_dirs,
            )
        self.assertIn("mode=pilot", pilot)
        self.assertIn("source_iteration=2300", pilot)
        self.assertIn("--train-iters 500", pilot)
        self.assertIn("--lr 5e-5", pilot)
        self.assertIn("--min-lr 5e-6", pilot)
        self.assertIn("--lr-warmup-iters 100", pilot)
        self.assertIn("--lr-decay-style cosine", pilot)
        self.assertIn("--lr-decay-iters 500", pilot)
        self.assertIn("--dataloader-type cyclic", pilot)
        self.assertIn("--finetune", pilot)
        self.assertIn("--no-load-optim", pilot)
        self.assertIn("--no-load-rng", pilot)
        self.assertNotIn("phase3", pilot.lower())

        self.assertIn("validate pilot TensorBoard through step 500", pipeline)
        self.assertIn("Phase3 remains disabled", pipeline)
        self.assertIn("mode=full", pipeline)
        self.assertIn("--train-iters 15381", pipeline)

    def test_phase2_recovery_v2_uses_global_shuffle_full_validation_and_low_lr(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = run_script(
                "scripts/run_qwen0p5b_unist198_phase2_recovery_v2_pipeline.sh",
                "--dry-run",
                extra_env={
                    "PILOT_SAVE_DIR": str(Path(tmp) / "pilot"),
                    "FULL_SAVE_DIR": str(Path(tmp) / "full"),
                },
            )
        self.assertIn("source_iteration=4600", output)
        self.assertIn("--train-iters 1000", output)
        self.assertIn("--lr 2e-5", output)
        self.assertIn("--min-lr 2e-6", output)
        self.assertIn("--lr-warmup-iters 200", output)
        self.assertIn("--dataloader-type cyclic", output)
        self.assertIn("--no-data-sharding", output)
        self.assertIn("--full-validation", output)
        self.assertIn("--eval-micro-batch-size 1", output)
        self.assertIn("--eval-global-batch-size 8", output)
        self.assertIn("--clip-grad 0.5", output)
        self.assertIn("absolute grad norm <= 20.0", output)
        self.assertIn("full continuation budget=10781", output)
        self.assertIn("effective final iteration=15381", output)
        self.assertIn("--train-iters 10781", output)
        self.assertIn("--nproc_per_node 8", output)

    def test_phase3_waiter_uses_recovered_phase2_and_cyclic_full_data(self):
        output = run_script(
            "scripts/run_qwen0p5b_unist198_phase3_after_phase2_recovery_v1.sh",
            "--dry-run",
        )
        self.assertIn("wait for Phase2 local checkpoint 15381", output)
        self.assertIn("validate final Phase2 TensorBoard/log", output)
        self.assertIn("phase3_unist198/packed_train.jsonl", output)
        self.assertIn("Phase3 packed count=1161587", output)
        self.assertIn("TRAIN_ITERS=9075", output)
        self.assertIn("NPROC_PER_NODE=8", output)
        self.assertIn("MICRO_BATCH_SIZE=2", output)
        self.assertIn("DATALOADER_TYPE=cyclic", output)
        self.assertIn("--dataloader-type cyclic", output)
        self.assertIn("--lr-decay-iters 9075", output)
        self.assertIn("phase3_unist198_from_phase2_recovery_v1", output)
        self.assertIn("port=6010", output)
        self.assertIn(
            "runs/uniss_qwen0p5b_phase3_unist198_from_phase2_recovery_v1/tensorboard",
            output,
        )
        self.assertIn("--load", output)
        self.assertIn("phase2_unist198_recovery_shuffle_lr5e5_v1/full", output)
        self.assertNotIn("uniss_qwen0p5b_phase3_unist198_full_v1", output)

    def test_phase3_v2_waiter_targets_global_shuffle_recovery(self):
        output = run_script(
            "scripts/run_qwen0p5b_unist198_phase3_after_phase2_recovery_v1.sh",
            "--dry-run",
            "--config",
            str(
                REPO_ROOT
                / "configs/experiments/uniss_qwen0p5b_unist198_phase3_after_phase2_recovery_v2.env"
            ),
        )
        self.assertIn("local checkpoint 10781", output)
        self.assertIn("source=4600, effective=15381", output)
        self.assertIn("phase2_unist198_recovery_global_shuffle_lr2e5_v2/full", output)
        self.assertIn("phase3_unist198_from_phase2_recovery_v2", output)
        self.assertIn("--dataloader-type cyclic", output)
        self.assertIn("--no-data-sharding", output)
        self.assertIn("--full-validation", output)
        self.assertIn("--eval-micro-batch-size 1", output)
        self.assertIn("--eval-global-batch-size 8", output)
        self.assertIn("TRAIN_ITERS=9075", output)
        self.assertIn("port=6010", output)

    def test_phase2_recovery_v3_halves_lr_without_early_stop(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = run_script(
                "scripts/run_qwen0p5b_unist198_phase2_recovery_v3_pipeline.sh",
                "--dry-run",
                extra_env={
                    "PILOT_SAVE_DIR": str(Path(tmp) / "pilot"),
                    "FULL_SAVE_DIR": str(Path(tmp) / "full"),
                },
            )
        self.assertIn("source_iteration=4600", output)
        self.assertIn("--lr 1e-5", output)
        self.assertIn("--min-lr 1e-6", output)
        self.assertIn("--lr-warmup-iters 200", output)
        self.assertIn("--clip-grad 0.5", output)
        self.assertIn("--no-data-sharding", output)
        self.assertIn("--full-validation", output)
        self.assertIn("--eval-micro-batch-size 1", output)
        self.assertIn("--eval-global-batch-size 8", output)
        self.assertIn("no early stop", output)
        self.assertIn("--train-iters 10781", output)

    def test_phase3_v3_uses_low_lr_warmup_and_strict_data_path(self):
        output = run_script(
            "scripts/run_qwen0p5b_unist198_phase3_after_phase2_recovery_v1.sh",
            "--dry-run",
            "--config",
            str(
                REPO_ROOT
                / "configs/experiments/uniss_qwen0p5b_unist198_phase3_after_phase2_recovery_v3.env"
            ),
        )
        self.assertIn("local checkpoint 10781", output)
        self.assertIn("phase2_unist198_recovery_global_shuffle_lr1e5_v3/full", output)
        self.assertIn("phase3_unist198_from_phase2_recovery_v3", output)
        self.assertIn("--lr 1e-5", output)
        self.assertIn("--min-lr 1e-6", output)
        self.assertIn("--lr-warmup-iters 200", output)
        self.assertIn("--clip-grad 0.5", output)
        self.assertIn("--no-data-sharding", output)
        self.assertIn("--full-validation", output)
        self.assertIn("port=6013", output)

    def test_phase2_v4_starts_from_phase1_and_preserves_pilot_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = run_script(
                "scripts/run_qwen0p5b_unist198_phase2_from_phase1_v4.sh",
                "--dry-run",
                extra_env={
                    "SAVE_DIR": str(Path(tmp) / "checkpoints"),
                    "RUN_DIR": str(Path(tmp) / "run"),
                    "TENSORBOARD_DIR": str(Path(tmp) / "tensorboard"),
                },
            )
        self.assertIn("clean Phase1 source iteration=15465", output)
        self.assertIn("no Phase2 recovery checkpoint", output)
        self.assertIn("--train-iters 15381", output)
        self.assertIn("--exit-interval 3000", output)
        self.assertIn("--lr 1e-5", output)
        self.assertIn("--min-lr 1e-6", output)
        self.assertIn("--lr-warmup-iters 400", output)
        self.assertIn("--lr-decay-iters 3000", output)
        self.assertIn("--clip-grad 0.5", output)
        self.assertIn("--dataloader-type cyclic", output)
        self.assertIn("--no-data-sharding", output)
        self.assertIn("--full-validation", output)
        self.assertIn("--eval-micro-batch-size 1", output)
        self.assertIn("--eval-global-batch-size 8", output)
        self.assertIn("FINETUNE=1", output)
        self.assertIn("LOAD_OPTIM=0", output)
        self.assertIn("LOAD_RNG=0", output)
        self.assertIn("FINETUNE=0", output)
        self.assertIn("LOAD_OPTIM=1", output)
        self.assertIn("LOAD_RNG=1", output)
        self.assertEqual(output.count("--exit-interval 3000"), 1)
        self.assertIn("preserve optimizer/RNG/data cursor", output)
        self.assertIn("port=6014", output)

    def test_packing_runner_completes_small_isolated_fixture(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            phase1_source = root / "phase1"
            phase3_source = root / "phase3"
            phase1_source.mkdir()
            phase3_source.mkdir()
            sample = json.dumps(
                {
                    "id": "fixture",
                    "task": "quality",
                    "prompt_ids": [1],
                    "target_ids": [2],
                }
            ) + "\n"
            for index in range(198):
                name = f"train-{index:05d}.jsonl"
                (phase1_source / name).write_text(sample, encoding="utf-8")
                (phase3_source / name).write_text(sample, encoding="utf-8")

            phase2_source = root / "phase2.jsonl"
            phase3_dev = root / "phase3_dev.jsonl"
            phase2_source.write_text(sample, encoding="utf-8")
            performance_sample = json.dumps(
                {
                    "id": "fixture-performance",
                    "task": "performance",
                    "prompt_ids": [3],
                    "target_ids": [4],
                }
            ) + "\n"
            phase3_dev.write_text(sample + performance_sample, encoding="utf-8")
            fake_dev = root / "dev.parquet"
            fake_dev.touch()

            phase1_output = root / "out" / "phase1.jsonl"
            phase2_output = root / "out" / "phase2.jsonl"
            phase3_output = root / "out" / "phase3.jsonl"
            phase3_valid = root / "out" / "phase3_valid.jsonl"
            marker = root / "run" / "PACKING_COMPLETE_V1"
            run_script(
                "scripts/pack_unist198_full.sh",
                extra_env={
                    "PHASE1_SOURCE_DIR": str(phase1_source),
                    "PHASE2_SOURCE": str(phase2_source),
                    "PHASE3_SOURCE_DIR": str(phase3_source),
                    "UNIST_DEV_PARQUET": str(fake_dev),
                    "PHASE3_DEV_SOURCE": str(phase3_dev),
                    "PHASE1_TRAIN": str(phase1_output),
                    "PHASE2_TRAIN": str(phase2_output),
                    "PHASE3_TRAIN": str(phase3_output),
                    "PHASE3_VALID": str(phase3_valid),
                    "PACK_RUN_DIR": str(marker.parent),
                    "PACKING_COMPLETE_MARKER": str(marker),
                    "SEQ_LENGTH": "6",
                    "PACK_WORKERS": "2",
                },
            )
            for output in (phase1_output, phase2_output, phase3_output, phase3_valid):
                self.assertTrue(output.is_file())
                self.assertTrue(Path(f"{output}.count").is_file())
                validate_packed_jsonl.validate_file(output, seq_length=6)
            self.assertTrue(marker.is_file())

    def test_parallel_packer_preserves_every_sample_in_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_a = root / "source-a.jsonl"
            source_b = root / "source-b.jsonl"
            output = root / "packed.jsonl"
            samples = [
                {
                    "id": f"sample-{index:03d}",
                    "task": "quality",
                    "prompt_ids": [index + 1],
                    "target_ids": [1000 + index, 2000 + index],
                }
                for index in range(37)
            ]
            source_a.write_text(
                "".join(json.dumps(sample) + "\n" for sample in samples[:19]),
                encoding="utf-8",
            )
            source_b.write_text(
                "".join(json.dumps(sample) + "\n" for sample in samples[19:]),
                encoding="utf-8",
            )
            report = pack_sequences_parallel.parallel_pack(
                paths=[source_a, source_b],
                output=output,
                seq_length=6,
                workers=4,
            )
            packed = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
            source_ids = [source_id for item in packed for source_id in item["source_ids"]]
            self.assertEqual(source_ids, [sample["id"] for sample in samples])
            self.assertEqual(report["packed_sequences"], len(packed))
            self.assertEqual(report["workers"], 4)
            self.assertLessEqual(
                len(packed),
                13 + report["boundary_padding_records_at_most"],
            )
            validate_packed_jsonl.validate_file(output, seq_length=6)

    def test_pipeline_and_tensorboard_dry_runs_are_isolated(self):
        pipeline = run_script("scripts/run_unist198_full_pipeline.sh", "--dry-run")
        self.assertLess(pipeline.index("pack_unist198_full.sh"), pipeline.index("run_qwen0p5b_unist198_all_phases.sh"))
        self.assertEqual(pipeline.count("--start-phase phase1"), 2)

        resumed = run_script(
            "scripts/run_unist198_full_pipeline.sh",
            "--dry-run",
            extra_env={"PACK_START_PHASE": "phase2", "TRAIN_START_PHASE": "phase1"},
        )
        self.assertIn("pack_unist198_full.sh", resumed)
        self.assertIn("--start-phase phase2", resumed.splitlines()[0])
        self.assertIn("--start-phase phase1", resumed.splitlines()[1])

        pack_dry_run = run_script(
            "scripts/pack_unist198_full.sh", "--dry-run", "--start-phase", "phase2"
        )
        self.assertIn("pack phase2 with 16 worker(s)", pack_dry_run)
        self.assertIn("pack_sequences_parallel.py --workers 16", pack_dry_run)

        tensorboard = run_script("scripts/start_unist198_tensorboard.sh", "--dry-run")
        self.assertIn("tensorboard", tensorboard)
        self.assertIn("uniss_qwen0p5b_unist198_full_v1/tensorboard", tensorboard)
        self.assertIn("--host 0.0.0.0", tensorboard)
        self.assertIn("--port 6006", tensorboard)

        guard = run_script(
            "scripts/guard_unist198_parallel_stages.sh",
            "--dry-run",
            extra_env={"PHASE1_PACKED_COUNT_OVERRIDE": "256"},
        )
        self.assertIn("run in parallel", guard)
        self.assertIn("Phase1 iteration 6", guard)
        self.assertIn("start unist198_phase23_train at phase2", guard)

        monitor = run_script(
            "scripts/monitor_unist198_phase2_phase3.sh",
            "--dry-run",
            extra_env={
                "PHASE2_PACKED_COUNT_OVERRIDE": "129",
                "PHASE3_PACKED_COUNT_OVERRIDE": "128",
                "MIN_PHASE3_ITERATION": "1",
            },
        )
        self.assertIn("without starting, stopping, or modifying training", monitor)
        self.assertIn("Phase2 target=2", monitor)
        self.assertIn("Phase3 target=1", monitor)
        self.assertIn("finite TensorBoard lm loss", monitor)


if __name__ == "__main__":
    unittest.main()

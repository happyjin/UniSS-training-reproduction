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
        output = run_script(
            "scripts/run_qwen0p5b_unist198_all_phases.sh",
            "--dry-run",
            extra_env={
                "PHASE1_PACKED_COUNT_OVERRIDE": "256",
                "PHASE2_PACKED_COUNT_OVERRIDE": "129",
                "PHASE3_PACKED_COUNT_OVERRIDE": "128",
            },
        )
        self.assertIn("phase1=6/2", output)
        self.assertIn("phase2=2/1", output)
        self.assertIn("phase3=1/0", output)
        self.assertIn(r"CUDA_VISIBLE_DEVICES=0\,1\,2\,3\,4\,5\,6\,7", output)
        self.assertIn("NPROC_PER_NODE=8", output)
        self.assertIn("MICRO_BATCH_SIZE=2", output)
        self.assertEqual(output.count("--attention-backend flash"), 3)
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
        output = run_script(
            "scripts/run_qwen0p5b_unist198_all_phases.sh",
            "--dry-run",
            "--start-phase",
            "phase1",
            "--end-phase",
            "phase1",
            extra_env={"PHASE1_PACKED_COUNT_OVERRIDE": "256"},
        )
        self.assertIn("START=phase1, END=phase1", output)
        self.assertIn("phase1=6/2", output)
        self.assertIn("TRAIN_ITERS=6", output)
        self.assertIn("phase2 skipped because END_PHASE=phase1", output)
        self.assertIn("phase3 skipped because END_PHASE=phase1", output)
        self.assertNotIn("Missing packed count sidecar", output)

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


if __name__ == "__main__":
    unittest.main()

import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import after_claude_workload as monitor


class CompletionTests(unittest.TestCase):
    def fixture(self, root):
        config = {"armed_at": 1, "queues": ["queue"],
                  "papers": [{"directory": "paper", "report": "response.md"}],
                  "qabc_metrics": "metrics.json", "qabc_output": "output"}
        (root / "queue/add").mkdir(parents=True)
        (root / "queue/state.json").write_text(json.dumps({"test": "done"}))
        (root / "queue/add/jobs.json").write_text(json.dumps([{"id": "test"}]))
        (root / "paper").mkdir()
        report = "Final response with experimental results.\n" * 30
        (root / "paper/response.md").write_text(report)
        (root / "paper/main.tex").write_text("paper source")
        (root / "paper/main.pdf").write_bytes(b"%PDF-1.4 final output")
        os.utime(root / "paper/main.tex", (2, 2))
        os.utime(root / "paper/main.pdf", (3, 3))
        (root / "metrics.json").write_text(json.dumps({"dirs": {
            d: {"missing": [], "points": {"p5": {}}} for d in ("en2zh", "zh2en")}}))
        (root / "output/speech").mkdir(parents=True)
        (root / "output/blocks.jsonl").write_text('{"id": "a"}\n{"id": "b"}\n')
        (root / "output/cont_tasks_s0.jsonl").write_text(json.dumps({
            "key": "test", "out": "output/speech", "blocks": "output/blocks.jsonl"}) + "\n")
        (root / "output/speech/asr_speech.summary.json").write_text('{"transcribed": 2}')
        (root / "output/speech/asr_speech.jsonl").write_text('{"id": "a"}\n{"id": "b"}\n')
        def git(args, **kwargs):
            return SimpleNamespace(returncode=0, stdout=report if "show" in args else "")
        return config, git

    def test_complete_artifacts_and_jobs_allow_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, git = self.fixture(root)
            with patch.object(monitor.subprocess, "run", side_effect=git):
                self.assertEqual(monitor.experiment_gate(config, root)[0], [])

    def test_running_or_unconsumed_job_blocks_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, git = self.fixture(root)
            with patch.object(monitor.subprocess, "run", side_effect=git):
                (root / "queue/state.json").write_text('{"test": "run"}')
                self.assertTrue(monitor.experiment_gate(config, root)[0])
                (root / "queue/state.json").write_text('{"test": "done"}')
                (root / "queue/add/next.json").write_text('[{"id": "next"}]')
                self.assertTrue(any("unconsumed" in r for r in monitor.experiment_gate(config, root)[0]))

    def test_partial_asr_is_not_completion(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, git = self.fixture(root)
            (root / "output/speech/asr_speech.jsonl").write_text('{"id": "a"}\n')
            with patch.object(monitor.subprocess, "run", side_effect=git):
                self.assertTrue(any("ASR incomplete" in r for r in monitor.experiment_gate(config, root)[0]))

    def test_draft_report_and_stale_pdf_block_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, git = self.fixture(root)
            (root / "paper/response.md").write_text("<!-- BODY -->" * 70)
            os.utime(root / "paper/main.pdf", (1, 1))
            with patch.object(monitor.subprocess, "run", side_effect=git):
                reasons = monitor.experiment_gate(config, root)[0]
                self.assertTrue(any("draft" in r for r in reasons))
                self.assertTrue(any("predates" in r for r in reasons))

    def test_common_failures_adjust_parameters_unknown_error_stops(self):
        tuning = {"micro_batch_size": 4, "min_free_mib": 8192, "lr": 1e-4, "num_workers": 2}
        self.assertEqual(monitor.recover(tuning, "CUDA out of memory"), "retry")
        self.assertEqual((tuning["micro_batch_size"], tuning["min_free_mib"]), (2, 12288))
        self.assertEqual(monitor.recover(tuning, "non-finite training loss"), "retry")
        self.assertAlmostEqual(tuning["lr"], 1e-5)
        self.assertEqual(monitor.recover(tuning, "DataLoader worker died"), "retry")
        self.assertEqual(tuning["num_workers"], 0)
        self.assertEqual(monitor.recover(tuning, "ModuleNotFoundError"), "failed")

    def test_never_launches_while_experiments_pending(self):
        event = SimpleNamespace(stopped=False)
        event.is_set = lambda: event.stopped
        event.wait = lambda _: setattr(event, "stopped", True)
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(monitor, "STOP", event), \
                patch.object(monitor, "experiment_gate", return_value=(["jobs pending"], "hash")), \
                patch.object(monitor.subprocess, "Popen") as popen:
            monitor.supervise({"poll_seconds": 0}, Path(directory))
            popen.assert_not_called()

    def test_reused_pid_is_not_signalled(self):
        with patch.object(monitor, "process_info", return_value={"start": "new"}), \
                patch.object(monitor.os, "kill") as kill:
            monitor.signal_same_process({"pid": 123, "start": "old"}, 15)
            kill.assert_not_called()


if __name__ == "__main__":
    unittest.main()

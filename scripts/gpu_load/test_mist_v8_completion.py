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
import mist_v8_completion as gate


class V8CompletionTests(unittest.TestCase):
    def fixture(self, root):
        def write(name, value):
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(value)
            return path
        test = {"diff": -0.5, "p": 0.8, "p_holm": 1.0, "ci": [-1, 0], "meetings": 16, "holds": False}
        stats = {f"{test_name}-{dataset}": test for dataset in ("en2zh", "zh2en", "ali")
                 for test_name in ("E-ov1", "E-ov2", "E-lat", "E-pol")}
        write("tests.json", json.dumps(stats))
        write("windows.jsonl", '{"id":"a","meeting":"m1"}\n{"id":"b","meeting":"m2"}\n')
        write("subset.json", '{"ami":["m1"],"ali":["m1"]}')
        write("runs/on_en2zh_S2dlnosd/hyp.jsonl", '{"id":"a"}\n{"id":"b"}\n')
        write("runs/on_en2zh_S2dllag2/hyp.jsonl", '{"id":"a"}\n')
        write("b6.jsonl", '{"id":"a","read_ms":[30,40]}\n{"id":"b","read_ms":[10]}\n')
        write("b6_jobs.json", json.dumps([{"id": "b6", "cmd": "python infer.py --windows windows.jsonl --out b6.jsonl --time-reads"}]))
        session = write("claude/session.jsonl", '{"type":"assistant"}\n')
        os.utime(session, (1, 1))
        config = {"statistics": "tests.json", "runs": "runs",
                  "expected_runs": ["on_en2zh_S2dlnosd", "on_en2zh_S2dllag2"],
                  "windows": {key: "windows.jsonl" for key in ("en2zh", "zh2en", "ali")},
                  "subset_manifest": "subset.json", "b6_manifest": "b6_jobs.json",
                  "session_files": ["claude/session.jsonl"], "session_globs": ["claude/agent-*.jsonl"],
                  "quiet_seconds": 60,
                  "git_delivery": [{"directory": ".", "remote_ref": "refs/remotes/private/main"}]}
        return config

    def check(self, config, root):
        reasons, evidence = [], []
        def git(args, **kwargs):
            return SimpleNamespace(returncode=0, stdout="abc\n" if "rev-parse" in args else "")
        with patch.object(gate.subprocess, "run", side_effect=git):
            gate.extra_checks(config, root, reasons, evidence)
        return reasons, evidence

    def test_complete_negative_results_are_allowed(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.assertEqual(self.check(self.fixture(root), root)[0], [])

    def test_missing_required_run_blocks_completion(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            config = self.fixture(root)
            config["expected_runs"].append("on_ali_S2dlLA2")
            self.assertTrue(any("missing" in r for r in self.check(config, root)[0]))

    def test_matching_line_count_does_not_hide_missing_windows(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            config = self.fixture(root)
            (root / "runs/on_en2zh_S2dlnosd/hyp.jsonl").write_text('{"id":"a"}\n{"id":"a"}\n')
            self.assertTrue(any("duplicates=1" in r for r in self.check(config, root)[0]))

    def test_b6_requires_actual_wall_clock_timings(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            config = self.fixture(root)
            (root / "b6.jsonl").write_text('{"id":"a"}\n{"id":"b"}\n')
            self.assertTrue(any("wall-clock" in r for r in self.check(config, root)[0]))

    def test_missing_statistic_blocks_completion(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            config = self.fixture(root)
            stats = json.loads((root / "tests.json").read_text())
            stats["E-pol-ali"] = None
            (root / "tests.json").write_text(json.dumps(stats))
            self.assertTrue(any("E-pol-ali" in r for r in self.check(config, root)[0]))

    def test_new_agent_activity_blocks_and_changes_fingerprint(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            config = self.fixture(root)
            first = self.check(config, root)[1]
            (root / "claude/agent-new.jsonl").write_text('{"type":"assistant"}\n')
            reasons, second = self.check(config, root)
            self.assertNotEqual(first, second)
            self.assertTrue(any("still active" in r for r in reasons))

    def test_unpushed_commit_blocks_completion(self):
        with tempfile.TemporaryDirectory() as folder:
            reasons, evidence = [], []
            with patch.object(gate.subprocess, "run", return_value=SimpleNamespace(returncode=1, stdout="")):
                gate.git_delivery({"directory": ".", "remote_ref": "refs/remotes/private/main"},
                                  Path(folder), reasons, evidence)
            self.assertTrue(any("push record" in r for r in reasons))

    def test_observe_only_gate_cannot_allow_launch(self):
        reasons, _ = monitor.experiment_gate({"queues": [], "papers": [], "launch_enabled": False})
        self.assertTrue(any("monitor-only" in r for r in reasons))

    def test_monitor_changes_do_not_hide_dirty_experiments(self):
        item = {"directory": ".", "remote_ref": "refs/remotes/private/main",
                "ignore_dirty_prefixes": ["scripts/gpu_load/"]}
        for dirty, expected_block in [("scripts/gpu_load/after_claude_workload.py\0", False),
                                      ("reports/gstar_si_v8/after_v8.py\0", True)]:
            def git(args, **kwargs):
                return SimpleNamespace(returncode=0, stdout=dirty if "diff" in args else "abc")
            with patch.object(gate.subprocess, "run", side_effect=git):
                reasons, evidence = [], []
                gate.git_delivery(item, Path("/tmp"), reasons, evidence)
                self.assertEqual(bool(reasons), expected_block)

    def test_unrecognized_completion_data_never_starts_training(self):
        event = SimpleNamespace(stopped=False)
        event.is_set = lambda: event.stopped
        event.wait = lambda _: setattr(event, "stopped", True)
        with tempfile.TemporaryDirectory() as folder, \
                patch.object(monitor, "STOP", event), \
                patch.object(monitor, "experiment_gate", side_effect=KeyError("changed schema")), \
                patch.object(monitor.subprocess, "Popen") as popen:
            monitor.supervise({"poll_seconds": 0}, Path(folder))
            popen.assert_not_called()


if __name__ == "__main__":
    unittest.main()

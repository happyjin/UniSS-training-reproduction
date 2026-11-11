"""CPU-only checks of scripts/gpu_load/train_mist_workload.py --ignore-k8s-pods (opt-in, user 2026-11-11):
with the flag a process in a Kubernetes pod cgroup neither blocks the start nor stops a running load, and its memory
is not counted as busy; any other process still does both; without the flag nothing changes; --tolerate-keeper
keeps its behaviour and combines with the new flag.

    python experiments/mist_w3po_v1/scripts/test_workload_ignore_k8s.py
"""
import importlib.util
from pathlib import Path
from unittest.mock import mock_open, patch
import unittest

REPO = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location("training_load_k8s", REPO / "scripts/gpu_load/train_mist_workload.py")
load = importlib.util.module_from_spec(spec)
spec.loader.exec_module(load)

POD, OURS, KEEPER = 700, 800, 500


def fake_smi(query):
    if query.startswith("--query-gpu"):
        return [["0", "GPU-a", "143000", "40000"]]
    if query == "--query-compute-apps=gpu_uuid,pid,used_memory":
        return [["GPU-a", str(POD), "18000"]]
    if query == "--query-compute-apps=gpu_uuid,pid":
        return [["GPU-a", str(POD)]]
    raise AssertionError(query)


def fake_smi_mixed(query):
    if query.startswith("--query-gpu"):
        return [["0", "GPU-a", "143000", "90000"]]
    if query == "--query-compute-apps=gpu_uuid,pid,used_memory":
        return [["GPU-a", str(POD), "18000"], ["GPU-a", str(OURS), "12000"], ["GPU-a", str(KEEPER), "59000"]]
    if query == "--query-compute-apps=gpu_uuid,pid":
        return [["GPU-a", str(POD)], ["GPU-a", str(OURS)], ["GPU-a", str(KEEPER)]]
    raise AssertionError(query)


class IgnoreK8sPods(unittest.TestCase):
    def test_flag_default_off_and_parsed(self):
        self.assertFalse(load.parse_args(["--devices", "0"]).ignore_k8s_pods)
        self.assertTrue(load.parse_args(["--devices", "0", "--ignore-k8s-pods"]).ignore_k8s_pods)

    def test_pod_blocks_without_flag(self):
        a = load.parse_args(["--devices", "0"])
        with patch.object(load, "smi", side_effect=fake_smi), patch.object(load, "is_k8s_pod", return_value=True):
            cards = load.gpu_snapshot(a)
        self.assertEqual(cards[0]["pids"], {POD})
        self.assertEqual(load.busy_cards(a, cards), [0])

    def test_pod_ignored_with_flag(self):
        a = load.parse_args(["--devices", "0", "--ignore-k8s-pods"])
        with patch.object(load, "smi", side_effect=fake_smi), patch.object(load, "is_k8s_pod", return_value=True):
            cards = load.gpu_snapshot(a)
        self.assertEqual(cards[0]["pids"], set())
        self.assertEqual(cards[0]["used"], 40000 - 18000)
        self.assertEqual(load.busy_cards(a, cards), [0])   # 22000 MiB still used > idle_memory_mib
        a2 = load.parse_args(["--devices", "0", "--ignore-k8s-pods", "--idle-memory-mib", "30000"])
        self.assertEqual(load.busy_cards(a2, cards), [])

    def test_own_process_still_counts_and_keeper_combines(self):
        a = load.parse_args(["--devices", "0", "--ignore-k8s-pods", "--tolerate-keeper"])
        with patch.object(load, "smi", side_effect=fake_smi_mixed), \
                patch.object(load, "is_k8s_pod", side_effect=lambda p: p == POD), \
                patch.object(load, "is_keeper", side_effect=lambda p: p == KEEPER):
            cards = load.gpu_snapshot(a)
        self.assertEqual(cards[0]["pids"], {OURS})
        self.assertEqual(cards[0]["used"], 90000 - 18000 - 59000)

    def test_tolerate_keeper_alone_unchanged(self):
        a = load.parse_args(["--devices", "0", "--tolerate-keeper"])
        with patch.object(load, "smi", side_effect=fake_smi_mixed), \
                patch.object(load, "is_k8s_pod", side_effect=lambda p: p == POD), \
                patch.object(load, "is_keeper", side_effect=lambda p: p == KEEPER):
            cards = load.gpu_snapshot(a)
        self.assertEqual(cards[0]["pids"], {POD, OURS})
        self.assertEqual(cards[0]["used"], 90000 - 59000)

    def test_is_k8s_pod_reads_cgroup(self):
        with patch("builtins.open", mock_open(read_data="0::/kubepods.slice/kubepods-podx.slice/cri.scope\n")):
            self.assertTrue(load.is_k8s_pod(1))
        with patch("builtins.open", mock_open(read_data="0::/user.slice/user-0.slice/session-1.scope\n")):
            self.assertFalse(load.is_k8s_pod(1))
        with patch("builtins.open", side_effect=OSError):
            self.assertFalse(load.is_k8s_pod(1))


class SmiRetries(unittest.TestCase):
    """--smi-retries (opt-in, user 2026-11-11): a timed-out nvidia-smi query is retried instead of stopping the load."""

    def test_default_one_attempt_and_retry(self):
        import subprocess
        self.assertEqual(load.parse_args(["--devices", "0"]).smi_retries, 0)
        calls = {"n": 0}

        def fake_run(*a, **k):
            calls["n"] += 1
            if calls["n"] < 3:
                raise subprocess.TimeoutExpired("nvidia-smi", 15)
            return subprocess.CompletedProcess(a, 0, stdout="0, x\n", stderr="")

        with patch.object(load.subprocess, "run", side_effect=fake_run), patch.object(load.time, "sleep"):
            load.SMI_RETRIES = 0
            with self.assertRaises(subprocess.TimeoutExpired):
                load.smi("--q")
            calls["n"] = 0
            load.SMI_RETRIES = 3
            self.assertEqual(load.smi("--q"), [["0", "x"]])
            self.assertEqual(calls["n"], 3)
            load.SMI_RETRIES = 0


if __name__ == "__main__":
    unittest.main()

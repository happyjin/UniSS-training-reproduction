"""CPU-only checks of scripts/gpu_load/train_mist_workload.py --tolerate-small-mib (opt-in, user 2026-11-12:
"改守护脚本补上，防止空卡"). With the flag, small compute processes already on a GPU (e.g. an idle 2.5 GB CTC
scorer at the tail of an eval) neither block the start nor count as busy memory; a process that appears after the
load started, or a tolerated one that grows above the limit, still stops the load; without the flag nothing changes.

    python experiments/mist_w3po_v1/scripts/test_workload_tolerate_small.py
    WORKLOAD_PY=<candidate file> python ...   (check a candidate before it replaces the live file)
"""
import importlib.util
import os
from pathlib import Path
import unittest
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[3]
TARGET = Path(os.environ.get("WORKLOAD_PY", REPO / "scripts/gpu_load/train_mist_workload.py"))
spec = importlib.util.spec_from_loader("training_load", loader=None)
load = importlib.util.module_from_spec(spec)
load.__file__ = str(REPO / "scripts/gpu_load/train_mist_workload.py")
exec(compile(TARGET.read_text(), str(TARGET), "exec"), load.__dict__)

SCORER, BIG, WORKER, NEWSMALL = 500, 600, 9001, 700


def smi_with(apps):
    """apps: list of (pid, used_mib) on GPU-a (total 143000)."""
    def fake(query):
        if query.startswith("--query-gpu"):
            return [["0", "GPU-a", "143000", str(sum(u for _, u in apps))]]
        if query == "--query-compute-apps=gpu_uuid,pid,used_memory":
            return [["GPU-a", str(p), str(u)] for p, u in apps]
        if query == "--query-compute-apps=gpu_uuid,pid":
            return [["GPU-a", str(p)] for p, _ in apps]
        raise AssertionError(query)
    return fake


def snap(a, apps):
    with patch.object(load, "smi", side_effect=smi_with(apps)):
        return load.gpu_snapshot(a)


class TolerateSmall(unittest.TestCase):
    def test_flag_default_off_and_parsed(self):
        self.assertEqual(load.parse_args(["--devices", "0"]).tolerate_small_mib, 0)
        self.assertEqual(load.parse_args(["--devices", "0", "--tolerate-small-mib", "8192"]).tolerate_small_mib, 8192)

    def test_negative_rejected(self):
        with self.assertRaises(SystemExit):
            load.parse_args(["--devices", "0", "--tolerate-small-mib", "-1"])

    def test_default_unchanged(self):
        a = load.parse_args(["--devices", "0"])
        cards = snap(a, [(SCORER, 2500)])
        self.assertEqual(cards[0], {"uuid": "GPU-a", "total": 143000, "used": 2500, "pids": {SCORER}})
        self.assertEqual(load.busy_cards(a, cards), [0])

    def test_small_tolerated_at_start(self):
        a = load.parse_args(["--devices", "0", "--tolerate-small-mib", "8192"])
        cards = snap(a, [(SCORER, 2500)])
        self.assertEqual(cards[0]["pids"], set())
        self.assertEqual(cards[0]["small"], {SCORER: 2500})
        self.assertEqual(cards[0]["used"], 0)
        self.assertEqual(load.raw_used(cards[0]), 2500)
        self.assertEqual(load.busy_cards(a, cards), [])

    def test_big_still_busy(self):
        a = load.parse_args(["--devices", "0", "--tolerate-small-mib", "8192"])
        cards = snap(a, [(SCORER, 2500), (BIG, 64000)])
        self.assertEqual(cards[0]["pids"], {BIG})
        self.assertEqual(load.busy_cards(a, cards), [0])

    def test_running_tolerates_initial_small_only(self):
        a = load.parse_args(["--devices", "0", "--tolerate-small-mib", "8192"])
        tolerated = set(snap(a, [(SCORER, 2500)])[0]["small"])
        # the load's own worker may be small while warming up: not "another process"
        warm = snap(a, [(SCORER, 2500), (WORKER, 600)])
        self.assertEqual(warm[0]["pids"] | load.small_others(warm[0], tolerated, WORKER), set())
        # a new small process appears -> it must stop the load
        new = snap(a, [(SCORER, 2500), (WORKER, 120000), (NEWSMALL, 300)])
        others = (new[0]["pids"] - {WORKER}) | load.small_others(new[0], tolerated, WORKER)
        self.assertEqual(others, {NEWSMALL})
        # the tolerated scorer grows above the limit -> it must stop the load
        grown = snap(a, [(SCORER, 9000), (WORKER, 120000)])
        others = (grown[0]["pids"] - {WORKER}) | load.small_others(grown[0], tolerated, WORKER)
        self.assertEqual(others, {SCORER})

    def test_unparseable_memory_is_not_small(self):
        a = load.parse_args(["--devices", "0", "--tolerate-small-mib", "8192"])
        with patch.object(load, "smi", side_effect=lambda q: [["0", "GPU-a", "143000", "100"]] if q.startswith(
                "--query-gpu") else [["GPU-a", str(SCORER), "[N/A]"]]):
            cards = load.gpu_snapshot(a)
        self.assertEqual(cards[0]["pids"], {SCORER})

    def test_composes_with_k8s_flag(self):
        a = load.parse_args(["--devices", "0", "--tolerate-small-mib", "8192", "--ignore-k8s-pods"])
        with patch.object(load, "is_k8s_pod", side_effect=lambda p: p == BIG):
            cards = snap(a, [(SCORER, 2500), (BIG, 64000)])
        self.assertEqual(cards[0]["pids"], set())
        self.assertEqual(cards[0]["small"], {SCORER: 2500})
        self.assertEqual(cards[0]["used"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)

"""CPU-only checks of scripts/gpu_load/train_mist_workload.py --tolerate-keeper (opt-in, user approval 2026-11-06):
with the flag a root-owned W3PO keeper neither blocks the start nor stops a running load, and its memory is not
counted as busy; any other process still does both; without the flag nothing changes.

    python experiments/mist_w3po_v1/scripts/test_workload_tolerate_keeper.py
"""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location("training_load", REPO / "scripts/gpu_load/train_mist_workload.py")
load = importlib.util.module_from_spec(spec)
spec.loader.exec_module(load)

KEEPER, OTHER, WORKER = 500, 600, 9001


def fake_smi(query):
    if query.startswith("--query-gpu"):
        return [["0", "GPU-a", "143000", "60000"]]
    if query == "--query-compute-apps=gpu_uuid,pid,used_memory":
        return [["GPU-a", str(KEEPER), "59500"]]
    if query == "--query-compute-apps=gpu_uuid,pid":
        return [["GPU-a", str(KEEPER)]]
    raise AssertionError(query)


class TolerateKeeper(unittest.TestCase):
    def test_flag_default_off_and_parsed(self):
        self.assertFalse(load.parse_args(["--devices", "0"]).tolerate_keeper)
        self.assertTrue(load.parse_args(["--devices", "0", "--tolerate-keeper"]).tolerate_keeper)

    def test_keeper_blocks_without_flag(self):
        a = load.parse_args(["--devices", "0"])
        with patch.object(load, "smi", side_effect=fake_smi), patch.object(load, "is_keeper", return_value=True):
            cards = load.gpu_snapshot(a)
        self.assertEqual(cards[0]["pids"], {KEEPER})
        self.assertEqual(load.busy_cards(a, cards), [0])

    def test_keeper_tolerated_with_flag(self):
        a = load.parse_args(["--devices", "0", "--tolerate-keeper"])
        with patch.object(load, "smi", side_effect=fake_smi), \
                patch.object(load, "is_keeper", side_effect=lambda p: p == KEEPER):
            cards = load.gpu_snapshot(a)
        self.assertEqual(cards[0]["pids"], set())
        self.assertEqual(cards[0]["used"], 500)
        self.assertEqual(load.busy_cards(a, cards), [])

    def test_non_keeper_still_busy_with_flag(self):
        a = load.parse_args(["--devices", "0", "--tolerate-keeper"])
        with patch.object(load, "smi", side_effect=fake_smi), patch.object(load, "is_keeper", return_value=False):
            cards = load.gpu_snapshot(a)
        self.assertEqual(load.busy_cards(a, cards), [0])

    def test_running_load_ignores_keeper_but_stops_on_foreign(self):
        a = load.parse_args(["--devices", "0", "--tolerate-keeper"])
        idle = {0: {"uuid": "GPU-a", "used": 0, "total": 143000, "pids": set()}}
        child = SimpleNamespace(pid=WORKER, poll=lambda: None, wait=lambda **kw: 0)

        def apps(query):
            if query.startswith("--query-gpu"):   # before launch only the keeper (29.5 GB) is on the card
                return [["0", "GPU-a", "143000", "29500" if getattr(apps, "n", 0) == 0 else "100000"]]
            n = apps.n = getattr(apps, "n", 0) + 1
            rows = [["GPU-a", str(KEEPER), "29000"]] + ([["GPU-a", str(WORKER), "70000"]] if n >= 2 else [])
            if n >= 3:
                rows.append(["GPU-a", str(OTHER), "1000"])
            return rows

        with patch.object(load, "wait_until_ready", return_value=idle), patch.object(load, "smi", side_effect=apps), \
                patch.object(load, "is_keeper", side_effect=lambda p: p == KEEPER), \
                patch.object(load.subprocess, "Popen", return_value=child), patch.object(load.os, "killpg"), \
                patch.object(load.time, "sleep"):
            with self.assertRaisesRegex(RuntimeError, rf"another GPU process appeared: \[{OTHER}\]"):
                load.supervise(a)
        self.assertGreaterEqual(apps.n, 3)   # the keeper alone (polls 1-2) did not stop it


if __name__ == "__main__":
    unittest.main()

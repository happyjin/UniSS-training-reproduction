"""Toy tests of b6_wallclock.analyse.

    /opt/dlami/nvme/jasonleeeli/conda_envs/uniss-eval/bin/python -m unittest reports/gstar_si_v8/test_b6_wallclock.py -v
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import b6_wallclock as B  # noqa: E402


def win(i, start, dur, open_ms=0, close_ms=None):
    return {"id": f"w{i}", "start_s": start, "dur_ms": dur, "open_detect_ms": open_ms,
            "close_detect_ms": dur if close_ms is None else close_ms}


class T(unittest.TestCase):
    def test_keeps_pace(self):
        # one 3 s window, reads of 0.5 s: no queueing, RTF 1.5 s / 3 s = 0.5
        W = [win(0, 10.0, 3000)]
        H = {"w0": {"read_ms": [500, 500, 500], "segments": [{"start_s": 0.0, "end_s": 1.5, "write_ms": 2000}]}}
        r = B.analyse(W, H)
        self.assertEqual(r["reads"], 3)
        self.assertAlmostEqual(r["rtf"], 0.5)
        self.assertEqual(r["queue_wait_max"], 0.0)
        # segment written at read 2 (due 12.0 s), read ends 12.5 s; source end 11.5 s -> 1.0 s
        self.assertAlmostEqual(r["text_lag_wall"]["p50"], 1.0)
        self.assertAlmostEqual(r["text_lag_nocompute_p50"], 0.5)

    def test_falls_behind(self):
        # reads of 2 s on a 1 s cadence: read k ends at 1 + 2k s; RTF 6/3 = 2
        W = [win(0, 0.0, 3000)]
        H = {"w0": {"read_ms": [2000, 2000, 2000], "segments": [{"start_s": 2.0, "end_s": 2.5, "write_ms": 3000}]}}
        r = B.analyse(W, H)
        self.assertAlmostEqual(r["rtf"], 2.0)
        self.assertAlmostEqual(r["queue_wait_max"], 2.0)   # read 3 due 3 s starts 5 s
        self.assertAlmostEqual(r["text_lag_wall"]["p50"], 7.0 - 2.5)
        self.assertEqual(r["reads_over_cadence"], 1.0)

    def test_queue_across_windows_and_missing(self):
        # two overlapping windows of one meeting share the decoder; a third window has no output
        W = [win(0, 0.0, 2000), win(1, 0.5, 2000), win(2, 5.0, 1000)]
        H = {"w0": {"read_ms": [800, 800], "segments": []}, "w1": {"read_ms": [800, 800], "segments": []}}
        r = B.analyse(W, H)
        self.assertEqual(r["windows_missing"], 1)
        # dues 1.0, 1.5, 2.0, 2.5 -> ends 1.8, 2.6, 3.4, 4.2; waits 0, 0.3, 0.6, 0.9
        self.assertAlmostEqual(r["queue_wait_max"], 0.9)

    def test_open_close_clamp(self):
        # opening known at 1.2 s, end known at 4.0 s
        w = win(0, 0.0, 2000, open_ms=1200, close_ms=4000)
        self.assertAlmostEqual(B.read_due(w, 1000), 1.2)
        self.assertAlmostEqual(B.read_due(w, 2000), 4.0)


if __name__ == "__main__":
    unittest.main()

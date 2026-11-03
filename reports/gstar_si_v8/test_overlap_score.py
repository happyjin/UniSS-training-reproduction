"""Toy tests of overlap_score.score (Appendix C's example plus overlap cases).

    /opt/dlami/nvme/jasonleeeli/conda_envs/uniss-eval/bin/python -m unittest reports/gstar_si_v8/test_overlap_score.py -v
"""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import overlap_score as O  # noqa: E402

# A says u1 (0-4 s) and u2 (4.5-8 s); B says "yes" (2-2.5 s) inside u1 and u3 (6-9 s) overlapping u2
U = {"m": [("u1", "A", 0.0, 4.0, False), ("u2", "A", 4.5, 8.0, False), ("y", "B", 2.0, 2.5, False),
           ("u3", "B", 6.0, 9.0, False)]}
REF = [{"meeting": "m", "start_s": 0.0, "utts": [
    {"record_id": "u1", "speaker_id": "A", "begin_s": 0.0, "end_s": 4.0, "hesitation": False, "zh": "", "ref": "we start the meeting now"},
    {"record_id": "u2", "speaker_id": "A", "begin_s": 4.5, "end_s": 8.0, "hesitation": False, "zh": "", "ref": "the budget is too high"},
    {"record_id": "y", "speaker_id": "B", "begin_s": 2.0, "end_s": 2.5, "hesitation": False, "zh": "", "ref": "yes"},
    {"record_id": "u3", "speaker_id": "B", "begin_s": 6.0, "end_s": 9.0, "hesitation": False, "zh": "", "ref": "i do not agree with that"}]}]
W = [{"id": "w", "meeting": "m", "start_s": 0.0, "dur_ms": 10000, "open_detect_ms": 0, "close_detect_ms": 10000}]


def run(segments, **kw):
    H = {"w": {"segments": segments}}
    return O.score("zh2en", W, H, REF, U, online=True, **kw)


class T(unittest.TestCase):
    def test_perfect(self):
        r = run([{"spk": 1, "start_s": 0.0, "end_s": 8.0, "text": "we start the meeting now the budget is too high", "write_ms": 8500},
                 {"spk": 2, "start_s": 2.0, "end_s": 2.5, "text": "yes", "write_ms": 3000},
                 {"spk": 2, "start_s": 6.0, "end_s": 9.0, "text": "i do not agree with that", "write_ms": 9500}])
        self.assertEqual(r["tcpBLEU_all"], 100.0)
        self.assertEqual(r["CR"], 1.0)          # pairs (u1, yes) and (u2, u3)
        self.assertEqual(r["pairs"], 2)
        self.assertEqual(r["collapse"], 0.0)

    def test_regions(self):
        # u1 overlaps "yes" for 0.5/4 s (12.5% < 20%) -> 1spk... but u1 and u2 form one unit (0.5 s apart): 0-8 s,
        # overlapped by yes (0.5) + u3 (2.0) = 2.5/8 = 31% -> ov; "yes" (fully inside u1) -> ov; u3 overlaps u2 2/3 -> ov
        r = run([{"spk": 1, "start_s": 0.0, "end_s": 8.0, "text": "we start the meeting now the budget is too high", "write_ms": 8500}])
        self.assertEqual(r["units_ov"], 3)
        self.assertEqual(r["units_1spk"], 0)
        self.assertAlmostEqual(r["omitted_ov"], 2 / 3, places=3)

    def test_collapse(self):
        # B's "yes" written under A's slot: not recalled for B, covered by A's slot -> collapse
        r = run([{"spk": 1, "start_s": 0.0, "end_s": 8.0, "text": "we start the meeting now the budget is too high", "write_ms": 8500},
                 {"spk": 1, "start_s": 2.0, "end_s": 2.5, "text": "yes", "write_ms": 3000},
                 {"spk": 2, "start_s": 6.0, "end_s": 9.0, "text": "i do not agree with that", "write_ms": 9500}])
        self.assertEqual(r["pairs"], 2)
        self.assertAlmostEqual(r["CR"], 0.5)
        self.assertAlmostEqual(r["collapse"], 0.5)

    def test_split(self):
        # B says b1 (0-3 s) and b2 (5-8 s), 2 s apart -> two units; one B segment over 1-7 s covers b1 for 2 s and b2 for
        # 2 s (each 33% of it) -> split half and half
        U2 = {"m": [("b1", "B", 0.0, 3.0, False), ("b2", "B", 5.0, 8.0, False)]}
        R2 = [{"meeting": "m", "start_s": 0.0, "utts": [
            {"record_id": "b1", "speaker_id": "B", "begin_s": 0.0, "end_s": 3.0, "hesitation": False, "zh": "", "ref": "one two three four five"},
            {"record_id": "b2", "speaker_id": "B", "begin_s": 5.0, "end_s": 8.0, "hesitation": False, "zh": "", "ref": "six seven eight nine ten"}]}]
        H = {"w": {"segments": [{"spk": 1, "start_s": 1.0, "end_s": 7.0, "text": "one two three four five six seven eight nine ten", "write_ms": 8000}]}}
        r = O.score("zh2en", W, H, R2, U2, online=True)
        self.assertEqual(r["split_segments"], 1)
        self.assertEqual(r["tcpBLEU_all"], 100.0)
        r0 = O.score("zh2en", W, H, R2, U2, online=True, split=False)
        self.assertLess(r0["tcpBLEU_all"], 100.0)

    def test_relabel_causal(self):
        post = np.zeros((125, 8), dtype=np.float16)
        post[:, 3] = 0.9          # slot 4 active everywhere
        s = {"a": 1.0, "b": 2.0}
        self.assertEqual(O.relabel_causal(s, post, 0.0, 4.0), 4)
        # at commit 1.5 s nothing of the span is available yet (1.5 - 1.04 = 0.46 s < 1.0 s): fall back to [a-2, avail]
        self.assertEqual(O.relabel_causal(s, post, 0.0, 1.5), 4)
        self.assertEqual(O.relabel_causal(s, post, 0.0, 1.0), 0)   # no frame available at all -> unknown


if __name__ == "__main__":
    unittest.main()

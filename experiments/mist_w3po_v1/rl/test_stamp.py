"""Hand-made cases for STAMP (rl/stamp.py); run: python -m experiments.mist_w3po_v1.rl.test_stamp"""
from __future__ import annotations

import numpy as np

from experiments.mist_w3po_v1.rl import stamp as S

UTTS = [
    {"speaker_id": "A", "begin_s": 0.5, "end_s": 4.0, "ref": "We should finish the remote control design today.", "zh": "我们今天应该完成遥控器设计。"},
    {"speaker_id": "B", "begin_s": 3.0, "end_s": 3.6, "ref": "Yeah.", "zh": "是的。"},
    {"speaker_id": "B", "begin_s": 6.0, "end_s": 9.0, "ref": "The battery should last for two years.", "zh": "电池应该能用两年。"},
]
DUR = 10.0
U = S.units_of(UTTS)


def seg(spk, a, b, text, r, t):
    return {"spk": spk, "a": a, "b": b, "text": text, "r": r, "t": t}


GOOD = [seg(1, 0.5, 4.0, "We should finish the remote control design today.", 4, 5.0),
        seg(2, 3.0, 3.6, "Yeah.", 4, 5.0),
        seg(2, 6.0, 9.0, "The battery should last for two years.", 9, 10.0)]


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    assert cond, name


def main():
    check("units: two B utterances 2.4 s apart stay separate", len(U) == 3)
    merged = S.units_of(UTTS[:2] + [{**UTTS[2], "begin_s": 4.3}])
    check("units: same-speaker gap < 1 s merged", len(merged) == 2)

    sc = S.score(GOOD, U, S.speaker_map(GOOD, U), DUR)
    check(f"perfect output R ~ 1 ({sc.R:.3f})", sc.R > 0.95)

    swapped = [{**s, "spk": {1: 5, 2: 3}[s["spk"]]} for s in GOOD]
    sc2 = S.score(swapped, U, S.speaker_map(swapped, U), DUR)
    check("slot permutation invariance", abs(sc2.R - sc.R) < 1e-9)

    # chunked output: one unit as two segments -- no penalty beyond the text quality
    chunked = [seg(1, 0.5, 2.0, "We should finish", 2, 3.0), seg(1, 2.0, 4.0, "the remote control design today.", 4, 5.0)] + GOOD[1:]
    sc3 = S.score(chunked, U, S.speaker_map(chunked, U), DUR)
    check(f"chunked segments of one unit keep R ({sc3.R:.3f})", sc3.R > 0.9)

    # duplicate: the long segment committed twice -> each copy's LOO credit <= 0
    dup = GOOD + [seg(1, 0.6, 4.0, "We should finish the remote control design today.", 5, 6.0)]
    cr = S.credit(dup, U, DUR, wait_reads=list(range(9)), read_t=[float(r + 1) for r in range(10)])
    check(f"duplicate: R lower than without ({cr.R:.3f})", cr.R < sc.R)
    check(f"duplicate: both copies LOO <= 0 ({cr.delta[0]:.3f}, {cr.delta[3]:.3f})", cr.delta[0] <= 0 and cr.delta[3] <= 0)
    check("duplicate: the other segments LOO > 0", cr.delta[1] > 0 and cr.delta[2] > 0)

    # hallucination outside any unit -> spurious, negative credit
    hal = GOOD + [seg(1, 9.6, 9.9, "And then we go to lunch.", 9, 10.0)]
    ch = S.credit(hal, U, DUR, wait_reads=list(range(9)), read_t=[float(r + 1) for r in range(10)])
    check(f"hallucination within the 2 s collar of a unit: inserted there, LOO < 0 ({ch.delta[3]:.3f})", ch.delta[3] < 0)
    far = GOOD + [seg(1, 12.0, 13.0, "And then we go to lunch.", 13, 14.0)]
    cf = S.credit(far, U, 14.0, wait_reads=list(range(13)), read_t=[float(r + 1) for r in range(14)])
    check(f"hallucination away from every unit: spurious, LOO = -LAMBDA_SP/|U| ({cf.delta[3]:.3f})",
          cf.spurious == 1 and abs(cf.delta[3] + S.LAMBDA_SP / 3) < 1e-9)

    # missed interjection: no "Yeah." -> R loses 1/3, the waits at reads ending >= 3.6 + 0.5 share -1/3
    miss = [GOOD[0], GOOD[2]]
    cm = S.credit(miss, U, DUR, wait_reads=list(range(9)), read_t=[float(r + 1) for r in range(10)])
    check(f"miss: R ~ 2/3 ({cm.R:.3f})", abs(cm.R - 2 / 3) < 0.05)
    check(f"miss credit to reads t >= 4.1 only, sum -1/3 ({sorted(cm.miss)})",
          min(cm.miss) == 4 and abs(sum(cm.miss.values()) + 1 / 3) < 1e-9)
    cm2 = S.credit(miss, U, DUR, wait_reads=[0, 1, 2], read_t=[float(r + 1) for r in range(10)])
    check("miss: no wait after the unit -> no when penalty", not cm2.miss)

    # wrong speaker on the interjection; posteriors say slot 2 -> RBC positive for... the model chose 1
    wrong = [GOOD[0], {**GOOD[1], "spk": 1}, GOOD[2]]
    post = np.zeros((125, 8), dtype=np.float32)
    post[6:50, 0] = 1.0          # slot 1: 0.5-4.0 s
    post[37:46, 1] = 0.95         # slot 2: 3.0-3.6 s (overlap; A quieter there)
    post[37:46, 0] = 0.3
    post[75:113, 1] = 1.0         # slot 2: 6-9 s
    cw = S.credit(wrong, U, DUR, wait_reads=[], read_t=[float(r + 1) for r in range(10)], post=post)
    check(f"wrong speaker lowers R ({cw.R:.3f} < {sc.R:.3f})", cw.R < sc.R - 0.05)
    check(f"RBC: the wrong slot gets negative who credit ({cw.who[1]:.3f})", cw.who[1] < 0)
    cr2 = S.credit(GOOD, U, DUR, wait_reads=[], read_t=[float(r + 1) for r in range(10)], post=post)
    check(f"RBC: agreeing with the relabel -> 0 ({cr2.who})", all(abs(w) < 1e-9 for w in cr2.who))
    post2 = post.copy()
    post2[37:46, 0] = 1.0         # A dominant over B's interjection: the relabel itself picks slot 1
    cw2 = S.credit(wrong, U, DUR, wait_reads=[], read_t=[float(r + 1) for r in range(10)], post=post2)
    check(f"RBC: wrong like the relabel -> 0, no advantage over it ({cw2.who[1]:.3f})", abs(cw2.who[1]) < 1e-9)
    right = S.credit(GOOD, U, DUR, wait_reads=[], read_t=[float(r + 1) for r in range(10)], post=post2)
    check(f"RBC: right where the relabel is wrong -> positive ({right.who[1]:.3f})", right.who[1] > 0)

    # language gate: Chinese copied into the English output
    zh = [GOOD[0], seg(2, 3.0, 3.6, "是的。", 4, 5.0), GOOD[2]]
    cz = S.credit(zh, U, DUR, wait_reads=[], read_t=[1.0])
    check(f"language gate: counted, R penalised ({cz.lang}, {cz.R:.3f})", cz.lang == 1 and cz.R < 2 / 3)

    # format: zero-length segment
    zl = [{**GOOD[0], "b": 0.5}] + GOOD[1:]
    check("format: zero-length counted", S.score(zl, U, S.speaker_map(zl, U), DUR).fmt == 1)

    # empty output
    ce = S.credit([], U, DUR, wait_reads=list(range(9)), read_t=[float(r + 1) for r in range(10)])
    check(f"empty output: R = 0; the two units ending before the last wait charge it -2/3 ({ce.R})",
          ce.R == 0 and abs(sum(ce.miss.values()) + 2 / 3) < 1e-9)

    # FLAD
    anchors = [(0.8, 3.0), (0.9, 4.0), (0.6, 1.0)]
    check("FLAD: fast and as good as every anchor within its lag -> -L", S.flad_reward(0.85, 3.0, anchors) == -3.0)
    check("FLAD: dominated by a not-slower anchor -> -L_MAX", S.flad_reward(0.7, 3.0, anchors) == -S.L_MAX)
    check("FLAD: faster than every anchor -> -L", S.flad_reward(0.3, 0.5, anchors) == -0.5)
    check("FLAD: slow and worse than the best -> -L_MAX", S.flad_reward(0.85, 5.0, anchors) == -S.L_MAX)

    # latency
    check(f"latency of GOOD (mean of 1.0, 1.4, 1.0) = {cr2.lat:.3f}", abs(cr2.lat - (1.0 + 1.4 + 1.0) / 3) < 1e-6)
    print("ALL STAMP TESTS PASSED")


if __name__ == "__main__":
    main()

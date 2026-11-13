"""Tests of joint_select.py (the pre-registered early-stop / selection rule of (b)), CPU only, synthetic outputs.

    python experiments/mist_w3po_v1/scripts/test_joint_select.py
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import joint_select as S  # noqa: E402

REF = {"zh2en": 3.0, "en2zh": 2.0}


def pt(step, jz, je=None, lz=3.0, le=2.0, oz=20.0, oe=20.0, hz=None, he=None, status="ok"):
    je = jz if je is None else je
    p = {"step": step, "status": status, "names": {"zh2en": f"z{step}", "en2zh": f"e{step}"}, "zh2en": None, "en2zh": None}
    if status == "ok":
        p["zh2en"] = {"Q": jz, "O": oz, "L": lz, "H": hz}
        p["en2zh"] = {"Q": je, "O": oe, "L": le, "H": he}
    return p


def seq(js, **kw):
    return [pt(500 * (i + 1), j, **kw) for i, j in enumerate(js)]


def test_no_stop_before_smin():
    r = S.decide(seq([20.0] * 5), REF, "s1")          # 500..2500, flat after 500
    assert r["action"] == "continue" and r["prefix_end"] == 2500 and r["last_improvement"] == 500, r
    r = S.decide(seq([20.0] * 6), REF, "s1")          # 3000: 3000 - 500 >= 1000
    assert r["action"] == "stop" and r["decision_step"] == 3000 and r["chosen"] == 3000, r   # ties -> later step


def test_patience_and_choice():
    js = [20, 21, 22, 23, 24, 25, 26, 27, 27.1, 27.15]      # improvements to 4000, then +0.1, +0.15 (< eps)
    r = S.decide(seq(js), REF, "s1")
    assert r["action"] == "stop" and r["decision_step"] == 5000 and r["last_improvement"] == 4000, r
    assert r["chosen"] == 5000 and r["best_J"] == 27, r      # the highest J, also a non-improving point
    r = S.decide(seq(js[:9]), REF, "s1")
    assert r["action"] == "continue", r


def test_eps_strict_and_best_not_moved():
    # 20.2 is not > 20 + 0.2 (strict); best stays 20, so 20.3 improves (it would not if 20.2 had moved best)
    assert 20 + S.EPS == 20.2
    r = S.decide(seq([20, 20.2, 20.3, 20.45, 20.45, 20.45]), REF, "s1")
    imp = [p["step"] for p in r["points"] if p.get("improved")]
    assert imp == [500, 1500], imp
    assert r["action"] == "stop" and r["decision_step"] == 3000 and r["chosen"] == 3000, r


def test_pending_cuts_prefix():
    pts = seq([20, 21, 22, 23, 24]) + [pt(3000, 0, status="pending"), pt(3500, 30)]
    r = S.decide(pts, REF, "s1")
    assert r["prefix_end"] == 2500 and r["action"] == "continue", r
    pts = seq([20] * 5) + [pt(3500, 30)]               # 3000 not saved / not passed: same
    assert S.decide(pts, REF, "s1")["prefix_end"] == 2500


def test_infeasible_and_failed():
    pts = seq([20, 21, 22, 23, 24, 25])
    pts[5] = pt(3000, 40, lz=3.2)                      # 3.2 > 3.0 + 0.1: infeasible, no improvement
    pts[4] = pt(2500, 0, status="failed")              # failed: no improvement, not a candidate
    pts += [pt(3500, 26, le=2.1)]                       # 2.1 <= 2.0 + 0.1: feasible (evaluated, beyond the decision)
    r = S.decide(pts, REF, "s1")
    imp = [p["step"] for p in r["points"] if p.get("improved")]
    assert imp == [500, 1000, 1500, 2000], imp
    assert r["action"] == "stop" and r["decision_step"] == 3000 and r["chosen"] == 3500, r
    assert 3000 not in r["candidates"] and 2500 not in r["candidates"] and 3500 in r["candidates"], r
    pts[6] = pt(3500, 26, le=2.11)                      # infeasible -> best feasible J = 23 at 2000
    r = S.decide(pts, REF, "s1")
    assert r["chosen"] == 2000, r


def test_ok_with_missing_outputs_is_failed():
    pts = seq([20] * 6)
    pts[0]["en2zh"] = None
    r = S.decide(pts, REF, "s1")
    assert r["points"][0]["status"] == "failed" and r["last_improvement"] == 1000, r
    assert r["action"] == "stop" and r["decision_step"] == 3000, r


def test_no_feasible():
    r = S.decide(seq([20] * 6, lz=9.0), REF, "s1")
    assert r["action"] == "no_feasible" and "chosen" not in r, r


def test_ceiling():
    r = S.decide(seq([20 + i for i in range(30)]), REF, "s1")
    assert r["prefix_end"] == 15000 and r["action"] == "ceiling", r


def test_ties():
    base = seq([20, 21, 22, 23, 24, 25])                 # improvement at 3000, then flat 25.004 / 25.001
    pts = base + [pt(3500, 25.004, oz=21, hz=1.0, he=1.0), pt(4000, 25.001, oz=20, hz=0.9, he=0.9)]
    r1 = S.decide([dict(p) for p in pts], REF, "s1")     # s1: no H -> higher O wins (3500)
    assert r1["action"] == "stop" and r1["chosen"] == 3500, r1
    r2 = S.decide([dict(p) for p in pts], REF, "s2")     # s2: lower H wins (4000)
    assert r2["chosen"] == 4000, r2
    pts = base + [pt(3500, 25.0, lz=3.05), pt(4000, 25.0, lz=3.0)]   # equal J, O: lower L - L_ref (4000)
    assert S.decide(pts, REF, "s1")["chosen"] == 4000
    pts = base + [pt(3500, 25.0, lz=2.9), pt(4000, 25.0, lz=3.0)]
    assert S.decide(pts, REF, "s1")["chosen"] == 3500
    pts = base + [pt(3500, 25.0, hz=0.5, he=None), pt(4000, 25.0, hz=0.9, he=0.9)]   # missing H counts as +inf
    assert S.decide(pts, REF, "s2")["chosen"] == 4000


def test_beyond_prefix_is_candidate():
    pts = seq([20, 21, 22, 23, 24, 25, 25, 25]) + [pt(4500, 0, status="pending"), pt(5000, 25.5)]
    r = S.decide(pts, REF, "s1")
    assert r["decision_step"] == 4000 and r["chosen"] == 5000, r


def test_main_files():
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        S.DEC, S.TE = d / "decode", d / "t_eval"

        def write(name, q, o, lat, h=None):
            (S.DEC / name).mkdir(parents=True)
            (S.DEC / name / "score_text.json").write_text(json.dumps({"tcpBLEU_all": q, "tcpBLEU_ov": o, "tcpBLEU_1spk": q}))
            (S.TE / name).mkdir(parents=True)
            (S.TE / name / f"textlat_{name}.json").write_text(json.dumps({"unit": {"text_laal_online": {"all": {"mean": lat}}}}))
            if h is not None:
                (S.TE / name / f"heldout_{name}.json").write_text(json.dumps({"ce_label": h}))

        args = ["--stage", "s2", "--ref-zh2en", "rz", "--ref-en2zh", "re", "--out", str(d / "sel.json")]
        import contextlib
        import io
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            S.main(args + ["--cand", "500", "z500", "e500", "pending"])
        assert json.loads(buf.getvalue())["action"] == "wait_ref"
        write("rz", 0, 0, 3.0)
        write("re", 0, 0, 2.0)
        for i in range(6):
            s = 500 * (i + 1)
            write(f"z{s}", 20.0, 18.0, 3.0, h=1.0)
            write(f"e{s}", 30.0, 28.0, 2.0, h=1.0 if s != 1500 else 0.5)
        c = []
        for i in range(5):
            s = 500 * (i + 1)
            c += ["--cand", str(s), f"z{s}", f"e{s}", "ok"]
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            S.main(args + c)
        assert json.loads(buf.getvalue())["action"] == "continue" and not (d / "sel.json").exists()
        c += ["--cand", "3000", "z3000", "e3000", "ok"]
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            S.main(args + c)
        r = json.loads(buf.getvalue())
        assert r["action"] == "stop" and r["chosen"] == 1500, r          # all J equal: lowest mean H
        assert r["chosen_names"] == {"zh2en": "z1500", "en2zh": "e1500"}
        assert json.load(open(d / "sel.json"))["chosen"] == 1500
        try:
            S.main(args + c)
            raise AssertionError("overwrote")
        except SystemExit:
            pass


def test_full_wait_and_select():
    # --full-steps 6000 (修正 4): no decision before every point 500 .. 6000 is resolved; never an early stop
    js = [20, 21, 22, 23, 24, 25, 25, 25, 25, 25, 25, 24]
    r = S.decide_full(seq(js[:11]), "s2", 6000)
    assert r["action"] == "continue" and r["prefix_end"] == 5500, r
    pts = seq(js[:11]) + [pt(6000, 24, status="pending")]
    assert S.decide_full(pts, "s2", 6000)["action"] == "continue"
    pts = seq(js[:5]) + seq(js)[6:]                  # 3000 not listed yet
    r = S.decide_full(pts, "s2", 6000)
    assert r["action"] == "continue" and r["prefix_end"] == 2500, r
    r = S.decide_full(seq(js), "s2", 6000)
    assert r["action"] == "select" and r["decision_step"] == 6000 and r["chosen"] == 5500, r   # J 25 ties -> later step
    assert r["chosen_names"] == {"zh2en": "z5500", "en2zh": "e5500"} and r["ref_L"] == {"zh2en": 3.0, "en2zh": 2.0}


def test_full_feasible_vs_last():
    # reference = the run's own last point: L_d <= L_d(6000) + 0.1 in each direction
    pts = seq([20] * 11) + [pt(6000, 20, lz=2.5, le=1.5)]
    pts[2] = pt(1500, 30, lz=2.6, le=1.6)            # within +0.1 of the last point in both directions
    pts[3] = pt(2000, 31, lz=2.61, le=1.5)           # zh2en over the cap
    pts[4] = pt(2500, 32, lz=2.5, le=1.61)           # en2zh over the cap
    r = S.decide_full(pts, "s2", 6000)
    assert r["action"] == "select" and r["chosen"] == 1500, r
    assert sorted(r["candidates"]) == [1500, 6000], r
    pts[5] = pt(3000, 29, status="failed")
    assert S.decide_full(pts, "s2", 6000)["chosen"] == 1500


def test_full_ties_and_no_last():
    pts = seq([20] * 12, hz=1.0, he=1.0)
    pts[1] = pt(1000, 20.004, hz=0.5, he=0.5)        # round(J, 2) equal -> lower mean H
    assert S.decide_full(pts, "s2", 6000)["chosen"] == 1000
    pts = seq([20] * 11) + [pt(6000, 0, status="failed")]
    r = S.decide_full(pts, "s2", 6000)
    assert r["action"] == "no_last" and "chosen" not in r, r
    pts = seq([20] * 11) + [pt(6000, 20)]
    pts[-1]["en2zh"] = None                          # ok with missing outputs = failed
    assert S.decide_full(pts, "s2", 6000)["action"] == "no_last"


def test_first_step():
    # 修正 4 stage 2: points 3000, 3500, ...; the first feasible one improves; stop after 2 non-improving points
    sq = lambda js: [pt(3000 + 500 * i, j) for i, j in enumerate(js)]
    r = S.decide(sq([20, 20.1]), REF, "s2", 3000)
    assert r["action"] == "continue" and r["prefix_end"] == 3500 and r["last_improvement"] == 3000, r
    r = S.decide(sq([20, 20.1, 20.2]), REF, "s2", 3000)
    assert r["action"] == "stop" and r["decision_step"] == 4000 and r["chosen"] == 4000 and r["first_step"] == 3000, r
    r = S.decide(sq([20, 21, 22, 23, 23, 23]) + [pt(1000, 40)], REF, "s2", 3000)   # points before 3000 are ignored
    assert r["action"] == "stop" and r["decision_step"] == 5500 and r["chosen"] == 5500, r
    assert all(p["step"] >= 3000 for p in r["points"])
    r = S.decide([pt(3500, 20)], REF, "s2", 3000)                   # 3000 missing: no prefix
    assert r["action"] == "continue" and r["prefix_end"] == 0, r
    js = [20 + i for i in range(25)]                                # still improving at 15000: ceiling
    r = S.decide(sq(js), REF, "s2", 3000)
    assert r["action"] == "ceiling" and r["prefix_end"] == 15000, r
    pts = sq([20, 21, 21, 21])
    pts[0] = pt(3000, 25, lz=3.2)                                    # infeasible: not an improvement
    r = S.decide(pts, REF, "s2", 3000)
    assert r["action"] == "stop" and r["decision_step"] == 4500 and r["last_improvement"] == 3500 and r["chosen"] == 4500, r
    for bad in (["--first-step", "750"], ["--first-step", "3000", "--full-steps", "6000"]):
        try:
            S.main(["--stage", "s2", "--ref-zh2en", "a", "--ref-en2zh", "b"] + bad)
            raise AssertionError(bad)
        except SystemExit:
            pass
    assert S.decide(seq([20.0] * 6), REF, "s1") == S.decide(seq([20.0] * 6), REF, "s1", 500)   # default unchanged


def test_full_main_files():
    import contextlib
    import io
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        S.DEC, S.TE = d / "decode", d / "t_eval"
        for i in range(4):
            s = 500 * (i + 1)
            for n, q, lat in ((f"z{s}", 20.0 + i, 3.0), (f"e{s}", 30.0, 2.0)):
                (S.DEC / n).mkdir(parents=True)
                (S.DEC / n / "score_text.json").write_text(json.dumps({"tcpBLEU_all": q, "tcpBLEU_ov": q, "tcpBLEU_1spk": q}))
                (S.TE / n).mkdir(parents=True)
                (S.TE / n / f"textlat_{n}.json").write_text(json.dumps({"unit": {"text_laal_online": {"all": {"mean": lat}}}}))
        out = d / "sel.json"
        c = []
        for i in range(4):
            s = 500 * (i + 1)
            c += ["--cand", str(s), f"z{s}", f"e{s}", "ok"]
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            S.main(["--stage", "s2", "--full-steps", "2500", "--out", str(out)] + c)   # 2500 not listed: wait
        assert json.loads(buf.getvalue())["action"] == "continue" and not out.exists()
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            S.main(["--stage", "s2", "--full-steps", "2000", "--out", str(out)] + c)
        r = json.loads(buf.getvalue())
        assert r["action"] == "select" and r["chosen"] == 2000 and json.load(open(out))["chosen"] == 2000, r
        for bad in (["--stage", "s2"] + c, ["--stage", "s2", "--full-steps", "750"] + c):   # refs needed / bad N
            try:
                S.main(bad)
                raise AssertionError(bad)
            except SystemExit:
                pass


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
    print(f"ALL PASSED ({len(tests)} tests)")


if __name__ == "__main__":
    main()

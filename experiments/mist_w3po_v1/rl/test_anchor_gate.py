"""Tests for advantage.admit_anchors (anchor gate) on hand-made values and on the pilot's real degenerate Lag1 anchor
(step 6, ES2015a_on_0014, R -27.5). Run (CPU):
    python -m experiments.mist_w3po_v1.rl.test_anchor_gate [<run dir>]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from experiments.mist_w3po_v1.rl import stamp as S
from experiments.mist_w3po_v1.rl.advantage import admit_anchors, group_advantages
from experiments.mist_w3po_v1.rl.check_anchor_rewards import anchor_trajs
from experiments.mist_w3po_v1.rl.rollout import load_window


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    assert cond, name


def unit_cases():
    check("none admits all", admit_anchors([0.3, 0.5], [-27.0, 0.6, 0.1], "none") == [True, True, True])
    check("minpol drops below min policy", admit_anchors([0.3, 0.5], [-27.0, 0.6, 0.3], "minpol") == [False, True, True])
    check("minpol, no policy -> all", admit_anchors([], [-1.0], "minpol") == [True])
    try:
        admit_anchors([0.1], [0.1], "bogus")
        check("bad rule raises", False)
    except ValueError:
        check("bad rule raises", True)


def real_case(run: Path, wid: str = "ES2015a_on_0014"):
    args = json.loads((run / "args.json").read_text())
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args["base"])
    w = next(x for x in map(json.loads, open(args["windows"], encoding="utf-8")) if x["id"] == wid)
    rec = None
    for f in sorted((run / "anchors").glob("rank*.jsonl")):
        for l in open(f, encoding="utf-8"):
            if f'"{wid}"' in l:
                rec = json.loads(l)
                break
        if rec:
            break
    check("record found", rec is not None)
    units = S.units_of(w["utts"])
    dur = w["dur_ms"] / 1000
    _, post = load_window(w)
    trs = anchor_trajs(rec, w, tok)
    cred = lambda tr: S.credit(tr.segs, units, dur, tr.wait_reads, [rd.t_ms / 1000 for rd in tr.reads], post=post)  # noqa: E731
    cs = [cred(tr) for tr in trs]
    check("Lag1 anchor degenerate (R < -10)", cs[0].R < -10)
    # stand-in policy group: the two good anchors, each twice (policy-like members with R ~0.7)
    members, mc = [trs[1], trs[2], trs[1], trs[2]], [cs[1], cs[2], cs[1], cs[2]]
    ok = admit_anchors([c.R for c in mc], [c.R for c in cs], "minpol")
    check("minpol drops only Lag1", ok == [False, True, True])
    aq = [(c.R, c.lat) for c in cs]

    def mag(group, creds):
        adv = group_advantages("w3po", group, creds, units, dur, aq)
        return float(np.mean([abs(t[4]) for r in adv["tok"][0] for t in r] + [abs(t[4]) for r in adv["tok"][1] for t in r]))
    a_none = mag(members + trs, mc + cs)
    a_gate = mag(members + [t for t, k in zip(trs, ok) if k], mc + [c for c, k in zip(cs, ok) if k])
    print(f"mean |what advantage| of the first two members: gate none {a_none:.4f}, minpol {a_gate:.4f}")
    check("degenerate anchor shrinks policy advantages; gate restores them", a_gate > 2 * a_none)


if __name__ == "__main__":
    unit_cases()
    real_case(Path(sys.argv[1] if len(sys.argv) > 1 else "eval_outputs/mist_w3po_v1/rl/pilot_w3po_v8"))
    print("ALL PASS")

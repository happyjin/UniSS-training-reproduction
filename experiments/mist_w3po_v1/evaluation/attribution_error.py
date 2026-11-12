"""Speaker-attribution error of a dev decode (SpecClose prereg §5 A1; CPU, read-only, dev only).

Per hyp segment with text (a, b = window start + start_s / end_s, as metric_headroom.build_segs): the annotated
utterance of the meeting (meeting_score.universe, zh_mix_v8 clock, hesitations included) with the largest time overlap
(> 0, no collar) gives the true speaker; the segment's slot is mapped to a reference speaker by the scorer's own
meeting-level map (metric_headroom.assign -> mp, the tcpBLEU speaker matching). A segment is MISATTRIBUTED when its
mapped speaker differs from the true speaker (an unmapped slot counts as misattributed). Word-weighted rate over the
segments that overlap an utterance; segments overlapping none are counted apart (no_overlap_words).
With --ref: the same for a reference decode, and the paired difference cand - ref with a meeting bootstrap
(rate of pooled sums per resample; 2000 resamples, seed 20261112).

    python -m experiments.mist_w3po_v1.evaluation.attribution_error --cand <decode dir> --cand-win <windows jsonl> \
        [--ref <decode dir> --ref-win <windows jsonl>] --out <json>
Refuses to overwrite <out>.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from experiments.mist_w3po_v1.evaluation import metric_headroom as M


def per_meeting(run: str, win: str) -> dict:
    from meeting_score import universe
    W = [json.loads(l) for l in open(M.REPO / win, encoding="utf-8")]
    H = {h["id"]: h for h in map(json.loads, open(Path(run) / "hyp.jsonl", encoding="utf-8"))}
    W = [w for w in W if w["id"] in H]
    RW = [json.loads(l) for l in open(M.REPO / M.REF_WIN, encoding="utf-8")]
    U = universe(M.DR, {w["meeting"] for w in W})
    sc = M.scored_utts(W, RW)
    segs = M.build_segs(W, H)
    out = {}
    for m in sorted(segs):
        A = M.assign(m, segs[m], sc, U[m])
        mp = A["mp"]
        st = {"words": 0, "mis_words": 0, "no_overlap_words": 0, "segs": 0, "mis_segs": 0}
        for s in segs[m]:
            if not s["text"]:
                continue
            nw = len(s["text"].split())
            best, bu = 0.0, None
            for u in U[m]:
                ov = min(s["b"], u[3]) - max(s["a"], u[2])
                if ov > best:
                    best, bu = ov, u
            if bu is None:
                st["no_overlap_words"] += nw
                continue
            st["words"] += nw
            st["segs"] += 1
            if mp.get(s["spk"]) != bu[1]:
                st["mis_words"] += nw
                st["mis_segs"] += 1
        out[m] = st
    return out


def rate(pm: dict, ms: list) -> float:
    w = sum(pm[m]["words"] for m in ms)
    return sum(pm[m]["mis_words"] for m in ms) / max(w, 1)


def summary(pm: dict) -> dict:
    ms = sorted(pm)
    tot = {k: sum(pm[m][k] for m in ms) for k in ("words", "mis_words", "no_overlap_words", "segs", "mis_segs")}
    return {**tot, "mis_rate_words": round(rate(pm, ms), 5), "mis_rate_segs": round(tot["mis_segs"] / max(tot["segs"], 1), 5)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cand", required=True)
    ap.add_argument("--cand-win", required=True)
    ap.add_argument("--ref", default="")
    ap.add_argument("--ref-win", default="")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if Path(a.out).exists():
        raise SystemExit(f"拒绝覆盖已有输出: {a.out}")
    c = per_meeting(a.cand, a.cand_win)
    res = {"cand": a.cand, "cand_win": a.cand_win, "cand_summary": summary(c), "cand_per_meeting": c}
    if a.ref:
        r = per_meeting(a.ref, a.ref_win or a.cand_win)
        ms = sorted(set(c) & set(r))
        rng = np.random.default_rng(20261112)
        diffs = []
        for _ in range(a.n):
            s = list(rng.choice(ms, len(ms)))
            diffs.append(rate(c, s) - rate(r, s))
        d = rate(c, ms) - rate(r, ms)
        res.update({"ref": a.ref, "ref_win": a.ref_win or a.cand_win, "ref_summary": summary(r), "ref_per_meeting": r,
                    "diff_mis_rate_words": round(d, 5), "ci95": [round(float(np.percentile(diffs, 2.5)), 5),
                                                             round(float(np.percentile(diffs, 97.5)), 5)],
                    "p_gt0": round(float(np.mean(np.asarray(diffs) > 0)), 4), "meetings": len(ms)})
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "x", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    print(json.dumps({k: v for k, v in res.items() if not k.endswith("per_meeting")}))


if __name__ == "__main__":
    main()

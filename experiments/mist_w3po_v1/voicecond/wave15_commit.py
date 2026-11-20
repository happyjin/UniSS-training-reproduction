"""Robust wave 15 CV-LA, commit statistics (prereg: auxiliary, no gate), wave8_aul_commit.py's convention: per system,
direction, set (a / b), condition, the mean over sentences with a non-final commit of the share of committed non-final
units the reference does not support (1 - LCS(committed, reference) / |committed|, wave3_d_cpu.toks / lcs), and the mean
non-final committed share; cvla - control paired over sentences (wave3_d_cpu.boot_diff, seed 0). For the CV-LA run also
the share of non-final reads where v1 cut LA2's commit (cv_new < la2_new) and the units withheld.
--group 1: cvla_base vs base (S1); --group 2: cvla_unist vs unist_wise05.
    <uniss-eval python> experiments/mist_w3po_v1/voicecond/wave15_commit.py --group 1 --out eval_outputs/mist_w3po_v1/wave15/commit_g1.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from experiments.mist_w3po_v1.voicecond.wave3_d_cpu import boot_diff, lcs, toks  # noqa: E402

ROOTS = {"base": REPO / "eval_outputs/mist_w3po_v1/wave2_S1",
         "cvla_base": REPO / "eval_outputs/mist_w3po_v1/wave15/eval_cvla_base",
         "unist_wise05": REPO / "eval_outputs/mist_w3po_v1/wave14/eval_unist_wise05",
         "cvla_unist": REPO / "eval_outputs/mist_w3po_v1/wave15/eval_cvla_unist"}
GROUP = {1: ("base", "cvla_base"), 2: ("unist_wise05", "cvla_unist")}
TGT = {"en2zh": "zh", "zh2en": "en"}


def jl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8")]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", type=int, choices=(1, 2), required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    ctl, arm = GROUP[a.group]
    res = {"group": a.group, "control": ctl, "arm": arm}
    for d in ("en2zh", "zh2en"):
        lang = TGT[d]
        S1 = ROOTS["base"]
        meta = {m["id"]: m for m in jl(S1 / f"data/meta_{d}.jsonl")}
        ref = {r["id"]: r["translation"] for r in jl(S1 / f"data/items_{d}.jsonl")}
        per, rows = {}, {}
        for sysn in (ctl, arm):
            rows[sysn] = {r["id"]: r for r in jl(ROOTS[sysn] / f"thinker_{d}" / f"blocks_{d}_w1.jsonl")}
            per[sysn] = {}
            for i in meta:
                bl = rows[sysn][i]["blocks"]
                c = toks("".join(b["text"] for b in bl if not b.get("final")), lang)
                n_all = len(toks("".join(b["text"] for b in bl), lang))
                per[sysn][i] = (np.nan if not c else 1.0 - lcs(c, toks(ref[i], lang)) / len(c), len(c) / max(1, n_all))
        res[d] = {}
        for set_ in ("a", "b"):
            for cond in ("clean", "noisy"):
                ids = sorted(i for i, m in meta.items() if m["set"] == set_ and m["cond"] == cond)
                r = {"n": len(ids)}
                for sysn in (ctl, arm):
                    v = np.array([per[sysn][i][0] for i in ids])
                    r[sysn] = {"unsupported_share": round(float(np.nanmean(v)), 4),
                               "nonfinal_share": round(float(np.mean([per[sysn][i][1] for i in ids])), 4)}
                rng = np.random.default_rng(0)
                X = np.array([per[ctl][i][0] for i in ids])
                Y = np.array([per[arm][i][0] for i in ids])
                r[f"{arm}-{ctl}_unsupported"] = [round(v, 4) for v in boot_diff(X, Y, rng)]
                X = np.array([per[ctl][i][1] for i in ids])
                Y = np.array([per[arm][i][1] for i in ids])
                r[f"{arm}-{ctl}_nonfinal"] = [round(v, 4) for v in boot_diff(X, Y, rng)]
                recs = [x for i in ids for x in rows[arm][i]["drafts"] if "cv_new" in x]
                r["cvla_reads"] = {"nonfinal_reads": len(recs),
                                   "cut_share": round(float(np.mean([x["cv_new"] < x["la2_new"] for x in recs])), 4),
                                   "v1_differs_share": round(float(np.mean([x["draft_v1"] != x["draft"] for x in recs])), 4),
                                   "la2_units": int(sum(x["la2_new"] for x in recs)),
                                   "cv_units": int(sum(x["cv_new"] for x in recs))}
                res[d][f"{set_}/{cond}"] = r
    with open(out, "x", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=1)
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()

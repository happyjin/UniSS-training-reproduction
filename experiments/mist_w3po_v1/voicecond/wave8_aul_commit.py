"""Robust wave 8 AUL, auxiliary commit statistic (prereg: no gate): the share of LA2-committed units before the final block
that the reference does not support -- wave3_d_cpu's regret (1 - LCS(committed, x) / |committed|, its tokenisation
toks()) with x = the reference instead of the whole-sentence read. Per system (S1 base, wave 5 erm_wise05, aul,
aul_wise05), direction, set (a / b), condition: the mean over sentences with a non-final commit, the mean non-final
committed share; aul_wise05 - erm_wise05 and aul_wise05 - base paired over sentences (bootstrap 2000, seed 0).
    <uniss-eval python> experiments/mist_w3po_v1/voicecond/wave8_aul_commit.py --out eval_outputs/mist_w3po_v1/wave8/AUL/commit_stats.json
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
         "erm_wise05": REPO / "eval_outputs/mist_w3po_v1/wave5/M/eval_erm_wise05",
         "aul": REPO / "eval_outputs/mist_w3po_v1/wave8/AUL/eval_aul",
         "aul_wise05": REPO / "eval_outputs/mist_w3po_v1/wave8/AUL/eval_aul_wise05"}
TGT = {"en2zh": "zh", "zh2en": "en"}


def jl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8")]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    res = {}
    for d in ("en2zh", "zh2en"):
        lang = TGT[d]
        S1 = ROOTS["base"]
        meta = {m["id"]: m for m in jl(S1 / f"data/meta_{d}.jsonl")}
        ref = {r["id"]: r["translation"] for r in jl(S1 / f"data/items_{d}.jsonl")}
        per = {}
        for sysn, root in ROOTS.items():
            f = root / f"thinker_{d}" / f"blocks_{d}_w1.jsonl"
            if not f.exists():
                continue
            bl = {r["id"]: r["blocks"] for r in jl(f)}
            per[sysn] = {}
            for i in meta:
                c = toks("".join(b["text"] for b in bl[i] if not b.get("final")), lang)
                n_all = len(toks("".join(b["text"] for b in bl[i]), lang))
                per[sysn][i] = (np.nan if not c else 1.0 - lcs(c, toks(ref[i], lang)) / len(c), len(c) / max(1, n_all))
        res[d] = {}
        for set_ in ("a", "b"):
            for cond in ("clean", "noisy"):
                ids = sorted(i for i, m in meta.items() if m["set"] == set_ and m["cond"] == cond)
                r = {"n": len(ids)}
                for sysn in per:
                    v = np.array([per[sysn][i][0] for i in ids])
                    r[sysn] = {"unsupported_share": round(float(np.nanmean(v)), 4),
                               "nonfinal_share": round(float(np.mean([per[sysn][i][1] for i in ids])), 4)}
                rng = np.random.default_rng(0)
                for x, y in (("erm_wise05", "aul_wise05"), ("base", "aul_wise05"), ("base", "erm_wise05")):
                    if x in per and y in per:
                        X = np.array([per[x][i][0] for i in ids])
                        Y = np.array([per[y][i][0] for i in ids])
                        r[f"{y}-{x}"] = [round(v, 4) for v in boot_diff(X, Y, rng)]
                res[d][f"{set_}/{cond}"] = r
    with open(out, "x", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=1)
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()

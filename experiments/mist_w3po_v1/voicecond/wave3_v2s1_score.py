"""Robust wave 3, V2 second-seed confirmation (robust_wave3_single_prereg.zh-CN.md, "V2 第二种子确认").

seed 1: v2_s1 vs B_s1 (wave3_combos/V/{v2_s1,B_s1}/talker_<d>), gate on the main cells {a, b_noxt} x {en2zh, zh2en}, noisy:
  ASR-BLEU diff >= -0.3 and CI high >= 0;  PESQ paired median diff CI low > 0.  LAAL / similarity reported only.
two seeds pooled (reported, no gate): per item PESQ / similarity / LAAL = mean of the two seeds; ASR-BLEU from the two
seeds' sufficient statistics summed; v2 + v2_s1 vs B + B_s1.
Units, bootstrap (2000, seed 0) and metric files as wave3_v_score.py.

    qwen3omni python -m experiments.mist_w3po_v1.voicecond.wave3_v2s1_score --out <json>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from experiments.mist_w3po_v1.voicecond.wave2_s1_score import LANG, Bleu, jl, laal_defs  # noqa: E402
from experiments.mist_w3po_v1.voicecond.wave3_v_score import S1, load  # noqa: E402

NB = 2000


def compare(meta, ref, items, Xs, Ys, lang, unit_key="stream"):
    """arm runs Xs vs reference runs Ys (lists of per-item dicts, one per seed)"""
    bl = Bleu(lang)
    units = sorted({meta[i][unit_key] for i in items})
    up = {u: k for k, u in enumerate(units)}
    pu = np.array([up[meta[i][unit_key]] for i in items])
    nU = len(units)

    def ustats(Ps):
        U = None
        for P in Ps:
            st = bl.stats([P[i]["asr"] for i in items], [ref[i] for i in items])
            u = np.zeros((nU, st.shape[1]), np.int64)
            np.add.at(u, pu, st)
            U = u if U is None else U + u
        return U

    def avg(Ps, k):
        v = np.full(len(items), np.nan)
        for n, i in enumerate(items):
            xs = [P[i][k] for P in Ps if P[i][k] is not None]
            if xs:
                v[n] = float(np.mean(xs))
        return v

    def unit_mean(v):
        s, c = np.zeros(nU), np.zeros(nU)
        ok = ~np.isnan(v)
        np.add.at(s, pu[ok], v[ok])
        np.add.at(c, pu[ok], 1)
        return s, c

    rng = np.random.default_rng(0)
    W = [np.ones(nU)] + [np.bincount(rng.integers(0, nU, nU), minlength=nU).astype(float) for _ in range(NB)]
    out = {"n_items": len(items), "n_units": nU}
    UX, UY = ustats(Xs), ustats(Ys)
    vals = [bl.score((UX * w[:, None]).sum(0)) - bl.score((UY * w[:, None]).sum(0)) for w in W]
    out["asr"] = {"arm": round(bl.score(UX.sum(0)), 3), "ref": round(bl.score(UY.sum(0)), 3), "diff": round(vals[0], 3),
                  "ci95": [round(float(np.percentile(vals[1:], 2.5)), 3), round(float(np.percentile(vals[1:], 97.5)), 3)]}
    for k in ("pesq", "sim"):
        xv, yv = avg(Xs, k), avg(Ys, k)
        s, c = unit_mean(xv - yv)
        idx = np.nonzero(c > 0)[0]
        um = s[idx] / c[idx]
        boots = [float(np.median(um[rng.integers(0, len(idx), len(idx))])) for _ in range(NB)]
        out[k] = {"arm_median": round(float(np.nanmedian(xv)), 4), "ref_median": round(float(np.nanmedian(yv)), 4),
                  "paired_median_diff": round(float(np.median(um)), 4),
                  "ci95": [round(float(np.percentile(boots, 2.5)), 4), round(float(np.percentile(boots, 97.5)), 4)]}
    for k in ("d1", "d2"):
        xv, yv = avg(Xs, k), avg(Ys, k)
        s, c = unit_mean(xv - yv)
        vals = [float((s * w).sum() / max((c * w).sum(), 1e-9)) for w in W]
        out[k] = {"arm_mean": round(float(np.nanmean(xv)), 4), "ref_mean": round(float(np.nanmean(yv)), 4),
                  "diff": round(vals[0], 4),
                  "ci95": [round(float(np.percentile(vals[1:], 2.5)), 4), round(float(np.percentile(vals[1:], 97.5)), 4)]}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    LD = laal_defs()
    res = {"seed1": {}, "pooled": {}, "gate": {}}
    for d in ("en2zh", "zh2en"):
        meta = {m["id"]: m for m in jl(S1 / "data" / f"meta_{d}.jsonl")}
        ref = {r["id"]: r["translation"] for r in jl(S1 / "data" / f"items_{d}.jsonl")}
        P = {arm: load(arm, d, LD) for arm in ("B", "v2", "B_s1", "v2_s1")}
        common = set.intersection(*[set(p) for p in P.values()])
        for set_ in ("a", "b", "b_noxt"):
            for cond in ("noisy", "clean"):
                items = sorted(i for i in common if meta[i]["set"] == set_[0] and meta[i]["cond"] == cond
                               and (set_ != "b_noxt" or not meta[i]["ovl"]))
                key = f"{d}/{set_}/{cond}"
                res["seed1"][key] = compare(meta, ref, items, [P["v2_s1"]], [P["B_s1"]], LANG[d])
                res["pooled"][key] = compare(meta, ref, items, [P["v2"], P["v2_s1"]], [P["B"], P["B_s1"]], LANG[d])
                if cond == "noisy" and set_ in ("a", "b_noxt"):
                    c = res["seed1"][key]
                    g = {"asr": c["asr"]["diff"] >= -0.3 and c["asr"]["ci95"][1] >= 0, "pesq": c["pesq"]["ci95"][0] > 0}
                    g["pass"] = g["asr"] and g["pesq"]
                    res["gate"][key] = g
    res["gate"]["confirmed"] = all(v["pass"] for k, v in res["gate"].items() if isinstance(v, dict))
    with open(out, "x", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=1)
    print(json.dumps(res["gate"], indent=1))


if __name__ == "__main__":
    main()

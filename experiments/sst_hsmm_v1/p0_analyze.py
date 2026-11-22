"""sst_hsmm_v1 P0 analysis (reports/sst_hsmm_v1/prereg.zh-CN.md 3.1) -> eval_outputs/sst_hsmm_v1/p0/summary.json.
P0a: text-unit LAAL (si_metrics.laal, units at their commit time capped at X) of plain LA2 vs the oracle (reference
     units at their token's lower-bound chunk); + the speech-dynamics proxy (hsmm.path_stats) of both.
     GO: mean(LA2 - oracle) >= 0.3 s in both directions (clean + noisy pooled).
P0b: emission sanity (median lp(g) - lp(g-1) > 0), token-mismatch count, timing -> training steps (2500, or fewer to stay
     within 30 h at 16 samples / step on 8 cards).
P0c: lambda grid with uniform-transition Viterbi on the P0b emissions; rule as in the prereg.
"""
from __future__ import annotations

import itertools
import json
import math
import multiprocessing as mp
import re
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from experiments.sst_hsmm_v1 import hsmm as H  # noqa: E402
from experiments.uniss_qwen3_omni_v1.evaluation.si_metrics import laal as si_laal, units as si_units  # noqa: E402

D = REPO / "eval_outputs/sst_hsmm_v1/data"
P0 = REPO / "eval_outputs/sst_hsmm_v1/p0"
LANG = {"en2zh": "cmn", "zh2en": "eng"}
PUNCT = re.compile(r"[，。,.!?！？；;：:、…]")
LAM_L = (0.02, 0.05, 0.1, 0.2, 0.5)
LAM_P = (0.5, 1.0, 2.0, 4.0)
_TOK = None


def tok():
    global _TOK
    if _TOK is None:
        from transformers import AutoTokenizer
        _TOK = AutoTokenizer.from_pretrained(str(REPO / "eval_outputs/mist_w3po_v1/wave17/merged_enc_wise05"))
    return _TOK


def jl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def unit_char_pos(text, lang):
    """first character index of every si_metrics unit, in order"""
    us = si_units(text, lang)
    low, cur, pos = text.lower(), 0, []
    for u in us:
        p = low.find(u.lower(), cur)
        p = cur if p < 0 else p
        pos.append(p)
        cur = p + max(len(u), 1)
    return pos


def oracle_laal(row, lang):
    ref = row["ref"]
    enc = tok()(ref, add_special_tokens=False, return_offsets_mapping=True)
    assert enc["input_ids"] == row["tok"]
    offs = enc["offset_mapping"]
    X = row["X"]
    times = []
    for p in unit_char_pos(ref, lang):
        k = next((j for j, (a, b) in enumerate(offs) if a <= p < b), len(offs) - 1)
        g = row["g"][k]
        times.append(X if g >= row["C"] else min(g * 1000, X))
    T = len(si_units(ref, lang))
    return si_laal(times, X, T) / 1000


def la2_measures(b, row, lang, dmean):
    X, C = row["X"], row["C"]
    T = len(si_units(row["ref"], lang))
    times, path, dbin, pu = [], [], [], []
    for bl in b["blocks"]:
        w = min(bl["write_ms"], X)
        times += [w] * len(si_units(bl["text"], lang))
        c = C if bl["write_ms"] >= X else max(1, math.ceil(bl["write_ms"] / 1000))
        for t in tok()(bl["text"], add_special_tokens=False)["input_ids"]:
            s = tok().decode([t])
            p = int(bool(PUNCT.search(s))) and not re.search(r"\w", s)
            path.append(c); pu.append(int(p)); dbin.append(0 if p else int(round(dmean / 100)))
    lat, npi = H.path_stats(path, dbin, pu, X) if path else (X / 1000, 0)
    return (si_laal(times, X, T) / 1000 if times else X / 1000), lat, npi


def _vit(args):
    key, row, lp = args
    out = {}
    dbin = [int(round(x / 100)) for x in row["d_ms"]]
    import torch
    torch.set_num_threads(1)
    lpt = torch.tensor(lp, dtype=torch.float32)
    for lL, lP in itertools.product(LAM_L, LAM_P):
        sc, path = H.lattice(lpt, None, row["g"], dbin, row["punct"], row["X"], lL, lP, None, viterbi=True, uniform=True)
        lat, npi = H.path_stats(path, dbin, row["punct"], row["X"])
        ll = float(sum(lp[c - 1, i] for i, c in enumerate(path))) / len(path)
        out[f"{lL}|{lP}"] = (lat, npi, ll)
    return key, out


def main():
    out_p = P0 / "summary.json"
    if out_p.exists():
        sys.exit(f"拒绝覆盖已有输出: {out_p}")
    res = {"P0a": {}, "P0b": {}, "P0c": {}}
    rows = {d: {r["id"]: r for r in jl(D / f"heldout_{d}.jsonl")} for d in LANG}
    dmean = {}
    for d in LANG:
        v = [x for r in jl(D / f"train_{d}.jsonl")[:5000] for x, p in zip(r["d_ms"], r["punct"]) if not p and x > 0]
        dmean[d] = float(np.mean(v))
    res["token_speech_ms_mean"] = dmean
    # ---- P0a
    la2stats = {}
    go = True
    for d, lang in LANG.items():
        blocks = {b["id"]: b for b in jl(P0 / "la2" / f"thinker_{d}" / f"blocks_{d}_w1.jsonl")}
        diff, per = [], {"clean": [], "noisy": []}
        for iid, b in blocks.items():
            cond, rid = iid.split("__", 1)
            row = rows[d][rid]
            l2, lat2, np2 = la2_measures(b, row, lang, dmean[d])
            lo = oracle_laal(row, lang)
            dif = l2 - lo
            diff.append(dif); per[cond].append((dif, l2, lo))
            la2stats[(d, cond, rid)] = (lat2, np2)
        res["P0a"][d] = {"n": len(diff), "laal_la2_minus_oracle_mean": float(np.mean(diff)),
                         **{f"{c}_mean_diff": float(np.mean([x[0] for x in v])) for c, v in per.items()},
                         **{f"{c}_laal_la2": float(np.mean([x[1] for x in v])) for c, v in per.items()},
                         **{f"{c}_laal_oracle": float(np.mean([x[2] for x in v])) for c, v in per.items()}}
        go &= res["P0a"][d]["laal_la2_minus_oracle_mean"] >= 0.3
    res["P0a"]["GO"] = bool(go)
    # ---- P0b
    sh = [json.load(open(p)) for p in sorted(P0.glob("shard_*.json"))]
    tim = [t for s in sh for t in s["timing"]]
    dg = [s["delta_g_median"] for s in sh if s["delta_g_median"] is not None]
    res["P0b"] = {"n": sum(s["n"] for s in sh), "mismatch": sum(s["mismatch"] for s in sh),
                  "delta_g_median_by_shard": dg, "sanity_pass": bool(np.median(dg) > 0), "timing": tim}
    sec = float(np.mean([t["sec"] for t in tim]))
    trows = [r for d in LANG for r in jl(D / f"train_{d}.jsonl")[:2000]]
    hrows = [r for d in LANG for r in rows[d].values()][:600]
    scale = np.mean([r["C"] * len(r["tok"]) for r in trows]) / np.mean([t["C"] * t["n"] for t in tim])
    step_sec = 2 * sec * scale * 1.15
    hours = 2500 * step_sec / 3600
    steps = 2500 if hours <= 30 else int(30 * 3600 / step_sec)
    ells = []
    for f in sorted((P0 / "em").glob("*.npz")):
        cond, rid = f.stem.split("__", 1)
        r = rows["en2zh"].get(rid) or rows["zh2en"][rid]
        lw = np.load(f)["lp_wait"].astype(np.float64)
        for i, gi in enumerate(r["g"]):
            for c in range(gi, r["C"]):            # writable, not the last chunk
                l1 = np.log1p(-min(np.exp(lw[c - 1, i]), 1 - 1e-6))
                ells.append(l1 - lw[c - 1, i])
    res["P0b"]["ell_median"] = float(np.median(ells)) if ells else None
    res["P0b"]["gate_bias_init"] = -float(np.median(ells)) if ells else 0.0
    res["P0b"].update({"sec_per_sample_timed": sec, "train_size_scale": float(scale), "est_step_sec": step_sec,
                       "est_hours_2500": hours, "steps": steps})
    # ---- P0c
    jobs = []
    for f in sorted((P0 / "em").glob("*.npz")):
        cond, rid = f.stem.split("__", 1)
        d = "en2zh" if rid in rows["en2zh"] else "zh2en"
        jobs.append(((d, cond, rid), rows[d][rid], np.load(f)["lp_tok"].astype(np.float32)))
    with mp.get_context("fork").Pool(96) as P:
        vit = dict(P.map(_vit, jobs, chunksize=4))
    keys = [k for k in vit if k in la2stats]
    la2_lat = float(np.mean([la2stats[k][0] for k in keys]))
    la2_np = float(np.mean([la2stats[k][1] for k in keys]))
    grid = {}
    for lL, lP in itertools.product(LAM_L, LAM_P):
        g = f"{lL}|{lP}"
        v = np.array([vit[k][g] for k in keys])
        grid[g] = {"lat": float(v[:, 0].mean()), "np_in": float(v[:, 1].mean()), "ll": float(v[:, 2].mean())}
    ok = {g: x for g, x in grid.items() if x["lat"] <= la2_lat - 0.3 and x["np_in"] <= 0.7 * la2_np}
    if ok:
        best = max(ok, key=lambda g: (ok[g]["ll"], -ok[g]["np_in"]))
        rule = "max ll among satisfying"
    else:
        best = min(grid, key=lambda g: grid[g]["lat"])
        rule = "none satisfies: lowest latency proxy"
    lL, lP = map(float, best.split("|"))
    res["P0c"] = {"n_items": len(keys), "la2_lat_proxy": la2_lat, "la2_np_in": la2_np, "grid": grid,
                  "chosen": {"lam_L": lL, "lam_P": lP, "rule": rule}}
    with open(out_p, "x") as fh:
        json.dump(res, fh, indent=1)
    print(json.dumps({"P0a": res["P0a"], "P0b": {k: v for k, v in res["P0b"].items() if k != "timing"},
                      "P0c": res["P0c"]["chosen"]}, indent=1))


if __name__ == "__main__":
    main()

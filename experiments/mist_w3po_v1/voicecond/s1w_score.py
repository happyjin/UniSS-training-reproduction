"""Robust S1 "witness algebra" scoring (reports/mist_w3po_v1/robust_s1_witness_prereg.zh-CN.md). Point estimates only.

Systems (Thinker text only, S1 dev, both directions): LA1 / LA2 / LA3 / AoV / EITHER from s1w_si_thinker.py
(<s1w>/<rule>/thinker_<d>), CV-LA = wave15/eval_cvla_base (existing), LA2_S1 = wave2_S1 (the original si_thinker_stream
la2 w1 run; batch-noise reference only, not on the curve). Per cell (direction x set a / b x condition clean / noisy):
  text BLEU  wave2_s1_score.Bleu (paper normalization, sacrebleu zh / 13a), corpus over the cell's sentences, hypothesis =
             the concatenated block texts (wave2_s1_score.load_dir);
  D1 (s)     text-level segment LAAL (wave5_viewmarg.d1): one delay per committed unit = its block's write time capped at
             |X|, si_metrics.laal with |Y*| = reference units (laal_defs.si_units); mean over sentences;
  D2 (s)     the same delays through laal_defs.laal_unitwise (speech_latency_lit.laal, the D2 formula); mean over sentences;
  unsup      wave15_commit.py: mean over sentences with a non-final commit of 1 - LCS(non-final committed, reference) /
             |non-final committed| (wave3_d_cpu.toks / lcs) = committed-then-contradicted share; nonfinal = mean share of
             units committed before the final read.
Curve: the LA family (LA1, LA2, LA3) as (D1, BLEU) points sorted by D1; a system's curve value = linear interpolation of
BLEU between the two adjacent LA points bracketing its D1 (outside the LA range: extrapolation of the nearest segment,
flagged). margin = BLEU - curve value. Pre-written rule (prereg section 4): a rule (AoV, EITHER) is "effective" iff on BOTH
zh2en noisy cells (a, b) margin >= +0.5 and on BOTH zh2en clean cells margin >= -0.3; en2zh reported only.

    <uniss-eval python> experiments/mist_w3po_v1/voicecond/s1w_score.py --out eval_outputs/mist_w3po_v1/s1w/score.json
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
from experiments.mist_w3po_v1.voicecond.wave3_d_cpu import lcs, toks  # noqa: E402

S1 = REPO / "eval_outputs/mist_w3po_v1/wave2_S1"
S1W = REPO / "eval_outputs/mist_w3po_v1/s1w"
ROOTS = {"LA1": S1W / "la1", "LA2": S1W / "la2", "LA3": S1W / "la3", "CVLA": REPO / "eval_outputs/mist_w3po_v1/wave15/eval_cvla_base",
         "AoV": S1W / "aov", "EITHER": S1W / "either", "LA2_S1": S1}
CURVE = ("LA1", "LA2", "LA3")
TESTED = ("AoV", "EITHER")
TGT = {"en2zh": "zh", "zh2en": "en"}


def curve_value(pts, x):
    """pts: [(d1, bleu)] of the LA family; linear interpolation between D1-adjacent points (extrapolated outside)"""
    p = sorted(pts)
    if x <= p[0][0]:
        a, b, ext = p[0], p[1], x < p[0][0]
    elif x >= p[-1][0]:
        a, b, ext = p[-2], p[-1], x > p[-1][0]
    else:
        k = max(i for i in range(len(p) - 1) if p[i][0] <= x)
        a, b, ext = p[k], p[k + 1], False
    v = a[1] + (b[1] - a[1]) * (x - a[0]) / (b[0] - a[0]) if b[0] != a[0] else (a[1] + b[1]) / 2
    return v, ext


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    LD = laal_defs()
    res = {"cells": {}, "verdict": {}, "checks": {}}
    for d in ("zh2en", "en2zh"):
        lang, tl = LANG[d], TGT[d]
        bl = Bleu(lang)
        meta = {m["id"]: m for m in jl(S1 / f"data/meta_{d}.jsonl")}
        items = {r["id"]: r for r in jl(S1 / f"data/items_{d}.jsonl")}
        per = {}
        for name, root in ROOTS.items():
            rows = {r["id"]: r for r in jl(root / f"thinker_{d}" / f"blocks_{d}_w1.jsonl")}
            assert set(rows) == set(meta), (name, d, len(rows), len(meta))
            per[name] = {}
            ws_mismatch = 0
            for i, r in rows.items():
                X = items[i]["dur_ms"]
                T = len(LD.si_units(items[i]["translation"], lang))
                hyp = "".join(b["text"] for b in r["blocks"])
                ws_mismatch += int(hyp.split() != r["translation"].split()) if tl == "en" else int(hyp.replace(" ", "") != r["translation"].replace(" ", ""))
                dw = [min(b["write_ms"], X) for b in r["blocks"] for _ in LD.si_units(b["text"], lang)]
                c = toks("".join(b["text"] for b in r["blocks"] if not b.get("final")), tl)
                n_all = len(toks(hyp, tl))
                per[name][i] = {"hyp": hyp,
                                "d1": LD.si_laal(dw, X, T) / 1000 if dw else np.nan,
                                "d2": LD.laal_unitwise(dw, X, T) / 1000 if dw else np.nan,
                                "unsup": np.nan if not c else 1.0 - lcs(c, toks(items[i]["translation"], tl)) / len(c),
                                "nonfinal": len(c) / max(1, n_all)}
            res["checks"][f"{d}/{name}/hyp_vs_translation_mismatch"] = ws_mismatch
        for set_ in ("a", "b"):
            for cond in ("clean", "noisy"):
                ids = sorted(i for i, m in meta.items() if m["set"] == set_ and m["cond"] == cond)
                refs = [items[i]["translation"] for i in ids]
                cell = {"n": len(ids), "systems": {}}
                for name in ROOTS:
                    P = per[name]
                    cell["systems"][name] = {
                        "bleu": round(bl.score(bl.stats([P[i]["hyp"] for i in ids], refs).sum(0)), 3),
                        "d1": round(float(np.nanmean([P[i]["d1"] for i in ids])), 3),
                        "d2": round(float(np.nanmean([P[i]["d2"] for i in ids])), 3),
                        "unsup": round(float(np.nanmean([P[i]["unsup"] for i in ids])), 4),
                        "nonfinal": round(float(np.mean([P[i]["nonfinal"] for i in ids])), 4),
                        "translation_identical_to_LA2_S1": round(float(np.mean([P[i]["hyp"] == per["LA2_S1"][i]["hyp"] for i in ids])), 4)}
                pts = [(cell["systems"][s]["d1"], cell["systems"][s]["bleu"]) for s in CURVE]
                for name in ("CVLA",) + TESTED:
                    s = cell["systems"][name]
                    v, ext = curve_value(pts, s["d1"])
                    s.update(curve_bleu=round(v, 3), margin=round(s["bleu"] - v, 3), extrapolated=ext)
                res["cells"][f"{d}/{set_}/{cond}"] = cell
    for name in TESTED + ("CVLA",):
        m = {k: res["cells"][k]["systems"][name]["margin"] for k in res["cells"]}
        noisy_ok = all(m[f"zh2en/{s}/noisy"] >= 0.5 for s in ("a", "b"))
        clean_ok = all(m[f"zh2en/{s}/clean"] >= -0.3 for s in ("a", "b"))
        res["verdict"][name] = {"margins": m, "zh2en_noisy_ge_0.5": noisy_ok, "zh2en_clean_ge_-0.3": clean_ok,
                                "effective": noisy_ok and clean_ok,
                                "extrapolated_cells": [k for k in res["cells"] if res["cells"][k]["systems"][name]["extrapolated"]]}
    with open(out, "x", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=1)
    # curve table
    for k, cell in res["cells"].items():
        print(f"\n{k} (n={cell['n']})  system  D1  D2  BLEU  curve  margin  unsup  nonfinal")
        for name, s in sorted(cell["systems"].items(), key=lambda x: x[1]["d1"]):
            extra = f"  {s['curve_bleu']:.2f}  {s['margin']:+.2f}{'*' if s['extrapolated'] else ''}" if "margin" in s else "  -  -"
            print(f"  {name:7s} {s['d1']:.3f} {s['d2']:.3f} {s['bleu']:.2f}{extra}  {s['unsup']:.3f}  {s['nonfinal']:.3f}")
    print("\nverdict:", json.dumps({k: {x: v[x] for x in ('effective', 'zh2en_noisy_ge_0.5', 'zh2en_clean_ge_-0.3', 'extrapolated_cells')} for k, v in res["verdict"].items()}, ensure_ascii=False))
    print("checks:", json.dumps(res["checks"]))


if __name__ == "__main__":
    main()

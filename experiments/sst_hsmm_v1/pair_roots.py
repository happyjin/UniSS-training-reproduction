"""overnight 2026-11-27: paired per-cell comparison of two test roots (X vs Y) with p3_score.pair (stream-level bootstrap, 2000
resamples, seed 0), same cells / drops as p3_score. CPU only.
    qwen3omni PYTHONPATH=. python -m experiments.sst_hsmm_v1.pair_roots --x p3p2fxff --y p3hk --dirs zh2en --out <json>"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.mist_w3po_v1.voicecond.combo_score import LANG, jl
from experiments.mist_w3po_v1.voicecond.test_score import EXCL
from experiments.sst_hsmm_v1 import p3_score as PS

KEYS = ("asr", "text_bleu", "d2", "d1", "EndOffset")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--x", required=True); ap.add_argument("--y", required=True)
    ap.add_argument("--dirs", default="en2zh,zh2en"); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if Path(a.out).exists():
        raise SystemExit(f"refuse to overwrite {a.out}")
    res = {}
    for d in a.dirs.split(","):
        lang = LANG[d]
        meta = {m["id"]: m for m in jl(PS.T / "data" / f"meta_{d}.jsonl")}
        ref = {r["id"]: r["translation"] for r in jl(PS.T / "data" / f"items_{d}.jsonl")}
        P = {}
        for s in (a.x, a.y):
            PS.R = PS.REPO / "eval_outputs/sst_hsmm_v1" / s
            P[s] = PS.load("NEW", d)
        common = set(P[a.x]) & set(P[a.y]) - PS.DROP
        for cell in PS.CELLS:
            items = sorted(i for i in common if f"{meta[i]['set']}/{meta[i]['cond']}" == cell
                           and not (meta[i]["set"] == "obsi" and meta[i]["pair"] in EXCL[d]))
            c = PS.pair(meta, ref, items, P[a.x], P[a.y], lang)
            res[f"{d}/{cell}"] = {k: {kk: c[k][kk] for kk in ("diff", "ci95") if kk in c[k]} for k in KEYS if k in c}
            res[f"{d}/{cell}"]["n"] = len(items)
            print(d, cell, len(items), {k: (round(c[k]["diff"], 3), [round(v, 3) for v in c[k]["ci95"]]) for k in KEYS if k in c}, flush=True)
    Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()

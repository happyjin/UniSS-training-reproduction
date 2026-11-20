"""Generic scorer for the overnight robust schemes R5+ (reports/mist_w3po_v1/robust_r5_gradsurgery_prereg.zh-CN.md,
section 5; same rule in later preregs). Point estimates only (no bootstrap).

Items / meta: eval_outputs/mist_w3po_v1/r1/data/{items,meta}_<d>.jsonl (D-U set u + bench_dev set b, clean / noisy).
Systems: --sys NAME=TEMPLATE[,TEMPLATE...]; TEMPLATE contains {d}; rows of all templates are united by id (first wins).
Metrics as r3_score.py: paper-protocol text BLEU (evaluation.text_metrics normalisation, sacrebleu zh / 13a), length
ratio, repetition rate (normalised hyp length > 3 x ref), text LAAL (block write times). Only ids present in every system.
Rule (--method M --erm E; cells = direction x {u, b}):
  1 gap(M) >= gap(base) - 0.3 in all 4 cells and gap(M) >= gap(base) + 0.3 in >= 2 cells  (gap = noisy - clean)
  2 clean BLEU(M) >= base - 0.3 in all 4 cells
  3 mean(clean, noisy) |len ratio - 1| (M) <= base + 0.02 and rep rate (M) <= base + 0.005 (clean and noisy), 4 cells
  4 text LAAL(M) <= base + 0.1 s in all 8 cells
  5 on b only, both directions: clean BLEU(M) >= E - 0.3 and gap(M) >= gap(E) - 0.3
-> effective iff 1-5 all hold.

    qwen3omni PYTHONPATH=. python -m experiments.mist_w3po_v1.voicecond.rx_score --out <json> --method M --erm E \
        --sys base=eval_outputs/mist_w3po_v1/r1/eval_base/thinker_{d}/blocks_{d}_w1.jsonl --sys M=... --sys E=...
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import sacrebleu

from evaluation.text_metrics import normalize_for_bleu

REPO = Path(__file__).resolve().parents[3]
DATA = REPO / "eval_outputs/mist_w3po_v1/r1/data"
LANG = {"en2zh": "cmn", "zh2en": "eng"}


def jl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8")]


def units(text, lang):
    return normalize_for_bleu(text, lang).split()


def laal(row, dur_ms, ref, lang):
    hyp_u, d = [], []
    for b in row["blocks"]:
        u = units(b["text"], lang)
        hyp_u += u
        d += [b["write_ms"]] * len(u)
    if not hyp_u:
        return None
    R = max(len(hyp_u), len(units(ref, lang)))
    s, tau = 0.0, 0
    for i, di in enumerate(d):
        s += di - i * dur_ms / R
        tau = i + 1
        if di >= dur_ms:
            break
    return s / tau / 1000.0


def cell(rows, ids, ref, dur, lang):
    hn = [normalize_for_bleu(rows[i]["translation"], lang) for i in ids]
    rn = [normalize_for_bleu(ref[i], lang) for i in ids]
    b = sacrebleu.corpus_bleu(hn, [rn], tokenize="zh" if lang == "cmn" else "13a")
    rep = sum(len(x.split()) > 3 * max(1, len(y.split())) for x, y in zip(hn, rn)) / len(ids)
    la = [x for x in (laal(rows[i], dur[i], ref[i], lang) for i in ids) if x is not None]
    return {"n": len(ids), "bleu": round(b.score, 2), "len_ratio": round(b.sys_len / max(1, b.ref_len), 4),
            "rep_rate": round(rep, 4), "text_laal": round(sum(la) / max(1, len(la)), 3)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--sys", action="append", required=True)
    ap.add_argument("--method", required=True)
    ap.add_argument("--erm", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    systems = {}
    for x in a.sys:
        k, v = x.split("=", 1)
        systems[k] = v.split(",")
    assert "base" in systems and a.method in systems and a.erm in systems
    res = {"systems": systems, "cells": {}, "gaps": {}, "rule": {}}
    for d in ("en2zh", "zh2en"):
        lang = LANG[d]
        meta = {m["id"]: m for m in jl(DATA / f"meta_{d}.jsonl")}
        items = {r["id"]: r for r in jl(DATA / f"items_{d}.jsonl")}
        ref = {i: r["translation"] for i, r in items.items()}
        dur = {i: int(r["dur_ms"]) for i, r in items.items()}
        R = {}
        for s, tps in systems.items():
            rows = {}
            for t in tps:
                p = REPO / t.format(d=d)
                if not p.exists():
                    print(f"note: missing {p} (skipped)")
                    continue
                for r in jl(p):
                    if r["id"] in meta:
                        rows.setdefault(r["id"], r)
            R[s] = rows
        for st in ("u", "b"):
            for cond in ("clean", "noisy"):
                ids = sorted(i for i, m in meta.items() if m["set"] == st and m["cond"] == cond and all(i in R[s] for s in R))
                n_all = sum(1 for m in meta.values() if m["set"] == st and m["cond"] == cond)
                res["cells"][f"{d}/{st}/{cond}"] = {"n_all": n_all, **{s: cell(R[s], ids, ref, dur, lang) for s in R}}
    M, E = a.method, a.erm
    allok = True
    n_better = 0
    for d in ("en2zh", "zh2en"):
        for st in ("u", "b"):
            c, n = res["cells"][f"{d}/{st}/clean"], res["cells"][f"{d}/{st}/noisy"]
            gap = {s: round(n[s]["bleu"] - c[s]["bleu"], 2) for s in systems}
            res["gaps"][f"{d}/{st}"] = gap
            lm = {s: (abs(c[s]["len_ratio"] - 1) + abs(n[s]["len_ratio"] - 1)) / 2 for s in systems}
            ok = {"1_gap_not_worse": gap[M] >= gap["base"] - 0.3,
                  "2_clean_vs_base": c[M]["bleu"] >= c["base"]["bleu"] - 0.3,
                  "3_len": lm[M] <= lm["base"] + 0.02,
                  "3_rep": all(x[M]["rep_rate"] <= x["base"]["rep_rate"] + 0.005 for x in (c, n)),
                  "4_laal": all(x[M]["text_laal"] <= x["base"]["text_laal"] + 0.1 for x in (c, n))}
            if st == "b":
                ok["5_clean_vs_erm"] = c[M]["bleu"] >= c[E]["bleu"] - 0.3
                ok["5_gap_vs_erm"] = gap[M] >= gap[E] - 0.3
            n_better += gap[M] >= gap["base"] + 0.3
            res["rule"][f"{d}/{st}"] = ok
            allok &= all(ok.values())
    res["rule"]["1_cells_gap_better_by_0.3"] = int(n_better)
    res["effective"] = bool(allok and n_better >= 2)
    with open(out, "x", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=1)
    for k, v in res["cells"].items():
        print(k, f"(n_all {v['n_all']})", " | ".join(f"{s} {v[s]['bleu']:.2f} r{v[s]['len_ratio']:.3f} rep{v[s]['rep_rate']:.3f} "
                                                     f"L{v[s]['text_laal']:.2f} n{v[s]['n']}" for s in systems))
    print(json.dumps(res["gaps"]))
    print(json.dumps(res["rule"]))
    print("effective:", res["effective"])


if __name__ == "__main__":
    main()

"""Robust K1 scorer (reports/mist_w3po_v1/robust_k1_unmerged_prereg.zh-CN.md, sections 3-5). Point estimates only.

Items / meta / metric: rx_score.py (r1/data; cell(): paper-protocol text BLEU, length ratio, repetition rate, text D1
LAAL), only ids present in every system. --sys NAME=TEMPLATE ({d} = direction).
  path check (section 3): --check-lora L --check-merged M: |BLEU(L) - BLEU(M)| <= 0.5 in all 8 cells; also the share of
      identical final translations L vs M, and --method vs base.
  rule (section 4): --method K vs --ref R (cells = direction x {u, b}):
      1 mean over the 4 cells of noisy BLEU (K - R) > 0 and K > R in >= 3 cells
      2 clean BLEU(K) >= R - 0.3 in every cell
      3 mean(|len ratio - 1|, clean and noisy)(K) <= R + 0.005 in every cell
  Refuses to overwrite.

    qwen3omni PYTHONPATH=. python -m experiments.mist_w3po_v1.voicecond.k1_score --out <json> --method k1 --ref r1 \
        [--check-lora unist_lora --check-merged unist] --sys base=... --sys k1=... --sys r1=...
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.mist_w3po_v1.voicecond.rx_score import DATA, LANG, REPO, cell, jl


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--sys", action="append", required=True)
    ap.add_argument("--method", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--check-lora", default="")
    ap.add_argument("--check-merged", default="")
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    systems = dict(x.split("=", 1) for x in a.sys)
    need = {"base", a.method, a.ref} | ({a.check_lora, a.check_merged} - {""})
    assert need <= set(systems), (need, systems.keys())
    res = {"systems": systems, "cells": {}, "gaps": {}, "rule": {}, "check": {}}
    rows_all = {}
    for d in ("en2zh", "zh2en"):
        lang = LANG[d]
        meta = {m["id"]: m for m in jl(DATA / f"meta_{d}.jsonl")}
        items = {r["id"]: r for r in jl(DATA / f"items_{d}.jsonl")}
        ref = {i: r["translation"] for i, r in items.items()}
        dur = {i: int(r["dur_ms"]) for i, r in items.items()}
        R = {}
        for s, t in systems.items():
            p = REPO / t.format(d=d)
            R[s] = {r["id"]: r for r in jl(p) if r["id"] in meta}
            assert R[s], (s, p)
        rows_all[d] = R
        for st in ("u", "b"):
            for cond in ("clean", "noisy"):
                n_all = sum(1 for m in meta.values() if m["set"] == st and m["cond"] == cond)
                ids = sorted(i for i, m in meta.items() if m["set"] == st and m["cond"] == cond and all(i in R[s] for s in R))
                c = {"n_all": n_all, **{s: cell(R[s], ids, ref, dur, lang) for s in R}}
                if a.check_lora:
                    c["same_lora_vs_merged"] = round(sum(R[a.check_lora][i]["translation"] == R[a.check_merged][i]["translation"]
                                                         for i in ids) / len(ids), 4)
                c["same_method_vs_base"] = round(sum(R[a.method][i]["translation"] == R["base"][i]["translation"]
                                                     for i in ids) / len(ids), 4)
                res["cells"][f"{d}/{st}/{cond}"] = c
    K, Rf = a.method, a.ref
    if a.check_lora:
        diffs = {k: round(v[a.check_lora]["bleu"] - v[a.check_merged]["bleu"], 2) for k, v in res["cells"].items()}
        res["check"] = {"bleu_lora_minus_merged": diffs, "pass": all(abs(x) <= 0.5 for x in diffs.values())}
    noisy_d, ok1 = [], {}
    for d in ("en2zh", "zh2en"):
        for st in ("u", "b"):
            c, n = res["cells"][f"{d}/{st}/clean"], res["cells"][f"{d}/{st}/noisy"]
            res["gaps"][f"{d}/{st}"] = {s: round(n[s]["bleu"] - c[s]["bleu"], 2) for s in systems}
            lm = {s: (abs(c[s]["len_ratio"] - 1) + abs(n[s]["len_ratio"] - 1)) / 2 for s in (K, Rf)}
            noisy_d.append(n[K]["bleu"] - n[Rf]["bleu"])
            res["rule"][f"{d}/{st}"] = {"noisy_diff": round(noisy_d[-1], 2),
                                        "2_clean": c[K]["bleu"] >= c[Rf]["bleu"] - 0.3,
                                        "3_len": lm[K] <= lm[Rf] + 0.005,
                                        "len_dev": {K: round(lm[K], 4), Rf: round(lm[Rf], 4)}}
    r1_ok = sum(noisy_d) / 4 > 0 and sum(x > 0 for x in noisy_d) >= 3
    res["rule"]["1_noisy"] = {"mean": round(sum(noisy_d) / 4, 3), "n_better": sum(x > 0 for x in noisy_d), "ok": r1_ok}
    res["pass"] = bool(r1_ok and all(res["rule"][f"{d}/{st}"]["2_clean"] and res["rule"][f"{d}/{st}"]["3_len"]
                                     for d in ("en2zh", "zh2en") for st in ("u", "b")))
    with open(out, "x", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=1)
    names = list(systems)
    for k, v in res["cells"].items():
        print(k, f"(n_all {v['n_all']})", " | ".join(f"{s} {v[s]['bleu']:.2f} r{v[s]['len_ratio']:.3f} rep{v[s]['rep_rate']:.3f} "
                                                     f"L{v[s]['text_laal']:.2f} n{v[s]['n']}" for s in names),
              {x: v[x] for x in v if x.startswith("same_")})
    print("gaps", json.dumps(res["gaps"]))
    print("check", json.dumps(res["check"]))
    print("rule", json.dumps(res["rule"]))
    print("pass:", res["pass"])


if __name__ == "__main__":
    main()

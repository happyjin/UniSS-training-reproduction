"""Robust R1 scoring (reports/mist_w3po_v1/robust_r1_taskdiff_prereg.zh-CN.md): Thinker-only point estimates, no bootstrap.

Data eval_outputs/mist_w3po_v1/r1/data/{items,meta}_<dir>.jsonl (UniST in-domain dev set "u" + S1 bench_dev set "b").
Systems: base, unist (wave 14, alpha 1), unist_wise05, r1 (theta_base + tau_noisy - tau_clean); blocks from
r1/eval_<sys>/thinker_<dir>/blocks_<dir>_w1.jsonl. Per direction x cell (u, b, b_noxt) x {clean, noisy}:
  text BLEU (paper protocol, evaluation.text_metrics.corpus_bleu), length ratio sys_len / ref_len of that BLEU,
  repetition rate (share of sentences whose normalised hypothesis is > 3x the normalised reference in units: Chinese
  characters / English words), text LAAL (si_text_eval.laal: every unit delayed to its block's write time), and the
  noise gap = noisy - clean BLEU.
Rule (point estimates, r1 vs unist_wise05, per direction x {u, b}): gap not worse (>=), clean BLEU not lower (>=),
|ratio - 1| closer to 0 (<=, on clean and noisy pooled), repetition rate not higher (<=).

    qwen3omni PYTHONPATH=. python -m experiments.mist_w3po_v1.voicecond.r1_score --out eval_outputs/mist_w3po_v1/r1/score.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from evaluation.text_metrics import corpus_bleu, normalize_for_bleu
from experiments.uniss_qwen3_omni_v1.evaluation.si_text_eval import CJK, EN, laal

REPO = Path(__file__).resolve().parents[3]
R = REPO / "eval_outputs/mist_w3po_v1/r1"
SYSTEMS = ("base", "unist", "unist_wise05", "r1")
LANG = {"en2zh": "cmn", "zh2en": "eng"}


def jl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8")]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--systems", default=",".join(SYSTEMS))
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    systems = a.systems.split(",")
    res = {"cells": {}, "rule": {}}
    for d in ("en2zh", "zh2en"):
        lang = LANG[d]
        unit = CJK if lang == "cmn" else EN
        meta = {m["id"]: m for m in jl(R / "data" / f"meta_{d}.jsonl")}
        items = {r["id"]: r for r in jl(R / "data" / f"items_{d}.jsonl")}
        H = {s: {r["id"]: r for r in jl(R / f"eval_{s}" / f"thinker_{d}" / f"blocks_{d}_w1.jsonl")} for s in systems}
        for s in systems:
            assert set(H[s]) == set(items), (s, d, len(H[s]), len(items))
        for cell in ("u", "b", "b_noxt"):
            for cond in ("clean", "noisy"):
                ids = sorted(i for i, m in meta.items() if m["set"] == cell[0] and m["cond"] == cond
                             and (cell != "b_noxt" or not m["ovl"]))
                if not ids:
                    continue
                refs = [items[i]["translation"] for i in ids]
                row = {"n": len(ids)}
                for s in systems:
                    hyp = [H[s][i]["translation"] for i in ids]
                    b = corpus_bleu(hyp, refs, language=lang)
                    rep = np.mean([len(normalize_for_bleu(h, lang).split()) > 3 * max(1, len(normalize_for_bleu(r, lang).split()))
                                   for h, r in zip(hyp, refs)])
                    lags = []
                    for i in ids:
                        dl = [bl["write_ms"] for bl in H[s][i]["blocks"] for _ in unit.findall(bl["text"])]
                        if dl:
                            lags.append(laal(dl, items[i]["dur_ms"], len(unit.findall(items[i]["translation"]))))
                    row[s] = {"bleu": round(b["score"], 2), "ratio": round(b["sys_len"] / max(1, b["ref_len"]), 3),
                              "rep": round(float(rep), 4), "laal_text_s": round(float(np.mean(lags)) / 1000, 3)}
                res["cells"][f"{d}/{cell}/{cond}"] = row
        for cell in ("u", "b"):
            c, n = res["cells"][f"{d}/{cell}/clean"], res["cells"][f"{d}/{cell}/noisy"]
            gap = {s: round(n[s]["bleu"] - c[s]["bleu"], 2) for s in systems}
            res["cells"][f"{d}/{cell}/gap"] = gap
            if "r1" in systems and "unist_wise05" in systems:
                r, w = "r1", "unist_wise05"
                dev = lambda s: abs((c[s]["ratio"] + n[s]["ratio"]) / 2 - 1)
                checks = {"gap_not_worse": gap[r] >= gap[w], "clean_not_lower": c[r]["bleu"] >= c[w]["bleu"],
                          "ratio_closer_to_1": dev(r) <= dev(w),
                          "rep_not_higher": (c[r]["rep"] + n[r]["rep"]) <= (c[w]["rep"] + n[w]["rep"])}
                res["rule"][f"{d}/{cell}"] = {**checks, "all": all(checks.values())}
    with open(out, "x", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=1)
    for k, v in res["cells"].items():
        if k.endswith("/gap"):
            print(k, v)
        else:
            print(k, f"n={v['n']}", " | ".join(f"{s} {v[s]['bleu']:.2f} r{v[s]['ratio']:.2f} rep{v[s]['rep']:.3f} L{v[s]['laal_text_s']:.2f}"
                                             for s in systems))
    print(json.dumps(res["rule"], indent=1))


if __name__ == "__main__":
    main()

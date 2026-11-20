"""Robust R4 scoring, point estimates only (reports/mist_w3po_v1/robust_r4_route_prereg.zh-CN.md): per direction x
{u, b, b_noxt} x {clean, noisy}: paper-normalised text BLEU, length ratio (units: CJK chars / English words), repetition
rate (hypothesis > 3x reference units), text D1 LAAL (s; delay of a unit = write_ms of its block, si_metrics.laal), and for
R4 arms the share of reads drafted by the adapter. Systems: base, enc (enc_wise05 LA2), uni (unist_wise05 LA2), R4e, R4w.
D-U references from r4/ref_*; (b) references reused (wave2_S1, wave17/eval_enc_wise05, wave14/eval_unist_wise05).
Pre-registered verdict per arm (vs its own adapter A and vs base B):
  (1) clean text BLEU vs A >= -0.3 in u and b, and the noisy-clean gap not worse than A's by more than 0.5;
  (2) some noisy cell (u or b) with text BLEU > B; (3) text D1 vs B <= +0.1 s in every cell.

    qwen3omni PYTHONPATH=. python -m experiments.mist_w3po_v1.voicecond.r4_score --out eval_outputs/mist_w3po_v1/r4/score.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from evaluation.text_metrics import corpus_bleu
from experiments.uniss_qwen3_omni_v1.evaluation.si_metrics import laal

REPO = Path(__file__).resolve().parents[3]
E = REPO / "eval_outputs/mist_w3po_v1"
M = E / "r4"
SRC = {"u": {"base": M / "ref_base", "enc": M / "ref_enc", "uni": M / "ref_uni", "R4e": M / "R4e", "R4w": M / "R4w"},
       "b": {"base": E / "wave2_S1", "enc": E / "wave17/eval_enc_wise05", "uni": E / "wave14/eval_unist_wise05",
             "R4e": M / "R4e", "R4w": M / "R4w"}}
ARMS = {"R4e": "enc", "R4w": "uni"}


def jl(p):
    return [json.loads(x) for x in open(p, encoding="utf-8")]


def toks(t, zh):
    return [c for c in t if not c.isspace()] if zh else t.split()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    res = {"cells": {}, "verdict": {}}
    for d in ("en2zh", "zh2en"):
        zh = d.endswith("zh")
        lang = "cmn" if zh else "eng"
        meta = {m["id"]: m for m in jl(M / "data" / f"meta_{d}.jsonl")}
        item = {r["id"]: r for r in jl(M / "data" / f"items_{d}.jsonl")}
        rows = {}
        for s_, srcs in SRC.items():
            for name, root in srcs.items():
                f = root / f"thinker_{d}" / f"blocks_{d}_w1.jsonl"
                rows.setdefault(name, {}).update({r["id"]: r for r in jl(f) if r["id"] in meta and meta[r["id"]]["set"] == s_})
        for cell_set in ("u", "b", "b_noxt"):
            for cond in ("clean", "noisy"):
                ids = sorted(i for i, m in meta.items() if m["set"] == cell_set[0] and m["cond"] == cond
                             and (cell_set != "b_noxt" or not m["ovl"]))
                cell = {}
                for name, R in rows.items():
                    if not all(i in R for i in ids):
                        continue
                    hyp = [R[i]["translation"] for i in ids]
                    ref = [item[i]["translation"] for i in ids]
                    b = corpus_bleu(hyp, ref, language=lang)
                    hl = [len(toks(h, zh)) for h in hyp]
                    rl = [len(toks(r, zh)) for r in ref]
                    d1 = []
                    for i, rl_ in zip(ids, rl):
                        delays = []
                        for blk in R[i]["blocks"]:
                            delays += [blk["write_ms"]] * len(toks(blk["text"], zh))
                        v = laal(delays, item[i]["dur_ms"], rl_)
                        if v == v:
                            d1.append(v / 1000)
                    c = {"bleu": round(b["score"], 2), "len_ratio": round(sum(hl) / max(sum(rl), 1), 3),
                         "repeat_rate": round(float(np.mean([h > 3 * max(r, 1) for h, r in zip(hl, rl)])), 4),
                         "d1": round(float(np.mean(d1)), 3), "n": len(ids)}
                    if name in ARMS:
                        rd = [x.get("model") for i in ids for x in R[i]["drafts"]]
                        c["adapter_read_share"] = round(sum(m == "adapter" for m in rd) / max(len(rd), 1), 4)
                    cell[name] = c
                res["cells"][f"{d}/{cell_set}/{cond}"] = cell
    C = res["cells"]
    for arm, ad in ARMS.items():
        v1, v2, v3 = [], [], []
        for d in ("en2zh", "zh2en"):
            for s_ in ("u", "b"):
                cl, no = C[f"{d}/{s_}/clean"], C[f"{d}/{s_}/noisy"]
                if arm not in cl or arm not in no:
                    continue
                dc = cl[arm]["bleu"] - cl[ad]["bleu"]
                dg = (no[arm]["bleu"] - cl[arm]["bleu"]) - (no[ad]["bleu"] - cl[ad]["bleu"])
                if dc < -0.3 or dg < -0.5:
                    v1.append([f"{d}/{s_}", round(dc, 2), round(dg, 2)])
                if no[arm]["bleu"] > no["base"]["bleu"]:
                    v2.append([f"{d}/{s_}/noisy", round(no[arm]["bleu"] - no["base"]["bleu"], 2)])
                for cc, x in (("clean", cl), ("noisy", no)):
                    if x[arm]["d1"] - x["base"]["d1"] > 0.1:
                        v3.append([f"{d}/{s_}/{cc}", round(x[arm]["d1"] - x["base"]["d1"], 3)])
        res["verdict"][arm] = {"rule1_failures": v1, "rule2_noisy_gains_vs_base": v2, "rule3_d1_failures": v3,
                               "EFFECTIVE": not v1 and bool(v2) and not v3}
    with open(out, "x", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=1)
    for k, cell in C.items():
        print(k, " | ".join(f"{n} {c['bleu']:.2f} r{c['len_ratio']:.2f} rep{c['repeat_rate']:.3f} D1 {c['d1']:.2f}"
                             + (f" ad{c['adapter_read_share']:.2f}" if "adapter_read_share" in c else "") for n, c in cell.items()))
    print(json.dumps(res["verdict"], ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()

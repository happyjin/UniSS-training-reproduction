"""R3 scoring (reports/mist_w3po_v1/robust_r3_noisesub_prereg.zh-CN.md, section 5): point estimates only (no bootstrap).
Systems: r3_wise05, base, enc_wise05. Cells: devU {clean, noisy} (set u), bench_dev (b) {clean, noisy} and (b) without
crosstalk, (a) CoVoST {clean, noisy} (reference only). Thinker outputs:
  devU: eval_outputs/mist_w3po_v1/r3/eval_<sys>/devU/thinker_<d>/
  s1  : r3_wise05 -> r3/eval_r3_wise05/s1/; base -> wave2_S1/thinker_<d>/; enc_wise05 -> wave17/eval_enc_wise05/thinker_<d>/
Metrics: paper-protocol text BLEU (evaluation.text_metrics normalisation, sacrebleu zh / 13a), length ratio sys/ref,
repetition rate (normalised hypothesis length > 3 x reference), text LAAL (s; unit delays = block write times, units =
characters (zh) / words (en) of the normalised text, LAAL with max(|hyp|, |ref|), cut at the source duration).
Rule (written before numbers): r3_wise05 vs enc_wise05 on devU and (b): gap (noisy - clean) not worse, clean BLEU not
lower, |length ratio - 1| smaller, repetition rate not higher -> "effective".

    qwen3omni PYTHONPATH=. python -m experiments.mist_w3po_v1.voicecond.r3_score --out <json>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import sacrebleu

from evaluation.text_metrics import normalize_for_bleu

REPO = Path(__file__).resolve().parents[3]
M = REPO / "eval_outputs/mist_w3po_v1"
LANG = {"en2zh": "cmn", "zh2en": "eng"}
SYS = ("r3_wise05", "base", "enc_wise05")


def jl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8")]


def thinker(sys_, dset, d):
    if dset == "devU":
        return M / f"r3/eval_{sys_}/devU/thinker_{d}/blocks_{d}_w1.jsonl"
    return {"r3_wise05": M / f"r3/eval_r3_wise05/s1/thinker_{d}/blocks_{d}_w1.jsonl",
            "base": M / f"wave2_S1/thinker_{d}/blocks_{d}_w1.jsonl",
            "enc_wise05": M / f"wave17/eval_enc_wise05/thinker_{d}/blocks_{d}_w1.jsonl"}[sys_]


def units(text, lang):
    n = normalize_for_bleu(text, lang)
    return n.split()


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
    h = [rows[i]["translation"] for i in ids]
    r = [ref[i] for i in ids]
    hn = [normalize_for_bleu(x, lang) for x in h]
    rn = [normalize_for_bleu(x, lang) for x in r]
    b = sacrebleu.corpus_bleu(hn, [rn], tokenize="zh" if lang == "cmn" else "13a")
    rep = sum(len(x.split()) > 3 * max(1, len(y.split())) for x, y in zip(hn, rn)) / len(ids)
    la = [laal(rows[i], dur[i], ref[i], lang) for i in ids]
    la = [x for x in la if x is not None]
    return {"n": len(ids), "bleu": round(b.score, 2), "len_ratio": round(b.sys_len / max(1, b.ref_len), 4),
            "rep_rate": round(rep, 4), "text_laal": round(sum(la) / max(1, len(la)), 3)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    res = {"cells": {}, "rule": {}}
    for d in ("en2zh", "zh2en"):
        lang = LANG[d]
        for dset, droot in (("devU", M / "devU/data"), ("s1", M / "wave2_S1/data")):
            meta = {m["id"]: m for m in jl(droot / f"meta_{d}.jsonl")}
            items = {r["id"]: r for r in jl(droot / f"items_{d}.jsonl")}
            ref = {i: r["translation"] for i, r in items.items()}
            dur = {i: int(r["dur_ms"]) for i, r in items.items()}
            R = {s: {r["id"]: r for r in jl(thinker(s, dset, d))} for s in SYS}
            sets = [("u", None)] if dset == "devU" else [("b", None), ("b_noxt", False), ("a", None)]
            for st, ovl in sets:
                for cond in ("clean", "noisy"):
                    ids = sorted(i for i, m in meta.items() if m["set"] == st[0] and m["cond"] == cond
                                 and (st != "b_noxt" or not m["ovl"]) and all(i in R[s] for s in SYS))
                    key = f"{d}/{'devU' if dset == 'devU' else st}/{cond}"
                    res["cells"][key] = {s: cell(R[s], ids, ref, dur, lang) for s in SYS}
    # rule: r3 vs enc_wise05 on devU and (b)
    checks = []
    for d in ("en2zh", "zh2en"):
        for st in ("devU", "b"):
            c, n = res["cells"][f"{d}/{st}/clean"], res["cells"][f"{d}/{st}/noisy"]
            gap = {s: round(n[s]["bleu"] - c[s]["bleu"], 2) for s in SYS}
            ok = {"gap_not_worse": gap["r3_wise05"] >= gap["enc_wise05"],
                  "clean_not_lower": c["r3_wise05"]["bleu"] >= c["enc_wise05"]["bleu"],
                  "len_closer_to_1": all(abs(x["r3_wise05"]["len_ratio"] - 1) <= abs(x["enc_wise05"]["len_ratio"] - 1) for x in (c, n)),
                  "rep_not_higher": all(x["r3_wise05"]["rep_rate"] <= x["enc_wise05"]["rep_rate"] for x in (c, n))}
            res["rule"][f"{d}/{st}"] = {"gap": gap, **ok}
            checks += list(ok.values())
    res["effective"] = all(checks)
    with open(out, "x", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=1)
    for k, v in res["cells"].items():
        print(k, " | ".join(f"{s} {v[s]['bleu']:.2f} r{v[s]['len_ratio']:.3f} rep{v[s]['rep_rate']:.3f} L{v[s]['text_laal']:.2f}" for s in SYS))
    print(json.dumps(res["rule"], indent=1))
    print("effective:", res["effective"])


if __name__ == "__main__":
    main()

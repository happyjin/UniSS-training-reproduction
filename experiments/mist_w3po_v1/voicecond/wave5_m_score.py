"""Robust wave 5, main line M scoring (reports/mist_w3po_v1/robust_wave5_prereg.zh-CN.md, incl. the unified rule and
change 2): the merged ERM / Fishr Thinkers against the S1 base (same LA2 chain, same T1c Talker, seed 20261021, 4 shards)
and Fishr against ERM.

Per direction x set (a: CoVoST 2 pairs; b: OB-SI dev single-speaker cuts; b_no_crosstalk) x condition (clean / noisy):
text BLEU and ASR-BLEU (S1 convention: normalize_for_bleu, sacrebleu zh / 13a; wave2_s1_score.Bleu), D1 / D2 LAAL means
(auxiliary, no gate); system differences with a paired bootstrap 95 % CI (2000, seed 0) over units = sentences (a) or
single-speaker streams (b).
Primary rule per arm vs base: (1) no BLEU degradation in every direction x {a, b} x condition, for text BLEU and ASR-BLEU:
point >= -0.3 and CI upper >= 0; (2) at least one noisy cell (direction x {a, b, b_no_crosstalk}) with a text BLEU gain
whose CI lower > 0. Secondary (reported): the original G1-G3 and Fishr - ERM.
    <qwen3omni python> experiments/mist_w3po_v1/voicecond/wave5_m_score.py --out eval_outputs/mist_w3po_v1/wave5/M/summary.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from experiments.mist_w3po_v1.voicecond import wave2_s1_score as S  # noqa: E402

ROOTS = {"base": REPO / "eval_outputs/mist_w3po_v1/wave2_S1",
         "erm": REPO / "eval_outputs/mist_w3po_v1/wave5/M/eval_erm",
         "fishr": REPO / "eval_outputs/mist_w3po_v1/wave5/M/eval_fishr"}
SEED = "20261021"
NB = 2000


def cell(meta, pers, lang, set_, cond, keep):
    bl = S.Bleu(lang)
    ids = sorted(i for i, m in meta.items() if m["set"] == set_ and m["cond"] == cond and keep(m))
    units = sorted({meta[i]["stream"] for i in ids})
    up = {u: k for k, u in enumerate(units)}
    pu = np.array([up[meta[i]["stream"]] for i in ids])
    st, mean = {}, {}
    for sysn, per in pers.items():
        for kind in ("text", "asr"):
            hyp = [per[i]["hyp"] if kind == "text" else per[i][f"asr{SEED}"] for i in ids]
            x = bl.stats(hyp, [per[i]["ref"] for i in ids])
            U = np.zeros((len(units), x.shape[1]), dtype=np.int64)
            np.add.at(U, pu, x)
            st[(sysn, kind)] = U
        for k in ("d1", "d2"):
            s_, n_ = np.zeros(len(units)), np.zeros(len(units))
            for j, i in enumerate(ids):
                v = per[i][f"{k}_{SEED}"]
                if v is not None:
                    s_[pu[j]] += v
                    n_[pu[j]] += 1
            mean[(sysn, k)] = (s_, n_)
    rng = np.random.default_rng(0)
    W = [np.ones(len(units))] + [np.bincount(rng.integers(0, len(units), len(units)), minlength=len(units)).astype(float)
                                 for _ in range(NB)]

    def val(sysn, k, w):
        if k in ("text", "asr"):
            return bl.score((st[(sysn, k)] * w[:, None]).sum(0))
        s_, n_ = mean[(sysn, k)]
        return float((s_ * w).sum() / max((n_ * w).sum(), 1e-9))
    out = {"n_sentences": len(ids), "n_units": len(units)}
    for k in ("text", "asr", "d1", "d2"):
        vals = {sysn: [val(sysn, k, w) for w in W] for sysn in pers}
        r = {sysn: round(v[0], 3) for sysn, v in vals.items()}
        for x, y in (("erm", "base"), ("fishr", "base"), ("fishr", "erm")):
            if x in vals and y in vals:
                d = np.array(vals[x]) - np.array(vals[y])
                r[f"{x}-{y}"] = [round(float(d[0]), 3), round(float(np.percentile(d[1:], 2.5)), 3),
                                 round(float(np.percentile(d[1:], 97.5)), 3)]
        out[k] = r
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    LD = S.laal_defs()
    res = {"systems": {k: str(v.relative_to(REPO)) for k, v in ROOTS.items()}, "seed": SEED}
    gates = {}
    for d in ("en2zh", "zh2en"):
        pers, meta = {}, None
        for sysn, root in ROOTS.items():
            if not (root / f"talker_{d}_s{SEED}" / "asr_speech.jsonl").exists():
                continue
            meta, pers[sysn] = S.load_dir(root, d, [SEED], LD)
        lang = S.LANG[d]
        ovl = {m["pair"] for m in meta.values() if m["set"] == "b" and m["ovl"]}
        res[d] = {}
        for sub, set_, keep in (("a", "a", lambda m: True), ("b", "b", lambda m: True),
                                ("b_no_crosstalk", "b", lambda m, o=ovl: m["pair"] not in o)):
            res[d][sub] = {c: cell(meta, pers, lang, set_, c, keep) for c in ("clean", "noisy")}
    for arm in ("erm", "fishr"):
        if not all(arm in res[d]["a"]["clean"]["text"] for d in ("en2zh", "zh2en")):
            continue
        k = f"{arm}-base"
        nodeg = {f"{d}/{sub}/{c}/{m}": res[d][sub][c][m][k] for d in ("en2zh", "zh2en") for sub in ("a", "b")
                 for c in ("clean", "noisy") for m in ("text", "asr")}
        nd_ok = all(v[0] >= -0.3 and v[2] >= 0 for v in nodeg.values())
        imp = {f"{d}/{sub}/noisy/text": res[d][sub]["noisy"]["text"][k] for d in ("en2zh", "zh2en")
               for sub in ("a", "b", "b_no_crosstalk")}
        imp_ok = any(v[1] > 0 for v in imp.values())
        g = {"no_degradation": nd_ok, "no_degradation_failures": {x: v for x, v in nodeg.items() if not (v[0] >= -0.3 and v[2] >= 0)},
             "improvement": imp_ok, "improvement_cells": {x: v for x, v in imp.items() if v[1] > 0},
             "PASS": nd_ok and imp_ok}
        if arm == "fishr" and "fishr-erm" in res["zh2en"]["a"]["noisy"]["text"]:
            g1 = res["zh2en"]["a"]["noisy"]["text"]["fishr-erm"]
            g2e = res["en2zh"]["b_no_crosstalk"]["noisy"]["text"]
            g["secondary_G1_zh2en_a_noisy_fishr_minus_erm>=+0.5_ci_lo>0"] = [g1, g1[0] >= 0.5 and g1[1] > 0]
            g["secondary_G2_en2zh_bnoct_noisy"] = [g2e.get("fishr-erm"), g2e.get("fishr-base"),
                                                   g2e["fishr-erm"][0] >= 0 and g2e["fishr-base"][0] >= 1.0]
            g3 = [res[d]["a"]["clean"]["text"]["fishr-base"] for d in ("en2zh", "zh2en")]
            g["secondary_G3_clean_a_>=-0.3"] = [g3, all(v[0] >= -0.3 for v in g3)]
        gates[arm] = g
    res["gates"] = gates
    with open(out, "x", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=1)
    print(json.dumps(gates, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()

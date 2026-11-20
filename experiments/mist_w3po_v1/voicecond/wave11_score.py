"""Robust wave 11 scoring (reports/mist_w3po_v1/robust_wave11_laplace_prereg.zh-CN.md), a copy of wave10_score.py with the
arm laplace (clean-anchored Laplace posterior merge: W0 + F_n / (F_c + F_n) * Delta_ERM) against the S1 base (3 base runs,
change 3 of wave 5) and against wave 5's erm_wise05 (same chain, same T1c Talker, seed 20261021, 4 shards).

Rule 1 (laplace vs base; wave 5's unified rule, unchanged): every direction x {a, b} x {clean, noisy}, text BLEU
(vs the mean of the 3 base runs) and ASR-BLEU (vs S1): point >= -0.3 and CI upper >= 0; and at least one noisy cell
(direction x {a, b, b_no_crosstalk}) with a text BLEU gain over the base mean whose CI lower > 0 and point > noise floor.
Rule 2 (increment vs erm_wise05): at least one noisy cell (direction x {a, b, b_no_crosstalk}) with text BLEU
laplace - erm_wise05 CI lower > 0 and point > the base noise floor of that cell; and no direction x {a, b} x
{clean, noisy} x {text, ASR-BLEU} cell with point < -0.3 and CI upper < 0.
Verdict: PASS = rule 1 and rule 2; "Laplace merge: no increment over WiSE" = rule 1 only (analysis only);
FAIL = rule 1 fails.
    <qwen3omni python> experiments/mist_w3po_v1/voicecond/wave11_score.py --out eval_outputs/mist_w3po_v1/wave11/summary.json

--- wave5_m_score.py docstring ---
Robust wave 5, main line M scoring (reports/mist_w3po_v1/robust_wave5_prereg.zh-CN.md, incl. the unified rule and
change 2): the merged ERM / Fishr Thinkers against the S1 base (same LA2 chain, same T1c Talker, seed 20261021, 4 shards)
and Fishr against ERM.

Per direction x set (a: CoVoST 2 pairs; b: OB-SI dev single-speaker cuts; b_no_crosstalk) x condition (clean / noisy):
text BLEU and ASR-BLEU (S1 convention: normalize_for_bleu, sacrebleu zh / 13a; wave2_s1_score.Bleu), D1 / D2 LAAL means
(auxiliary, no gate); system differences with a paired bootstrap 95 % CI (2000, seed 0) over units = sentences (a) or
single-speaker streams (b).
Change 3: two Thinker-only reruns of the base (wave5/base_rerun{1,2}) join S1 as 3 base runs; text BLEU is judged
against their mean, and a gain must also exceed the largest pairwise difference of the 3 base runs in that cell.
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
         "erm_wise05": REPO / "eval_outputs/mist_w3po_v1/wave5/M/eval_erm_wise05",
         "laplace": REPO / "eval_outputs/mist_w3po_v1/wave11/eval_laplace"}
RERUNS = {"base_r1": REPO / "eval_outputs/mist_w3po_v1/wave5/base_rerun1",
          "base_r2": REPO / "eval_outputs/mist_w3po_v1/wave5/base_rerun2"}   # change 3: Thinker-only reruns of the base
BASES = ("base", "base_r1", "base_r2")
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
            if kind == "asr" and f"asr{SEED}" not in per[ids[0]]:
                continue
            hyp = [per[i]["hyp"] if kind == "text" else per[i][f"asr{SEED}"] for i in ids]
            x = bl.stats(hyp, [per[i]["ref"] for i in ids])
            U = np.zeros((len(units), x.shape[1]), dtype=np.int64)
            np.add.at(U, pu, x)
            st[(sysn, kind)] = U
        for k in ("d1", "d2"):
            if f"{k}_{SEED}" not in per[ids[0]]:
                continue
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
        have = [sysn for sysn in pers if (sysn, k) in st or (sysn, k) in mean]
        vals = {sysn: [val(sysn, k, w) for w in W] for sysn in have}
        r = {sysn: round(v[0], 3) for sysn, v in vals.items()}
        if k == "text" and all(b in vals for b in BASES):
            vals["base_mean"] = list(np.mean([vals[b] for b in BASES], axis=0))
            r["base_mean"] = round(vals["base_mean"][0], 3)
            r["base_floor"] = round(max(abs(vals[x][0] - vals[y][0]) for i, x in enumerate(BASES) for y in BASES[i + 1:]), 3)
        for x, y in (("laplace", "base"), ("laplace", "base_mean"), ("laplace", "erm_wise05"),
                     ("erm_wise05", "base"), ("erm_wise05", "base_mean")):
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
        for rn, rr in RERUNS.items():
            bf = rr / f"thinker_{d}" / f"blocks_{d}_w1.jsonl"
            if bf.exists():
                blk = {r["id"]: "".join(b["text"] for b in r["blocks"]) for r in S.jl(bf)}
                assert set(blk) >= set(meta), f"{rn} {d}: missing items"
                pers[rn] = {i: {"hyp": blk[i], "ref": pers["base"][i]["ref"]} for i in meta}
        lang = S.LANG[d]
        ovl = {m["pair"] for m in meta.values() if m["set"] == "b" and m["ovl"]}
        res[d] = {}
        for sub, set_, keep in (("a", "a", lambda m: True), ("b", "b", lambda m: True),
                                ("b_no_crosstalk", "b", lambda m, o=ovl: m["pair"] not in o)):
            res[d][sub] = {c: cell(meta, pers, lang, set_, c, keep) for c in ("clean", "noisy")}
    for arm in ("laplace", "erm_wise05"):
        if not all(arm in res[d]["a"]["clean"]["text"] for d in ("en2zh", "zh2en")):
            continue
        k, km = f"{arm}-base", f"{arm}-base_mean"
        # change 3: text BLEU against the mean of the 3 base runs; ASR-BLEU only S1 has a Talker run
        nodeg = {f"{d}/{sub}/{c}/{m}": res[d][sub][c][m][km if m == "text" else k] for d in ("en2zh", "zh2en")
                 for sub in ("a", "b") for c in ("clean", "noisy") for m in ("text", "asr")}
        nd_ok = all(v[0] >= -0.3 and v[2] >= 0 for v in nodeg.values())
        imp = {f"{d}/{sub}/noisy/text": res[d][sub]["noisy"]["text"][km] + [res[d][sub]["noisy"]["text"]["base_floor"]]
               for d in ("en2zh", "zh2en") for sub in ("a", "b", "b_no_crosstalk")}
        imp_ok = any(v[1] > 0 and v[0] > v[3] for v in imp.values())
        g = {"rule1_no_degradation": nd_ok,
             "rule1_no_degradation_failures": {x: v for x, v in nodeg.items() if not (v[0] >= -0.3 and v[2] >= 0)},
             "rule1_improvement": imp_ok, "rule1_improvement_cells_[d,lo,hi,floor]": {x: v for x, v in imp.items() if v[1] > 0},
             "rule1_PASS": nd_ok and imp_ok}
        if arm == "laplace":
            kk = "laplace-erm_wise05"
            inc = {f"{d}/{sub}/noisy/text": res[d][sub]["noisy"]["text"][kk] + [res[d][sub]["noisy"]["text"]["base_floor"]]
                   for d in ("en2zh", "zh2en") for sub in ("a", "b", "b_no_crosstalk")}
            inc_ok = any(v[1] > 0 and v[0] > v[3] for v in inc.values())
            harm = {f"{d}/{sub}/{c}/{m}": res[d][sub][c][m][kk] for d in ("en2zh", "zh2en")
                    for sub in ("a", "b") for c in ("clean", "noisy") for m in ("text", "asr")}
            harm_bad = {x: v for x, v in harm.items() if v[0] < -0.3 and v[2] < 0}
            g.update({"rule2_increment": inc_ok, "rule2_increment_cells_[d,lo,hi,floor]": {x: v for x, v in inc.items() if v[1] > 0},
                      "rule2_all_noisy_text_vs_erm_wise05_[d,lo,hi,floor]": inc,
                      "rule2_no_harm": not harm_bad, "rule2_harm_cells": harm_bad,
                      "rule2_PASS": inc_ok and not harm_bad})
            g["VERDICT"] = ("PASS" if g["rule1_PASS"] and g["rule2_PASS"] else
                            "NO_INCREMENT (rule 1 only: Laplace merge has no increment over WiSE; analysis only)" if g["rule1_PASS"] else "FAIL (rule 1)")
        gates[arm] = g
    res["gates"] = gates
    with open(out, "x", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=1)
    print(json.dumps(gates, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()

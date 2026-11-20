"""Robust wave 17b scoring (reports/mist_w3po_v1/robust_wave17b_enclora_covost_prereg.zh-CN.md; rules: robust_mainline_plan
4.3 with the rule-2 reference changed to CoVoST erm_wise05). Copy of wave17_score.py: arms enc_wise05 / enc are wave 17b's
(encoder-only LoRA on wave 5's CoVoST data; directory names kept, called enc_cv_wise05 / enc_cv in the report);
unist_wise05 dropped; rule 2 = enc_wise05 vs erm_wise05 (increment: any direction x {a, b, b_no_crosstalk} x {clean, noisy}
text cell with CI lower > 0 and point > base floor; no harm: no direction x {a, b} x condition x {text, ASR} cell with
point < -0.3 and CI upper < 0). Self-check: erm_wise05 vs base equals wave5/M/summary_erm_wise05.json exactly.
    <qwen3omni python> experiments/mist_w3po_v1/voicecond/wave17b_score.py --out eval_outputs/mist_w3po_v1/wave17b/summary.json

--- wave17_score.py docstring ---
Robust wave 17 scoring (reports/mist_w3po_v1/robust_wave17_enclora_prereg.zh-CN.md; rules: robust_mainline_plan_20261120
4.3), a copy of wave14_score.py with the arms enc_wise05 (encoder-only LoRA on wave 14's UniST data, WiSE 0.5; the final
system) and enc (alpha 1, auxiliary, no gate) against the S1 base (3 base runs), wave 14's unist_wise05 and wave 5's
erm_wise05 (same chain, same T1c Talker, seed 20261021, 4 shards).

Rule 1 (enc_wise05 vs base; the unified rule): every direction x {a, b} x {clean, noisy}, text BLEU (vs the mean of the
3 base runs) and ASR-BLEU (vs S1): point >= -0.3 and CI upper >= 0; and at least one noisy cell (direction x
{a, b, b_no_crosstalk}) with a text BLEU gain over the base mean whose CI lower > 0 and point > noise floor.
Rule 2 (vs unist_wise05): at least one cell (direction x {a, b, b_no_crosstalk} x {clean, noisy}) with text BLEU
enc_wise05 - unist_wise05 CI lower > 0 and point > the base noise floor of that cell; and no direction x {a, b} x
{clean, noisy} x {text, ASR-BLEU} cell with point < -0.3 and CI upper < 0.
Verdict: PASS = rule 1 and rule 2; "NO_INCREMENT over unist_wise05" = rule 1 only; FAIL = rule 1 fails.
Auxiliary: enc (alpha 1) vs base / unist_wise05; mechanism prediction 1 = zh2en/a/clean text enc - base_mean CI upper >= 0.
Self-checks (else no output): erm_wise05 vs base equals wave5/M/summary_erm_wise05.json exactly (as in wave 14); if
wave14/summary.json exists, unist_wise05 and unist_wise05 - base / - base_mean equal it exactly in every cell.
    <qwen3omni python> experiments/mist_w3po_v1/voicecond/wave17_score.py --out eval_outputs/mist_w3po_v1/wave17/summary.json

--- wave14_score.py docstring ---
Robust wave 14 scoring (reports/mist_w3po_v1/robust_wave14_unist_prereg.zh-CN.md), a copy of wave10_score.py with the
arms unist_wise05 (ERM on UniST pilot data, WiSE 0.5) and unist (alpha 1, auxiliary, no gate) against the S1 base (3 base
runs) and against wave 5's erm_wise05 (same chain, same T1c Talker, seed 20261021, 4 shards).

Rule 1 (unist_wise05 vs base; wave 5's unified rule): every direction x {a, b} x {clean, noisy}, text BLEU (vs the mean
of the 3 base runs) and ASR-BLEU (vs S1): point >= -0.3 and CI upper >= 0; and at least one noisy cell (direction x
{a, b, b_no_crosstalk}) with a text BLEU gain over the base mean whose CI lower > 0 and point > noise floor.
Rule 2 (increment vs erm_wise05): at least one cell (direction x {a, b, b_no_crosstalk} x {clean, noisy}) with text BLEU
unist_wise05 - erm_wise05 CI lower > 0 and point > the base noise floor of that cell; and no direction x {a, b} x
{clean, noisy} x {text, ASR-BLEU} cell with point < -0.3 and CI upper < 0.
Verdict: PASS = rule 1 and rule 2; "no increment from the data swap" = rule 1 only; FAIL = rule 1 fails.
Self-check: erm_wise05 vs base (every cell's text / ASR difference and CI, and its rule-1 gate) must equal
wave5/M/summary_erm_wise05.json exactly; the script stops otherwise.
    <qwen3omni python> experiments/mist_w3po_v1/voicecond/wave14_score.py --out eval_outputs/mist_w3po_v1/wave14/summary.json

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
         "enc_wise05": REPO / "eval_outputs/mist_w3po_v1/wave17b/eval_enc_wise05",
         "enc": REPO / "eval_outputs/mist_w3po_v1/wave17b/eval_enc"}
REF5 = REPO / "eval_outputs/mist_w3po_v1/wave5/M/summary_erm_wise05.json"
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
        for x, y in (("enc_wise05", "base"), ("enc_wise05", "base_mean"), ("enc_wise05", "erm_wise05"),
                     ("enc", "base"), ("enc", "base_mean"), ("enc", "erm_wise05"), ("enc", "enc_wise05"),
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
    for arm in ("enc_wise05", "erm_wise05"):
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
        if arm == "enc_wise05":
            kk = "enc_wise05-erm_wise05"
            # wave 17b rule 2 (plan 4.3, reference erm_wise05): clean OR noisy cells count for the increment
            inc = {f"{d}/{sub}/{c}/text": res[d][sub][c]["text"][kk] + [res[d][sub][c]["text"]["base_floor"]]
                   for d in ("en2zh", "zh2en") for sub in ("a", "b", "b_no_crosstalk") for c in ("clean", "noisy")}
            inc_ok = any(v[1] > 0 and v[0] > v[3] for v in inc.values())
            harm = {f"{d}/{sub}/{c}/{m}": res[d][sub][c][m][kk] for d in ("en2zh", "zh2en")
                    for sub in ("a", "b") for c in ("clean", "noisy") for m in ("text", "asr")}
            harm_bad = {x: v for x, v in harm.items() if v[0] < -0.3 and v[2] < 0}
            g.update({"rule2_increment": inc_ok, "rule2_increment_cells_[d,lo,hi,floor]": {x: v for x, v in inc.items() if v[1] > 0},
                      "rule2_all_text_vs_erm_wise05_[d,lo,hi,floor]": inc,
                      "rule2_no_harm": not harm_bad, "rule2_harm_cells": harm_bad,
                      "rule2_PASS": inc_ok and not harm_bad})
            g["VERDICT"] = ("PASS" if g["rule1_PASS"] and g["rule2_PASS"] else
                            "NO_INCREMENT (rule 1 only: no increment over erm_wise05)" if g["rule1_PASS"]
                            else "FAIL (rule 1)")
        gates[arm] = g
    res["gates"] = gates
    # auxiliary: enc (alpha 1) vs base / erm_wise05, no gate; mechanism prediction 1 on zh2en/a/clean
    if all("enc" in res[d]["a"]["clean"]["text"] for d in ("en2zh", "zh2en")):
        z = res["zh2en"]["a"]["clean"]["text"]
        res["aux_enc_alpha1"] = {
            "zh2en/a/clean": {m: {k: res["zh2en"]["a"]["clean"][m].get(k) for k in ("base", "base_mean", "erm_wise05", "enc",
                              "enc-base", "enc-base_mean", "enc-erm_wise05")} for m in ("text", "asr")},
            "prediction1_no_significant_clean_zh2en_drop": bool(z["enc-base_mean"][2] >= 0),
            "no_degradation_failures_vs_base": {f"{d}/{sub}/{c}/{m}": v for d in ("en2zh", "zh2en") for sub in ("a", "b")
                                                for c in ("clean", "noisy") for m in ("text", "asr")
                                                for v in [res[d][sub][c][m]["enc-base_mean" if m == "text" else "enc-base"]]
                                                if not (v[0] >= -0.3 and v[2] >= 0)},
            "noisy_text_gains_vs_base_mean_[d,lo,hi,floor]": {
                f"{d}/{sub}/noisy/text": res[d][sub]["noisy"]["text"]["enc-base_mean"] + [res[d][sub]["noisy"]["text"]["base_floor"]]
                for d in ("en2zh", "zh2en") for sub in ("a", "b", "b_no_crosstalk")}}
    # self-check against wave 5's erm_wise05 summary
    ref = json.load(open(REF5))
    nchk = 0
    for d in ("en2zh", "zh2en"):
        for sub in ("a", "b", "b_no_crosstalk"):
            for c in ("clean", "noisy"):
                for m in ("text", "asr", "d1", "d2"):
                    for k in ("erm_wise05-base", "erm_wise05-base_mean", "base", "erm_wise05", "base_mean", "base_floor"):
                        if k in ref[d][sub][c].get(m, {}):
                            assert res[d][sub][c][m][k] == ref[d][sub][c][m][k], (d, sub, c, m, k, res[d][sub][c][m][k],
                                                                                 ref[d][sub][c][m][k])
                            nchk += 1
    re5 = ref["gates"]["erm_wise05"]
    assert gates["erm_wise05"]["rule1_PASS"] == re5["PASS"]
    assert gates["erm_wise05"]["rule1_no_degradation_failures"] == re5["no_degradation_failures"]
    assert gates["erm_wise05"]["rule1_improvement_cells_[d,lo,hi,floor]"] == re5["improvement_cells_[d,lo,hi,floor]"]
    res["self_check"] = f"erm_wise05 vs base equals {REF5.relative_to(REPO)} ({nchk} values + rule-1 gate)"
    print(res["self_check"], flush=True)
    with open(out, "x", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=1)
    print(json.dumps(gates, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()

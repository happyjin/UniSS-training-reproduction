"""Robust wave 16 NCWI, group ncwi_erm (change B: CoVoST ERM endpoints, reference erm_wise05; wave16_score.py with
those systems). Original docstring:
Robust wave 16 NCWI scoring (reports/mist_w3po_v1/robust_wave16_ncwi_prereg.zh-CN.md section 4; method and rule: main-line
plan 4.2), a copy of wave14_score.py with the system ncwi (the root named in wave16/NCWI_ROOT: eval_ncwi = LoRA path,
eval_ncwi_merged = merged fallback; both cross-fit halves in one output) against unist_wise05, base (3 runs), erm_wise05 and
unist (alpha 1, listed only).

Main rule (4.2, against unist_wise05), on the merged halves:
  no harm   no direction x {a, b} x {clean, noisy} x {text, ASR-BLEU} cell with ncwi - unist_wise05 point < -0.3 and CI
            upper < 0;
  gain      at least one direction x {a, b, b_no_crosstalk} x {clean, noisy} cell with text ncwi - unist_wise05 CI lower
            > 0 and point > that cell's base noise floor (largest pairwise difference of the 3 base runs).
  PASS = both. Paired bootstrap 95 % CI (2000, seed 0) over sentences (a) / single-speaker streams (b), as waves 5 / 14.
Auxiliary (no gate): ncwi vs base with wave 5's unified rule; per-half (cross-fit half 0 / 1) ncwi - unist_wise05 on a, b;
D1 / D2 LAAL; the fitted alpha* per bin; b-hat vs true BVR; the LoRA consistency check.
Self-checks (the script stops otherwise): erm_wise05 vs base equals wave5/M/summary_erm_wise05.json; every unist_wise05 /
unist / base value wave 14 reported equals wave14/summary.json.
    <qwen3omni python> experiments/mist_w3po_v1/voicecond/wave16_score.py --out eval_outputs/mist_w3po_v1/wave16/summary.json
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

M16 = REPO / "eval_outputs/mist_w3po_v1/wave16/erm"
NOISE = REPO / "eval_outputs/mist_w3po_v1/wave16/noise"
ROOTS = {"base": REPO / "eval_outputs/mist_w3po_v1/wave2_S1",
         "erm_wise05": REPO / "eval_outputs/mist_w3po_v1/wave5/M/eval_erm_wise05",
         "erm": REPO / "eval_outputs/mist_w3po_v1/wave5/M/eval_erm"}
REF5 = REPO / "eval_outputs/mist_w3po_v1/wave5/M/summary_erm_wise05.json"
REF14 = REPO / "eval_outputs/mist_w3po_v1/wave5/M/summary.json"   # wave 5 M summary (erm / base values)
RERUNS = {"base_r1": REPO / "eval_outputs/mist_w3po_v1/wave5/base_rerun1",
          "base_r2": REPO / "eval_outputs/mist_w3po_v1/wave5/base_rerun2"}
BASES = ("base", "base_r1", "base_r2")
SEED = "20261021"
NB = 2000
PAIRS = (("ncwi", "erm_wise05"), ("ncwi", "base"), ("ncwi", "base_mean"), ("ncwi", "erm"),
         ("erm", "base"), ("erm", "base_mean"),
         ("erm_wise05", "base"), ("erm_wise05", "base_mean"))


def rel(p: Path) -> str:
    try:
        return str(p.relative_to(REPO))
    except ValueError:
        return str(p)


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
        for x, y in PAIRS:
            if x in vals and y in vals:
                d = np.array(vals[x]) - np.array(vals[y])
                r[f"{x}-{y}"] = [round(float(d[0]), 3), round(float(np.percentile(d[1:], 2.5)), 3),
                                 round(float(np.percentile(d[1:], 97.5)), 3)]
        out[k] = r
    return out


def rule1(res, arm):
    """wave 5's unified rule against the base (text vs the 3-run mean, ASR vs S1)"""
    k, km = f"{arm}-base", f"{arm}-base_mean"
    nodeg = {f"{d}/{sub}/{c}/{m}": res[d][sub][c][m][km if m == "text" else k] for d in ("en2zh", "zh2en")
             for sub in ("a", "b") for c in ("clean", "noisy") for m in ("text", "asr")}
    nd_ok = all(v[0] >= -0.3 and v[2] >= 0 for v in nodeg.values())
    imp = {f"{d}/{sub}/noisy/text": res[d][sub]["noisy"]["text"][km] + [res[d][sub]["noisy"]["text"]["base_floor"]]
           for d in ("en2zh", "zh2en") for sub in ("a", "b", "b_no_crosstalk")}
    imp_ok = any(v[1] > 0 and v[0] > v[3] for v in imp.values())
    return {"rule1_no_degradation": nd_ok,
            "rule1_no_degradation_failures": {x: v for x, v in nodeg.items() if not (v[0] >= -0.3 and v[2] >= 0)},
            "rule1_improvement": imp_ok, "rule1_improvement_cells_[d,lo,hi,floor]": {x: v for x, v in imp.items() if v[1] > 0},
            "rule1_PASS": nd_ok and imp_ok}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    rootn = (M16 / "NCWI_ROOT").read_text().strip()
    assert rootn in ("eval_ncwi", "eval_ncwi_merged"), rootn
    roots = dict(ROOTS, ncwi=M16 / rootn)
    assert (roots["ncwi"] / "EVAL_DONE").exists(), roots["ncwi"]
    LD = S.laal_defs()
    res = {"systems": {k: rel(v) for k, v in roots.items()}, "seed": SEED,
           "ncwi_path": "LoRA (per-request adapters)" if rootn == "eval_ncwi" else "merged fallback (one copy per alpha*)"}
    for d in ("en2zh", "zh2en"):
        pers, meta = {}, None
        for sysn, root in roots.items():
            if not (root / f"talker_{d}_s{SEED}" / "asr_speech.jsonl").exists():
                raise SystemExit(f"{sysn} {d}: no ASR")
            meta, pers[sysn] = S.load_dir(root, d, [SEED], LD)
        for rn, rr in RERUNS.items():
            blk = {r["id"]: "".join(b["text"] for b in r["blocks"]) for r in S.jl(rr / f"thinker_{d}" / f"blocks_{d}_w1.jsonl")}
            assert set(blk) >= set(meta), f"{rn} {d}: missing items"
            pers[rn] = {i: {"hyp": blk[i], "ref": pers["base"][i]["ref"]} for i in meta}
        half = {r["id"]: r["half"] for r in S.jl(M16 / "fit" / f"assign_{d}.jsonl")}
        assert set(half) == set(meta)
        lang = S.LANG[d]
        ovl = {m["pair"] for m in meta.values() if m["set"] == "b" and m["ovl"]}
        res[d] = {}
        for sub, set_, keep in (("a", "a", lambda m: True), ("b", "b", lambda m: True),
                                ("b_no_crosstalk", "b", lambda m, o=ovl: m["pair"] not in o),
                                ("a_half0", "a", lambda m: half[m["id"]] == 0), ("a_half1", "a", lambda m: half[m["id"]] == 1),
                                ("b_half0", "b", lambda m: half[m["id"]] == 0), ("b_half1", "b", lambda m: half[m["id"]] == 1)):
            res[d][sub] = {c: cell(meta, pers, lang, set_, c, keep) for c in ("clean", "noisy")}
    # main rule (4.2): ncwi against unist_wise05
    kk = "ncwi-erm_wise05"
    harm = {f"{d}/{sub}/{c}/{m}": res[d][sub][c][m][kk] for d in ("en2zh", "zh2en")
            for sub in ("a", "b") for c in ("clean", "noisy") for m in ("text", "asr")}
    harm_bad = {x: v for x, v in harm.items() if v[0] < -0.3 and v[2] < 0}
    gain = {f"{d}/{sub}/{c}/text": res[d][sub][c]["text"][kk] + [res[d][sub][c]["text"]["base_floor"]]
            for d in ("en2zh", "zh2en") for sub in ("a", "b", "b_no_crosstalk") for c in ("clean", "noisy")}
    gain_ok = any(v[1] > 0 and v[0] > v[3] for v in gain.values())
    g = {"no_harm": not harm_bad, "harm_cells": harm_bad, "all_cells_[d,lo,hi]": harm,
         "gain": gain_ok, "gain_cells_[d,lo,hi,floor]": {x: v for x, v in gain.items() if v[1] > 0 and v[0] > v[3]},
         "all_text_cells_[d,lo,hi,floor]": gain, "PASS": (not harm_bad) and gain_ok}
    g["VERDICT"] = "PASS" if g["PASS"] else "FAIL"
    res["gate_ncwi_erm_vs_erm_wise05"] = g
    res["aux_ncwi_vs_base_rule1"] = rule1(res, "ncwi")
    res["aux_per_half_ncwi-erm_wise05_[d,lo,hi]"] = {
        f"{d}/{sub}/{c}/{m}": res[d][sub][c][m][kk] for d in ("en2zh", "zh2en")
        for sub in ("a_half0", "a_half1", "b_half0", "b_half1") for c in ("clean", "noisy") for m in ("text", "asr")}
    res["aux_laal_ncwi-erm_wise05_[d,lo,hi]"] = {
        f"{d}/{sub}/{c}/{m}": res[d][sub][c][m].get(kk) for d in ("en2zh", "zh2en")
        for sub in ("a", "b") for c in ("clean", "noisy") for m in ("d1", "d2")}
    fit = json.load(open(M16 / "fit" / "fit.json"))
    res["aux_alpha_star"] = {d: {f"fit_half{h}": {"edges_db": fit[d][f"fit_half{h}"]["edges_db"],
                                                  "alpha_per_bin": [b["alpha"] for b in fit[d][f"fit_half{h}"]["bins"]],
                                                  "rule_per_bin": [b["rule"] for b in fit[d][f"fit_half{h}"]["bins"]]}
                                 for h in (0, 1)} for d in ("en2zh", "zh2en")}
    res["aux_bhat_vs_bvr"] = json.load(open(NOISE / "report.json"))
    if (M16 / "agree.json").exists():
        res["aux_lora_consistency"] = json.load(open(M16 / "agree.json"))
    # self-check 1: wave 5's erm_wise05 summary
    ref = json.load(open(REF5))
    nchk = 0
    for d in ("en2zh", "zh2en"):
        for sub in ("a", "b", "b_no_crosstalk"):
            for c in ("clean", "noisy"):
                for m in ("text", "asr", "d1", "d2"):
                    for k in ("erm_wise05-base", "erm_wise05-base_mean", "base", "erm_wise05", "base_mean", "base_floor"):
                        if k in ref[d][sub][c].get(m, {}):
                            assert res[d][sub][c][m][k] == ref[d][sub][c][m][k], (d, sub, c, m, k)
                            nchk += 1
    ge = rule1(res, "erm_wise05")
    re5 = ref["gates"]["erm_wise05"]
    assert ge["rule1_PASS"] == re5["PASS"]
    assert ge["rule1_no_degradation_failures"] == re5["no_degradation_failures"]
    assert ge["rule1_improvement_cells_[d,lo,hi,floor]"] == re5["improvement_cells_[d,lo,hi,floor]"]
    # self-check 2: wave 5's M summary (erm / erm_wise05 / base values that this script also computes)
    r14 = json.load(open(REF14))
    n14 = 0
    for d in ("en2zh", "zh2en"):
        for sub in ("a", "b", "b_no_crosstalk"):
            for c in ("clean", "noisy"):
                for m in ("text", "asr", "d1", "d2"):
                    for k, v in r14[d][sub][c].get(m, {}).items():
                        if k in res[d][sub][c][m]:
                            assert res[d][sub][c][m][k] == v, (d, sub, c, m, k, res[d][sub][c][m][k], v)
                            n14 += 1
    res["self_check"] = (f"erm_wise05 vs base equals {rel(REF5)} ({nchk} values + rule-1 gate); "
                         f"{n14} values equal {rel(REF14)}")
    print(res["self_check"], flush=True)
    with open(out, "x", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=1)
    print(json.dumps({"gate": {k: g[k] for k in ("VERDICT", "no_harm", "harm_cells", "gain", "gain_cells_[d,lo,hi,floor]")},
                      "aux_rule1_vs_base": res["aux_ncwi_vs_base_rule1"]["rule1_PASS"]}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()

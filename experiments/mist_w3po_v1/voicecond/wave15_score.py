"""Robust wave 15 CV-LA scoring (reports/mist_w3po_v1/robust_wave15_cvla_prereg.zh-CN.md; rule: robust_mainline_plan
section 4.1), a copy of wave14_score.py (same cells, bootstrap, base runs, noise floor and erm_wise05 self-check).

--group 1: cvla_base (base Thinker + CV-LA) against la2_base. Text BLEU against the mean of the 3 base runs (S1 +
  wave5/base_rerun{1,2}), ASR-BLEU against S1 (wave 5 change 3). Gate: every direction x {a, b} x {clean, noisy} x
  {text, ASR-BLEU}: point >= -0.3 and CI upper >= 0; and at least one noisy cell (direction x {a, b, b_no_crosstalk})
  with a text gain whose CI lower > 0 and point > the base noise floor (largest pairwise difference of the 3 base runs).
--group 0: only the self-check below (base and erm_wise05).
--group 2: cvla_unist (merged_unist_wise05 + CV-LA) against unist_wise05 (wave 14's own LA2 run), text and ASR-BLEU
  against that run; same cells; noise floor = the base noise floor (wave 11 rule 2's convention).
Verdict per group: PASS = no degradation and improvement; else FAIL (with the failing part).
Self-check: erm_wise05 vs base (every cell's values and the rule-1 gate) must equal wave5/M/summary_erm_wise05.json.
    <qwen3omni python> experiments/mist_w3po_v1/voicecond/wave15_score.py --group 1 --out eval_outputs/mist_w3po_v1/wave15/summary_g1.json
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

ROOTS_ALL = {"base": REPO / "eval_outputs/mist_w3po_v1/wave2_S1",
             "erm_wise05": REPO / "eval_outputs/mist_w3po_v1/wave5/M/eval_erm_wise05",
             "cvla_base": REPO / "eval_outputs/mist_w3po_v1/wave15/eval_cvla_base",
             "unist_wise05": REPO / "eval_outputs/mist_w3po_v1/wave14/eval_unist_wise05",
             "cvla_unist": REPO / "eval_outputs/mist_w3po_v1/wave15/eval_cvla_unist"}
GROUP = {0: (None, None), 1: ("cvla_base", "base"), 2: ("cvla_unist", "unist_wise05")}   # 0: the self-check alone
REF5 = REPO / "eval_outputs/mist_w3po_v1/wave5/M/summary_erm_wise05.json"
RERUNS = {"base_r1": REPO / "eval_outputs/mist_w3po_v1/wave5/base_rerun1",
          "base_r2": REPO / "eval_outputs/mist_w3po_v1/wave5/base_rerun2"}   # wave 5 change 3: Thinker-only reruns
BASES = ("base", "base_r1", "base_r2")
SEED = "20261021"
NB = 2000
PAIRS = (("cvla_base", "base"), ("cvla_base", "base_mean"), ("cvla_unist", "unist_wise05"), ("cvla_unist", "base"),
         ("cvla_unist", "base_mean"), ("unist_wise05", "base"), ("unist_wise05", "base_mean"),
         ("erm_wise05", "base"), ("erm_wise05", "base_mean"))


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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", type=int, choices=(0, 1, 2), required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    arm, ctl = GROUP[a.group]
    keep_sys = {"base", "erm_wise05", arm, ctl}
    ROOTS = {k: v for k, v in ROOTS_ALL.items() if k in keep_sys}
    for k, v in ROOTS.items():
        for d in ("en2zh", "zh2en"):
            assert (v / f"talker_{d}_s{SEED}" / "asr_speech.jsonl").exists(), f"{k} {d}: no asr_speech.jsonl"
    LD = S.laal_defs()
    res = {"group": a.group, "arm": arm, "control": ctl, "systems": {k: str(v.relative_to(REPO)) for k, v in ROOTS.items()},
           "seed": SEED}
    for d in ("en2zh", "zh2en"):
        pers, meta = {}, None
        for sysn, root in ROOTS.items():
            meta, pers[sysn] = S.load_dir(root, d, [SEED], LD)
        for rn, rr in RERUNS.items():
            bf = rr / f"thinker_{d}" / f"blocks_{d}_w1.jsonl"
            blk = {r["id"]: "".join(b["text"] for b in r["blocks"]) for r in S.jl(bf)}
            assert set(blk) >= set(meta), f"{rn} {d}: missing items"
            pers[rn] = {i: {"hyp": blk[i], "ref": pers["base"][i]["ref"]} for i in meta}
        lang = S.LANG[d]
        ovl = {m["pair"] for m in meta.values() if m["set"] == "b" and m["ovl"]}
        res[d] = {}
        for sub, set_, keep in (("a", "a", lambda m: True), ("b", "b", lambda m: True),
                                ("b_no_crosstalk", "b", lambda m, o=ovl: m["pair"] not in o)):
            res[d][sub] = {c: cell(meta, pers, lang, set_, c, keep) for c in ("clean", "noisy")}
    gates = {}
    # wave 14's erm_wise05 rule-1 gate, needed for the self-check
    k, km = "erm_wise05-base", "erm_wise05-base_mean"
    nodeg = {f"{d}/{sub}/{c}/{m}": res[d][sub][c][m][km if m == "text" else k] for d in ("en2zh", "zh2en")
             for sub in ("a", "b") for c in ("clean", "noisy") for m in ("text", "asr")}
    imp = {f"{d}/{sub}/noisy/text": res[d][sub]["noisy"]["text"][km] + [res[d][sub]["noisy"]["text"]["base_floor"]]
           for d in ("en2zh", "zh2en") for sub in ("a", "b", "b_no_crosstalk")}
    nd_ok = all(v[0] >= -0.3 and v[2] >= 0 for v in nodeg.values())
    imp_ok = any(v[1] > 0 and v[0] > v[3] for v in imp.values())
    gates["erm_wise05"] = {"rule1_PASS": nd_ok and imp_ok,
                           "rule1_no_degradation_failures": {x: v for x, v in nodeg.items() if not (v[0] >= -0.3 and v[2] >= 0)},
                           "rule1_improvement_cells_[d,lo,hi,floor]": {x: v for x, v in imp.items() if v[1] > 0}}
    # the group's gate (plan 4.1)
    kt = ka = None
    if arm is not None:
        kt = f"{arm}-base_mean" if ctl == "base" else f"{arm}-{ctl}"
        ka = f"{arm}-{ctl}"
        nodeg = {f"{d}/{sub}/{c}/{m}": res[d][sub][c][m][kt if m == "text" else ka] for d in ("en2zh", "zh2en")
                 for sub in ("a", "b") for c in ("clean", "noisy") for m in ("text", "asr")}
        nd_ok = all(v[0] >= -0.3 and v[2] >= 0 for v in nodeg.values())
        imp = {f"{d}/{sub}/noisy/text": res[d][sub]["noisy"]["text"][kt] + [res[d][sub]["noisy"]["text"]["base_floor"]]
               for d in ("en2zh", "zh2en") for sub in ("a", "b", "b_no_crosstalk")}
        imp_ok = any(v[1] > 0 and v[0] > v[3] for v in imp.values())
        gates[arm] = {"vs": ctl, "text_key": kt, "asr_key": ka,
                      "no_degradation": nd_ok, "all_cells_[d,lo,hi]": nodeg,
                      "no_degradation_failures": {x: v for x, v in nodeg.items() if not (v[0] >= -0.3 and v[2] >= 0)},
                      "improvement": imp_ok, "noisy_text_cells_[d,lo,hi,floor]": imp,
                      "improvement_cells_[d,lo,hi,floor]": {x: v for x, v in imp.items() if v[1] > 0 and v[0] > v[3]},
                      "laal_d1_d2_[d,lo,hi]": {f"{d}/{sub}/{c}/{m}": res[d][sub][c][m].get(ka) for d in ("en2zh", "zh2en")
                                               for sub in ("a", "b") for c in ("clean", "noisy") for m in ("d1", "d2")},
                      "VERDICT": "PASS" if nd_ok and imp_ok else
                                 "FAIL (" + ", ".join(x for x, ok in (("degradation", nd_ok), ("no significant noisy gain", imp_ok))
                                                      if not ok) + ")"}
    res["gates"] = gates
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
    assert nchk > 0
    res["self_check"] = f"erm_wise05 vs base equals {REF5.relative_to(REPO)} ({nchk} values + rule-1 gate)"
    print(res["self_check"], flush=True)
    with open(out, "x", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=1)
    print(json.dumps(gates.get(arm, gates["erm_wise05"]), ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()

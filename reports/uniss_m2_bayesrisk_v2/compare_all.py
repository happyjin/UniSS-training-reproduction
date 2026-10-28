"""Every RealSI system of the commit-policy line in one table per direction: the non-Bayesian LocalAgreement-2 baselines,
M2, the v3 bias curve and its variants (no SDR, calibrated SDR, per-talk Beta), bayesrisk v1 and v2.

ASR-BLEU, UTMOS and abrupt pauses are also shown against the v3 curve interpolated at the system's LAAL (outside the
curve: the nearest segment extrapolated, LAAL flagged *); the last column is the paired bootstrap against v3 where one
was run (v1/v2: the nearest-LAAL v3 point; v3-config variants: the default v3 point), shown only when that point
is within 0.25 s LAAL; bold = the paired 95% CI excludes 0 (BLEU in the last column, UTMOS in its own column). Abrupt pauses come from the
continuous-timeline analysis (s1b1.2q playback; LA2 originals and M2 use plain playback, as they were run).
    python3 compare_all.py > compare_all.md            [--only v3,br2Ah3t-b1,...]  (a subset of runs)
"""
import argparse
import json
import os
import re

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.dirname(HERE)
V3C, V1, V2 = f"{R}/uniss_m2_fluency_bayes_v3cfg", f"{R}/uniss_m2_bayesrisk_v1", HERE
FINALS = [f"{V3C}/final_{{d}}.json", f"{V1}/final_{{d}}.json", f"{V2}/final_{{d}}.json", f"{V2}/final_r2_{{d}}.json"]
ABRUPT = {  # run -> abrupt file stem when it is not <run>_s1b1.2q in the v1/v2 abrupt dirs
    "m2_en2zh": "uniss_m2_fluency_results_v2/abrupt/m2_en2zh", "m2_zh2en": "uniss_m2_fluency_results_v2/abrupt/m2_zh2en",
    "v3_en2zh": "uniss_m2_fluency_results_v3/abrupt/sqb1_en2zh_s1b1.2q",
    "v3b0.5_en2zh": "uniss_m2_fluency_results_v3/abrupt/sq_en2zh_s1b1.2q",
    "v3_zh2en": "uniss_m2_fluency_results_v3/abrupt/sq_zh2en_s1b1.2q",
    "v3bm0.5_zh2en": "uniss_m2_fluency_results_v3/abrupt/sqbm0p5_zh2en_s1b1.2q",
    "v3b1.5_en2zh": "uniss_m2_fluency_bayes_v3cfg/abrupt/bay-v3-b1.5_m0.5_en2zh_s1b1.2q",
    "v3b2_en2zh": "uniss_m2_fluency_bayes_v3cfg/abrupt/bay-v3-b2_m0.5_en2zh_s1b1.2q",
    "v3b0_zh2en": "uniss_m2_fluency_bayes_v3cfg/abrupt/bay-v3-zh-b0_m1_zh2en_s1b1.2q",
    "v3b0.5_zh2en": "uniss_m2_fluency_bayes_v3cfg/abrupt/bay-v3-zh-b0.5_m1_zh2en_s1b1.2q",
    "nosdr_en2zh": "uniss_m2_fluency_bayes_v3cfg/abrupt/nosdr_en2zh_s1b1.2q",
    "nosdr_zh2en": "uniss_m2_fluency_bayes_v3cfg/abrupt/nosdr_zh2en_s1b1.2q",
    "sdrcal_en2zh": "uniss_m2_fluency_bayes_v3cfg/abrupt/sdrcal_en2zh_s1b1.2q",
    "sdrcal_zh2en": "uniss_m2_fluency_bayes_v3cfg/abrupt/sdrcal_zh2en_s1b1.2q",
    "betab0.5_en2zh": "uniss_m2_fluency_bayes_v3cfg/abrupt/bay-beta-b0.5_m0.5_en2zh_s1b1.2q",
    "betab1_en2zh": "uniss_m2_fluency_bayes_v3cfg/abrupt/beta_en2zh_s1b1.2q",
    "betabm1.5_zh2en": "uniss_m2_fluency_bayes_v3cfg/abrupt/bay-beta-zh-b-1.5_m1_zh2en_s1b1.2q",
    "betabm1_zh2en": "uniss_m2_fluency_bayes_v3cfg/abrupt/beta_zh2en_s1b1.2q",
    "la2w0.75_en2zh": "uniss_m2_fluency_bayes_v3cfg/abrupt/la2w0.75_en2zh_plain",
    "la2w1_en2zh": "uniss_m2_fluency_bayes_v3cfg/abrupt/la2w1_en2zh_plain",
    "la2w0.5_zh2en": "uniss_m2_fluency_bayes_v3cfg/abrupt/la2w0.5_zh2en_plain",
    "la2w0.75_zh2en": "uniss_m2_fluency_bayes_v3cfg/abrupt/la2w0.75_zh2en_plain",
    "la2v3w0.5_en2zh": "uniss_m2_fluency_bayes_v3cfg/abrupt/la2v3_m0.5_en2zh_s1b1.2q",
    "la2v3w0.6_en2zh": "uniss_m2_fluency_bayes_v3cfg/abrupt/la2v3_m0.6_en2zh_s1b1.2q",
    "la2v3w0.75_en2zh": "uniss_m2_fluency_bayes_v3cfg/abrupt/la2v3w0.75_en2zh_s1b1.2q",
    "la2v3w1_en2zh": "uniss_m2_fluency_bayes_v3cfg/abrupt/la2v3w1_en2zh_s1b1.2q",
    "la2v3w0.4_zh2en": "uniss_m2_fluency_bayes_v3cfg/abrupt/la2v3-zh_m0.4_zh2en_s1b1.2q",
    "la2v3w0.5_zh2en": "uniss_m2_fluency_bayes_v3cfg/abrupt/la2v3w0.5_zh2en_s1b1.2q",
    "la2v3w0.75_zh2en": "uniss_m2_fluency_bayes_v3cfg/abrupt/la2v3w0.75_zh2en_s1b1.2q",
}
GROUPS = [  # (group label, run-name prefix, label of a run)
    ("非 Bayesian：LA2 原版", "la2w", lambda b: f"LA2 w{b}"),
    ("非 Bayesian：LA2 + v3 规则", "la2v3w", lambda b: f"LA2+v3 w{b}"),
    ("Bayesian 无规则：M2（ABC+SDR）", "m2", lambda b: "M2"),
    ("**基线 v3**（ABC+SDR+规则）", "v3", lambda b: f"v3 b{b}"),
    ("v3 消融：去掉 SDR（只 ABC）", "nosdr", lambda b: "去掉 SDR"),
    ("v3 变体：SDR 按实测语速校准", "sdrcal", lambda b: "SDR 校准"),
    ("v3 变体：按场 Beta 在线后验", "beta", lambda b: f"Beta b{b}"),
    ("Bayes 风险 v1", "br", lambda b: pretty(b)),
    ("Bayes 风险 v2", "br2", lambda b: pretty(b)),
    ("Bayes 风险 v2（第二次 RealSI，zh→en）", "br3", lambda b: pretty(b)),
]
PAIR_MAX_DLAAL = 0.25  # a paired test against a v3 point further away in LAAL measures latency, not the policy
DEFAULT_B = {"v3_en2zh": "1", "v3_zh2en": "-1", "nosdr": "", "sdrcal": "", "m2": ""}


def pretty(code):
    """bayesrisk run code -> readable name: A = ABC prior, P/PF = fitted posterior, PO/PFO = fitted + online posterior,
    h<m>t = hard hygiene (min m units, trim), a<l> = abrupt-pause cost lambda, d = W/P-fw seams excluded."""
    m = re.match(r"(PFO|PF|PO|P|A)(\d*)(d?)(?:h(\d)t)?(?:-a([\d.]+))?-b(m?-?[\d.]+)$", code)
    if not m:
        return code
    post, _, dangle, hyg, lam, b = m.groups()
    name = {"A": "A", "P": "P", "PF": "P", "PO": "PO", "PFO": "PO"}[post]
    name += (f"+硬卫生m{hyg}" if hyg else "") + ("+排除虚词" if dangle else "") + (f"+λ{lam}" if lam else "")
    return f"{name} b{b.replace('m', '-')}"


def abrupt(run):
    stems = [f"{V2}/abrupt/{run}_s1b1.2q", f"{V1}/abrupt/{run}_s1b1.2q"]
    if run in ABRUPT:
        stems.append(f"{R}/{ABRUPT[run]}")
    for s in stems:
        if os.path.exists(s + ".txt"):
            rows = [line.split(" ", 1)[1] for line in open(s + ".txt") if ' {"pauses_ge0.5"' in line]
            if rows:
                return json.loads(rows[-1])["abrupt_per_min_speech"]
    return None


def paired(run):
    """Paired bootstrap of run against v3: v1/v2 pair files (nearest-LAAL v3 point), else the v3-config final json
    (the default v3 point)."""
    for f in [f"{V2}/pair_{run}.json"] + [f"{V1}/{p}" for p in sorted(os.listdir(V1)) if p.startswith("pair_v3") and p.endswith(".json")]:
        if os.path.exists(f):
            res = json.load(open(f))
            if run in res and "vs_base_bleu" in res[run]:
                base = next(k for k in res if k != run and k.startswith("v3"))
                return base, res[run]["vs_base_bleu"], res[run].get("vs_base_utmos")
    return None


def group_of(run):
    stem = run.rsplit("_", 1)[0]
    if stem.startswith("br3"):
        return 9, stem[3:]
    if stem.startswith("br2"):
        return 8, stem[3:]
    if stem.startswith("br"):
        return 7, stem[2:]
    for i, (_, p, _) in enumerate(GROUPS[:7]):
        if stem.startswith(p) and not (p == "la2w" and stem.startswith("la2v3")):
            b = stem[len(p):].lstrip("b").replace("m", "-") if p in ("v3", "beta") else stem[len(p):]
            return i, b
    raise KeyError(run)


def fmt_delta(x, d, nd):
    return f"{x:.{nd}f}" if d is None else f"{x:.{nd}f} ({d:+.{nd}f})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="", help="comma-separated run names (without _<direction>) to print; default all")
    only = set(filter(None, ap.parse_args().only.split(",")))
    for d in ("en2zh", "zh2en"):
        res, base_pair = {}, {}
        for f in FINALS:
            f = f.format(d=d)
            if os.path.exists(f):
                for k, v in json.load(open(f)).items():
                    res.setdefault(k, v)
                    if "vs_base_bleu" in v and f.startswith(V3C):
                        base_pair[k] = (f"v3_{d}", v["vs_base_bleu"], v.get("vs_base_utmos"))
        v3 = sorted((k for k in res if k.startswith("v3")), key=lambda k: res[k]["laal_s"])
        xs = np.array([res[k]["laal_s"] for k in v3])

        def at(x, m):
            ys = np.array([res[k][m] for k in v3])
            if xs[0] <= x <= xs[-1]:
                return float(np.interp(x, xs, ys))
            i = 0 if x < xs[0] else len(xs) - 2
            return float(ys[i] + (ys[i + 1] - ys[i]) * (x - xs[i]) / (xs[i + 1] - xs[i]))

        ab3 = [(res[k]["laal_s"], abrupt(k)) for k in v3]
        ab3 = [p for p in ab3 if p[1] is not None]
        print(f"\n## {'en→zh' if d == 'en2zh' else 'zh→en'}（RealSI，{d}）\n")
        print("| 组 | 系统 | LAAL s | ASR-BLEU（Δ 同 LAAL v3） | UTMOS（Δ） | 突兀停顿/分（×v3） | 单字块 | 多余停顿 s/100u | "
              "≥1 s 停顿/100u | 静音占比 | Talker 读错 | 配对 ΔBLEU vs v3 [95% CI] |")
        print("|---" * 12 + "|")
        rows = sorted(res, key=lambda k: (group_of(k)[0], res[k]["laal_s"]))
        last = None
        for k in rows:
            if only and k.rsplit("_", 1)[0] not in only:
                continue
            g, b = group_of(k)
            v, x = res[k], res[k]["laal_s"]
            isv3 = g == 3
            ext = "" if xs[0] <= x <= xs[-1] else "*"
            db = None if isv3 else v["asr_bleu"] - at(x, "asr_bleu")
            du = None if isv3 or "utmos" not in v else v["utmos"] - at(x, "utmos")
            a = abrupt(k)
            if a is None:
                ac = "—"
            elif isv3:
                ac = f"{a:.2f}"
            else:
                ya = float(np.interp(x, [p[0] for p in ab3], [p[1] for p in ab3]))
                ac = f"{a:.2f}（×{a / ya:.2f}）"
            p = None if isv3 else (paired(k) or base_pair.get(k))
            if p and abs(res[p[0]]["laal_s"] - x) > PAIR_MAX_DLAAL:
                p = None
            pc = "—" if p is None else (
                f"{p[1]['diff']:+.2f} [{p[1]['ci95'][0]:+.2f}, {p[1]['ci95'][1]:+.2f}]（{p[0].split('_')[0]}）")
            if p and (p[1]["ci95"][0] > 0 or p[1]["ci95"][1] < 0):
                pc = f"**{pc}**"
            name = GROUPS[g][2](b)
            if isv3 and k == f"v3_{d}":
                name = f"**v3 b{DEFAULT_B[k]}（默认）**"
            label = GROUPS[g][0] if g != last else ""
            last = g
            u = f"{v['utmos']:.3f}" if du is None and "utmos" in v else ("—" if "utmos" not in v else f"{v['utmos']:.3f} ({du:+.3f})")
            pu = p[2] if p else None
            if pu and "ci95" in pu and (pu["ci95"][0] > 0 or pu["ci95"][1] < 0):
                u = f"**{u}**"  # paired UTMOS difference significant
            print(f"| {label} | {name} | {x:.2f}{ext} | {fmt_delta(v['asr_bleu'], db, 2)} | {u} | {ac} | "
                  f"{100 * v['one_unit_blocks']:.1f}% | {v['excess_pause_s_per_100u']:.2f} | {v['pauses_ge1_per_100u']:.2f} | "
                  f"{100 * v['silence_ratio']:.1f}% | {100 * v['talker_err']:.1f}% | {pc} |")


if __name__ == "__main__":
    main()

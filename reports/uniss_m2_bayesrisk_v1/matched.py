"""Matched-latency view of the bayesrisk systems against the v3 bias curve (RealSI).

For each bayesrisk run: the v3 curve (runs named v3*) interpolated linearly in LAAL at the run's LAAL (no
extrapolation: outside the curve the nearest end is reported and flagged), for ASR-BLEU and the fluency metrics;
the paired bootstrap against the nearest v3 point comes from the pair_<d>_<v3 run>.json files (pair_cmd.sh).
    python3 matched.py <compare json en2zh> <compare json zh2en> > matched.md
"""
import glob
import json
import os
import sys

import numpy as np

M = ["asr_bleu", "laal_s", "utmos", "pauses_ge1_per_100u", "excess_pause_s_per_100u", "one_unit_blocks",
     "adjacent_rate_jump_gt1.5x", "talker_err", "silence_ratio", "end_offset_s_p90"]
HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.dirname(HERE)
# v3 points whose continuous analysis was run for earlier reports (same abrupt.py, mode s1b1.2q)
ABRUPT_OLD = {"v3_en2zh": "uniss_m2_fluency_results_v3/abrupt/sqb1_en2zh", "v3b0.5_en2zh": "uniss_m2_fluency_results_v3/abrupt/sq_en2zh",
              "v3_zh2en": "uniss_m2_fluency_results_v3/abrupt/sq_zh2en", "v3bm0.5_zh2en": "uniss_m2_fluency_results_v3/abrupt/sqbm0p5_zh2en",
              "v3b1.5_en2zh": "uniss_m2_fluency_bayes_v3cfg/abrupt/bay-v3-b1.5_m0.5_en2zh", "v3b2_en2zh": "uniss_m2_fluency_bayes_v3cfg/abrupt/bay-v3-b2_m0.5_en2zh",
              "v3b0_zh2en": "uniss_m2_fluency_bayes_v3cfg/abrupt/bay-v3-zh-b0_m1_zh2en", "v3b0.5_zh2en": "uniss_m2_fluency_bayes_v3cfg/abrupt/bay-v3-zh-b0.5_m1_zh2en"}


def abrupt(name):
    f = f"{R}/{ABRUPT_OLD[name]}_s1b1.2q.txt" if name in ABRUPT_OLD else f"{HERE}/abrupt/{name}_s1b1.2q.txt"
    if not os.path.exists(f):
        return None
    rows = [line.split(" ", 1)[1] for line in open(f) if ' {"pauses_ge0.5"' in line]
    return json.loads(rows[-1]) if rows else None


def pairs():
    out = {}
    for f in glob.glob(f"{HERE}/pair_*.json"):
        for k, v in json.load(open(f)).items():
            if "vs_base_bleu" in v:
                out.setdefault(k, []).append((os.path.basename(f), v["vs_base_bleu"], v.get("vs_base_utmos")))
    return out


def main():
    pr = pairs()
    for f in sys.argv[1:]:
        res = json.load(open(f))
        d = "en2zh" if "en2zh" in f else "zh2en"
        v3 = sorted((k for k in res if k.startswith("v3")), key=lambda k: res[k]["laal_s"])
        xs = np.array([res[k]["laal_s"] for k in v3])
        print(f"\n## {d}\n")
        print("v3 curve: " + ", ".join(f"{k} ({res[k]['laal_s']:.2f} s, {res[k]['asr_bleu']:.2f})" for k in v3) + "\n")
        cols = [m for m in M if m != "laal_s" and all(m in res[k] for k in v3)]
        print("| run | LAAL | " + " | ".join(f"{m} (v3@LAAL)" for m in cols) + " | paired vs nearest v3 |")
        print("|---" * (len(cols) + 3) + "|")
        for k in sorted((k for k in res if k.startswith("br")), key=lambda k: res[k]["laal_s"]):
            x = res[k]["laal_s"]
            flag = "" if xs[0] <= x <= xs[-1] else " (extrap.)"
            cells = []
            for m in cols:
                ys = np.array([res[v][m] for v in v3])
                y = float(np.interp(x, xs, ys))
                a = res[k].get(m)
                cells.append("-" if a is None else f"{a:.3g} ({y:.3g}, {a - y:+.2g})")
            pv = "; ".join(f"{p[0][5:-5]}: {p[1]['diff']:+.2f} [{p[1]['ci95'][0]:+.2f},{p[1]['ci95'][1]:+.2f}]"
                           + (f" U{p[2]['diff']:+.3f}" if p[2] else "") for p in pr.get(k, []))
            print(f"| {k} | {x:.2f}{flag} | " + " | ".join(cells) + f" | {pv} |")
        print("\nabrupt (continuous timeline, s1b1.2q):\n")
        for k in v3 + sorted(k for k in res if k.startswith("br")):
            a = abrupt(k)
            if a:
                print(f"- {k}: " + ", ".join(f"{m}={a[m]}" for m in a if not isinstance(a[m], (dict, list))))


if __name__ == "__main__":
    main()

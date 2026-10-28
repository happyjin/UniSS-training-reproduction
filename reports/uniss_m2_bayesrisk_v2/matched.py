"""Matched-latency view of the bayesrisk v2 systems against the v3 bias curve on RealSI (as the v1 report's matched.py).

For each br2 run: the v3 curve interpolated linearly in LAAL at the run's LAAL (outside the curve: the nearest end,
flagged) for ASR-BLEU, UTMOS and the fluency metrics, the abrupt pauses per minute of speech on the continuous
timeline (ratio to v3 at the same LAAL: the v2 gate is <= 1.2x), and the paired bootstrap against the nearest-LAAL v3
point (pair_<run>.json, post.py).
    python3 matched.py final_en2zh.json final_zh2en.json > matched.md
"""
import json
import os
import sys

import numpy as np

M = ["asr_bleu", "utmos", "pauses_ge1_per_100u", "excess_pause_s_per_100u", "one_unit_blocks",
     "adjacent_rate_jump_gt1.5x", "talker_err", "silence_ratio"]
HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.dirname(HERE)
ABRUPT_OLD = {"v3_en2zh": "uniss_m2_fluency_results_v3/abrupt/sqb1_en2zh", "v3b0.5_en2zh": "uniss_m2_fluency_results_v3/abrupt/sq_en2zh",
              "v3_zh2en": "uniss_m2_fluency_results_v3/abrupt/sq_zh2en", "v3bm0.5_zh2en": "uniss_m2_fluency_results_v3/abrupt/sqbm0p5_zh2en",
              "v3b1.5_en2zh": "uniss_m2_fluency_bayes_v3cfg/abrupt/bay-v3-b1.5_m0.5_en2zh", "v3b2_en2zh": "uniss_m2_fluency_bayes_v3cfg/abrupt/bay-v3-b2_m0.5_en2zh",
              "v3b0_zh2en": "uniss_m2_fluency_bayes_v3cfg/abrupt/bay-v3-zh-b0_m1_zh2en", "v3b0.5_zh2en": "uniss_m2_fluency_bayes_v3cfg/abrupt/bay-v3-zh-b0.5_m1_zh2en"}


def abrupt(name):
    for f in (f"{HERE}/abrupt/{name}_s1b1.2q.txt", f"{R}/uniss_m2_bayesrisk_v1/abrupt/{name}_s1b1.2q.txt",
              f"{R}/{ABRUPT_OLD.get(name, '-')}_s1b1.2q.txt"):
        if os.path.exists(f):
            rows = [line.split(" ", 1)[1] for line in open(f) if ' {"pauses_ge0.5"' in line]
            if rows:
                return json.loads(rows[-1])
    return None


def pair(name):
    f = f"{HERE}/pair_{name}.json"
    if not os.path.exists(f):
        return ""
    res = json.load(open(f))
    v = res[name]
    base = next(k for k in res if k != name)
    b = v["vs_base_bleu"]
    s = f"{base}: {b['diff']:+.2f} [{b['ci95'][0]:+.2f},{b['ci95'][1]:+.2f}]"
    if v.get("vs_base_utmos"):
        u = v["vs_base_utmos"]
        s += f" U{u['diff']:+.3f}" + (f" [{u['ci95'][0]:+.3f},{u['ci95'][1]:+.3f}]" if "ci95" in u else "")
    return s


def main():
    for f in sys.argv[1:]:
        res = json.load(open(f))
        d = "en2zh" if "en2zh" in f else "zh2en"
        v3 = sorted((k for k in res if k.startswith("v3")), key=lambda k: res[k]["laal_s"])
        xs = np.array([res[k]["laal_s"] for k in v3])
        ab3 = {k: abrupt(k) for k in v3}
        ok3 = [k for k in v3 if ab3[k]]
        xa = np.array([res[k]["laal_s"] for k in ok3])
        ya = np.array([ab3[k]["abrupt_per_min_speech"] for k in ok3])
        print(f"\n## {d}\n")
        print("v3 curve: " + ", ".join(f"{k} ({res[k]['laal_s']:.2f} s, {res[k]['asr_bleu']:.2f})" for k in v3) + "\n")
        cols = [m for m in M if all(m in res[k] for k in v3)]
        print("| run | LAAL | " + " | ".join(f"{m} (v3@LAAL, diff)" for m in cols)
              + " | abrupt/min (v3@LAAL, ratio) | paired vs nearest v3 (BLEU, UTMOS) |")
        print("|---" * (len(cols) + 4) + "|")
        for k in sorted((k for k in res if k.startswith("br")), key=lambda k: res[k]["laal_s"]):
            x = res[k]["laal_s"]
            flag = "" if xs[0] <= x <= xs[-1] else " (extrap.)"
            cells = []
            for m in cols:
                y = float(np.interp(x, xs, np.array([res[v][m] for v in v3])))
                a = res[k].get(m)
                cells.append("-" if a is None else f"{a:.3g} ({y:.3g}, {a - y:+.2g})")
            a = abrupt(k)
            if a and len(ok3):
                y = float(np.interp(x, xa, ya))
                cells.append(f"{a['abrupt_per_min_speech']:.2f} ({y:.2f}, x{a['abrupt_per_min_speech'] / y:.2f})")
            else:
                cells.append("-")
            print(f"| {k} | {x:.2f}{flag} | " + " | ".join(cells) + f" | {pair(k)} |")
        print("\nabrupt (continuous timeline, s1b1.2q):\n")
        for k in v3 + sorted(k for k in res if k.startswith("br")):
            a = abrupt(k)
            if a:
                print(f"- {k}: " + ", ".join(f"{m}={a[m]}" for m in a if not isinstance(a[m], (dict, list))))


if __name__ == "__main__":
    main()

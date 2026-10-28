"""Dev systems against the v3 dev curve at the same LAAL (linear interpolation; outside the curve's range the nearest
segment is extrapolated and marked). Inputs: si_bayesrisk_dev_eval --timeline full json files; runs named v3* form the
curve.
    python dev_matched.py [--abrupt-key talk_abrupt_per_min] it0b_zh2en.json [more.json ...]
Prints, per system: LAAL, BLEU (delta vs v3), abrupt per minute (ratio vs v3), one-unit blocks, gaps >= 1 s."""
import json
import sys

import numpy as np


def curve(res, key):
    pts = sorted((v["laal_s"], v[key]) for k, v in res.items() if k.startswith("v3"))
    return np.array([p[0] for p in pts]), np.array([p[1] for p in pts])


def at(x, xs, ys):
    if x < xs[0] or x > xs[-1]:
        i = 0 if x < xs[0] else len(xs) - 2
        return ys[i] + (ys[i + 1] - ys[i]) * (x - xs[i]) / (xs[i + 1] - xs[i]), True
    return float(np.interp(x, xs, ys)), False


def main():
    args = sys.argv[1:]
    key = "abrupt_per_min"
    if args[:1] == ["--abrupt-key"]:
        key, args = args[1], args[2:]
    res = {}
    for f in args:
        res.update(json.load(open(f)))
    xb, yb = curve(res, "bleu")
    xa, ya = curve(res, key)
    for k, v in res.items():
        if k.startswith("v3"):
            continue
        b, ex = at(v["laal_s"], xb, yb)
        a, _ = at(v["laal_s"], xa, ya)
        print(f"{k:14s} LAAL {v['laal_s']:.2f}{'*' if ex else ' '} BLEU {v['bleu']:.2f} ({v['bleu'] - b:+.2f})  "
              f"abrupt {v[key]:.2f}/min (x{v[key] / a:.2f})  one-unit {v['one_unit_blocks']:.3f}  "
              f"gaps>=1s {v['gaps_ge1s_per_100u']:.2f}/100u")


if __name__ == "__main__":
    main()

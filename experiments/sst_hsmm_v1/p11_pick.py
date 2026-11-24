"""sst_hsmm_v1 P11 (prereg change 19): dev operating-point rule for lam0 (the controller's start multiplier), fixed in advance.
  second: dd = mean over the 2 directions of the pooled dev D2 diff (NEW - N) of run a (lam0 default).
          |dd| <= 0.05 -> no second run (prints "none"); dd < -0.05 (too early) -> control_b with lam0 / 8;
          dd > +0.05 (too late) -> lam0 * 8 (clipped to [lam_lo, lam_hi]); prints "run".
  choose: among the dev runs, the Pareto-gate passers (both directions) with the highest mean ASR diff; if none passes, the
          highest mean ASR diff among runs with dd <= +0.3; if none, the smallest dd. Copies its controller to --out and
          writes choice.json next to it.
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from pathlib import Path


def stats(summary):
    s = json.load(open(summary))
    v = s["verdict"]
    dirs = ("en2zh", "zh2en")
    return {"dd": sum(v[d]["d2_diff"] for d in dirs) / 2, "asr": sum(v[d]["asr_diff"] for d in dirs) / 2,
            "pass": bool(v["GO"]), "np_rel": sum(v[d]["np_in_rel"] for d in dirs) / 2}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("second", "choose"))
    ap.add_argument("--runs", nargs="+", required=True, help="summary.json:control.json pairs")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if Path(a.out).exists():
        sys.exit(f"拒绝覆盖已有输出: {a.out}")
    pairs = [x.split(":") for x in a.runs]
    if a.mode == "second":
        (sm, cf), = pairs
        st = stats(sm)
        if abs(st["dd"]) <= 0.05:
            print("none", json.dumps(st)); return
        c = json.load(open(cf))
        f = 1 / 8 if st["dd"] < 0 else 8.0
        c["lam0"] = min(max(c["lam0"] * f, c["lam_lo"]), c["lam_hi"])
        c["p11_second"] = {"from": cf, "factor": f, "dev_a": st}
        json.dump(c, open(a.out, "x"), indent=1)
        print("run", json.dumps(st), "lam0", c["lam0"])
        return
    rows = [(sm, cf, stats(sm)) for sm, cf in pairs]
    ps = [r for r in rows if r[2]["pass"]]
    if ps:
        best, why = max(ps, key=lambda r: r[2]["asr"]), "gate pass, highest ASR"
    elif [r for r in rows if r[2]["dd"] <= 0.3]:
        best, why = max([r for r in rows if r[2]["dd"] <= 0.3], key=lambda r: r[2]["asr"]), "no pass; highest ASR with dd <= 0.3"
    else:
        best, why = min(rows, key=lambda r: r[2]["dd"]), "no pass; smallest dd"
    shutil.copyfile(best[1], a.out)
    json.dump({"chosen": best[1], "why": why, "runs": [{"summary": r[0], "control": r[1], **r[2]} for r in rows]},
              open(Path(a.out).with_name("choice.json"), "x"), indent=1)
    print("chosen", best[1], why)


if __name__ == "__main__":
    main()

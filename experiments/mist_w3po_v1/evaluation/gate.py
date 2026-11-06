"""Stage gate (reports/mist_w3po_v1/LOG.zh-CN.md, rule fixed before any S0 / RL result): candidate B vs reference A on
AMI dev, three items -- text tcpBLEU all, text tcpBLEU overlap, speech ASR-tcpBLEU all. An item passes when
B - A >= -0.2 (point) and P(B - A <= 0) < 0.95 under the paired meeting bootstrap (10k, compare.py), i.e. B is not
significantly worse. The stage passes only if all three pass.

    python -m experiments.mist_w3po_v1.evaluation.gate <cand decode dir> --refs <v8 ctx0 dir> <v8 ctx2 dir> [--out gate.json]
Amendment 2026-11-06 22:00 UTC (before any S0 result): per gated metric the reference is the better of the refs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.mist_w3po_v1.evaluation.compare import compare

TOL = 0.2
P_MAX = 0.95


def gate(refs: list, cand: Path, n: int = 10000) -> dict:
    """refs: reference decode dirs (amendment 2026-11-06 22:00 UTC: v8 ctx0 and v8 ctx2); per gated metric the
    reference is the one with the higher point score."""
    ct = json.load(open(cand / "score_text.json"))
    cs = json.load(open(cand / "score_asr.json"))
    items, info = {}, {}
    for key, fname, reg, cj in (("text_all", "score_text.json", "all", ct), ("text_ov", "score_text.json", "ov", ct),
                                ("speech_all", "score_asr.json", "all", cs)):
        best = None
        for r in refs:
            res = compare(json.load(open(Path(r) / fname)), cj, regs=(reg,), n=n)[reg]
            res["ref"] = str(r)
            info[f"{key}@{Path(r).name}"] = res
            if best is None or res["A"] > best["A"]:
                best = res
        best = dict(best)
        best["pass"] = bool(best["diff"] >= -TOL and best["p_le0"] < P_MAX)
        items[key] = best
    return {"refs": [str(r) for r in refs], "cand": str(cand), "items": items, "per_ref": info,
            "pass": all(v["pass"] for v in items.values())}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cand")
    ap.add_argument("--refs", nargs="+", required=True)
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    r = gate([Path(x) for x in a.refs], Path(a.cand), a.n)
    print(json.dumps(r, indent=1))
    if a.out:
        with open(a.out, "x") as fh:
            json.dump(r, fh, indent=1)


if __name__ == "__main__":
    main()

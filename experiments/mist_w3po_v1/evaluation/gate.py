"""Stage gate (reports/mist_w3po_v1/LOG.zh-CN.md, rule fixed before any S0 / RL result): candidate B vs reference A on
AMI dev, three items -- text tcpBLEU all, text tcpBLEU overlap, speech ASR-tcpBLEU all. An item passes when
B - A >= -0.2 (point) and P(B - A <= 0) < 0.95 under the paired meeting bootstrap (10k, compare.py), i.e. B is not
significantly worse. The stage passes only if all three pass.

    python -m experiments.mist_w3po_v1.evaluation.gate <ref decode dir> <cand decode dir> [--out gate.json]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.mist_w3po_v1.evaluation.compare import compare

TOL = 0.2
P_MAX = 0.95


def gate(ref: Path, cand: Path, n: int = 10000) -> dict:
    t = compare(json.load(open(ref / "score_text.json")), json.load(open(cand / "score_text.json")),
                regs=("all", "ov", "1spk"), n=n)
    s = compare(json.load(open(ref / "score_asr.json")), json.load(open(cand / "score_asr.json")),
                regs=("all", "ov", "1spk"), n=n)
    items = {"text_all": t["all"], "text_ov": t["ov"], "speech_all": s["all"]}
    for v in items.values():
        v["pass"] = bool(v["diff"] >= -TOL and v["p_le0"] < P_MAX)
    return {"ref": str(ref), "cand": str(cand), "items": items, "info": {"text_1spk": t["1spk"], "speech_ov": s["ov"],
            "speech_1spk": s["1spk"]}, "pass": all(v["pass"] for v in items.values())}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("ref")
    ap.add_argument("cand")
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    r = gate(Path(a.ref), Path(a.cand), a.n)
    print(json.dumps(r, indent=1))
    if a.out:
        with open(a.out, "x") as fh:
            json.dump(r, fh, indent=1)


if __name__ == "__main__":
    main()

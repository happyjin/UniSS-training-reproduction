"""Paired UTMOS difference of a candidate talker output vs references, scripts/pmc_listen_compare.py's method unchanged
(per-stream items of talker/talker_quality.py, paired on common ids within each TTS seed, the two seeds pooled, 2000
meeting-level bootstrap resamples, seed 20261112) except the meeting key: id.rsplit("_on_", 1)[0] instead of the prefix
before the first "_" (AliMeeting ids "R8002_M8002_on_0000__S0" would otherwise be grouped by room; for AMI ids both keys
are the meeting). Final test (reports/mist_w3po_v1/final_test_prereg_v1.zh-CN.md). Prints JSON.

    python -m experiments.mist_w3po_v1.evaluation.listen_paired_meeting --cand <q dir s1> <q dir s2> \\
        --ref <name> <q dir s1> <q dir s2> [--ref ...]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from pmc_listen_compare import items, summary  # noqa: E402


def meeting(k: str) -> str:
    return k.split("__S")[0].rsplit("_on_", 1)[0]


def paired(cand: list[str], ref: list[str], n_boot: int = 2000, seed: int = 20261112) -> dict:
    by = defaultdict(list)
    n = 0
    for c, r in zip(cand, ref):
        a, b = items(c), items(r)
        for k in sorted(set(a) & set(b)):
            by[meeting(k)].append(a[k] - b[k])
            n += 1
    ms = sorted(by)
    sums = np.array([sum(by[m]) for m in ms])
    cnts = np.array([len(by[m]) for m in ms])
    diff = float(sums.sum() / cnts.sum())
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(n_boot):
        ix = rng.integers(0, len(ms), len(ms))
        bs.append(sums[ix].sum() / cnts[ix].sum())
    bs = np.asarray(bs)
    return {"pairs": n, "meetings": len(ms), "diff": round(diff, 4),
            "ci95": [round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)],
            "p_le0": round(float((bs <= 0).mean()), 4)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cand", nargs=2, required=True)
    ap.add_argument("--ref", nargs=3, action="append", required=True, metavar=("NAME", "S1", "S2"))
    a = ap.parse_args()
    out = {"cand": {"dirs": a.cand, "seeds": [summary(d) for d in a.cand]}, "vs": {}, "meeting_key": "rsplit _on_"}
    for name, s1, s2 in a.ref:
        out["vs"][name] = {"ref_seeds": [summary(s1), summary(s2)], "utmos_paired": paired(a.cand, [s1, s2])}
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()

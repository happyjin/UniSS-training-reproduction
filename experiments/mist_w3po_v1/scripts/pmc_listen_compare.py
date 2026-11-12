"""Paired UTMOS difference of a candidate talker output vs references (PMC prereg §3.5; the method of
aprime_rl_results_v1's listening section): per-stream items of talker/talker_quality.py (<quality dir>/quality.json.items.jsonl,
id = <window>__S<k>), paired on common ids within each TTS seed (seed 1 with seed 1, seed 2 with seed 2), the two seeds
pooled, meeting-level bootstrap (meeting = id prefix before the first "_"; 2000 resamples, seed 20261112).
Also the SIM means. Prints JSON.
    python pmc_listen_compare.py --cand <q dir s1> <q dir s2> --ref <name> <q dir s1> <q dir s2> [--ref ...]"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def items(d: str) -> dict:
    return {json.loads(l)["id"]: json.loads(l)["utmos"] for l in open(Path(d) / "quality.json.items.jsonl")}


def summary(d: str) -> dict:
    q = json.loads((Path(d) / "quality.json").read_text())
    return {"utmos_mean": q.get("utmos_mean"), "sim_mean": q.get("sim_mean"), "n_utmos": q.get("n_utmos")}


def paired(cand: list[str], ref: list[str], n_boot: int = 2000, seed: int = 20261112) -> dict:
    by = defaultdict(list)   # meeting -> list of diffs (both seeds)
    n = 0
    for c, r in zip(cand, ref):
        a, b = items(c), items(r)
        for k in sorted(set(a) & set(b)):
            by[k.split("_")[0]].append(a[k] - b[k])
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
    out = {"cand": {"dirs": a.cand, "seeds": [summary(d) for d in a.cand]}, "vs": {}}
    for name, s1, s2 in a.ref:
        out["vs"][name] = {"ref_seeds": [summary(s1), summary(s2)], "utmos_paired": paired(a.cand, [s1, s2])}
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()

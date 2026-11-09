"""Falsification check for the finalpost dev test (CPU): the flag changes only FINAL reads, so a window whose NON-final
read differs from the reference must have first diverged at a batch step that contained some other window's final
read (bf16 MoE batch coupling), never at a step without one. online_decode's schedule is deterministic for --context 0
(one queue per shard: meetings sorted, windows by start; up to --batch active; each step reads every active window at
t = min(r * 1 s, dur); a window leaves after its final read), so the steps are re-simulated from the windows file and
checked against the read logs (number of reads). Prints and writes a json (new file).

    python -m experiments.mist_w3po_v1.evaluation.finalpost_batchcheck --ref <dir> --cand <dir> --out <json>
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

WIN = "eval_outputs/mist_w3po_v1/data/windows_zh2en_v1/windows_dev_online.jsonl"


def schedule(W: list[dict], shards: int = 8, batch: int = 16, period: int = 1000):
    """-> {window id: [(shard, step) per read]}, {(shard, step): [(window id, is_final)]}"""
    ms = sorted({w["meeting"] for w in W})
    steps, members = {}, defaultdict(list)
    for k in range(shards):
        mine = set(ms[k::shards])
        by_m = defaultdict(list)
        for w in W:
            if w["meeting"] in mine:
                by_m[w["meeting"]].append(w)
        flat = [w for m in sorted(by_m) for w in sorted(by_m[m], key=lambda w: w["start_s"])]
        active, step = [], 0
        while flat or active:
            while flat and len(active) < batch:
                active.append({"w": flat.pop(0), "r": 0})
            fin = []
            for s in active:
                s["r"] += 1
                t = min(s["r"] * period, s["w"]["dur_ms"])
                final = t >= s["w"]["dur_ms"]
                steps.setdefault(s["w"]["id"], []).append((k, step))
                members[(k, step)].append((s["w"]["id"], final))
                if final:
                    fin.append(s)
            for s in fin:
                active.remove(s)
            step += 1
    return steps, members


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--cand", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    W = [json.loads(l) for l in open(WIN, encoding="utf-8")]
    R = {r["id"]: r for r in map(json.loads, open(Path(a.ref) / "hyp.jsonl", encoding="utf-8"))}
    C = {r["id"]: r for r in map(json.loads, open(Path(a.cand) / "hyp.jsonl", encoding="utf-8"))}
    steps, members = schedule(W)
    n_reads_ok = sum(len(steps[i]) == len(R[i]["read_log"]) for i in R)
    div = {}   # window -> index of its first differing NON-final read
    for i in R:
        lr, lc = R[i]["read_log"], C[i]["read_log"]
        d = next((j for j, (x, y) in enumerate(zip(lr[:-1], lc[:-1])) if x["new"] != y["new"]), None)
        if d is not None:
            div[i] = d
    no_final_mate, unexplained = [], []
    for i, d in div.items():
        st = steps[i][d]
        mates = [(m, f) for m, f in members[st] if m != i]
        if not any(f for _, f in mates):
            no_final_mate.append(i)
        # a mate's input differs at this step: it is at its final read (P), or it diverged at an earlier read
        diff_in = [m for m, f in mates if f or (m in div and div[m] < steps[m].index(st))]
        if not diff_in:
            unexplained.append(i)
    res = {"windows": len(R), "read_count_matches_schedule": n_reads_ok, "perturbed_windows": len(div),
           "first_divergence_without_final_batchmate": len(no_final_mate),
           "first_divergence_without_any_differing_batchmate_input": len(unexplained), "unexplained": unexplained[:10]}
    print(json.dumps(res))
    with open(out, "x", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main()

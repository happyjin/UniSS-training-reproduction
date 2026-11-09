"""CPU pre-check for --mode ecf (event-triggered counterfactual flush branches), before launch. Same groups as
precheck_eagrpo (existing AMI dev decodes of the same online windows by 8 systems; a diagnostic, nothing is trained or
tuned on it). For every short reference unit (<= 1 s):
  (a) how many of the 8 members cover it (k): k = 0 means no in-group contrast at all (EA-GRPO's event term is zero
      there; only a counterfactual branch can produce signal), 0 < k < 8 means in-group contrast exists;
  (b) per missed (unit, member) pair: is it branchable (a non-final read at or after the unit's end whose block ends
      with <wait>, i.e. the member chose to keep waiting after the unit was fully heard);
  (c) for units with 0 < k < 8: mean z(cpBLEU) of the members covering it minus that of the members missing it (the
      correlational proxy of "emitting it helps the window reward"), and the share of such units where it is > 0;
  (d) branches per window under the launch cap (at most --cap branches, distinct members).

    python -m experiments.mist_w3po_v1.rl.precheck_ecf --out reports/mist_w3po_v1/precheck_ecf.json [--n 3000]
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from experiments.mist_w3po_v1.rl.precheck_eagrpo import D, SYSTEMS

SHORT_S = 1.0


def one_group(job):
    from experiments.mist_w3po_v1.core.engine import WAIT_ID
    from experiments.mist_w3po_v1.rl import advantage as A
    from experiments.mist_w3po_v1.rl import stamp as S
    w, utts, trajs, cap = job
    units = S.units_of(utts)
    dur = w["dur_ms"] / 1000
    post = np.load(w["post"]["lowlat"]).astype(np.float32)
    creds = [S.credit(t.segs, units, dur, t.wait_reads, [rd.t_ms / 1000 for rd in t.reads], post=post) for t in trajs]
    zq = A.z(np.array([S.seq_cpbleu(t.segs, units) for t in trajs]))
    out = {"k": [], "missed": 0, "branchable": 0, "gap": [], "members_branchable": 0}
    mem_ok = set()
    for i, u in enumerate(units):
        if u["b"] - u["a"] > SHORT_S:
            continue
        cov = [any(x == i for x in c.unit_of) for c in creds]
        k = sum(cov)
        out["k"].append(k)
        if 0 < k < len(trajs):
            out["gap"].append(float(np.mean([zq[g] for g in range(len(trajs)) if cov[g]])
                                    - np.mean([zq[g] for g in range(len(trajs)) if not cov[g]])))
        for g, tr in enumerate(trajs):
            if cov[g]:
                continue
            out["missed"] += 1
            r = next((r for r, rd in enumerate(tr.reads) if rd.t_ms / 1000 >= u["b"]), None)
            if r is not None and not tr.reads[r].final and tr.reads[r].ids and tr.reads[r].ids[-1] == WAIT_ID:
                out["branchable"] += 1
                mem_ok.add(g)
    out["members_branchable"] = min(cap, len(mem_ok))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--cap", type=int, default=4)
    ap.add_argument("--workers", type=int, default=24)
    a = ap.parse_args()
    if Path(a.out).exists():
        raise SystemExit(f"refusing to overwrite {a.out}")
    from experiments.mist_w3po_v1.core.engine import Prompter
    from experiments.mist_w3po_v1.evaluation.diag_reward_noise import WIN, ref_utts, window_utts
    from experiments.mist_w3po_v1.rl.test_pipeline_cpu import traj_from
    tok = Prompter("zh2en").tok
    W = [json.loads(l) for l in open(WIN, encoding="utf-8")]
    by = ref_utts()
    H = [{r["id"]: r for r in map(json.loads, open(f"{D}/{s}/hyp.jsonl", encoding="utf-8"))} for s in SYSTEMS]
    cand = [w for w in W if window_utts(w, by) and all(h.get(w["id"], {}).get("read_log") for h in H)]
    pick = [cand[round(i * (len(cand) - 1) / (a.n - 1))] for i in range(min(a.n, len(cand)))] if a.n > 1 else cand
    jobs = [(w, window_utts(w, by), [traj_from(h[w["id"]], tok, w) for h in H], a.cap) for w in pick]
    with ProcessPoolExecutor(a.workers) as ex:
        res = list(ex.map(one_group, jobs, chunksize=4))
    k = np.array([x for r in res for x in r["k"]])
    G = len(SYSTEMS)
    gap = np.array([x for r in res for x in r["gap"]])
    missed = sum(r["missed"] for r in res)
    rep = {"systems": SYSTEMS, "groups": len(res), "short_units": int(len(k)),
           "share_k0": float(np.mean(k == 0)), "share_k_mid": float(np.mean((k > 0) & (k < G))),
           "share_kG": float(np.mean(k == G)), "missed_pairs": int(missed),
           "share_missed_in_k0_units": float(np.sum(k == 0) * G / max(missed, 1)),
           "branchable_share_of_missed": float(sum(r["branchable"] for r in res) / max(missed, 1)),
           "gap_cover_minus_miss_mean": float(gap.mean()) if len(gap) else None,
           "gap_positive_share": float(np.mean(gap > 0)) if len(gap) else None,
           "branches_per_window_cap": float(np.mean([r["members_branchable"] for r in res])), "cap": a.cap}
    Path(a.out).write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()

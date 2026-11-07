"""Pick the RL micro-batch token budget from step benches (same step resumed from the same checkpoint with different
--tok-budget; rollouts are seeded per step and rank, so they are identical and only the update differs), then write
the next RL job file with that budget. Waits until every bench job has an exit code in <qdir>/state.

Equivalence (the token budget only regroups the same per-token losses into micro-batches; gradients are summed
before the optimiser step): R, groups, dropped must be equal; kl, ratio_dev, clip_frac, gnorm within --rtol of the
reference budget's values (bf16 accumulation order). Among equivalent successful benches the one with the lowest
t_step wins. Also prints mean power / SM util per card over each bench step from an nvidia-smi CSV.

    python pick_bench.py --qdir Q --bench 12000:rl_bench_s14_tb12000:<run dir> ... --ref 12000 \
        --template <job json with @TB@> --out-job <qdir>/jobs/<name>.json --smi <csv> --report <json>
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import time
from pathlib import Path


def smi_stats(csv: str, t0: float, t1: float) -> dict:
    p, u, n = 0.0, 0.0, 0
    for l in open(csv, encoding="utf-8", errors="ignore"):
        x = [s.strip() for s in l.split(",")]
        if len(x) < 5:
            continue
        try:
            ts = dt.datetime.strptime(x[0], "%Y/%m/%d %H:%M:%S.%f").replace(tzinfo=dt.timezone.utc).timestamp()
            if t0 <= ts <= t1:
                p += float(x[2])
                u += float(x[3])
                n += 1
        except ValueError:
            continue
    return {"power_w": p / n if n else None, "sm_util": u / n if n else None, "samples": n}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--qdir", required=True)
    ap.add_argument("--bench", nargs="+", required=True, help="tb:job_id:run_dir")
    ap.add_argument("--ref", type=int, required=True)
    ap.add_argument("--step", type=int, default=14)
    ap.add_argument("--rtol", type=float, default=0.05)
    ap.add_argument("--template", required=True)
    ap.add_argument("--out-job", required=True)
    ap.add_argument("--smi", default="")
    ap.add_argument("--report", required=True)
    a = ap.parse_args()
    q = Path(a.qdir)
    benches = []
    for b in a.bench:
        tb, jid, run = b.split(":")
        benches.append((int(tb), jid, Path(run)))
    while not all((q / "state" / f"{jid}.rc").exists() or (q / "state" / f"{jid}.skipped").exists()
                  for _, jid, _ in benches):
        time.sleep(20)
    res = {}
    for tb, jid, run in benches:
        rc = (q / "state" / f"{jid}.rc").read_text().strip() if (q / "state" / f"{jid}.rc").exists() else "skipped"
        m = None
        if (run / "metrics.jsonl").exists():
            for l in open(run / "metrics.jsonl", encoding="utf-8"):
                r = json.loads(l)
                if r["step"] == a.step:
                    m = r
        rec = {"rc": rc, "metrics": m}
        if m:
            t1 = (run / "metrics.jsonl").stat().st_mtime
            rec["t_end"] = t1
            if a.smi:
                rec["smi_step"] = smi_stats(a.smi, t1 - m["t_step"], t1)
                rec["smi_update"] = smi_stats(a.smi, t1 - m["t_update"], t1)
                rec["smi_rollout"] = smi_stats(a.smi, t1 - m["t_step"], t1 - m["t_step"] + m["t_rollout"])
        res[tb] = rec
    ref = res.get(a.ref, {}).get("metrics")
    ok = []
    for tb, rec in res.items():
        m = rec["metrics"]
        if rec["rc"] != "0" or not m:
            rec["equiv"] = False
            continue
        if ref is None:
            rec["equiv"] = tb == a.ref
        else:
            same = all(m[k] == ref[k] for k in ("R", "groups", "dropped"))
            close = all(abs(m[k] - ref[k]) <= a.rtol * max(abs(ref[k]), 1e-6) for k in ("kl", "ratio_dev", "clip_frac", "gnorm"))
            rec["equiv"] = bool(same and close)
            rec["rel_diff"] = {k: (m[k] - ref[k]) / max(abs(ref[k]), 1e-12) for k in ("kl", "ratio_dev", "clip_frac", "gnorm")}
        if rec["equiv"]:
            ok.append((m["t_step"], tb))
    pick = min(ok)[1] if ok else a.ref
    out = {"pick": pick, "benches": res}
    Path(a.report).write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps(out, indent=1, default=str), flush=True)
    job = Path(a.template).read_text().replace("@TB@", str(pick))
    json.loads(job)
    tmp = Path(a.out_job + ".tmp")
    tmp.write_text(job)
    os.replace(tmp, a.out_job)
    print(f"PICK {pick}; wrote {a.out_job}", flush=True)


if __name__ == "__main__":
    main()

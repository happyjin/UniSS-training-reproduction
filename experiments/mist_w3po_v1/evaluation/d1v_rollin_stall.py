"""D1v roll-in: stall classes on the training plan windows with the learner's rho~ (thinker_dagger_d1v_prereg_v1 §2.3
descriptive item "训练窗口上的停顿分类（用 ρ̃）"; D1 prereg §4.2). CPU only, read-only.

Per plan window: e = the label's due read per item (precheck_serial.chunk_reads, as in d1_rollin.run_manifest), rho~ =
the learner's write read per item (roll-in rho.jsonl), final read = n_read - 1. The classes are those of
t1_stall_diag.py with read indices in place of times:
  A  something due before the final read, nothing written before it            ("never starts")
  B  wrote before the final read, then silent while more items became due      ("stalls after progress")
  C  keeps writing until its last due non-final item                           ("keeps up")
Windows with no item due before the final read are not classed (as in t1_stall_diag).
    python -m experiments.mist_w3po_v1.evaluation.d1v_rollin_stall --sched <sched dir> --root <roll-in root> --out <new json>
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from experiments.mist_w3po_v1.crf import d1_rollin as R
from experiments.mist_w3po_v1.crf import precheck_serial as S


def classify(e: list[int], rho: list[int], n_read: int) -> dict | None:
    final = n_read - 1
    due_nf = sorted(t for t in e if t < final)
    if not due_nf:
        return None
    early = sorted(r for r in rho if r < final)
    if not early:
        return {"cls": "A", "silent": sum(1 for r in range(due_nf[0], final))}
    last = early[-1]
    later = [t for t in due_nf if t > last]
    return {"cls": "B" if later else "C", "silent": (final - last - 1) if later else 0}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sched", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if Path(a.out).exists():
        raise SystemExit(f"拒绝覆盖已有输出: {a.out}")
    recs, plan = R.load_plan(a.sched)
    rho = {}
    for f in sorted(Path(a.root).glob("rollin_*of*/rho.jsonl")):
        for l in open(f, encoding="utf-8"):
            x = json.loads(l)
            rho[x["id"]] = x["rho"]
    used = sorted({i for s in plan for i in s})
    res = []
    for i in used:
        rec = recs[i]
        c = classify(S.chunk_reads(rec), rho[rec["id"]], rec["n_read"])
        if c:
            c["dur_s"] = rec["dur_ms"] / 1000
            res.append(c)
    n = len(res)
    cnt = Counter(r["cls"] for r in res)
    out = {"sched": a.sched, "root": a.root, "plan_windows": len(used), "classed": n,
           "share": {k: round(cnt[k] / n, 4) for k in "ABC"}, "count": {k: cnt[k] for k in "ABC"},
           "B_mean_silent_reads": round(sum(r["silent"] for r in res if r["cls"] == "B") / max(1, cnt["B"]), 3)}
    by = defaultdict(Counter)
    for r in res:
        by["<3s" if r["dur_s"] < 3 else ("3-6s" if r["dur_s"] < 6 else ">=6s")][r["cls"]] += 1
    out["by_dur"] = {b: dict(v) for b, v in sorted(by.items())}
    g6 = by[">=6s"]
    out["A_share_ge6s"] = round(g6["A"] / max(1, sum(g6.values())), 4)
    with open(a.out, "x", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, ensure_ascii=False)
    print(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()

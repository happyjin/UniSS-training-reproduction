"""Audit of the prefix-constrained start oracle (build_start_oracle.py) before it becomes training data.

Per direction: start reads, share whose target gets longer (best_L > lcp), units added, how often the old target was
empty and the new one is not, sentence chrF of the constrained whole translation vs the free oracle (both against the
human reference), length ratio, and checks that the new target is a prefix of the draft and that the constrained whole
starts with it. Writes <in>.audit.json and prints random samples (lengthened and kept) for reading.

    python -m experiments.uniss_qwen3_omni_v1.data_clean.audit_start_oracle <start_oracle.jsonl> [--samples 20]
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("path")
    ap.add_argument("--samples", type=int, default=20)
    args = ap.parse_args()
    src = Path(args.path)
    out = src.with_suffix(".audit.json")
    if out.exists():
        raise SystemExit(f"refusing to overwrite {out}")
    rows = [json.loads(l) for l in open(src, encoding="utf-8")]
    summ = {}
    for d in ("en2zh", "zh2en"):
        R = [r for r in rows if r["dir"] == d]
        if not R:
            continue
        zh = d == "en2zh"
        U = lambda t: [c for c in t if not c.isspace()] if zh else t.split()
        up = [r for r in R if r["best_L"] > r["lcp"]]
        prefix_ok = sum(1 for r in up if U(r["draft"])[: r["best_L"]] == U(r["target"]))
        full_ok = sum(1 for r in up if r["full"] and U(r["full"])[: r["best_L"]] == U(r["target"]))
        diff = np.array([r["chrf_full"] - r["chrf_oracle"] for r in up]) if up else np.zeros(1)
        summ[d] = {
            "start_reads": len(R), "lengthened": len(up), "lengthened_share": round(len(up) / len(R), 3),
            "empty_to_nonempty": sum(1 for r in up if r["lcp"] == 0), "empty_before": sum(1 for r in R if r["lcp"] == 0),
            "units_added_mean": round(float(np.mean([r["best_L"] - r["lcp"] for r in up])), 2) if up else 0,
            "target_is_draft_prefix": round(prefix_ok / max(1, len(up)), 4),
            "constrained_full_starts_with_target": round(full_ok / max(1, len(up)), 4),
            "chrf_full_minus_oracle_p10_p50_p90": [round(float(x), 2) for x in np.percentile(diff, [10, 50, 90])],
            "chrf_full_mean": round(float(np.mean([r["chrf_full"] for r in up])), 2) if up else None,
            "chrf_oracle_mean_on_these": round(float(np.mean([r["chrf_oracle"] for r in up])), 2) if up else None,
        }
    out.write_text(json.dumps(summ, ensure_ascii=False, indent=1))
    print(json.dumps(summ, ensure_ascii=False, indent=1))
    rng = random.Random(0)
    for d in ("en2zh", "zh2en"):
        R = [r for r in rows if r["dir"] == d]
        up = [r for r in R if r["best_L"] > r["lcp"]]
        keep = [r for r in R if r["best_L"] <= r["lcp"]]
        for tag, pool in (("NEW", up), ("KEEP", keep)):
            for r in rng.sample(pool, min(args.samples // 2, len(pool))):
                print(f"\n[{d} {tag}] t={r['t']} lcp={r['lcp']} -> L={r['best_L']}  chrF {r['chrf_oracle']} -> {r['chrf_full']}"
                      f"\n  draft : {r['draft']}\n  old tgt: {r['draft'][:0] if r['lcp'] == 0 else '(first ' + str(r['lcp']) + ' units)'}"
                      f"\n  new tgt: {r['target']}\n  oracle: {r['oracle']}\n  full  : {r['full']}\n  ref   : {r['ref']}")


if __name__ == "__main__":
    main()

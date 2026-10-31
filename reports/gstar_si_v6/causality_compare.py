"""Paper v6, review M02: comparison step of the causality test, redone on the saved prefix posteriors.

causality_test.py compared the prefix run's raw output (10 ms frames) with the whole-session dumps, which hold 80 ms
frames (10 ms posteriors mean-pooled over 8 frames, as dump_nemotron_posteriors.py does): a frame-rate mismatch, so its
per-cut numbers in cuts.jsonl are void. Here every saved prefix output (<dir>/post/<set>_<meeting>_<t_ms>.pt) is pooled
exactly as the dump script pools (pool = round(frames / duration * 0.08)), then compared on the frames a reader at t may
see (<= t - 1.04 s) with
  dump      the whole-session low-latency dump the streaming runs read
  prefix    every longer prefix run of the same meeting (self-consistency: does later audio change earlier frames?)
Writes <dir>/cuts_v2.jsonl and prints a summary.

    python reports/gstar_si_v6/causality_compare.py --dir eval_outputs/uniss_qwen3_omni_v1/gstar_v6_20261031/causality
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from causality_test import LOOKAHEAD_S, SETS  # noqa: E402


def pool80(pr, dur):
    pool = int(round(pr.shape[0] / max(dur, 1e-6) * 0.08))
    t80 = pr.shape[0] // max(pool, 1)
    return pr[: t80 * pool].reshape(t80, pool, -1).mean(1) if pool > 1 else pr


def cmp(a, b, n):
    a16 = a[:n].astype(np.float16).astype(np.float32)
    b = b[:n]
    d = np.abs(a16 - b)
    live = (b.max(1) > 0.5) | (a16.max(1) > 0.5)
    return {"frames": int(n), "max_abs": float(d.max()) if n else 0.0, "mean_abs": float(d.mean()) if n else 0.0,
            "frac_decision_diff": float(((a16 > 0.5) != (b > 0.5)).any(1).mean()) if n else 0.0,
            "argmax_agree_active": float((a16.argmax(1) == b.argmax(1))[live].mean()) if live.any() else 1.0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    a = ap.parse_args()
    d = Path(a.dir)
    out = d / "cuts_v2.jsonl"
    if out.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    runs = defaultdict(list)
    for f in sorted((d / "post").glob("*.pt")):
        st, rest = f.stem.split("_", 1)
        m, t = rest.rsplit("_", 1)
        runs[(st, m)].append((int(t) / 1000, f))
    rows = []
    for (st, m), lst in sorted(runs.items()):
        dump = torch.load(SETS[st][2](m), weights_only=False)["post80"].float().numpy()
        pooled = []
        for t, f in sorted(lst):
            pre = torch.load(f)["post80"].float().numpy()
            pooled.append((t, pool80(pre, t)))
        for i, (t, p) in enumerate(pooled):
            n = max(0, min(int((t - LOOKAHEAD_S) / 0.08), len(p), len(dump)))
            r = {"set": st, "meeting": m, "t_s": t, "vs_dump": cmp(p, dump, n)}
            later = [q for t2, q in pooled[i + 1:]]
            if later:
                r["vs_longer_prefix"] = cmp(p, later[0], min(n, len(later[0])))
            rows.append(r)
    with open(out, "x") as fh:
        fh.writelines(json.dumps(r) + "\n" for r in rows)
    for key in ("vs_dump", "vs_longer_prefix"):
        xs = [r[key] for r in rows if key in r]
        print(key, len(xs), "max_abs", max(x["max_abs"] for x in xs), "mean dec diff",
              round(float(np.mean([x["frac_decision_diff"] for x in xs])), 6),
              "min argmax", round(min(x["argmax_agree_active"] for x in xs), 4), "frames", sum(x["frames"] for x in xs))


if __name__ == "__main__":
    main()

"""Paired comparison of two commit logs: per-utterance mean errors, utterance-level bootstrap 95% CI of B - A.

  python compare.py A_run B_run --align-dir ... [--n-boot 5000]
Negative deltas mean B has lower error than A (better). Lookahead delta is reported too.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "vits"))
sys.path.insert(0, str(ROOT / "scripts"))
from eval_metrics import load_runs, plan_durations, word_and_pauses  # noqa: E402


def per_utt(runs, rows, gt):
  out = {}
  for uid, rec in runs.items():
    pw, pp = word_and_pauses(rows[uid], plan_durations(rec))
    tw, tq = word_and_pauses(rows[uid], gt[uid])
    ew = np.abs(np.log(np.maximum(pw, 1)) - np.log(np.maximum(tw, 1)))
    ep = np.abs(np.log1p(pp) - np.log1p(tq))
    look = np.array([u["j"] - u["i"] for u in rec["units"]], dtype=float)
    out[uid] = (ew.sum(), len(ew), ep.sum(), len(ep), look.sum(), len(look))
  return out


def boot(a, b, keys, idx_num, idx_den, n_boot, rng):
  A = np.array([[a[k][idx_num], a[k][idx_den]] for k in keys])
  B = np.array([[b[k][idx_num], b[k][idx_den]] for k in keys])
  point = B[:, 0].sum() / B[:, 1].sum() - A[:, 0].sum() / A[:, 1].sum()
  sel = rng.integers(0, len(keys), size=(n_boot, len(keys)))
  d = B[sel, 0].sum(1) / B[sel, 1].sum(1) - A[sel, 0].sum(1) / A[sel, 1].sum(1)
  return point, np.percentile(d, 2.5), np.percentile(d, 97.5), float((d >= 0).mean())


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("a")
  ap.add_argument("b")
  ap.add_argument("--align-dir", required=True)
  ap.add_argument("--manifest", default=str(ROOT / "data/manifest_v2.jsonl"))
  ap.add_argument("--n-boot", type=int, default=5000)
  args = ap.parse_args()
  rows = {r["utt_id"]: r for r in map(json.loads, open(args.manifest))}
  gt = dict(np.load(Path(args.align_dir) / "durations.npz"))
  a, b = per_utt(load_runs(args.a), rows, gt), per_utt(load_runs(args.b), rows, gt)
  keys = sorted(set(a) & set(b))
  rng = np.random.default_rng(0)
  res = {"a": Path(args.a).name, "b": Path(args.b).name, "n_utts": len(keys)}
  for name, i, j in [("word_mae", 0, 1), ("pause_mae", 2, 3), ("look", 4, 5)]:
    p, lo, hi, p_ge0 = boot(a, b, keys, i, j, args.n_boot, rng)
    res[name] = {"delta": round(p, 4), "ci95": [round(lo, 4), round(hi, 4)], "boot_p(delta>=0)": round(p_ge0, 4)}
  print(json.dumps(res))


if __name__ == "__main__":
  main()

"""Paired utterance-bootstrap of audio metrics between rendered systems (per_utt_audio.jsonl from eval_audio.py).

  python compare_audio.py A_run B_run [B_run ...]
WER is a ratio of sums (per-utterance WER x reference word count, reference words from the manifest text, which
approximates the normalised count); UTMOS and word log-F0 error are utterance means. Delta = B - A.
"""
import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def load(run):
  return {d["utt_id"]: d for d in map(json.loads, open(Path(run) / "per_utt_audio.jsonl"))}


def paired(a_run, b_run, n_boot=5000):
  """Delta B - A with utterance-bootstrap 95% CI for wer (ratio of sums), utmos and f0_err."""
  nref = {r["utt_id"]: len(r["clean"].split()) for r in map(json.loads, open(ROOT / "data/manifest_v2.jsonl"))}
  A, B = load(a_run), load(b_run)
  keys = sorted(set(A) & set(B))
  n = np.array([nref[k] for k in keys], dtype=float)
  rng = np.random.default_rng(0)
  sel = rng.integers(0, len(keys), size=(n_boot, len(keys)))
  res = {"a": str(a_run), "b": str(b_run), "n_utts": len(keys)}
  for m in ["wer", "utmos", "f0_err"]:
    xa = np.array([np.nan if A[k][m] is None else A[k][m] for k in keys])
    xb = np.array([np.nan if B[k][m] is None else B[k][m] for k in keys])
    ok = ~(np.isnan(xa) | np.isnan(xb))
    w = (n if m == "wer" else np.ones_like(n)) * ok
    xa, xb = np.nan_to_num(xa), np.nan_to_num(xb)
    point = (w * xb).sum() / w.sum() - (w * xa).sum() / w.sum()
    d = (w[sel] * xb[sel]).sum(1) / w[sel].sum(1) - (w[sel] * xa[sel]).sum(1) / w[sel].sum(1)
    res[m] = {"delta": round(float(point), 4), "ci95": [round(float(np.percentile(d, 2.5)), 4),
                                                       round(float(np.percentile(d, 97.5)), 4)]}
  return res


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("a")
  ap.add_argument("b", nargs="+")
  ap.add_argument("--n-boot", type=int, default=5000)
  args = ap.parse_args()
  for b in args.b:
    print(json.dumps(paired(args.a, b, args.n_boot)))


if __name__ == "__main__":
  main()

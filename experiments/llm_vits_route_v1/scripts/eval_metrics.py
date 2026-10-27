"""Prosody/latency metrics of commit logs against the real recordings (and optionally the E0 offline plan).

Per unit: Y = (log word frames, log1p pause frames). Reports MAE per coordinate, pause-boundary F1 (pause >= 3
frames, about 35 ms), mean lookahead in units (end-of-stream units count their actual lookahead), wait rate, and
T3 certification / violation statistics when present.
"""
import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "vits"))
from stream_common import prosody_y  # noqa: E402

PAUSE_FR = 3


def load_runs(path):
  files = sorted(glob.glob(str(Path(path) / "commits*.jsonl")))
  return {d["utt_id"]: d for f in files for d in map(json.loads, open(f))}


def metrics(runs, gt, ref=None, eps=None, w=(0.5, 0.5), ystd=None):
  err, ref_err, look, tp, fp, fn = [], [], [], 0, 0, 0
  traj_viol, cert, n_units, compute = 0, 0, 0, []
  for uid, rec in runs.items():
    ys = gt[uid]
    viol = False
    for u in rec["units"]:
      y, t = np.array(u["y"]), ys[u["i"]]
      err.append(np.abs(y - t))
      if ref is not None:
        ref_err.append(np.abs(y - np.array(ref[uid]["units"][u["i"]]["y"])))
      look.append(u["j"] - u["i"])
      p_hat, p_true = np.expm1(y[1]) >= PAUSE_FR, np.expm1(t[1]) >= PAUSE_FR
      tp += p_hat and p_true
      fp += p_hat and not p_true
      fn += p_true and not p_hat
      n_units += 1
      if u.get("certified"):
        cert += 1
        if eps is not None and float((np.array(w) * np.abs(y - t) / ystd).sum()) > eps:
          viol = True
    traj_viol += viol
    if "compute_s" in rec:
      compute.append(rec["compute_s"] / len(rec["units"]))
  err = np.stack(err)
  out = {"n_utts": len(runs), "n_units": n_units, "mae_word": float(err[:, 0].mean()), "mae_pause": float(err[:, 1].mean()),
         "mae_mean": float(err.mean()), "pause_f1": 2 * tp / max(1, 2 * tp + fp + fn), "look_mean": float(np.mean(look)),
         "wait_rate": float(np.mean(np.array(look) > 0))}
  if ref_err:
    r = np.stack(ref_err)
    out.update({"ref_mae_word": float(r[:, 0].mean()), "ref_mae_pause": float(r[:, 1].mean())})
  if cert or eps is not None:
    out.update({"certified_frac": cert / n_units, "traj_violation_rate": traj_viol / len(runs)})
  if compute:
    out["compute_ms_per_unit"] = 1000 * float(np.mean(compute))
  return out


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("runs", nargs="+")
  ap.add_argument("--align-dir", required=True)
  ap.add_argument("--ref", default=None, help="E0 offline commits dir")
  ap.add_argument("--risk", default=None, help="T3 risk head (for standardised violation checks)")
  ap.add_argument("--out", default=None)
  args = ap.parse_args()
  gt = {d["utt_id"]: prosody_y(d["units"]) for d in map(json.loads, open(Path(args.align_dir) / "unit_prosody.jsonl"))}
  ref = load_runs(args.ref) if args.ref else None
  ystd = None
  if args.risk:
    import torch
    ystd = np.array(torch.load(args.risk, map_location="cpu")["meta"]["y_std"])
  table = {}
  for p in args.runs:
    eps = None
    name = Path(p).name
    if ":" in name or name.startswith("t3_"):
      try:
        eps = float(name.split("_")[-1])
      except ValueError:
        eps = None
    table[name] = metrics(load_runs(p), gt, ref, eps if ystd is not None else None, ystd=ystd)
    print(name, json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in table[name].items()}), flush=True)
  if args.out:
    json.dump(table, open(args.out, "w"), indent=2)


if __name__ == "__main__":
  main()

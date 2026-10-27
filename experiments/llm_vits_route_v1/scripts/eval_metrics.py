"""Prosody/latency metrics of commit logs against the real recordings (and optionally the E0 offline plan).

From token durations (plan: concatenated committed blocks; truth: d* alignment of the recording):
  mae_word   per unit |log word frames - truth|
  mae_pause  per word boundary |log1p(pause frames) - truth|, pause = trailing non-word frames of unit i plus
             leading non-word frames of unit i+1
  pause_f1   boundary pause >= 10 frames (116 ms) detection F1
  rate_err   |log total frames - truth| per utterance
plus mean lookahead in units, wait rate, compute, and T3 certification / trajectory violation statistics.
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
from stream_common import prosody_y, token_is_word, unit_token_spans  # noqa: E402

PAUSE_FR = 10


def load_runs(path):
  files = sorted(glob.glob(str(Path(path) / "commits*.jsonl")))
  return {d["utt_id"]: d for f in files for d in map(json.loads, open(f))}


def word_and_pauses(row, dur):
  isw = token_is_word(row["clean"])
  words, lead, trail = [], [], []
  for a, b in unit_token_spans(row):
    seg, wm = np.asarray(dur[a:b], dtype=np.float64), isw[a:b]
    idx = np.where(wm)[0]
    words.append(seg[wm].sum())
    lead.append(seg[:idx[0]].sum() if len(idx) else seg.sum())
    trail.append(seg[idx[-1] + 1:].sum() if len(idx) else 0.0)
  pauses = [trail[i] + lead[i + 1] for i in range(len(words) - 1)]
  return np.array(words), np.array(pauses)


def plan_durations(rec):
  return np.concatenate([np.asarray(u["dur"]) for u in sorted(rec["units"], key=lambda u: u["i"])])


def metrics(runs, rows, gt_dur, gt_y, ref=None, eps=None, w=(0.5, 0.5), ystd=None):
  ew, ep, er, rw, rp, look = [], [], [], [], [], []
  tp = fp = fn = 0
  traj_viol, cert, n_units, compute = 0, 0, 0, []
  for uid, rec in runs.items():
    row = rows[uid]
    pw, pp = word_and_pauses(row, plan_durations(rec))
    tw, tq = word_and_pauses(row, gt_dur[uid])
    ew.extend(np.abs(np.log(np.maximum(pw, 1)) - np.log(np.maximum(tw, 1))))
    ep.extend(np.abs(np.log1p(pp) - np.log1p(tq)))
    er.append(abs(np.log(plan_durations(rec).sum()) - np.log(gt_dur[uid].sum())))
    tp += int(((pp >= PAUSE_FR) & (tq >= PAUSE_FR)).sum())
    fp += int(((pp >= PAUSE_FR) & (tq < PAUSE_FR)).sum())
    fn += int(((pp < PAUSE_FR) & (tq >= PAUSE_FR)).sum())
    if ref is not None:
      fw, fq = word_and_pauses(row, plan_durations(ref[uid]))
      rw.extend(np.abs(np.log(np.maximum(pw, 1)) - np.log(np.maximum(fw, 1))))
      rp.extend(np.abs(np.log1p(pp) - np.log1p(fq)))
    viol = False
    for u in rec["units"]:
      look.append(u["j"] - u["i"])
      n_units += 1
      if u.get("certified"):
        cert += 1
        if eps is not None and float((np.array(w) * np.abs(np.array(u["y"]) - gt_y[uid][u["i"]]) / ystd).sum()) > eps:
          viol = True
    traj_viol += viol
    if "compute_s" in rec:
      compute.append(rec["compute_s"] / len(rec["units"]))
  out = {"n_utts": len(runs), "n_units": n_units, "mae_word": float(np.mean(ew)), "mae_pause": float(np.mean(ep)),
         "pause_f1": 2 * tp / max(1, 2 * tp + fp + fn), "rate_err": float(np.mean(er)),
         "look_mean": float(np.mean(look)), "wait_rate": float(np.mean(np.array(look) > 0))}
  if rw:
    out.update({"ref_mae_word": float(np.mean(rw)), "ref_mae_pause": float(np.mean(rp))})
  if cert or eps is not None:
    out.update({"certified_frac": cert / n_units, "traj_violation_rate": traj_viol / len(runs)})
  if compute:
    out["compute_ms_per_unit"] = 1000 * float(np.mean(compute))
  return out


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("runs", nargs="+")
  ap.add_argument("--align-dir", required=True)
  ap.add_argument("--manifest", default=str(ROOT / "data/manifest_v2.jsonl"))
  ap.add_argument("--ref", default=None, help="E0 offline commits dir")
  ap.add_argument("--risk", default=None, help="T3 risk head (for standardised violation checks)")
  ap.add_argument("--out", default=None)
  args = ap.parse_args()
  rows = {r["utt_id"]: r for r in map(json.loads, open(args.manifest))}
  gt_dur = dict(np.load(Path(args.align_dir) / "durations.npz"))
  gt_y = {d["utt_id"]: prosody_y(d["units"]) for d in map(json.loads, open(Path(args.align_dir) / "unit_prosody.jsonl"))}
  ref = load_runs(args.ref) if args.ref else None
  ystd = None
  if args.risk:
    import torch
    ystd = np.array(torch.load(args.risk, map_location="cpu")["meta"]["y_std"])
  table = {}
  for p in args.runs:
    name = Path(p).name
    eps = None
    if name.startswith("t3_"):
      eps = float(name.split("_")[-1])
    table[name] = metrics(load_runs(p), rows, gt_dur, gt_y, ref, eps if ystd is not None else None, ystd=ystd)
    print(name, json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in table[name].items()}), flush=True)
  if args.out:
    if Path(args.out).exists():
      raise FileExistsError(args.out)
    json.dump(table, open(args.out, "w"), indent=2)


if __name__ == "__main__":
  main()

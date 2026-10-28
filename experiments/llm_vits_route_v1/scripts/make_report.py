"""Collect the final tables: fixed lookahead per model (seed mean), policy operating points chosen on val and
reported on test, fixed-k interpolation at the same lookahead, and paired bootstraps. Writes <exp>/results.json.

Thresholds are chosen on val only (closest val lookahead to each target); test is touched once per chosen point.
"""
import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "vits"))
sys.path.insert(0, str(ROOT / "scripts"))
from compare import boot, per_utt  # noqa: E402
from eval_metrics import load_runs, metrics  # noqa: E402
from stream_common import prosody_y  # noqa: E402

KEYS = ["mae_word", "mae_pause", "pause_f1", "rate_err", "word_bias", "look_mean", "wait_rate", "certified_frac",
        "traj_violation_rate", "compute_ms_per_unit"]


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--exp", required=True)
  ap.add_argument("--targets", default="0.25,0.5")
  args = ap.parse_args()
  exp = Path(args.exp)
  rows = {r["utt_id"]: r for r in map(json.loads, open(ROOT / "data/manifest_v2.jsonl"))}
  gt_dur = dict(np.load(exp / "align/durations.npz"))
  gt_y = {d["utt_id"]: prosody_y(d["units"]) for d in map(json.loads, open(exp / "align/unit_prosody.jsonl"))}
  ystd = {"full": np.array(torch.load(exp / "t3/risk_s6f.pt", map_location="cpu")["meta"]["y_std"]),
          "lite": np.array(torch.load(exp / "t3/risk_s6f_lite.pt", map_location="cpu")["meta"]["y_std"])}
  cache = {}

  def M(path, eps=None, kind="full"):
    key = (str(path), eps)
    if key not in cache:
      m = metrics(load_runs(path), rows, gt_dur, gt_y, None, eps, ystd=ystd[kind] if eps is not None else None)
      cache[key] = {k: m[k] for k in KEYS if k in m}
    return cache[key]

  out = {"fixed": {}, "policies": {}, "pairs": {}}
  models = ["e1", "e2", "s6_future", "s6_rho", "s6"]
  for split in ["test", "val"]:
    for v in models:
      for k in [0, 1, 2, 3, 99]:
        runs = [p for p in [exp / f"eval_{split}/{v}_fixed{k}", exp / f"eval_{split}/{v}_seed29_fixed{k}"] if p.exists()]
        ms = [M(p) for p in runs]
        if not ms:
          continue
        out["fixed"][f"{split}/{v}/k{k}"] = {kk: float(np.mean([m[kk] for m in ms])) for kk in ms[0]} | {
            "seeds": len(ms), "seed_spread_word": float(np.ptp([m["mae_word"] for m in ms])) if len(ms) > 1 else None}
    out["fixed"][f"{split}/e0_offline"] = M(exp / f"eval_{split}/e0_offline")

  fam = {"entropy": ("eval_{}/s6_future_entropy_*", "full"), "spread": ("eval_{}/s6_future_spread_*", "full"),
         "t3": ("eval_{}/t3_s6_future_*", "full"), "t3_lite": ("eval_lite_{}/t3_s6_future_*", "lite")}
  fixed_test = [(out["fixed"][f"test/s6_future/k{k}"]["look_mean"], out["fixed"][f"test/s6_future/k{k}"]["mae_word"],
                 out["fixed"][f"test/s6_future/k{k}"]["mae_pause"]) for k in [0, 1, 2, 3] if f"test/s6_future/k{k}" in out["fixed"]]
  for name, (pat, kind) in fam.items():
    cands = []
    for p in sorted(glob.glob(str(exp / pat.format("val")))):
      thr = float(Path(p).name.split("_")[-1])
      cands.append((thr, M(p, thr if name.startswith("t3") else None, kind)["look_mean"], p))
    for tgt in map(float, args.targets.split(",")):
      thr, vlook, vp = min(cands, key=lambda c: abs(c[1] - tgt))
      tp = vp.replace("eval_val", "eval_test").replace("eval_lite_val", "eval_lite_test")
      tm = M(tp, thr if name.startswith("t3") else None, kind)
      lx = [f[0] for f in fixed_test]
      interp = {"mae_word": float(np.interp(tm["look_mean"], lx, [f[1] for f in fixed_test])),
                "mae_pause": float(np.interp(tm["look_mean"], lx, [f[2] for f in fixed_test]))}
      out["policies"][f"{name}@{tgt}"] = {"threshold": thr, "val_look": vlook, "test": tm,
                                         "fixed_k_interp_same_look": interp, "path": tp}

  E = lambda n: str(exp / f"eval_test/{n}")
  pairs = [("e1_fixed1", "e2_fixed1"), ("e2_fixed1", "s6_future_fixed1"), ("e2_fixed0", "s6_future_fixed0"),
           ("e1_fixed1", "s6_future_fixed0"), ("e2_fixed1", "s6_fixed1"), ("e2_fixed1", "s6_rho_fixed1"),
           ("s6_future_fixed1", out["policies"]["t3_lite@0.5"]["path"]),
           ("e1_fixed1", out["policies"]["t3_lite@0.25"]["path"]),
           (out["policies"]["entropy@0.25"]["path"], out["policies"]["t3_lite@0.25"]["path"]),
           (out["policies"]["spread@0.25"]["path"], out["policies"]["t3@0.25"]["path"])]
  rng = np.random.default_rng(0)
  for a, b in pairs:
    pa, pb = (a if a.startswith("/") else E(a)), (b if b.startswith("/") else E(b))
    ua, ub = per_utt(load_runs(pa), rows, gt_dur), per_utt(load_runs(pb), rows, gt_dur)
    keys = sorted(set(ua) & set(ub))
    res = {}
    for nm, i, j in [("word_mae", 0, 1), ("pause_mae", 2, 3), ("look", 4, 5)]:
      p, lo, hi, pge = boot(ua, ub, keys, i, j, 5000, rng)
      res[nm] = {"delta": float(p), "ci95": [float(lo), float(hi)], "boot_p_delta_ge_0": float(pge)}
    out["pairs"][f"{Path(pa).parent.name}/{Path(pa).name} -> {Path(pb).parent.name}/{Path(pb).name}"] = res
  if (exp / "results.json").exists():
    raise FileExistsError(exp / "results.json")
  json.dump(out, open(exp / "results.json", "w"), indent=1)
  for k, v in out["fixed"].items():
    if k.startswith("test"):
      print(k, {kk: round(x, 4) for kk, x in v.items() if isinstance(x, float)})
  for k, v in out["policies"].items():
    print(k, v["threshold"], {kk: round(x, 4) for kk, x in v["test"].items()}, "interp", v["fixed_k_interp_same_look"])
  for k, v in out["pairs"].items():
    print(k, {n: (round(r["delta"], 4), [round(x, 4) for x in r["ci95"]]) for n, r in v.items()})


if __name__ == "__main__":
  main()

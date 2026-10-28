"""T1 controlled audit (handbook T1.3, T1.4): bias-corrected future impact Gamma_hat with repeated samples.

For each val unit i at visible prefix j = i: J imagined suffixes (the LLM candidates of that prefix) x n independent
SDP noise seeds; the text-only E1 model plans unit i with prefix + suffix visible, giving Y_jb (standardised
2-d unit prosody). Gamma_hat (T1.5) = J^-1 sum_j ||Ybar_j - Ybar||^2 - (1 - 1/J)/(J n) sum_j tr(S_j), unbiased for the
between-suffix variance of conditional means (T1.6). Reported against the T1 Gamma head and the actual revision.
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
from fit_policies import PolicyHead, std_y  # noqa: E402
from run_stream_eval import features, load_stream  # noqa: E402
from simulator import Simulator  # noqa: E402
from stream_data import load_cands  # noqa: E402


def gamma_hat(Y):
  """Y [J, n, q] -> unbiased between-class variance of means (T1.5)."""
  J, n, _ = Y.shape
  yb = Y.mean(1)
  between = ((yb - yb.mean(0)) ** 2).sum(-1).mean()
  tr_s = sum(np.trace(np.atleast_2d(np.cov(Y[j].T, ddof=1))) for j in range(J))
  return float(between - (1 - 1 / J) / (J * n) * tr_s)


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--model", required=True, help="S6-future checkpoint (its plans give the actual revision)")
  ap.add_argument("--probe", required=True, help="E1 checkpoint (text-only planner for imagined suffixes)")
  ap.add_argument("--head", required=True, help="T1 Gamma head")
  ap.add_argument("--backbone-config", required=True)
  ap.add_argument("--backbone-ckpt", required=True)
  ap.add_argument("--n-seeds", type=int, default=8)
  ap.add_argument("--limit", type=int, default=100)
  ap.add_argument("--out", required=True)
  args = ap.parse_args()
  out = Path(args.out)
  if out.exists():
    raise FileExistsError(out)
  dev = torch.device("cuda")
  model = load_stream(args.model, args.backbone_config, args.backbone_ckpt, dev)
  probe_model = load_stream(args.probe, args.backbone_config, args.backbone_ckpt, dev)
  head = PolicyHead(args.head, "cpu")
  cands = load_cands(sorted(glob.glob(str(ROOT / "data/future_cands/shard*.jsonl"))))
  rows = [r for r in map(json.loads, open(ROOT / "data/manifest_v2.jsonl")) if r["split"] == "val"][:args.limit]
  recs = []
  for r in rows:
    qz = np.load(ROOT / f"data/qwen_prefix/{r['utt_id']}.npz")
    sim = Simulator(model, r, dev, qwen=qz, cands=cands, noise_w=0.8)
    probes = [Simulator(probe_model, r, dev, cands=cands, noise_w=0.8, seed=1234 + b) for b in range(args.n_seeds)]
    for i in range(sim.U - 1):
      plan = sim.plan(i, i)
      seqs = probes[0].future_seqs(i)
      Y = np.stack([np.stack([std_y(y, head.meta) for y in p.plan_imagined(i, i, seqs)]) for p in probes], 1)
      feat = features(sim, probes[0], i, i, plan, qz)
      y_full = sim.plan(i, sim.U - 1)["y"]
      rev = float(((std_y(y_full, head.meta) - std_y(plan["y"], head.meta)) ** 2 * np.array([0.5, 0.5])).sum())
      recs.append({"utt": r["utt_id"], "i": i, "gamma_hat": gamma_hat(Y), "gamma_head": head.gamma(feat),
                   "revision": rev, "naive_between": float(((Y.mean(1) - Y.mean((0, 1))) ** 2).sum(-1).mean())})
      sim.commit(i, i, plan)
      for p in probes:
        p.committed_y = list(sim.committed_y)
  g, h, v, nb = (np.array([x[k] for x in recs]) for k in ["gamma_hat", "gamma_head", "revision", "naive_between"])
  summary = {"n_units": len(recs), "J": 8, "n_seeds": args.n_seeds, "gamma_hat_mean": float(g.mean()),
             "naive_between_mean": float(nb.mean()), "noise_correction_mean": float((nb - g).mean()),
             "frac_negative": float((g < 0).mean()),
             "corr_gamma_hat_revision": float(np.corrcoef(g, v)[0, 1]),
             "corr_gamma_head_revision": float(np.corrcoef(h, v)[0, 1]),
             "corr_gamma_hat_head": float(np.corrcoef(g, h)[0, 1])}
  json.dump({"summary": summary, "units": recs}, open(out, "w"))
  print(json.dumps(summary), flush=True)


if __name__ == "__main__":
  main()

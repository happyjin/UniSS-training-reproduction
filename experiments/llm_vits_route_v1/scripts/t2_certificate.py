"""T2.6 finite-class certificate on an independent dump (val), pooled per lookahead L = j - i.

P_hat: class frequencies of the full-context plans y_full (the realised future's acoustic target); Q_hat: class
frequencies of the E1 probe's plans under the M imagined LLM futures (the forecast); classes, centroid metric C and
audited e^U come from fit_policies --kind t2 (train/calib dumps, frozen before this split). Bound:
W1(PK, Q Khat) <= sum_j P_hat_j e^U_j + OT_C(P_hat, Q_hat) + 2 Delta t_P + Delta t_Q (prob. >= 1 - delta_conf if the
samples were i.i.d.; rows of one utterance are dependent, so the n used here overstates independence: the
certificate is reported as an empirical check, as T2.4 asks for natural data). Compared with the empirical W1 between
the y_full sample and the forecast centroid sample (weighted L1 ground metric, exact LP on subsamples).
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from fit_policies import W, load_rows, ot_cost, std_y  # noqa: E402


def emp_w1(a, b, n=300, seed=0):
  from scipy.optimize import linear_sum_assignment
  rng = np.random.default_rng(seed)
  a = a[rng.choice(len(a), min(n, len(a)), replace=False)]
  b = b[rng.choice(len(b), len(a), replace=len(b) < len(a))]
  cost = (W * np.abs(a[:, None] - b[None])).sum(-1)
  r, c = linear_sum_assignment(cost)
  return float(cost[r, c].mean())


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--dump", required=True)
  ap.add_argument("--head", required=True)
  ap.add_argument("--out", required=True)
  args = ap.parse_args()
  if Path(args.out).exists():
    raise FileExistsError(args.out)
  m = torch.load(args.head, map_location="cpu", weights_only=False)["meta"]
  cen, C, eU = np.array(m["centroids"]), np.array(m["C"]), np.array(m["e_upper"])
  J, delta, dc = len(cen), m["diameter"], m["delta_conf"]
  cls = lambda Y: np.argmin(((Y[:, None] - cen[None]) ** 2).sum(-1), 1)
  rows = load_rows(args.dump)
  res = {}
  for L in range(4):
    sel = [r for r in rows if r["j"] - r["i"] == L]
    yf = np.stack([std_y(r["y_full"], m) for r in sel])
    yq = np.concatenate([np.stack([std_y(y, m) for y in r["probe_ys"]]) for r in sel])
    p = np.bincount(cls(yf), minlength=J) / len(yf)
    q = np.bincount(cls(yq), minlength=J) / len(yq)
    t = lambda n: min(1.0, J / 2 * math.sqrt(math.log(6 * J / dc) / (2 * n)))
    ot, resid = ot_cost(p, q, C)
    bound = float((p * eU).sum() + ot + 2 * delta * t(len(yf)) + delta * t(len(yq)))
    actual = emp_w1(yf, cen[cls(yq)])
    res[f"L{L}"] = {"n_P": len(yf), "n_Q": len(yq), "sum_P_eU": float((p * eU).sum()), "OT": ot, "lp_residual": resid,
                    "sampling_terms": 2 * delta * t(len(yf)) + delta * t(len(yq)), "bound": bound,
                    "empirical_W1": actual, "covered": bool(actual <= bound), "diameter": delta}
  json.dump(res, open(args.out, "w"), indent=1)
  print(json.dumps(res, indent=1))


if __name__ == "__main__":
  main()

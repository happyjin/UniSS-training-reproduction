"""T3: fit the Gaussian risk head F_eta on training trajectories, calibrate one trajectory-level radius Q_conf.

Targets Y* (unit prosody: log word frames, log1p pause frames) are by default the same model's full-context plan of
the unit (handbook T1/T3: a full-context reference), so the guarantee is about commitment regret: a certified commit
is within eps of what waiting for the whole sentence would have produced. --target gt uses the recording instead
(its irreducible variability makes the set width almost independent of lookahead). Standardised with training stats. Calibration score of trajectory i (T3.2): S_i = max over its predetermined grid (unit, visible j <= unit +
k_max) and coordinates of |Y* - mu| / s; Q_conf = S_(ceil((n+1)(1-a))). Each LJSpeech utterance is one exchangeable
trajectory. The certificate (T3.4) B = sum_l w_l (|y_hat_l - mu_l| + Q_conf s_l) bounds the weighted L1 distance from
the plan to every point of the set, so P(exists certified commit with d_w(y_hat, Y*) > eps) <= a (T3.5).
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "vits"))
from stream_common import prosody_y  # noqa: E402

W = np.array([0.5, 0.5])


def feat_vec(f, lite=False, k_max=3):
  """lite drops the acoustic-impact probe (M imagined futures), keeping only features that cost nothing extra."""
  look = np.zeros(k_max + 1)
  look[min(f["look"], k_max)] = 1.0
  probe = [] if lite else [*f["probe_mean"], *f["probe_mad"]]
  return np.concatenate([f["y_hat"], probe, [f["entropy"], f["p_punct"], f["end"], f["n_chars"], f["punct_end"]],
                         look]).astype(np.float32)


class RiskNet(nn.Module):
  def __init__(self, d_in, s_min=0.05):
    super().__init__()
    self.net = nn.Sequential(nn.Linear(d_in, 256), nn.GELU(), nn.Linear(256, 256), nn.GELU(), nn.Linear(256, 4))
    self.s_min = s_min

  def forward(self, x):
    mu, a = self.net(x).chunk(2, -1)
    return mu, nn.functional.softplus(a) + self.s_min


class RiskHead:
  def __init__(self, net, meta, dev):
    self.net, self.meta, self.dev = net.to(dev).eval(), meta, dev
    self.fm, self.fs = np.array(meta["f_mean"]), np.array(meta["f_std"])
    self.lite = bool(meta.get("lite", False))
    self.ym, self.ys = np.array(meta["y_mean"]), np.array(meta["y_std"])

  @classmethod
  def load(cls, path, dev):
    ck = torch.load(path, map_location="cpu")
    net = RiskNet(len(ck["meta"]["f_mean"]))
    net.load_state_dict(ck["net"])
    return cls(net, ck["meta"], dev)

  def predict(self, feats):
    x = (np.stack([feat_vec(f, self.lite) for f in feats]) - self.fm) / self.fs
    with torch.no_grad():
      mu, s = self.net(torch.from_numpy(x.astype(np.float32)).to(self.dev))
    return mu.cpu().numpy(), s.cpu().numpy()

  def bound(self, f):
    mu, s = self.predict([f])
    yh = (np.array(f["y_hat"]) - self.ym) / self.ys
    return float((W * (np.abs(yh - mu[0]) + self.meta["q_conf"] * s[0])).sum())


def load_dump(path, gt, target="full", lite=False):
  """path: a commits jsonl or a directory of commits*.jsonl shards."""
  p = Path(path)
  files = sorted(p.glob("commits*.jsonl")) if p.is_dir() else [p]
  X, Y, YH, traj = [], [], [], []
  for line in (l for f in files for l in open(f)):
    rec = json.loads(line)
    ys = gt[rec["utt_id"]]
    for d in rec["dump"]:
      X.append(feat_vec(d, lite))
      Y.append(d["y_full"] if target == "full" else ys[d["i"]])
      YH.append(d["y_hat"])
      traj.append(rec["utt_id"])
  return np.stack(X), np.stack(Y), np.stack(YH), np.array(traj)


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--train-dump", required=True)
  ap.add_argument("--calib-dump", required=True)
  ap.add_argument("--align-dir", required=True)
  ap.add_argument("--a-risk", type=float, default=0.1)
  ap.add_argument("--target", default="full", choices=["full", "gt"],
                  help="full: the model's own full-context plan (commitment regret); gt: the recording")
  ap.add_argument("--epochs", type=int, default=30)
  ap.add_argument("--lite", action="store_true", help="no acoustic-impact probe features (no extra compute)")
  ap.add_argument("--out", required=True)
  args = ap.parse_args()
  out = Path(args.out)
  if out.exists():
    raise FileExistsError(out)
  gt = {d["utt_id"]: prosody_y(d["units"]) for d in map(json.loads, open(Path(args.align_dir) / "unit_prosody.jsonl"))}
  Xt, Yt, _, _ = load_dump(args.train_dump, gt, args.target, args.lite)
  Xc, Yc, YHc, Tc = load_dump(args.calib_dump, gt, args.target, args.lite)
  fm, fs = Xt.mean(0), Xt.std(0) + 1e-6
  ym, ys = Yt.mean(0), Yt.std(0) + 1e-6
  torch.manual_seed(0)
  net = RiskNet(Xt.shape[1])
  opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-4)
  xt = torch.from_numpy((Xt - fm) / fs).float()
  yt = torch.from_numpy((Yt - ym) / ys).float()
  for ep in range(args.epochs):
    perm = torch.randperm(len(xt))
    tot = 0.0
    for s in range(0, len(xt), 1024):
      idx = perm[s:s + 1024]
      mu, sd = net(xt[idx])
      loss = (0.5 * ((yt[idx] - mu) / sd) ** 2 + torch.log(sd)).sum(-1).mean()   # T3.1
      opt.zero_grad()
      loss.backward()
      opt.step()
      tot += loss.item() * len(idx)
    print(json.dumps({"epoch": ep, "nll": tot / len(xt)}), flush=True)
  net.eval()
  with torch.no_grad():
    mu, sd = net(torch.from_numpy((Xc - fm) / fs).float())
  r = (np.abs((Yc - ym) / ys - mu.numpy()) / sd.numpy()).max(-1)
  scores = np.array([r[Tc == t].max() for t in np.unique(Tc)])   # one max score per trajectory (T3.2)
  n = len(scores)
  b = math.ceil((n + 1) * (1 - args.a_risk))
  q = float(np.sort(scores)[b - 1]) if b <= n else float("inf")
  meta = {"f_mean": fm.tolist(), "f_std": fs.tolist(), "y_mean": ym.tolist(), "y_std": ys.tolist(), "q_conf": q,
          "a_risk": args.a_risk, "target": args.target, "lite": args.lite, "n_cal": n, "b": b, "w": W.tolist(),
          "calib_score_quantiles": np.quantile(scores, [0.5, 0.9, 0.95]).tolist(),
          "train_rows": int(len(Xt)), "calib_rows": int(len(Xc))}
  torch.save({"net": net.state_dict(), "meta": meta}, out)
  print(json.dumps(meta | {"f_mean": None, "f_std": None}), flush=True)


if __name__ == "__main__":
  main()

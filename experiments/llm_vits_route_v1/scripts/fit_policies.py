"""Policy heads for T1, T2 and T5 on the T3 feature dumps (S6-future trajectories, target = full-context plan).

  --kind t1  Gamma head (T1.1-T1.4): predicts the expected squared revision ||(y_full - y_hat)/y_std||^2 of a commit
             (missing-information loss plus model gap, T1.3) from prefix features; policy gamma:TAU commits when the
             prediction is <= TAU.
  --kind t2  finite acoustic classes (T2.4-T2.5): k-means (J classes) on standardised full-context plans of the
             training dump, centroid metric C (weighted L1, coordinates clipped to +-3 so the diameter is bounded),
             class audit e_j^U on the calibration dump (one-sided Hoeffding, T2.6). Policy ot:TAU commits when
             OT_C(Q_hat, delta_class(y_hat)) + e^U_class(y_hat) <= TAU, Q_hat = classes of the M imagined-future probe plans.
  --kind t5  finite-horizon stopping (T5.1-T5.3): exact backward DP along each dumped trajectory with STOP loss
             d_w(y_hat_j, y_full) and WAIT cost lam per word; an MLP regresses Q(STOP), Q(WAIT) from features, T3 risk
             statistics and lam. Policy t5:LAM[:EPS] stops when Q_STOP <= Q_WAIT (with EPS: only when T3-certified).
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
sys.path.insert(0, str(ROOT / "scripts"))
from fit_t3 import W, RiskHead, feat_vec  # noqa: E402

CLIP, K_MAX = 3.0, 3


def load_rows(path):
  p = Path(path)
  files = sorted(p.glob("commits*.jsonl"))
  out = []
  for line in (l for f in files for l in open(f)):
    rec = json.loads(line)
    for d in rec["dump"]:
      out.append({"utt": rec["utt_id"], **d})
  return out


def std_y(y, meta):
  return np.clip((np.asarray(y) - meta["y_mean"]) / meta["y_std"], -CLIP, CLIP)


def dist_w(a, b):
  return float((W * np.abs(a - b)).sum(-1)) if np.ndim(a) == 1 else (W * np.abs(a - b)).sum(-1)


def mlp(d_in, d_out):
  return nn.Sequential(nn.Linear(d_in, 256), nn.GELU(), nn.Linear(256, 256), nn.GELU(), nn.Linear(256, d_out))


def fit_regressor(X, Y, epochs=30, seed=0):
  torch.manual_seed(seed)
  fm, fs = X.mean(0), X.std(0) + 1e-6
  net = mlp(X.shape[1], Y.shape[1])
  opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-4)
  x, y = torch.from_numpy((X - fm) / fs).float(), torch.from_numpy(Y).float()
  for ep in range(epochs):
    perm = torch.randperm(len(x))
    tot = 0.0
    for s in range(0, len(x), 1024):
      idx = perm[s:s + 1024]
      loss = nn.functional.smooth_l1_loss(net(x[idx]), y[idx])
      opt.zero_grad()
      loss.backward()
      opt.step()
      tot += loss.item() * len(idx)
    print(json.dumps({"epoch": ep, "loss": tot / len(x)}), flush=True)
  return net.eval(), fm, fs


def kmeans(X, k, iters=50, seed=0):
  rng = np.random.default_rng(seed)
  c = X[rng.choice(len(X), k, replace=False)]
  for _ in range(iters):
    lab = np.argmin(((X[:, None] - c[None]) ** 2).sum(-1), 1)
    c = np.stack([X[lab == j].mean(0) if (lab == j).any() else c[j] for j in range(k)])
  return c


def ot_cost(p, q, C):
  """Exact unregularised OT between two J-point distributions (small LP via scipy)."""
  from scipy.optimize import linprog
  J = len(p)
  A = []
  for i in range(J):
    row = np.zeros((J, J)); row[i] = 1; A.append(row.ravel())
  for j in range(J):
    col = np.zeros((J, J)); col[:, j] = 1; A.append(col.ravel())
  res = linprog(C.ravel(), A_eq=np.array(A), b_eq=np.concatenate([p, q]), bounds=(0, None), method="highs")
  return float(res.fun), float(np.abs(np.array(A) @ res.x - np.concatenate([p, q])).max())


def t5_features(rows, risk, lam):
  mu, s = risk.predict(rows)
  base = np.stack([feat_vec(r, risk.lite) for r in rows])
  return np.concatenate([base, mu, s, np.full((len(rows), 1), lam)], 1).astype(np.float32)


class PolicyHead:
  """Loaded at evaluation time by run_stream_eval (policies gamma / ot / t5)."""

  def __init__(self, path, dev):
    self.ck = torch.load(path, map_location="cpu", weights_only=False)
    self.kind, self.meta = self.ck["kind"], self.ck["meta"]
    if self.kind in ("t1", "t5"):
      self.net = mlp(len(self.meta["f_mean"]), 1 if self.kind == "t1" else 2)
      self.net.load_state_dict(self.ck["net"])
      self.net.eval()
    if self.kind == "t5":
      self.risk = RiskHead.load(self.meta["risk"], dev)

  def _run(self, x):
    x = (x - np.array(self.meta["f_mean"])) / np.array(self.meta["f_std"])
    with torch.no_grad():
      return self.net(torch.from_numpy(x.astype(np.float32))).numpy()

  def gamma(self, f):
    return float(self._run(feat_vec(f, False)[None])[0, 0])

  def ot_risk(self, f, probe_ys):
    m = self.meta
    cen, C, eU = np.array(m["centroids"]), np.array(m["C"]), np.array(m["e_upper"])
    cls = lambda y: int(np.argmin(((std_y(y, m) - cen) ** 2).sum(-1)))
    q = np.bincount([cls(y) for y in probe_ys], minlength=len(cen)) / len(probe_ys)
    c0 = cls(f["y_hat"])
    return float((q * C[:, c0]).sum() + eU[c0])

  def t5_q(self, f, lam):
    return self._run(t5_features([f], self.risk, lam))[0]


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--kind", required=True, choices=["t1", "t2", "t5"])
  ap.add_argument("--train-dump", required=True)
  ap.add_argument("--calib-dump", required=True)
  ap.add_argument("--risk", required=True, help="T3 risk head (standardisation stats; T5 risk features)")
  ap.add_argument("--classes", type=int, default=8)
  ap.add_argument("--delta-conf", type=float, default=0.1)
  ap.add_argument("--lams", default="0.02,0.05,0.1,0.2,0.4")
  ap.add_argument("--out", required=True)
  args = ap.parse_args()
  out = Path(args.out)
  if out.exists():
    raise FileExistsError(out)
  risk = RiskHead.load(args.risk, "cpu")
  ym = {"y_mean": np.array(risk.meta["y_mean"]), "y_std": np.array(risk.meta["y_std"])}
  tr, ca = load_rows(args.train_dump), load_rows(args.calib_dump)
  meta = {"kind": args.kind, "y_mean": ym["y_mean"].tolist(), "y_std": ym["y_std"].tolist(), "train_rows": len(tr)}
  ck = {"kind": args.kind}
  if args.kind == "t1":
    X = np.stack([feat_vec(r, False) for r in tr])
    Y = np.array([[((std_y(r["y_full"], ym) - std_y(r["y_hat"], ym)) ** 2 * W).sum()] for r in tr], dtype=np.float32)
    net, fm, fs = fit_regressor(X, Y)
    Xc = np.stack([feat_vec(r, False) for r in ca])
    Yc = np.array([((std_y(r["y_full"], ym) - std_y(r["y_hat"], ym)) ** 2 * W).sum() for r in ca])
    with torch.no_grad():
      pc = net(torch.from_numpy(((Xc - fm) / fs).astype(np.float32))).numpy()[:, 0]
    meta.update(f_mean=fm.tolist(), f_std=fs.tolist(), calib_corr=float(np.corrcoef(pc, Yc)[0, 1]),
                calib_mean_target=float(Yc.mean()))
    ck["net"] = net.state_dict()
  elif args.kind == "t2":
    Yt = np.stack([std_y(r["y_full"], ym) for r in tr])
    cen = kmeans(Yt, args.classes)
    C = (W * np.abs(cen[:, None] - cen[None])).sum(-1)
    delta = float((W * 2 * CLIP).sum())
    J = args.classes
    Yc = np.stack([std_y(r["y_full"], ym) for r in ca])
    lab = np.argmin(((Yc[:, None] - cen[None]) ** 2).sum(-1), 1)
    eU, ebar, nj = [], [], []
    for j in range(J):
      dj = (W * np.abs(Yc[lab == j] - cen[j])).sum(-1)
      n = len(dj)
      e = float(dj.mean()) if n else delta
      eU.append(min(delta, e + delta * math.sqrt(math.log(3 * J / args.delta_conf) / (2 * n))) if n else delta)
      ebar.append(e)
      nj.append(n)
    meta.update(centroids=cen.tolist(), C=C.tolist(), e_upper=eU, e_bar=ebar, n_audit=nj, diameter=delta,
                delta_conf=args.delta_conf)
  else:
    lams = [float(l) for l in args.lams.split(",")]
    groups = {}
    for k, r in enumerate(tr):
      groups.setdefault((r["utt"], r["i"]), []).append(k)
    Xs, Ys = [], []
    loss = np.array([dist_w(std_y(r["y_hat"], ym), std_y(r["y_full"], ym)) for r in tr])
    for lam in lams:
      X = t5_features(tr, risk, lam)
      Y = np.zeros((len(tr), 2), dtype=np.float32)
      for idx in groups.values():
        idx = sorted(idx, key=lambda k: tr[k]["j"])
        v_next = None
        for k in reversed(idx):                          # exact DP on the observed trajectory (T5.1)
          q_stop = loss[k]
          # at the deadline WAIT is unavailable: its target is set one unit above STOP so the head learns to stop
          q_wait = lam + v_next if v_next is not None else q_stop + 1.0
          Y[k] = [q_stop, q_wait]
          v_next = min(q_stop, q_wait)
      Xs.append(X)
      Ys.append(Y)
    X, Y = np.concatenate(Xs), np.concatenate(Ys)
    net, fm, fs = fit_regressor(X, Y)
    meta.update(f_mean=fm.tolist(), f_std=fs.tolist(), lams=lams, risk=str(Path(args.risk).resolve()))
    ck["net"] = net.state_dict()
  ck["meta"] = meta
  torch.save(ck, out)
  print(json.dumps({k: v for k, v in meta.items() if k not in ("f_mean", "f_std", "centroids", "C")}), flush=True)


if __name__ == "__main__":
  main()

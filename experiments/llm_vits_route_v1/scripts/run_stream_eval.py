"""Simulate streaming synthesis for one (model, policy) pair over a split and record every commit.

Policies (k_max caps the wait; the end of stream always allows commitment):
  fixed:K            commit unit i when unit min(i+K, U-1) has arrived (E1 protocol)
  entropy:TAU        E5 lexical gate: wait while Qwen's next-token entropy after the visible prefix exceeds TAU
  spread:TAU         E6 acoustic-impact gate (T2 geometry): wait while the E1 probe's plans for unit i under the M
                     imagined LLM futures disagree by more than TAU (mean abs deviation of Y, averaged over coords)
  t3:EPS             T3 certificate: commit when B = sum_l w_l (|y_hat - mu_l| + Q s_l) <= EPS (risk head + Q from
                     calibration); a deadline commit is an uncertified override and is flagged. The certified target is
                     the same model's full-context plan y_full of the unit (recorded), so a certified commit is one that
                     waiting for the whole sentence would not have changed by more than EPS
  gamma:TAU          T1 Gamma head: commit when the predicted squared revision of the plan is <= TAU
  ot:TAU             T2 finite-class acoustic OT risk of the imagined futures (+ class audit bound) <= TAU
  t5:LAM[:EPS]       T5 value policy, WAIT cost LAM per word: stop when Q_STOP <= Q_WAIT (EPS: only T3-certified)
  dump               no policy: record the T3 feature vector at every (i, j <= i+k_max) along the k_max trajectory
Outputs <out>/commits.jsonl (+ wavs/ when --render).
"""
import argparse
import glob
import json
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "vits"))
sys.path.insert(0, str(ROOT / "scripts"))
from simulator import Simulator  # noqa: E402
from stream_data import load_cands  # noqa: E402
from stream_model import StreamVITS  # noqa: E402
from stream_common import load_backbone  # noqa: E402
from train_stream import VARIANTS  # noqa: E402


def load_stream(path, backbone_config, backbone_ckpt, dev):
  ck = torch.load(path, map_location="cpu")
  m = StreamVITS(load_backbone(backbone_config, backbone_ckpt), **VARIANTS[ck["variant"]])
  missing, unexpected = m.load_state_dict(ck["model"], strict=False)
  assert not unexpected and all(k.startswith("frozen.") for k in missing), (missing, unexpected)
  return m.to(dev).eval()


def features(sim, probe, i, j, plan, qz):
  """T3/T2 features of committing unit i with units 0..j visible (probe None: no impact-probe features)."""
  if probe is not None:
    ys = np.stack(probe.plan_imagined(i, j, probe.future_seqs(j)))
  else:
    ys = np.zeros((1, 2))
  dev = np.abs(ys - ys.mean(0)).mean(0)
  word = sim.row["clean"][sim.row["units"][i][0]:sim.row["units"][i][1]].strip()
  sim.last_probe_ys = ys
  return {"y_hat": plan["y"].tolist(), "probe_mean": ys.mean(0).tolist(), "probe_mad": dev.tolist(),
          "entropy": float(qz["unit_entropy"][j]), "p_punct": float(qz["unit_p_punct"][j]), "look": j - i,
          "end": int(j == sim.U - 1), "n_chars": len(word), "punct_end": int(word[-1] in ",.;:!?")}


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--model", required=True, help="stream checkpoint")
  ap.add_argument("--probe", default=None, help="E1 stream checkpoint for the acoustic-impact probe")
  ap.add_argument("--policy", required=True)
  ap.add_argument("--k-max", type=int, default=3)
  ap.add_argument("--risk", default=None, help="t3 risk head json (from fit_t3.py)")
  ap.add_argument("--policy-head", default=None, help="T1 / T2 / T5 head (from fit_policies.py)")
  ap.add_argument("--backbone-config", required=True)
  ap.add_argument("--backbone-ckpt", required=True)
  ap.add_argument("--manifest", default=str(ROOT / "data/manifest_v2.jsonl"))
  ap.add_argument("--qwen-dir", default=str(ROOT / "data/qwen_prefix"))
  ap.add_argument("--cands", default=str(ROOT / "data/future_cands/shard*.jsonl"))
  ap.add_argument("--split", default="test")
  ap.add_argument("--limit", type=int, default=0)
  ap.add_argument("--shard", type=int, default=0)
  ap.add_argument("--num-shards", type=int, default=1)
  ap.add_argument("--noise-w", type=float, default=0.0)
  ap.add_argument("--noise-z", type=float, default=0.667)
  ap.add_argument("--render", action="store_true")
  ap.add_argument("--out", required=True)
  args = ap.parse_args()
  dev = torch.device("cuda")
  out = Path(args.out)
  out.mkdir(parents=True, exist_ok=True)
  res_path = out / (f"commits.shard{args.shard}.jsonl" if args.num_shards > 1 else "commits.jsonl")
  if res_path.exists():
    raise FileExistsError(res_path)
  if args.render:
    (out / "wavs").mkdir(exist_ok=True)
  model = load_stream(args.model, args.backbone_config, args.backbone_ckpt, dev)
  pname, _, pval = args.policy.partition(":")
  risk, head = None, None
  if pname == "t3" or (pname == "t5" and ":" in pval):
    from fit_t3 import RiskHead
    risk = RiskHead.load(args.risk, dev)
  if pname in ("gamma", "ot", "t5"):
    from fit_policies import PolicyHead
    head = PolicyHead(args.policy_head, "cpu")
  need_probe = pname in ("spread", "dump", "gamma", "ot", "t5") or (pname == "t3" and not risk.lite)
  probe_model = load_stream(args.probe, args.backbone_config, args.backbone_ckpt, dev) if need_probe else None
  cands = load_cands(sorted(glob.glob(args.cands))) if (model.use_future or need_probe) else None
  rows = [json.loads(l) for l in open(args.manifest)]
  rows = [r for r in rows if r["split"] == args.split]
  if args.limit:
    rows = rows[:args.limit]
  rows = rows[args.shard::args.num_shards]
  with open(res_path, "w") as f:
    for r in rows:
      qz = np.load(f"{args.qwen_dir}/{r['utt_id']}.npz")
      kw = dict(cands=cands, noise_w=args.noise_w, noise_z=args.noise_z)
      sim = Simulator(model, r, dev, qwen=qz if model.use_qwen else None, **kw)
      probe = Simulator(probe_model, r, dev, **kw) if need_probe else None
      t0, excluded = time.time(), 0.0
      rec = {"utt_id": r["utt_id"], "units": [], "dump": []}
      for i in range(sim.U):
        t_ref = time.time()   # the full-context reference plan is evaluation-only: excluded from compute_s
        y_full = sim.plan(i, sim.U - 1)["y"].tolist() if pname in ("t3", "dump", "gamma", "ot", "t5") else None
        excluded += time.time() - t_ref
        j = i
        while True:
          plan = sim.plan(i, j)
          deadline = j - i >= args.k_max or j == sim.U - 1
          info = {}
          if pname == "fixed":
            go = j >= min(i + int(pval), sim.U - 1)
          elif pname == "entropy":
            go = deadline or qz["unit_entropy"][j] <= float(pval)
          elif pname in ("spread", "t3", "dump", "gamma", "ot", "t5"):
            if probe is not None:
              probe.committed_y = list(sim.committed_y)
            feat = features(sim, probe, i, j, plan, qz)
            if pname == "spread":
              info["mad"] = float(np.mean(feat["probe_mad"]))
              go = deadline or info["mad"] <= float(pval)
            elif pname == "t3":
              b = risk.bound(feat)
              info["B"] = b
              go = deadline or b <= float(pval)
              info["certified"] = bool(b <= float(pval))
            elif pname == "gamma":
              info["gamma"] = head.gamma(feat)
              go = deadline or info["gamma"] <= float(pval)
            elif pname == "ot":
              info["ot"] = head.ot_risk(feat, sim.last_probe_ys)
              go = deadline or info["ot"] <= float(pval)
            elif pname == "t5":
              lam, _, eps = pval.partition(":")
              q_stop, q_wait = head.t5_q(feat, float(lam))
              info["q"] = [float(q_stop), float(q_wait)]
              ok = True
              if eps:
                info["B"] = risk.bound(feat)
                ok = info["B"] <= float(eps)
                info["certified"] = bool(ok)
              go = deadline or (q_stop <= q_wait and ok)
            else:
              rec["dump"].append({"i": i, "j": j, "y_full": y_full, **feat})
              go = deadline
          if go:
            break
          j += 1
        sim.commit(i, j, plan)
        if y_full is not None:
          info["y_full"] = y_full
        rec["units"].append({"i": i, "j": j, "y": plan["y"].tolist(), "dur": plan["dur"].tolist(), **info})
      rec["compute_s"] = time.time() - t0 - excluded
      if args.render:
        wav = sim.render_incremental()
        sf.write(out / "wavs" / f"{r['utt_id']}.wav", wav, 22050)
      if not rec["dump"]:
        rec.pop("dump")
      f.write(json.dumps(rec) + "\n")
      f.flush()
  print(f"done {len(rows)} -> {res_path}", flush=True)


if __name__ == "__main__":
  main()

"""All handbook schemes on LJSpeech test: one results file and the pick of the best scheme per role.

  --stage select  choose T1 / T2 / T5 (and T5 + T3 mask) operating points on val (closest val lookahead to each
                  target, as make_report.py) and write <all>/chosen_policies.json for the test renders
  --stage final   write <all>/results_all.json:
    acoustic   S5, T4, S2, S3, S4 behind identical S6-future fixed:1 plans: WER / UTMOS / word log-F0 and acoustic
               real-time factor, paired bootstrap against the E0 acoustic path (vits) on the same plans
    planner    S1 against S6-future at fixed lookahead (duration metrics, paired bootstrap, audio)
    policies   T1 / T2 / T5 test operating points next to T3, T3-lite, entropy (E5) and spread (E6) from
               results_v2.json, paired bootstraps, audio at the chosen points
    theory     T1 Gamma audit and T2.6 certificate summaries
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
from compare_audio import paired  # noqa: E402
from eval_metrics import load_runs, metrics  # noqa: E402
from stream_common import prosody_y  # noqa: E402

KEYS = ["mae_word", "mae_pause", "pause_f1", "rate_err", "word_bias", "look_mean", "wait_rate", "certified_frac",
        "traj_violation_rate", "compute_ms_per_unit"]
FAMILIES = {"t1": "t1_s6_future_*", "t2": "t2_s6_future_*", "t5": "t5_s6_future_*"}


def fam_paths(base, fam):
  ps = sorted(glob.glob(str(base / FAMILIES[fam.replace("t5m", "t5")])))
  masked = [p for p in ps if Path(p).name.count("_") == 4]          # t5_s6_future_LAM_EPS
  if fam == "t5m":
    return masked
  return [p for p in ps if p not in masked] if fam == "t5" else ps


class Ctx:
  def __init__(self, final):
    self.rows = {r["utt_id"]: r for r in map(json.loads, open(ROOT / "data/manifest_v2.jsonl"))}
    self.gt_dur = dict(np.load(final / "align/durations.npz"))
    self.gt_y = {d["utt_id"]: prosody_y(d["units"]) for d in map(json.loads, open(final / "align/unit_prosody.jsonl"))}
    self.ystd = np.array(torch.load(final / "t3/risk_s6f.pt", map_location="cpu")["meta"]["y_std"])
    self.cache = {}

  def m(self, path, eps=None):
    key = (str(path), eps)
    if key not in self.cache:
      mm = metrics(load_runs(path), self.rows, self.gt_dur, self.gt_y, None, eps, ystd=self.ystd if eps else None)
      self.cache[key] = {k: float(mm[k]) for k in KEYS if k in mm}
    return self.cache[key]

  def pair(self, a, b):
    ua, ub = per_utt(load_runs(a), self.rows, self.gt_dur), per_utt(load_runs(b), self.rows, self.gt_dur)
    keys = sorted(set(ua) & set(ub))
    rng = np.random.default_rng(0)
    out = {}
    for nm, i, j in [("word_mae", 0, 1), ("pause_mae", 2, 3), ("look", 4, 5)]:
      p, lo, hi, _ = boot(ua, ub, keys, i, j, 5000, rng)
      out[nm] = {"delta": round(float(p), 4), "ci95": [round(float(lo), 4), round(float(hi), 4)]}
    return out


def eps_of(path):
  name = Path(path).name
  return float(name.split("_")[-1]) if name.count("_") == 4 else None


def select(ctx, pol_root, targets):
  chosen = {}
  for fam in ["t1", "t2", "t5", "t5m"]:
    cands = [(Path(p).name, ctx.m(p, eps_of(p))["look_mean"]) for p in fam_paths(pol_root / "eval_pol_val", fam)]
    for t in targets:
      name, look = min(cands, key=lambda c: abs(c[1] - t))
      chosen[f"{fam}@{t}"] = {"name": name, "val_look": look}
  return chosen


def acoustic_rtf(d):
  recs = [json.loads(l) for f in sorted(Path(d).glob("commits*.jsonl")) for l in open(f)]
  if not recs or "acoustic_s" not in recs[0]:
    return None
  return float(sum(r["acoustic_s"] for r in recs) / sum(r["audio_s"] for r in recs))


def audio(d):
  f = Path(d) / "audio_metrics.json"
  return json.load(open(f)) if f.exists() else None


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--stage", required=True, choices=["select", "final"])
  ap.add_argument("--final", default=str(ROOT / "runs/final"))
  ap.add_argument("--all", default=str(ROOT / "runs/all_schemes"))
  ap.add_argument("--targets", default="0.25,0.5")
  args = ap.parse_args()
  final, al = Path(args.final), Path(args.all)
  ctx = Ctx(final)
  targets = [float(t) for t in args.targets.split(",")]
  if args.stage == "select":
    out = al / "chosen_policies.json"
    if out.exists():
      raise FileExistsError(out)
    chosen = select(ctx, final, targets)
    json.dump(chosen, open(out, "w"), indent=1)
    print(json.dumps(chosen, indent=1))
    return

  out_path = al / "results_all.json"
  if out_path.exists():
    raise FileExistsError(out_path)
  v2 = json.load(open(final / "results_v2.json"))
  res = {"acoustic": {}, "planner": {}, "policies": {}, "theory": {}}

  base = final / "eval_render_test/s6_future_fixed1"
  ref_rows = {"e0_offline": final / "eval_render_test/e0_offline", "e1_k1": final / "eval_render_test/e1_fixed1",
              "vits(S6-future k1)": base}
  for k, d in ref_rows.items():
    res["acoustic"][k] = {"audio": audio(d), "rtf": acoustic_rtf(al / "render_test/vits") if k.startswith("vits") else None}
  for s in ["s5", "t4", "s2", "s3", "s4"]:
    d = al / "render_test" / s
    if audio(d) is None:
      continue
    res["acoustic"][s] = {"audio": audio(d), "rtf": acoustic_rtf(d), "vs_vits": paired(base, d),
                          "vs_e0_offline": paired(final / "eval_render_test/e0_offline", d)}

  for k in [0, 1]:
    s1, s6 = al / f"eval_test/s1_fixed{k}", final / f"eval_test/s6_future_fixed{k}"
    if s1.exists():
      res["planner"][f"k{k}"] = {"s1": ctx.m(s1), "s6_future": ctx.m(s6), "s1_minus_s6_future": ctx.pair(s6, s1),
                                 "s1_minus_e1": ctx.pair(final / f"eval_test/e1_fixed{k}", s1)}
  s1r = al / "eval_render_test/s1_fixed1"
  if audio(s1r):
    res["planner"]["audio_k1"] = {"s1": audio(s1r), "vs_s6_future": paired(base, s1r)}

  chosen = json.load(open(al / "chosen_policies.json"))
  for key, c in chosen.items():
    fam, t = key.split("@")
    tp = final / "eval_pol_test" / c["name"]
    e = eps_of(tp)
    entry = {"threshold": c["name"], "val_look": c["val_look"], "test": ctx.m(tp, e)}
    t3p = Path(v2["policies"][f"t3@{t}"]["path"])
    entry["vs_t3"] = ctx.pair(t3p, tp)
    entry["vs_entropy"] = ctx.pair(Path(v2["policies"][f"entropy@{t}"]["path"]), tp)
    entry["vs_e1_k1"] = ctx.pair(final / "eval_test/e1_fixed1", tp)
    rd = final / "eval_render_pol_test" / c["name"]
    if audio(rd):
      entry["audio"] = audio(rd)
      entry["audio_vs_e1_k1"] = paired(final / "eval_render_test/e1_fixed1", rd)
    res["policies"][key] = entry
  for key in ["t3@0.25", "t3@0.5", "t3_lite@0.25", "t3_lite@0.5", "entropy@0.25", "entropy@0.5", "spread@0.25", "spread@0.5"]:
    res["policies"][key] = {"threshold": v2["policies"][key]["threshold"], "test": v2["policies"][key]["test"]}

  for k, f in [("t1_audit", al / "policies/t1_audit.json"), ("t2_certificate", al / "policies/t2_certificate.json")]:
    if f.exists():
      d = json.load(open(f))
      res["theory"][k] = d.get("summary", d)
  json.dump(res, open(out_path, "w"), indent=1)
  print(json.dumps(res, indent=1)[:20000])


if __name__ == "__main__":
  main()

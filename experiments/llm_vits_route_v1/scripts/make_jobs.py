"""Write a jobs.json of run_stream_eval / offline_plans commands for one experiment stage.

  python make_jobs.py --stage fixed --exp runs/final --models e1,e2,s6 --split test --ks 0,1,2,3 --out jobs.json
Stages:
  fixed   every model x fixed:K (+ the E0 offline reference)
  dump    T3 feature dumps of one model (train subset and calib), probe = e1
  sweep   threshold sweeps for entropy / spread / t3 on one model
Model checkpoints are read from <exp>/<variant>/best.json.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = "/opt/dlami/nvme/jasonleeeli/conda_envs/uniss-train/bin/python"


def ckpt(exp, v):
  best = json.load(open(Path(exp) / v / "best.json"))["best_step"]
  return str(Path(exp) / v / f"stream_{best}.pth")


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--stage", required=True, choices=["fixed", "dump", "sweep"])
  ap.add_argument("--exp", required=True)
  ap.add_argument("--backbone-ckpt", required=True)
  ap.add_argument("--models", default="e1,e2,s6")
  ap.add_argument("--model", default="s6")
  ap.add_argument("--split", default="test")
  ap.add_argument("--ks", default="0,1,2,3")
  ap.add_argument("--shards", type=int, default=4)
  ap.add_argument("--train-limit", type=int, default=1500)
  ap.add_argument("--entropy", default="")
  ap.add_argument("--spread", default="")
  ap.add_argument("--t3", default="")
  ap.add_argument("--risk", default="")
  ap.add_argument("--gamma", default="", help="T1 thresholds (needs --t1-head)")
  ap.add_argument("--ot", default="", help="T2 thresholds (needs --t2-head)")
  ap.add_argument("--t5", default="", help="T5 LAM or LAM:EPS values, comma separated (needs --t5-head)")
  ap.add_argument("--t1-head", default="")
  ap.add_argument("--t2-head", default="")
  ap.add_argument("--t5-head", default="")
  ap.add_argument("--render", action="store_true")
  ap.add_argument("--noise-w", type=float, default=0.0)
  ap.add_argument("--tag", default="")
  ap.add_argument("--out", required=True)
  a = ap.parse_args()
  exp = Path(a.exp).resolve()
  base = ["--backbone-config", str(ROOT / "configs/ljs_char_base.json"), "--backbone-ckpt", a.backbone_ckpt,
          "--noise-w", str(a.noise_w)]
  ev = str(exp / f"eval{a.tag}_{a.split}")
  jobs = []

  def add(name, extra, shards=a.shards, split=a.split, out_root=ev, render=a.render):
    out = f"{out_root}/{name}"
    for s in range(shards):
      cmd = [PY, "scripts/run_stream_eval.py", *base, *extra, "--split", split, "--out", out]
      if shards > 1:
        cmd += ["--shard", str(s), "--num-shards", str(shards)]
      if render:
        cmd.append("--render")
      done = f"{out}/commits.shard{s}.jsonl" if shards > 1 else f"{out}/commits.jsonl"
      jobs.append({"id": f"{Path(out_root).name}-{name}-{s}", "cmd": cmd, "done": done})

  if a.stage == "fixed":
    for v in a.models.split(","):
      for k in a.ks.split(","):
        add(f"{v}_fixed{k}", ["--model", ckpt(exp, v), "--policy", f"fixed:{k}"])
    off = f"{ev}/e0_offline"
    cmd = [PY, "scripts/offline_plans.py", "--backbone-config", str(ROOT / "configs/ljs_char_base.json"),
           "--backbone-ckpt", a.backbone_ckpt, "--split", a.split, "--noise-w", str(a.noise_w), "--out", off]
    if a.render:
      cmd.append("--render")
    jobs.append({"id": f"{Path(ev).name}-e0_offline", "cmd": cmd, "done": f"{off}/commits.jsonl"})
  elif a.stage == "dump":
    m = ["--model", ckpt(exp, a.model), "--probe", ckpt(exp, "e1"), "--policy", "dump"]
    add(f"dump_{a.model}_train", m + ["--limit", str(a.train_limit)], split="train", out_root=str(exp / f"t3{a.tag}"), render=False)
    add(f"dump_{a.model}_calib", m, split="calib", out_root=str(exp / f"t3{a.tag}"), render=False)
  else:
    m = ["--model", ckpt(exp, a.model)]
    pr = ["--probe", ckpt(exp, "e1")]
    for t in filter(None, a.entropy.split(",")):
      add(f"{a.model}_entropy_{t}", m + ["--policy", f"entropy:{t}"])
    for t in filter(None, a.spread.split(",")):
      add(f"{a.model}_spread_{t}", m + pr + ["--policy", f"spread:{t}"])
    for t in filter(None, a.t3.split(",")):
      add(f"t3_{a.model}_{t}", m + pr + ["--policy", f"t3:{t}", "--risk", a.risk])
    for t in filter(None, a.gamma.split(",")):
      add(f"t1_{a.model}_{t}", m + pr + ["--policy", f"gamma:{t}", "--policy-head", a.t1_head])
    for t in filter(None, a.ot.split(",")):
      add(f"t2_{a.model}_{t}", m + pr + ["--policy", f"ot:{t}", "--policy-head", a.t2_head])
    for t in filter(None, a.t5.split(",")):
      add(f"t5_{a.model}_{t.replace(':', '_')}", m + pr + ["--policy", f"t5:{t}", "--policy-head", a.t5_head, "--risk", a.risk])
  if Path(a.out).exists():
    sys.exit(f"refusing to overwrite {a.out}")
  json.dump(jobs, open(a.out, "w"), indent=1)
  print(f"{len(jobs)} jobs -> {a.out}")


if __name__ == "__main__":
  main()

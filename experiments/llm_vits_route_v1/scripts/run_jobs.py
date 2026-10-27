"""Run a list of evaluation commands over GPUs, several processes per card; a job is skipped when its done-file exists.

  python run_jobs.py jobs.json --cards 0,1,2 --per-card 3 --log-dir runs/logs/jobs
jobs.json: [{"id": str, "cmd": [argv...], "done": path}]; each command runs from the experiment root.
"""
import argparse
import json
import os
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("jobs")
  ap.add_argument("--cards", required=True)
  ap.add_argument("--per-card", type=int, default=2)
  ap.add_argument("--log-dir", required=True)
  args = ap.parse_args()
  jobs = [j for j in json.load(open(args.jobs)) if not Path(j["done"]).exists()]
  slots = [c for c in args.cards.split(",") for _ in range(args.per_card)]
  log_dir = Path(args.log_dir)
  log_dir.mkdir(parents=True, exist_ok=True)
  running = {}
  while jobs or running:
    for s, (j, p) in list(running.items()):
      if p.poll() is not None:
        print(f"{time.strftime('%H:%M:%S')} {'done' if p.returncode == 0 else 'FAIL'} {j['id']}", flush=True)
        del running[s]
    for s in range(len(slots)):
      if s not in running and jobs:
        j = jobs.pop(0)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=slots[s], OMP_NUM_THREADS="4", NUMBA_NUM_THREADS="4")
        log = open(log_dir / f"{j['id']}.log", "w")
        running[s] = (j, subprocess.Popen(j["cmd"], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT))
        print(f"{time.strftime('%H:%M:%S')} start {j['id']} on card {slots[s]}", flush=True)
    time.sleep(5)
  print("ALL_DONE", flush=True)


if __name__ == "__main__":
  main()

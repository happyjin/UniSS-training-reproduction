"""Paper v7: every failed decode job of queue2 (stopped by the memory watchdog, or any other failure) whose windows are
not all decoded is queued again once, through requeue_rest.py (remaining windows only, batch 16, new output files).
Run by run_after_v7.sh every round; requeued ids are listed in logs/multispk/v7/fu/requeued.txt. A job is requeued at
most 3 times along its chain (<id>_r1_r1_r1); after that it is left for a human look.
"""
import glob
import json
import subprocess

Q = "logs/multispk/v7/queue3"   # writes here; reads the job lists of queue2 and queue3
QS = ["logs/multispk/v7/queue2", "logs/multispk/v7/queue3"]
F = "logs/multispk/v7/fu/requeued.txt"
state = {}
for q in QS:
    try:
        state.update(json.load(open(f"{q}/state.json")))
    except FileNotFoundError:
        pass
jobs = {}
for f in sorted(g for q in QS for g in glob.glob(f"{q}/add/*.json")):
    for j in json.load(open(f)):
        jobs.setdefault(j["id"], j)
try:
    done = set(open(F).read().split())
except FileNotFoundError:
    done = set()
todo = [k for k, v in state.items() if v == "fail" and k not in done and jobs.get(k, {}).get("kind") == "decode"
        and "--windows" in jobs[k]["cmd"] and k.count("_r") < 3]
if todo:
    n = len(glob.glob("reports/gstar_si_v7/jobs/auto_rq*.json"))
    r = subprocess.run(["python3", "reports/gstar_si_v7/requeue_rest.py", f"auto_rq{n:03d}", "--batch", "8"] + todo,
                       capture_output=True, text=True)
    print(r.stdout, r.stderr)
    if r.returncode == 0:
        with open(F, "a") as fh:
            fh.write("\n".join(todo) + "\n")

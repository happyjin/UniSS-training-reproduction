"""Paper v7 queue maintenance: re-add every waiting job of queue2 without its "cards" lane (new id <id>_x, deps renamed
when they point to another re-added job), after writing the old ids to the queue's cancel file. Used when the training
lane has no training to run (the seed-3 training was stopped by the watchdog on 2026-11-02 03:36 UTC when a third
user's job came onto every card). Writes reports/gstar_si_v7/jobs/<name>.json and the queue's add/ file (mode "x").

    python3 reports/gstar_si_v7/relane.py <name>
"""
import glob
import json
import os
import sys
import time

Q = "logs/multispk/v7/queue3"
name = sys.argv[1]
state = json.load(open(f"{Q}/state.json"))
jobs = {}
for f in sorted(glob.glob(f"{Q}/add/*.json")):
    for j in json.load(open(f)):
        jobs.setdefault(j["id"], j)
wait = [k for k, v in state.items() if v == "wait" and not k.startswith("train_") and not k.startswith("qabc_")]   # QABC v11 jobs are never touched
with open(f"{Q}/cancel", "a") as fh:
    fh.write("\n".join(wait) + "\n")
time.sleep(20)   # the queue reads the cancel file every 15 s
state = json.load(open(f"{Q}/state.json"))
assert all(state[k] == "cancel" for k in wait), [k for k in wait if state[k] != "cancel"]
new = []
for k in wait:
    j = dict(jobs[k])
    j.pop("cards", None)
    j["id"] = k + "_x"
    j["deps"] = [d + "_x" if d in wait else d for d in j["deps"]]
    new.append(j)
for p in (f"reports/gstar_si_v7/jobs/{name}.json", f"{Q}/add/{len(glob.glob(Q + '/add/*.json')):03d}_{name}.json"):
    with open(p, "x") as fh:
        json.dump(new, fh, indent=1)
print(len(new), "jobs re-added without lanes")

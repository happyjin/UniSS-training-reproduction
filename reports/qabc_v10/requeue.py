"""The jobs of gpuq7/gpuq8 job files that no queue log shows as started (or that failed, with --failed), into a new job
file -- to restart a queue on other cards after stopping its process by PID (running jobs keep running). A failed job
is re-queued only if its output directory is gone or empty (the runs refuse to overwrite), so check before using --failed.

    python reports/qabc_v10/requeue.py <out.json> --jobs a.json b.json --logs q1_queue.log q4_queue.log [--failed]
"""
import argparse
import json
import re

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--jobs", nargs="+", required=True)
ap.add_argument("--logs", nargs="+", required=True)
ap.add_argument("--failed", action="store_true")
a = ap.parse_args()
started, failed = set(), set()
for f in a.logs:
    for line in open(f):
        m = re.match(r"\S+ (start|fail) (\S+)", line)
        if m:
            (started if m.group(1) == "start" else failed).add(m.group(2))
jobs, seen = [], set()
for f in a.jobs:
    for j in json.load(open(f)):
        if j["id"] in seen:
            continue
        seen.add(j["id"])
        if j["id"] not in started or (a.failed and j["id"] in failed):
            jobs.append(j)
with open(a.out, "x") as fh:
    json.dump(jobs, fh, indent=1)
print(len(jobs), "jobs ->", a.out, "| started", len(started), "failed", len(failed))

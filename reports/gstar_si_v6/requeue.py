"""Paper v6: keep ONE gpuq7 queue for this branch. Given the running queue's job file and log, writes <out> with its
jobs that have not started yet (dependencies on started jobs dropped) plus the jobs of any extra job files; the caller
then stops the old queue by PID and starts <out>. New file only (mode "x").

    python3 reports/gstar_si_v6/requeue.py <old jobs.json> <old logdir> <out jobs.json> [<extra jobs.json> ...]
"""
import json
import re
import sys

old, logdir, out, *extra = sys.argv[1:]
started = set()
for line in open(f"{logdir}/queue.log"):
    m = re.search(r" (?:start|done|fail|skip) (\S+)", line)
    if m:
        started.add(m.group(1))
jobs = [j for j in json.load(open(old)) if j["id"] not in started]
for f in extra:
    jobs += json.load(open(f))
ids = {j["id"] for j in jobs}
for j in jobs:
    j["deps"] = [d for d in j["deps"] if d in ids]
with open(out, "x") as fh:
    json.dump(jobs, fh, indent=1)
print(len(jobs), "jobs ->", out)

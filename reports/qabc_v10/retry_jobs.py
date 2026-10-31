"""Re-runs of v10 jobs that failed (e.g. a vLLM engine that failed its memory profile while another started on the card).
A speech job writes to a new folder system_<name>[-seed<k>]-r<n>_m<w> (the failed one is kept as it is; the loaders take
the most recent complete run of a window and talker seed); any other job is re-queued unchanged, which is safe only
because its outputs were never written (the runs refuse existing files).

    python reports/qabc_v10/retry_jobs.py <out.json> <n> --jobs a.json ... --ids id1 id2 ...
"""
import argparse
import json
import re

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("n", type=int)
ap.add_argument("--jobs", nargs="+", required=True)
ap.add_argument("--ids", nargs="+", required=True)
a = ap.parse_args()
alljobs = {}
for f in a.jobs:
    for j in json.load(open(f)):
        alljobs.setdefault(j["id"], j)
out = []
for i in a.ids:
    j = dict(alljobs[i])
    if i.startswith("sp_"):
        cmd, k = re.subn(r"(speech/system_[^/ ]+?)(_m[\d.]+)", rf"\1-r{a.n}\2", j["cmd"])
        assert k >= 2, i
        j["cmd"], j["id"] = cmd, f"{i}-r{a.n}"
    out.append(j)
with open(a.out, "x") as fh:
    json.dump(out, fh, indent=1)
print(len(out), "jobs ->", a.out)

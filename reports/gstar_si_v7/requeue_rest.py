"""Paper v7 queue maintenance: for decode jobs that failed (e.g. stopped by the memory watchdog), queue the windows of
their shard that no hyp_s*.jsonl of their output directory holds yet, as one new job (id <id>_r<n>, output
hyp_s<k>r<n>.jsonl, windows <dir>/todo_<id>_r<n>.windows.jsonl; same command otherwise, batch optionally lowered).
Nothing existing is changed; the job list also goes to reports/gstar_si_v7/jobs/<name>.json.

    python3 reports/gstar_si_v7/requeue_rest.py <name> [--batch 16] <failed job id> ...
"""
import glob
import json
import re
import sys

Q = "logs/multispk/v7/queue3"   # writes here; reads the job lists of queue2 and queue3
QS = ["logs/multispk/v7/queue2", "logs/multispk/v7/queue3"]
args = sys.argv[1:]
name = args.pop(0)
batch = None
if args and args[0] == "--batch":
    batch = args[1]
    args = args[2:]
jobs = {}
for f in sorted(g for q in QS for g in glob.glob(f"{q}/add/*.json")):
    for j in json.load(open(f)):
        jobs.setdefault(j["id"], j)
new = []
for jid in args:
    j = dict(jobs[jid])
    cmd = j["cmd"]
    win = re.search(r"--windows (\S+)", cmd).group(1)
    out = re.search(r"--out (\S+)", cmd).group(1)
    k, n = int(re.search(r"--shard (\d+)", cmd).group(1)), int(re.search(r"--num-shards (\d+)", cmd).group(1))
    d = out.rsplit("/", 1)[0]
    done = set()
    for f in glob.glob(f"{d}/hyp_s*.jsonl"):
        for l in open(f, encoding="utf-8"):
            try:
                done.add(json.loads(l)["id"])
            except ValueError:
                pass
    W = [l for l in open(win, encoding="utf-8")][k::n]
    rest = [l for l in W if json.loads(l)["id"] not in done]
    r = 1
    while glob.glob(f"{d}/todo_{jid}_r{r}.windows.jsonl"):
        r += 1
    tw = f"{d}/todo_{jid}_r{r}.windows.jsonl"
    with open(tw, "x", encoding="utf-8") as fh:
        fh.writelines(rest)
    o2 = out.replace(".jsonl", f"r{r}.jsonl")
    cmd = cmd.replace(f"--windows {win}", f"--windows {tw}").replace(f"--out {out}", f"--out {o2}")
    cmd = cmd.replace(f"--shard {k} --num-shards {n}", "--shard 0 --num-shards 1")
    cmd = re.sub(r"> (\S+)\.log 2>&1", lambda m: f"> {m.group(1)}r{r}.log 2>&1", cmd)
    if batch:
        cmd = re.sub(r"--batch \d+", f"--batch {batch}", cmd)
    j.update(id=f"{jid}_r{r}", cmd=cmd, deps=[])
    j.pop("cards", None)
    new.append(j)
    print(jid, "->", j["id"], len(rest), "windows")
for p in (f"reports/gstar_si_v7/jobs/{name}.json", f"{Q}/add/{len(glob.glob(Q + '/add/*.json')):03d}_{name}.json"):
    with open(p, "x") as fh:
        json.dump(new, fh, indent=1)

"""Paper v7 queue maintenance: for a failed single-card shared-Thinker job (gstar_multi_v6 runs file), queue the windows
of its runs that no hyp_s*.jsonl / hyp_*.jsonl of the output directory holds yet, as one new job with W = 3 workers (less
memory than the 5 that took a card to 137 GB), batch 8, outputs hyp_<id>r<n>m<k>.jsonl. Nothing existing is changed.

    python3 reports/gstar_si_v7/requeue_multi.py <name> <failed job id> ...
"""
import glob
import json
import re
import sys

Q = "logs/multispk/v7/queue3"
QS = ["logs/multispk/v7/queue2", "logs/multispk/v7/queue3"]
name, ids = sys.argv[1], sys.argv[2:]
jobs = {}
for f in sorted(g for q in QS for g in glob.glob(f"{q}/add/*.json")):
    for j in json.load(open(f)):
        jobs.setdefault(j["id"], j)
new = []
for jid in ids:
    j = jobs[jid]
    rf = j["cmd"].split()[-1]
    runs = json.load(open(rf))
    a = runs[0]["argv"]
    win = a[a.index("--windows") + 1]
    d = a[a.index("--out") + 1].rsplit("/", 1)[0]
    th = a[a.index("--thinker") + 1]
    done = set()
    for f in glob.glob(f"{d}/hyp_*.jsonl"):
        for l in open(f, encoding="utf-8"):
            try:
                done.add(json.loads(l)["id"])
            except ValueError:
                pass
    left = [l for l in open(win, encoding="utf-8") if json.loads(l)["id"] not in done]
    r = 1
    while glob.glob(f"{d}/todo_{jid}_r{r}.windows.jsonl"):
        r += 1
    tw = f"{d}/todo_{jid}_r{r}.windows.jsonl"
    with open(tw, "x", encoding="utf-8") as fh:
        fh.writelines(left)
    W = 3
    base = [x for x in a]
    nr = []
    for k in range(W):
        b = list(base)
        b[b.index("--windows") + 1] = tw
        b[b.index("--out") + 1] = f"{d}/hyp_{jid}r{r}m{k}.jsonl"
        b[b.index("--shard") + 1] = str(k)
        b[b.index("--num-shards") + 1] = str(W)
        b[b.index("--batch") + 1] = "8"
        nr.append({"module": runs[0]["module"], "argv": b, "log": f"{d}/infer_{jid}r{r}m{k}.log"})
    nf = rf.replace(".json", f"_r{r}.json")
    with open(nf, "x") as fh:
        json.dump(nr, fh, indent=1)
    cmd = re.sub(r"\S+\.json$", nf, j["cmd"])
    new.append({"id": f"{jid}_r{r}", "gpus": 1, "mem": 72000 + 7000 * W + 10000, "kind": "decode", "deps": [], "cmd": cmd})
    print(jid, "->", f"{jid}_r{r}", len(left), "windows")
for p in (f"reports/gstar_si_v7/jobs/{name}.json", f"{Q}/add/{len(glob.glob(Q + '/add/*.json')):03d}_{name}.json"):
    with open(p, "x") as fh:
        json.dump(new, fh, indent=1)

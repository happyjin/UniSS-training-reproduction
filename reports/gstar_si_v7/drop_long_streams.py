"""Paper v7 (prereg_v7 amendment 1): copy a Talker input dir (blocks.jsonl, items.jsonl, voice dirs by symlink) leaving
out streams with more than N blocks (a runaway generation; v6: 473 blocks exceeded the Talker's context). The left-out
stream ids go to <out>/DROPPED.txt ("<id>: <n> blocks ..."). The input dir is not changed.

    python3 reports/gstar_si_v7/drop_long_streams.py <talker_in_all> <talker_in> 400
"""
import json
import os
import sys

src, dst, n_max = sys.argv[1], sys.argv[2], int(sys.argv[3])
os.makedirs(dst, exist_ok=False)
blocks = [json.loads(l) for l in open(f"{src}/blocks.jsonl", encoding="utf-8")]
drop = {b["id"]: len(b["blocks"]) for b in blocks if len(b["blocks"]) > n_max}
for name in ("blocks.jsonl", "items.jsonl"):
    with open(f"{dst}/{name}", "x", encoding="utf-8") as fh:
        for l in open(f"{src}/{name}", encoding="utf-8"):
            if json.loads(l)["id"] not in drop:
                fh.write(l)
for e in os.listdir(src):
    if e not in ("blocks.jsonl", "items.jsonl"):
        os.symlink(os.path.abspath(f"{src}/{e}"), f"{dst}/{e}")
with open(f"{dst}/DROPPED.txt", "x") as fh:
    for k, v in drop.items():
        fh.write(f"{k}: {v} blocks; more than {n_max} blocks (runaway generation); not synthesised\n")
print(f"{len(blocks) - len(drop)} streams kept, {len(drop)} dropped")

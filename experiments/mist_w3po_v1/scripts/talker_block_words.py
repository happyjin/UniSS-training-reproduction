"""Talker-fragment risk (user 2026-11-11, T4 vs T5): words per unit that reaches the talker, from a decode's hyp.jsonl.
  segment   one Thinker segment (hyp "segments": what the talker plays as one piece)
  read_spk  the segments of one speaker written at one read, merged (T3's A6 unit)
Reports mean / p10 / p50 / p90 and the shares of 1-word and <= 2-word units, over the windows of <windows.jsonl>.
    python talker_block_words.py <hyp.jsonl> <windows.jsonl>"""
import json
import re
import sys
from collections import defaultdict

import numpy as np

TAG = re.compile(r"<\|[^|]*\|>")


def nw(s: str) -> int:
    return len(TAG.sub(" ", s).split())


def stats(x):
    x = np.asarray([v for v in x if v > 0])
    return {"n": int(len(x)), "mean": round(float(x.mean()), 3), "p10": float(np.percentile(x, 10)),
            "p50": float(np.median(x)), "p90": float(np.percentile(x, 90)), "share_1w": round(float((x == 1).mean()), 4),
            "share_le2w": round(float((x <= 2).mean()), 4)} if len(x) else {"n": 0}


def main():
    ids = {json.loads(l)["id"] for l in open(sys.argv[2], encoding="utf-8")}
    seg, grp = [], []
    for l in open(sys.argv[1], encoding="utf-8"):
        r = json.loads(l)
        if r["id"] not in ids:
            continue
        g = defaultdict(int)
        for s in r.get("segments", []):
            n = nw(s["text"])
            seg.append(n)
            g[(s["write_ms"], s["spk"])] += n
        grp += list(g.values())
    print(json.dumps({"segment": stats(seg), "read_spk": stats(grp)}))


if __name__ == "__main__":
    main()

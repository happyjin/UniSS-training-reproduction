"""Where the contextual alignment cuts the source into prefixes.

English sources: after every word. Chinese sources: the forced aligner's units are
characters; prefixes end after every jieba word instead, which halves the prefixes
(each a full Thinker pass) without coarsening the timing -- a word is aligned to
its last character. bounds[k] = how many source units prefix k holds.

    python prefix_bounds.py --meta data/unist_clean_v1/meta --keys keys.parquet --out bounds.parquet
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from pathlib import Path

import jieba
import pyarrow as pa
import pyarrow.parquet as pq


def one(job):
    path, keys = job
    jieba.setLogLevel(60)
    out = []
    for r in pq.read_table(path, columns=["key", "src_lang", "source_words"]).to_pylist():
        if r["key"] not in keys:
            continue
        units = [w["text"] for w in json.loads(r["source_words"])]
        bounds = list(range(len(units) + 1))
        if r["src_lang"] in ("cmn", "zh") and all(len(u) == 1 for u in units):
            bounds, pos = [0], 0
            for w in jieba.cut("".join(units)):
                pos += len(w)
                bounds.append(pos)
        out.append({"key": r["key"], "bounds": json.dumps(bounds)})
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--keys", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--jobs", type=int, default=30)
    args = ap.parse_args()
    if Path(args.out).exists():
        raise SystemExit(f"拒绝覆盖已有输出: {args.out}")
    keys = set(pq.read_table(args.keys, columns=["key"]).column("key").to_pylist())
    rows = []
    with mp.Pool(args.jobs) as pool:
        for part in pool.imap_unordered(one, [(str(p), keys) for p in sorted(Path(args.meta).glob("*.parquet"))]):
            rows += part
    pq.write_table(pa.Table.from_pylist(rows), args.out)
    print(f"{len(rows)} rows -> {args.out}", flush=True)


if __name__ == "__main__":
    main()

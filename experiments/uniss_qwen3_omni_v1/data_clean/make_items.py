"""The ASR work list for both cleaning plans, pilot rows first.

Pilot: 2,000 rows -- 500 Chinese targets with Latin words (code-switched),
500 pure Chinese targets, 1,000 English targets -- drawn at random across all
30 chunks. Items are (key, side, path) for the target and the source audio of
every row; the pilot's come first, so with the list sharded over the GPUs each
shard's first batch carries its share of the pilot.

    python make_items.py --meta data/unist_clean_v1/meta --out data/unist_clean_v1
"""

from __future__ import annotations

import argparse
import random
import re
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

LATIN = re.compile(r"[A-Za-z]{2,}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=20261021)
    args = ap.parse_args()
    out = Path(args.out)
    for name in ("pilot_rows.parquet", "asr_items.parquet"):
        if (out / name).exists():
            raise SystemExit(f"拒绝覆盖已有输出: {out / name}")
    cols = ["key", "tgt_lang", "translation", "source_audio", "target_audio"]
    rows = pa.concat_tables([pq.read_table(f, columns=cols) for f in sorted(Path(args.meta).glob("*.parquet"))]).to_pylist()
    groups = {"zh_codeswitch": [], "zh_pure": [], "en": []}
    for r in rows:
        if r["tgt_lang"] in ("cmn", "zh"):
            groups["zh_codeswitch" if LATIN.search(r["translation"]) else "zh_pure"].append(r)
        else:
            groups["en"].append(r)
    rng = random.Random(args.seed)
    want = {"zh_codeswitch": 500, "zh_pure": 500, "en": 1000}
    pilot = []
    for g, n in want.items():
        for r in rng.sample(groups[g], n):
            pilot.append({"key": r["key"], "group": g})
    print({g: len(v) for g, v in groups.items()}, "rows;", len(pilot), "pilot rows", flush=True)
    pilot_keys = {p["key"] for p in pilot}
    order = [p["key"] for p in pilot] + [r["key"] for r in rows if r["key"] not in pilot_keys]
    by_key = {r["key"]: r for r in rows}
    items = {"key": [], "side": [], "path": []}
    for k in order:
        for side, col in (("tgt", "target_audio"), ("src", "source_audio")):
            items["key"].append(k); items["side"].append(side); items["path"].append(by_key[k][col])
    out.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(pilot), out / "pilot_rows.parquet")
    pq.write_table(pa.table(items), out / "asr_items.parquet")
    print(f"{len(items['key'])} items written", flush=True)


if __name__ == "__main__":
    main()

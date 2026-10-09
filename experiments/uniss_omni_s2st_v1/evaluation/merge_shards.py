"""Merge sharded zero-shot runs and score the union under the repo protocol.

Each shard scores only its own slice, which is not the number anyone wants;
corpus BLEU is not an average of per-shard BLEUs. This concatenates the rows
and scores once.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from evaluation import text_metrics

LANGUAGE = {"en2zh": "cmn", "zh2en": "eng"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shards", nargs="+", required=True)
    ap.add_argument("--direction", choices=("en2zh", "zh2en"), required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    rows: list[dict] = []
    for path in args.shards:
        rows.extend(json.loads(Path(path).read_text(encoding="utf-8"))["rows"])
    seen: set[str] = set()
    unique = []
    for row in rows:
        if row["id"] in seen:
            continue
        seen.add(row["id"])
        unique.append(row)

    score = text_metrics.corpus_bleu(
        [r["hypothesis"] for r in unique],
        [r["reference"] for r in unique],
        language=LANGUAGE[args.direction],
    )
    report = {
        "direction": args.direction,
        "pairs": len(unique),
        "duplicates_dropped": len(rows) - len(unique),
        "bleu": score["score"],
        "sys_len": score["sys_len"],
        "ref_len": score["ref_len"],
        "bp": score["bp"],
        "empty": sum(1 for r in unique if not r["hypothesis"]),
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps({"report": report, "rows": unique}, ensure_ascii=False, indent=1),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=1))
    print(f"-> {out}")


if __name__ == "__main__":
    main()

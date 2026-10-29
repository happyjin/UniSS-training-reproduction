"""Listening measures of G-STAR-SI speech on the single interpreter channel (paper v3, review R4), from the stitched
transcript of gstar_talker (one entry per spoken block: speaker, source segment start, play start, duration, text).

  abrupt_per_min   pauses >= 0.5 s between two consecutive blocks of the same speaker in a window, after a block that does
                   not end at a clause boundary (a stop mid-phrase), per minute of translated speech; by speaker, not by
                   segment, since streaming models write one segment per chunk
  single_unit      share of blocks with one Chinese character or one English word
  lag_start_med    median time from the end of a source segment to the start of its first spoken block
  lag_end_med      median time from the end of a source segment to the end of its last spoken block

    python -m experiments.uniss_qwen3_omni_v1.evaluation.gstar_listen --transcript <stitched/transcript.jsonl> --out <json>
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path

CLAUSE_END = tuple(",.;:!?，。；：！？、…")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--transcript", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-pause", type=float, default=0.5)
    args = ap.parse_args()
    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    abrupt, speech_s, blocks, single = 0, 0.0, 0, 0
    starts, ends = [], []
    for line in open(args.transcript, encoding="utf-8"):
        t = json.loads(line)
        segs, spk = defaultdict(list), defaultdict(list)
        for b in t["segments"]:
            segs[(b["spk"], b["src_start_s"])].append(b)
            spk[b["spk"]].append(b)
            speech_s += b["dur_s"]
            blocks += 1
            txt = b["text"].strip()
            n = len(re.findall(r"[一-鿿]", txt)) or len(re.findall(r"[A-Za-z0-9']+", txt))
            single += n <= 1
        for bs in spk.values():
            bs = sorted(bs, key=lambda b: b["play_start_s"])
            for a, b in zip(bs, bs[1:]):
                gap = b["play_start_s"] - (a["play_start_s"] + a["dur_s"])
                if gap >= args.min_pause and not a["text"].strip().endswith(CLAUSE_END):
                    abrupt += 1
        for bs in segs.values():
            bs = sorted(bs, key=lambda b: b["play_start_s"])
            starts.append(bs[0]["play_start_s"] - bs[0]["src_end_s"])
            ends.append(bs[-1]["play_start_s"] + bs[-1]["dur_s"] - bs[-1]["src_end_s"])
    res = {"abrupt_per_min": round(abrupt / max(speech_s / 60, 1e-6), 3), "abrupt": abrupt,
           "speech_min": round(speech_s / 60, 2), "blocks": blocks, "single_unit": round(single / max(blocks, 1), 4),
           "lag_start_med": round(statistics.median(starts), 3) if starts else None,
           "lag_end_med": round(statistics.median(ends), 3) if ends else None}
    out.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res))


if __name__ == "__main__":
    main()

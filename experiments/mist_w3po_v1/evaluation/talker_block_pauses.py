"""Block and pause statistics of finished MIST talker runs (CPU, read-only; coordinator 2026-11-12 ~07:45 Prague: does
the read-level block input hurt how the output sounds?). Works on talker/results.jsonl only (no re-synthesis).

Per run (one results.jsonl = one TTS seed), over all streams (window x output speaker) without an error:
  blocks         number of talker blocks, forced (truncated) blocks and their share
  words/block    mean words per block; share of 1-word blocks (words = whitespace tokens with a letter or digit)
  block dur      mean / p50 / p90 block audio duration (dur_ms)
  pauses         gaps between consecutive blocks of one stream: start_ms[k+1] - (start_ms[k] + dur_ms[k]) (the talker
                 plays blocks in order, so a gap is silence inside that speaker's output); counted when >= --min-gap
                 (default 0.5 s), per minute of produced speech (sum of dur_ms) and as total seconds per minute:
                   all        every such gap
                   mid_sent   the earlier block does not end a sentence (last char not . ? !): a pause inside a
                              sentence = the "abrupt pause" proxy
                   after_sent the earlier block ends a sentence
    python -m experiments.mist_w3po_v1.evaluation.talker_block_pauses --talker <talker dir> [--talker ...] [--min-gap 0.5]
Prints one JSON line per talker dir.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

WORD = re.compile(r"[A-Za-z0-9]")


def n_words(text: str) -> int:
    return sum(1 for t in (text or "").split() if WORD.search(t))


def sent_end(text: str) -> bool:
    t = (text or "").rstrip().rstrip("\"')]")
    return bool(t) and t[-1] in ".?!"


def stats(rows: list[dict], min_gap_ms: float = 500.0) -> dict:
    nb = nf = n1 = 0
    words, durs = [], []
    speech_ms = 0.0
    gaps = {"all": [], "mid_sent": [], "after_sent": []}
    streams = 0
    for r in rows:
        if r.get("error"):
            continue
        bl = r.get("blocks") or []
        if not bl:
            continue
        streams += 1
        for k, b in enumerate(bl):
            nb += 1
            nf += bool(b.get("forced"))
            w = n_words(b.get("text", ""))
            words.append(w)
            n1 += w == 1
            durs.append(float(b.get("dur_ms") or 0))
            speech_ms += float(b.get("dur_ms") or 0)
            if k:
                p = bl[k - 1]
                g = float(b["start_ms"]) - (float(p["start_ms"]) + float(p.get("dur_ms") or 0))
                if g >= min_gap_ms:
                    gaps["all"].append(g)
                    gaps["after_sent" if sent_end(p.get("text", "")) else "mid_sent"].append(g)
    mins = speech_ms / 60000.0 if speech_ms else float("nan")
    d = np.array(durs) if durs else np.zeros(1)
    out = {"streams": streams, "blocks": nb, "forced_blocks": nf, "forced_share": round(nf / max(nb, 1), 4),
           "words_per_block": round(float(np.mean(words)) if words else 0.0, 3), "share_1w": round(n1 / max(nb, 1), 4),
           "block_dur_s": {"mean": round(float(d.mean()) / 1000, 3), "p50": round(float(np.percentile(d, 50)) / 1000, 3),
                           "p90": round(float(np.percentile(d, 90)) / 1000, 3)},
           "speech_min": round(mins, 2), "min_gap_s": min_gap_ms / 1000}
    for k, v in gaps.items():
        out[f"pauses_{k}"] = {"n": len(v), "per_min": round(len(v) / mins, 3), "sec_per_min": round(sum(v) / 1000 / mins, 3),
                              "mean_s": round(float(np.mean(v)) / 1000, 3) if v else 0.0}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--talker", action="append", required=True)
    ap.add_argument("--min-gap", type=float, default=0.5)
    a = ap.parse_args()
    for t in a.talker:
        rows = [json.loads(l) for l in open(Path(t) / "results.jsonl", encoding="utf-8")]
        print(json.dumps({"talker": t, **stats(rows, a.min_gap * 1000)}, ensure_ascii=False))


if __name__ == "__main__":
    main()

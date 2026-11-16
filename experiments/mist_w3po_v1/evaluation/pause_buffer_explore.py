"""Exploratory (descriptive, 2026-11-16; not a registered test): how much would a playout buffer reduce MIST's pauses,
and at what latency? Re-times the existing talker blocks (durations and audio unchanged; talker layout, write_ms
based), per stream (window x output speaker):

  none      start_n = max(write_n, end_{n-1})                      (= the recorded layout)
  sent(d)   the first block of a sentence (n = 0 or block n-1 ends in .?!。？！) waits d: max(write_n + d, end_{n-1})
  idle(d)   every block that would start on an idle stream waits d

Pauses: gaps >= 0.5 s between consecutive blocks of a stream per minute of speech (all / inside a sentence);
delay: word-weighted mean of start_n - write_n (s); end: mean over streams of the last block's end - its write (s).

    python -m experiments.mist_w3po_v1.evaluation.pause_buffer_explore --run <speech run> [--run ...] [--units char]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.mist_w3po_v1.talker import talker_rows as TR
from experiments.mist_w3po_v1.talker.tspa_precheck_p1 import sent_end

DS = (0.0, 0.25, 0.5, 0.75, 1.0, 1.5)


def schedule(bl, policy: str, d_ms: float):
    st, end = [], None
    for n, b in enumerate(bl):
        w = float(b["write_ms"])
        first = n == 0 or sent_end(bl[n - 1].get("text", ""))
        idle = end is None or w >= end
        hold = d_ms if (policy == "sent" and first) or (policy == "idle" and idle) else 0.0
        s = max(w + hold, end if end is not None else -1e9)
        st.append(s)
        end = s + float(b["dur_ms"])
    return st


def stats(results: Path, policy: str, d: float) -> dict:
    p = pm = 0
    sp = wd = ww = 0.0
    ends = []
    for line in open(results, encoding="utf-8"):
        r = json.loads(line)
        bl = r.get("blocks") or []
        if r.get("error") or not bl:
            continue
        st = schedule(bl, policy, 1000 * d)
        for n, b in enumerate(bl):
            W = max(1, TR.n_units(b.get("text", "")))
            wd += W * (st[n] - float(b["write_ms"]))
            ww += W
            sp += float(b["dur_ms"])
            if n:
                g = st[n] - (st[n - 1] + float(bl[n - 1]["dur_ms"]))
                if g >= 500:
                    p += 1
                    pm += not sent_end(bl[n - 1].get("text", ""))
        ends.append(st[-1] + float(bl[-1]["dur_ms"]) - float(bl[-1]["write_ms"]))
    mins = sp / 60000
    return {"policy": policy, "d_s": d, "pauses_per_min": round(p / mins, 2), "mid_pauses_per_min": round(pm / mins, 2),
            "start_delay_s": round(wd / ww / 1000, 3), "end_lag_s": round(float(np.mean(ends)) / 1000, 3)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", action="append", required=True)
    ap.add_argument("--units", choices=("word", "char"), default="word")
    a = ap.parse_args()
    TR.UNITS = a.units
    for run in a.run:
        res = Path(run) / "talker" / "results.jsonl"
        rows = [stats(res, "none", 0.0)] + [stats(res, pol, d) for pol in ("sent", "idle") for d in DS[1:]]
        print(json.dumps({"run": Path(run).name, "rows": rows}, ensure_ascii=False))


if __name__ == "__main__":
    main()

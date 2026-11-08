"""Runaway-loop metric (fixed 2026-11-08, LOG): a dev window is a *runaway* when its output has more than
max(3 x ref words, ref words + 100) words, ref words = the words of the reference utterances that overlap the online
window by >= half their length (diag_reward_noise.window_utts). Reported per decode: the number of runaway windows,
their meetings, and tcpBLEU all / ov recomputed from score_text.json's per-meeting statistics with those meetings
left out (secondary row only; the formal gate is unchanged).

    python -m experiments.mist_w3po_v1.evaluation.loop_stats <decode dir> [...]
"""
from __future__ import annotations

import json
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np

from experiments.mist_w3po_v1.evaluation.compare import bleu_of
from experiments.mist_w3po_v1.evaluation.diag_reward_noise import WIN, ref_utts, window_utts


@lru_cache(maxsize=1)
def ref_words() -> dict:
    by = ref_utts()
    out = {}
    for w in map(json.loads, open(WIN, encoding="utf-8")):
        out[w["id"]] = (w["meeting"], sum(len(u["ref"].split()) for u in window_utts(w, by) if u.get("ref")))
    return out


def runaway(words: int, ref: int) -> bool:
    return words > max(3 * ref, ref + 100)


def loop_stats(d: Path) -> dict:
    rw = ref_words()
    bad = []
    for l in open(d / "hyp.jsonl", encoding="utf-8"):
        h = json.loads(l)
        m, r = rw.get(h["id"], ("", 0))
        n = len(h.get("translation", "").split())
        if runaway(n, r):
            bad.append({"id": h["id"], "meeting": m, "words": n, "ref_words": r})
    meet = sorted({b["meeting"] for b in bad})
    out = {"runaway_windows": len(bad), "runaway_meetings": meet, "windows": bad}
    sc = d / "score_text.json"
    if sc.exists():
        pm = json.loads(sc.read_text())["per_meeting"]
        keep = [m for m in pm if m not in meet]
        for reg in ("all", "ov"):
            st = [np.array(pm[m][f"{reg}_stats"] or [0] * 10, dtype=float) for m in keep]
            out[f"tcpBLEU_{reg}_excl_runaway_meetings"] = float(bleu_of(np.sum(st, 0))) if st else None
    return out


def main():
    for d in sys.argv[1:]:
        s = loop_stats(Path(d))
        print(Path(d).name, s["runaway_windows"], s["runaway_meetings"],
              s.get("tcpBLEU_all_excl_runaway_meetings"), s.get("tcpBLEU_ov_excl_runaway_meetings"))


if __name__ == "__main__":
    main()

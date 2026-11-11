"""Tests of crf/merge_read_segments.py (CPU):
  (a) toy: same-read runs per speaker merged (other speakers / empty entries in between do not break a run), different
      reads not merged, members ordered by start_s, end_s = max, write_ms kept, inputs untouched
  (b) real dev decodes (T2 0.5 s reads, A' 1.0 s reads): per window and speaker the word sequence of the segments (in
      the talker prep's (write_ms, start_s) order) and the multiset of (write_ms, words) are unchanged; no two
      remaining non-empty segments of a speaker share a write_ms consecutively; merged blocks pair with merged segments
      (spk, start_s, text) wherever the source's blocks pair with its segments; idempotent

    PYTHONPATH=. python -m experiments.mist_w3po_v1.crf.test_merge_read_segments
"""
from __future__ import annotations

import copy
import json
from collections import Counter, defaultdict

from experiments.mist_w3po_v1.crf import merge_read_segments as MR

DEC = "eval_outputs/mist_w3po_v1/decode"
RUNS = ["dev_t2_latent_t0_s3000_w05", "dev_crf_v2p_s3000_finalpost_s8_c2g06"]


def seg(spk, w, a, b, t):
    return {"spk": spk, "write_ms": w, "start_s": a, "end_s": b, "text": t}


def test_toy():
    S = [seg(1, 1000, 0.5, 0.9, "b c"), seg(2, 1000, 0.0, 0.4, "x"), seg(1, 1000, 0.0, 0.4, "a"),
         seg(1, 1000, 1.0, 1.2, " "), seg(1, 1000, 0.9, 1.0, "d."), seg(1, 2000, 1.2, 1.6, "e"),
         seg(2, 2000, 0.4, 0.8, "y"), seg(2, 2000, 0.8, 0.9, "z")]
    S0 = copy.deepcopy(S)
    out = MR.merge_entries(S, "end_s")
    assert S == S0
    ne = [(o["spk"], o["write_ms"], o["text"], o["start_s"], o["end_s"], o["_n"]) for o in out if o["text"].strip()]
    assert ne == [(1, 1000, "a b c d.", 0.0, 1.0, 3), (2, 1000, "x", 0.0, 0.4, 1), (1, 2000, "e", 1.2, 1.6, 1),
                  (2, 2000, "y z", 0.4, 0.9, 2)], ne
    assert [o["text"] for o in out][2] == " "            # the empty entry keeps its place among the run heads
    h = {"id": "w", "segments": S, "blocks": [{k: v for k, v in s.items() if k != "end_s"} for s in S]}
    h2, st = MR.merge_hyp(h)
    assert MR.check_pair(h2) and st["seg_in"] == 7 and st["seg_out"] == 4 and st["blk_out"] == 4
    print("  (a) toy: runs per speaker merged across other speakers / empties, ordered by start_s; pairing holds")


def order(segs, spk):
    xs = [s for s in segs if s["spk"] == spk and s["text"].strip()]
    return sorted(xs, key=lambda s: (s["write_ms"], s["start_s"] or 0.0))


def test_real():
    for run in RUNS:
        n = nm = pair_src = pair_ok = 0
        for l in open(f"{DEC}/{run}/hyp.jsonl", encoding="utf-8"):
            h = json.loads(l)
            h2, st = MR.merge_hyp(h)
            n += 1
            nm += st["seg_merged_members"]
            for k in {s["spk"] for s in h["segments"]}:
                a, b = order(h["segments"], k), order(h2["segments"], k)
                assert " ".join(s["text"].strip() for s in a).split() == " ".join(s["text"] for s in b).split(), h["id"]
                ca, cb = Counter(), Counter()
                for s in a:
                    ca[s["write_ms"]] += len(s["text"].split())
                for s in b:
                    cb[s["write_ms"]] += len(s["text"].split())
                assert ca == cb, h["id"]
                ws = [s["write_ms"] for s in h2["segments"] if s["spk"] == k and s["text"].strip()]
                assert all(x != y for x, y in zip(ws, ws[1:])), h["id"]
            if h.get("blocks"):
                src_pairs = MR.check_pair({"segments": h["segments"], "blocks": h["blocks"]})
                pair_src += src_pairs
                pair_ok += src_pairs and MR.check_pair(h2)
            h3, st3 = MR.merge_hyp(h2)
            assert st3["seg_merged_members"] == 0 and h3["segments"] == h2["segments"]
        assert pair_ok == pair_src, (run, pair_ok, pair_src)
        print(f"  (b) {run}: {n} windows, {nm} segments merged into their read's block; words / write times kept; "
              f"pairing kept in {pair_ok}/{pair_src} windows whose source blocks pair; idempotent")


def main():
    test_toy()
    test_real()
    print("ALL 2 PASSED")


if __name__ == "__main__":
    main()

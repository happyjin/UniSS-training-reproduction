"""Final test (coordinator 2026-11-12 ~22:20 Prague): per-window check of the read-level merge's lost pairings
(merge_read_segments.py, pair_lost). A window counts as the KNOWN mechanism when its source hyp pairs (check_pair) and,
among the source segment / block pairs of the same (speaker, start_s, text), at least one pair has a different write_ms
(a segment's write_ms is the read that saw its END marker, a block's the read that saw its TEXT), so the per-read merge
groups segments and blocks differently. Prints JSON: windows, pair_lost, share of the set's windows, per lost window
the mechanism flag and the first mismatching merged block. CPU, read-only.

    python -m experiments.mist_w3po_v1.evaluation.merge_pairlost_check --src <decode name> [--dst <merged name>]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.mist_w3po_v1.crf.merge_read_segments import check_pair

DEC = Path("eval_outputs/mist_w3po_v1/decode")


def mechanism(h: dict) -> bool:
    seg = {(s["spk"], s["start_s"], s["text"]): s.get("write_ms") for s in h.get("segments", []) if s["text"].strip()}
    return any(seg.get((b["spk"], b["start_s"], b["text"])) not in (None, b.get("write_ms"))
               for b in h.get("blocks", []) if b["text"].strip())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", default="")
    a = ap.parse_args()
    src = {json.loads(l)["id"]: json.loads(l) for l in open(DEC / a.src / "hyp.jsonl", encoding="utf-8")}
    lost = []
    for l in open(DEC / (a.dst or a.src + "_rl") / "hyp.jsonl", encoding="utf-8"):
        h = json.loads(l)
        h0 = src[h["id"]]
        if check_pair(h0) and not check_pair(h):
            seg = {(s["spk"], s["start_s"]): s["text"] for s in h["segments"] if s["text"].strip()}
            bad = next(((b["spk"], b["start_s"], b["text"][:80], seg.get((b["spk"], b["start_s"]), "<none>")[:80])
                        for b in h["blocks"] if b["text"].strip() and seg.get((b["spk"], b["start_s"])) != b["text"]), None)
            lost.append({"id": h["id"], "mechanism": mechanism(h0), "first_mismatch": bad})
    out = {"src": a.src, "windows": len(src), "pair_lost": len(lost), "share": round(len(lost) / len(src), 5),
           "all_known_mechanism": all(x["mechanism"] for x in lost), "lost": lost}
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()

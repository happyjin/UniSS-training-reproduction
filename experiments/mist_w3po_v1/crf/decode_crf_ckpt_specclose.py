"""SpecClose dev decode (reports/mist_w3po_v1/thinker_detector_specclose_prereg_v1.zh-CN.md §1.2): exactly
decode_crf_ckpt_finalpost --final-read-post close_detect, except that a window's FINAL read sees the posteriors P up to
the window jsonl's final_post_cut_ms (the frames the lowlat streaming preset has emitted at the window's close time,
data/specclose.py) instead of min(t, close_detect_ms - 1.04 s). Non-final reads: unchanged (t - 1.04 s).

Implementation: online_decode_finalpost.install with the per-window value final_post_cut_ms + LOOKAHEAD_MS, so its
final_cut_ms(t, v) = max(0, t - 1040, min(t, final_post_cut_ms)) -- no copy of the masking code. Every window row must
carry final_post_cut_ms. No --final-read-post flag (passing one is an error). The checkpoint is applied by
decode_crf_ckpt.main as for the finalpost decode.

    python -m experiments.mist_w3po_v1.crf.decode_crf_ckpt_specclose --crf-ckpt <run>/ckpt/step_NNNNN \
        --thinker <v8 merged> --windows <specclose windows jsonl> --out <new jsonl> <online_decode args>
"""
from __future__ import annotations

import json
import sys

from experiments.mist_w3po_v1.core import engine as E
from experiments.mist_w3po_v1.crf import decode_crf_ckpt as dc
from experiments.mist_w3po_v1.evaluation import online_decode_finalpost as fp


def load_cut(path: str) -> dict[str, int]:
    out = {}
    for w in map(json.loads, open(path, encoding="utf-8")):
        if "final_post_cut_ms" not in w:
            raise SystemExit(f"{path}: window {w.get('id')} has no final_post_cut_ms (not a SpecClose windows file)")
        out[w["id"]] = int(w["final_post_cut_ms"]) + E.LOOKAHEAD_MS
    return out


def main() -> None:
    rest = sys.argv[1:]
    if any(a == "--final-read-post" or a.startswith("--final-read-post=") for a in rest):
        raise SystemExit("decode_crf_ckpt_specclose takes no --final-read-post (the final-read cut is the windows' field)")
    fp.install(load_cut(fp.windows_arg(rest)))
    print("final-read-post specclose: final reads see P up to the windows' final_post_cut_ms", flush=True)
    sys.argv = [sys.argv[0]] + rest
    dc.main()
    print(f"FINALPOST_STATS {json.dumps(fp.STATS)}", flush=True)


if __name__ == "__main__":
    main()

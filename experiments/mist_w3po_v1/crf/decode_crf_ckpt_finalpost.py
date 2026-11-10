"""crf/decode_crf_ckpt.py + evaluation/online_decode_finalpost.py in one process (C0 / C0-fp dev eval, 2026-11-10):
a train_crf_sft checkpoint applied in memory to v8 merged AND the final-read causal P protocol. Neither wrapper is
changed; both patch different class attributes of core.engine.Engine and then hand the argv on:

  1. --final-read-post is taken out of the argv (online_decode_finalpost.split_argv); with close_detect,
     online_decode_finalpost.install wraps Engine.read (final reads see P up to min(t, close_detect - 1.04 s))
  2. decode_crf_ckpt.main takes --crf-ckpt out, wraps Engine.from_merged (LoRA fold, speaker projector, new token rows)
     and runs online_decode.main with the remaining argv

  --final-read-post off            (default) exactly decode_crf_ckpt.main with the argv unchanged; Engine.read untouched
  --final-read-post close_detect   as online_decode_finalpost (FINALPOST_STATS printed at the end)

    python -m experiments.mist_w3po_v1.crf.decode_crf_ckpt_finalpost --crf-ckpt <run>/ckpt/step_NNNNN \
        --final-read-post close_detect --thinker <v8 merged> --windows <jsonl> --out <new jsonl> <online_decode args>
"""
from __future__ import annotations

import json
import sys

from experiments.mist_w3po_v1.core import engine as E
from experiments.mist_w3po_v1.crf import decode_crf_ckpt as dc
from experiments.mist_w3po_v1.evaluation import online_decode_finalpost as fp


def main() -> None:
    mode, rest = fp.split_argv(sys.argv[1:])
    if mode == "close_detect":
        fp.install(fp.load_close_detect(fp.windows_arg(rest)))
        print(f"final-read-post close_detect: final reads see P up to min(t, close_detect - {E.LOOKAHEAD_MS} ms)",
              flush=True)
    sys.argv = [sys.argv[0]] + rest
    dc.main()
    if mode == "close_detect":
        print(f"FINALPOST_STATS {json.dumps(fp.STATS)}", flush=True)


if __name__ == "__main__":
    main()

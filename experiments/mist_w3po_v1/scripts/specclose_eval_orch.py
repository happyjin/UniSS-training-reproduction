#!/usr/bin/env python
"""SpecClose dev evaluation orchestrator (reports/mist_w3po_v1/thinker_detector_specclose_prereg_v1.zh-CN.md §7):
thinker_t_eval_orch.py unchanged, with ONE substitution in crf_pair_dev_orch.dec_cmd's decode command:
  -m experiments.mist_w3po_v1.crf.decode_crf_ckpt_finalpost --crf-ckpt <ck> --final-read-post close_detect
  -> -m experiments.mist_w3po_v1.crf.decode_crf_ckpt_specclose --crf-ckpt <ck>
(the final read sees P up to the windows' final_post_cut_ms). Every other job (score, speech prep, talkers, ASR,
speech_latency_lit, gates) is thinker_t_eval_orch's; pass --win <SpecClose windows jsonl> so all of them use it.

    python specclose_eval_orch.py <thinker_t_eval_orch arguments> --win <specclose windows> [--dry]
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import crf_pair_dev_orch as cp  # noqa: E402
import thinker_t_eval_orch as T  # noqa: E402

SPEC_MOD = "experiments.mist_w3po_v1.crf.decode_crf_ckpt_specclose"
_orig_dec_cmd = cp.dec_cmd


def dec_cmd(ck: str, out: str, extra: str, prev: str = "") -> str:
    c = _orig_dec_cmd(ck, out, extra, prev)
    old = f"-m {cp.MOD} --crf-ckpt {ck} --final-read-post close_detect "
    assert c.count(old) == 1, c
    return c.replace(old, f"-m {SPEC_MOD} --crf-ckpt {ck} ", 1)


def main() -> None:
    if "--win" not in sys.argv[1:] and not any(a.startswith("--win=") for a in sys.argv[1:]):
        raise SystemExit("specclose_eval_orch: --win <SpecClose windows jsonl> is required")
    cp.dec_cmd = dec_cmd
    T.main()


if __name__ == "__main__":
    main()

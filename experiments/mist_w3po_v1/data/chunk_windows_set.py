"""Chunk-protocol windows of a TEST set (reports/mist_w3po_v1/final_test_prereg_v1.zh-CN.md; the user's latency
reporting standard, latency_chunk_protocol_table_v1): specclose.cmd_build with --judge off -- the same C2 windows
(id, meeting, start, duration, audio, posteriors) with the chunk times of the lowlat preset (open = T(st), a gap / cap
close = T(j - 1), session end unchanged) -- on that set's meeting posteriors / session audio instead of the dev ones
(specclose.DATA / DIAR / MIX). The judge-off build asserts that its simulated windows equal --base-win row by row.
Additive: specclose.py is unchanged. CPU only. Writes <out>/windows_dev_online.jsonl (the name is specclose's; the rows
are the test windows), windows_dev_retime.jsonl (identical times with judge off), judge.jsonl (empty), summary.json.

    python -m experiments.mist_w3po_v1.data.chunk_windows_set --set zh2en|ali --base-win <test C2 windows> --out <dir>
"""
from __future__ import annotations

import argparse
from pathlib import Path

from experiments.mist_w3po_v1.data import specclose as S

SETS = {"zh2en": ("data/ami_multispk_si_v1", "nemotron_lowlat_zh8", "zh_mix_v8"),
        "ali": ("data/alimeeting_si_v1", "nemotron_lowlat", "near_mix")}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", required=True, choices=list(SETS))
    ap.add_argument("--base-win", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    data, diar, mix = SETS[a.set]
    S.DATA, S.DIAR, S.MIX = Path(data), diar, mix
    S.cmd_build(argparse.Namespace(judge="off", base_win=a.base_win, out=a.out))


if __name__ == "__main__":
    main()

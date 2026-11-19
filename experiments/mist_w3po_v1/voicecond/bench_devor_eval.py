"""Robust wave 2 S2 (reports/mist_w3po_v1/robust_wave2_diag_prereg.zh-CN.md): bench_dev_eval (= bench_eval UNCHANGED with
the dev roots) for the oracle-diarization runs on id2-dev, with only the windows swapped in this process:
  windows_obsidev_id2_<dir>_c2g06{,_chunk}_v1 -> windows_obsidev_id2or_<dir>_c2g06{,_chunk}_v1
  (the C2 windows and the chunk-protocol windows of the speech LAAL, both built on the oracle posteriors by
  s2_oracle_build). References, meetings, utterances, clean speaker clips: id2-dev's. --cond id2 only, --name obsidev_bor_*.

    python -m experiments.mist_w3po_v1.voicecond.bench_devor_eval --cond id2 --dir zh2en|en2zh --name obsidev_bor_<run>
        [--seeds 1|2] [--procs 32]
"""
from __future__ import annotations

import sys

from experiments.mist_w3po_v1.voicecond import bench_dev_eval as DE

E = DE.E


def wins(cond: str, d: str) -> tuple[str, str]:
    assert cond == "id2", cond
    return (f"{E.WD}/windows_obsidev_id2or_{d}_c2g06_v1/windows_test_online_{d}.jsonl",
            f"{E.WD}/windows_obsidev_id2or_{d}_c2g06_chunk_v1/windows_dev_online.jsonl")


def main() -> None:
    argv = sys.argv[1:]
    if "--cond" not in argv or argv[argv.index("--cond") + 1] != "id2":
        raise SystemExit("--cond id2 required")
    if "--name" not in argv or not argv[argv.index("--name") + 1].startswith("obsidev_bor_"):
        raise SystemExit("--name obsidev_bor_<...> required")
    DE.wins = wins   # bench_dev_eval.chunk_lat looks up its module-level wins
    E.wins = wins    # bench_eval.main (C2 windows)
    E.main()


if __name__ == "__main__":
    main()

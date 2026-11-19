#!/usr/bin/env python
"""OpenBroadcast-SI DEV twin (D0, reports/mist_w3po_v1/robust_wave1_prereg.zh-CN.md): scripts/voicecond_bench_orch.py
UNCHANGED, with only its set roots swapped in this process:
  data     data/open_voicecond_v1/bench -> data/open_voicecond_v1/bench_dev   (experiments/.../voicecond/bench_dev_build.py)
  windows  windows_obsi_<cond>_<dir>_c2g06_v1 -> windows_obsidev_<cond>_<dir>_c2g06_v1
  --cond   id2 | clean only (the dev twin has no ood1); run names must start with "obsidev_"
Everything else (--voice-override hook, --exclude check, decode flags, talkers, placement, --cap135, ledger, scoring,
C2 latency) is voicecond_bench_orch's / the wrapped orchestrators'. Meeting ids d[bc][ze]NNN are resolved by
reports/gstar_si_v6/meeting_score.universe (bench_dev fallback).

    python scripts/voicecond_bench_dev_orch.py --cond id2|clean --dir zh2en|en2zh [--voice-override <json>] --base <merged> \\
        --run <new run dir> --name obsidev_<...> --ckpt <ckpt|-> [--speech-from <name>] --talker none|t1c --window 1 \\
        --cap135 --exclude <cards|none> [--dry]
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import voicecond_bench_orch as V  # noqa: E402

V.BD = "data/open_voicecond_v1/bench_dev"


def win(cond: str, d: str) -> str:
    return f"{V.WD}/windows_obsidev_{cond}_{d}_c2g06_v1/windows_test_online_{d}.jsonl"


V.win = win


def main() -> None:
    argv = sys.argv[1:]
    if "--cond" not in argv or argv[argv.index("--cond") + 1] not in ("id2", "clean"):
        raise SystemExit("--cond id2|clean required (dev twin)")
    if "--name" not in argv or not argv[argv.index("--name") + 1].startswith("obsidev_"):
        raise SystemExit("--name obsidev_<...> required (dev twin run names)")
    if "--speech-from" in argv and not argv[argv.index("--speech-from") + 1].startswith("obsidev_"):
        raise SystemExit("--speech-from must be a dev twin text run (obsidev_<...>)")
    V.main()


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Robust wave 2 S2 (reports/mist_w3po_v1/robust_wave2_diag_prereg.zh-CN.md): baseline B on the OpenBroadcast-SI DEV
noisy condition (id2-dev) with REFERENCE-turn diarizer posteriors (experiments/mist_w3po_v1/voicecond/s2_oracle_build.py).
scripts/voicecond_bench_dev_orch.py (= voicecond_bench_orch UNCHANGED with the dev roots) with, in this process, only:
  windows   windows_obsidev_id2_<dir>_c2g06_v1 -> windows_obsidev_id2or_<dir>_c2g06_v1 (C2 gap 0.6 on the oracle posteriors)
  talker    the talker prep's speaker-voice prompts read diar/oracle_lowlat instead of diar/nemotron_lowlat
            ("--diar nemotron_lowlat " -> "--diar oracle_lowlat " after the wrapped orchestrators' own checks)
  --cond    id2 only; run names must start with "obsidev_bor_"
Data (meetings, references, session audio), Thinker, decode flags, Talker T1c, seeds, placement, scoring: B's.

    python scripts/voicecond_bench_devor_orch.py --cond id2 --dir zh2en|en2zh --base <merged> --run <new run dir> \\
        --name obsidev_bor_<...> --ckpt <ckpt|-> [--speech-from obsidev_bor_<...>] --talker none|t1c --window 1 \\
        --cap135 --exclude none [--dry]
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import voicecond_bench_dev_orch as DV  # noqa: E402  (sets the dev roots on voicecond_bench_orch)

V = DV.V
OLD, NEW = " --diar nemotron_lowlat ", " --diar oracle_lowlat "


def win(cond: str, d: str) -> str:
    assert cond == "id2", cond
    return f"{V.WD}/windows_obsidev_id2or_{d}_c2g06_v1/windows_test_online_{d}.jsonl"


def oracle_diar(orig):
    def fix(cmd, *a, **k):
        cmd = orig(cmd, *a, **k)
        if "gstar_talker_v6 prep" in cmd:
            assert OLD in cmd, cmd
            cmd = cmd.replace(OLD, NEW)
        assert OLD not in cmd, cmd
        return cmd
    return fix


def main() -> None:
    argv = sys.argv[1:]
    if "--cond" not in argv or argv[argv.index("--cond") + 1] != "id2":
        raise SystemExit("--cond id2 required (S2 oracle diarization on id2-dev)")
    if "--name" not in argv or not argv[argv.index("--name") + 1].startswith("obsidev_bor_"):
        raise SystemExit("--name obsidev_bor_<...> required")
    if "--speech-from" in argv and not argv[argv.index("--speech-from") + 1].startswith("obsidev_bor_"):
        raise SystemExit("--speech-from must be an S2 text run (obsidev_bor_<...>)")
    V.win = win
    V.F.fix = oracle_diar(V.F.fix)   # zh2en: final_test_orch.build_jobs calls the module-level fix
    V.E.fix = oracle_diar(V.E.fix)   # en2zh: voicecond_bench_orch wraps E.fix again at its main (outermost)
    DV.main()


if __name__ == "__main__":
    main()

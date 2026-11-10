#!/usr/bin/env python
"""Speech-gap plan C2 (reports/mist_w3po_v1/text_to_speech_gap_plan.zh-CN.md §5 C2; user go 2026-11-10 ~15:50 UTC):
the dev online windows rebuilt with a shorter closing gap (build_windows_zh2en.py --gap 0.5: a window closes after
0.5 s without diarizer activity instead of 1.0 s; dev detector wait 1.64 -> 1.21 s, utterances cut by a window
boundary 2.5 % -> 3.6 %, c2_gap_stats.py), then the current best system on them: SerialCRF v2' (step_03000,
--final-read-post close_detect) + talker c1+b1 (speech_var_orch.talker_cmd), both TTS seeds.

Every job is crf_pair_dev_orch's (decode 8 shards -> score -> 2 speech chains), verbatim, with two changes only:
crf_pair_dev_orch.WIN (read by its decode / score / talker-prep / speech-score commands at call time) points at the
C2 windows, and each talker command gets c1+b1's flags. The diagnostic diagA job is left out. Placement, locks, caps:
crf_pair_dev_orch.PairOrch (--spread). Ref windows (scoring) unchanged. Dev only.

    python c2_dev_orch.py --run logs/mist_w3po_v1/c2_dev --scorers 4 --spread [--dry]
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import crf_pair_dev_orch as cp  # noqa: E402
import finalpost_dev_orch as fo  # noqa: E402
import speech_var_orch as sv  # noqa: E402
from finalpost_dev_orch import ENV, REPO  # noqa: E402

WIN_C2 = "eval_outputs/mist_w3po_v1/data/windows_zh2en_c2g05_v1/windows_dev_online.jsonl"
NAME = "dev_crf_v2p_s3000_finalpost_s8_c2g05"
CKPT = f"{cp.TRAIN}/crf_v2p_20261110T075131Z/ckpt/{cp.STEP}"
TALKER_VAR = "c1+b1"


def setup() -> None:
    cp.WIN = WIN_C2
    cp.CANDS = {"v2p_c2g05": (NAME, CKPT)}
    cp.DIAG = "eval_outputs/mist_w3po_v1/crf/v2p_c2g05_dev_diagA"   # not run (diagA dropped); keeps names distinct


def build_jobs(scorers: int) -> list[fo.Job]:
    setup()
    jobs = [j for j in cp.build_jobs() if not j.id.endswith("__diagA")]
    for j in jobs:
        if j.kind == "talk":
            k = int(j.id.rsplit("__talk", 1)[1])
            j.cmd = (lambda att, f=j.cmd, k=k: sv.talker_cmd(f(att), TALKER_VAR, k, scorers))
    ids = {j.id for j in jobs}
    for j in jobs:
        assert all(d in ids for d in j.deps), (j.id, j.deps)
    return jobs


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True)
    ap.add_argument("--scorers", type=int, default=4)
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--spread", action="store_true")
    a = ap.parse_args()
    os.chdir(REPO)
    jobs = build_jobs(a.scorers)
    assert len({j.id for j in jobs}) == len(jobs)
    dirs = [d for d in cp.new_dirs() if not d.startswith(cp.DIAG)]
    run = Path(a.run)
    if a.dry:
        for j in jobs:
            print(j.prio, j.kind, j.id, "deps", len(j.deps), "|", j.cmd(1)[len(ENV):][:900])
        print(len(jobs), "jobs; new dirs:", [d for d in dirs if Path(d).exists()] or "none exist")
        return
    if a.resume:
        if not (run / "state").is_dir():
            raise SystemExit(f"--resume: no state in {run}")
        o = cp.PairOrch(run, jobs)
        o.adopt()
    else:
        if run.exists():
            raise SystemExit(f"拒绝覆盖已有输出: {run}")
        ex = [d for d in dirs if Path(d).exists()]
        if ex:
            raise SystemExit(f"拒绝覆盖已有输出: {ex}")
        if not Path(WIN_C2).exists():
            raise SystemExit(f"C2 windows missing: {WIN_C2}")
        missing = [f"{CKPT}/{f}" for f in ("DONE", "lora.pt", "extra.pt") if not Path(f"{CKPT}/{f}").exists()]
        if missing:
            raise SystemExit(f"checkpoint files missing: {missing}")
        run.mkdir(parents=True)
        o = cp.PairOrch(run, jobs)
    o.spread = a.spread
    o.log(f"spread {a.spread}; C2 windows {WIN_C2}; talker {TALKER_VAR}; scorers {a.scorers}")
    Path(fo.LOCK).parent.mkdir(parents=True, exist_ok=True)
    try:
        o.loop()
    except Exception as e:  # noqa: BLE001
        import traceback
        o.log("CRASH " + traceback.format_exc().replace("\n", " | "))
        o.alert(f"ORCHESTRATOR CRASH {e!r}; running jobs continue (their rc files land in state/)")
        raise


if __name__ == "__main__":
    main()

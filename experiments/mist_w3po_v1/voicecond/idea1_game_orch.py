#!/usr/bin/env python
"""Idea 1 (ZNE) E1.2b (reports/mist_w3po_v1/voicecond_idea1_prereg.zh-CN.md §6.3): baseline B's T1c speech chain on the 10
game demo excerpts, exactly as experiments/mist_w3po_v1/scripts/game_orch.py runs it (same text via --speech-from, same
talker flags, shards, both TTS seeds, ASR, latency), with ONE change: every Talker (job kind "talk") runs
voicecond/idea1_si_infer_vo.py with --voice-override <table> instead of si_infer, i.e. the 32 voice tokens of each stream
are the precomputed ZNE tokens (checked against the sha1 of the prompt file the chain prepares). The swap is applied to
the final command strings when the orchestrator object is built (after every wrapped orchestrator has applied its own
rewrites and checks); with --dry the swapped talk commands are printed after the usual dry listing.

    python experiments/mist_w3po_v1/voicecond/idea1_game_orch.py --voice-override <tokens.jsonl> <game_orch.py arguments>
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "experiments/mist_w3po_v1/scripts"))

OLD = "-m experiments.uniss_qwen3_omni_v1.evaluation.si_infer "
NEW = "-m experiments.mist_w3po_v1.voicecond.idea1_si_infer_vo --voice-override {} "


def main() -> None:
    argv = sys.argv[1:]
    i = argv.index("--voice-override")
    table = str(Path(argv[i + 1]).resolve())
    del argv[i:i + 2]
    if not Path(table).is_file():
        raise SystemExit(f"--voice-override: no table {table}")
    import finalpost_dev_orch as fo
    import game_orch as G
    import talker_crf_orch as tc
    made = []
    orig_job_init = fo.Job.__init__

    def job_init(self, *a, **k):
        orig_job_init(self, *a, **k)
        made.append(self)

    fo.Job.__init__ = job_init

    def swap(f):
        def cmd(att):
            c = f(att)
            assert c.count(OLD) == 1, c[:300]
            return c.replace(OLD, NEW.format(table))
        return cmd

    orig_init = tc.YieldOrch.__init__

    def orch_init(self, run, jobs, *a, **k):
        n = 0
        for j in jobs:
            if j.kind == "talk":
                j.cmd = swap(j.cmd)
                j.cmd(1)
                n += 1
        assert n > 0, "no talk jobs"
        print(f"[idea1] voice-override swapped into {n} talk jobs ({table})", flush=True)
        orig_init(self, run, jobs, *a, **k)

    tc.YieldOrch.__init__ = orch_init
    sys.argv = [sys.argv[0]] + argv
    G.main()
    if "--dry" in argv:
        talks = [j for j in made if j.kind == "talk"]
        for j in talks:
            print("[idea1 dry]", j.id, "|", swap(j.cmd)(1)[:900])
        print(f"[idea1 dry] {len(talks)} talk jobs would run idea1_si_infer_vo")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Speech-gap plan batch 1 (reports/mist_w3po_v1/text_to_speech_gap_plan.zh-CN.md §5; user go 2026-11-10 ~11:50 UTC):
the talker-only variants on C0-fp's dev text, each with both TTS seeds, so text tcpBLEU is that of C0-fp by
construction and only the speech changes.

  b1  --select-k 2, the candidate an independent CTC ASR (si_ctc_scorer.py, wav2vec2; not the Whisper evaluator)
      finds likeliest to say the block            [shard k asks the scorer on port SCORER_PORT0 + k % --scorers]
  b3  --src-mask-own-ms 250: a stream reads only its own segments of the mixed window audio (system segments, causal)
  b2  --rerun-forced 3: guard-forced (runaway) blocks are sampled again, first that closes on its own kept
  c1  --speed-policy queue (si_infer's existing fluency plan v3 policy, its defaults) instead of fixed

Every chain is crf_pair_dev_orch.speech verbatim (prep -> 8 talkers -> 16 ASR sub-shards -> merge -> asr_hyp ->
score_run --asr), into new dirs decode|speech/<C0-fp>__<v>[_tts2]; the talker command gets the variant's flags and
nothing else. Placement, gpu_launch.lock, caps (<= 135 GiB), talker pair gap: crf_pair_dev_orch.PairOrch (--spread).
Seeds: the C0-fp runs' own (talker default seed / SEED2), so each variant is paired with C0-fp sample for sample.

    python speech_var_orch.py --run logs/mist_w3po_v1/speech_var_b1 --vars b1,b3,b2,c1 --spread [--dry]
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import crf_pair_dev_orch as cp  # noqa: E402
import finalpost_dev_orch as fo  # noqa: E402
from finalpost_dev_orch import DEC, ENV, REPO, SEED2, SPD  # noqa: E402

BASE = "dev_crf_c0fp_s3000_finalpost_s8"
SCORER_PORT0 = 18765
VARS = {
    "b1": "--select-k 2",
    "b3": "--src-mask-own-ms 250",
    "b2": "--rerun-forced 3",
    "c1": "",   # speed policy swap below
}


def talker_cmd(cmd: str, v: str, shard: int, scorers: int) -> str:
    """The C0-fp talker command with variant v's flags (inserted before the shard flags)."""
    assert cmd.count("--speed-policy fixed") == 1 and cmd.count(" --shard ") == 1, cmd
    if v == "c1":
        return cmd.replace("--speed-policy fixed", "--speed-policy queue")
    extra = VARS[v]
    if v == "b1":
        extra += f" --scorer-url http://127.0.0.1:{SCORER_PORT0 + shard % scorers}/score"
    return cmd.replace(" --shard ", f" {extra} --shard ", 1)


def build_jobs(vs: list[str], scorers: int) -> list[fo.Job]:
    jobs: list[fo.Job] = []

    def add(*a, **k):
        jobs.append(fo.Job(*a, **k))
        jobs[-1].order = len(jobs)

    for v in vs:
        for name, seed in ((f"{BASE}__{v}", None), (f"{BASE}__{v}_tts2", SEED2)):
            n0 = len(jobs)
            cp.speech(add, name, BASE, seed, 2, [])
            for j in jobs[n0:]:
                if j.kind == "talk":
                    k = int(j.id.rsplit("__talk", 1)[1])
                    j.cmd = (lambda att, f=j.cmd, v=v, k=k: talker_cmd(f(att), v, k, scorers))
    return jobs


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True)
    ap.add_argument("--vars", default="b1,b3,b2,c1")
    ap.add_argument("--scorers", type=int, default=4, help="b1: CTC scorer servers (ports SCORER_PORT0 + i)")
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--spread", action="store_true")
    a = ap.parse_args()
    vs = a.vars.split(",")
    assert vs and all(v in VARS for v in vs), vs
    os.chdir(REPO)
    cp.CANDS = {}   # PairOrch.loop: no text-score flags in this job list
    run = Path(a.run)
    jobs = build_jobs(vs, a.scorers)
    assert len({j.id for j in jobs}) == len(jobs)
    ids = {j.id for j in jobs}
    for j in jobs:
        for d in j.deps:
            assert d in ids, (j.id, d)
    dirs = [f"{R}/{BASE}__{v}{s}" for v in vs for s in ("", "_tts2") for R in (DEC, SPD)]
    if a.dry:
        for j in jobs:
            print(j.prio, j.kind, j.id, "deps", len(j.deps), "|", j.cmd(1)[len(ENV):][:700])
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
        for f in ("hyp.jsonl", "score_text.json"):
            if not Path(f"{DEC}/{BASE}/{f}").exists():
                raise SystemExit(f"C0-fp {f} missing")
        run.mkdir(parents=True)
        o = cp.PairOrch(run, jobs)
    o.spread = a.spread
    o.log(f"spread {a.spread}; vars {vs}; scorers {a.scorers}")
    try:
        o.loop()
    except Exception as e:  # noqa: BLE001
        import traceback
        o.log("CRASH " + traceback.format_exc().replace("\n", " | "))
        o.alert(f"ORCHESTRATOR CRASH {e!r}; running jobs continue (their rc files land in state/)")
        raise


if __name__ == "__main__":
    main()

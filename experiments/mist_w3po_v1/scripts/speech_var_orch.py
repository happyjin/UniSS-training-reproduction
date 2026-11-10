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
    # after b1 / c1 passed (batch 1-2), their second points (2026-11-10 ~15:55 UTC, stacked next to C2):
    "b1k4": "--select-k 4",     # b1 with 4 candidates per block
    "s2": "--speed-shift 2",    # with c1: every block asks 2 speed steps (0.2 source/target ratio) faster
}


def talker_cmd(cmd: str, v: str, shard: int, scorers: int) -> str:
    """The C0-fp talker command with variant v's flags (inserted before the shard flags). v may combine variants with
    '+' (batch 2, e.g. c1+b1): each one's change applied in turn, a single variant exactly as in batch 1."""
    assert cmd.count("--speed-policy fixed") == 1 and cmd.count(" --shard ") == 1, cmd
    for u in v.split("+"):
        if u == "c1":
            cmd = cmd.replace("--speed-policy fixed", "--speed-policy queue")
            continue
        extra = VARS[u]
        if u.startswith("b1"):
            extra += f" --scorer-url http://127.0.0.1:{SCORER_PORT0 + shard % scorers}/score"
        cmd = cmd.replace(" --shard ", f" {extra} --shard ", 1)
    return cmd


def tag(v: str) -> str:
    """dir / job-id suffix of variant v ('c1+b1' -> 'c1_b1')."""
    return v.replace("+", "_")


def split_var(item: str, base: str) -> tuple[str, str]:
    """'c1+b1' -> ('c1+b1', base); 'c1@<decode dir name>' -> ('c1', that dir name)."""
    v, _, b = item.partition("@")
    return v, (b or base)


def build_jobs(vs: list[str], scorers: int, base: str = BASE) -> list[fo.Job]:
    jobs: list[fo.Job] = []

    def add(*a, **k):
        jobs.append(fo.Job(*a, **k))
        jobs[-1].order = len(jobs)

    for item in vs:
        v, b = split_var(item, base)
        for name, seed in ((f"{b}__{tag(v)}", None), (f"{b}__{tag(v)}_tts2", SEED2)):
            n0 = len(jobs)
            cp.speech(add, name, b, seed, 2, [])
            for j in jobs[n0:]:
                if j.kind == "talk":
                    k = int(j.id.rsplit("__talk", 1)[1])
                    j.cmd = (lambda att, f=j.cmd, v=v, k=k: talker_cmd(f(att), v, k, scorers))
    return jobs


def main():
    global SCORER_PORT0
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True)
    ap.add_argument("--vars", default="b1,b3,b2,c1",
                    help="comma list; an item is v or v@<base decode dir>, v = one variant or several joined by +")
    ap.add_argument("--scorers", type=int, default=4, help="b1: CTC scorer servers (ports SCORER_PORT0 + i)")
    ap.add_argument("--scorer-port0", type=int, default=SCORER_PORT0, help="first scorer port (default 18765)")
    ap.add_argument("--base", default=BASE, help="decode dir whose text the variants speak (default C0-fp)")
    ap.add_argument("--win", default=cp.WIN,
                    help="online windows the base was decoded on (default crf_pair_dev_orch.WIN, unchanged; a C2 base "
                         "passes its own, e.g. windows_zh2en_c2g05_v1); read by talker prep / speech score")
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--spread", action="store_true")
    a = ap.parse_args()
    vs = a.vars.split(",")
    SCORER_PORT0 = a.scorer_port0   # read by talker_cmd when the talk commands are built
    assert vs and all(u in VARS for v in vs for u in split_var(v, a.base)[0].split("+")), vs
    os.chdir(REPO)
    cp.WIN = a.win   # crf_pair_dev_orch.speech's commands read it at call time
    cp.CANDS = {}   # PairOrch.loop: no text-score flags in this job list
    run = Path(a.run)
    jobs = build_jobs(vs, a.scorers, a.base)
    assert len({j.id for j in jobs}) == len(jobs)
    ids = {j.id for j in jobs}
    for j in jobs:
        for d in j.deps:
            assert d in ids, (j.id, d)
    pairs = [split_var(v, a.base) for v in vs]
    dirs = [f"{R}/{b}__{tag(v)}{s}" for v, b in pairs for s in ("", "_tts2") for R in (DEC, SPD)]
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
        for b in sorted({b for _, b in pairs}):
            for f in ("hyp.jsonl", "score_text.json"):
                if not Path(f"{DEC}/{b}/{f}").exists():
                    raise SystemExit(f"base {b} {f} missing")
        run.mkdir(parents=True)
        o = cp.PairOrch(run, jobs)
    o.spread = a.spread
    o.log(f"spread {a.spread}; base {a.base}; vars {vs}; scorers {a.scorers}; windows {a.win}")
    try:
        o.loop()
    except Exception as e:  # noqa: BLE001
        import traceback
        o.log("CRASH " + traceback.format_exc().replace("\n", " | "))
        o.alert(f"ORCHESTRATOR CRASH {e!r}; running jobs continue (their rc files land in state/)")
        raise


if __name__ == "__main__":
    main()

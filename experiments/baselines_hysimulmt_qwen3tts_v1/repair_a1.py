"""Bookkeeping for the a1 runs while no worker or pipeline runs (nothing is deleted; superseded files are moved aside).

  orphans     a claimed attempt with neither a done file nor a failed marker (its worker was stopped by PID to keep the
              cards under the memory limit): a failed marker is added, so the next worker pass retries it
  off-by-one  a done job written before the cap fix (tts_worker.py, 2026-11-12 20:23 Prague) in which some block kept
              cap frames, one more than one request with max_new_tokens = cap returns (cap - 1): its done file moves to
              done_superseded/ and the attempt gets a failed marker, so it is synthesised again with the fixed worker
  runs        an assembled run (or a partial run folder) that holds a superseded job moves to runs_superseded/

    PYTHONPATH=. venvs/tts_schemes/bin/python experiments/baselines_hysimulmt_qwen3tts_v1/repair_a1.py [--apply]
"""
import argparse
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from assemble import R, run_dir, tts_dir  # noqa: E402

CLAIMS = R / "claims"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()
    jobs = [json.loads(l) for l in open(R / "jobs.jsonl", encoding="utf-8")]
    orphans, offby, bad_runs = [], [], set()
    for j in jobs:
        t = tts_dir(j["point"], j["direction"], j["seed"])
        done = t / "done" / f"{j['job']}.json"
        n = max((n for n in (1, 2, 3) if (CLAIMS / f"{j['job']}_a{n}").exists()), default=None)
        if done.exists():
            r = json.load(open(done, encoding="utf-8"))
            if any(b["frames"] >= b["cap"] for b in r["blocks"]):
                offby.append((j, done, r["attempt"]))
                bad_runs.add((j["point"], j["direction"], j["seed"]))
        elif n is not None and not (CLAIMS / f"failed_{j['job']}_a{n}").exists():
            orphans.append((j, n))
    print(f"orphans {len(orphans)}: {[f'{j[0]}_a{n}' for j, n in [((x['job'],), n) for x, n in orphans]]}")
    print(f"off-by-one done jobs {len(offby)}: {[x[0]['job'] for x in offby]}")
    partial = {k for k in [(p, d, s) for p in ("hy-std-w4", "hy-sc-w4") for d in ("en2zh", "zh2en") for s in (0, 1, 2)]
               if run_dir(*k).exists() and not (run_dir(*k) / "asr_speech.jsonl").exists()}
    moves = sorted(k for k in bad_runs | partial if run_dir(*k).exists())
    print(f"run folders to move aside: {moves}")
    if not a.apply:
        print("dry run; --apply to act")
        return
    for j, n in orphans:
        os.close(os.open(CLAIMS / f"failed_{j['job']}_a{n}", os.O_CREAT | os.O_EXCL | os.O_WRONLY))
    for j, done, att in offby:
        dst = done.parent.parent / "done_superseded" / f"{j['job']}_a{att}.json"
        dst.parent.mkdir(exist_ok=True)
        assert not dst.exists()
        done.rename(dst)
        os.close(os.open(CLAIMS / f"failed_{j['job']}_a{att}", os.O_CREAT | os.O_EXCL | os.O_WRONLY))
    for k in moves:
        src = run_dir(*k)
        dst = R / "runs_superseded" / src.parent.name / src.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        assert not dst.exists()
        src.rename(dst)
    print("applied")


if __name__ == "__main__":
    main()

"""Paper v9: launcher that places decode/train jobs by REAL nvidia-smi memory, beside other people's jobs.

Jobs are read every loop from <dir>/*.json (lists of {"id", "gpus", "need", "cmd", "prio"}); a job id is started once
only. Waiting jobs are tried in (prio, file name, list order); the first one whose "gpus" cards (the cards with the most
free memory) all have need + 8 GiB free starts, then 60 s pass before the next start so its memory shows. A job is
never placed if that would leave a card with less than 8 GiB. <dir>/cancel (job ids, one per line) drops waiting jobs.
Exits when <dir>/FINISH exists and nothing waits or runs. Every start / exit (with rc) goes to <log>.

    setsid nohup python3 reports/gstar_si_v9/memq9.py <dir> <log> &
"""
import glob
import json
import os
import subprocess
import sys
import time

REPO = "/opt/dlami/nvme/jasonleeeli/projects/UniSS"
HEAD = 8192


def free():
    out = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.used,memory.total",
                                   "--format=csv,noheader,nounits"], text=True)
    return {int(i): int(t) - int(u) for i, u, t in (ln.split(", ") for ln in out.strip().splitlines())}


def say(log, msg):
    with open(log, "a") as fh:
        fh.write(time.strftime("%m-%d %H:%M:%S ", time.gmtime()) + msg + "\n")


def load(d, seen):
    todo = []
    cancel = set()
    if os.path.exists(os.path.join(d, "cancel")):
        cancel = {ln.strip() for ln in open(os.path.join(d, "cancel")) if ln.strip()}
    for f in sorted(glob.glob(os.path.join(d, "*.json"))):
        try:
            jobs = json.load(open(f))
        except (json.JSONDecodeError, OSError):
            continue
        for i, j in enumerate(jobs):
            if j["id"] not in seen and j["id"] not in cancel:
                todo.append((j.get("prio", 5), os.path.basename(f), i, j))
    return [t[3] for t in sorted(todo, key=lambda t: t[:3])]


def main():
    d, log = sys.argv[1], sys.argv[2]
    seen, run = set(), {}
    while True:
        for k, p in list(run.items()):
            if p.poll() is not None:
                say(log, f"exit {k} rc {p.returncode}")
                del run[k]
        todo = load(d, seen)
        if not todo and not run and os.path.exists(os.path.join(d, "FINISH")):
            break
        f = free()
        for j in todo:
            cards = sorted(f, key=lambda g: -f[g])[: j.get("gpus", 2)]
            if all(f[g] >= j["need"] + HEAD for g in cards):
                cvd = ",".join(map(str, sorted(cards)))
                env = dict(os.environ, MIST_V7_JOB=j["id"], MIST_V7_KIND=j.get("kind", "decode"), CVD=cvd)
                run[j["id"]] = subprocess.Popen(["bash", "-c", j["cmd"]], env=env, cwd=REPO)
                seen.add(j["id"])
                say(log, f"start {j['id']} on {cvd} (free {', '.join(str(f[g]) for g in cards)} MiB)")
                time.sleep(60)
                break
        else:
            time.sleep(30)
    say(log, "MEMQ_DONE")


if __name__ == "__main__":
    main()

"""Paper v8: a minimal launcher that places decode jobs ("gpus" cards each, default 2) by REAL nvidia-smi memory (the v8 queue's declared
bookkeeping and steadiness rule left cards idle). Jobs (a json list of {"id", "need", "cmd"}; need = MiB per card a
job may grow to) start in list order on the two cards with the most free memory, if both have need + 8 GiB free;
60 s between starts, so the previous job's memory shows before the next placement. Every job's exit code goes to the log.

    setsid nohup python3 reports/gstar_si_v8/memq8.py <jobs.json> <log> &
"""
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


def main():
    jobs, log = json.load(open(sys.argv[1])), sys.argv[2]
    todo, run = list(jobs), {}
    while todo or run:
        for k, p in list(run.items()):
            if p.poll() is not None:
                say(log, f"exit {k} rc {p.returncode}")
                del run[k]
        if todo:
            f = free()
            j = todo[0]
            cards = sorted(f, key=lambda g: -f[g])[: j.get("gpus", 2)]
            if all(f[g] >= j["need"] + HEAD for g in cards):
                cvd = ",".join(map(str, sorted(cards)))
                env = dict(os.environ, MIST_V7_JOB=j["id"], MIST_V7_KIND="decode", CVD=cvd)
                run[j["id"]] = subprocess.Popen(["bash", "-c", j["cmd"]], env=env, cwd=REPO)
                say(log, f"start {j['id']} on {cvd} (free {f[cards[0]]}, {f[cards[1]]} MiB)")
                todo.pop(0)
                time.sleep(60)
                continue
        time.sleep(30)
    say(log, "MEMQ_DONE")


if __name__ == "__main__":
    main()

"""v10 GPU memory watch (log only, stops nothing): every 10 s, each card's used memory against its limit -- 126,000 MiB on
a card that carries a colleague's process (working directory under /opt/dlami/nvme/zezhong), 136,000 MiB otherwise.
Each card over its limit is written, with every compute process on it (pid, MiB, owner, command), to
logs/qabc_v10/mem_hot.log; a one-line summary of all cards goes to logs/qabc_v10/mem_watch.log each minute.
Runs until logs/qabc_v10/MEM_WATCH_STOP exists.

    setsid nohup /opt/dlami/nvme/jasonleeeli/conda_envs/uniss-eval/bin/python -u reports/qabc_v10/mem_watch.py &
"""
import os
import subprocess
import time
from datetime import datetime, timezone

R = "/opt/dlami/nvme/jasonleeeli/projects/UniSS"
LOG = f"{R}/logs/qabc_v10/mem_watch.log"
HOT = f"{R}/logs/qabc_v10/mem_hot.log"
STOP = f"{R}/logs/qabc_v10/MEM_WATCH_STOP"
COLLEAGUE = "/opt/dlami/nvme/zezhong"


def now():
    return datetime.now(timezone.utc).strftime("%F %T")


def smi(q):
    out = subprocess.run(["nvidia-smi", f"--query-{q}", "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
    return [[x.strip() for x in l.split(",")] for l in out.strip().splitlines() if l.strip()]


def cwd(pid):
    try:
        return os.readlink(f"/proc/{pid}/cwd")
    except OSError:
        return ""


def args(pid):
    try:
        return open(f"/proc/{pid}/cmdline", "rb").read().replace(b"\0", b" ").decode(errors="replace")
    except OSError:
        return ""


last = 0
while not os.path.exists(STOP):
    try:
        uuid2idx = {u: int(i) for i, u in smi("gpu=index,uuid")}
        used = {int(i): int(m) for i, m in smi("gpu=index,memory.used")}
        apps = [(uuid2idx.get(u, -1), int(p), int(m)) for u, p, m in smi("compute-apps=gpu_uuid,pid,used_memory")]
    except Exception as e:  # noqa: BLE001
        open(LOG, "a").write(f"[{now()}] nvidia-smi failed: {e}\n")
        time.sleep(10)
        continue
    for g, mem in sorted(used.items()):
        on = [(p, m, cwd(p)) for gg, p, m in apps if gg == g]
        limit = 126000 if any(c.startswith(COLLEAGUE) for _, _, c in on) else 136000
        if mem > limit:
            with open(HOT, "a") as fh:
                fh.write(f"[{now()}] HOT card {g}: {mem} MiB > {limit}\n")
                for p, m, c in sorted(on, key=lambda x: -x[1]):
                    who = "colleague" if c.startswith(COLLEAGUE) else "ours"
                    fh.write(f"    {p} {m} MiB {who} {args(p)[:150]}\n")
    if time.time() - last > 60:
        open(LOG, "a").write(f"[{now()}] " + " ".join(f"{g}:{used[g] // 1000}" for g in sorted(used)) + "\n")
        last = time.time()
    time.sleep(10)
open(LOG, "a").write(f"[{now()}] stopped\n")

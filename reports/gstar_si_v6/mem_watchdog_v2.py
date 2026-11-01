"""Memory watchdog v2 of the paper-v6 (MIST) branch (2026-11-01: the colleague's restarted job swings by up to
~20 GB, so we keep ~6 GB headroom): a card with another user's process is acted on when it stays above --soft
(122,000 MiB) for --hold consecutive checks, or at once above --limit-foreign (126,000 MiB). Otherwise as v1: Every 15 s, per card: if the card's used memory exceeds its limit
(126,000 MiB when a process of another user runs on it -- told by the process's working directory, since vLLM renames
its engine -- else 136,000 MiB), the largest decode worker OF THIS BRANCH on that card is stopped by PID (SIGTERM).
A worker of this branch is a process whose own or parent command line contains "gstar_multi_v6"; nothing else is ever
touched (another user's jobs, the single-speaker branch's jobs, GPU holders). Every action is logged here and in
logs/gpu_coord/events.log. Stops itself when --until-file exists.

    setsid nohup python3 reports/gstar_si_v6/mem_watchdog_v2.py --until-file logs/multispk/v6/WATCHDOG_STOP > logs/multispk/v6/watchdog.log 2>&1 &
"""
import argparse
import os
import signal
import subprocess
import time

ME = "/opt/dlami/nvme/jasonleeeli/"
EVENTS = "/opt/dlami/nvme/jasonleeeli/projects/UniSS/logs/gpu_coord/events.log"


def q(args):
    return subprocess.check_output(["nvidia-smi"] + args, text=True).strip().splitlines()


def cmd(pid):
    try:
        return open(f"/proc/{pid}/cmdline", "rb").read().replace(b"\0", b" ").decode(errors="replace")
    except OSError:
        return ""


def ppid(pid):
    try:
        return int(open(f"/proc/{pid}/stat").read().rsplit(")", 1)[1].split()[1])
    except (OSError, ValueError, IndexError):
        return 0


def foreign(pid):
    try:
        return not os.readlink(f"/proc/{pid}/cwd").startswith(ME)
    except OSError:
        return True


def ours(pid):
    return "gstar_multi_v6" in cmd(pid) or "gstar_multi_v6" in cmd(ppid(pid))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--until-file", required=True)
    ap.add_argument("--limit-foreign", type=int, default=126000)
    ap.add_argument("--limit", type=int, default=136000)
    ap.add_argument("--soft", type=int, default=122000)
    ap.add_argument("--hold", type=int, default=2)
    a = ap.parse_args()
    over = {}
    while not os.path.exists(a.until_file):
        try:
            bus = {}
            used = {}
            for line in q(["--query-gpu=index,pci.bus_id,memory.used", "--format=csv,noheader,nounits"]):
                i, b, u = [x.strip() for x in line.split(",")]
                bus[b.lower()], used[int(i)] = int(i), int(u)
            procs = {i: [] for i in used}
            for line in q(["--query-compute-apps=pid,gpu_bus_id,used_memory", "--format=csv,noheader,nounits"]):
                p, b, m = [x.strip() for x in line.split(",")]
                if b.lower() in bus:
                    procs[bus[b.lower()]].append((int(m), int(p)))
            for c, ps in procs.items():
                fgn = any(foreign(p) for _, p in ps)
                lim = a.limit_foreign if fgn else a.limit
                over[c] = over.get(c, 0) + 1 if (fgn and used[c] > a.soft) else 0
                if used[c] <= lim and over[c] < a.hold:
                    continue
                over[c] = 0
                mine = sorted(((m, p) for m, p in ps if ours(p) and "gstar_multi_v6" not in cmd(p)), reverse=True)
                if not mine:
                    print(time.strftime("%F %T"), f"card {c} at {used[c]} MiB > {lim}: no worker of ours on it", flush=True)
                    continue
                m, p = mine[0]
                os.kill(p, signal.SIGTERM)
                msg = (f"MIST(v6) watchdog v2: card {c} at {used[c]} MiB (limit {lim}, soft {a.soft} held); stopped our decode worker {p} "
                       f"({m} MiB) by PID")
                print(time.strftime("%F %T"), msg, flush=True)
                with open(EVENTS, "a") as fh:
                    fh.write(f"[{time.strftime('%F %T', time.gmtime())} UTC] {msg}\n")
        except Exception as e:   # noqa: BLE001
            print("watchdog error", e, flush=True)
        time.sleep(15)


if __name__ == "__main__":
    main()

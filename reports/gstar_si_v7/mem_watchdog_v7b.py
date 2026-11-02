"""Memory watchdog of the paper-v7 (MIST) branch. Every 15 s, per card:

  * a card that also holds another user's process (told by the process's working directory, since vLLM renames its
    engine) is acted on when its used memory stays above --soft (122,000 MiB) for --hold consecutive checks, or at
    once above --limit-foreign (126,000 MiB); a card with only our processes at --limit (136,000 MiB);
  * acting = stopping, by PID (SIGTERM), the most recently started GPU process of THIS branch on that card that is not a training
    rank (a decode or speech job; for a vLLM engine its si_infer parent is stopped too). If the card holds only our
    training ranks and stays above --limit-foreign for --hold checks, the training's launcher (torchrun) is stopped.

A process of this branch carries MIST_V7_JOB in its environment (every v7 queue job exports it; children inherit it);
MIST_V7_KIND tells train / decode / speech apart. Nothing else is ever touched (another user's jobs, GPU holders).
Every action is logged here and in logs/gpu_coord/events.log. Stops itself when --until-file exists.

    setsid nohup python3 reports/gstar_si_v7/mem_watchdog_v7.py --until-file logs/multispk/v7/WATCHDOG_STOP \
        >> logs/multispk/v7/watchdog.log 2>&1 < /dev/null &
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


def env(pid):
    try:
        raw = open(f"/proc/{pid}/environ", "rb").read().split(b"\0")
    except OSError:
        return {}
    out = {}
    for kv in raw:
        if kv.startswith(b"MIST_V7_"):
            k, _, v = kv.partition(b"=")
            out[k.decode()] = v.decode(errors="replace")
    return out


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


def start_time(pid):
    try:
        return int(open(f"/proc/{pid}/stat").read().rsplit(")", 1)[1].split()[19])
    except (OSError, ValueError, IndexError):
        return 0


def foreign(pid):
    try:
        return not os.readlink(f"/proc/{pid}/cwd").startswith(ME)
    except OSError:
        return True


def launcher(pid):
    """The torchrun process above a training rank (or the rank itself)."""
    p = pid
    for _ in range(6):
        if "torch.distributed.run" in cmd(p):
            return p
        p = ppid(p)
        if p <= 1:
            break
    return pid


def log(msg):
    print(time.strftime("%F %T"), msg, flush=True)
    with open(EVENTS, "a") as fh:
        fh.write(f"[{time.strftime('%F %T', time.gmtime())} UTC] MIST(v7) watchdog: {msg}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--until-file", required=True)
    ap.add_argument("--limit-foreign", type=int, default=126000)
    ap.add_argument("--limit", type=int, default=136000)
    ap.add_argument("--soft", type=int, default=122000)
    ap.add_argument("--hold", type=int, default=2)
    a = ap.parse_args()
    over, hard = {}, {}
    while not os.path.exists(a.until_file):
        try:
            bus, used = {}, {}
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
                hard[c] = hard.get(c, 0) + 1 if used[c] > lim else 0
                if hard[c] == 0 and over[c] < a.hold:
                    continue
                mine = [(m, p, env(p)) for m, p in ps if not foreign(p)]
                mine = [(m, p, e) for m, p, e in mine if e.get("MIST_V7_JOB")]
                stoppable = sorted([(m, p, e) for m, p, e in mine if e.get("MIST_V7_KIND") != "train"],
                                   key=lambda t: start_time(t[1]), reverse=True)   # newest-started first
                if stoppable:
                    m, p, e = stoppable[0]
                    targets = [p]
                    if "VLLM" in cmd(p) or e.get("MIST_V7_KIND") == "speech":
                        pp = ppid(p)
                        if pp > 1 and env(pp).get("MIST_V7_JOB"):
                            targets.append(pp)
                    for t in targets:
                        try:
                            os.kill(t, signal.SIGTERM)
                        except OSError:
                            pass
                    log(f"card {c} at {used[c]} MiB (limit {lim}, soft {a.soft}); stopped our {e.get('MIST_V7_KIND')} "
                        f"process(es) {targets} of job {e.get('MIST_V7_JOB')} ({m} MiB) by PID")
                    over[c], hard[c] = 0, 0
                    continue
                trains = [(m, p, e) for m, p, e in mine if e.get("MIST_V7_KIND") == "train"]
                if trains and hard[c] >= a.hold:
                    m, p, e = trains[0]
                    t = launcher(p)
                    try:
                        os.kill(t, signal.SIGTERM)
                    except OSError:
                        pass
                    log(f"card {c} at {used[c]} MiB > {lim} for {hard[c]} checks with only our training on it; stopped "
                        f"training launcher {t} of job {e.get('MIST_V7_JOB')} by PID")
                    over[c], hard[c] = 0, 0
                elif over[c] >= a.hold or hard[c] >= a.hold:
                    print(time.strftime("%F %T"), f"card {c} at {used[c]} MiB: nothing of ours to stop "
                          f"({'training only' if trains else 'no v7 process'})", flush=True)
        except Exception as ex:   # noqa: BLE001
            print("watchdog error", ex, flush=True)
        time.sleep(15)


if __name__ == "__main__":
    main()

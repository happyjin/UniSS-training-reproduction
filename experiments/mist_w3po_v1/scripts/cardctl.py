"""Hand-over of OUR cards (0,1,2,3,7) between the reservation keeper (gpu_keeper.py), W3PO jobs (gpuq_w3.py) and the
user's no-checkpoint workload (scripts/gpu_load/run_mist_workload.sh), user rules of 2026-11-06.

Per card k there is at most one keeper and at most one workload supervisor (one per card: `--devices k`, so a card can
be handed back alone). State files under <qdir>/keeper/card<k>.state (keeper, written by the keeper itself) and
<qdir>/workload/card<k>.json (workload supervisor pid, written here). Every hand-over of card k holds the flock
<qdir>/keeper/card<k>.lock, so the idle-fallback watcher and the queue never act on a card at the same time.

  keeper -> workload, gap-free (user approval 2026-11-06 23:30 UTC): start_workload() with the opt-in
      --tolerate-keeper flag while the keeper stays; the keeper hands its blocks over by itself (it releases whenever
      free < 12 GiB, and the workload pads up to 8 GiB free); once the worker holds its memory (handover_ready:
      worker >= READY_MIB and the keeper holds nothing or the card has < 30000 MiB free) drop_keeper() stops it.
  workload -> keeper (to_keeper): start a keeper next to the workload (it allocates nothing while the card is full),
      then SIGTERM the workload supervisor; the keeper refills within ~1 s of the release, so the card never shows
      >= 80000 MiB free.
Only processes whose /proc cmdline matches ours are signalled, always by PID.
"""
from __future__ import annotations

import fcntl
import json
import os
import signal
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path

REPO = Path("/opt/dlami/nvme/jasonleeeli/projects/UniSS")
PY = "/opt/dlami/nvme/jasonleeeli/conda_envs/qwen3omni/bin/python"
CARDS = (0, 1, 2, 3, 7)
KEEPER = "experiments/mist_w3po_v1/scripts/gpu_keeper.py"
WORKLOAD = "scripts/gpu_load/run_mist_workload.sh"
WORKLOAD_PY = "train_mist_workload.py"
READY_MIB = 56000


DEFAULT_HEADROOM = {4: 25.0, 5: 25.0, 6: 25.0}   # kenxsong's cards while his pipeline runs there


def load_cfg(qdir) -> dict:
    """<qdir>/cards.json {"ours": [...], "headroom": {"card": GB}, "keepers": bool, "idle_s": s} (written by
    takeover_4_6.py / by hand); default 0,1,2,3,7. keepers false (dedicated server, user order 2026-11-07): no
    reservation keeper anywhere -- an idle card goes straight to the workload, a job card holds only the job;
    idle_s: how long a card must be without a job of ours before it gets the workload (default 600 s)."""
    try:
        d = json.loads((Path(qdir) / "cards.json").read_text())
        return {"ours": tuple(int(c) for c in d["ours"]),
                "headroom": {int(k): float(v) for k, v in d.get("headroom", {}).items()},
                "keepers": bool(d.get("keepers", True)), "idle_s": float(d.get("idle_s", 600))}
    except Exception:
        return {"ours": CARDS, "headroom": dict(DEFAULT_HEADROOM), "keepers": True, "idle_s": 600.0}


def log(msg: str):
    print(f"[{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}] {msg}", flush=True)


def cmdline(pid: int) -> str:
    try:
        return open(f"/proc/{pid}/cmdline", "rb").read().replace(b"\0", b" ").decode(errors="ignore")
    except OSError:
        return ""


def is_ours(pid: int) -> bool:
    """root-owned and started from this user's tree (other containers also run as root, 2026-11-07)."""
    try:
        if os.stat(f"/proc/{pid}").st_uid != 0:
            return False
    except OSError:
        return False
    for _ in range(4):   # the process or one of its parents (vLLM engine cores are children of our si_infer)
        if "/opt/dlami/nvme/jasonleeeli/" in cmdline(pid):
            return True
        try:
            pid = int(open(f"/proc/{pid}/stat").read().rsplit(")", 1)[1].split()[1])
        except (OSError, IndexError, ValueError):
            return False
        if pid <= 1:
            return False
    return False


def alive(pid: int) -> bool:
    if not pid:
        return False
    try:
        st = open(f"/proc/{pid}/stat").read()
        return st.rsplit(")", 1)[1].split()[0] != "Z"
    except OSError:
        return False


def smi_apps() -> list[tuple[str, int, int]]:
    out = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout
    res = []
    for l in out.strip().splitlines():
        p = [x.strip() for x in l.split(",")]
        if len(p) == 3 and p[1].isdigit():
            res.append((p[0], int(p[1]), int(p[2]) if p[2].isdigit() else 0))
    return res


def smi_cards() -> dict:
    out = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,memory.used,memory.total", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout
    res = {}
    for l in out.strip().splitlines():
        i, u, used, tot = [x.strip() for x in l.split(",")]
        res[int(i)] = {"uuid": u, "used": float(used), "total": float(tot)}
    return res


class Card:
    def __init__(self, qdir: Path, c: int):
        self.q, self.c = Path(qdir), c
        self.kdir, self.wdir = self.q / "keeper", self.q / "workload"
        self.kdir.mkdir(parents=True, exist_ok=True)
        self.wdir.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def lock(self):
        with open(self.kdir / f"card{self.c}.lock", "a") as fh:
            fcntl.flock(fh, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(fh, fcntl.LOCK_UN)

    # keeper
    def keeper_pid(self) -> int:
        try:
            pid = int(json.loads((self.kdir / f"card{self.c}.state").read_text())["pid"])
        except Exception:
            return 0
        cl = cmdline(pid)
        return pid if alive(pid) and "gpu_keeper.py" in cl and cl.rstrip().endswith(f" {self.c}") else 0

    def start_keeper(self, wait_s: float = 90) -> int:
        old = self.keeper_pid()
        if old:
            return old
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(self.c))
        with open(self.kdir / f"card{self.c}.log", "a") as fh:
            # setsid -f: re-parented to init (the caller must not be the parent of a GPU process, 2026-11-07)
            subprocess.run(["setsid", "-f", PY, KEEPER, str(self.kdir), str(self.c)], env=env, stdout=fh,
                           stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, cwd=REPO)
        t0 = time.time()
        while time.time() - t0 < wait_s:   # up when it has written its own (fresh) state
            pid = self.keeper_pid()
            if pid and self.keeper_held_mib() >= 0:
                log(f"card {self.c}: keeper up, pid {pid}")
                return pid
            time.sleep(1)
        log(f"card {self.c}: keeper did not come up")
        return 0

    def stop_keeper(self, wait_s: float = 30) -> bool:
        pid = self.keeper_pid()
        if not pid:
            return True
        os.kill(pid, signal.SIGTERM)
        t0 = time.time()
        while time.time() - t0 < wait_s:
            if not alive(pid):
                log(f"card {self.c}: keeper pid {pid} stopped")
                return True
            time.sleep(0.5)
        log(f"card {self.c}: keeper pid {pid} still alive after SIGTERM")
        return False

    # workload
    def workload_pid(self) -> int:
        try:
            pid = int(json.loads((self.wdir / f"card{self.c}.json").read_text())["pid"])
        except Exception:
            return 0
        cl = cmdline(pid)
        return pid if alive(pid) and WORKLOAD_PY in cl and f"--devices {self.c} " in cl + " " else 0

    def workload_pids(self) -> set:
        """supervisor + its workers (direct children, each in its own session)."""
        sup = self.workload_pid()
        if not sup:
            return set()
        kids = set()
        for d in os.listdir("/proc"):
            if d.isdigit():
                try:
                    if int(open(f"/proc/{d}/stat").read().rsplit(")", 1)[1].split()[1]) == sup:
                        kids.add(int(d))
                except (OSError, IndexError, ValueError):
                    pass
        return {sup} | kids

    def _write_wl(self, rec: dict):
        tmp = self.wdir / f".card{self.c}.json.tmp"
        tmp.write_text(json.dumps(rec))
        os.replace(tmp, self.wdir / f"card{self.c}.json")

    def start_workload(self) -> int:
        old = self.workload_pid()
        if old:
            return old
        logf = REPO / "logs/mist_w3po_v1" / f"workload_card{self.c}.log"
        with open(logf, "a") as fh:
            pidf = self.wdir / f".card{self.c}.{int(time.time() * 1000)}.pid"   # fresh name: no stale pid
            # setsid -f + exec: re-parented to init; the pid file holds the supervisor's pid (bash execs python)
            subprocess.run(["setsid", "-f", "bash", "-c", f"echo $$ > {pidf}; exec bash {WORKLOAD} --devices {self.c} "
                            "--tolerate-keeper --wait-idle --idle-seconds 0 --poll-seconds 1 --max-used-mib 0"],
                           stdout=fh, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, cwd=REPO)
        pid = 0
        for _ in range(50):
            try:
                pid = int(pidf.read_text().strip())
                break
            except (OSError, ValueError):
                time.sleep(0.1)
        self._write_wl({"pid": pid, "time": time.time(), "log": str(logf)})
        log(f"card {self.c}: workload supervisor pid {pid} (log {logf})")
        return pid

    def stop_workload(self, wait_s: float = 40) -> bool:
        pids = self.workload_pids()
        sup = self.workload_pid()
        if sup:
            os.kill(sup, signal.SIGTERM)   # its finally: SIGTERM, then SIGKILL after 10 s, to each worker group
        t0 = time.time()
        while time.time() - t0 < wait_s:
            if not any(alive(p) for p in pids):
                break
            time.sleep(0.5)
        ok = not any(alive(p) for p in pids)
        self._write_wl({"pid": 0, "time": time.time(), "stopped": sup})
        log(f"card {self.c}: workload pid {sup} stopped" if ok else f"card {self.c}: workload {sorted(pids)} still alive")
        return ok

    # hand-overs
    def to_workload(self) -> bool:
        """keeper -> workload, step 1: start the workload next to the keeper (step 2: drop_keeper when ready)."""
        if self.workload_pid():
            return True
        sup = self.start_workload()
        time.sleep(3)
        if not alive(sup):
            log(f"card {self.c}: workload supervisor exited at once; keeping the keeper")
            return False
        return True

    def keeper_held_mib(self) -> float:
        try:
            s = json.loads((self.kdir / f"card{self.c}.state").read_text())
            return float(s["held_mib"]) if time.time() - float(s["time"]) < 20 else -1.0
        except Exception:
            return -1.0

    def handover_ready(self, uuid: str, free_mib: float, apps=None) -> bool:
        """the workload's worker holds its memory, the keeper has (almost) nothing left to hand over."""
        return self.workload_mib(uuid, apps) >= READY_MIB and (self.keeper_held_mib() == 0 or free_mib < 30000)

    def drop_keeper(self) -> bool:
        return self.stop_keeper()

    def to_keeper(self) -> bool:
        """workload -> keeper (no free gap: the keeper is up before the workload releases)."""
        kp = self.start_keeper()
        if self.workload_pid():
            self.stop_workload()
        return bool(kp)

    def workload_started_worker(self) -> bool:
        return len(self.workload_pids()) > 1

    def foreign_pids(self, uuid: str, apps=None) -> set:
        """GPU processes on this card that are not ours (other users, also root jobs from other containers)."""
        return {pid for u, pid, _ in (apps if apps is not None else smi_apps()) if u == uuid and not is_ours(pid)}

    def our_job_pids(self, uuid: str, apps=None) -> set:
        """our GPU processes on this card other than its keeper / workload (= W3PO jobs)."""
        skip = {self.keeper_pid()} | self.workload_pids()
        return {pid for u, pid, _ in (apps if apps is not None else smi_apps())
                if u == uuid and pid not in skip and is_ours(pid)}

    def workload_mib(self, uuid: str, apps=None) -> float:
        pids = self.workload_pids()
        return float(sum(m for u, p, m in (apps if apps is not None else smi_apps()) if u == uuid and p in pids))

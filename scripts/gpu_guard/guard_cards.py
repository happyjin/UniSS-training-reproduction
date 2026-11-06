"""Keep the user's GPUs exclusive: every POLL seconds, find compute processes on the protected cards that are not
owned by root (all of the user's jobs run as root; kenxsong's coordinator launches as ec2-user over ssh) and send
them SIGTERM, then SIGKILL after GRACE seconds if still alive. Logs every action. Never touches other cards.

usage: python3 guard_cards.py --cards 0,1,2,3,7 --log logs/gpu_guard/guard.log
stop:  kill -TERM <pid in logs/gpu_guard/guard.pid>
"""
import argparse, os, pwd, signal, subprocess, time

ap = argparse.ArgumentParser()
ap.add_argument("--cards", default="0,1,2,3,7")
ap.add_argument("--poll", type=float, default=5.0)
ap.add_argument("--grace", type=float, default=15.0)
ap.add_argument("--log", required=True)
ap.add_argument("--dry-run", action="store_true")
a = ap.parse_args()
cards = {int(c) for c in a.cards.split(",")}
os.makedirs(os.path.dirname(os.path.abspath(a.log)), exist_ok=True)
with open(os.path.join(os.path.dirname(os.path.abspath(a.log)), "guard.pid"), "w") as f:
    f.write(str(os.getpid()))


def log(msg):
    with open(a.log, "a") as f:
        f.write(time.strftime("[%Y-%m-%d %H:%M:%S UTC] ", time.gmtime()) + msg + "\n")


def smi(*q):
    return subprocess.run(["nvidia-smi", *q, "--format=csv,noheader"], capture_output=True, text=True).stdout


def owner(pid):
    try:
        return pwd.getpwuid(os.stat(f"/proc/{pid}").st_uid).pw_name
    except (FileNotFoundError, KeyError):
        return None


def cmdline(pid):
    try:
        return open(f"/proc/{pid}/cmdline", "rb").read().replace(b"\0", b" ").decode(errors="replace")[:200]
    except FileNotFoundError:
        return ""


pending = {}  # pid -> time SIGTERM was sent
log(f"guard started on cards {sorted(cards)} (dry_run={a.dry_run})")
running = True
signal.signal(signal.SIGTERM, lambda *_: globals().__setitem__("running", False))
while running:
    uuid2idx = {}
    for line in smi("--query-gpu=index,uuid").splitlines():
        i, u = [x.strip() for x in line.split(",")]
        uuid2idx[u] = int(i)
    for line in smi("--query-compute-apps=gpu_uuid,pid,used_memory").splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) < 3 or not parts[1].isdigit():
            continue
        idx, pid = uuid2idx.get(parts[0]), int(parts[1])
        if idx not in cards:
            continue
        user = owner(pid)
        if user is None or user == "root":
            continue
        now = time.time()
        if pid not in pending:
            log(f"card {idx}: pid {pid} user {user} mem {parts[2]} -> SIGTERM | {cmdline(pid)}")
            if not a.dry_run:
                try:
                    os.kill(pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
            pending[pid] = now
        elif now - pending[pid] > a.grace:
            log(f"card {idx}: pid {pid} still alive after {a.grace:.0f}s -> SIGKILL")
            if not a.dry_run:
                try:
                    os.kill(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            pending[pid] = now
    pending = {p: t for p, t in pending.items() if os.path.exists(f"/proc/{p}")}
    time.sleep(a.poll)
log("guard stopped")

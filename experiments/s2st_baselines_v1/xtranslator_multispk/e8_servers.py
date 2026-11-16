"""E8 server helpers: X-Translator's backend services, launched exactly as experiments/baselines_xtranslator_v1/
start_servers.sh does (same commands, same resource flags: LMT bf16 max-model-len 1024 with a fixed 3G KV cache,
Qwen3-ASR via serve_qwen3asr.py with a fixed 1536M KV cache and the adapter's 0.6 s chunk, X-Voice Stage 2 via
serve_xvoice.py), one detached process group per service (setsid), PID / port / card recorded in
eval_outputs/mist_w3po_v1/e8_xtranslator/servers/<name>.json. Resource flags only; decoding is the released config.

Also: GPU memory per card (nvidia-smi), the placeholder stop for one card (train_mist_workload.py supervisor + its
--worker, by PID after a cmdline check; never the guard), health checks, stop-by-PID of our own servers.

CLI (smoke tests / manual use):
    python e8_servers.py start <kind lmt|asr|xvoice> <card> <port> [--dry-run]
    python e8_servers.py stop <name> [--dry-run]       python e8_servers.py status
    python e8_servers.py stop-placeholder <card> [--dry-run]
"""
from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

REPO = Path("/opt/dlami/nvme/jasonleeeli/projects/UniSS")
J = "/opt/dlami/nvme/jasonleeeli"
V = f"{J}/venvs/xtranslator"
M = f"{J}/models/xtranslator"
XV = f"{J}/research_sources/baselines_20261112/X-Voice"
EXP = REPO / "experiments/baselines_xtranslator_v1"
SRV = REPO / "eval_outputs/mist_w3po_v1/e8_xtranslator/servers"
ALLOWED_CARDS = (0, 1, 2, 3)
PROTECTED = {2726951, 52045}   # keep_cards_busy_v3.sh guard, gpu_watch_shell.sh: never stopped
EST_MIB = {"lmt": 22 * 1024, "asr": 14 * 1024, "xvoice": 6 * 1024}   # observed + margin (RealSI notes / smoke)
GIB = 1024


def env_for(card: int) -> dict:
    e = dict(os.environ)
    e.pop("PYTHONPATH", None)   # as start_servers.sh (no repo modules on the servers' path)
    e.update(PATH=f"{V}/bin:" + e.get("PATH", ""), CUDA_VISIBLE_DEVICES=str(card),
             HF_HOME=f"{J}/cache/huggingface", MODELSCOPE_CACHE=f"{J}/cache/modelscope",
             PHONEMIZER_ESPEAK_LIBRARY=f"{V}/opt/espeak-ng/lib/libespeak-ng.so",
             PHONEMIZER_ESPEAK_DATA_PATH=f"{V}/opt/espeak-ng/share/espeak-ng-data",
             TMPDIR=f"{J}/work/e8_xtranslator_tmp")
    e["SSL_CERT_FILE"] = subprocess.run([f"{V}/bin/python", "-m", "certifi"], capture_output=True, text=True,
                                        env=e).stdout.strip()
    os.makedirs(e["TMPDIR"], exist_ok=True)
    return e


def command(kind: str, port: int) -> tuple[list[str], str | None]:
    """start_servers.sh's command lines (cwd for X-Voice)."""
    if kind == "lmt":
        return ([f"{V}/bin/python", "-m", "vllm.entrypoints.openai.api_server", "--host", "127.0.0.1", "--port", str(port),
                 "--trust-remote-code", "--model", f"{M}/LMT-60-8B", "--tensor-parallel-size", "1", "--dtype", "bfloat16",
                 "--served-model-name", "lmt", "--gpu-memory-utilization", "0.30", "--kv-cache-memory-bytes", "3G",
                 "--max-model-len", "1024", "--max-num-seqs", "128"], None)
    if kind == "asr":
        return ([f"{V}/bin/python", str(EXP / "serve_qwen3asr.py"), "--model", f"{M}/Qwen3-ASR-1.7B",
                 "--gpu-memory-utilization", "0.12", "--kv-cache-memory-bytes", "1536M", "--port", str(port),
                 "--chunk-size-sec", "0.6"], None)
    if kind == "xvoice":
        return ([f"{V}/bin/python", str(EXP / "serve_xvoice.py"), "--host", "127.0.0.1", "--port", str(port), "--stage", "2",
                 "--no-auto-detect-lang", "--ckpt-file", f"{M}/X-Voice/XVoice_Base_Stage2/model_70000.safetensors",
                 "--srp-ckpt-file", f"{M}/X-Voice/SpeedPredictor/model_28000.safetensors", "--device", "cuda"], XV)
    raise ValueError(kind)


def name_of(kind: str, port: int) -> str:
    return f"{kind}_{port}"


def cmdline(pid: int) -> str:
    try:
        return Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode(errors="ignore")
    except OSError:
        return ""


def alive(pid: int) -> bool:
    try:
        st = Path(f"/proc/{pid}/stat").read_text()
    except OSError:
        return False
    return st.rsplit(")", 1)[1].split()[0] != "Z"


def gpu_used() -> dict[int, int]:
    q = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
                       capture_output=True, text=True, timeout=60).stdout
    out = {}
    for l in q.strip().split("\n"):
        p = [x.strip() for x in l.split(",")]
        if len(p) == 2 and p[0].isdigit():
            out[int(p[0])] = int(p[1])
    return out


def gpu_apps() -> dict[int, list[tuple[int, int]]]:
    """card -> [(pid, MiB)] of compute processes."""
    u = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"], capture_output=True, text=True,
                       timeout=60).stdout
    u2c = {}
    for l in u.strip().split("\n"):
        p = [x.strip() for x in l.split(",")]
        if len(p) == 2 and p[0].isdigit():
            u2c[p[1]] = int(p[0])
    a = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,gpu_uuid,used_memory", "--format=csv,noheader,nounits"],
                       capture_output=True, text=True, timeout=60).stdout
    out: dict[int, list] = {c: [] for c in u2c.values()}
    for l in a.strip().split("\n"):
        p = [x.strip() for x in l.split(",")]
        if len(p) == 3 and p[0].isdigit() and p[1] in u2c:
            out[u2c[p[1]]].append((int(p[0]), int(p[2]) if p[2].isdigit() else -1))
    return out


def fits(card: int, est_mib: int, cap_gib: int = 135, margin_gib: int = 6) -> tuple[bool, int]:
    used = gpu_used()[card]
    return used + est_mib + margin_gib * GIB <= cap_gib * GIB, used


def placeholder_pids(card: int) -> list[tuple[int, str]]:
    """[(pid, role)] of train_mist_workload.py processes for this card: supervisors (--devices <card>, no --worker) and
    their --worker children (found via the supervisor's children, cmdline checked)."""
    out = []
    ps = subprocess.run(["ps", "-eo", "pid=,ppid=,args="], capture_output=True, text=True).stdout
    rows = []
    for l in ps.split("\n"):
        p = l.strip().split(None, 2)
        if len(p) == 3 and p[0].isdigit():
            rows.append((int(p[0]), int(p[1]), p[2]))
    sups = []
    for pid, ppid, args in rows:
        a = args.split()
        if "train_mist_workload.py" not in args or "--worker" in a:
            continue
        if "--devices" in a and a.index("--devices") + 1 < len(a) and a[a.index("--devices") + 1] == str(card):
            sups.append(pid)
            out.append((pid, "supervisor"))
    for pid, ppid, args in rows:
        if ppid in sups and "train_mist_workload.py" in args and "--worker" in args.split():
            out.append((pid, "worker"))
    return out


def stop_placeholder(card: int, dry: bool = False, log=print) -> bool:
    """stop the card's placeholder supervisor(s) and worker(s) by PID (cmdline re-checked right before each signal)."""
    assert card in ALLOWED_CARDS, card
    P = placeholder_pids(card)
    if not P:
        log(f"card {card}: no placeholder")
        return True
    for pid, role in sorted(P, key=lambda x: x[1] != "supervisor"):   # supervisor first (it stops its worker too)
        assert pid not in PROTECTED
        c = cmdline(pid)
        ok = "train_mist_workload.py" in c and f"--devices {card} " in c + " " and (role == "worker") == ("--worker" in c.split())
        if not ok:
            log(f"card {card}: pid {pid} cmdline changed ({c[:120]}); not stopped")
            return False
        if dry:
            log(f"DRY card {card}: would stop placeholder {role} {pid}: {c[:160]}")
            continue
        try:
            os.kill(pid, signal.SIGTERM)
            log(f"card {card}: SIGTERM placeholder {role} {pid}")
        except ProcessLookupError:
            pass
    if dry:
        return True
    for _ in range(60):
        if not any(alive(p) for p, _ in P):
            break
        time.sleep(1)
    left = [p for p, _ in P if alive(p)]
    if left:
        log(f"card {card}: placeholder pids still alive after 60 s: {left}")
        return False
    return True


def port_open(port: int) -> bool:
    with socket.socket() as s:
        s.settimeout(2)
        try:
            s.connect(("127.0.0.1", port))
            return True
        except OSError:
            return False


def healthy(kind: str, port: int) -> bool:
    if not port_open(port):
        return False
    try:
        if kind == "lmt":
            urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=5).read()
        elif kind == "xvoice":
            r = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{port}/healthz", timeout=5).read())
            return isinstance(r, dict) and r.get("status") == "ok"
        return True
    except Exception:  # noqa: BLE001
        return False


def record(name: str) -> dict | None:
    p = SRV / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else None


def running(name: str) -> dict | None:
    """the recorded server if its process is alive with the recorded command line."""
    r = record(name)
    if r and alive(r["pid"]) and f"--port {r['port']}" in cmdline(r["pid"]):
        return r
    return None


def start(kind: str, card: int, port: int, dry: bool = False, log=print) -> dict | None:
    assert card in ALLOWED_CARDS, card
    name = name_of(kind, port)
    if running(name):
        log(f"{name} already running: {record(name)}")
        return record(name)
    if port_open(port):
        log(f"refuse: port {port} is in use by another process")
        return None
    cmd, cwd = command(kind, port)
    SRV.mkdir(parents=True, exist_ok=True)
    k = 0
    logf = SRV / f"{name}.log"
    while logf.exists():
        k += 1
        logf = SRV / f"{name}.r{k}.log"
    if dry:
        log(f"DRY start {name} card {card} cwd {cwd}: {' '.join(cmd)} > {logf}")
        return None
    env = env_for(card)
    if kind == "xvoice":
        env["XVOICE_ROOT"] = XV
    with open(logf, "x") as fh:
        p = subprocess.Popen(cmd, cwd=cwd, env=env, stdin=subprocess.DEVNULL, stdout=fh, stderr=subprocess.STDOUT,
                             start_new_session=True)
    r = {"name": name, "kind": kind, "card": card, "port": port, "pid": p.pid, "log": str(logf),
         "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "cmd": cmd}
    rec = SRV / f"{name}.json"
    if rec.exists():   # keep the old record (a previous, dead instance)
        k = 0
        while (SRV / f"{name}.dead{k}.json").exists():
            k += 1
        os.rename(rec, SRV / f"{name}.dead{k}.json")
    rec.write_text(json.dumps(r, indent=1))
    log(f"started {name} card {card} pid {p.pid} log {logf}")
    return r


def stop(name: str, dry: bool = False, log=print) -> bool:
    """SIGTERM the server's process group (setsid: pgid = pid; vLLM EngineCore children included), PID checked."""
    r = record(name)
    if not r:
        log(f"{name}: no record")
        return True
    pid = r["pid"]
    if not alive(pid):
        log(f"{name}: pid {pid} not running")
        return True
    c = cmdline(pid)
    if f"--port {r['port']}" not in c or pid in PROTECTED or os.getpgid(pid) != pid:
        log(f"{name}: pid {pid} cmdline / group mismatch ({c[:120]}); not stopped")
        return False
    kids = subprocess.run(["pgrep", "-g", str(pid)], capture_output=True, text=True).stdout.split()
    if dry:
        log(f"DRY stop {name}: group {pid} ({len(kids)} processes)")
        return True
    os.killpg(pid, signal.SIGTERM)
    for _ in range(60):
        if not any(alive(int(k)) for k in kids):
            break
        time.sleep(1)
    left = [int(k) for k in kids if alive(int(k))]
    if left:
        log(f"{name}: still alive after 60 s: {left}; SIGKILL to group {pid}")
        try:
            os.killpg(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    log(f"stopped {name} (group {pid}, {len(kids)} processes)")
    return True


def main() -> None:
    a = sys.argv[1:]
    dry = "--dry-run" in a
    a = [x for x in a if x != "--dry-run"]
    if a and a[0] == "start":
        start(a[1], int(a[2]), int(a[3]), dry)
    elif a and a[0] == "stop":
        stop(a[1], dry)
    elif a and a[0] == "stop-placeholder":
        stop_placeholder(int(a[1]), dry)
    elif a and a[0] == "status":
        for f in sorted(SRV.glob("*.json")):
            if ".dead" in f.name:
                continue
            r = json.loads(f.read_text())
            print(r["name"], "card", r["card"], "pid", r["pid"], "alive", alive(r["pid"]), "healthy",
                  healthy(r["kind"], r["port"]))
        print("gpu used MiB", gpu_used())
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()

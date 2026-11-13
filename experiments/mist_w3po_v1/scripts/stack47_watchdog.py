#!/usr/bin/env python
"""Watchdog of the dev5 decodes stacked on the (b) stage-1 training cards 4-7 (user 2026-11-13 ~06:50 / ~06:55 Prague;
reports/mist_w3po_v1/joint_bidir_plan_prereg_v1.zh-CN.md 修正 2). Detached (setsid nohup). It never touches the training.

  start      creates --stack-file (exclusive; refuses if it exists) with the cards: from now on the orchestrators
             started with --stack-cards <file> may place one shard decode per card there (stack_cards.py)
  watch      every --tick s: nvidia-smi per card; the training log's new lines. From the first decode seen on the cards:
               - any card > --cap-gib (135) GiB used, or < --min-free-gib physically free (default 0: off) -> trip
               - --n-slow (3) consecutive iteration times > baseline x (1 + --slow)    -> trip (baseline 4650 ms, +40 %)
               - a new training-log line with OOM / Traceback / CUDA / NCCL error     -> trip (from the start)
  trip       command mv the stack file to <file>.withdrawn_<UTC> (the orchestrators exclude the cards again at their
             next tick); then SIGTERM by PID every decode process on the cards (cmdline checked: decode_crf_ckpt /
             online_decode, never "pretrain_"), SIGKILL after 30 s what is left (re-checked); the orchestrators retry
             those shards elsewhere. Logs STACK47_WITHDRAWN <reason> and appends one line to LOG.zh-CN.md. Exits.
  training gone (--train-pid exits): the stack file is renamed (STACK47_TRAIN_EXITED), running decodes are left to end.
  stack file renamed by someone else: STACK47_FILE_GONE, exit.

    setsid nohup python stack47_watchdog.py --stack-file F --cards 4 5 6 7 --train-log L --train-pid P \\
        [--baseline-ms 4650] [--slow 0.4] [--n-slow 3] [--cap-gib 135] [--tick 20] [--dry] > log 2>&1 < /dev/null &
"""
from __future__ import annotations

import argparse
import os
import re
import signal
import subprocess
import time
from pathlib import Path

REPO = Path("/opt/dlami/nvme/jasonleeeli/projects/UniSS")
LOG_MD = REPO / "reports/mist_w3po_v1/LOG.zh-CN.md"
ERR_RE = re.compile(r"out of memory|OutOfMemoryError|Traceback \(most recent call last\)|CUDA error|NCCL error|"
                    r"ChildFailedError|RuntimeError", re.I)
IT_RE = re.compile(r"iteration\s+(\d+)/\s*\d+ .*elapsed time per iteration \(ms\): ([0-9.]+)")
DEC_PAT = ("decode_crf_ckpt", "online_decode")


def utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def prague() -> str:
    return time.strftime("%Y-%m-%d %H:%M", time.gmtime(time.time() + 3600))


def log(msg: str) -> None:
    print(f"{utc()} {msg}", flush=True)


def cmdline(pid: int) -> str:
    try:
        return Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode(errors="ignore")
    except OSError:
        return ""


def is_decode(cmd: str) -> bool:
    return any(p in cmd for p in DEC_PAT) and "pretrain_" not in cmd


def smi() -> tuple[dict, dict]:
    """card -> (used MiB, total MiB); card -> [(pid, MiB)]."""
    q = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,memory.used,memory.total", "--format=csv,noheader,nounits"],
                       capture_output=True, text=True, timeout=60).stdout
    mem, u2c = {}, {}
    for l in q.strip().split("\n"):
        p = [x.strip() for x in l.split(",")]
        if len(p) == 4 and p[0].isdigit():
            mem[int(p[0])] = (float(p[2]), float(p[3]))
            u2c[p[1]] = int(p[0])
    a = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,gpu_uuid,used_memory", "--format=csv,noheader,nounits"],
                       capture_output=True, text=True, timeout=60).stdout
    procs: dict = {c: [] for c in mem}
    for l in a.strip().split("\n"):
        p = [x.strip() for x in l.split(",")]
        if len(p) == 3 and p[0].isdigit() and p[1] in u2c:
            procs[u2c[p[1]]].append((int(p[0]), float(p[2]) if p[2].replace(".", "").isdigit() else 0.0))
    return mem, procs


def slow_trip(points: list[tuple[int, float]], baseline: float, slow: float, n: int) -> bool:
    return len(points) >= n and all(ms > baseline * (1 + slow) for _, ms in points[-n:])


def append_log_md(line: str) -> None:
    fd = os.open(LOG_MD, os.O_WRONLY | os.O_APPEND)
    try:
        os.write(fd, ("\n" + line + "\n").encode())
    finally:
        os.close(fd)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stack-file", required=True)
    ap.add_argument("--cards", nargs="+", type=int, required=True)
    ap.add_argument("--train-log", required=True)
    ap.add_argument("--train-pid", type=int, required=True)
    ap.add_argument("--baseline-ms", type=float, default=4650.0)
    ap.add_argument("--slow", type=float, default=0.4)
    ap.add_argument("--n-slow", type=int, default=3)
    ap.add_argument("--cap-gib", type=float, default=135.0)
    ap.add_argument("--min-free-gib", type=float, default=0.0,
                    help="also trip when total - used < this on a card (default 0 = off; user 2026-11-13 07:35: 1.5)")
    ap.add_argument("--tick", type=float, default=20.0)
    ap.add_argument("--warmup-s", type=float, default=0.0,
                    help="create the stack file EMPTY, wait this long, then append the cards (default 0 = off): the "
                         "--stack-cards orchestrators publish their real ready counts first, so the group order holds "
                         "from the first placement")
    ap.add_argument("--status-s", type=float, default=600.0)
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    os.chdir(REPO)
    sf = Path(a.stack_file)
    tl = Path(a.train_log)
    if "pretrain_" not in cmdline(a.train_pid) and "torch.distributed.run" not in cmdline(a.train_pid):
        raise SystemExit(f"--train-pid {a.train_pid} is not a training: {cmdline(a.train_pid)[:200]}")
    if a.dry:
        mem, procs = smi()
        print(f"stack file {sf} exists={sf.exists()}; cards {a.cards}; cap {a.cap_gib} GiB; slow > "
              f"{a.baseline_ms * (1 + a.slow):.0f} ms x {a.n_slow}; train pid {a.train_pid} alive")
        for c in a.cards:
            print(c, f"{mem[c][0] / 1024:.1f} GiB", [(p, m, is_decode(cmdline(p))) for p, m in procs[c]])
        return
    sf.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(sf, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)   # refuses an existing file
    if a.warmup_s > 0:
        log(f"STACK47_WARMUP {sf} created empty; cards appended in {a.warmup_s:.0f} s")
        time.sleep(a.warmup_s)
    os.write(fd, (" ".join(map(str, a.cards)) + "\n").encode())
    os.close(fd)
    log(f"STACK47_ON {sf} = {a.cards}; pid {os.getpid()}; cap {a.cap_gib} GiB; min free {a.min_free_gib} GiB; slow > "
        f"{a.baseline_ms * (1 + a.slow):.0f} ms x {a.n_slow}; training pid {a.train_pid} (never touched)")
    off = tl.stat().st_size if tl.exists() else 0
    buf = ""
    first = None            # time the first decode was seen on the cards
    pts: list[tuple[int, float]] = []
    t_status = 0.0
    seen: set[int] = set()

    def withdraw(reason: str, kill: bool) -> None:
        dst = Path(f"{sf}.withdrawn_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}")
        os.rename(sf, dst)   # new name: nothing is overwritten
        log(f"STACK47_WITHDRAWN {reason}; {sf} -> {dst}")
        killed = []
        if kill:
            _, procs = smi()
            for c in a.cards:
                for p, m in procs.get(c, []):
                    cmd = cmdline(p)
                    if is_decode(cmd):
                        try:
                            os.kill(p, signal.SIGTERM)
                            killed.append(p)
                            log(f"SIGTERM decode pid {p} on card {c} ({m:.0f} MiB): {cmd[:200]}")
                        except ProcessLookupError:
                            pass
            for _ in range(30):
                if not any(Path(f"/proc/{p}").exists() for p in killed):
                    break
                time.sleep(1)
            for p in killed:
                cmd = cmdline(p)
                if cmd and is_decode(cmd):
                    try:
                        os.kill(p, signal.SIGKILL)
                        log(f"SIGKILL decode pid {p}")
                    except ProcessLookupError:
                        pass
        try:
            append_log_md(f"- {prague()} 布拉格（看门狗 stack47_watchdog，pid {os.getpid()}）：STACK47_WITHDRAWN —— {reason}；"
                          f"4–7 号卡重新排除（{dst.name}），按 PID 停了 4–7 上的解码 {killed or '无'}，编排器会在别处重试；"
                          f"(b) 训练未动。")
        except OSError as e:
            log(f"LOG.zh-CN.md append failed: {e!r}")

    while True:
        if not sf.exists():
            log("STACK47_FILE_GONE: the stack file was renamed / removed by someone else; exiting")
            return
        if not Path(f"/proc/{a.train_pid}").exists():
            withdraw(f"training pid {a.train_pid} exited (STACK47_TRAIN_EXITED); running decodes left to end", kill=False)
            return
        try:
            mem, procs = smi()
        except Exception as e:  # noqa: BLE001
            log(f"nvidia-smi failed: {e!r}")
            time.sleep(a.tick)
            continue
        decs = {c: [(p, m) for p, m in procs.get(c, []) if is_decode(cmdline(p))] for c in a.cards}
        for c, ds in decs.items():
            for p, m in ds:
                if p not in seen:
                    seen.add(p)
                    log(f"decode pid {p} on card {c}: card {mem[c][0] / 1024:.1f} GiB")
        if first is None and any(decs.values()):
            first = time.time()
            log("STACK47_FIRST_DECODE " + "; ".join(f"card {c} {mem[c][0] / 1024:.1f} GiB" for c in a.cards))
        # training log: new lines only
        if tl.exists():
            sz = tl.stat().st_size
            if sz > off:
                with open(tl, "rb") as fh:
                    fh.seek(off)
                    buf += fh.read(sz - off).decode(errors="ignore")
                off = sz
            lines, buf = buf.split("\n")[:-1], buf.split("\n")[-1]
            for l in lines:
                if ERR_RE.search(l):
                    withdraw(f"训练日志报错: {l.strip()[:200]}", kill=True)
                    return
                m = IT_RE.search(l)
                if m and first is not None:
                    pts.append((int(m.group(1)), float(m.group(2))))
        over = [c for c in a.cards if mem[c][0] > a.cap_gib * 1024 or mem[c][1] - mem[c][0] < a.min_free_gib * 1024]
        if over and first is not None:
            withdraw("卡 " + ", ".join(f"{c}: {mem[c][0] / 1024:.1f} GiB（剩 {(mem[c][1] - mem[c][0]) / 1024:.1f}）" for c in over)
                     + f" > {a.cap_gib} GiB 或剩余 < {a.min_free_gib} GiB",
                     kill=True)
            return
        if slow_trip(pts, a.baseline_ms, a.slow, a.n_slow):
            withdraw(f"训练步时连续 {a.n_slow} 个日志点 > 基线 {a.baseline_ms:.0f} ms 的 {1 + a.slow:.1f} 倍: "
                     f"{pts[-a.n_slow:]}", kill=True)
            return
        if time.time() - t_status >= a.status_s:
            t_status = time.time()
            log("status: " + "; ".join(f"card {c} {mem[c][0] / 1024:.1f} GiB dec {[p for p, _ in decs[c]]}"
                                       for c in a.cards) + f"; last iterations {pts[-3:]}")
        time.sleep(a.tick)


if __name__ == "__main__":
    main()

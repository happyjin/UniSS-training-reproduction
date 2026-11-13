#!/usr/bin/env python
"""Restart one running eval orchestrator (thinker_t_eval_orch / joint_orch / en2zh_orch) with --resume and extra
arguments (2026-11-13: --stack-cards ... for the (b) dev5 proxies), keeping its exact command line, environment, cwd and
launch log: reads /proc/<pid>/{cmdline,environ,cwd,fd/1}, checks the cmdline (an *_orch.py with --run), SIGTERM by
PID (the orchestrator has no handler; its jobs run in their own sessions and are adopted by --resume), waits until it is
gone, then starts the same argv + --resume + EXTRA in a new session (stdin /dev/null, stdout/stderr appended to the same
launch log, no inherited fds: Popen close_fds). Prints the new PID.

    python relaunch_orch_stack.py --pid P [--dry] -- <extra args...>
"""
from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pid", type=int, required=True)
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("extra", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    extra = [x for x in a.extra if x != "--"] if a.extra[:1] == ["--"] else a.extra
    P = Path(f"/proc/{a.pid}")
    argv = [x.decode() for x in (P / "cmdline").read_bytes().split(b"\0") if x]
    if not (len(argv) >= 2 and argv[1].endswith("_orch.py") and "--run" in argv):
        raise SystemExit(f"pid {a.pid} is not an eval orchestrator: {argv[:4]}")
    env = dict(x.decode(errors="surrogateescape").split("=", 1) for x in (P / "environ").read_bytes().split(b"\0")
               if b"=" in x)
    cwd = os.readlink(P / "cwd")
    out = os.readlink(P / "fd" / "1")
    if not out.startswith("/") or not Path(out).exists():
        raise SystemExit(f"stdout of {a.pid} is {out}: not a log file")
    new = argv + ([] if "--resume" in argv else ["--resume"]) + extra
    print(f"pid {a.pid}: cwd {cwd}; log {out}; env {len(env)} vars\n  old: {' '.join(argv)}\n  new: {' '.join(new)}")
    if a.dry:
        return
    os.kill(a.pid, signal.SIGTERM)
    for _ in range(60):
        if not P.exists():
            break
        time.sleep(0.5)
    if P.exists():
        raise SystemExit(f"pid {a.pid} still alive after 30 s; not relaunched")
    with open(out, "ab") as fh:
        fh.write(f"\n=== {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} relaunch_orch_stack: pid {a.pid} stopped; "
                 f"restarted with --resume {' '.join(extra)} ===\n".encode())
        fh.flush()
        p = subprocess.Popen(new, cwd=cwd, env=env, stdin=subprocess.DEVNULL, stdout=fh, stderr=subprocess.STDOUT,
                             start_new_session=True, close_fds=True)
    print(f"new pid {p.pid}")
    sys.stdout.flush()


if __name__ == "__main__":
    main()

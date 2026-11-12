"""R1 queue for the registered SeamlessStreaming RealSI grid (paper/baselines_run_plan_seamless_v1.md). CPU only; it
launches, watches and post-processes the GPU jobs, never touching anyone else's process.

Jobs per (grid point, direction): K generation shards (run_seamless.py, --resume) -> merge + pre-ASR score ->
speech_only.py + evaluation.asr_transcribe (Whisper-large-v3 / Paraformer-large) -> score_seamless.py --require-asr ->
paper/baselines_results/seamless/<tag>_<dir>.json (refuses existing files).

GPU rules (coordinator, 2026-11-12):
  cards 4-7: card total <= 135 GiB; launch only if used + job estimate + 6 GiB margin fits; stagger starts 60 s per
  card, and no launch on a card where someone else's GPU process (e.g. a vLLM engine) appeared < 90 s ago.
  cards 0-3 (D1v training): only when --use-train-cards and the training has logged >= 20 steps; my jobs <= 45 GiB per
  card, at most 2 jobs there (~24 GiB, margin for the training peak); at most 6 of my jobs on each of cards 4-7; evicted (all my jobs there, cards disabled) if the training log shows an OOM / error, or
  its median step time over the last 15 steps exceeds 1.4 x the baseline measured before my first launch there.
  Any card whose total passes its limit: kill MY newest job there (PID + cmdline check), requeue it (--resume).
  Placeholders (train_mist_workload.py) on a card are left alone: launches there only use the free memory.
Everything is logged to logs/s2st_baselines_v1/realsi_queue_<ts>.log; state in <work>/queue_state.json.

    PYTHONPATH=. /opt/dlami/nvme/jasonleeeli/conda_envs/uniss-eval/bin/python \
        experiments/s2st_baselines_v1/seamless/realsi_queue.py --grid <grid.json> [--use-train-cards] [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import signal
import statistics as st
import subprocess
import sys
import time
from pathlib import Path

REPO = Path("/opt/dlami/nvme/jasonleeeli/projects/UniSS")
OUT = REPO / "eval_outputs/s2st_baselines_v1/seamless/realsi"
RES = REPO / "paper/baselines_results/seamless"
LOGD = REPO / "logs/s2st_baselines_v1/realsi"
ENV = "source experiments/s2st_baselines_v1/seamless/env.sh"
PYS = "/opt/dlami/nvme/jasonleeeli/venvs/s2st_baselines/bin/python"
PYE = "/opt/dlami/nvme/jasonleeeli/conda_envs/uniss-eval/bin/python"
TRAIN_LOG = REPO / "logs/mist_w3po_v1/d1v_20261112T120413Z.log"
GIB = 1024
LIMIT = {g: 135 * GIB for g in range(4, 8)} | {g: 139 * GIB for g in range(4)}   # card totals (MiB)
EST = {"gen": 12 * GIB, "asr": 10 * GIB}
MARGIN = 6 * GIB   # coordinator: new job peak + existing <= 135 GiB, plus ~5 GiB headroom
TRAIN_MAX_MINE = 45 * GIB
STAGGER = 60
MAX_MINE = {g: 6 for g in range(4, 8)} | {g: 2 for g in range(4)}   # leave room for other agents / the training peak
SHARDS = {"en2zh": 4, "zh2en": 5}


def now() -> str:
    return time.strftime("%H:%M:%S")


class Q:
    def __init__(self, a):
        self.a = a
        self.ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        LOGD.mkdir(parents=True, exist_ok=True)
        self.log = open(REPO / f"logs/s2st_baselines_v1/realsi_queue_{self.ts}.log", "x", encoding="utf-8")
        self.grid = json.load(open(a.grid, encoding="utf-8"))
        self.jobs = []   # dicts: kind, tag, dir, shard, state (wait/run/done/fail), pid, card, attempts
        for p in self.grid["points"]:
            for d in self.grid["directions"]:
                for k in range(SHARDS[d]):
                    self.jobs.append({"kind": "gen", "tag": p["tag"], "dir": d, "shard": k, "cfg": p, "state": "wait",
                                      "attempts": 0})
                self.jobs.append({"kind": "asr", "tag": p["tag"], "dir": d, "shard": None, "cfg": p, "state": "wait",
                                  "attempts": 0})
        self.last_start = {}
        self.cooldown = {}
        self.train_ok = a.use_train_cards
        self.train_base = None
        self.cards = list(range(4, 8)) + (list(range(4)) if a.use_train_cards else [])
        self.seen = None   # (card, pid) -> first seen; processes present at start count as old

    def say(self, msg: str) -> None:
        line = f"[{now()}] {msg}"
        print(line, flush=True)
        self.log.write(line + "\n")
        self.log.flush()

    # ------------------------------------------------------------- GPU state
    def gpu(self) -> tuple[dict, dict]:
        """card -> used MiB; card -> [(pid, MiB)] of all compute processes"""
        q = lambda f: subprocess.run(["nvidia-smi", f"--query-{f}", "--format=csv,noheader,nounits"],  # noqa: E731
                                     capture_output=True, text=True, timeout=60).stdout.splitlines()
        used, uuid = {}, {}
        for l in q("gpu=index,uuid,memory.used"):
            i, u, m = [x.strip() for x in l.split(",")]
            used[int(i)] = int(m)
            uuid[u] = int(i)
        procs = {i: [] for i in used}
        for l in q("compute-apps=gpu_uuid,pid,used_memory"):
            u, p, m = [x.strip() for x in l.split(",")]
            if u in uuid and m.isdigit():
                procs[uuid[u]].append((int(p), int(m)))
        return used, procs

    def mine(self, card: int) -> list[dict]:
        return [j for j in self.jobs if j["state"] == "run" and j["card"] == card]

    # ------------------------------------------------------------- training guard
    def train_steps(self) -> list[float]:
        t = []
        try:
            lines = open(TRAIN_LOG, encoding="utf-8", errors="replace").read().splitlines()[-400:]
        except OSError:
            return t
        for l in lines:
            m = re.match(r"(\d\d):(\d\d):(\d\d) \{\"step\"", l)
            if m:
                t.append(int(m.group(1)) * 3600 + int(m.group(2)) * 60 + int(m.group(3)))
        return t

    def step_time(self, n: int = 15):
        t = self.train_steps()
        if len(t) < n + 1:
            return None, len(t)
        d = [b - a if b >= a else b + 86400 - a for a, b in zip(t[-n - 1:-1], t[-n:])]
        return st.median(d), len(t)

    def train_check(self) -> None:
        if not self.train_ok:
            return
        try:
            tail = open(TRAIN_LOG, encoding="utf-8", errors="replace").read()[-20000:]
        except OSError:
            tail = ""
        bad = re.search(r"out of memory|OutOfMemoryError|Traceback", tail)
        med, n = self.step_time()
        on_train = any(self.mine(c) for c in range(4))
        if self.train_base is None:
            if med is not None and n >= 20:
                self.train_base = med
                self.say(f"training baseline step time {med:.2f} s (n steps {n})")
            return
        if bad or (on_train and med is not None and med > 1.4 * self.train_base):
            self.say(f"EVICT training cards: {'log error' if bad else f'step {med:.2f} s > 1.4 x {self.train_base:.2f}'}")
            self.train_ok = False
            for c in range(4):
                for j in self.mine(c):
                    self.kill(j, "training guard")
            self.cards = [c for c in self.cards if c >= 4]
            Path(REPO / f"logs/s2st_baselines_v1/TRAIN_EVICT_{self.ts}").write_text(
                f"{now()} {'log error' if bad else f'median step {med:.2f} s vs baseline {self.train_base:.2f}'}\n")

    # ------------------------------------------------------------- jobs
    def run_dir(self, j) -> Path:
        return OUT / j["tag"] / f"realsi_{j['dir']}"

    def logf(self, j) -> Path:
        s = f"_s{j['shard']}" if j["kind"] == "gen" else "_asr"
        return LOGD / f"{j['tag']}_{j['dir']}{s}.a{j['attempts']}.log"

    def cmd(self, j, card: int) -> str:
        c, rd = j["cfg"], self.run_dir(j)
        if j["kind"] == "gen":
            return (f"cd {REPO} && {ENV} && CUDA_VISIBLE_DEVICES={card} exec $PYS "
                    f"experiments/s2st_baselines_v1/seamless/run_seamless.py --direction {j['dir']} "
                    f"--decision-threshold {c['decision_threshold']} --min-starting-wait-w2vbert {c['min_starting_wait_w2vbert']} "
                    f"--min-unit-chunk-size {c['min_unit_chunk_size']} --dtype {c.get('dtype', 'fp32')} --out {rd} "
                    f"--shard {j['shard']} --num-shards {SHARDS[j['dir']]} --resume")
        res = RES / f"{j['tag']}_{j['dir']}.json"
        return (f"cd {REPO} && {ENV} && set -e && "
                f"$PYE experiments/s2st_baselines_v1/seamless/score_seamless.py {rd} && "
                f"$PYE reports/uniss_si_v4/speech_only.py {rd} && "
                f"CUDA_VISIBLE_DEVICES={card} MODELSCOPE_CACHE=/opt/dlami/nvme/jasonleeeli/cache/modelscope "
                f"$PYE -m evaluation.asr_transcribe --input {rd}/results_speechonly.jsonl --output {rd}/asr_speech.jsonl "
                f"--device cuda:0 --batch-size 4 --resume && "
                f"$PYE experiments/s2st_baselines_v1/seamless/score_seamless.py {rd} --require-asr --out {res} && "
                f"echo ASR_SCORE_DONE")

    def ready(self, j) -> bool:
        if j["kind"] == "gen":
            return True
        return all(x["state"] == "done" for x in self.jobs
                   if x["kind"] == "gen" and x["tag"] == j["tag"] and x["dir"] == j["dir"])

    def launch(self, j, card: int) -> None:
        j["attempts"] += 1
        lf = self.logf(j)
        if lf.exists():
            raise SystemExit(f"refuse: {lf} exists")
        c = self.cmd(j, card)
        if self.a.dry_run:
            self.say(f"DRY {j['kind']} {j['tag']} {j['dir']} {j['shard']} card {card}: {c}")
            j["state"], j["card"], j["pid"] = "done", card, None
            return
        p = subprocess.Popen(["bash", "-c", c], stdout=open(lf, "x"), stderr=subprocess.STDOUT, cwd=REPO,
                             start_new_session=True)
        j.update(state="run", card=card, pid=p.pid, proc=p, t0=time.time(), log=str(lf))
        self.last_start[card] = time.time()
        self.say(f"start {j['kind']} {j['tag']} {j['dir']} shard {j['shard']} card {card} pid {p.pid} -> {lf.name}")

    def kill(self, j, why: str) -> None:
        pid = j.get("pid")
        try:
            cl = open(f"/proc/{pid}/cmdline", "rb").read().replace(b"\0", b" ").decode()
        except OSError:
            cl = ""
        if pid and ("run_seamless.py" in cl or "asr_transcribe" in cl or "score_seamless" in cl or "speech_only" in cl
                    or cl.startswith("bash -c cd /opt/dlami/nvme/jasonleeeli/projects/UniSS")):
            try:
                os.killpg(pid, signal.SIGTERM)   # its own session (start_new_session): only my job's processes
            except ProcessLookupError:
                pass
            self.say(f"KILL {j['kind']} {j['tag']} {j['dir']} shard {j['shard']} card {j['card']} pid {pid}: {why}")
        else:
            self.say(f"kill skipped (cmdline mismatch) pid {pid}: {cl[:80]!r}")
        try:
            j["proc"].wait(timeout=60)
        except Exception:
            pass
        j["state"] = "wait"
        self.cooldown[j["card"]] = time.time() + 300

    def poll(self) -> None:
        for j in self.jobs:
            if j["state"] != "run":
                continue
            rc = j["proc"].poll()
            if rc is None:
                continue
            txt = open(j["log"], encoding="utf-8", errors="replace").read()[-4000:]
            ok = ("SHARD_DONE" in txt) if j["kind"] == "gen" else ("ASR_SCORE_DONE" in txt)
            if rc == 0 and ok:
                j["state"] = "done"
                self.say(f"done {j['kind']} {j['tag']} {j['dir']} shard {j['shard']} ({(time.time() - j['t0']) / 60:.1f} min)")
            else:
                j["state"] = "wait" if j["attempts"] < 3 else "fail"
                self.say(f"FAILED rc {rc} {j['kind']} {j['tag']} {j['dir']} shard {j['shard']} -> {j['state']}: "
                         f"{txt[-300:]!r}")

    def track(self, procs: dict) -> None:
        t = time.time()
        first = self.seen is None
        if first:
            self.seen = {}
        for c, ps in procs.items():
            for p, _ in ps:
                if (c, p) not in self.seen:
                    self.seen[(c, p)] = 0.0 if first else t

    def foreign_recent(self, c: int, procs: dict, window: float = 90.0) -> bool:
        """someone else's GPU process (e.g. a vLLM engine) appeared on card c less than `window` s ago"""
        mine = {j["pid"] for j in self.jobs if j["state"] == "run"}
        t = time.time()
        return any(t - self.seen.get((c, p), 0.0) < window for p, _ in procs.get(c, [])
                   if p not in mine and not self._child_of(p, mine))

    def guard(self, used: dict) -> None:
        for c in self.cards:
            if used.get(c, 0) > LIMIT[c]:
                mine = sorted(self.mine(c), key=lambda j: j["t0"])
                if mine:
                    self.kill(mine[-1], f"card {c} at {used[c]} MiB > {LIMIT[c]}")

    def schedule(self, used: dict, procs: dict) -> None:
        t = time.time()
        for c in self.cards:
            if t < self.cooldown.get(c, 0) or t - self.last_start.get(c, 0) < STAGGER:
                continue
            if c < 4:
                if not self.train_ok or self.train_base is None:
                    continue
            if self.foreign_recent(c, procs):
                continue
            nxt = next((j for j in self.jobs if j["state"] == "wait" and self.ready(j) and
                        (j["kind"] == "asr") ), None) or next(
                (j for j in self.jobs if j["state"] == "wait" and self.ready(j)), None)
            if nxt is None:
                return
            if len(self.mine(c)) >= MAX_MINE[c]:
                continue
            est = EST[nxt["kind"]]
            if used.get(c, 0) + est + MARGIN > LIMIT[c] - (4 * GIB if c < 4 else 0):
                continue
            if c < 4:
                mine_pids = {j["pid"] for j in self.mine(c)}
                mine_mem = sum(m for p, m in procs.get(c, []) if p in mine_pids or self._child_of(p, mine_pids))
                if mine_mem + est > TRAIN_MAX_MINE:
                    continue
            self.launch(nxt, c)
            used[c] = used.get(c, 0) + est
            t = time.time()

    @staticmethod
    def _child_of(pid: int, roots: set) -> bool:
        p = pid
        for _ in range(6):
            try:
                p = int(open(f"/proc/{p}/stat").read().rsplit(")", 1)[1].split()[1])
            except (OSError, ValueError, IndexError):
                return False
            if p in roots:
                return True
            if p <= 1:
                return False
        return False

    def loop(self) -> None:
        self.say(f"queue {len(self.jobs)} jobs; grid {self.a.grid}; cards {self.cards}; train cards {self.train_ok}")
        while True:
            self.poll()
            self.train_check()
            used, procs = self.gpu()
            self.track(procs)
            self.guard(used)
            self.schedule(used, procs)
            left = [j for j in self.jobs if j["state"] in ("wait", "run")]
            if not left:
                break
            time.sleep(1 if self.a.dry_run else 15)
        fails = [j for j in self.jobs if j["state"] == "fail"]
        self.say(f"QUEUE_DONE fails {len(fails)}: {[(j['kind'], j['tag'], j['dir'], j['shard']) for j in fails]}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--grid", required=True)
    ap.add_argument("--use-train-cards", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="print the commands, launch nothing")
    a = ap.parse_args()
    Q(a).loop()


if __name__ == "__main__":
    main()

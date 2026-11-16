"""E8 queue v2 (2026-11-16 ~20:05 Prague): e8_queue.py (not changed; stopped by PID) with three scheduling / logging fixes,
adopting everything the first queue left running. Jobs, servers, chain, outputs and refusals are e8_queue.py's.
  * ASR (Whisper / Paraformer) jobs: estimate 24 GiB (observed 22.7 GiB, was 16); while the X-Translator servers run,
    at most 1 of my ASR jobs per card (2 after the servers are stopped) and no ASR launch on a card within 120 s of my
    previous ASR launch there (two jobs launched 11 s apart on one card before the first one's memory showed: 140 GiB)
  * the job log redirect wraps the whole command (it bound only to the trailing `touch`: ASR logs were lost)
  * adoption: running run_e8.py drivers (as before) and running evaluation.asr_transcribe jobs of our runs (watched by
    PID; done iff the step marker the old wrapper touches exists, else retried with --resume)

e8_queue.py docstring:
E8 queue (reports/mist_w3po_v1/multispk_s2st_baselines_prereg_v2.zh-CN.md §1 S2 / §2 / §4): X-Translator xt-default on
the three multi-speaker online test sets, each decoded once, then scored with our own tools exactly as E7
(experiments/s2st_baselines_v1/seamless_multispk/e7_queue_v2.py, copied, not changed). CPU process; launches and
watches only its own servers and jobs, on cards 0-3 only.

Servers (e8_servers.py = start_servers.sh's commands): LMT (vLLM) per PLAN, and per slot one Qwen3-ASR (vLLM) + one
X-Voice; a slot = one driver at a time (as RealSI: one ASR / X-Voice pair per shard). Placeholders: a card's
train_mist_workload.py supervisor + worker are stopped by PID (cmdline checked) right before the first launch there;
the guard keep_cards_busy_v3.sh / gpu_watch_shell.sh are never touched. vLLM engines start >= 60 s apart (global);
every launch only if the card's used memory + estimate + 6 GiB <= 135 GiB (nvidia-smi read right before). Running
servers recorded by e8_servers are adopted (smoke-test servers, queue restarts); a server that dies is restarted
(<= 3 times) and its slot waits until it is healthy again.

Per set (ami: AMI zh->en, en: AMI En->Zh, ali: AliMeeting zh->en; the E7 window / reference files):
  1. decode     run_e8.py shards (W[k::K], --resume), each on an idle healthy slot; exit 3 = one of our servers
                was unreachable: requeued (the window was not written), the slot rechecked
  2. post (thread)  build_e8 -> score_run (main, _1spk) -> p80_speechonly -> chunk-protocol text latency
  3. ASR        evaluation.asr_transcribe (Whisper-large-v3 en / Paraformer zh), 4 GPU shards, same commands as E7
  4. post (thread)  asr_merge -> asr_hyp -> score_run --asr -> chunk-protocol speech latency (E7's lat) ->
                table1_lat.sh (lat_units_dump perword + blockstart, the Table 1 latency units)
  5. marker logs/mist_w3po_v1/e8/E8_<set>_DONE
When no decode job is left, all X-Translator servers are stopped by PID (the guard then refills idle cards); an ASR
job that needs a card held by a placeholder stops that placeholder first (PID, cmdline checked).
Step markers (logs/mist_w3po_v1/e8/markers) make a restarted queue skip finished steps; outputs are never overwritten.

    PYTHONPATH=. $PY experiments/s2st_baselines_v1/xtranslator_multispk/e8_queue.py [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e8_servers as ES  # noqa: E402

REPO = ES.REPO
J = ES.J
CODE = "experiments/s2st_baselines_v1/xtranslator_multispk"
RAW = "eval_outputs/mist_w3po_v1/e8_xtranslator/raw"
DEC = "eval_outputs/mist_w3po_v1/decode"
SPD = "eval_outputs/mist_w3po_v1/speech"
TE = "eval_outputs/mist_w3po_v1/crf/t_eval"
LOGD = REPO / "logs/mist_w3po_v1/e8"
MK = LOGD / "markers"
PY = f"{J}/conda_envs/qwen3omni/bin/python"
PYE = f"{J}/conda_envs/uniss-eval/bin/python"
V = ES.V
ENV_E = (f"export TMPDIR={J}/work/mist_w3po_v1/tmp HF_HOME={J}/cache/huggingface TORCH_HOME={J}/cache/torch "
         f"XDG_CACHE_HOME={J}/cache MODELSCOPE_CACHE={J}/cache/modelscope PYTHONPATH=.:third_party/Megatron-LM "
         f"OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false; ")
ENV_D = (f"unset PYTHONPATH; export PATH={V}/bin:$PATH HF_HOME={J}/cache/huggingface MODELSCOPE_CACHE={J}/cache/modelscope "
         f"TMPDIR={J}/work/e8_xtranslator_tmp OMP_NUM_THREADS=4 SSL_CERT_FILE=$({V}/bin/python -m certifi); ")
DD = "eval_outputs/mist_w3po_v1/data"
SETS = {   # e7_queue_v2.SETS (+ direction, shard count ~ 12 min of source audio per shard)
    "ami": {"win": f"{DD}/windows_test_c2g06_v1/windows_test_online_zh2en.jsonl",
            "cwin": f"{DD}/windows_test_c2g06_chunk_v1/zh2en/windows_dev_online.jsonl",
            "ref": "data/ami_multispk_si_v1/e2e_gstar_zh2en_v1/windows_test_ref.jsonl", "set": "zh2en",
            "tgt": "eng", "src": "cmn", "dir": "zh2en", "shards": 30},
    "en": {"win": f"{DD}/windows_test_en2zh_c2g06_v1/windows_test_online_en2zh.jsonl",
           "cwin": f"{DD}/windows_test_en2zh_c2g06_chunk_v1/windows_dev_online.jsonl",
           "ref": "data/ami_multispk_si_v1/e2e_gstar_v1/windows_test_ref.jsonl", "set": "en2zh",
           "tgt": "cmn", "src": "eng", "dir": "en2zh", "shards": 36},
    "ali": {"win": f"{DD}/windows_test_c2g06_v1/windows_test_online_ali.jsonl",
            "cwin": f"{DD}/windows_test_c2g06_chunk_v1/ali/windows_dev_online.jsonl",
            "ref": "data/diarist_alimeeting_refs_v1/windows_test_ref_human.jsonl", "set": "ali",
            "tgt": "eng", "src": "cmn", "dir": "zh2en", "shards": 50},
}
ASR_SHARDS = 4
GIB = 1024
EST = {"dec": 3 * GIB, "asr": 24 * GIB}   # v2: asr 24 (observed 22.7)
ASR_HOLD = 120   # v2: s between two of my ASR launches on one card
CAP_GIB, MARGIN_GIB = 135, 6
STAGGER_VLLM = 60
STAGGER_ANY = 10
MAX_ASR_PER_CARD = 2
CARDS = (0, 1, 2, 3)
# servers: LMT per card that hosts one; slots per card (ASR port 19001 + 8c + i, X-Voice 19201 + 8c + i)
LMT = {0: 19100, 2: 19101}
LMT_OF = {0: 19100, 1: 19100, 2: 19101, 3: 19101}
SLOTS_PER_CARD = {0: 6, 1: 7, 2: 6, 3: 6}   # measured: ASR 13.0, X-Voice ~1.8, driver ~0.7, LMT 20.7 GiB


def slot_list() -> list[dict]:
    out = []
    for i in range(max(SLOTS_PER_CARD.values())):
        for c in CARDS:
            if i < SLOTS_PER_CARD[c]:
                out.append({"id": f"c{c}s{i}", "card": c, "asr": 19001 + 8 * c + i, "tts": 19201 + 8 * c + i,
                            "mt": LMT_OF[c], "job": None})
    return out


def launch_plan(slots: list[dict]) -> list[tuple[str, int, int]]:
    """(kind, card, port) in launch order: a card's LMT before its first slot; X-Voice right before its ASR."""
    plan, seen = [], set()
    for s in slots:
        c = s["card"]
        if c not in seen:
            seen.add(c)
            if c in LMT:
                plan.append(("xvoice", c, s["tts"]))
                plan.append(("lmt", c, LMT[c]))
                plan.append(("asr", c, s["asr"]))
                continue
        plan.append(("xvoice", c, s["tts"]))
        plan.append(("asr", c, s["asr"]))
    return plan


def name(s: str) -> str:
    return f"jtest_{s}_xtranslator_default"


class Job:
    def __init__(self, jid: str, kind: str, cmd: str, log: Path, setk: str, shard: int = -1):
        self.id, self.kind, self.cmd, self.log, self.set, self.shard = jid, kind, cmd, log, setk, shard
        self.proc = None
        self.pid = None
        self.card = None
        self.slot = None
        self.started = 0.0
        self.tries = 0


class Q:
    def __init__(self, a):
        self.a = a
        LOGD.mkdir(parents=True, exist_ok=True)
        MK.mkdir(parents=True, exist_ok=True)
        self.logf = open(LOGD / ("e8_queue_dry.log" if a.dry_run else "e8_queue.log"), "a", encoding="utf-8")
        self.lock = threading.RLock()
        self.slots = slot_list()
        self.plan = launch_plan(self.slots)
        self.restarts: dict[str, int] = {}
        self.pending: list[Job] = []
        self.running: list[Job] = []
        self.last_launch = 0.0
        self.last_srv = 0.0
        self.last_vllm = 0.0
        self.stage = {s: "decode" for s in SETS}
        self.threads: dict[str, threading.Thread] = {}
        self.failed: list[str] = []
        self.servers_stopped = False
        self.adopted_asr: dict = {}
        self.asr_launch: dict[int, float] = {}
        self.health: dict[str, bool] = {}
        self.last_health = 0.0
        self.expected = {}
        for s, S in SETS.items():
            W = [json.loads(l)["id"] for l in open(REPO / S["win"], encoding="utf-8")]
            self.expected[s] = {k: set(W[k:: S["shards"]]) for k in range(S["shards"])}

    def log(self, msg: str) -> None:
        line = f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {msg}"
        with self.lock:
            print(line, flush=True)
            self.logf.write(line + "\n")
            self.logf.flush()

    def mark(self, s: str, step: str) -> Path:
        return MK / f"{s}.{step}.ok"

    # ------------------------------------------------------------------ servers
    def server_ok(self, kind: str, port: int) -> bool:
        return self.health.get(ES.name_of(kind, port), False)

    def refresh_health(self) -> None:
        if time.time() - self.last_health < 30:
            return
        self.last_health = time.time()
        for kind, c, port in self.plan:
            n = ES.name_of(kind, port)
            r = ES.running(n)
            self.health[n] = bool(r) and ES.healthy(kind, port)

    def slot_ready(self, sl: dict) -> bool:
        return (self.server_ok("asr", sl["asr"]) and self.server_ok("xvoice", sl["tts"])
                and self.server_ok("lmt", sl["mt"]))

    def card_touched(self, c: int) -> bool:
        return any(ES.record(ES.name_of(k, p)) for k, cc, p in self.plan if cc == c)

    def manage_servers(self) -> None:
        """launch the next planned server that is not running (first launches and restarts of dead ones)."""
        if self.servers_stopped or self.a.dry_run:
            return
        now = time.time()
        if now - self.last_srv < STAGGER_ANY:
            return
        for kind, c, port in self.plan:
            n = ES.name_of(kind, port)
            if ES.running(n):
                continue
            rec = ES.record(n)
            if rec is not None:   # launched before and now dead
                k = self.restarts.get(n, 0)
                if k >= 3:
                    continue
                if not rec.get("dead_logged"):
                    self.log(f"SERVER_DEAD {n} pid {rec['pid']} (restart {k + 1}/3); log {rec['log']}")
                    rec["dead_logged"] = True
                    (ES.SRV / f"{n}.json").write_text(json.dumps(rec, indent=1))
            vllm = kind in ("lmt", "asr")
            if vllm and now - self.last_vllm < STAGGER_VLLM:
                return
            if not self.card_touched(c) or ES.placeholder_pids(c):
                mine = [j for j in self.running if j.card == c]
                if ES.placeholder_pids(c) and not mine and not any(ES.running(ES.name_of(k_, p_))
                                                                    for k_, cc, p_ in self.plan if cc == c):
                    self.log(f"card {c}: stopping placeholder before the first launch")
                    if not ES.stop_placeholder(c, log=self.log):
                        self.log(f"card {c}: placeholder stop failed; retry later")
                        return
                    for _ in range(30):
                        if ES.gpu_used()[c] < 4 * GIB:
                            break
                        time.sleep(2)
            ok, used = ES.fits(c, ES.EST_MIB[kind], CAP_GIB, MARGIN_GIB)
            if not ok:
                return   # wait (launch order is kept)
            if rec is not None:
                self.restarts[n] = self.restarts.get(n, 0) + 1
            r = ES.start(kind, c, port, log=self.log)
            self.last_srv = time.time()
            if vllm:
                self.last_vllm = self.last_srv
            if r is None:
                self.log(f"launch of {n} refused")
            return

    def stop_all_servers(self) -> None:
        for kind, c, port in reversed(self.plan):
            ES.stop(ES.name_of(kind, port), log=self.log)
        self.servers_stopped = True
        self.log("SERVERS_STOPPED (all X-Translator servers of E8)")

    # ------------------------------------------------------------------ jobs
    def decode_jobs(self, s: str) -> list[Job]:
        S = SETS[s]
        out = []
        for k in range(S["shards"]):
            if self.shard_done(s, k):
                continue
            out.append(Job(f"{s}__dec{k:02d}", "dec", "", LOGD / f"{s}_dec{k:02d}.log", s, k))
        return out

    def shard_done(self, s: str, k: int) -> bool:
        p = REPO / RAW / s / f"part_{k:02d}.jsonl"
        if not p.exists():
            return False
        got = set()
        for l in open(p, encoding="utf-8"):
            try:
                got.add(json.loads(l)["id"])
            except (json.JSONDecodeError, KeyError):
                pass
        return self.expected[s][k] <= got

    def dec_cmd(self, j: Job, sl: dict) -> str:
        S = SETS[j.set]
        return (ENV_D + f"{V}/bin/python -u {CODE}/run_e8.py --windows {S['win']} --direction {S['dir']} "
                f"--out {RAW}/{j.set} --asr-port {sl['asr']} --tts-port {sl['tts']} --mt-port {sl['mt']} "
                f"--shard {j.shard} --num-shards {S['shards']} --resume")

    def launch(self, j: Job, c: int, cmd: str) -> None:
        j.tries += 1
        j.card = c
        full = f"cd {REPO} && export CUDA_VISIBLE_DEVICES={c}; ( {cmd} ) >> {j.log} 2>&1"   # v2: whole command
        j.proc = subprocess.Popen(["bash", "-c", full], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                  stderr=subprocess.DEVNULL, start_new_session=True)
        j.started = self.last_launch = time.time()
        self.running.append(j)
        self.log(f"LAUNCH {j.id} card {c}{' slot ' + j.slot['id'] if j.slot else ''} pid {j.proc.pid} try {j.tries}")

    def adopt(self) -> None:
        """running run_e8.py drivers of a previous queue (by cmdline), watched by PID."""
        inv = {f"{RAW}/{s}": s for s in SETS}
        for p in Path("/proc").iterdir():
            if not p.name.isdigit():
                continue
            a = ES.cmdline(int(p.name)).split()
            if not any(x.endswith("run_e8.py") for x in a) or "--out" not in a or not a[0].endswith("python"):
                continue
            s = inv.get(a[a.index("--out") + 1])
            k = int(a[a.index("--shard") + 1])
            tts = int(a[a.index("--tts-port") + 1])
            sl = next((x for x in self.slots if x["tts"] == tts), None)
            j = next((x for x in self.pending if x.set == s and x.shard == k), None)
            if s is None or sl is None or j is None or not ES.alive(int(p.name)):
                continue
            j.pid, j.card, j.slot, j.started, j.tries = int(p.name), sl["card"], sl, time.time(), 1
            sl["job"] = j
            self.pending.remove(j)
            self.running.append(j)
            self.log(f"ADOPT {j.id} pid {j.pid} slot {sl['id']}")
        # v2: running ASR jobs of our runs (python processes of evaluation.asr_transcribe)
        outs = {f"{SPD}/{name(s)}/talker/asr_speech.p{k}.jsonl": (s, k) for s in SETS for k in range(ASR_SHARDS)}
        for p in Path("/proc").iterdir():
            if not p.name.isdigit():
                continue
            a = ES.cmdline(int(p.name)).split()
            if "evaluation.asr_transcribe" not in a or "--output" not in a or not a[0].endswith("python"):
                continue
            key = outs.get(a[a.index("--output") + 1])
            if key is None or not ES.alive(int(p.name)) or key in self.adopted_asr:
                continue
            s, k = key
            try:
                env = (p / "environ").read_bytes().split(b"\0")
                card = next(int(e.split(b"=", 1)[1]) for e in env if e.startswith(b"CUDA_VISIBLE_DEVICES="))
            except (OSError, StopIteration, ValueError):
                card = -1
            T = f"{SPD}/{name(s)}/talker"
            cmd = (ENV_E + f"{PYE} -m evaluation.asr_transcribe --input {T}/results_speechonly.jsonl --output "
                   f"{T}/asr_speech.p{k}.jsonl --device cuda:0 --batch-size 8 --resume --num-shards {ASR_SHARDS} "
                   f"--shard-index {k} && touch {MK}/{s}.asr{k}.ok")
            j = Job(f"{s}__asr{k}", "asr", cmd, LOGD / f"{s}_asr{k}.log", s)
            j.pid, j.card, j.started, j.tries = int(p.name), card, time.time(), 1
            self.adopted_asr[key] = j
            self.running.append(j)
            self.log(f"ADOPT {j.id} pid {j.pid} card {card}")

    def job_exit(self, j: Job):
        if j.proc is not None:
            return j.proc.poll()
        if ES.alive(j.pid):
            return None
        if j.kind == "asr":   # v2: adopted ASR job: the old wrapper touches the marker on success
            k = int(j.id.split("asr")[-1])
            for _ in range(15):
                if self.mark(j.set, f"asr{k}").exists():
                    return 0
                time.sleep(1)
            return 1
        return -1

    def stop_job(self, j: Job) -> None:
        if j.proc is not None:
            try:
                os.killpg(j.proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass

    def reap(self) -> None:
        for j in list(self.running):
            rc = self.job_exit(j)
            if rc is None:
                continue
            self.running.remove(j)
            if j.slot is not None:
                j.slot["job"] = None
            mins = (time.time() - j.started) / 60
            if j.kind == "dec":
                good = self.shard_done(j.set, j.shard)
                if good:
                    self.log(f"DONE {j.id} rc {rc} ({mins:.1f} min)")
                    continue
                if rc == 3:
                    self.log(f"SERVER_DOWN {j.id} on slot {j.slot['id']}: requeued; health refresh")
                    self.last_health = 0
                    j.tries -= 1   # not the job's fault
                    self.pending.insert(0, j)
                    continue
            elif rc == 0:
                self.log(f"DONE {j.id} rc 0 ({mins:.1f} min)")
                continue
            if j.tries < 4:
                self.log(f"RETRY {j.id} rc {rc} ({mins:.1f} min)")
                self.pending.insert(0, j)
            else:
                self.log(f"GIVEUP {j.id} rc {rc}")
                self.fail(j.set, f"{j.id} failed {j.tries} times")

    def schedule_decode(self) -> None:
        if self.a.dry_run:
            return
        for sl in self.slots:
            if sl["job"] is not None or not self.slot_ready(sl):
                continue
            j = next((x for x in self.pending if x.kind == "dec"), None)
            if j is None:
                return
            if time.time() - self.last_launch < 2:
                return
            ok, used = ES.fits(sl["card"], EST["dec"], CAP_GIB, MARGIN_GIB)
            if not ok:
                continue
            self.pending.remove(j)
            j.slot = sl
            sl["job"] = j
            self.launch(j, sl["card"], self.dec_cmd(j, sl))

    def schedule_asr(self) -> None:
        if self.a.dry_run:
            return
        j = next((x for x in self.pending if x.kind == "asr"), None)
        if j is None or time.time() - self.last_launch < STAGGER_ANY:
            return
        used = ES.gpu_used()
        mine = {c: sum(1 for x in self.running if x.kind == "asr" and x.card == c) for c in CARDS}
        for c in sorted(CARDS, key=lambda c: used[c]):
            if mine[c] >= (MAX_ASR_PER_CARD if self.servers_stopped else 1):
                continue
            if time.time() - self.asr_launch.get(c, 0) < ASR_HOLD:
                continue
            ph = ES.placeholder_pids(c)
            if ph and any(r == "worker" for _, r in ph) and self.servers_stopped and \
                    not any(x.card == c for x in self.running):
                self.log(f"card {c}: placeholder holds the card; stopping it for {j.id}")
                if not ES.stop_placeholder(c, log=self.log):
                    continue
                for _ in range(30):
                    if ES.gpu_used()[c] < 4 * GIB:
                        break
                    time.sleep(2)
            ok, u = ES.fits(c, EST["asr"], CAP_GIB, MARGIN_GIB)
            if ok:
                self.pending.remove(j)
                self.launch(j, c, j.cmd)
                self.asr_launch[c] = time.time()
                return

    # ------------------------------------------------------------------ chain
    def sh(self, cmd: str, log: Path) -> int:
        if self.a.dry_run:
            self.log(f"DRY {cmd[-400:]}")
            return 0
        r = subprocess.run(["bash", "-c", f"cd {REPO} && {cmd} >> {log} 2>&1"], stdin=subprocess.DEVNULL)
        return r.returncode

    def step(self, s: str, key: str, cmd: str, log: Path, must: bool = True) -> bool:
        if self.mark(s, key).exists():
            return True
        rc = self.sh(cmd, log)
        if rc:
            if must:
                self.fail(s, f"step {key} rc {rc}: {cmd[-200:]}")
                return False
            self.log(f"{s}: step {key} failed rc {rc} (see {log}); continuing")
            return True
        if not self.a.dry_run:
            self.mark(s, key).write_text(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()) + "\n")
        self.log(f"{s}: step {key} ok")
        return True

    def lat_cmd(self, s: str, n: str, talker: str | None) -> tuple[str, str]:
        S = SETS[s]
        O = f"{TE}/{n}"
        if talker is None:
            f, A = f"{O}/textlat_chunk_{n}.json", "--text-only"
        else:
            f, A = f"{O}/lat_chunk_{n}.json", f"--talker {talker}"
        if S["set"] == "en2zh":
            cmd = (ENV_E + f"mkdir -p {O}; CUDA_VISIBLE_DEVICES= {PY} -m experiments.mist_w3po_v1.evaluation.en2zh_lat "
                   f"--ref-win {S['ref']} --work {f}.work --run {DEC}/{n} {A} --win {S['cwin']} --out {f}")
        else:
            cmd = (ENV_E + f"mkdir -p {O}; CUDA_VISIBLE_DEVICES= {PY} -m experiments.mist_w3po_v1.evaluation.on_refset "
                   f"--ref-win {S['ref']} --dr {S['set']} experiments.mist_w3po_v1.evaluation.speech_latency_lit "
                   f"--run {DEC}/{n} {A} --win {S['cwin']} --out {f}")
        return cmd, f"{REPO}/{f}.log"

    def post1(self, s: str) -> None:
        S, n = SETS[s], name(s)
        L = LOGD / f"{s}_post.log"
        T = f"{SPD}/{n}/talker"
        steps = [
            ("build", ENV_E + f"{PY} {CODE}/build_e8.py --raw {RAW}/{s} --windows {S['win']} --name {n} "
                              f"--tgt {S['tgt']} --src {S['src']}"),
            ("score_text", ENV_E + f"{PY} -m experiments.mist_w3po_v1.evaluation.score_run --run {DEC}/{n} "
                                   f"--windows {S['win']} --ref-windows {S['ref']} --set {S['set']}"),
            ("score_text_1spk", ENV_E + f"{PY} -m experiments.mist_w3po_v1.evaluation.score_run --run {DEC}/{n}_1spk "
                                        f"--windows {S['win']} --ref-windows {S['ref']} --set {S['set']}"),
            ("speechonly", ENV_E + f"{PYE} scripts/gpu_queue/p80_speechonly.py {T}"),
        ]
        for key, c in steps:
            if not self.step(s, key, c, L):
                return
        cmd, lg = self.lat_cmd(s, n, None)
        self.step(s, "textlat", cmd, Path(lg), must=False)
        with self.lock:
            for k in range(ASR_SHARDS):
                if (REPO / T / f"asr_speech.p{k}.done").exists() or self.mark(s, f"asr{k}").exists():
                    continue
                if (s, k) in self.adopted_asr:   # v2: running since the first queue
                    continue
                cmd = (ENV_E + f"{PYE} -m evaluation.asr_transcribe --input {T}/results_speechonly.jsonl --output "
                       f"{T}/asr_speech.p{k}.jsonl --device cuda:0 --batch-size 8 --resume --num-shards {ASR_SHARDS} "
                       f"--shard-index {k} && touch {MK}/{s}.asr{k}.ok")
                self.pending.append(Job(f"{s}__asr{k}", "asr", cmd, LOGD / f"{s}_asr{k}.log", s))
            self.stage[s] = "asr"
        self.log(f"{s}: build + text scoring done; ASR queued")

    def post2(self, s: str) -> None:
        S, n = SETS[s], name(s)
        L = LOGD / f"{s}_post.log"
        T = f"{SPD}/{n}/talker"
        steps = [
            ("asr_merge", ENV_E + f"{PY} -m experiments.mist_w3po_v1.evaluation.asr_merge {T} {ASR_SHARDS}"),
            ("asr_hyp", ENV_E + f"{PY} -m experiments.mist_w3po_v1.evaluation.asr_hyp {DEC}/{n}/hyp.jsonl "
                                f"{T}/asr_speech_merged.jsonl {DEC}/{n}/hyp_asr.jsonl"),
            ("score_asr", ENV_E + f"{PY} -m experiments.mist_w3po_v1.evaluation.score_run --run {DEC}/{n} "
                                  f"--windows {S['win']} --ref-windows {S['ref']} --set {S['set']} "
                                  f"--asr {DEC}/{n}/hyp_asr.jsonl"),
        ]
        for key, c in steps:
            if not self.step(s, key, c, L):
                return
        cmd, lg = self.lat_cmd(s, n, T)
        self.step(s, "speechlat", cmd, Path(lg), must=False)
        self.step(s, "table1lat", f"bash experiments/mist_w3po_v1/scripts/table1_lat.sh {s} {n}",
                  LOGD / f"{s}_table1lat.log", must=False)
        with self.lock:
            self.stage[s] = "done"
        if not self.a.dry_run:
            (LOGD / f"E8_{s}_DONE").write_text(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()) + "\n")
        self.log(f"E8_SET_DONE {s}")

    def fail(self, s: str, why: str) -> None:
        with self.lock:
            self.stage[s] = "failed"
            self.failed.append(s)
        self.log(f"E8_SET_FAILED {s}: {why}")

    def start_thread(self, s: str, fn, stage: str) -> None:
        self.stage[s] = stage
        t = threading.Thread(target=fn, args=(s,), daemon=False)
        self.threads[s] = t
        t.start()

    # ------------------------------------------------------------------ main loop
    def run(self) -> None:
        for s in SETS:
            n = name(s)
            if not self.mark(s, "build").exists():
                for p in (f"{DEC}/{n}", f"{DEC}/{n}_1spk", f"{SPD}/{n}", f"{TE}/{n}"):
                    if (REPO / p).exists():
                        raise SystemExit(f"拒绝覆盖已有输出: {p}")
            self.pending += self.decode_jobs(s)
        self.adopt()
        self.log(f"E8 queue v2 start: {len(self.pending)} decode jobs pending, {len(self.running)} adopted; "
                 f"{len(self.slots)} slots; dry={self.a.dry_run}")
        if self.a.dry_run:
            for kind, c, port in self.plan:
                ES.start(kind, c, port, dry=True, log=self.log)
            for c in CARDS:
                ES.stop_placeholder(c, dry=True, log=self.log)
            for j in self.pending[:3]:
                self.log(f"DRY {j.id}: {self.dec_cmd(j, self.slots[0])}")
            for s in SETS:
                self.post1(s)
                self.post2(s)
            self.log(f"DRY gpu used {ES.gpu_used()}")
            return
        while True:
            with self.lock:
                stages = dict(self.stage)
            if all(v in ("done", "failed") for v in stages.values()):
                break
            self.reap()
            self.refresh_health()
            for s in SETS:
                busy = [j for j in self.running + self.pending if j.set == s]
                if stages[s] == "decode" and not busy:
                    self.start_thread(s, self.post1, "post1")
                elif stages[s] == "asr" and not busy:
                    self.start_thread(s, self.post2, "post2")
            with self.lock:
                self.pending = [j for j in self.pending if self.stage[j.set] != "failed"]
                for j in [j for j in self.running if self.stage[j.set] == "failed"]:
                    self.stop_job(j)
            left_dec = [j for j in self.running + self.pending if j.kind == "dec"]
            if not left_dec and not self.servers_stopped:
                self.stop_all_servers()
            try:
                self.manage_servers()
                self.schedule_decode()
                self.schedule_asr()
            except Exception as e:  # noqa: BLE001  (nvidia-smi hiccup etc.)
                self.log(f"loop error: {type(e).__name__}: {e}")
            time.sleep(10)
        if not self.servers_stopped:
            self.stop_all_servers()
        self.log(f"E8_QUEUE_DONE stages {self.stage} failed {self.failed}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    os.chdir(REPO)
    Q(a).run()


if __name__ == "__main__":
    main()

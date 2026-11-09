"""Final-read causal P test on dev (LOG 2026-11-09 19:35 UTC; sched_precheck_v8.zh-CN.md §8): one detached orchestrator
for every GPU / CPU step of the test. Inference / scoring only.

Runs (all NEW dirs; a fresh start refuses existing ones):
  CAND  decode/dev_v8_ctx0_guard_finalpost_s8: evaluation.online_decode_finalpost --final-read-post close_detect + the
        exact reference config (logs/mist_w3po_v1/queue_v2/jobs/032_dev_v8_ctx0_guard_s8_00.json: v8 merged Thinker,
        windows_dev_online, --policy wait --lag-ms 2000 --context 0 --batch 16 --max-gib 120, 8 shards by meeting,
        --loop-guard-k 6 --loop-guard-maxp 12 --budget-tps 25 --budget-slack 50), then score_run (score_text.json)
  OFF   decode/dev_v8_ctx0_guard_fpoff_s8: the same launcher with --final-read-post off (noise floor vs the reference)
  speech, the make_eval_jobs --speech --asr-per-talk 2 chain (prep -> 8 talker shards -> per-shard speech-only ASR in
        2 sub-shards -> asr_merge -> asr_hyp + score_run --asr); seed 2 = talker --seed 20261109 in <name>_tts2 with
        hyp.jsonl / score_text.json copied (cp -n), as mk_tts_seed2_jobs: CAND, CAND_tts2, OFF, OFF_tts2 and the
        reference's seed 2 in dev_v8_ctx0_guard_s8_tts2b (the queued dev_v8_ctx0_guard_s8_tts2 never ran its talkers;
        its files are left untouched)
GPU launches (coordinator rule 2026-11-09): under flock(LOCK): read nvidia-smi, check every target card stays
<= 135 GiB with the job's expected footprint (and planned use <= total - PLAN_MARGIN), launch (setsid -f, own session),
wait until the job's memory shows in nvidia-smi (>= VISIBLE GiB) or it exits (300 s cap), then release. vLLM talkers:
at most one start per GPU pair per 60 s. Headroom < 8 GiB on a card -> our latest job there is stopped (its session's
PIDs) and retried later, logged as events/alert_<n>. Placement: priority order (CAND decode > CAND speech > reference
seed 2 speech > OFF decode > OFF speech), backfill allowed except a decode never jumps a waiting higher-priority job.
State: <run>/state/<id>.a<n>.{pid,card,rc}, <run>/logs/<id>.a<n>.log, <run>/orch.log, <run>/status.json,
<run>/events/{alert_<n>,text_done,DONE}. Stop launching: touch <run>/STOP (running jobs continue; DONE when drained).

    setsid nohup python finalpost_dev_orch.py --run logs/mist_w3po_v1/finalpost_dev > /dev/null 2>&1 < /dev/null &
    python finalpost_dev_orch.py --run <dir> --dry          # print the jobs, launch nothing
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

REPO = Path("/opt/dlami/nvme/jasonleeeli/projects/UniSS")
J = "/opt/dlami/nvme/jasonleeeli"
LOCK = f"{J}/work/mist_w3po_v1/gpu_launch.lock"
ENV = (f"export TMPDIR={J}/work/mist_w3po_v1/tmp HF_HOME={J}/cache/huggingface TORCH_HOME={J}/cache/torch "
       f"XDG_CACHE_HOME={J}/cache TRITON_CACHE_DIR={J}/cache/triton VLLM_CACHE_ROOT={J}/cache/vllm_omni_030 "
       f"PYTHONPATH=.:third_party/Megatron-LM PATH={J}/venvs/vllm_omni_030/bin:$PATH OMP_NUM_THREADS=8 "
       f"TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; ")   # make_eval_jobs.ENV
PY = f"{J}/conda_envs/qwen3omni/bin/python"
PYE = f"{J}/conda_envs/uniss-eval/bin/python"
PYV = f"{J}/venvs/vllm_omni_030/bin/python"
T1C = "eval_outputs/uniss_qwen3_omni_v1/si_t1c_20261021T181440Z/autoext/export_1500/uniss"
M = "experiments.uniss_qwen3_omni_v1.evaluation"
WIN = "eval_outputs/mist_w3po_v1/data/windows_zh2en_v1/windows_dev_online.jsonl"
REF_WIN = "data/ami_multispk_si_v1/e2e_gstar_zh2en_v1/windows_validation_ref.jsonl"
THINKER = "eval_outputs/uniss_qwen3_omni_v1/thinker_gstar_s2dl_zh2en_3k_20261030/thinker_merged"
DEC = "eval_outputs/mist_w3po_v1/decode"
SPD = "eval_outputs/mist_w3po_v1/speech"
REF = "dev_v8_ctx0_guard_s8"
CAND = "dev_v8_ctx0_guard_finalpost_s8"
OFF = "dev_v8_ctx0_guard_fpoff_s8"
REF2 = REF + "_tts2b"
SEED2 = 20261109
GUARD = "--loop-guard-k 6 --loop-guard-maxp 12 --budget-tps 25 --budget-slack 50"
SHARDS_BIG_FIRST = [6, 5, 4, 1, 3, 2, 0, 7]   # reference windows per shard: 588 579 443 374 352 341 285 88
NEED = {"dec": 68.0, "talk": 26.0, "asr": 14.0}      # GiB per job (decode ~66, make_eval_jobs: talker 26, ASR 14)
VISIBLE = {"dec": 45.0, "talk": 10.0, "asr": 2.5}    # GiB of the job in nvidia-smi = "its memory is visible"
CAP_TOTAL = 135.0
HEADROOM = 8.0
PLAN_MARGIN = 10.0
OTHER_WINDOW_S = 180
VISIBLE_WAIT_S = 300
TALK_PAIR_GAP_S = 60
LAUNCH_GAP_S = {"dec": 15, "talk": 5, "asr": 10}
MAX_CPU = 8
TICK_S = 10


def now() -> float:
    return time.time()


class Job:
    def __init__(self, jid, kind, prio, deps, cmd, retries=0):
        self.id, self.kind, self.prio, self.deps, self.cmd, self.retries = jid, kind, prio, list(deps), cmd, retries
        self.state, self.attempt, self.card, self.sid, self.t_start, self.t_end = "wait", 0, None, None, None, None
        self.not_before, self.rc, self.order = 0.0, None, 0

    @property
    def gpu(self):
        return self.kind != "cpu"


def build_jobs() -> list[Job]:
    jobs: list[Job] = []

    def add(*a, **k):
        jobs.append(Job(*a, **k))
        jobs[-1].order = len(jobs)

    def decode(name, mode, prio):
        D = f"{DEC}/{name}"
        for k in SHARDS_BIG_FIRST:
            def cmd(att, k=k, D=D):
                return (ENV + f"mkdir -p {D} && PREV=$(ls {D}/hyp_s{k}_a*.jsonl 2>/dev/null); {PY} -m "
                        f"experiments.mist_w3po_v1.evaluation.online_decode_finalpost --final-read-post {mode} "
                        f"--thinker {THINKER} --windows {WIN} --out {D}/hyp_s{k}_a{att}.jsonl "
                        f"${{PREV:+--resume-from $PREV}} --policy wait --lag-ms 2000 --context 0 --dir zh2en "
                        f"--batch 16 --max-gib 120 --shard {k} --num-shards 8 --shard-by meeting {GUARD} "
                        f"> {D}/infer_s{k}_a{att}.log 2>&1 && grep -q W3PO_ONLINE_DONE {D}/infer_s{k}_a{att}.log")
            add(f"{name}__dec{k}", "dec", prio, [], cmd, retries=2)
        add(f"{name}__score", "cpu", prio, [f"{name}__dec{k}" for k in range(8)],
            lambda att, D=D: ENV + f"[ -e {D}/score_text.json ] || {PY} -m experiments.mist_w3po_v1.evaluation.score_run "
                                   f"--run {D} --windows {WIN} --ref-windows {REF_WIN} --set zh2en > {D}/score.log 2>&1")

    def speech(name, base, seed, prio, base_dep):
        D, SP = f"{DEC}/{name}", f"{SPD}/{name}"
        pre = (f"mkdir -p {D} && cp -n {DEC}/{base}/hyp.jsonl {DEC}/{base}/score_text.json {D}/ && "
               if name != base else "")
        add(f"{name}__sprep", "cpu", prio, base_dep,
            lambda att: ENV + pre + f"mkdir -p {SP}/talker && GSTAR_DATA=data/ami_multispk_si_v1 {PYE} -m "
                        f"{M}.gstar_talker_v6 prep --hyp {D}/hyp.jsonl --windows {WIN} --causal --blocks --preset lowlat "
                        f"--session zh_mix_v8 --diar nemotron_lowlat_zh8 --sep ' ' --out {SP}/talker_in_all > "
                        f"{SP}/prep.log 2>&1 && python3 reports/gstar_si_v7/drop_long_streams.py {SP}/talker_in_all "
                        f"{SP}/talker_in 400 >> {SP}/prep.log 2>&1")
        sd = f" --seed {seed}" if seed else ""
        parts = []
        for k in range(8):
            add(f"{name}__talk{k}", "talk", prio, [f"{name}__sprep"],
                lambda att, k=k: ENV + f"VLLM_USE_FLASHINFER_SAMPLER=0 OMP_NUM_THREADS=16 {PYV} -m {M}.si_infer "
                                       f"--model {T1C} --direction zh2en --window 1 --output {SP}/talker --text-policy file "
                                       f"--blocks-file {SP}/talker_in/blocks.jsonl --items-file {SP}/talker_in/items.jsonl "
                                       f"--speed-policy fixed --history-ban 8 --max-model-len 32768 --gpu-mem 0.1 "
                                       f"--shard {k} --num-shards 8{sd} > {SP}/talk{k}.log 2>&1", retries=5)
            add(f"{name}__asrprep{k}", "cpu", prio, [f"{name}__talk{k}"],
                lambda att, k=k: ENV + f"[ -s {SP}/talker/part_{k}.jsonl ] && {PYE} experiments/mist_w3po_v1/evaluation/"
                                       f"speechonly_part.py {SP}/talker {k} > {SP}/asrprep{k}.log 2>&1")
            for s in range(2):
                parts.append(f"{name}__asrp{k}_{s}")
                add(parts[-1], "asr", prio, [f"{name}__asrprep{k}"],
                    lambda att, k=k, s=s: ENV + f"{PYE} -m evaluation.asr_transcribe --input "
                                                f"{SP}/talker/results_speechonly.t{k}.jsonl --output "
                                                f"{SP}/talker/asr_speech.t{k}s{s}.jsonl --device cuda:0 --batch-size 8 "
                                                f"--resume --num-shards 2 --shard-index {s} >> {SP}/asr_t{k}s{s}.log 2>&1",
                    retries=3)
        add(f"{name}__asrprep", "cpu", prio, [f"{name}__talk{k}" for k in range(8)],
            lambda att: ENV + f"for k in 0 1 2 3 4 5 6 7; do [ -s {SP}/talker/part_$k.jsonl ] || MISSING=1; done; "
                              f"[ -z \"$MISSING\" ] && cat {SP}/talker/part_*.jsonl > {SP}/talker/results.jsonl && "
                              f"{PYE} scripts/gpu_queue/p80_speechonly.py {SP}/talker > {SP}/asrprep.log 2>&1")
        add(f"{name}__asr", "cpu", prio, parts + [f"{name}__asrprep"],
            lambda att: ENV + f"{PY} -m experiments.mist_w3po_v1.evaluation.asr_merge {SP}/talker 16 "
                              f"'asr_speech.t*s*.jsonl' > {SP}/asr_merge.log 2>&1")
        add(f"{name}__sscore", "cpu", prio, [f"{name}__asr"],
            lambda att: ENV + f"{PY} -m experiments.mist_w3po_v1.evaluation.asr_hyp {D}/hyp.jsonl "
                              f"{SP}/talker/asr_speech_merged.jsonl {D}/hyp_asr.jsonl && {PY} -m "
                              f"experiments.mist_w3po_v1.evaluation.score_run --run {D} --windows {WIN} --ref-windows "
                              f"{REF_WIN} --set zh2en --asr {D}/hyp_asr.jsonl > {D}/score_asr.log 2>&1")

    decode(CAND, "close_detect", 1)
    speech(CAND, CAND, None, 2, [f"{CAND}__score"])
    speech(CAND + "_tts2", CAND, SEED2, 2, [f"{CAND}__score"])
    speech(REF2, REF, SEED2, 3, [])
    decode(OFF, "off", 4)
    speech(OFF, OFF, None, 5, [f"{OFF}__score"])
    speech(OFF + "_tts2", OFF, SEED2, 5, [f"{OFF}__score"])
    return jobs


NEW_DIRS = [f"{DEC}/{n}" for n in (CAND, CAND + "_tts2", OFF, OFF + "_tts2", REF2)] + \
           [f"{SPD}/{n}" for n in (CAND, CAND + "_tts2", OFF, OFF + "_tts2", REF2)]


class Orch:
    def __init__(self, run: Path, jobs: list[Job]):
        self.run, self.jobs = run, jobs
        self.by_id = {j.id: j for j in jobs}
        self.other_hist: list[tuple[float, dict]] = []
        self.pair_last = {p: 0.0 for p in range(4)}
        self.last_gpu_launch = 0.0
        self.n_alert = 0
        self.t0 = now()
        for d in ("state", "logs", "events"):
            (run / d).mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------- bookkeeping
    def log(self, msg: str):
        with open(self.run / "orch.log", "a") as fh:
            fh.write(f"[{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}] {msg}\n")

    def alert(self, msg: str):
        self.n_alert += 1
        with open(self.run / "events" / f"alert_{self.n_alert}", "x") as fh:
            fh.write(msg + "\n")
        self.log("ALERT " + msg)

    def f(self, j: Job, att: int, ext: str) -> Path:
        return self.run / "state" / f"{j.id}.a{att}.{ext}"

    def sids(self) -> dict[int, int]:
        out = subprocess.run(["ps", "-eo", "pid=,sid="], capture_output=True, text=True).stdout.split("\n")
        res = {}
        for l in out:
            p = l.split()
            if len(p) == 2:
                res[int(p[0])] = int(p[1])
        return res

    def snapshot(self) -> dict:
        q = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,memory.used,memory.total",
                            "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=60).stdout
        cards, uuid2c = {}, {}
        for l in q.strip().split("\n"):
            i, u, used, tot = [x.strip() for x in l.split(",")]
            cards[int(i)] = {"used": float(used) / 1024, "total": float(tot) / 1024}
            uuid2c[u] = int(i)
        a = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,gpu_uuid,used_memory",
                            "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=60).stdout
        sid_of = self.sids()
        run_sids = {j.sid: j for j in self.jobs if j.state == "run" and j.sid}
        per_job = {}
        ours = {c: 0.0 for c in cards}
        for l in a.strip().split("\n"):
            p = [x.strip() for x in l.split(",")]
            if len(p) != 3 or not p[0].isdigit():
                continue
            pid, c, mem = int(p[0]), uuid2c.get(p[1]), (float(p[2]) / 1024 if p[2].replace(".", "").isdigit() else 0.0)
            j = run_sids.get(sid_of.get(pid))
            if j is not None and c is not None:
                per_job[j.id] = per_job.get(j.id, 0.0) + mem
                ours[c] += mem
        other = {c: cards[c]["used"] - ours[c] for c in cards}
        t = now()
        self.other_hist = [(tt, o) for tt, o in self.other_hist if t - tt < OTHER_WINDOW_S] + [(t, other)]
        other_plan = {c: max(o[c] for _, o in self.other_hist) for c in cards}
        return {"cards": cards, "per_job": per_job, "ours": ours, "other": other, "other_plan": other_plan,
                "sid_of": sid_of}

    def free_plan(self, snap: dict) -> dict:
        """GiB a new job may still take per card (planned use <= min(135, total - margin))."""
        res = {}
        for c, v in snap["cards"].items():
            mine = sum(max(NEED[j.kind], snap["per_job"].get(j.id, 0.0)) for j in self.jobs
                       if j.state == "run" and j.gpu and j.card == c)
            cap = min(CAP_TOTAL, v["total"] - PLAN_MARGIN)
            res[c] = cap - snap["other_plan"][c] - mine
            # real-time check as well: the card now + the not yet visible part of our jobs
            pend = sum(max(0.0, NEED[j.kind] - snap["per_job"].get(j.id, 0.0)) for j in self.jobs
                       if j.state == "run" and j.gpu and j.card == c)
            res[c] = min(res[c], CAP_TOTAL - v["used"] - pend, v["total"] - HEADROOM - 2.0 - v["used"] - pend)
        return res

    # ---------------------------------------------------------------- processes
    def alive(self, sid: int, sid_of: dict | None = None) -> list[int]:
        sid_of = sid_of if sid_of is not None else self.sids()
        return [p for p, s in sid_of.items() if s == sid]

    def stop(self, j: Job, why: str):
        """SIGTERM every process of the job's session (by PID), SIGKILL what is left after 20 s."""
        pids = self.alive(j.sid)
        self.log(f"stop {j.id} a{j.attempt} card {j.card} ({why}): pids {pids}")
        for p in pids:
            try:
                os.kill(p, signal.SIGTERM)
            except ProcessLookupError:
                pass
        for _ in range(20):
            time.sleep(1)
            if not self.alive(j.sid):
                break
        for p in self.alive(j.sid):
            try:
                os.kill(p, signal.SIGKILL)
            except ProcessLookupError:
                pass

    def start(self, j: Job, card: int | None):
        j.attempt += 1
        pidf, rcf, logf = self.f(j, j.attempt, "pid"), self.f(j, j.attempt, "rc"), self.run / "logs" / f"{j.id}.a{j.attempt}.log"
        env = os.environ.copy()
        if card is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(card)
            self.f(j, j.attempt, "card").write_text(str(card))
        else:
            env["CUDA_VISIBLE_DEVICES"] = ""
        body = (f"echo $$ > {pidf}.tmp && command mv -f {pidf}.tmp {pidf}\n{j.cmd(j.attempt)}\n"
                f"echo $? > {rcf}.tmp && command mv -f {rcf}.tmp {rcf}")
        with open(logf, "ab") as fh:
            subprocess.run(["setsid", "-f", "bash", "-c", body], cwd=REPO, env=env, stdin=subprocess.DEVNULL,
                           stdout=fh, stderr=subprocess.STDOUT)
        for _ in range(100):
            if pidf.exists():
                break
            time.sleep(0.1)
        j.sid = int(pidf.read_text().strip()) if pidf.exists() else None
        j.state, j.card, j.t_start, j.rc = "run", card, now(), None
        self.log(f"start {j.id} a{j.attempt} card {card} sid {j.sid}")

    def finish(self, j: Job, rc: int):
        j.rc, j.t_end = rc, now()
        dur = j.t_end - (j.t_start or j.t_end)
        if rc == 0:
            j.state = "ok"
            self.log(f"ok {j.id} a{j.attempt} {dur:.0f}s")
        elif j.attempt <= j.retries:
            j.state, j.not_before = "wait", now() + 60
            self.log(f"failed {j.id} a{j.attempt} rc {rc} {dur:.0f}s -> retry")
        else:
            j.state = "fail"
            self.alert(f"FAILED {j.id} a{j.attempt} rc {rc} after {dur:.0f}s (log {self.run}/logs/{j.id}.a{j.attempt}.log)")

    def poll(self, sid_of: dict):
        for j in self.jobs:
            if j.state != "run":
                continue
            rcf = self.f(j, j.attempt, "rc")
            if rcf.exists():
                try:
                    self.finish(j, int(rcf.read_text().strip() or 1))
                except ValueError:
                    pass
            elif j.sid is None or not self.alive(j.sid, sid_of):
                time.sleep(2)
                if rcf.exists():
                    self.finish(j, int(rcf.read_text().strip() or 1))
                else:
                    self.finish(j, 137)

    def headroom(self, snap: dict) -> bool:
        hit = False
        for c, v in snap["cards"].items():
            if v["total"] - v["used"] >= HEADROOM:
                continue
            mine = [j for j in self.jobs if j.state == "run" and j.gpu and j.card == c]
            if not mine:
                self.log(f"card {c}: headroom {v['total'] - v['used']:.1f} GiB, none of ours there")
                continue
            j = max(mine, key=lambda x: x.t_start)
            self.alert(f"HEADROOM card {c}: used {v['used']:.1f} of {v['total']:.1f} GiB (ours {snap['ours'][c]:.1f}, "
                       f"other {snap['other'][c]:.1f}); stopping our latest job {j.id} a{j.attempt}, retried later")
            self.stop(j, "headroom")
            j.state, j.not_before, j.t_end = "wait", now() + 300, now()
            j.retries += 1   # a stop for headroom does not use up a retry
            hit = True
        return hit

    # ---------------------------------------------------------------- scheduling
    def deps_state(self, j: Job) -> str:
        st = [self.by_id[d].state for d in j.deps]
        if any(s == "fail" or s == "skip" for s in st):
            return "fail"
        return "ok" if all(s == "ok" for s in st) else "wait"

    def plan(self, snap: dict, ready: list[Job]) -> list[tuple[Job, int]]:
        free = self.free_plan(snap)
        dec_on = {c: any(j.state == "run" and j.kind == "dec" and j.card == c for j in self.jobs) for c in free}
        reserved, plan, pair_used, blocked = set(), [], set(), False
        t = now()
        for j in sorted(ready, key=lambda x: (x.prio, x.order)):
            need = NEED[j.kind]
            cands = [c for c in free if c not in reserved and free[c] >= need and not (j.kind == "dec" and dec_on[c])]
            if j.kind == "dec" and blocked:
                continue
            if j.kind == "talk":
                ok = [c for c in cands if c // 2 not in pair_used and t - self.pair_last[c // 2] >= TALK_PAIR_GAP_S]
                if cands and not ok:
                    reserved.add(min(cands, key=lambda c: free[c] - need))
                    blocked = True
                    continue
                cands = ok
            if not cands:
                blocked = True
                continue
            c = max(cands, key=lambda c: free[c]) if j.kind == "dec" else min(cands, key=lambda c: free[c] - need)
            plan.append((j, c))
            free[c] -= need
            if j.kind == "dec":
                dec_on[c] = True
            if j.kind == "talk":
                pair_used.add(c // 2)
            if len(plan) >= 8:
                break
        return plan

    def launch_gpu(self, plan: list[tuple[Job, int]]):
        lk = open(LOCK, "a")
        t_lock = now()
        while True:
            try:
                fcntl.flock(lk, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if now() - t_lock > 900:
                    self.log("gpu_launch.lock busy for 15 min; skipping this round")
                    lk.close()
                    return
                time.sleep(2)
        try:
            self.log(f"lock held after {now() - t_lock:.0f}s; plan {[(j.id, c) for j, c in plan]}")
            launched = []
            for j, c in plan:
                snap = self.snapshot()
                fr = self.free_plan(snap)
                if fr[c] < NEED[j.kind]:
                    self.log(f"skip {j.id} on card {c}: free {fr[c]:.1f} < {NEED[j.kind]} at re-check")
                    continue
                if launched:
                    time.sleep(LAUNCH_GAP_S[j.kind])
                if j.kind == "talk":
                    self.pair_last[c // 2] = now()
                self.start(j, c)
                launched.append(j)
            # wait until each launched job's memory shows (or it exits); 300 s cap per job
            while launched:
                snap = self.snapshot()
                self.headroom(snap)
                sid_of = snap["sid_of"]
                rest = []
                for j in launched:
                    vis = snap["per_job"].get(j.id, 0.0)
                    gone = j.state != "run" or self.f(j, j.attempt, "rc").exists() or not self.alive(j.sid, sid_of)
                    if vis >= VISIBLE[j.kind] or gone:
                        self.log(f"visible {j.id}: {vis:.1f} GiB{' (exited)' if gone else ''} after {now() - j.t_start:.0f}s")
                    elif now() - j.t_start > VISIBLE_WAIT_S:
                        self.log(f"visible-wait timeout {j.id}: {vis:.1f} GiB after {now() - j.t_start:.0f}s")
                    else:
                        rest.append(j)
                launched = rest
                if launched:
                    time.sleep(5)
        finally:
            fcntl.flock(lk, fcntl.LOCK_UN)
            lk.close()
            self.last_gpu_launch = now()

    def status(self, snap: dict | None):
        cnt = {}
        for j in self.jobs:
            cnt[j.state] = cnt.get(j.state, 0) + 1
        st = {"t": time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), "elapsed_min": round((now() - self.t0) / 60, 1),
              "counts": cnt, "running": [[j.id, j.card, round((snap or {}).get("per_job", {}).get(j.id, 0.0), 1)]
                                         for j in self.jobs if j.state == "run"],
              "cards_used": {c: round(v["used"], 1) for c, v in (snap or {}).get("cards", {}).items()},
              "times": {j.id: [j.t_start, j.t_end, j.attempt, j.card] for j in self.jobs if j.t_start}}
        tmp = self.run / "status.json.tmp"
        tmp.write_text(json.dumps(st, indent=1))
        os.replace(tmp, self.run / "status.json")

    def adopt(self):
        """--resume after an orchestrator crash: rebuild job states from state/<id>.a<n>.{pid,card,rc}."""
        sid_of = self.sids()
        while (self.run / "events" / f"alert_{self.n_alert + 1}").exists():
            self.n_alert += 1
        for j in self.jobs:
            n = 0
            while self.f(j, n + 1, "pid").exists():
                n += 1
            if not n:
                continue
            j.attempt = n
            rcf, cf = self.f(j, n, "rc"), self.f(j, n, "card")
            j.card = int(cf.read_text()) if cf.exists() else None
            j.sid = int(self.f(j, n, "pid").read_text().strip())
            j.t_start = self.f(j, n, "pid").stat().st_mtime
            if rcf.exists():
                j.state = "run"
                self.finish(j, int(rcf.read_text().strip() or 1))
            elif self.alive(j.sid, sid_of):
                j.state = "run"
            else:
                j.state = "run"
                self.finish(j, 137)
            if j.kind == "talk" and j.card is not None:
                self.pair_last[j.card // 2] = max(self.pair_last[j.card // 2], j.t_start)
        self.log("resumed: " + json.dumps({s: sum(1 for j in self.jobs if j.state == s) for s in
                                           ("ok", "run", "wait", "fail")}))

    def loop(self):
        self.log(f"orchestrator pid {os.getpid()} start, {len(self.jobs)} jobs")
        text_flag = self.run / "events" / "text_done"
        while True:
            snap = self.snapshot()
            self.poll(snap["sid_of"])
            self.headroom(snap)
            for j in self.jobs:
                if j.state == "wait" and self.deps_state(j) == "fail":
                    j.state = "skip"
                    self.alert(f"SKIPPED {j.id}: a dependency failed")
            stop = (self.run / "STOP").exists()
            ready = [j for j in self.jobs if j.state == "wait" and self.deps_state(j) == "ok" and now() >= j.not_before]
            if not stop:
                n_cpu = sum(1 for j in self.jobs if j.state == "run" and not j.gpu)
                for j in [x for x in ready if not x.gpu][: max(0, MAX_CPU - n_cpu)]:
                    self.start(j, None)
                plan = self.plan(snap, [x for x in ready if x.gpu])
                if plan:
                    self.launch_gpu(plan)
            if not text_flag.exists() and all(self.by_id[f"{n}__score"].state == "ok" for n in (CAND, OFF)):
                text_flag.write_text("1\n")
            cand_flag = self.run / "events" / "cand_text_done"
            if not cand_flag.exists() and self.by_id[f"{CAND}__score"].state == "ok":
                cand_flag.write_text("1\n")
            snap2 = self.snapshot()
            self.status(snap2)
            active = [j for j in self.jobs if j.state == "run" or (j.state == "wait" and not stop)]
            if not active:
                bad = [j.id for j in self.jobs if j.state in ("fail", "skip")]
                (self.run / "events" / "DONE").write_text(json.dumps({"failed_or_skipped": bad,
                                                                       "elapsed_min": round((now() - self.t0) / 60, 1)}) + "\n")
                self.log(f"DONE; failed/skipped {bad}")
                return
            time.sleep(TICK_S)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True)
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--resume", action="store_true", help="continue a crashed orchestrator's run dir (adopt its jobs)")
    a = ap.parse_args()
    os.chdir(REPO)
    jobs = build_jobs()
    assert len({j.id for j in jobs}) == len(jobs)
    for j in jobs:
        for d in j.deps:
            assert d in {x.id for x in jobs}, (j.id, d)
    if a.dry:
        for j in jobs:
            print(j.prio, j.kind, j.id, "deps", len(j.deps), "|", j.cmd(1)[len(ENV):][:260])
        print(len(jobs), "jobs; new dirs:", [d for d in NEW_DIRS if Path(d).exists()] or "none exist")
        return
    run = Path(a.run)
    if a.resume:
        if not (run / "state").is_dir():
            raise SystemExit(f"--resume: no state in {run}")
        o = Orch(run, jobs)
        o.adopt()
    else:
        if run.exists():
            raise SystemExit(f"拒绝覆盖已有输出: {run}")
        ex = [d for d in NEW_DIRS if Path(d).exists()]
        if ex:
            raise SystemExit(f"拒绝覆盖已有输出: {ex}")
        run.mkdir(parents=True)
        o = Orch(run, jobs)
    Path(LOCK).parent.mkdir(parents=True, exist_ok=True)
    try:
        o.loop()
    except Exception as e:  # noqa: BLE001
        import traceback
        o.log("CRASH " + traceback.format_exc().replace("\n", " | "))
        o.alert(f"ORCHESTRATOR CRASH {e!r}; running jobs continue (their rc files land in state/)")
        raise


if __name__ == "__main__":
    main()

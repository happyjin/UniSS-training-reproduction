"""E7 queue (reports/mist_w3po_v1/e7_seamless_multispk_prereg.zh-CN.md): SeamlessStreaming on the three multi-speaker
test sets, each decoded once, then scored with our own tools. CPU process; launches and watches only its own jobs.

Per set (ami: AMI zh->en, ali: AliMeeting zh->en, en: AMI En->Zh):
  1. decode shards  run_e7.py on the set's online test windows (joint_lib WT_*), --resume on relaunch
  2. build          build_e7.py -> decode/jtest_<set>_seamless_default{,_1spk}/hyp_seamless.jsonl, speech/.../talker
  3. text           score_run (score_text.json; both runs), chunk-protocol text LAAL (--text-only) of the main run
  4. speech         p80_speechonly -> evaluation.asr_transcribe (Whisper-large-v3 / Paraformer-zh), ASR_SHARDS GPU
                    shards -> asr_merge -> asr_hyp -> score_run --asr (score_asr.json) -> chunk-protocol speech latency
  5. marker         logs/mist_w3po_v1/e7/E7_<set>_DONE
GPU rule (coordinator 2026-11-14): launch a job on a card only if its used memory + the job's estimate + 6 GiB margin
<= 135 GiB, no launch on a card within 60 s of my previous launch there, no card holding a training process, at most
MAX_PER_CARD of my jobs per card and MAX_GPU of my jobs in all. A card whose used memory stays above 136 GiB for two
polls: my NEWEST job there is stopped by PID (its cmdline checked) and requeued (decode: --resume; ASR: --resume).

    PYTHONPATH=. $PY experiments/s2st_baselines_v1/seamless_multispk/e7_queue.py [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

REPO = Path("/opt/dlami/nvme/jasonleeeli/projects/UniSS")
J = "/opt/dlami/nvme/jasonleeeli"
CODE = "experiments/s2st_baselines_v1/seamless_multispk"
RAW = "eval_outputs/mist_w3po_v1/e7_seamless/raw"
DEC = "eval_outputs/mist_w3po_v1/decode"
SPD = "eval_outputs/mist_w3po_v1/speech"
TE = "eval_outputs/mist_w3po_v1/crf/t_eval"
LOGD = REPO / "logs/mist_w3po_v1/e7"
PY = f"{J}/conda_envs/qwen3omni/bin/python"
PYE = f"{J}/conda_envs/uniss-eval/bin/python"
PYS = f"{J}/venvs/s2st_baselines/bin/python"
ENV_S = "source experiments/s2st_baselines_v1/seamless/env.sh; export PYTHONPATH=.; "
ENV_E = (f"export TMPDIR={J}/work/mist_w3po_v1/tmp HF_HOME={J}/cache/huggingface TORCH_HOME={J}/cache/torch "
         f"XDG_CACHE_HOME={J}/cache MODELSCOPE_CACHE={J}/cache/modelscope PYTHONPATH=.:third_party/Megatron-LM "
         f"OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false; ")
DD = "eval_outputs/mist_w3po_v1/data"
SETS = {
    "ami": {"win": f"{DD}/windows_test_c2g06_v1/windows_test_online_zh2en.jsonl",
            "cwin": f"{DD}/windows_test_c2g06_chunk_v1/zh2en/windows_dev_online.jsonl",
            "ref": "data/ami_multispk_si_v1/e2e_gstar_zh2en_v1/windows_test_ref.jsonl", "set": "zh2en",
            "tgt": "eng", "src": "cmn", "shards": 6},
    "ali": {"win": f"{DD}/windows_test_c2g06_v1/windows_test_online_ali.jsonl",
            "cwin": f"{DD}/windows_test_c2g06_chunk_v1/ali/windows_dev_online.jsonl",
            "ref": "data/diarist_alimeeting_refs_v1/windows_test_ref_human.jsonl", "set": "ali",
            "tgt": "eng", "src": "cmn", "shards": 10},
    "en": {"win": f"{DD}/windows_test_en2zh_c2g06_v1/windows_test_online_en2zh.jsonl",
           "cwin": f"{DD}/windows_test_en2zh_c2g06_chunk_v1/windows_dev_online.jsonl",
           "ref": "data/ami_multispk_si_v1/e2e_gstar_v1/windows_test_ref.jsonl", "set": "en2zh",
           "tgt": "cmn", "src": "eng", "shards": 8},
}
ASR_SHARDS = 4
GIB = 1024
EST = {"dec": 12 * GIB, "asr": 12 * GIB}
MARGIN = 6 * GIB
CAP = 135 * GIB
EVICT = 136 * GIB
MAX_PER_CARD = 3
MAX_GPU = 10
STAGGER = 60
TRAIN_PAT = ("train_crf_sft", "train_v2p", "torch.distributed", "megatron", "pretrain_", "train_talker")


def name(s: str) -> str:
    return f"jtest_{s}_seamless_default"


class Job:
    def __init__(self, jid: str, kind: str, cmd: str, log: Path, setk: str):
        self.id, self.kind, self.cmd, self.log, self.set = jid, kind, cmd, log, setk
        self.proc = None
        self.card = None
        self.started = 0.0
        self.tries = 0
        self.done = False


class Q:
    def __init__(self, a):
        self.a = a
        LOGD.mkdir(parents=True, exist_ok=True)
        self.logf = open(LOGD / "e7_queue.log", "a", encoding="utf-8")
        self.pending: list[Job] = []
        self.running: list[Job] = []
        self.last_launch: dict[int, float] = {}
        self.over: dict[int, int] = {}
        self.stage = {s: "decode" for s in SETS}
        self.failed: list[str] = []

    def log(self, msg: str) -> None:
        line = f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {msg}"
        print(line, flush=True)
        self.logf.write(line + "\n")
        self.logf.flush()

    # ------------------------------------------------------------------ GPU
    def cards(self) -> dict[int, dict]:
        q = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,memory.used", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=60).stdout
        out, u2c = {}, {}
        for l in q.strip().split("\n"):
            p = [x.strip() for x in l.split(",")]
            if len(p) == 3 and p[0].isdigit():
                out[int(p[0])] = {"used": int(p[2]), "train": False}
                u2c[p[1]] = int(p[0])
        a = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,gpu_uuid", "--format=csv,noheader"],
                           capture_output=True, text=True, timeout=60).stdout
        for l in a.strip().split("\n"):
            p = [x.strip() for x in l.split(",")]
            if len(p) != 2 or not p[0].isdigit() or p[1] not in u2c:
                continue
            try:
                cmd = Path(f"/proc/{p[0]}/cmdline").read_bytes().replace(b"\0", b" ").decode(errors="ignore")
            except OSError:
                continue
            if any(x in cmd for x in TRAIN_PAT):
                out[u2c[p[1]]]["train"] = True
        return out

    def pick(self, C: dict, kind: str) -> int | None:
        now = time.time()
        mine = {}
        for j in self.running:
            mine[j.card] = mine.get(j.card, 0) + 1
        best = None
        for c, v in sorted(C.items()):
            if v["train"] or mine.get(c, 0) >= MAX_PER_CARD or now - self.last_launch.get(c, 0) < STAGGER:
                continue
            if v["used"] + EST[kind] + MARGIN > CAP:
                continue
            if best is None or v["used"] < C[best]["used"]:
                best = c
        return best

    def launch(self, j: Job, c: int) -> None:
        j.tries += 1
        j.card = c
        full = f"cd {REPO} && export CUDA_VISIBLE_DEVICES={c}; {j.cmd} >> {j.log} 2>&1"
        j.proc = subprocess.Popen(["bash", "-c", full], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                  stderr=subprocess.DEVNULL)
        j.started = self.last_launch[c] = time.time()
        self.running.append(j)
        self.log(f"LAUNCH {j.id} card {c} pid {j.proc.pid} try {j.tries}")

    def evict(self, C: dict) -> None:
        for c, v in C.items():
            if v["used"] > EVICT:
                self.over[c] = self.over.get(c, 0) + 1
            else:
                self.over[c] = 0
            if self.over[c] < 2:
                continue
            mine = [j for j in self.running if j.card == c]
            if not mine:
                continue
            j = max(mine, key=lambda x: x.started)
            self.stop(j, f"card {c} at {v['used']} MiB > {EVICT} for two polls")
            self.over[c] = 0

    def stop(self, j: Job, why: str) -> None:
        """stop my job by PID: the wrapper pid itself (bash may exec the last command) and its children, each only
        after a cmdline check (run_e7.py / asr_transcribe), then reap the wrapper and requeue the job."""
        pids = [str(j.proc.pid)] + subprocess.run(["ps", "-o", "pid=", "--ppid", str(j.proc.pid)], capture_output=True,
                                                  text=True).stdout.split()
        for p in pids:
            try:
                cmd = Path(f"/proc/{p}/cmdline").read_bytes().replace(b"\0", b" ").decode(errors="ignore")
            except OSError:
                continue
            if ("run_e7.py" in cmd or "asr_transcribe" in cmd) and "e7_queue.py" not in cmd:
                os.kill(int(p), signal.SIGTERM)
                self.log(f"STOP {j.id} pid {p} ({why})")
        try:
            j.proc.wait(timeout=120)
        except subprocess.TimeoutExpired:
            self.log(f"STOP {j.id}: wrapper {j.proc.pid} still running after 120 s; left alone, job not requeued")
            return
        self.running.remove(j)
        j.proc = None
        self.pending.insert(0, j)

    # ------------------------------------------------------------------ chain
    def decode_jobs(self, s: str) -> list[Job]:
        S = SETS[s]
        out = []
        for k in range(S["shards"]):
            cmd = (ENV_S + f"{PYS} -u {CODE}/run_e7.py --windows {S['win']} --tgt {S['tgt']} --out {RAW}/{s} "
                   f"--shard {k} --num-shards {S['shards']} --resume")
            out.append(Job(f"{s}__dec{k}", "dec", cmd, LOGD / f"{s}_dec{k}.log", s))
        return out

    def ok_decode(self, s: str, k: int) -> bool:
        return "E7_SHARD_DONE" in (LOGD / f"{s}_dec{k}.log").read_text(errors="ignore")

    def sh(self, cmd: str, log: Path) -> int:
        if self.a.dry_run:
            self.log(f"DRY {cmd}")
            return 0
        r = subprocess.run(["bash", "-c", f"cd {REPO} && {cmd} >> {log} 2>&1"], stdin=subprocess.DEVNULL)
        return r.returncode

    def lat(self, s: str, n: str, talker: str | None) -> int:
        S = SETS[s]
        O = f"{TE}/{n}"
        if not self.a.dry_run:
            os.makedirs(REPO / O, exist_ok=True)
        if talker is None:
            f, A = f"{O}/textlat_chunk_{n}.json", "--text-only"
        else:
            f, A = f"{O}/lat_chunk_{n}.json", f"--talker {talker}"
        if S["set"] == "en2zh":
            cmd = (ENV_E + f"CUDA_VISIBLE_DEVICES= {PY} -m experiments.mist_w3po_v1.evaluation.en2zh_lat --ref-win "
                   f"{S['ref']} --work {f}.work --run {DEC}/{n} {A} --win {S['cwin']} --out {f}")
        else:
            cmd = (ENV_E + f"CUDA_VISIBLE_DEVICES= {PY} -m experiments.mist_w3po_v1.evaluation.on_refset --ref-win "
                   f"{S['ref']} --dr {S['set']} experiments.mist_w3po_v1.evaluation.speech_latency_lit --run {DEC}/{n} "
                   f"{A} --win {S['cwin']} --out {f}")
        return self.sh(cmd, Path(f"{REPO}/{f}.log"))

    def after_decode(self, s: str) -> None:
        S, n = SETS[s], name(s)
        L = LOGD / f"{s}_post.log"
        T = f"{SPD}/{n}/talker"
        steps = [
            ENV_E + f"{PY} {CODE}/build_e7.py --raw {RAW}/{s} --windows {S['win']} --name {n} --tgt {S['tgt']} --src {S['src']}",
            ENV_E + f"{PY} -m experiments.mist_w3po_v1.evaluation.score_run --run {DEC}/{n} --windows {S['win']} "
                    f"--ref-windows {S['ref']} --set {S['set']}",
            ENV_E + f"{PY} -m experiments.mist_w3po_v1.evaluation.score_run --run {DEC}/{n}_1spk --windows {S['win']} "
                    f"--ref-windows {S['ref']} --set {S['set']}",
            ENV_E + f"{PYE} scripts/gpu_queue/p80_speechonly.py {T}",
        ]
        for c in steps:
            rc = self.sh(c, L)
            if rc:
                self.fail(s, f"post step rc {rc}: {c[-200:]}")
                return
        if self.lat(s, n, None):
            self.log(f"{s}: text latency failed (see its .log); continuing")
        for k in range(ASR_SHARDS):
            cmd = (ENV_E + f"{PYE} -m evaluation.asr_transcribe --input {T}/results_speechonly.jsonl --output "
                   f"{T}/asr_speech.p{k}.jsonl --device cuda:0 --batch-size 8 --resume --num-shards {ASR_SHARDS} "
                   f"--shard-index {k}")
            self.pending.append(Job(f"{s}__asr{k}", "asr", cmd, LOGD / f"{s}_asr{k}.log", s))
        self.stage[s] = "asr"
        self.log(f"{s}: build + text scoring done; ASR queued")

    def after_asr(self, s: str) -> None:
        S, n = SETS[s], name(s)
        L = LOGD / f"{s}_post.log"
        T = f"{SPD}/{n}/talker"
        steps = [
            ENV_E + f"{PY} -m experiments.mist_w3po_v1.evaluation.asr_merge {T} {ASR_SHARDS}",
            ENV_E + f"{PY} -m experiments.mist_w3po_v1.evaluation.asr_hyp {DEC}/{n}/hyp.jsonl {T}/asr_speech_merged.jsonl "
                    f"{DEC}/{n}/hyp_asr.jsonl",
            ENV_E + f"{PY} -m experiments.mist_w3po_v1.evaluation.score_run --run {DEC}/{n} --windows {S['win']} "
                    f"--ref-windows {S['ref']} --set {S['set']} --asr {DEC}/{n}/hyp_asr.jsonl",
        ]
        for c in steps:
            rc = self.sh(c, L)
            if rc:
                self.fail(s, f"asr post step rc {rc}: {c[-200:]}")
                return
        if self.lat(s, n, T):
            self.log(f"{s}: speech latency failed (see its .log)")
        self.stage[s] = "done"
        if not self.a.dry_run:
            (LOGD / f"E7_{s}_DONE").write_text(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()) + "\n")
        self.log(f"E7_SET_DONE {s}")

    def fail(self, s: str, why: str) -> None:
        self.stage[s] = "failed"
        self.failed.append(s)
        self.log(f"E7_SET_FAILED {s}: {why}")

    # ------------------------------------------------------------------ main loop
    def run(self) -> None:
        for s in SETS:
            n = name(s)
            for p in (f"{DEC}/{n}", f"{DEC}/{n}_1spk", f"{SPD}/{n}", f"{TE}/{n}", f"{RAW}/{s}"):
                if (REPO / p).exists() and not self.a.resume:
                    raise SystemExit(f"拒绝覆盖已有输出: {p}")
            self.pending += self.decode_jobs(s)
        self.log(f"E7 queue start: {len(self.pending)} decode jobs; dry={self.a.dry_run}")
        if self.a.dry_run:
            for j in self.pending:
                self.log(f"DRY {j.id}: {j.cmd}")
            for s in SETS:
                self.after_decode(s)
                self.after_asr(s)
            C = self.cards()
            self.log(f"DRY cards: {C}; dec pick: {self.pick(C, 'dec')}")
            return
        while any(v not in ("done", "failed") for v in self.stage.values()):
            for j in list(self.running):
                rc = j.proc.poll()
                if rc is None:
                    continue
                self.running.remove(j)
                good = rc == 0 and (j.kind != "dec" or self.ok_decode(j.set, int(j.id.split("dec")[1])))
                if good:
                    j.done = True
                    self.log(f"DONE {j.id} rc {rc} ({(time.time() - j.started) / 60:.1f} min)")
                elif j.tries < 4:
                    self.log(f"RETRY {j.id} rc {rc}")
                    self.pending.insert(0, j)
                else:
                    self.log(f"GIVEUP {j.id} rc {rc}")
                    self.fail(j.set, f"{j.id} failed {j.tries} times")
            for s in SETS:
                busy = [j for j in self.running + self.pending if j.set == s]
                if self.stage[s] == "decode" and not busy:
                    self.after_decode(s)
                elif self.stage[s] == "asr" and not busy:
                    self.after_asr(s)
            self.pending = [j for j in self.pending if self.stage[j.set] != "failed"]
            try:
                C = self.cards()
            except Exception as e:  # nvidia-smi hiccup
                self.log(f"nvidia-smi failed: {e}")
                time.sleep(30)
                continue
            self.evict(C)
            if self.pending and len(self.running) < MAX_GPU:
                j = self.pending[0]
                c = self.pick(C, j.kind)
                if c is not None:
                    self.pending.pop(0)
                    self.launch(j, c)
            time.sleep(15)
        self.log(f"E7_QUEUE_DONE stages {self.stage} failed {self.failed}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--resume", action="store_true", help="allow existing outputs (relaunch after a stopped queue)")
    a = ap.parse_args()
    os.chdir(REPO)
    Q(a).run()


if __name__ == "__main__":
    main()

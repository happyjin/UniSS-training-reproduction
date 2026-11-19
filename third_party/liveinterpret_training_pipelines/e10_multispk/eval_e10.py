"""E10 / E10-0 test (or dev) evaluation: decode -> build -> exactly E8's scoring chain
(experiments/s2st_baselines_v1/xtranslator_multispk/e8_queue.py post1/post2 + table1_lat.sh).

Per set and TTS seed (Table 1 convention: two seeds, the default and 20261109, run names <n> and <n>_tts2):
  decode     run_e10.py, contiguous shards over GPU slots (simuls2s venv)
  build      build_e10.py (--attr own for E10, relabel for E10-0; E10 also builds <n>_rel for the appendix)
  score_text score_run (main, _1spk)                         (qwen3omni env, as E8)
  speechonly scripts/gpu_queue/p80_speechonly.py              (uniss-eval)
  asr        evaluation.asr_transcribe, 4 shards on GPU slots (uniss-eval; Whisper-large-v3 / Paraformer-zh)
  asr_merge -> asr_hyp -> score_run --asr
  lat        chunk-protocol text/speech latency (on_refset+speech_latency_lit, en2zh_lat) and table1_lat.sh
Nothing is run without --run (the default prints every command). Markers make it resumable; existing outputs are
never overwritten (each tool refuses on its own).

    $PYE e10_multispk/eval_e10.py --mode e10 --adapter <ckpt> --sets ami,ali,en           # dry run
    setsid nohup $PYE e10_multispk/eval_e10.py --mode e10 --adapter <ckpt> --run > log 2>&1 &
"""
from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = Path("/opt/dlami/nvme/jasonleeeli/projects/UniSS")
J = "/opt/dlami/nvme/jasonleeeli"
SIMULS2S = f"{J}/venvs/simuls2s/bin/python"
PY = f"{J}/conda_envs/qwen3omni/bin/python"
PYE = f"{J}/conda_envs/uniss-eval/bin/python"
DEC, SPD, TE = "eval_outputs/mist_w3po_v1/decode", "eval_outputs/mist_w3po_v1/speech", "eval_outputs/mist_w3po_v1/crf/t_eval"
ENV_E = (f"export TMPDIR={J}/work/mist_w3po_v1/tmp HF_HOME={J}/cache/huggingface TORCH_HOME={J}/cache/torch "
         f"XDG_CACHE_HOME={J}/cache MODELSCOPE_CACHE={J}/cache/modelscope PYTHONPATH=.:third_party/Megatron-LM "
         f"OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false; ")
DD = "eval_outputs/mist_w3po_v1/data"
SETS = {   # e8_queue.SETS + run_e10 set names
    "ami": {"win": f"{DD}/windows_test_c2g06_v1/windows_test_online_zh2en.jsonl",
            "cwin": f"{DD}/windows_test_c2g06_chunk_v1/zh2en/windows_dev_online.jsonl",
            "ref": "data/ami_multispk_si_v1/e2e_gstar_zh2en_v1/windows_test_ref.jsonl", "set": "zh2en",
            "tgt": "eng", "src": "cmn", "run_set": "ami_zh2en"},
    "en": {"win": f"{DD}/windows_test_en2zh_c2g06_v1/windows_test_online_en2zh.jsonl",
           "cwin": f"{DD}/windows_test_en2zh_c2g06_chunk_v1/windows_dev_online.jsonl",
           "ref": "data/ami_multispk_si_v1/e2e_gstar_v1/windows_test_ref.jsonl", "set": "en2zh",
           "tgt": "cmn", "src": "eng", "run_set": "ami_en2zh"},
    "ali": {"win": f"{DD}/windows_test_c2g06_v1/windows_test_online_ali.jsonl",
            "cwin": f"{DD}/windows_test_c2g06_chunk_v1/ali/windows_dev_online.jsonl",
            "ref": "data/diarist_alimeeting_refs_v1/windows_test_ref_human.jsonl", "set": "ali",
            "tgt": "eng", "src": "cmn", "run_set": "ali"},
    # dev (joint_lib.sh WZ_DEV / WE_DEV + their reference windows); no Table 1 latency for dev
    "devz": {"win": f"{DD}/windows_zh2en_c2g06_v1/windows_dev_online.jsonl", "cwin": None,
             "ref": "data/ami_multispk_si_v1/e2e_gstar_zh2en_v1/windows_validation_ref.jsonl", "set": "zh2en",
             "tgt": "eng", "src": "cmn", "run_set": "ami_zh2en"},
    "deve": {"win": f"{DD}/windows_en2zh_c2g06_v1/windows_dev_online.jsonl", "cwin": None,
             "ref": "data/ami_multispk_si_v1/e2e_gstar_v1/windows_validation_ref.jsonl", "set": "en2zh",
             "tgt": "cmn", "src": "eng", "run_set": "ami_en2zh"},
}
SEEDS = (("", 0), ("_tts2", 20261109))
ASR_SHARDS = 4


def stop_placeholders(card, log):
    """takeover_slotA_train.sh's rule: SIGTERM the train_mist_workload supervisor (ppid 1, --devices <card>) and its
    children, by PID; slot-A cards only."""
    import os
    import signal
    if str(card) not in ("0", "1", "2", "3"):
        return
    ps = subprocess.run(["ps", "-eo", "pid=,ppid=,args="], capture_output=True, text=True).stdout.splitlines()
    rows = [(int(p), int(pp), a) for p, pp, a in (l.split(None, 2) for l in ps if len(l.split(None, 2)) == 3)]
    for pid, ppid, args in rows:
        if ppid == 1 and "train_mist_workload.py" in args and f"--devices {card} " in args + " ":
            kids = [p for p, pp, _ in rows if pp == pid]
            log(f"card {card}: stop placeholder supervisor {pid} (children {kids})")
            for p in [pid] + kids:
                try:
                    os.kill(p, signal.SIGTERM)
                except ProcessLookupError:
                    pass
            time.sleep(15)


class Runner:
    def __init__(self, a):
        self.a = a
        self.mk = HERE / "output" / "eval" / "markers"
        self.logd = HERE / "output" / "eval" / "logs"
        if a.run:
            self.mk.mkdir(parents=True, exist_ok=True)
            self.logd.mkdir(parents=True, exist_ok=True)
        self.slots = [c for c in a.gpus.split(",") for _ in range(a.procs_per_gpu)]

    def log(self, msg):
        print(time.strftime("%H:%M:%S"), msg, flush=True)

    def done(self, key):
        return (self.mk / f"{key}.ok").exists()

    def mark(self, key):
        if self.a.run:
            (self.mk / f"{key}.ok").write_text(time.strftime("%Y-%m-%dT%H:%M:%SZ\n", time.gmtime()))

    def sh(self, key, cmd, must=True):
        if self.done(key):
            return True
        if not self.a.run:
            print(f"[DRY] {key}: {cmd}")
            return True
        with open(self.logd / f"{key}.log", "a") as fh:
            rc = subprocess.run(["bash", "-lc", f"cd {REPO}; {cmd}"], stdout=fh, stderr=subprocess.STDOUT).returncode
        if rc:
            self.log(f"FAILED {key} rc={rc} (log {self.logd / (key + '.log')})")
            if must:
                raise SystemExit(1)
            return False
        self.mark(key)
        return True

    def parallel(self, key, cmds):
        """cmds: list of (subkey, card_independent_cmd_template with {card}); one per free slot."""
        todo = [(k, c) for k, c in cmds if not self.done(f"{key}.{k}")]
        if not self.a.run:
            for k, c in todo:
                print(f"[DRY] {key}.{k}: {c.format(card=self.slots[0])}")
            return
        free, running, failed = list(self.slots), [], []
        while todo or running:
            while todo and free and not failed:
                k, c = todo.pop(0)
                card = free.pop(0)
                if self.a.stop_placeholders:
                    stop_placeholders(card, self.log)
                fh = open(self.logd / f"{key}.{k}.log", "a")
                p = subprocess.Popen(["bash", "-lc", f"cd {REPO}; {c.format(card=card)}"], stdout=fh,
                                     stderr=subprocess.STDOUT)
                running.append((k, card, p, fh))
                time.sleep(self.a.stagger)
            for item in list(running):
                k, card, p, fh = item
                if p.poll() is not None:
                    fh.close()
                    running.remove(item)
                    free.append(card)
                    if p.returncode:
                        self.log(f"FAILED {key}.{k} rc={p.returncode}")
                        failed.append(k)
                    else:
                        self.mark(f"{key}.{k}")
            if failed and not running:     # let the started jobs finish (no orphans), then stop
                raise SystemExit(1)
            time.sleep(5)

    def resume_decode(self, n, raw, K):
        """A shard with part + stats is complete (mark it); a part without stats is an aborted shard: set it aside
        (rename) so run_e10 can redo it."""
        for k in range(K):
            part, stats = raw / f"part_{k:02d}.jsonl", raw / f"part_{k:02d}.stats.json"
            if self.done(f"{n}.decode.{k:02d}") or not part.exists():
                continue
            if stats.exists():
                self.log(f"{n}: shard {k:02d} complete on disk -> marked")
                self.mark(f"{n}.decode.{k:02d}")
            else:
                dst = part.with_name(part.name + time.strftime(".incomplete_%Y%m%dT%H%M%SZ", time.gmtime()))
                self.log(f"{n}: shard {k:02d} incomplete -> {dst.name}")
                if self.a.run:
                    part.rename(dst)

    def one(self, s, tag, seed):
        a, S = self.a, SETS[s]
        n = f"{'jdev' if s.startswith('dev') else 'jtest'}_{s}_{a.mode}{tag}"
        raw = HERE / "output" / "eval" / n / "raw"
        K = len(self.slots)
        dec = [(f"{k:02d}", f"CUDA_VISIBLE_DEVICES={{card}} {SIMULS2S} {HERE}/run_e10.py --mode {a.mode} "
                            f"--set {S['run_set']} --windows {S['win']} --adapter {a.adapter} --out {raw} "
                            f"--shard {k} --num-shards {K} --seed {seed}") for k in range(K)]
        self.resume_decode(n, raw, K)
        self.parallel(f"{n}.decode", dec)
        system = "LiveInterpret S5 (E10)" if a.mode == "e10" else "LiveInterpret S5 (E10-0)"
        builds = [(n, "own" if a.mode == "e10" else "relabel")]
        if a.mode == "e10":
            builds.append((f"{n}_rel", "relabel"))
        for name, attr in builds:
            T = f"{SPD}/{name}/talker"
            self.sh(f"{name}.build", ENV_E + f"{PYE} {HERE}/build_e10.py --raw {raw} --windows {S['win']} "
                    f"--name {name} --tgt {S['tgt']} --src {S['src']} --attr {attr} --system {shlex.quote(system)}")
            for v in (name, f"{name}_1spk"):
                self.sh(f"{v}.score_text", ENV_E + f"{PY} -m experiments.mist_w3po_v1.evaluation.score_run "
                        f"--run {DEC}/{v} --windows {S['win']} --ref-windows {S['ref']} --set {S['set']}")
            self.sh(f"{name}.speechonly", ENV_E + f"{PYE} scripts/gpu_queue/p80_speechonly.py {T}")
            self.parallel(f"{name}.asr", [(f"p{k}", ENV_E + f"CUDA_VISIBLE_DEVICES={{card}} {PYE} -m "
                                          f"evaluation.asr_transcribe --input {T}/results_speechonly.jsonl --output "
                                          f"{T}/asr_speech.p{k}.jsonl --device cuda:0 --batch-size 8 --resume "
                                          f"--num-shards {ASR_SHARDS} --shard-index {k}") for k in range(ASR_SHARDS)])
            self.sh(f"{name}.asr_merge", ENV_E + f"{PY} -m experiments.mist_w3po_v1.evaluation.asr_merge {T} {ASR_SHARDS}")
            self.sh(f"{name}.asr_hyp", ENV_E + f"{PY} -m experiments.mist_w3po_v1.evaluation.asr_hyp {DEC}/{name}/hyp.jsonl "
                    f"{T}/asr_speech_merged.jsonl {DEC}/{name}/hyp_asr.jsonl")
            self.sh(f"{name}.score_asr", ENV_E + f"{PY} -m experiments.mist_w3po_v1.evaluation.score_run --run {DEC}/{name} "
                    f"--windows {S['win']} --ref-windows {S['ref']} --set {S['set']} --asr {DEC}/{name}/hyp_asr.jsonl")
            if S["cwin"] is None:
                continue
            O = f"{TE}/{name}"
            for kind, f, A in (("textlat", f"{O}/textlat_chunk_{name}.json", "--text-only"),
                               ("speechlat", f"{O}/lat_chunk_{name}.json", f"--talker {T}")):
                if S["set"] == "en2zh":
                    c = (f"mkdir -p {O}; CUDA_VISIBLE_DEVICES= {PY} -m experiments.mist_w3po_v1.evaluation.en2zh_lat "
                         f"--ref-win {S['ref']} --work {f}.work --run {DEC}/{name} {A} --win {S['cwin']} --out {f}")
                else:
                    c = (f"mkdir -p {O}; CUDA_VISIBLE_DEVICES= {PY} -m experiments.mist_w3po_v1.evaluation.on_refset "
                         f"--ref-win {S['ref']} --dr {S['set']} experiments.mist_w3po_v1.evaluation.speech_latency_lit "
                         f"--run {DEC}/{name} {A} --win {S['cwin']} --out {f}")
                self.sh(f"{name}.{kind}", ENV_E + c, must=False)
            self.sh(f"{name}.table1lat", f"bash experiments/mist_w3po_v1/scripts/table1_lat.sh {s} {name}", must=False)
        self.log(f"E10_DONE {n}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["e10", "e10_0"], required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--sets", default="ami,ali,en")
    ap.add_argument("--seeds", default="both", choices=["both", "first"])
    ap.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    ap.add_argument("--procs-per-gpu", type=int, default=4)
    ap.add_argument("--stagger", type=float, default=20.0, help="s between process starts (vLLM-style memory race)")
    ap.add_argument("--run", action="store_true", help="execute (default: print the commands)")
    ap.add_argument("--stop-placeholders", action="store_true",
                    help="before each launch on card c (0-3 only), stop that card's no-checkpoint placeholder by PID")
    a = ap.parse_args()
    a.adapter = str(Path(a.adapter).resolve())
    r = Runner(a)
    for s in a.sets.split(","):
        for tag, seed in (SEEDS if a.seeds == "both" else SEEDS[:1]):
            r.one(s, tag, seed)


if __name__ == "__main__":
    sys.exit(main())

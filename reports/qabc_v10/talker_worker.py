"""v10 speech without a vLLM start per run: one long-lived Talker engine per worker speaks many blocks files.

Each task is one speech run as run_si_probe.sh / make_v10_jobs.py devspeech would make it -- si_infer with the same
arguments (T1c@1500, --text-policy file, block speed policy, history ban 8, two shards --shard 0/1 --num-shards 2 run one
after the other, so every request gets the seed si_infer gives it: seed + 1000 * item + round) -- but si_infer.main()
is called in-process and vllm.LLM is patched to hand back one cached engine, so the engine starts once per worker.
Requests are seeded per request, so the cached engine changes nothing but scheduling.

  talker  claims tasks in order (an O_EXCL file under <claims>/), skips a task whose run is complete (asr_speech.jsonl
          in its folder or in a -r<n> re-run folder), writes into the task's folder, or into the first free -r<n>
          folder when the folder already exists (a failed or partial earlier run is kept as it is)
  asr     for every task folder with both shard files and no asr_speech.jsonl: speech_only.py, then the speech-only ASR
          (evaluation.asr_transcribe, batch 4), claimed the same way
Task file: jsonl of {"key", "blocks", "direction", "window", "seed" (0 = si_infer's default), "out", "items" (optional)}.

    CUDA_VISIBLE_DEVICES=<card> python reports/qabc_v10/talker_worker.py talker <tasks.jsonl> --gpu-mem 0.10
    CUDA_VISIBLE_DEVICES=<card> python reports/qabc_v10/talker_worker.py asr <tasks.jsonl>
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
EXPORT = REPO / "eval_outputs/uniss_qwen3_omni_v1/si_t1c_20261021T181440Z/autoext/export_1500/uniss"
CLAIMS = REPO / "eval_outputs/uniss_qwen3_omni_v1/qabc_v10/claims"
PYE = "/opt/dlami/nvme/jasonleeeli/conda_envs/uniss-eval/bin/python"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def variants(out: Path):
    yield out
    m = re.match(r"(.*?)(_m[\d.]+)(/.*)?$", str(out))
    if "/speech/system_" in str(out):   # system_<name>_m<w>/_<d>/realsi_<d> -> system_<name>-r<n>_m<w>/...
        head, mw, tail = m.group(1), m.group(2), m.group(3) or ""
        for n in range(1, 10):
            yield Path(f"{head}-r{n}{mw}{tail}")
    else:                                 # devspeech/<run> -> devspeech/<run>-r<n>
        for n in range(1, 10):
            yield Path(f"{out}-r{n}")


def complete(folder: Path) -> bool:
    return (folder / "asr_speech.jsonl").exists()


def claim(name: str) -> bool:
    CLAIMS.mkdir(parents=True, exist_ok=True)
    try:
        os.close(os.open(CLAIMS / name, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
        return True
    except FileExistsError:
        return False


def talker(tasks, gpu_mem):
    import vllm
    real, cache = vllm.LLM, {}

    def cached(**kw):
        key = kw.get("model")
        if key not in cache:
            kw = {**kw, "gpu_memory_utilization": gpu_mem}
            cache[key] = real(**kw)
        return cache[key]
    vllm.LLM = cached
    sys.path.insert(0, str(REPO))
    from experiments.uniss_qwen3_omni_v1.evaluation import si_infer
    while True:
        waiting = 0
        for t in tasks:
            outs = list(variants(Path(t["out"])))
            if any(complete(o) for o in outs):
                continue
            if not os.path.exists(t["blocks"]):   # its Thinker run is not there yet: come back to it
                waiting += 1
                continue
            n = next((n for n in (1, 2, 3) if not (CLAIMS / f"talker_{t['key']}_a{n}").exists()
                      or not (CLAIMS / f"failed_{t['key']}_a{n}").exists()), None)
            if n is None or (CLAIMS / f"talker_{t['key']}_a{n}").exists() or not claim(f"talker_{t['key']}_a{n}"):
                continue   # three failed attempts, or another worker has it
            out = next(o for o in outs if not o.exists())
            out.mkdir(parents=True)
            log(f"talker {t['key']} attempt {n} -> {out}")
            ok = True
            for shard in (0, 1):
                sys.argv = ["si_infer", "--model", str(EXPORT), "--direction", t["direction"], "--window", str(t["window"]),
                            "--limit", "0", "--output", str(out), "--shard", str(shard), "--num-shards", "2",
                            "--text-policy", "file", "--min-words", "2", "--blocks-file", t["blocks"],
                            "--speed-policy", "block", "--history-ban", "8", "--gpu-mem", str(gpu_mem)]
                if t.get("seed"):
                    sys.argv += ["--seed", str(t["seed"])]
                if t.get("items"):
                    sys.argv += ["--items-file", t["items"]]
                try:
                    si_infer.main()
                except Exception as e:  # noqa: BLE001
                    log(f"FAILED {t['key']} shard {shard}: {e!r}")
                    ok = False
                    break
            if not ok:
                claim(f"failed_{t['key']}_a{n}")
            log(f"talker {'done' if ok else 'failed'} {t['key']}")
        if not waiting or os.path.exists(REPO / "logs/qabc_v10/TALKER_STOP"):
            log("TALKER_PASS_DONE" if not waiting else "stopped")
            return
        time.sleep(60)


def asr(tasks):
    while True:
        pending = 0
        for t in tasks:
            for o in variants(Path(t["out"])):
                if not o.exists():
                    break
                if complete(o) or not ((o / "part_0.jsonl").exists() and (o / "part_1.jsonl").exists()):
                    continue
                pending += 1
                if not claim("asr_" + str(o.relative_to(REPO)).replace("/", "__")):
                    continue
                log(f"asr {o}")
                ok = subprocess.run(f"{PYE} reports/uniss_si_v4/speech_only.py {o} > /dev/null && PYTHONPATH=. {PYE} -m "
                                    f"evaluation.asr_transcribe --input {o}/results_speechonly.jsonl --output {o}/asr_speech.jsonl "
                                    f"--device cuda:0 --batch-size 4 > {o}/asr.log 2>&1", shell=True, cwd=REPO).returncode
                log(f"asr {'done' if ok == 0 else 'FAILED'} {o}")
        if all(any(complete(o) for o in variants(Path(t["out"]))) for t in tasks):
            log("ASR_ALL_DONE")
            return
        if os.path.exists(REPO / "logs/qabc_v10/ASR_STOP"):
            return
        time.sleep(30)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("talker", "asr"))
    ap.add_argument("tasks")
    ap.add_argument("--gpu-mem", type=float, default=0.10)
    a = ap.parse_args()
    tasks = [json.loads(l) for l in open(a.tasks, encoding="utf-8")]
    talker(tasks, a.gpu_mem) if a.mode == "talker" else asr(tasks)


if __name__ == "__main__":
    main()

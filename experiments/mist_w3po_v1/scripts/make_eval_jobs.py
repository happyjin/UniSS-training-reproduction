"""Write gpuq_w3 job files for one evaluation: decode shards (one job each, ~66 GB on one card) -> merge + text score
(CPU) -> [speech: prep (CPU) -> 8 talker shards (24 GB) -> ASR (12 GB) -> hyp_asr + speech score (CPU)].

    python make_eval_jobs.py --qdir logs/mist_w3po_v1/queue --name dev_s0all --thinker <merged> [--set dev|test|ali]
        [--shards 8] [--context 2] [--policy wait|lag --lag-ms 2000] [--lora <lora.pt>] [--speech] [--prio 50]
        [--after <job id> ...]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

J = "/opt/dlami/nvme/jasonleeeli"
ENV = (f"export TMPDIR={J}/work/mist_w3po_v1/tmp HF_HOME={J}/cache/huggingface TORCH_HOME={J}/cache/torch "
       f"XDG_CACHE_HOME={J}/cache TRITON_CACHE_DIR={J}/cache/triton VLLM_CACHE_ROOT={J}/cache/vllm_omni_030 "
       f"PYTHONPATH=.:third_party/Megatron-LM PATH={J}/venvs/vllm_omni_030/bin:$PATH OMP_NUM_THREADS=8 "
       f"TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; ")
PY = f"{J}/conda_envs/qwen3omni/bin/python"
PYE = f"{J}/conda_envs/uniss-eval/bin/python"
PYV = f"{J}/venvs/vllm_omni_030/bin/python"
SETS = {
    "dev": ("eval_outputs/mist_w3po_v1/data/windows_zh2en_v1/windows_dev_online.jsonl",
            "data/ami_multispk_si_v1/e2e_gstar_zh2en_v1/windows_validation_ref.jsonl", "zh2en"),
    "test": ("data/online_v6_20261031/windows_test_online_zh2en.jsonl",
             "data/ami_multispk_si_v1/e2e_gstar_zh2en_v1/windows_test_ref.jsonl", "zh2en"),
    "ali": ("data/online_v6_20261031/windows_test_online_ali.jsonl", "data/alimeeting_si_v1/windows_test_ref.jsonl", "ali"),
}
T1C = "eval_outputs/uniss_qwen3_omni_v1/si_t1c_20261021T181440Z/autoext/export_1500/uniss"
M = "experiments.uniss_qwen3_omni_v1.evaluation"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--qdir", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--thinker", required=True)
    ap.add_argument("--set", default="dev", choices=list(SETS))
    ap.add_argument("--shards", type=int, default=8)
    ap.add_argument("--context", type=int, default=2)
    ap.add_argument("--policy", default="wait")
    ap.add_argument("--lag-ms", type=int, default=2000)
    ap.add_argument("--lora", default="")
    ap.add_argument("--speech", action="store_true")
    ap.add_argument("--prio", type=int, default=50)
    ap.add_argument("--after", nargs="*", default=[])
    ap.add_argument("--dec-cards", type=int, default=1, help="cards per decode shard (model split by --max-gib)")
    ap.add_argument("--dec-mem", type=float, default=66, help="GB per card of one decode shard")
    ap.add_argument("--max-gib", type=int, default=120)
    ap.add_argument("--dec-only", default="", help="comma list: cards the decode shards may use")
    ap.add_argument("--id-suffix", default="", help="suffix for job ids (re-chaining an existing run)")
    ap.add_argument("--extra-deps", nargs="*", default=[], help="with --existing: job ids the score job waits for")
    ap.add_argument("--existing", type=int, default=0,
                    help="N: the decode exists (N shards, started elsewhere): wait for its shards, no decode jobs")
    a = ap.parse_args()
    win, ref, sset = SETS[a.set]
    D = f"eval_outputs/mist_w3po_v1/decode/{a.name}"
    if Path(D).exists() and not a.existing:
        raise SystemExit(f"拒绝覆盖已有输出: {D}")
    jobs = []
    extra = f"--lora {a.lora}" if a.lora else ""
    for k in range(0 if a.existing else a.shards):
        # every attempt writes a new file and resumes from the earlier attempts' windows (retries are safe)
        jobs.append({"id": f"{a.name}__dec{k}", "mem_gb": a.dec_mem, "cards": a.dec_cards, "deps": a.after,
                     "retries": 2,
                     "cmd": ENV + f"mkdir -p {D} && R=$(date +%s) && PREV=$(ls {D}/hyp_s{k}_a*.jsonl 2>/dev/null); "
                            f"{PY} -m experiments.mist_w3po_v1.evaluation.online_decode "
                            f"--thinker {a.thinker} --windows {win} --out {D}/hyp_s{k}_a$R.jsonl "
                            f"${{PREV:+--resume-from $PREV}} --policy {a.policy} "
                            f"--lag-ms {a.lag_ms} --context {a.context} --dir zh2en --batch 16 --max-gib {a.max_gib} "
                            f"--shard {k} --num-shards {a.shards} --shard-by meeting {extra} > {D}/infer_s{k}_a$R.log 2>&1 "
                            f"&& grep -q W3PO_ONLINE_DONE {D}/infer_s{k}_a$R.log"})
    if a.dec_only:
        for j in jobs:
            j["only"] = [int(c) for c in a.dec_only.split(",")]
    dec_ids = [j["id"] for j in jobs]
    wait = (f"until [ $(grep -l W3PO_ONLINE_DONE {D}/infer_s*.log | wc -l) -ge {a.existing} ]; do sleep 30; done; "
            if a.existing else "")
    jobs.append({"id": f"{a.name}__score", "mem_gb": 0, "deps": dec_ids + a.extra_deps,
                 "cmd": ENV + wait + f"[ -e {D}/score_text.json ] || {PY} -m experiments.mist_w3po_v1.evaluation.score_run --run {D} --windows {win} "
                        f"--ref-windows {ref} --set {sset} > {D}/score.log 2>&1"})
    if a.speech:
        SP = f"eval_outputs/mist_w3po_v1/speech/{a.name}"
        data, sess, diar = (("data/alimeeting_si_v1", "near_mix", "nemotron_lowlat") if a.set == "ali" else
                            ("data/ami_multispk_si_v1", "zh_mix_v8", "nemotron_lowlat_zh8"))
        jobs.append({"id": f"{a.name}__sprep", "mem_gb": 0, "deps": [f"{a.name}__score"],
                     "cmd": ENV + f"mkdir -p {SP}/talker && GSTAR_DATA={data} {PYE} -m {M}.gstar_talker_v6 prep "
                            f"--hyp {D}/hyp.jsonl --windows {win} --causal --blocks --preset lowlat --session {sess} "
                            f"--diar {diar} --sep ' ' --out {SP}/talker_in_all > {SP}/prep.log 2>&1 && python3 "
                            f"reports/gstar_si_v7/drop_long_streams.py {SP}/talker_in_all {SP}/talker_in 400 >> {SP}/prep.log 2>&1"})
        tk = []
        for k in range(8):
            tk.append(f"{a.name}__talk{k}")
            jobs.append({"id": tk[-1], "mem_gb": 26, "deps": [f"{a.name}__sprep"], "retries": 20,
                         "cmd": ENV + f"VLLM_USE_FLASHINFER_SAMPLER=0 OMP_NUM_THREADS=16 {PYV} -m {M}.si_infer "
                                f"--model {T1C} --direction zh2en --window 1 --output {SP}/talker --text-policy file "
                                f"--blocks-file {SP}/talker_in/blocks.jsonl --items-file {SP}/talker_in/items.jsonl "
                                f"--speed-policy fixed --history-ban 8 --max-model-len 32768 --gpu-mem 0.1 --shard {k} "
                                f"--num-shards 8 > {SP}/talk{k}.log 2>&1"})
        jobs.append({"id": f"{a.name}__asr", "mem_gb": 14, "deps": tk,
                     "cmd": ENV + f"for k in 0 1 2 3 4 5 6 7; do [ -s {SP}/talker/part_$k.jsonl ] || MISSING=1; done; "
                            f"[ -z \"$MISSING\" ] && cat {SP}/talker/part_*.jsonl > {SP}/talker/results.jsonl && "
                            f"{PYE} scripts/gpu_queue/p80_speechonly.py {SP}/talker && {PYE} -m evaluation.asr_transcribe "
                            f"--input {SP}/talker/results_speechonly.jsonl --output {SP}/talker/asr_speech.jsonl "
                            f"--device cuda:0 --batch-size 8 > {SP}/asr.log 2>&1"})
        jobs.append({"id": f"{a.name}__sscore", "mem_gb": 0, "deps": [f"{a.name}__asr"],
                     "cmd": ENV + f"{PY} -m experiments.mist_w3po_v1.evaluation.asr_hyp {D}/hyp.jsonl "
                            f"{SP}/talker/asr_speech.jsonl {D}/hyp_asr.jsonl && {PY} -m "
                            f"experiments.mist_w3po_v1.evaluation.score_run --run {D} --windows {win} --ref-windows {ref} "
                            f"--set {sset} --asr {D}/hyp_asr.jsonl > {D}/score_asr.log 2>&1"})
    if a.id_suffix:   # a second chain for the same outputs (the first one's ids are spent)
        ids = {j["id"] for j in jobs}
        for j in jobs:
            j["id"] += a.id_suffix
            j["deps"] = [d + a.id_suffix if d in ids else d for d in j["deps"]]
    qd = Path(a.qdir) / "jobs"
    qd.mkdir(parents=True, exist_ok=True)
    for i, j in enumerate(jobs):
        f = qd / f"{a.prio:03d}_{a.name}{a.id_suffix}_{i:02d}.json"
        with open(f, "x") as fh:
            json.dump(j, fh, indent=1)
    print(f"{len(jobs)} jobs for {a.name}")


if __name__ == "__main__":
    main()

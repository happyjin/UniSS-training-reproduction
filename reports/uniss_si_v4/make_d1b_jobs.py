"""gpuq7 jobs for D1 with A2 (prereg_dirsel.zh-CN.md: A2 joins the same grid once trained): A2+SDR at b0 = 0, 2, 3 on the
UniST dev and the spontaneous dev (P2 runs b0 = 1), into the folders P2 writes (dev_a2, devsp_a2). "mem" is what a TP2
engine at --gpu-mem 0.3 really holds.

    python3 reports/uniss_si_v4/make_d1b_jobs.py reports/uniss_si_v4/jobs/d1b_jobs.json
"""
import json, sys

O = "eval_outputs/uniss_qwen3_omni_v1"
A2 = f"{O}/thinker_op_v7sp_20261029/thinker_merged_i200"
RATE = {"en2zh": "0.191", "zh2en": "0.339"}
DEVW = "0.5,0.75,1,1.25,1.5,2,2.5,3"
DEVS = {"un": ("data/si_thinker_oracle_prefix_v1/dev_items_{d}.jsonl", "dev_a2"),
        "sp": ("data/si_spont_v1/dev_{d}_ref.jsonl", "devsp_a2")}
ST = ("CUDA_VISIBLE_DEVICES=${GPUS// /,} PATH=/opt/dlami/nvme/jasonleeeli/venvs/vllm_omni_030/bin:$PATH OMP_NUM_THREADS=16 "
      "VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_CACHE_ROOT=/opt/dlami/nvme/jasonleeeli/cache/vllm_omni_030 PYTHONPATH=.:.tmp/pylib_jieba "
      "/opt/dlami/nvme/jasonleeeli/venvs/vllm_omni_030/bin/python -m experiments.uniss_qwen3_omni_v1.evaluation.si_thinker_stream "
      f"--tp 2 --gpu-mem 0.3 --policy selfwait --thinker {A2} --thinker-tag a2")
jobs = [{"id": f"d1b_a2_{dev}_{d}", "gpus": 2, "vllm": True, "mem": 56000, "deps": [],
         "cmd": f"{ST} --wait-biases=0,2,3 --sw-queue-rate {RATE[d]} --direction {d} --windows {DEVW} "
                f"--items-file {items.format(d=d)} --out {O}/si_v4/{out}"}
        for d in ("zh2en", "en2zh") for dev, (items, out) in DEVS.items()]
with open(sys.argv[1], "x") as fh:
    json.dump(jobs, fh, indent=1)
print(len(jobs), "jobs")

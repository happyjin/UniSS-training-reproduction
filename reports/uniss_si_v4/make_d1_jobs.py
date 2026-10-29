"""gpuq7 jobs for D1 (prereg_dirsel.zh-CN.md): the per-direction choice of posterior and prior bias on dev only.
s0 and bma3 with SDR at b0 = 0, 2, 3 (b0 = 1 exists) on the UniST dev and the spontaneous dev, the windows of the A0 dev
runs; bma3 also at b0 = 1 on the spontaneous dev (not run before). One file per bias, beside the existing dev runs.

    python3 reports/uniss_si_v4/make_d1_jobs.py reports/uniss_si_v4/jobs/d1_jobs.json
"""
import json, sys

O = "eval_outputs/uniss_qwen3_omni_v1"
V4 = f"{O}/si_v4"
M = {"s0": f"{O}/thinker_op_v3_20261023T213631Z/thinker_merged_i200", "bma3": f"{O}/thinker_op_v3_bma3/thinker_merged"}
RATE = {"en2zh": "0.191", "zh2en": "0.339"}
DEVW = "0.5,0.75,1,1.25,1.5,2,2.5,3"
DEVS = {"un": ("data/si_thinker_oracle_prefix_v1/dev_items_{d}.jsonl", "dev_{m}"),
        "sp": ("data/si_spont_v1/dev_{d}_ref.jsonl", "devsp_{m}")}
ST = ("CUDA_VISIBLE_DEVICES=${GPUS// /,} PATH=/opt/dlami/nvme/jasonleeeli/venvs/vllm_omni_030/bin:$PATH OMP_NUM_THREADS=16 "
      "VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_CACHE_ROOT=/opt/dlami/nvme/jasonleeeli/cache/vllm_omni_030 PYTHONPATH=.:.tmp/pylib_jieba "
      "/opt/dlami/nvme/jasonleeeli/venvs/vllm_omni_030/bin/python -m experiments.uniss_qwen3_omni_v1.evaluation.si_thinker_stream "
      "--tp 2 --gpu-mem 0.3 --policy selfwait")
jobs = []
for d in ("zh2en", "en2zh"):   # Zh->En first: the direction the change is for
    for m in ("s0", "bma3"):
        for dev, (items, out) in DEVS.items():
            biases = "0,1,2,3" if (m, dev) == ("bma3", "sp") else "0,2,3"
            jobs.append({"id": f"d1_{m}_{dev}_{d}", "gpus": 2, "vllm": True, "mem": 46000, "deps": [],
                         "cmd": f"{ST} --thinker {M[m]} --thinker-tag {m} --wait-biases={biases} --sw-queue-rate {RATE[d]} "
                                f"--direction {d} --windows {DEVW} --items-file {items.format(d=d)} "
                                f"--out {V4}/{out.format(m=m)}"})
with open(sys.argv[1], "x") as fh:
    json.dump(jobs, fh, indent=1)
print(len(jobs), "jobs")

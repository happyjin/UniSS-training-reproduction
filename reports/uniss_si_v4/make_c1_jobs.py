"""gpuq7 jobs for the review controls of the single-speaker paper (v4/v5), RealSI text side on LA2's 15 windows unless noted:
  la2bw   LocalAgreement on the ABC adapter's own drafts (s0, wait marker banned): same drafts, only the rule differs
  la2qg   LocalAgreement on the frozen Thinker gated by the tracked speech queue (r = 0.191 / 0.339): a non-Bayesian queue rule
  la2low  LocalAgreement at read windows 0.2 / 0.3 / 0.4 s: can it reach the latencies the Bayesian policy reaches?
  s0qb    A0's posterior with the queue as a binary gate instead of SDR's derived threshold (--sw-queue-binary)

    python3 reports/uniss_si_v4/make_c1_jobs.py reports/uniss_si_v4/jobs/c1_jobs.json
"""
import json, sys

O = "eval_outputs/uniss_qwen3_omni_v1"
V4 = f"{O}/si_v4"
S0 = f"{O}/thinker_op_v3_20261023T213631Z/thinker_merged_i200"
RATE = {"en2zh": "0.191", "zh2en": "0.339"}
GRID = "0.5,0.6,0.75,0.9,1,1.1,1.25,1.5,1.75,2,2.5,3,4,5,6"
ST = ("CUDA_VISIBLE_DEVICES=${GPUS// /,} PATH=/opt/dlami/nvme/jasonleeeli/venvs/vllm_omni_030/bin:$PATH OMP_NUM_THREADS=16 "
      "VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_CACHE_ROOT=/opt/dlami/nvme/jasonleeeli/cache/vllm_omni_030 PYTHONPATH=.:.tmp/pylib_jieba "
      "/opt/dlami/nvme/jasonleeeli/venvs/vllm_omni_030/bin/python -m experiments.uniss_qwen3_omni_v1.evaluation.si_thinker_stream "
      "--tp 2 --gpu-mem 0.3")
jobs = []
for d in ("en2zh", "zh2en"):
    spec = {"la2bw": f"--policy la2 --thinker {S0} --ban-wait --windows {GRID} --out {V4}/realsi_la2_s0bw",
            "la2qg": f"--policy la2 --queue-rate {RATE[d]} --la2-queue-gate --windows {GRID} --out {V4}/realsi_la2qg",
            "la2low": f"--policy la2 --windows 0.2,0.3,0.4 --out {V4}/realsi_la2low",
            "s0qb": f"--policy selfwait --thinker {S0} --thinker-tag s0 --wait-biases=1 --sw-queue-rate {RATE[d]} --sw-queue-binary "
                    f"--windows {GRID} --out {V4}/realsi_grid_s0qb"}
    for k, a in spec.items():
        jobs.append({"id": f"{k}_{d}", "gpus": 2, "vllm": True, "mem": 46000, "deps": [], "cmd": f"{ST} --direction {d} {a}"})
with open(sys.argv[1], "x") as fh:
    json.dump(jobs, fh, indent=1)
print(len(jobs), "jobs")

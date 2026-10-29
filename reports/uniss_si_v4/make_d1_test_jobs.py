"""gpuq7 jobs for the D1 RealSI test (prereg_dirsel.zh-CN.md): the dev choice for Zh->En (bma3, b0 = 2, SDR) on LA2's
15 read windows, split in two jobs; En->Zh kept the default (s0, b0 = 1), which v6 already has. "mem" is what a TP2
engine at --gpu-mem 0.3 really holds (about 55 GB a card).

    python3 reports/uniss_si_v4/make_d1_test_jobs.py reports/uniss_si_v4/jobs/d1_test_jobs.json
"""
import json, sys

O = "eval_outputs/uniss_qwen3_omni_v1"
ST = ("CUDA_VISIBLE_DEVICES=${GPUS// /,} PATH=/opt/dlami/nvme/jasonleeeli/venvs/vllm_omni_030/bin:$PATH OMP_NUM_THREADS=16 "
      "VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_CACHE_ROOT=/opt/dlami/nvme/jasonleeeli/cache/vllm_omni_030 PYTHONPATH=.:.tmp/pylib_jieba "
      "/opt/dlami/nvme/jasonleeeli/venvs/vllm_omni_030/bin/python -m experiments.uniss_qwen3_omni_v1.evaluation.si_thinker_stream "
      f"--tp 2 --gpu-mem 0.3 --policy selfwait --thinker {O}/thinker_op_v3_bma3/thinker_merged --thinker-tag bma3")
SPLIT = ["0.5,0.6,0.75,0.9,1,1.1,1.25,1.5", "1.75,2,2.5,3,4,5,6"]
jobs = [{"id": f"d1test_bma3_b2_zh2en_{i}", "gpus": 2, "vllm": True, "mem": 56000, "deps": [],
         "cmd": f"{ST} --wait-biases=2 --sw-queue-rate 0.339 --direction zh2en --windows {w} --out {O}/si_v4/realsi_grid_bma3"}
        for i, w in enumerate(SPLIT)]
with open(sys.argv[1], "x") as fh:
    json.dump(jobs, fh, indent=1)
print(len(jobs), "jobs")

"""gpuq7 jobs for evaluating candidate A2 (ABC trained with spontaneous audio added; run_v3_eval5.sh trains it), each
waiting for data/si_thinker_oracle_prefix_v7_spont/TRAINED:
  dev_*   A2+SDR (b0 = 1) on the spontaneous dev and on the UniST dev, the windows of the A0 dev runs -- the selection
          rule of amendment 1 is computed from these alone
  test_*  depend on the dev runs: A2 whole-sentence on RealSI (content check) and A2 with and without SDR on LA2's
          15 windows (text side), reported whatever the selection says (unselected candidates go to the appendix)

    python3 reports/uniss_si_v4/make_p2_jobs.py reports/uniss_si_v4/jobs/p2_jobs.json
"""
import json, sys

O = "eval_outputs/uniss_qwen3_omni_v1"
V4 = f"{O}/si_v4"
A2 = f"{O}/thinker_op_v7sp_20261029/thinker_merged_i200"
RATE = {"en2zh": "0.191", "zh2en": "0.339"}
DEVW = "0.5,0.75,1,1.25,1.5,2,2.5,3"
GRID = "0.5,0.6,0.75,0.9,1,1.1,1.25,1.5,1.75,2,2.5,3,4,5,6"
WAIT = "until [ -f data/si_thinker_oracle_prefix_v7_spont/TRAINED ] && [ -f " + A2 + "/config.json ]; do sleep 60; done; "
ST = ("CUDA_VISIBLE_DEVICES=${GPUS// /,} PATH=/opt/dlami/nvme/jasonleeeli/venvs/vllm_omni_030/bin:$PATH OMP_NUM_THREADS=16 "
      "VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_CACHE_ROOT=/opt/dlami/nvme/jasonleeeli/cache/vllm_omni_030 PYTHONPATH=.:.tmp/pylib_jieba "
      "/opt/dlami/nvme/jasonleeeli/venvs/vllm_omni_030/bin/python -m experiments.uniss_qwen3_omni_v1.evaluation.si_thinker_stream "
      f"--tp 2 --gpu-mem 0.3 --policy selfwait --thinker {A2} --thinker-tag a2")
jobs, dev_ids = [], []
for d in ("en2zh", "zh2en"):
    for name, items, out in (("sp", f"data/si_spont_v1/dev_{d}_ref.jsonl", "devsp_a2"),
                             ("un", f"data/si_thinker_oracle_prefix_v1/dev_items_{d}.jsonl", "dev_a2")):
        jobs.append({"id": f"dev{name}_a2_{d}", "gpus": 2, "vllm": True, "mem": 46000, "deps": [],
                     "cmd": WAIT + f"{ST} --wait-biases=1 --sw-queue-rate {RATE[d]} --direction {d} --windows {DEVW} "
                                   f"--items-file {items} --out {V4}/{out}"})
        dev_ids.append(jobs[-1]["id"])
for d in ("en2zh", "zh2en"):
    jobs.append({"id": f"test_off_a2_{d}", "gpus": 2, "vllm": True, "mem": 46000, "deps": list(dev_ids),
                 "cmd": f"{ST} --wait-biases=0 --direction {d} --windows 60 --out {V4}/realsi_off_a2"})
    for sdr, extra in (("q", f"--sw-queue-rate {RATE[d]}"), ("noq", "")):
        jobs.append({"id": f"test_grid_a2_{sdr}_{d}", "gpus": 2, "vllm": True, "mem": 46000, "deps": list(dev_ids),
                     "cmd": f"{ST} --wait-biases=1 {extra} --direction {d} --windows {GRID} --out {V4}/realsi_grid_a2"})
with open(sys.argv[1], "x") as fh:
    json.dump(jobs, fh, indent=1)
print(len(jobs), "jobs")

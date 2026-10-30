"""gpuq7 jobs for paper v8 (the unfinished items of docs/migration/new_server_run_guide.zh-CN.md section 6.2), on the new
server, stacked beside the multi-speaker training:
  acl_la2_w15   LA2 on ACL60/60-dev at the registered windows it lacked (0.6 0.9 1.1 1.25 1.75 2.5 s), same command as
                acl6060_dev_v1 / acl6060_dev_v1_la2low (base Thinker), into a new folder acl6060_dev_v1_w15
  acl_a0_w15    A0 (s0 adapter, b0 = 1, SDR 0.191) at the registered windows it lacked (0.6 0.75 0.9 1.1 1.25 1.5 1.75
                2.5 s), same folder; with acl6060_dev_v1 both curves then have the 15 registered windows
  simo_*        SIM-O of the CVSS-T Table 1 runs (frozen Thinker, s0) with wavlm-base-plus-sv: the only speaker-verification
                WavLM here (the cached microsoft/wavlm-large has no x-vector head); CPU threads capped at 8
  sp_*          the s1 speech points of the tier table that were cut on the old server (tier2_speech_jobs_spec.json
                entries without asr_speech.jsonl), talker seed 0 as the rest of that table

    python3 reports/uniss_si_v4/make_v8_jobs.py reports/uniss_si_v4/jobs/v8_jobs.json
"""
import json
import os
import subprocess
import sys

O = "eval_outputs/uniss_qwen3_omni_v1"
V4 = f"{O}/si_v4"
SP = f"{V4}/speech"
A0 = f"{O}/thinker_op_v3_20261023T213631Z/thinker_merged_i200"
PYE = "/opt/dlami/nvme/jasonleeeli/conda_envs/uniss-eval/bin/python"
ST = ("CUDA_VISIBLE_DEVICES=${GPUS// /,} PATH=/opt/dlami/nvme/jasonleeeli/venvs/vllm_omni_030/bin:$PATH OMP_NUM_THREADS=16 "
      "VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_CACHE_ROOT=/opt/dlami/nvme/jasonleeeli/cache/vllm_omni_030 PYTHONPATH=.:.tmp/pylib_jieba "
      "/opt/dlami/nvme/jasonleeeli/venvs/vllm_omni_030/bin/python -m experiments.uniss_qwen3_omni_v1.evaluation.si_thinker_stream "
      "--tp 2 --gpu-mem 0.3")
ACL = "--direction en2zh --items-file data/acl6060_dev_v1/items.jsonl"
jobs = [
    {"id": "acl_la2_w15", "gpus": 2, "vllm": True, "mem": 56000, "deps": [],
     "cmd": f"{ST} {ACL} --out {O}/acl6060_dev_v1_w15 --windows 0.6,0.9,1.1,1.25,1.75,2.5"},
    {"id": "acl_a0_w15", "gpus": 2, "vllm": True, "mem": 56000, "deps": [],
     "cmd": f"{ST} --policy selfwait --thinker {A0} --thinker-tag op1 --wait-biases=1 --sw-queue-rate 0.191 {ACL} "
            f"--out {O}/acl6060_dev_v1_w15 --windows 0.6,0.75,0.9,1.1,1.25,1.5,1.75,2.5"},
]
for m in ("base", "s0"):
    for d in ("en2zh", "zh2en"):
        jobs.append({"id": f"simo_{m}_{d}", "gpus": 1, "vllm": False, "mem": 6000, "deps": [],
                     "cmd": f"OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONPATH=. {PYE} reports/uniss_si_v4/cvss_table1_metrics.py "
                            f"{SP}/cvss_{m}/cvss_{d} {V4}/cvss_table1/{m}_{d} $GPU0"})

spec = [e for e in json.load(open("reports/uniss_si_v4/jobs/tier2_speech_jobs_spec.json"))
        if not os.path.exists(f"{SP}/system_{e['name']}_m{e['window']}/_{e['dir']}/realsi_{e['dir']}/asr_speech.jsonl")]
spec_path = sys.argv[1].replace(".json", "_speech_spec.json")
with open(spec_path, "x") as fh:
    json.dump(spec, fh, indent=1)
sp = sys.argv[1].replace(".json", "_speech.json")
subprocess.run([sys.executable, "reports/uniss_si_v4/make_speech_jobs.py", sp, spec_path], check=True)
jobs += json.load(open(sp))
with open(sys.argv[1], "x") as fh:
    json.dump(jobs, fh, indent=1)
print(len(jobs), "jobs;", len(spec), "speech points:", [(e["name"], e["window"]) for e in spec])

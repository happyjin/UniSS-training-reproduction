"""gpuq7 jobs for the latency-tier table of paper v7 (SimulS2ST-Omni Table 3 style, m1-m6), cards 0-3:
  dev_s{1,2}_*  the dev-only choice of the adapter seed (prereg_v4 rule: mean E1 over UniST dev and spontaneous dev,
                b0 = 1 with SDR, the 8 dev windows; s0 exists in dev_s0 / devsp_s0)
  acl_la2low_*  LA2 on ACL60/60-dev at read windows 0.2 / 0.3 / 0.4 s (LA2's points start at 1.66 s StreamLAAL, above
                tier m1), into a new folder acl6060_dev_v1_la2low
  sp_la2low*    the Talker speaks LA2's RealSI drafts at windows 0.2 / 0.3 / 0.4 s (si_v4/realsi_la2low), three talker
                seeds, as the other RealSI speech points; named la2low / la2low-zh so that the pre-registered 15-window
                LA2 curve is unchanged
vLLM Thinkers are tensor-parallel on two cards at --gpu-mem 0.3 and declare 56000 MiB a card.

    python3 reports/uniss_si_v4/make_tier_jobs.py reports/uniss_si_v4/jobs/tier_jobs.json
"""
import json
import subprocess
import sys

O = "eval_outputs/uniss_qwen3_omni_v1"
V4 = f"{O}/si_v4"
M = {"s1": f"{O}/thinker_op_v3s1_20261025T143042Z/thinker_merged", "s2": f"{O}/thinker_op_v3s2_20261025T143042Z/thinker_merged"}
RATE = {"en2zh": "0.191", "zh2en": "0.339"}
DEVW = "0.5,0.75,1,1.25,1.5,2,2.5,3"
DEVS = {"un": ("data/si_thinker_oracle_prefix_v1/dev_items_{d}.jsonl", "dev_{m}"),
        "sp": ("data/si_spont_v1/dev_{d}_ref.jsonl", "devsp_{m}")}
ST = ("CUDA_VISIBLE_DEVICES=${GPUS// /,} PATH=/opt/dlami/nvme/jasonleeeli/venvs/vllm_omni_030/bin:$PATH OMP_NUM_THREADS=16 "
      "VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_CACHE_ROOT=/opt/dlami/nvme/jasonleeeli/cache/vllm_omni_030 PYTHONPATH=.:.tmp/pylib_jieba "
      "/opt/dlami/nvme/jasonleeeli/venvs/vllm_omni_030/bin/python -m experiments.uniss_qwen3_omni_v1.evaluation.si_thinker_stream "
      "--tp 2 --gpu-mem 0.3")
jobs = []
for m in ("s1", "s2"):
    for d in ("zh2en", "en2zh"):
        for dev, (items, out) in DEVS.items():
            jobs.append({"id": f"dev_{m}_{dev}_{d}", "gpus": 2, "vllm": True, "mem": 56000, "deps": [],
                         "cmd": f"{ST} --policy selfwait --thinker {M[m]} --thinker-tag {m} --wait-biases=1 "
                                f"--sw-queue-rate {RATE[d]} --direction {d} --windows {DEVW} --items-file {items.format(d=d)} "
                                f"--out {V4}/{out.format(m=m)}"})
for w in ("0.2", "0.3", "0.4"):
    jobs.append({"id": f"acl_la2low_{w}", "gpus": 2, "vllm": True, "mem": 56000, "deps": [],
                 "cmd": f"{ST} --direction en2zh --items-file data/acl6060_dev_v1/items.jsonl --out {O}/acl6060_dev_v1_la2low "
                        f"--windows {w}"})
spec = [{"kind": "realsi", "name": "la2low" + ("-zh" if d == "zh2en" else ""), "dir": d, "window": w,
         "blocks_dir": f"{V4}/realsi_la2low", "suffix": "", "seeds": [0, 1, 2]}
        for d in ("en2zh", "zh2en") for w in (0.2, 0.3, 0.4)]
with open(sys.argv[1].replace(".json", "_speech_spec.json"), "x") as fh:
    json.dump(spec, fh, indent=1)
sp = sys.argv[1].replace(".json", "_speech.json")
subprocess.run([sys.executable, "reports/uniss_si_v4/make_speech_jobs.py", sp, sys.argv[1].replace(".json", "_speech_spec.json")],
               check=True)
jobs += json.load(open(sp))
with open(sys.argv[1], "x") as fh:
    json.dump(jobs, fh, indent=1)
print(len(jobs), "jobs")

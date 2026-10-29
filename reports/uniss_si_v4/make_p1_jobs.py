"""gpuq7 jobs for candidate A2 of the single-speaker paper v4 (docs/prereg_v4.zh-CN.md, amendment 1):
  mt_*     silver references of the spontaneous dev sets (Hy-MT2-30B-A3B)
  dump_*   oracle labels on the spontaneous training audio (data/si_spont_v1), exactly as v1's UniST dumps: the TTB head
           policy (head_cons_all, tau 0.5, agreement weight 4, a virtual draft 500 ms back) with --oracle --dump, read
           windows 0.5/1/2/4 s
  devsp_*  LA2 and A0 (s0+SDR) on the spontaneous dev, for the selection rule
  rows     UniST + spontaneous oracle-prefix rows (build_thinker_oracle_prefix, v1's defaults), then READY for
           run_v3_eval4.sh, which trains A2 when the multi-speaker S2-dl ends
Each job waits for build_spont_items to finish.

    python3 reports/uniss_si_v4/make_p1_jobs.py reports/uniss_si_v4/jobs/p1_jobs.json
"""
import json, sys

O = "eval_outputs/uniss_qwen3_omni_v1"
V4 = f"{O}/si_v4"
SP = "data/si_spont_v1"
ROWS = "data/si_thinker_oracle_prefix_v7_spont"
S0 = f"{O}/thinker_op_v3_20261023T213631Z/thinker_merged_i200"
RATE = {"en2zh": "0.191", "zh2en": "0.339"}
TGT = {"en2zh": "zh", "zh2en": "en"}
DEVW = "0.5,0.75,1,1.25,1.5,2,2.5,3"
VENV = ("PATH=/opt/dlami/nvme/jasonleeeli/venvs/vllm_omni_030/bin:$PATH OMP_NUM_THREADS=16 VLLM_USE_FLASHINFER_SAMPLER=0 "
        "VLLM_CACHE_ROOT=/opt/dlami/nvme/jasonleeeli/cache/vllm_omni_030 PYTHONPATH=.:.tmp/pylib_jieba")
PYV = "/opt/dlami/nvme/jasonleeeli/venvs/vllm_omni_030/bin/python"
PYE = "/opt/dlami/nvme/jasonleeeli/conda_envs/uniss-eval/bin/python"
WAIT = "until grep -q SPONT_DONE logs/si_v4/spont_items.log; do sleep 30; done; "
ST = f"CUDA_VISIBLE_DEVICES=${{GPUS// /,}} {VENV} {PYV} -m experiments.uniss_qwen3_omni_v1.evaluation.si_thinker_stream --tp 2 --gpu-mem 0.3"

jobs = []
for d in ("en2zh", "zh2en"):
    jobs.append({"id": f"mt_{d}", "gpus": 1, "vllm": True, "mem": 46000, "deps": [],
                 "cmd": WAIT + f"CUDA_VISIBLE_DEVICES=$GPU0 {VENV} {PYV} -m experiments.uniss_qwen3_omni_v1.data_clean.translate_dev_refs "
                        f"--items {SP}/dev_{d}.jsonl --tgt {TGT[d]} --out {SP}/dev_{d}_ref.jsonl"})
    for w in ("0.5", "1", "2", "4"):
        out = f"{SP}/dump_{d}_w{w}"
        jobs.append({"id": f"dump_{d}_w{w}", "gpus": 2, "vllm": True, "mem": 46000, "deps": [],
                     "cmd": WAIT + f"{ST} --policy head --head data/unist_simul_thinker_v1/p1/heads/head_cons_all.pt --taus 0.5 "
                            f"--agree-beta 4 --virtual-ms 500 --oracle --dump {out}/dumps --items-file {SP}/items_{d}.jsonl "
                            f"--direction {d} --windows {w} --out {out}"})
    jobs.append({"id": f"devsp_la2_{d}", "gpus": 2, "vllm": True, "mem": 46000, "deps": [f"mt_{d}"],
                 "cmd": f"{ST} --policy la2 --direction {d} --windows {DEVW} --out {V4}/devsp_la2 --items-file {SP}/dev_{d}_ref.jsonl"})
    jobs.append({"id": f"devsp_s0_{d}", "gpus": 2, "vllm": True, "mem": 46000, "deps": [f"mt_{d}"],
                 "cmd": f"{ST} --policy selfwait --thinker {S0} --thinker-tag s0 --wait-biases=1 --sw-queue-rate {RATE[d]} "
                        f"--direction {d} --windows {DEVW} --out {V4}/devsp_s0 --items-file {SP}/dev_{d}_ref.jsonl"})
items_all = (f"set -C; for d in en2zh zh2en; do cat data/unist_simul_thinker_v1/items_$d.jsonl {SP}/items_$d.jsonl > {SP}/items_all_$d.jsonl; done")
dumps = "data/si_bayes_lr_v1/dump_*/dumps/*.jsonl,data/si_belief_v1/shard_*/dumps/*.jsonl," + f"{SP}/dump_*/dumps/*.jsonl"
jobs.append({"id": "rows", "gpus": 1, "vllm": False, "mem": 0, "deps": [j["id"] for j in jobs if j["id"].startswith("dump_")],
             "cmd": f"{items_all} && PYTHONPATH=. {PYE} -m experiments.uniss_qwen3_omni_v1.data_clean.build_thinker_oracle_prefix "
                    f"--dumps '{dumps}' --items '{SP}/items_all_{{d}}.jsonl' --out {ROWS} && touch {ROWS}/READY"})
order = ["dump_", "mt_", "devsp_", "rows"]
jobs.sort(key=lambda j: next(i for i, p in enumerate(order) if j["id"].startswith(p)))
with open(sys.argv[1], "x") as fh:
    json.dump(jobs, fh, indent=1)
print(len(jobs), "jobs")

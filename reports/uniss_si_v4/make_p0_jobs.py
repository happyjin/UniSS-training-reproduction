"""gpuq7 jobs for the single-speaker paper v4, step P0 (text side; paper/bayes_simul_s2st_v4/docs/prereg_v4.zh-CN.md):
  off_*   whole-sentence (60 s window) translation on RealSI by the ABC adapters s0/s1/s2 and their weight average
          (bma3): does the adapter change what is said?
  grid_*  ABC (b0 = 1) with and without SDR on LA2's 15 read windows (s0: the 8 windows v3 did not run)
  dev_*   the dev-only selection of the posterior: LA2, s0+SDR and bma3+SDR on the 805 dev items
  cvss_*  whole-sentence translation of CVSS-T test (4,897 a direction) by the base Thinker and by s0
Every Thinker runs tensor-parallel on two cards at --gpu-mem 0.3 (about 43 GB a card), beside other work.

    python3 reports/uniss_si_v4/make_p0_jobs.py reports/uniss_si_v4/jobs/p0_jobs.json
"""
import json, sys

O = "eval_outputs/uniss_qwen3_omni_v1"
V4 = f"{O}/si_v4"
M = {"base": "", "s0": f"{O}/thinker_op_v3_20261023T213631Z/thinker_merged_i200",
     "s1": f"{O}/thinker_op_v3s1_20261025T143042Z/thinker_merged", "s2": f"{O}/thinker_op_v3s2_20261025T143042Z/thinker_merged",
     "bma3": f"{O}/thinker_op_v3_bma3/thinker_merged"}
RATE = {"en2zh": "0.191", "zh2en": "0.339"}
GRID = [0.5, 0.6, 0.75, 0.9, 1, 1.1, 1.25, 1.5, 1.75, 2, 2.5, 3, 4, 5, 6]
V3_DONE = [0.5, 1, 2, 3, 4, 5, 6]
DEVW = [0.5, 0.75, 1, 1.25, 1.5, 2, 2.5, 3]
DEV = "data/si_thinker_oracle_prefix_v1/dev_items_{d}.jsonl"
CVSS = "data/cvss_t/si_items_v1/items_{d}.jsonl"
PY = ("CUDA_VISIBLE_DEVICES=${GPUS// /,} OMP_NUM_THREADS=16 VLLM_USE_FLASHINFER_SAMPLER=0 "
      "VLLM_CACHE_ROOT=/opt/dlami/nvme/jasonleeeli/cache/vllm_omni_030 PYTHONPATH=.:.tmp/pylib_jieba "
      "/opt/dlami/nvme/jasonleeeli/venvs/vllm_omni_030/bin/python -m experiments.uniss_qwen3_omni_v1.evaluation.si_thinker_stream "
      "--tp 2 --gpu-mem 0.3")
WAIT_BMA = "until grep -q AVERAGE_DONE logs/si_v4/average_bma3.log; do sleep 30; done; "


def ws(xs):
    return ",".join(f"{w:g}" for w in xs)


def job(jid, model, d, windows, out, bias="1", sdr=False, items="", policy=None):
    m = M[model]
    pol = policy or ("la2" if model == "base" else "selfwait")
    args = f"--policy {pol} --direction {d} --windows {windows} --out {V4}/{out}"
    if m:
        args += f" --thinker {m} --thinker-tag {model} --wait-biases={bias}"
    if sdr:
        args += f" --sw-queue-rate {RATE[d]}"
    if items:
        args += f" --items-file {items.format(d=d)}"
    return {"id": jid, "gpus": 2, "vllm": True, "mem": 46000, "deps": [],
            "cmd": (WAIT_BMA if model == "bma3" else "") + f"{PY} {args}"}


jobs = []
for d in ("en2zh", "zh2en"):
    for m in ("s0", "s1", "s2", "bma3"):
        jobs.append(job(f"off_{m}_{d}", m, d, "60", f"realsi_off_{m}", bias="0"))
    for m, grid in (("s0", [w for w in GRID if w not in V3_DONE]), ("bma3", GRID)):
        for sdr in (True, False):
            jobs.append(job(f"grid_{m}_{'q' if sdr else 'noq'}_{d}", m, d, ws(grid), f"realsi_grid_{m}", sdr=sdr))
    jobs.append(job(f"dev_la2_{d}", "base", d, ws(DEVW), "dev_la2", items=DEV))
    for m in ("s0", "bma3"):
        jobs.append(job(f"dev_{m}_q_{d}", m, d, ws(DEVW), f"dev_{m}", sdr=True, items=DEV))
    jobs.append(job(f"cvss_base_{d}", "base", d, "60", "cvss_off_base", items=CVSS))
    jobs.append(job(f"cvss_s0_{d}", "s0", d, "60", "cvss_off_s0", bias="0", items=CVSS))
# cheap and decisive first: dev selection and the offline checks, then the grids
order = ["dev_", "off_", "cvss_", "grid_"]
jobs.sort(key=lambda j: next(i for i, p in enumerate(order) if j["id"].startswith(p)))
with open(sys.argv[1], "x") as fh:
    json.dump(jobs, fh, indent=1)
print(len(jobs), "jobs")

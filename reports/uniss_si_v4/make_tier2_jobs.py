"""Tier table of paper v7, part 2: the adapter seed chosen on dev is s1 (paper/bayes_simul_s2st_v7/generated/
seed_selection.json), which so far has RealSI speech only at b0 = 1 and windows 0.5 / 1 / 2. Two steps, cards 0-3:
  text    QABC with adapter s1 on RealSI, LA2's 15 read windows, b0 in {-1, 0, 1, 2}, queue-aware threshold (SDR),
          one Thinker per direction into si_v4/realsi_grid_s1 (tensor-parallel on two cards, gpu-mem 0.3, 56000 MiB)
  speech  the Talker speaks the points that mirror s0's QABC speech grid -- b0 = -1 at 0.5 s, b0 = 0 at 0.5 / 1 s,
          b0 = 1 at every window not yet spoken, b0 = 2 at 3-6 s -- with talker seed 0 only (to save compute; the
          tier table uses talker seed 0 for every policy), named v7s1-q-b<b> / v7s1-q-b<b>-zh
    python3 reports/uniss_si_v4/make_tier2_jobs.py text   reports/uniss_si_v4/jobs/tier2_text_jobs.json
    python3 reports/uniss_si_v4/make_tier2_jobs.py speech reports/uniss_si_v4/jobs/tier2_speech_jobs.json
"""
import json
import subprocess
import sys

O = "eval_outputs/uniss_qwen3_omni_v1"
V4 = f"{O}/si_v4"
S1 = f"{O}/thinker_op_v3s1_20261025T143042Z/thinker_merged"
RATE = {"en2zh": "0.191", "zh2en": "0.339"}
GRID = [0.5, 0.6, 0.75, 0.9, 1, 1.1, 1.25, 1.5, 1.75, 2, 2.5, 3, 4, 5, 6]
ST = ("CUDA_VISIBLE_DEVICES=${GPUS// /,} PATH=/opt/dlami/nvme/jasonleeeli/venvs/vllm_omni_030/bin:$PATH OMP_NUM_THREADS=16 "
      "VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_CACHE_ROOT=/opt/dlami/nvme/jasonleeeli/cache/vllm_omni_030 PYTHONPATH=.:.tmp/pylib_jieba "
      "/opt/dlami/nvme/jasonleeeli/venvs/vllm_omni_030/bin/python -m experiments.uniss_qwen3_omni_v1.evaluation.si_thinker_stream "
      "--tp 2 --gpu-mem 0.3")
POINTS = {-1: [0.5], 0: [0.5, 1], 1: [w for w in GRID if w not in (0.5, 1, 2)], 2: [3, 4, 5, 6]}

kind, path = sys.argv[1], sys.argv[2]
if kind == "text":
    jobs = [{"id": f"grid_s1_{d}", "gpus": 2, "vllm": True, "mem": 56000, "deps": [],
             "cmd": f"{ST} --policy selfwait --thinker {S1} --thinker-tag s1 --wait-biases=-1,0,1,2 --sw-queue-rate {RATE[d]} "
                    f"--direction {d} --windows {','.join(f'{w:g}' for w in GRID)} --out {V4}/realsi_grid_s1"}
            for d in ("zh2en", "en2zh")]
    with open(path, "x") as fh:
        json.dump(jobs, fh, indent=1)
    print(len(jobs), "jobs")
else:
    spec = [{"kind": "realsi", "name": f"v7s1-q-b{b:g}".replace("b-1", "bm1") + ("-zh" if d == "zh2en" else ""), "dir": d,
             "window": w, "blocks_dir": f"{V4}/realsi_grid_s1", "suffix": f"_sws1b{b:g}q{RATE[d]}", "seeds": [0]}
            for d in ("en2zh", "zh2en") for b, ws in POINTS.items() for w in ws]
    sp = path.replace(".json", "_spec.json")
    with open(sp, "x") as fh:
        json.dump(spec, fh, indent=1)
    subprocess.run([sys.executable, "reports/uniss_si_v4/make_speech_jobs.py", path, sp], check=True)

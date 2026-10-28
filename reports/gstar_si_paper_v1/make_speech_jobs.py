"""Speech for the G-STAR-SI paper (docs/multi_speaker_si_results_v1.zh-CN.md §7.4): every test window (2,375) of five
systems spoken by the UniSS talker, one stream per window and speaker slot, then ASR, stitching and cp-ASR-BLEU.
The demo did this on 75 windows; here the whole test set, 8 talker shards per system on the gpuq6 queue.

    python reports/gstar_si_paper_v1/make_speech_jobs.py > reports/gstar_si_paper_v1/jobs/speech_jobs.json
"""
import json

EV = "eval_outputs/uniss_qwen3_omni_v1"
OFF = f"{EV}/gstar_offline_eval_o1_20261026T195403Z"
STR = f"{EV}/gstar_stream_eval_o1_20261026T195403Z"
OUTROOT = f"{EV}/gstar_speech_full_20261028"
DATA = "data/ami_multispk_si_v1/e2e_gstar_v1"
SDATA = "data/ami_multispk_si_v1/e2e_gstar_stream_v1"
PYE = "/opt/dlami/nvme/jasonleeeli/conda_envs/uniss-eval/bin/python"
PYV = "/opt/dlami/nvme/jasonleeeli/venvs/vllm_omni_030/bin/python"
T1C = f"{EV}/si_t1c_20261021T181440Z/autoext/export_1500/uniss"
SCR = "/tmp/claude-0/-opt-dlami-nvme-jasonleeeli-projects-UniSS/d490b1cc-23c2-464f-a450-9958e1eda107/scratchpad"
N = 8

SYSTEMS = [  # name, hypotheses, windows, mode, posterior preset (voices and Nemotron DER)
    ("O-G1", f"{OFF}/O-G1", f"{DATA}/windows_test_ref.jsonl", "offline", "offline"),
    ("B-off", f"{OFF}/B-off", f"{DATA}/windows_test_ref.jsonl", "offline", "offline"),
    ("S-T_d1", f"{STR}/S-T_d1", f"{SDATA}/windows_test_ref.jsonl", "stream", "lowlat"),
    ("S-LA2_d1", f"{STR}/S-LA2_d1", f"{SDATA}/windows_test_ref.jsonl", "stream", "lowlat"),
    ("B-LA2_d1", f"{STR}/B-LA2_d1", f"{SDATA}/windows_test_ref.jsonl", "stream", "lowlat"),
]

jobs = []
for name, sys_dir, win, mode, pk in SYSTEMS:
    out = f"{OUTROOT}/{name}"
    for k in range(N):
        jobs.append({"id": f"{name}_talk{k}", "gpus": 1, "vllm": True, "deps": [],
                     "cmd": f"CUDA_VISIBLE_DEVICES=$GPU0 VLLM_USE_FLASHINFER_SAMPLER=0 "
                            f"VLLM_CACHE_ROOT=/opt/dlami/nvme/jasonleeeli/cache/vllm_omni_030 PYTHONPATH=. "
                            f"OMP_NUM_THREADS=16 {PYV} -m experiments.uniss_qwen3_omni_v1.evaluation.si_infer "
                            f"--model {T1C} --direction en2zh --window 1 --output {out}/talker --text-policy file "
                            f"--blocks-file {out}/talker_in/blocks.jsonl --items-file {out}/talker_in/items.jsonl "
                            f"--speed-policy fixed --history-ban 8 --max-model-len 16384 --gpu-mem 0.3 "
                            f"--shard {k} --num-shards {N}"})
    jobs.append({"id": f"{name}_post", "gpus": 1, "vllm": False, "share": True, "need_mib": 8000,
                 "deps": [f"{name}_talk{k}" for k in range(N)],
                 "cmd": f"set -e; cat {out}/talker/part_*.jsonl > {out}/talker/results.jsonl; "
                        f"{PYE} {SCR}/p80_speechonly.py {out}/talker; "
                        f"CUDA_VISIBLE_DEVICES=$GPU0 PYTHONPATH=. {PYE} -m evaluation.asr_transcribe "
                        f"--input {out}/talker/results_speechonly.jsonl --output {out}/talker/asr_speech.jsonl "
                        f"--device cuda:0 --batch-size 16 > {out}/asr.log 2>&1; "
                        f"PYTHONPATH=. {PYE} -m experiments.uniss_qwen3_omni_v1.evaluation.gstar_talker stitch "
                        f"--results {out}/talker/results.jsonl --prep {out}/talker_in --windows {win} --mode {mode} "
                        f"--out {out}/stitched; "
                        f"PYTHONPATH=. {PYE} -m experiments.uniss_qwen3_omni_v1.evaluation.gstar_score --windows {win} "
                        f"--hyp {sys_dir}/hyp.jsonl --asr {out}/talker/asr_speech.jsonl --post-key {pk} "
                        f"--out {out}/score.json > {out}/score.log 2>&1"})
print(json.dumps(jobs, indent=1))

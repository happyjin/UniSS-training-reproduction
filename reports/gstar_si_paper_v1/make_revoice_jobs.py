"""Jobs for the one-voice-per-speaker Chinese AMI speech (build_ami_zh_revoice.py): 8 UniSS talker shards, then the
speech-only list and 8 Paraformer shards (the CER gate of plan v9 §11.5 on every new clip).

    python reports/gstar_si_paper_v1/make_revoice_jobs.py > reports/gstar_si_paper_v1/jobs/revoice_jobs.json
"""
import json

R = "data/ami_multispk_si_v1/zh_revoice_v1"
PYE = "/opt/dlami/nvme/jasonleeeli/conda_envs/uniss-eval/bin/python"
PYV = "/opt/dlami/nvme/jasonleeeli/venvs/vllm_omni_030/bin/python"
T1C = "eval_outputs/uniss_qwen3_omni_v1/si_t1c_20261021T181440Z/autoext/export_1500/uniss"
N = 8
jobs = [{"id": f"talk{k}", "gpus": 1, "vllm": True, "deps": [],
         "cmd": f"CUDA_VISIBLE_DEVICES=$GPU0 VLLM_USE_FLASHINFER_SAMPLER=0 "
                f"VLLM_CACHE_ROOT=/opt/dlami/nvme/jasonleeeli/cache/vllm_omni_030 PYTHONPATH=. OMP_NUM_THREADS=16 "
                f"{PYV} -m experiments.uniss_qwen3_omni_v1.evaluation.si_infer --model {T1C} --direction en2zh "
                f"--window 1 --output {R}/talker --text-policy file --blocks-file {R}/blocks.jsonl "
                f"--items-file {R}/items.jsonl --speed-policy fixed --history-ban 8 --max-model-len 16384 "
                f"--gpu-mem 0.8 --shard {k} --num-shards {N}"} for k in range(N)]
jobs.append({"id": "speechonly", "gpus": 1, "vllm": False, "share": True, "need_mib": 0,
             "deps": [f"talk{k}" for k in range(N)],
             "cmd": f"set -e; cat {R}/talker/part_*.jsonl > {R}/talker/results.jsonl; "
                    f"{PYE} scripts/gpu_queue/p80_speechonly.py {R}/talker"})
jobs += [{"id": f"asr{k}", "gpus": 1, "vllm": False, "deps": ["speechonly"],
          "cmd": f"CUDA_VISIBLE_DEVICES=$GPU0 PYTHONPATH=. {PYE} -m evaluation.asr_transcribe "
                 f"--input {R}/talker/results_speechonly.jsonl --output {R}/talker/asr_s{k}.jsonl --num-shards {N} "
                 f"--shard-index {k} --device cuda:0 --batch-size 32"} for k in range(N)]
print(json.dumps(jobs, indent=1))

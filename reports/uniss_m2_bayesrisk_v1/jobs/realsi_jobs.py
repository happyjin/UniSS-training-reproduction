"""Build a gpuq5 job list for RealSI bayesrisk systems: Thinker blocks -> Talker (SQ) -> speech-only ASR.
usage: python3 realsi_jobs.py <out_jobs.json> '<json list of [name, direction, window, bias, extra_args, suffix]>'"""
import json, os, sys
S = "/tmp/claude-0/-opt-dlami-nvme-jasonleeeli-projects-UniSS/88583ca0-585a-4fad-ada7-4b193b743b50/scratchpad"
PY = ("CUDA_VISIBLE_DEVICES=$GPU0 OMP_NUM_THREADS=16 VLLM_USE_FLASHINFER_SAMPLER=0 "
      "VLLM_CACHE_ROOT=/opt/dlami/nvme/jasonleeeli/cache/vllm_omni_030 PYTHONPATH=.:.tmp/pylib_jieba "
      "/opt/dlami/nvme/jasonleeeli/venvs/vllm_omni_030/bin/python -m experiments.uniss_qwen3_omni_v1.evaluation.si_thinker_stream "
      "--thinker eval_outputs/uniss_qwen3_omni_v1/thinker_op_v3_20261023T213631Z/thinker_merged_i200 --thinker-tag op1 "
      "--policy bayesrisk --clean")
TB = "eval_outputs/uniss_qwen3_omni_v1/bayesrisk_v1/realsi"
E = "eval_outputs/uniss_qwen3_omni_v1/si_t1c_20261021T181440Z/full_iter0001500"
EXP = "eval_outputs/uniss_qwen3_omni_v1/si_t1c_20261021T181440Z/autoext/export_1500"
EV = "/opt/dlami/nvme/jasonleeeli/conda_envs/uniss-eval/bin/python"
jobs = []
for name, d, w, b, extra, suffix in json.loads(sys.argv[2]):
    out = f"{E}/system_{name}_m{w}"
    assert not os.path.exists(out), out
    bf = f"{TB}/blocks_{d}_w{w}{suffix}.jsonl"
    assert not os.path.exists(bf), bf
    tx = f"tx_{name}"
    jobs.append({"id": tx, "gpus": 1, "vllm": True, "deps": [],
                 "cmd": f"{PY} {extra} --direction {d} --windows {w} --wait-biases={b} --out {TB} && [ -f {bf} ]"})
    jobs.append({"id": f"sp_{name}", "gpus": 2, "vllm": True, "deps": [tx],
                 "cmd": f"env EXPORT_DIR={EXP} LIMIT=0 SPEED_POLICY=queue HISTORY_BAN=8 EXTRA=0 TEXT_POLICY=file BLOCKS_DIR={TB} "
                        f"GPUS=\"$GPUS\" OUT={out} WINDOW={w} BLOCKS_SUFFIX={suffix} DIRS={d} bash experiments/uniss_qwen3_omni_v1/scripts/run_si_probe.sh "
                        f"&& [ -f {out}/realsi_{d}/results.jsonl ]"})
    jobs.append({"id": f"asr_{name}", "gpus": 1, "vllm": False, "deps": [f"sp_{name}"],
                 "cmd": f"{EV} {S}/p8_prep.py {out} > /dev/null && CUDA_VISIBLE_DEVICES=$GPU0 PYTHONPATH=. {EV} -m evaluation.asr_transcribe "
                        f"--input {out}/realsi_{d}/results_speechonly.jsonl --output {out}/realsi_{d}/asr_speech.jsonl --device cuda:0 --batch-size 16"})
with open(sys.argv[1], "x") as fh:
    json.dump(jobs, fh, indent=1)
print(len(jobs))

"""Paper v3 speech (plan v9 §6.3): the chosen systems of both directions spoken by the UniSS talker, then ASR,
cp-ASR-BLEU, voice similarity (gstar_voice), listening measures (gstar_listen) and UTMOS.

  offline systems: one voice prompt per meeting and slot from the session's solo frames, segments laid at their source
                   start (dubbing)
  streaming:       causal voice prompts (solo frames before the slot's first commit), each read's block spoken at its
                   commit time (gstar_talker prep --blocks), laid on one channel

    python reports/gstar_si_paper_v1/make_v3_speech_jobs.py <systems.json> > <jobs.json>
    systems.json: [{"name": "zh2en_O2", "dir": "zh2en", "hyp": "<dir with hyp.jsonl>", "mode": "offline" | "stream"}, ...]
"""
import json
import sys

D = "data/ami_multispk_si_v1"
EV = "eval_outputs/uniss_qwen3_omni_v1"
OUT = f"{EV}/gstar_v3_speech_20261029"
PYE = "/opt/dlami/nvme/jasonleeeli/conda_envs/uniss-eval/bin/python"
PYV = "/opt/dlami/nvme/jasonleeeli/venvs/vllm_omni_030/bin/python"
T1C = f"{EV}/si_t1c_20261021T181440Z/autoext/export_1500/uniss"
M = "experiments.uniss_qwen3_omni_v1.evaluation"
N = 8
WIN = {("en2zh", "offline"): f"{D}/e2e_gstar_v1/windows_test_ref.jsonl",
       ("en2zh", "stream"): f"{D}/e2e_gstar_stream_v1/windows_test_ref.jsonl",
       ("zh2en", "offline"): f"{D}/e2e_gstar_zh2en_v1/windows_test_ref.jsonl",
       ("zh2en", "stream"): f"{D}/e2e_gstar_zh2en_v1/windows_test_ref.jsonl"}
SESS = {"en2zh": "--session ihm_mix", "zh2en": "--session zh_mix_v8"}
DIAR = {("en2zh", "offline"): "", ("en2zh", "stream"): "",
        ("zh2en", "offline"): "--diar nemotron_offline_zh8", ("zh2en", "stream"): "--diar nemotron_lowlat_zh8"}
VOICES = {"en2zh": f"{D}/e2e_gstar_v1/voices_offline", "zh2en": f"{D}/e2e_gstar_zh2en_v1/voices_offline"}

jobs = [{"id": "voices_zh2en", "gpus": 1, "vllm": False, "share": True, "need_mib": 0, "deps": [],
                 "cmd": f"PYTHONPATH=. {PYE} -m {M}.gstar_talker voices --windows {WIN[('zh2en', 'offline')]} "
                        f"--preset offline {SESS['zh2en']} {DIAR[('zh2en', 'offline')]} --out {VOICES['zh2en']}"}]
for s in json.load(open(sys.argv[1])):
    name, dr, mode, hyp = s["name"], s["dir"], s["mode"], s["hyp"]
    win, out = WIN[(dr, mode)], f"{OUT}/{name}"
    sep = "--sep ' '" if dr == "zh2en" else ""
    prep = (f"--voices {VOICES[dr]}" if mode == "offline" else
            f"--causal --blocks --preset lowlat {SESS[dr]} {DIAR[(dr, mode)]}")
    jobs.append({"id": f"{name}_prep", "gpus": 1, "vllm": False, "share": True, "need_mib": 0,
                 "deps": ["voices_zh2en"] if dr == "zh2en" and mode == "offline" else [],
                 "cmd": f"mkdir -p {out} && PYTHONPATH=. {PYE} -m {M}.gstar_talker prep --hyp {hyp}/hyp.jsonl "
                        f"--windows {win} {prep} {sep} --out {out}/talker_in"})
    for k in range(N):
        jobs.append({"id": f"{name}_talk{k}", "gpus": 1, "vllm": True, "deps": [f"{name}_prep"],
                     "cmd": f"CUDA_VISIBLE_DEVICES=$GPU0 VLLM_USE_FLASHINFER_SAMPLER=0 "
                            f"VLLM_CACHE_ROOT=/opt/dlami/nvme/jasonleeeli/cache/vllm_omni_030 PYTHONPATH=. "
                            f"OMP_NUM_THREADS=16 {PYV} -m {M}.si_infer --model {T1C} --direction {dr} --window 1 "
                            f"--output {out}/talker --text-policy file --blocks-file {out}/talker_in/blocks.jsonl "
                            f"--items-file {out}/talker_in/items.jsonl --speed-policy fixed --history-ban 8 "
                            f"--max-model-len 32768 --gpu-mem 0.3 --shard {k} --num-shards {N}"})
    pk = "offline" if mode == "offline" else "lowlat"
    jobs.append({"id": f"{name}_post", "gpus": 1, "vllm": False, "share": True, "need_mib": 8000,
                 "deps": [f"{name}_talk{k}" for k in range(N)],
                 "cmd": f"set -e; cat {out}/talker/part_*.jsonl > {out}/talker/results.jsonl; "
                        f"{PYE} scripts/gpu_queue/p80_speechonly.py {out}/talker; "
                        f"CUDA_VISIBLE_DEVICES=$GPU0 PYTHONPATH=. {PYE} -m evaluation.asr_transcribe "
                        f"--input {out}/talker/results_speechonly.jsonl --output {out}/talker/asr_speech.jsonl "
                        f"--device cuda:0 --batch-size 16 > {out}/asr.log 2>&1; "
                        f"PYTHONPATH=. {PYE} -m {M}.gstar_talker stitch --results {out}/talker/results.jsonl "
                        f"--prep {out}/talker_in --windows {win} --mode {mode} --out {out}/stitched; "
                        f"PYTHONPATH=. {PYE} -m {M}.gstar_score --windows {win} --hyp {hyp}/hyp.jsonl "
                        f"--asr {out}/talker/asr_speech.jsonl --post-key {pk} --out {out}/score.json > {out}/score.log 2>&1; "
                        f"PYTHONPATH=. {PYE} -m {M}.gstar_listen --transcript {out}/stitched/transcript.jsonl "
                        f"--out {out}/listen.json; "
                        f"CUDA_VISIBLE_DEVICES=$GPU0 PYTHONPATH=. {PYE} -m {M}.gstar_voice --windows {win} "
                        f"--hyp {hyp}/hyp.jsonl --results {out}/talker/results.jsonl --out {out}/voice.json; "
                        f"CUDA_VISIBLE_DEVICES=$GPU0 PYTHONPATH=. {PYE} -m evaluation.utmos_metrics --input "
                        f"{out}/talker/results_speechonly.jsonl --output-dir {out}/utmos --device cuda:0 "
                        f"> {out}/utmos.log 2>&1 || true"})
print(json.dumps(jobs, indent=1))

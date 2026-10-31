"""gpuq7 jobs for paper v6 speech (review M03 listener-side latency, M07 AliMeeting speech), as
reports/gstar_si_paper_v1/make_v4_speech_jobs.py: the UniSS talker T1c@1500 speaks every output speaker in a voice
prompted from the diarizer's solo frames (offline: whole session; streaming: causal, before the slot's first commit,
each read's block at its commit time), then the speech-only ASR. Outputs go to
eval_outputs/.../gstar_v6_speech_20261031/<name>/ (new directories only). AliMeeting uses the v6 talker copy with
GSTAR_DATA=data/alimeeting_si_v1. Scoring is done afterwards (gstar_score --asr for reference windows,
listener_latency.py / meeting_score.py for online windows).

    python3 reports/gstar_si_v6/make_v6_speech_jobs.py <systems.json> > <jobs.json>
    systems.json: [{"name", "set": "en2zh" | "zh2en" | "ali", "hyp": <dir with hyp.jsonl>, "windows": <file>,
                    "mode": "offline" | "stream"}, ...]
"""
import json
import sys

EV = "eval_outputs/uniss_qwen3_omni_v1"
OUT = f"{EV}/gstar_v6_speech_20261031"
PYE = "/opt/dlami/nvme/jasonleeeli/conda_envs/uniss-eval/bin/python"
PYV = "/opt/dlami/nvme/jasonleeeli/venvs/vllm_omni_030/bin/python"
VENV = "PATH=/opt/dlami/nvme/jasonleeeli/venvs/vllm_omni_030/bin:$PATH"
T1C = f"{EV}/si_t1c_20261021T181440Z/autoext/export_1500/uniss"
M = "experiments.uniss_qwen3_omni_v1.evaluation"
N = 8
DATA = {"en2zh": "data/ami_multispk_si_v1", "zh2en": "data/ami_multispk_si_v1", "ali": "data/alimeeting_si_v1"}
SESS = {"en2zh": "--session ihm_mix", "zh2en": "--session zh_mix_v8", "ali": "--session near_mix"}
DIAR = {("en2zh", "offline"): "", ("en2zh", "stream"): "",
        ("zh2en", "offline"): "--diar nemotron_offline_zh8", ("zh2en", "stream"): "--diar nemotron_lowlat_zh8",
        ("ali", "offline"): "--diar nemotron_offline", ("ali", "stream"): "--diar nemotron_lowlat"}
VOICES = {"en2zh": "data/ami_multispk_si_v1/e2e_gstar_v1/voices_offline",
          "zh2en": "data/ami_multispk_si_v1/e2e_gstar_zh2en_v1/voices_offline",
          "ali": "data/alimeeting_si_v1/voices_offline_v6"}
DIRN = {"en2zh": "en2zh", "zh2en": "zh2en", "ali": "zh2en"}

jobs = []
systems = json.load(open(sys.argv[1]))
if any(s["set"] == "ali" and s["mode"] == "offline" for s in systems):
    jobs.append({"id": "voices_ali_v6", "gpus": 1, "vllm": False, "deps": [], "mem": 1000,
                 "cmd": f"GSTAR_DATA=data/alimeeting_si_v1 PYTHONPATH=. CUDA_VISIBLE_DEVICES= {PYE} -m {M}.gstar_talker_v6 "
                        f"voices --windows data/alimeeting_si_v1/windows_test_ref.jsonl --preset offline {SESS['ali']} "
                        f"{DIAR[('ali', 'offline')]} --out {VOICES['ali']}"})
for s in systems:
    name, st, mode, hyp, win = s["name"], s["set"], s["mode"], s["hyp"], s["windows"]
    out = f"{OUT}/{name}"
    sep = "--sep ' '" if st != "en2zh" else ""
    prep = (f"--voices {VOICES[st]}" if mode == "offline" else
            f"--causal --blocks --preset lowlat {SESS[st]} {DIAR[(st, mode)]}")
    jobs.append({"id": f"{name}_prep", "gpus": 1, "vllm": False, "mem": 1000,
                 "deps": ["voices_ali_v6"] if st == "ali" and mode == "offline" else [],
                 "cmd": f"mkdir -p {out} && GSTAR_DATA={DATA[st]} PYTHONPATH=. CUDA_VISIBLE_DEVICES= {PYE} -m "
                        f"{M}.gstar_talker_v6 prep --hyp {hyp}/hyp.jsonl --windows {win} {prep} {sep} --out {out}/talker_in"})
    for k in range(N):
        jobs.append({"id": f"{name}_talk{k}", "gpus": 1, "vllm": True, "deps": [f"{name}_prep"], "mem": 56000,
                     "cmd": f"CUDA_VISIBLE_DEVICES=$GPU0 {VENV} VLLM_USE_FLASHINFER_SAMPLER=0 "
                            f"VLLM_CACHE_ROOT=/opt/dlami/nvme/jasonleeeli/cache/vllm_omni_030 PYTHONPATH=. "
                            f"OMP_NUM_THREADS=16 {PYV} -m {M}.si_infer --model {T1C} --direction {DIRN[st]} --window 1 "
                            f"--output {out}/talker --text-policy file --blocks-file {out}/talker_in/blocks.jsonl "
                            f"--items-file {out}/talker_in/items.jsonl --speed-policy fixed --history-ban 8 "
                            f"--max-model-len 32768 --gpu-mem 0.3 --shard {k} --num-shards {N}"})
    jobs.append({"id": f"{name}_asr", "gpus": 1, "vllm": False, "mem": 22000,
                 "deps": [f"{name}_talk{k}" for k in range(N)],
                 "cmd": f"set -e; cat {out}/talker/part_*.jsonl > {out}/talker/results.jsonl; "
                        f"{PYE} scripts/gpu_queue/p80_speechonly.py {out}/talker; "
                        f"CUDA_VISIBLE_DEVICES=$GPU0 PYTHONPATH=. {PYE} -m evaluation.asr_transcribe "
                        f"--input {out}/talker/results_speechonly.jsonl --output {out}/talker/asr_speech.jsonl "
                        f"--device cuda:0 --batch-size 8 > {out}/asr.log 2>&1"})
print(json.dumps(jobs, indent=1))

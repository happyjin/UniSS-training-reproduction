"""Paper v2 experiments (the strict review of v1): two cascade baselines on the same 2,375 test windows, text and speech,
and the voice similarity of every system's speech.

  C-van  Nemotron-3 turns -> the original Qwen3-Omni Thinker translates each turn -> the turn's slot
  C-ft   Nemotron-3 turns -> the model trained on the same rows as the offline system, speaker stream zeroed (O-nosd)
  voice  gstar_voice.py on every system's talker streams (WavLM-SV cosine to the attributed speaker's clean voice)

    python reports/gstar_si_paper_v1/make_v2_jobs.py > reports/gstar_si_paper_v1/jobs/v2_jobs.json
"""
import json

EV = "eval_outputs/uniss_qwen3_omni_v1"
CAS = f"{EV}/gstar_cascade_20261028"
SPEECH = f"{EV}/gstar_speech_full_20261028"
DATA = "data/ami_multispk_si_v1/e2e_gstar_v1"
SDATA = "data/ami_multispk_si_v1/e2e_gstar_stream_v1"
PYE = "/opt/dlami/nvme/jasonleeeli/conda_envs/uniss-eval/bin/python"
PYM = "/opt/dlami/nvme/jasonleeeli/conda_envs/qwen3omni/bin/python"
PYV = "/opt/dlami/nvme/jasonleeeli/venvs/vllm_omni_030/bin/python"
T1C = f"{EV}/si_t1c_20261021T181440Z/autoext/export_1500/uniss"
SCR = "/tmp/claude-0/-opt-dlami-nvme-jasonleeeli-projects-UniSS/d490b1cc-23c2-464f-a450-9958e1eda107/scratchpad"
WIN = f"{DATA}/windows_test_ref.jsonl"
N = 8
M = "experiments.uniss_qwen3_omni_v1.evaluation"

CASCADES = [("C-van", "pretrained_models/Qwen3-Omni-30B-A3B-Instruct", "--inst sd"),
            ("C-ft", f"{EV}/thinker_gstar_o1nosd_20261026T195403Z/thinker_merged",
             "--inst gstar --post-key offline --zero-sd")]


def talker_jobs(name, out, deps):
    jobs = []
    for k in range(N):
        jobs.append({"id": f"{name}_talk{k}", "gpus": 1, "vllm": True, "deps": deps,
                     "cmd": f"CUDA_VISIBLE_DEVICES=$GPU0 VLLM_USE_FLASHINFER_SAMPLER=0 "
                            f"VLLM_CACHE_ROOT=/opt/dlami/nvme/jasonleeeli/cache/vllm_omni_030 PYTHONPATH=. "
                            f"OMP_NUM_THREADS=16 {PYV} -m {M}.si_infer --model {T1C} --direction en2zh --window 1 "
                            f"--output {out}/talker --text-policy file --blocks-file {out}/talker_in/blocks.jsonl "
                            f"--items-file {out}/talker_in/items.jsonl --speed-policy fixed --history-ban 8 "
                            f"--max-model-len 32768 --gpu-mem 0.3 --shard {k} --num-shards {N}"})
    return jobs


jobs = []
for name, thinker, extra in CASCADES:
    d = f"{CAS}/{name}"
    for k in range(N):
        jobs.append({"id": f"{name}_dec{k}", "gpus": 1, "vllm": False, "deps": [],
                     "cmd": f"mkdir -p {d} && CUDA_VISIBLE_DEVICES=$GPU0 PYTHONPATH=.:third_party/Megatron-LM {PYM} -m "
                            f"{M}.gstar_si_stream --thinker {thinker} --windows {CAS}/segments/segments.jsonl "
                            f"--out {d}/hyp_seg_s{k}.jsonl --shard {k} --num-shards {N} {extra}"})
    jobs.append({"id": f"{name}_asm", "gpus": 1, "vllm": False, "share": True, "need_mib": 0,
                 "deps": [f"{name}_dec{k}" for k in range(N)],
                 "cmd": f"set -e; cat {d}/hyp_seg_s*.jsonl > {d}/hyp_seg.jsonl; "
                        f"PYTHONPATH=. {PYE} -m {M}.gstar_cascade assemble --seg {CAS}/segments/segments.jsonl "
                        f"--hyp {d}/hyp_seg.jsonl --windows {WIN} --out {d}/hyp.jsonl; "
                        f"PYTHONPATH=. {PYE} -m {M}.gstar_score --windows {WIN} --hyp {d}/hyp.jsonl --post-key offline "
                        f"--out {d}/score.json > {d}/score.log 2>&1; "
                        f"PYTHONPATH=. {PYE} -m {M}.gstar_talker prep --hyp {d}/hyp.jsonl --windows {WIN} "
                        f"--voices {DATA}/voices_offline --out {d}/speech/talker_in"})
    out = f"{d}/speech"
    jobs += talker_jobs(name, out, [f"{name}_asm"])
    jobs.append({"id": f"{name}_post", "gpus": 1, "vllm": False, "share": True, "need_mib": 8000,
                 "deps": [f"{name}_talk{k}" for k in range(N)],
                 "cmd": f"set -e; cat {out}/talker/part_*.jsonl > {out}/talker/results.jsonl; "
                        f"{PYE} {SCR}/p80_speechonly.py {out}/talker; "
                        f"CUDA_VISIBLE_DEVICES=$GPU0 PYTHONPATH=. {PYE} -m evaluation.asr_transcribe "
                        f"--input {out}/talker/results_speechonly.jsonl --output {out}/talker/asr_speech.jsonl "
                        f"--device cuda:0 --batch-size 16 > {out}/asr.log 2>&1; "
                        f"PYTHONPATH=. {PYE} -m {M}.gstar_score --windows {WIN} --hyp {d}/hyp.jsonl "
                        f"--asr {out}/talker/asr_speech.jsonl --post-key offline --out {out}/score.json "
                        f"> {out}/score.log 2>&1"})

VOICE = [("O-G1", f"{EV}/gstar_offline_eval_o1_20261026T195403Z/O-G1/hyp.jsonl", f"{SPEECH}/O-G1", WIN, []),
         ("B-off", f"{EV}/gstar_offline_eval_o1_20261026T195403Z/B-off/hyp.jsonl", f"{SPEECH}/B-off", WIN, []),
         ("S-T", f"{EV}/gstar_stream_eval_o1_20261026T195403Z/S-T_d1/hyp.jsonl", f"{SPEECH}/S-T_d1",
          f"{SDATA}/windows_test_ref.jsonl", []),
         ("S-LA2", f"{EV}/gstar_stream_eval_o1_20261026T195403Z/S-LA2_d1/hyp.jsonl", f"{SPEECH}/S-LA2_d1",
          f"{SDATA}/windows_test_ref.jsonl", []),
         ("B-LA2", f"{EV}/gstar_stream_eval_o1_20261026T195403Z/B-LA2_d1/hyp.jsonl", f"{SPEECH}/B-LA2_d1",
          f"{SDATA}/windows_test_ref.jsonl", []),
         ("C-van", f"{CAS}/C-van/hyp.jsonl", f"{CAS}/C-van/speech", WIN, ["C-van_post"]),
         ("C-ft", f"{CAS}/C-ft/hyp.jsonl", f"{CAS}/C-ft/speech", WIN, ["C-ft_post"])]
for name, hyp, sp, win, deps in VOICE:
    jobs.append({"id": f"voice_{name}", "gpus": 1, "vllm": False, "share": True, "need_mib": 4000, "deps": deps,
                 "cmd": f"CUDA_VISIBLE_DEVICES=$GPU0 PYTHONPATH=. {PYE} -m {M}.gstar_voice --windows {win} --hyp {hyp} "
                        f"--results {sp}/talker/results.jsonl --out {sp}/voice.json"})
print(json.dumps(jobs, indent=1))

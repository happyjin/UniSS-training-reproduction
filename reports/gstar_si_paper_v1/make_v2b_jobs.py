"""Paper v2 experiments, second batch (strict review of v1):
  diar   realistic segmentation: the test windows the diarizer itself opens and closes (871 kept windows of the
         offline preset) for MIST (offline), the prompted base model, MIST w/o speaker stream and the same-training
         cascade -- the model no longer receives reference utterance groups
  ctrl   MIST-Simul decoded offline (one read at the window's end, full speaker stream): separates what its extra
         streaming training adds to translation from what its commit policy adds
  delta  latency-quality points: MIST-Simul with reads every 0.5 s and 2 s, LocalAgreement on MIST (offline) every 2 s

    python reports/gstar_si_paper_v1/make_v2b_jobs.py > reports/gstar_si_paper_v1/jobs/v2b_jobs.json
"""
import json

EV = "eval_outputs/uniss_qwen3_omni_v1"
OUT = f"{EV}/gstar_v2b_20261028"
DATA = "data/ami_multispk_si_v1/e2e_gstar_v1"
SDATA = "data/ami_multispk_si_v1/e2e_gstar_stream_v1"
PYE = "/opt/dlami/nvme/jasonleeeli/conda_envs/uniss-eval/bin/python"
PYM = "/opt/dlami/nvme/jasonleeeli/conda_envs/qwen3omni/bin/python"
M1 = f"{EV}/thinker_gstar_o1_20261026T195403Z/thinker_merged"
MS = f"{EV}/thinker_gstar_s1_20261026T195403Z/thinker_merged"
NOSD = f"{EV}/thinker_gstar_o1nosd_20261026T195403Z/thinker_merged"
REL = "pretrained_models/Qwen3-Omni-30B-A3B-Instruct"
DWIN = f"{DATA}/windows_test_diar_offline.jsonl"
SWIN = f"{SDATA}/windows_test_ref.jsonl"
M = "experiments.uniss_qwen3_omni_v1.evaluation"
N = 8

jobs = []


def decode(name, mod, thinker, win, extra, post_key, deps=(), score=True, n=N):
    d = f"{OUT}/{name}"
    for k in range(n):
        jobs.append({"id": f"{name}_dec{k}", "gpus": 1, "vllm": False, "deps": list(deps),
                     "cmd": f"mkdir -p {d} && CUDA_VISIBLE_DEVICES=$GPU0 PYTHONPATH=.:third_party/Megatron-LM {PYM} -m "
                            f"{M}.{mod} --thinker {thinker} --windows {win} --out {d}/hyp_s{k}.jsonl --shard {k} "
                            f"--num-shards {n} {extra} > {d}/infer_s{k}.log 2>&1"})
    cmd = f"set -e; cat {d}/hyp_s*.jsonl > {d}/hyp.jsonl"
    if score:
        cmd += (f"; PYTHONPATH=. {PYE} -m {M}.gstar_score --windows {win} --hyp {d}/hyp.jsonl --post-key {post_key} "
                f"--out {d}/score.json > {d}/score.log 2>&1")
    jobs.append({"id": f"{name}_score", "gpus": 1, "vllm": False, "share": True, "need_mib": 0,
                 "deps": [f"{name}_dec{k}" for k in range(n)], "cmd": cmd})


# realistic segmentation
decode("D-O-G1", "gstar_si_stream", M1, DWIN, "--inst gstar --post-key offline", "offline")
decode("D-B-off", "gstar_si_stream", REL, DWIN, "--inst sd", "offline")
decode("D-O-nosd", "gstar_si_stream", NOSD, DWIN, "--inst gstar --post-key offline --zero-sd", "offline")
jobs.append({"id": "D-seg", "gpus": 1, "vllm": False, "share": True, "need_mib": 0, "deps": [],
             "cmd": f"PYTHONPATH=. {PYE} -m {M}.gstar_cascade segment --windows {DWIN} --out {OUT}/D-segments"})
decode("D-C-ft_seg", "gstar_si_stream", NOSD, f"{OUT}/D-segments/segments.jsonl",
       "--inst gstar --post-key offline --zero-sd", "offline", deps=["D-seg"], score=False)
jobs.append({"id": "D-C-ft_score", "gpus": 1, "vllm": False, "share": True, "need_mib": 0, "deps": ["D-C-ft_seg_score"],
             "cmd": f"set -e; mkdir -p {OUT}/D-C-ft; PYTHONPATH=. {PYE} -m {M}.gstar_cascade assemble "
                    f"--seg {OUT}/D-segments/segments.jsonl --hyp {OUT}/D-C-ft_seg/hyp.jsonl --windows {DWIN} "
                    f"--out {OUT}/D-C-ft/hyp.jsonl; PYTHONPATH=. {PYE} -m {M}.gstar_score --windows {DWIN} "
                    f"--hyp {OUT}/D-C-ft/hyp.jsonl --post-key offline --out {OUT}/D-C-ft/score.json "
                    f"> {OUT}/D-C-ft/score.log 2>&1"})
# offline control of the streaming model: one read at the end, no look-ahead cut
decode("S-T-offline", "gstar_si_stream_online", MS, SWIN, "--policy wait --window 1000 --lookahead-ms 0 --post-key lowlat",
       "lowlat")
# latency-quality points
decode("S-T_d0.5", "gstar_si_stream_online", MS, SWIN, "--policy wait --window 0.5 --post-key lowlat", "lowlat")
decode("S-T_d2", "gstar_si_stream_online", MS, SWIN, "--policy wait --window 2 --post-key lowlat", "lowlat")
decode("S-LA2_d2", "gstar_si_stream_online", M1, SWIN, "--policy la2 --window 2 --post-key lowlat", "lowlat")
print(json.dumps(jobs, indent=1))

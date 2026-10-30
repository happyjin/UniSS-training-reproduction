"""Paper v4: the diarizer-defined En->Zh test windows (v2's realistic segmentation, 871 windows) for the v4 models --
the bidirectional MIST (O2), O2 without speaker stream, and the same-data cascade with O2's no-stream translator. The
prompted base model is unchanged since v2 (gstar_v2b_20261028/D-B-off). gpuq6 jobs, 4 shards per system.

    python3 reports/gstar_si_paper_v1/make_v4_diar_jobs.py > reports/gstar_si_paper_v1/jobs/v4_diar_jobs.json
"""
import json

EV = "eval_outputs/uniss_qwen3_omni_v1"
OUT = f"{EV}/gstar_v4_20261030"
DWIN = "data/ami_multispk_si_v1/e2e_gstar_v1/windows_test_diar_offline.jsonl"
PYE = "/opt/dlami/nvme/jasonleeeli/conda_envs/uniss-eval/bin/python"
PYM = "/opt/dlami/nvme/jasonleeeli/conda_envs/qwen3omni/bin/python"
O2 = f"{EV}/thinker_gstar_o2_20261029b/thinker_merged"
O2N = f"{EV}/thinker_gstar_o2nosd_20261029b/thinker_merged"
M = "experiments.uniss_qwen3_omni_v1.evaluation"
N = 4
jobs = []


def decode(name, thinker, win, extra, deps=(), score=True):
    d = f"{OUT}/{name}"
    for k in range(N):
        jobs.append({"id": f"{name}_dec{k}", "gpus": 1, "vllm": False, "deps": list(deps),
                     "cmd": f"mkdir -p {d} && CUDA_VISIBLE_DEVICES=$GPU0 PYTHONPATH=.:third_party/Megatron-LM {PYM} -m "
                            f"{M}.gstar_si_stream --thinker {thinker} --windows {win} --out {d}/hyp_s{k}.jsonl --shard {k} "
                            f"--num-shards {N} {extra} > {d}/infer_s{k}.log 2>&1"})
    cmd = f"set -e; cat {d}/hyp_s*.jsonl > {d}/hyp.jsonl"
    if score:
        cmd += (f"; PYTHONPATH=. {PYE} -m {M}.gstar_score --windows {win} --hyp {d}/hyp.jsonl --post-key offline "
                f"--out {d}/score.json > {d}/score.log 2>&1")
    jobs.append({"id": f"{name}_score", "gpus": 1, "vllm": False, "share": True, "need_mib": 0,
                 "deps": [f"{name}_dec{k}" for k in range(N)], "cmd": cmd})


decode("en2zh_D-O2", O2, DWIN, "--inst gstar --post-key offline --dir en2zh")
decode("en2zh_D-O2-nosd", O2N, DWIN, "--inst gstar --post-key offline --zero-sd --dir en2zh")
jobs.append({"id": "D-seg", "gpus": 1, "vllm": False, "share": True, "need_mib": 0, "deps": [],
             "cmd": f"PYTHONPATH=. {PYE} -m {M}.gstar_cascade segment --windows {DWIN} --out {OUT}/en2zh_D-segments"})
decode("en2zh_D-C-ft2_seg", O2N, f"{OUT}/en2zh_D-segments/segments.jsonl",
       "--inst gstar --post-key offline --zero-sd --dir en2zh", deps=["D-seg"], score=False)
jobs.append({"id": "en2zh_D-C-ft2_score", "gpus": 1, "vllm": False, "share": True, "need_mib": 0,
             "deps": ["en2zh_D-C-ft2_seg_score"],
             "cmd": f"set -e; mkdir -p {OUT}/en2zh_D-C-ft2; PYTHONPATH=. {PYE} -m {M}.gstar_cascade assemble "
                    f"--seg {OUT}/en2zh_D-segments/segments.jsonl --hyp {OUT}/en2zh_D-C-ft2_seg/hyp.jsonl --windows {DWIN} "
                    f"--out {OUT}/en2zh_D-C-ft2/hyp.jsonl; PYTHONPATH=. {PYE} -m {M}.gstar_score --windows {DWIN} "
                    f"--hyp {OUT}/en2zh_D-C-ft2/hyp.jsonl --post-key offline --out {OUT}/en2zh_D-C-ft2/score.json "
                    f"> {OUT}/en2zh_D-C-ft2/score.log 2>&1"})
print(json.dumps(jobs, indent=1))

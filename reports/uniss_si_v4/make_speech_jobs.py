"""gpuq7 jobs that speak Thinker block files with the Talker T1c@1500 and transcribe the speech (single-speaker paper v4).

  realsi  as v3's main curves (sys_runs_dirs.sh): run_si_probe.sh, text policy file, block speed policy, history ban 8,
          talker seed default / 1 / 2 (named <name>, <name>-seed1, <name>-seed2); then speech-only ASR
  cvss    CVSS-T test, whole sentences: si_infer on the items file, two shards on the job's two cards; then ASR
Talker engines at --gpu-mem 0.15 (RealSI; 3 GB of weights, about 16 GB of KV cache) or 0.25 (CVSS-T), so two
RealSI jobs share a card beside other work. Runs go to eval_outputs/uniss_qwen3_omni_v1/si_v4/speech/system_<name>[-seed<k>]_m<window>/realsi_<dir>
and .../si_v4/speech/cvss_<name>/cvss_<dir>.

    python3 reports/uniss_si_v4/make_speech_jobs.py <out_jobs.json> <spec.json>
spec: [{"kind": "realsi", "name": str, "dir": d, "window": w, "blocks_dir": dir, "suffix": str, "seeds": [0, 1, 2]},
       {"kind": "cvss", "name": str, "dir": d, "blocks": file}]
"""
import json, os, sys

O = "eval_outputs/uniss_qwen3_omni_v1"
SP = f"{O}/si_v4/speech"
EXPORT = f"{O}/si_t1c_20261021T181440Z/autoext/export_1500"
PYE = "/opt/dlami/nvme/jasonleeeli/conda_envs/uniss-eval/bin/python"
PYV = "/opt/dlami/nvme/jasonleeeli/venvs/vllm_omni_030/bin/python"
VENV = (f"PATH=/opt/dlami/nvme/jasonleeeli/venvs/vllm_omni_030/bin:$PATH VLLM_USE_FLASHINFER_SAMPLER=0 "
        f"VLLM_CACHE_ROOT=/opt/dlami/nvme/jasonleeeli/cache/vllm_omni_030 PYTHONPATH=.")
ITEMS = "data/cvss_t/si_items_v1/items_{d}.jsonl"


def asr(folder):
    return (f"{PYE} reports/uniss_si_v4/speech_only.py {folder} > /dev/null && CUDA_VISIBLE_DEVICES=$GPU0 PYTHONPATH=. {PYE} "
            f"-m evaluation.asr_transcribe --input {folder}/results_speechonly.jsonl --output {folder}/asr_speech.jsonl "
            f"--device cuda:0 --batch-size 16 && [ -f {folder}/asr_speech.jsonl ]")


jobs = []
for s in json.load(open(sys.argv[2])):
    d = s["dir"]
    if s["kind"] == "realsi":
        for seed in s["seeds"]:
            name = s["name"] + (f"-seed{seed}" if seed else "")
            out = f"{SP}/system_{name}_m{s['window']:g}"
            assert not os.path.exists(f"{out}/realsi_{d}"), out
            assert os.path.exists(f"{s['blocks_dir']}/blocks_{d}_w{s['window']:g}{s['suffix']}.jsonl"), (s, d)
            jobs.append({"id": f"sp_{name}_m{s['window']:g}_{d}", "gpus": 2, "vllm": True, "mem": 25000, "deps": [],
                         "cmd": f"env EXPORT_DIR={EXPORT} TALKER_GPU_MEM=0.15 LIMIT=0 SPEED_POLICY=block HISTORY_BAN=8 EXTRA=0 REPORT=0 "
                                f"TEXT_POLICY=file BLOCKS_DIR={s['blocks_dir']} GPUS=\"$GPUS\" OUT={out}/_{d} WINDOW={s['window']:g} "
                                f"BLOCKS_SUFFIX={s['suffix']} DIRS={d} " + (f"SEED={seed} " if seed else "") +
                                f"bash experiments/uniss_qwen3_omni_v1/scripts/run_si_probe.sh && mkdir -p {out} && "
                                f"command mv {out}/_{d}/realsi_{d} {out}/realsi_{d} && " + asr(f"{out}/realsi_{d}")})
    else:
        out = f"{SP}/cvss_{s['name']}/cvss_{d}"
        assert not os.path.exists(out), out
        shard = lambda k, g: (f"CUDA_VISIBLE_DEVICES={g} {VENV} {PYV} -m experiments.uniss_qwen3_omni_v1.evaluation.si_infer "  # noqa: E731
                              f"--model {EXPORT}/uniss --direction {d} --window 60 --output {out} --text-policy file "
                              f"--blocks-file {s['blocks']} --items-file {ITEMS.format(d=d)} --speed-policy block "
                              f"--history-ban 8 --gpu-mem 0.25 --shard {k} --num-shards 2 > {out}_shard{k}.log 2>&1")
        jobs.append({"id": f"cvss_{s['name']}_{d}", "gpus": 2, "vllm": True, "mem": 40000, "deps": [],
                     "cmd": f"mkdir -p {os.path.dirname(out)} && ({shard(0, '$GPU0')} & sleep 60; {shard(1, '$GPU1')}; wait) && "
                            f"[ $(ls {out}/part_*.jsonl | wc -l) -eq 2 ] && " + asr(out)})
with open(sys.argv[1], "x") as fh:
    json.dump(jobs, fh, indent=1)
print(len(jobs), "jobs")

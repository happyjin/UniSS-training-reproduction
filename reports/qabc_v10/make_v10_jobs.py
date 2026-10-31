"""gpuq7 jobs for paper v10 (paper/bayes_simul_s2st_v10/docs/prereg_v5.zh-CN.md). Everything goes under
eval_outputs/uniss_qwen3_omni_v1/qabc_v10/ (O10); nothing existing is rerun or overwritten.

  text <seed> ...     RealSI QABC grids (b0 = 1, SDR 0.191 / 0.339, the 15 registered windows) into O10/realsi_grid_<seed>
                      and ACL60/60 QABC into O10/acl_<seed> (then StreamLAAL into O10/acl_<seed>/seg); seed "qb" is s0 with
                      the binary queue gate (ACL only: its RealSI grid exists in si_v4/realsi_grid_s0qb)
  speech <name>=<blocks_dir>:<en2zh suffix>:<zh2en suffix> ...
                      the Talker T1c@1500 speaks every window of a RealSI grid with talker seeds 0/1/2 (run_si_probe.sh,
                      block speed policy, history ban 8, as v4) then the speech-only ASR; into O10/speech/system_<name>[-seed<k>]_m<w>
  rules <dirs> <rules> [--items dev|realsi] [--windows ...] [--tag ...] [--recal json]
                      si_qabc_rules on the s0 Thinker (--dump --label on dev)
  devspeech <name>=<blocks_dir>:<en2zh suffix>:<zh2en suffix>
                      one talker seed on the dev items (si_infer --items-file, two shards), then the speech-only ASR

    python3 reports/qabc_v10/make_v10_jobs.py <out_jobs.json> <mode> ...
"""
import json
import os
import sys
from pathlib import Path

O = "eval_outputs/uniss_qwen3_omni_v1"
O10 = f"{O}/qabc_v10"
EXPORT = f"{O}/si_t1c_20261021T181440Z/autoext/export_1500"
M = {"s0": f"{O}/thinker_op_v3_20261023T213631Z/thinker_merged_i200",
     "s1": f"{O}/thinker_op_v3s1_20261025T143042Z/thinker_merged",
     "s2": f"{O}/thinker_op_v3s2_20261025T143042Z/thinker_merged",
     "s3": f"{O}/thinker_op_v3s3_20261031/thinker_merged_i200",
     "s4": f"{O}/thinker_op_v3s4_20261031/thinker_merged_i200"}
RATE = {"en2zh": "0.191", "zh2en": "0.339"}
W15 = "0.5,0.6,0.75,0.9,1,1.1,1.25,1.5,1.75,2,2.5,3,4,5,6"
DEVW = "0.5,0.75,1,1.25,1.5,2,2.5,3"
DEV = "data/si_thinker_oracle_prefix_v1/dev_items_{d}.jsonl"
PYE = "/opt/dlami/nvme/jasonleeeli/conda_envs/uniss-eval/bin/python"
PYV = "/opt/dlami/nvme/jasonleeeli/venvs/vllm_omni_030/bin/python"
VENV = ("PATH=/opt/dlami/nvme/jasonleeeli/venvs/vllm_omni_030/bin:$PATH OMP_NUM_THREADS=16 VLLM_USE_FLASHINFER_SAMPLER=0 "
        "VLLM_CACHE_ROOT=/opt/dlami/nvme/jasonleeeli/cache/vllm_omni_030 PYTHONPATH=.:.tmp/pylib_jieba")
ST = f"CUDA_VISIBLE_DEVICES=${{GPUS// /,}} {VENV} {PYV} -m experiments.uniss_qwen3_omni_v1.evaluation.si_thinker_stream --tp 2 --gpu-mem 0.3"
QR = f"CUDA_VISIBLE_DEVICES=${{GPUS// /,}} {VENV} {PYV} -m experiments.uniss_qwen3_omni_v1.evaluation.si_qabc_rules --tp 2 --gpu-mem 0.3"
ACL = "--direction en2zh --items-file data/acl6060_dev_v1/items.jsonl"
SCORE = (f"{PYE} -m experiments.uniss_qwen3_omni_v1.evaluation.si_streamlaal score --refs data/acl6060_dev_v1/refs.jsonl "
         f"--items data/acl6060_dev_v1/items.jsonl --procs 32")


def asr(folder):
    return (f"{PYE} reports/uniss_si_v4/speech_only.py {folder} > /dev/null && CUDA_VISIBLE_DEVICES=$GPU0 PYTHONPATH=. {PYE} "
            f"-m evaluation.asr_transcribe --input {folder}/results_speechonly.jsonl --output {folder}/asr_speech.jsonl "
            f"--device cuda:0 --batch-size 4 && [ -f {folder}/asr_speech.jsonl ]")


def text_jobs(seeds):
    jobs = []
    for s in seeds:
        if s != "qb":
            for d in ("en2zh", "zh2en"):
                out = f"{O10}/realsi_grid_{s}"
                jobs.append({"id": f"grid_{s}_{d}", "gpus": 2, "vllm": True, "mem": 56000, "deps": [],
                             "cmd": f"{ST} --policy selfwait --thinker {M[s]} --thinker-tag {s} --wait-biases=1 "
                                    f"--sw-queue-rate {RATE[d]} --direction {d} --windows {W15} --out {out}"})
        m, extra, tag = (M["s0"], " --sw-queue-binary", "s0") if s == "qb" else (M[s], "", s)
        out = f"{O10}/acl_{s}"
        jobs.append({"id": f"acl_{s}", "gpus": 2, "vllm": True, "mem": 56000, "deps": [],
                     "cmd": f"{ST} --policy selfwait --thinker {m} --thinker-tag {tag} --wait-biases=1 --sw-queue-rate 0.191{extra} "
                            f"{ACL} --windows {W15} --out {out} && {SCORE} --blocks '{out}/blocks_en2zh_*.jsonl' --dump {out}/seg "
                            f"> {out}/streamlaal.jsonl"})
    return jobs


def speech_jobs(specs, seeds=(0, 1, 2)):
    jobs = []
    for spec in specs:
        name, rest = spec.split("=", 1)
        bdir, suf_en, suf_zh = rest.split(":")
        for d, suf in (("en2zh", suf_en), ("zh2en", suf_zh)):
            for w in W15.split(","):
                f = f"{bdir}/blocks_{d}_w{w}{suf}.jsonl"
                for seed in seeds:
                    nm = (name if d == "en2zh" else f"{name}-zh") + (f"-seed{seed}" if seed else "")
                    out = f"{O10}/speech/system_{nm}_m{w}"
                    assert not os.path.exists(f"{out}/_{d}"), out
                    jobs.append({"id": f"sp_{nm}_m{w}_{d}", "gpus": 2, "vllm": True, "mem": 25000, "deps": [], "blocks": f,
                                 "cmd": f"env EXPORT_DIR={EXPORT} TALKER_GPU_MEM=0.15 LIMIT=0 SPEED_POLICY=block HISTORY_BAN=8 "
                                        f"EXTRA=0 REPORT=0 TEXT_POLICY=file BLOCKS_DIR={bdir} GPUS=\"$GPUS\" OUT={out}/_{d} WINDOW={w} "
                                        f"BLOCKS_SUFFIX={suf} DIRS={d} " + (f"SEED={seed} " if seed else "") +
                                        f"bash experiments/uniss_qwen3_omni_v1/scripts/run_si_probe.sh && " + asr(f"{out}/_{d}/realsi_{d}")})
    return jobs


def rules_jobs(args):
    dirs, rules = args[0].split(","), args[1]
    opts = dict(zip(args[2::2], args[3::2]))
    items = opts.get("--items", "dev")
    windows = opts.get("--windows", DEVW if items == "dev" else W15)
    tag = opts.get("--tag", "s0")
    jobs = []
    for d in dirs:
        out = f"{O10}/{'dev' if items == 'dev' else 'realsi'}_rules"
        it = (f"--items-file {DEV.format(d=d)} --dump {out}/dump --label" if items == "dev" else
              f"--dump {out}/dump --label" if opts.get("--dump") == "1" else "")
        rc = f" --recal-json {opts['--recal']} --recal-tag {opts.get('--recal-tag', '')}" if "--recal" in opts else ""
        jobs.append({"id": f"rules_{items}_{rules.replace(',', '-')}_{d}", "gpus": 2, "vllm": True, "mem": 56000, "deps": [],
                     "cmd": f"{QR} --direction {d} --windows {windows} --rules {rules} --thinker {M['s0']} --thinker-tag {tag} "
                            f"--b0 1 --rate {RATE[d]} --out {out} {it}{rc}"})
    return jobs


def devspeech_jobs(specs, windows=DEVW):
    jobs = []
    for spec in specs:
        name, rest = spec.split("=", 1)
        bdir, suf_en, suf_zh = rest.split(":")
        for d, suf in (("en2zh", suf_en), ("zh2en", suf_zh)):
            for w in windows.split(","):
                f = f"{bdir}/blocks_{d}_w{w}{suf}.jsonl"
                out = f"{O10}/devspeech/{name}_{d}_m{w}"
                assert not os.path.exists(out), out
                shard = lambda k, g: (f"CUDA_VISIBLE_DEVICES={g} {VENV} {PYV} -m experiments.uniss_qwen3_omni_v1.evaluation.si_infer "  # noqa: E731
                                      f"--model {EXPORT}/uniss --direction {d} --window {w} --output {out} --text-policy file "
                                      f"--blocks-file {f} --items-file {DEV.format(d=d)} --speed-policy block --history-ban 8 "
                                      f"--gpu-mem 0.15 --shard {k} --num-shards 2 > {out}_shard{k}.log 2>&1")
                jobs.append({"id": f"dsp_{name}_{d}_m{w}", "gpus": 2, "vllm": True, "mem": 25000, "deps": [], "blocks": f,
                             "cmd": f"mkdir -p {os.path.dirname(out)} && ({shard(0, '$GPU0')} & sleep 60; {shard(1, '$GPU1')}; wait) && "
                                    f"[ $(ls {out}/part_*.jsonl | wc -l) -eq 2 ] && " + asr(out)})
    return jobs


def main():
    out, mode, args = sys.argv[1], sys.argv[2], sys.argv[3:]
    jobs = {"text": lambda: text_jobs(args), "speech": lambda: speech_jobs(args), "rules": lambda: rules_jobs(args),
            "devspeech": lambda: devspeech_jobs(args)}[mode]()
    for j in jobs:   # speech needs its blocks: fail here, not in the queue
        if "blocks" in j:
            assert os.path.exists(j.pop("blocks")), j["id"]
    with open(out, "x") as fh:
        json.dump(jobs, fh, indent=1)
    print(len(jobs), "jobs ->", out)


if __name__ == "__main__":
    main()

"""QABC v11 test runs on the continuous clock (paper/bayes_simul_s2st_v11/docs/prereg_v6.zh-CN.md), into
eval_outputs/uniss_qwen3_omni_v1/qabc_v11/ (Q11); nothing existing is rerun or overwritten. The control and gamma come
from generated/v11_devctrl.json (chosen on dev before any of this runs).

  thinker <queue_add.json> [--seed s0]   si_qabc_cont on RealSI, rules p5/gate/linlogit/seqb, windows split over three
                                         jobs per direction (TP2, 56 GB a card), dumps on -> Q11/cont_<seed>
  la2                                    the registered LA2 blocks (realsi_thinker_stream_v1) with the tracked queue and
                                         shift of the same control -> Q11/cont_la2 (CPU)
  talker <tasks.jsonl> <queue_add.json> <n_talker> <n_asr> [--systems ...] [--seeds 0,1]
                                         talker_worker.py tasks for every continuous blocks file and its queue jobs
                                         -> Q11/cont_speech/<system>_<d>_m<w>[-seed<k>]

    python3 reports/qabc_v11/make_v11_jobs.py <mode> ...
"""
import copy
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cont_queue as cq  # noqa: E402

R = cq.REPO
O = "eval_outputs/uniss_qwen3_omni_v1"
Q11 = f"{O}/qabc_v11"
M = {"s0": f"{O}/thinker_op_v3_20261023T213631Z/thinker_merged_i200",
     "s1": f"{O}/thinker_op_v3s1_20261025T143042Z/thinker_merged"}
W6 = ("0.5", "0.75", "1", "1.5", "2", "3")
GROUPS = (("0.5",), ("0.75", "1"), ("1.5", "2", "3"))
RULES = "p5,gate,linlogit,seqb"
ENV = ("PATH=/opt/dlami/nvme/jasonleeeli/venvs/vllm_omni_030/bin:$PATH OMP_NUM_THREADS=16 VLLM_USE_FLASHINFER_SAMPLER=0 "
       "VLLM_CACHE_ROOT=/opt/dlami/nvme/jasonleeeli/cache/vllm_omni_030 PYTHONPATH=.:.tmp/pylib_jieba PYTHONUNBUFFERED=1")
PYV = "/opt/dlami/nvme/jasonleeeli/venvs/vllm_omni_030/bin/python"


def ctrl():
    c = json.load(open(R / "paper/bayes_simul_s2st_v11/generated/v11_devctrl.json"))
    return c["choice"], {d: round(c["gamma"][d]["gamma"], 4) for d in ("en2zh", "zh2en")}


def thinker(args):
    add = args[0]
    seed = args[args.index("--seed") + 1] if "--seed" in args else "s0"
    ch, gam = ctrl()
    jobs = []
    for d in ("en2zh", "zh2en"):
        for k, ws in enumerate(GROUPS):
            out = f"{Q11}/cont_{seed}"
            jobs.append({"id": f"qabc_cont_{seed}_{d}_{k}", "gpus": 2, "mem": 56000, "kind": "decode", "deps": [],
                         "cmd": f"sleep {60 * (len(jobs) % 3)}; CUDA_VISIBLE_DEVICES=$CVD {ENV} {PYV} -m "
                                f"experiments.uniss_qwen3_omni_v1.evaluation.si_qabc_cont --tp 2 --gpu-mem 0.3 --direction {d} "
                                f"--windows {','.join(ws)} --rules {RULES} --thinker {M[seed]} --thinker-tag {seed} --b0 1 "
                                f"--ctrl {ch} --gamma {gam[d]} --out {out} --dump {out}/dump"})
    with open(add, "x") as fh:
        json.dump(jobs, fh, indent=1)
    print(len(jobs), "jobs ->", add, "control", ch, gam)


def la2(args):
    ch, gam = ctrl()
    out = R / Q11 / "cont_la2"
    out.mkdir(parents=True, exist_ok=True)
    for d in ("en2zh", "zh2en"):
        talks = cq.realsi_talks(d)
        where = {sid: (t, off) for t, s in talks for sid, off, _ in s}
        for w in W6:
            src = R / O / f"realsi_thinker_stream_v1/blocks_{d}_w{w}.jsonl"
            b = cq.read_blocks(src)
            cq.carry(b, talks, d, ch, gam[d])
            dst = out / f"blocks_{d}_w{w}_la2_{ch}.jsonl"
            n = 0
            with open(dst, "x", encoding="utf-8") as fh:
                for r in map(json.loads, open(src, encoding="utf-8")):
                    if r["id"] in b and r["id"] in where:
                        r["blocks"], r["talk"], r["offset_ms"] = b[r["id"]], where[r["id"]][0], where[r["id"]][1]
                        fh.write(json.dumps(r, ensure_ascii=False) + "\n")
                        n += 1
            print(dst.name, n)


def talker(args):
    tasks_path, add, n_t, n_a = args[0], args[1], int(args[2]), int(args[3])
    systems = args[args.index("--systems") + 1].split(",") if "--systems" in args else ["la2", "p5", "gate", "linlogit", "seqb"]
    seeds = [int(s) for s in (args[args.index("--seeds") + 1] if "--seeds" in args else "0,1").split(",")]
    tag = args[args.index("--tag") + 1] if "--tag" in args else "s0"
    ch, _ = ctrl()
    tasks = []
    for d in ("en2zh", "zh2en"):
        r = cq.RATE[d]
        for system in systems:
            for w in W6:
                f = (f"{Q11}/cont_la2/blocks_{d}_w{w}_la2_{ch}.jsonl" if system == "la2"
                     else f"{Q11}/cont_{tag}/blocks_{d}_w{w}_c{system}{tag}b1q{r:g}_{ch}.jsonl")
                for s in seeds:
                    name = f"{system}{'' if system == 'la2' or tag == 's0' else '_' + tag}_{d}_m{w}" + (f"-seed{s}" if s else "")
                    out = f"{Q11}/cont_speech/{name}"
                    assert not os.path.exists(R / out), out
                    tasks.append({"key": f"cont_{name}", "blocks": str(R / f), "direction": d, "window": float(w), "seed": s,
                                  "out": str(R / out), "speed_policy": "tracked"})
    with open(tasks_path, "x") as fh:
        fh.writelines(json.dumps(t) + "\n" for t in tasks)
    jobs = []
    for k in range(n_t):
        jobs.append({"id": f"qabc_conttalk_{tag}_{k}", "gpus": 1, "mem": 26000, "kind": "speech", "deps": [],
                     "cmd": f"sleep {75 * k}; env {ENV} CUDA_VISIBLE_DEVICES=$GPU0 {PYV} -u reports/qabc_v11/talker_worker.py talker "
                            f"{tasks_path} --gpu-mem 0.10"})
    for k in range(n_a):
        jobs.append({"id": f"qabc_contasr_{tag}_{k}", "gpus": 1, "mem": 24000, "kind": "speech", "deps": [],
                     "cmd": f"sleep {600 + 120 * k}; env {ENV} CUDA_VISIBLE_DEVICES=$GPU0 {PYV} -u reports/qabc_v11/talker_worker.py asr {tasks_path}"})
    with open(add, "x") as fh:
        json.dump(jobs, fh, indent=1)
    print(len(tasks), "tasks ->", tasks_path, ";", len(jobs), "jobs ->", add)


if __name__ == "__main__":
    {"thinker": thinker, "la2": la2, "talker": talker}[sys.argv[1]](sys.argv[2:])

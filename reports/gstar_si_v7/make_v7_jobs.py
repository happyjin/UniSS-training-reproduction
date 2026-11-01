"""Jobs of the paper-v7 (MIST) queue (paper/gstar_si_v7/docs/prereg_v7.zh-CN.md). Each phase writes one job list to
the queue's add/ directory (read once by gpuq7v7.py) and the same list to reports/gstar_si_v7/jobs/<phase>.json
(kept in git); both with mode "x", so a phase can never be written twice. Every decode output goes to a new directory
eval_outputs/uniss_qwen3_omni_v1/gstar_v7_20261101/<name>/ (hyp_s<k>.jsonl, infer_s<k>.log).

The Thinker is split over two cards (gstar_si_stream_online_v6s / gstar_si_stream_v7s, --max-gib 34; ~40 GB per card
with batch 32): a colleague's job holds ~50-55 GB of every card, so a whole copy (~61 GB) no longer fits under 120 GB.

  p1   whole-meeting online decodes of the meetings v6 did not decode (prereg_v7 §1: MIST-Simul and MIST + LA2, three
       test sets), the N-check (En->Zh MIST-Simul's v6 subset decoded again), the oracle-turn and context cascades
       (§2, O2-nosd), and a smoke test of the split offline decoder

  p2   the turn-trained cascade's decodes on the cascade's own diarizer turns (C-ft2 segments, three test sets; start
       once the model is merged) and the seed trainings (Zh->En 3k recipe seed 20261101, then En->Zh seed 20261102),
       training on the training lane (4 cards, MBS 4, GBS 64)

    python3 reports/gstar_si_v7/make_v7_jobs.py p1
"""
import json
import sys
from pathlib import Path

EV = "eval_outputs/uniss_qwen3_omni_v1"
OUT = f"{EV}/gstar_v7_20261101"
V6 = f"{EV}/gstar_v6_20261031"
ON = "data/online_v6_20261031"
D = "data/ami_multispk_si_v1"
T = {"S2DL": f"{EV}/thinker_gstar_s2dl_20261029b/thinker_merged",
     "S2DLZ": f"{EV}/thinker_gstar_s2dl_zh2en_3k_20261030/thinker_merged",
     "O2": f"{EV}/thinker_gstar_o2_20261029b/thinker_merged",
     "O2N": f"{EV}/thinker_gstar_o2nosd_20261029b/thinker_merged",
     "CTURN": f"{EV}/thinker_gstar_cturn_20261101/thinker_merged",
     "S2DL3": f"{EV}/thinker_gstar_s2dl_seed20261102/thinker_merged",
     "S2DL2": f"{EV}/thinker_gstar_s2dl_seed20261101/thinker_merged",
     "S2DLZ2": f"{EV}/thinker_gstar_s2dl_zh2en_3k_seed20261101/thinker_merged"}
V3 = f"{EV}/gstar_v3_20261029"
DECODE_LANE, TRAIN_LANE = [1, 2, 3, 4], [0, 5, 6, 7]
PYM = "/opt/dlami/nvme/jasonleeeli/conda_envs/qwen3omni/bin/python"
PRE = f"CUDA_VISIBLE_DEVICES=$CVD PYTHONPATH=.:third_party/Megatron-LM {PYM} -m experiments.uniss_qwen3_omni_v1.evaluation"
SPLIT_MIB, BATCH, GIB = 46000, 32, 34
QADD = Path("logs/multispk/v7/queue2/add")   # queue2 (gpuq7v7b) from 14:26 UTC
JDIR = Path("reports/gstar_si_v7/jobs")


def rest_windows(s, sysname, name):
    """Windows of the full online set that v6 did not decode for this system -> <OUT>/<name>/todo.windows.jsonl."""
    done = {json.loads(l)["id"] for l in open(f"{V6}/on_{s}_{sysname}/hyp.jsonl")}
    full = [l for l in open(f"{ON}/windows_test_online_{s}.jsonl")]
    rest = [l for l in full if json.loads(l)["id"] not in done]
    p = Path(OUT) / name / "todo.windows.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "x") as fh:
        fh.writelines(rest)
    return str(p), len(rest)


def online(jid, name, thinker, win, d, policy, k=0, n=1, extra=""):
    Path(OUT, name).mkdir(parents=True, exist_ok=True)
    return {"id": jid, "gpus": 2, "mem": SPLIT_MIB, "kind": "decode", "deps": [], "cards": DECODE_LANE,
            "cmd": f"{PRE}.gstar_si_stream_online_v6s --thinker {T[thinker]} --windows {win} --out {OUT}/{name}/hyp_s{k}.jsonl "
                   f"--shard {k} --num-shards {n} --policy {policy} --window 1 --post-key lowlat --dir {d} --batch {BATCH} "
                   f"--max-gib {GIB} {extra}> {OUT}/{name}/infer_s{k}.log 2>&1"}


def offline(jid, name, thinker, win, d, k=0, n=1, extra="--zero-sd", deps=()):
    Path(OUT, name).mkdir(parents=True, exist_ok=True)
    return {"id": jid, "gpus": 2, "mem": SPLIT_MIB, "kind": "decode", "deps": list(deps), "cards": DECODE_LANE,
            "cmd": f"{PRE}.gstar_si_stream_v7s --thinker {T[thinker]} --windows {win} --out {OUT}/{name}/hyp_s{k}.jsonl "
                   f"--shard {k} --num-shards {n} --inst gstar --post-key offline --dir {d} --batch {BATCH} --max-gib {GIB} "
                   f"{extra} > {OUT}/{name}/infer_s{k}.log 2>&1"}


def p1():
    jobs, info = [], {}
    jobs.append(offline("smoke_v7s", "smoke_v7s_offline", "O2N", f"{V6}/en2zh_OC-segments/segments.jsonl", "en2zh",
                        extra="--zero-sd --limit 64"))
    for s, th in (("en2zh", "S2DL"), ("zh2en", "S2DLZ"), ("ali", "S2DLZ")):
        w, n = rest_windows(s, "S2dl", f"on_{s}_S2dl_rest")
        info[f"on_{s}_S2dl_rest"] = n
        jobs.append(online(f"on_{s}_S2dl_rest", f"on_{s}_S2dl_rest", th, w, "en2zh" if s == "en2zh" else "zh2en", "wait"))
    for s in ("en2zh", "zh2en", "ali"):
        w, n = rest_windows(s, "O2_LA2", f"on_{s}_O2_LA2_rest")
        info[f"on_{s}_O2_LA2_rest"] = n
        for k in range(2):
            jobs.append(online(f"on_{s}_O2_LA2_rest_{k}", f"on_{s}_O2_LA2_rest", "O2", w,
                               "en2zh" if s == "en2zh" else "zh2en", "la2", k, 2))
    jobs.append(online("nchk_on_en2zh_S2dl_sub", "nchk_on_en2zh_S2dl_sub", "S2DL",
                       f"{ON}/windows_test_online_en2zh_sub.jsonl", "en2zh", "wait"))
    for d in ("en2zh", "zh2en"):
        for c in ("OC", "CC"):
            jobs.append(offline(f"{d}_{c}_seg", f"{d}_{c}_seg", "O2N", f"{V6}/{d}_{c}-segments/segments.jsonl", d,
                                deps=["smoke_v7s"]))
    return jobs, info


def train(jid, rid, port, env, deps=()):
    log = f"logs/multispk/v7/{rid}.log"
    return {"id": jid, "gpus": 4, "mem": 64000, "kind": "train", "deps": list(deps), "cards": TRAIN_LANE,
            "cmd": f"env CUDA_VISIBLE_DEVICES=$CVD NPROC=4 MBS=4 GBS=64 RUN_ID={rid} {env} MASTER_PORT={port} "
                   f"bash experiments/uniss_qwen3_omni_v1/scripts/run_thinker_gstar_v7.sh > {log} 2>&1 && "
                   f"grep -q THINKER_GSTAR_DONE {log}"}


def p2():
    jobs = []
    for s, d in (("en2zh", "en2zh"), ("zh2en", "zh2en"), ("ali", "zh2en")):
        for k in range(2):
            j = offline(f"{s}_Cturn_seg_{k}", f"{s}_Cturn_seg", "CTURN", f"{V3}/segments_{s}/segments.jsonl", d, k, 2)
            j["needs"] = [f"{T['CTURN']}/sd_proj.pt"]
            jobs.append(j)
    O2 = T["O2"].rsplit("/", 1)[0] + "/thinker_merged"
    jobs.append(train("train_seed_zh2", "thinker_gstar_s2dl_zh2en_3k_seed20261101", 29882,
                      f"BASE={O2} INIT_MERGED=1 ROWS={D}/e2e_gstar_stream_zh2en_v1/rows_train.jsonl TRAIN_ITERS=3000 "
                      f"TRAIN_SEED=20261101 GSTAR_SEED=20261101"))
    jobs.append(train("train_seed_en3", "thinker_gstar_s2dl_seed20261102", 29883,
                      f"BASE={O2} INIT_MERGED=1 ROWS={D}/e2e_gstar_stream_v1/rows_train.jsonl TRAIN_ITERS=3000 "
                      f"TRAIN_SEED=20261102 GSTAR_SEED=20261102", deps=["train_seed_zh2"]))
    return jobs, {}


if __name__ == "__main__":
    phase = sys.argv[1]
    jobs, info = globals()[phase]()
    JDIR.mkdir(parents=True, exist_ok=True)
    with open(JDIR / f"{phase}.json", "x") as fh:
        json.dump(jobs, fh, indent=1)
    n = len(list(QADD.glob("*.json")))
    with open(QADD / f"{n:03d}_{phase}.json", "x") as fh:
        json.dump(jobs, fh, indent=1)
    print(json.dumps({"phase": phase, "jobs": len(jobs), **info}))

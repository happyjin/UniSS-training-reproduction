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

  p4   reference-window decodes (prereg_v7 §3-§4): the seeds (En->Zh seed 20261101 on the 8 meetings v6 did not decode,
       seed 20261102, Zh->En and AliMeeting seed 20261101; each starts once its model is merged), LA2 on the prefix-trained
       models (En->Zh remaining 8 meetings, Zh->En, AliMeeting), the streaming cascade (AMI both directions) and fixed lag
       (En->Zh L = 1, 2 s; Zh->En L = 2 s). Writes reports/gstar_si_v7/refwin_systems.txt for after_v7.sh

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
QADD = Path("logs/multispk/v7/queue3/add")   # queue3 from 2026-11-02 13:24 UTC (queue2 from 11-01 14:26)
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


def online(jid, name, thinker, win, d, policy, k=0, n=1, extra="", batch=BATCH, tag=""):
    Path(OUT, name).mkdir(parents=True, exist_ok=True)
    return {"id": jid, "gpus": 2, "mem": SPLIT_MIB, "kind": "decode", "deps": [], "cards": DECODE_LANE,
            "cmd": f"{PRE}.gstar_si_stream_online_v6s --thinker {T[thinker]} --windows {win} --out {OUT}/{name}/hyp_s{k}{tag}.jsonl "
                   f"--shard {k} --num-shards {n} --policy {policy} --window 1 --post-key lowlat --dir {d} --batch {batch} "
                   f"--max-gib {GIB} {extra}> {OUT}/{name}/infer_s{k}{tag}.log 2>&1"}


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


REF = {"en2zh": f"{D}/e2e_gstar_stream_v1/windows_test_ref.jsonl", "zh2en": f"{D}/e2e_gstar_zh2en_v1/windows_test_ref.jsonl",
       "ali": "data/alimeeting_si_v1/windows_test_ref.jsonl"}


def minus(win, done_hyp, name):
    done = {json.loads(l)["id"] for l in open(done_hyp)}
    rest = [l for l in open(win) if json.loads(l)["id"] not in done]
    p = Path(OUT) / name / "todo.windows.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "x") as fh:
        fh.writelines(rest)
    return str(p)


def p4():
    jobs, lst = [], []
    def add(name, th, win, d, policy, n, extra="", needs=None, pk="lowlat"):
        for k in range(n):
            j = online(f"{name}_{k}", name, th, win, d, policy, k, n, extra)
            if needs:
                j["needs"] = [needs]
            jobs.append(j)
        lst.append(f"{OUT}/{name}:{win}:{pk}".replace(f"{OUT}/", "", 1))
    w = minus(REF["en2zh"], f"{V6}/en2zh_S2dl_seed1_d1/hyp.jsonl", "en2zh_S2dl_seed2_rest")
    add("en2zh_S2dl_seed2_rest", "S2DL2", w, "en2zh", "wait", 1)
    add("en2zh_S2dl_seed3_d1", "S2DL3", REF["en2zh"], "en2zh", "wait", 2, needs=f"{T['S2DL3']}/sd_proj.pt")
    add("zh2en_S2dlZ_seed2_d1", "S2DLZ2", REF["zh2en"], "zh2en", "wait", 2, needs=f"{T['S2DLZ2']}/sd_proj.pt")
    add("ali_S2dlZ_seed2_d1", "S2DLZ2", REF["ali"], "zh2en", "wait", 2, needs=f"{T['S2DLZ2']}/sd_proj.pt")
    w = minus(REF["en2zh"], f"{V6}/en2zh_S2dl_LA2_d1/hyp.jsonl", "en2zh_S2dl_LA2_rest")
    add("en2zh_S2dl_LA2_rest", "S2DL", w, "en2zh", "la2", 2)
    add("zh2en_S2dlZ_LA2_d1", "S2DLZ", REF["zh2en"], "zh2en", "la2", 3)
    add("ali_S2dlZ_LA2_d1", "S2DLZ", REF["ali"], "zh2en", "la2", 3)
    for d in ("en2zh", "zh2en"):
        add(f"{d}_SC_seg", "O2N", f"{V6}/{d}_SC-segments/segments.jsonl", d, "la2", 2, "--zero-sd ")
    add("en2zh_S2dl_lag1", "S2DL", REF["en2zh"], "en2zh", "lag", 2, "--lag-ms 1000 ")
    add("en2zh_S2dl_lag2", "S2DL", REF["en2zh"], "en2zh", "lag", 2, "--lag-ms 2000 ")
    add("zh2en_S2dlZ_lag2", "S2DLZ", REF["zh2en"], "zh2en", "lag", 3, "--lag-ms 2000 ")
    with open("reports/gstar_si_v7/refwin_systems.txt", "x") as fh:
        fh.write("\n".join(x for x in lst if "_SC_seg" not in x) + "\n")
    return jobs, {}


def p5():
    """prereg_v7 amendment 2: fixed lag on the 8-meeting En->Zh subset (950 reference windows), then LA2 on the
    prefix-trained Zh->En model for AliMeeting re-added last (its p4 jobs were cancelled before starting)."""
    jobs, lst = [], []
    sub = "data/online_v6_20261031/windows_test_ref_en2zh_stream_sub.jsonl"
    for L in (1, 2):
        name = f"en2zh_S2dl_lag{L}_sub"
        for k in range(2):
            jobs.append(online(f"{name}_{k}", name, "S2DL", sub, "en2zh", "lag", k, 2, f"--lag-ms {1000 * L} "))
        lst.append(f"{name}:{sub}:lowlat")
    for k in range(3):
        jobs.append(online(f"ali_S2dlZ_LA2_d1_late_{k}", "ali_S2dlZ_LA2_d1", "S2DLZ", REF["ali"], "zh2en", "la2", k, 3))
    keep = [l for l in open("reports/gstar_si_v7/refwin_systems.txt").read().split() if "_lag" not in l]
    with open("reports/gstar_si_v7/refwin_systems.txt.new", "x") as fh:
        fh.write("\n".join(keep + lst) + "\n")
    return jobs, {}


def p6():
    """AliMeeting reference QE (GEMBA-DA with the original Qwen3-Omni Thinker): a 96-utterance smoke run, then all."""
    Path(OUT, "qe").mkdir(parents=True, exist_ok=True)
    py = f"CUDA_VISIBLE_DEVICES=$CVD PYTHONPATH=. {PYM} reports/gstar_si_v7/qe_gemba.py"
    return [{"id": "qe_smoke", "gpus": 2, "mem": SPLIT_MIB, "kind": "decode", "deps": [], "cards": DECODE_LANE,
             "cmd": f"{py} --limit 96 --out {OUT}/qe/ali_ref_gemba_smoke.jsonl > {OUT}/qe/smoke.log 2>&1"},
            {"id": "qe_all", "gpus": 2, "mem": SPLIT_MIB, "kind": "decode", "deps": ["qe_smoke"], "cards": DECODE_LANE,
             "cmd": f"{py} --out {OUT}/qe/ali_ref_gemba.jsonl > {OUT}/qe/all.log 2>&1"}], {}


def p7():
    """AliMeeting decodes at batch 12 (16:09 UTC: the batch-32 AliMeeting decode reached 71 GB on one card and the
    watchdog stopped it): the rest of on_ali_S2dl_rest (windows not in hyp_s0.jsonl), and the not-yet-started AliMeeting
    jobs of p1 / p4 / p5 (cancelled in the queue) re-made with batch 12. Outputs: hyp_s<k>b.jsonl beside the others."""
    jobs = []
    d = Path(OUT) / "on_ali_S2dl_rest"
    done = {json.loads(l)["id"] for l in open(d / "hyp_s0.jsonl") if l.strip().endswith("}")}
    rest = [l for l in open(d / "todo.windows.jsonl") if json.loads(l)["id"] not in done]
    with open(d / "todo_b.windows.jsonl", "x") as fh:
        fh.writelines(rest)
    jobs.append(online("on_ali_S2dl_rest_b", "on_ali_S2dl_rest", "S2DLZ", str(d / "todo_b.windows.jsonl"), "zh2en", "wait",
                       batch=12, tag="b"))
    w = str(Path(OUT) / "on_ali_O2_LA2_rest" / "todo.windows.jsonl")
    for k in range(2):
        jobs.append(online(f"on_ali_O2_LA2_rest_{k}b", "on_ali_O2_LA2_rest", "O2", w, "zh2en", "la2", k, 2, batch=12, tag="b"))
    for k in range(2):
        j = online(f"ali_S2dlZ_seed2_d1_{k}b", "ali_S2dlZ_seed2_d1", "S2DLZ2", REF["ali"], "zh2en", "wait", k, 2, batch=12, tag="b")
        j["needs"] = [f"{T['S2DLZ2']}/sd_proj.pt"]
        jobs.append(j)
    for k in range(3):
        jobs.append(online(f"ali_S2dlZ_LA2_d1_{k}b", "ali_S2dlZ_LA2_d1", "S2DLZ", REF["ali"], "zh2en", "la2", k, 3, batch=12, tag="b"))
    return jobs, {"rest_left": len(rest)}


def p8(tag="", mem=11000):
    """prereg_v7 amendment 4 (2026-11-02 07:00 UTC): a third user's job holds ~52 GB of every card, leaving ~13-18 GB under
    the 120 GB rule, so every remaining Thinker decode runs on all eight cards (Thinker split ~9 GiB per card, batch 12),
    one job at a time, on the windows its system does not have yet (todo_p8.windows.jsonl, output hyp_s8.jsonl).
    Order: the confirmatory remainders, then the exploratory decodes in prereg order, then the reference QE."""
    import glob as _g
    jobs = []
    def rest(name, win):
        d = Path(OUT) / name
        done = set()
        for f in _g.glob(str(d / "hyp_s*.jsonl")):
            for l in open(f, encoding="utf-8"):
                try:
                    done.add(json.loads(l)["id"])
                except ValueError:
                    pass
        left = [l for l in open(win, encoding="utf-8") if json.loads(l)["id"] not in done]
        with open(d / f"todo_p8{tag}.windows.jsonl", "x") as fh:
            fh.writelines(left)
        return str(d / f"todo_p8{tag}.windows.jsonl"), len(left)
    def j8(name, th, win, d, policy, extra=""):
        w, n = rest(name, win)
        return {"id": f"p8{tag}_{name}", "gpus": 8, "mem": mem, "kind": "decode", "deps": [], "left": n,
                "cmd": f"{PRE}.gstar_si_stream_online_v6s --thinker {T[th]} --windows {w} --out {OUT}/{name}/hyp_s8{tag}.jsonl "
                       f"--shard 0 --num-shards 1 --policy {policy} --window 1 --post-key lowlat --dir {d} --batch 12 "
                       f"--max-gib 9 {extra}> {OUT}/{name}/infer_s8{tag}.log 2>&1"}
    sub = "data/online_v6_20261031/windows_test_ref_en2zh_stream_sub.jsonl"
    jobs.append(j8("on_ali_S2dl_rest", "S2DLZ", f"{OUT}/on_ali_S2dl_rest/todo.windows.jsonl", "zh2en", "wait"))
    jobs.append(j8("zh2en_S2dlZ_seed2_d1", "S2DLZ2", REF["zh2en"], "zh2en", "wait"))
    jobs.append(j8("on_ali_O2_LA2_rest", "O2", f"{OUT}/on_ali_O2_LA2_rest/todo.windows.jsonl", "zh2en", "la2"))
    jobs.append(j8("en2zh_S2dl_LA2_rest", "S2DL", f"{OUT}/en2zh_S2dl_LA2_rest/todo.windows.jsonl", "en2zh", "la2"))
    jobs.append(j8("zh2en_S2dlZ_LA2_d1", "S2DLZ", REF["zh2en"], "zh2en", "la2"))
    for d in ("en2zh", "zh2en"):
        jobs.append(j8(f"{d}_SC_seg", "O2N", f"{V6}/{d}_SC-segments/segments.jsonl", d, "la2", "--zero-sd "))
    for L in (1, 2):
        jobs.append(j8(f"en2zh_S2dl_lag{L}_sub", "S2DL", sub, "en2zh", "lag", f"--lag-ms {1000 * L} "))
    py = f"CUDA_VISIBLE_DEVICES=$CVD PYTHONPATH=. {PYM} reports/gstar_si_v7/qe_gemba.py"
    jobs.append({"id": f"p8{tag}_qe_all", "gpus": 8, "mem": mem, "kind": "decode", "deps": [],
                 "cmd": f"{py} --max-gib 9 --batch 32 --out {OUT}/qe/ali_ref_gemba_p8{tag}.jsonl > {OUT}/qe/p8{tag}.log 2>&1"})
    jobs.append(j8("ali_S2dlZ_seed2_d1", "S2DLZ2", REF["ali"], "zh2en", "wait"))
    jobs.append(j8("ali_S2dlZ_LA2_d1", "S2DLZ", REF["ali"], "zh2en", "la2"))
    info = {j["id"]: j.pop("left", None) for j in jobs}
    return jobs, info


def p8b():
    """As p8, declared at 17,000 MiB per card (the measured footprint; at 11,000 the queue placed the first job where it
    took cards to 122-124 GB and the watchdog stopped it, 2026-11-02 07:01 UTC), so a job starts only where it fits."""
    return p8("b", 17000)


def p9():
    """prereg_v7 amendment 7 (2026-11-02 14:00 UTC, other users gone): the remaining decodes on ONE card each with one
    Thinker copy shared by W = 5 decode workers (gstar_multi_v6, as v6: ~72 GB + ~7 GB per worker at batch 8), on cards
    0, 1, 2, 6 only, so that cards 3, 4, 5, 7 drain for the seed-3 training. Output hyp_sm<k>.jsonl, windows not yet
    decoded by the system (todo_p9.windows.jsonl)."""
    import glob as _g
    W, jobs = 5, []
    def rest(name, win):
        d = Path(OUT) / name
        d.mkdir(parents=True, exist_ok=True)
        done = set()
        for f in _g.glob(str(d / "hyp_s*.jsonl")):
            for l in open(f, encoding="utf-8"):
                try:
                    done.add(json.loads(l)["id"])
                except ValueError:
                    pass
        left = [l for l in open(win, encoding="utf-8") if json.loads(l)["id"] not in done]
        with open(d / "todo_p9.windows.jsonl", "x") as fh:
            fh.writelines(left)
        return str(d / "todo_p9.windows.jsonl"), len(left)
    def multi(name, th, win, d, policy, extra=(), needs=None, tpath=None):
        w, n = rest(name, win)
        tp = tpath or T[th]
        runs = [{"module": "gstar_si_stream_online_v6",
                 "argv": ["--thinker", tp, "--windows", w, "--out", f"{OUT}/{name}/hyp_sm{k}.jsonl", "--shard", str(k),
                          "--num-shards", str(W), "--policy", policy, "--window", "1", "--post-key", "lowlat", "--dir", d,
                          "--batch", "8"] + list(extra),
                 "log": f"{OUT}/{name}/infer_sm{k}.log"} for k in range(W)]
        f = JDIR / "p9" / f"{name}.json"
        f.parent.mkdir(parents=True, exist_ok=True)
        with open(f, "x") as fh:
            json.dump(runs, fh, indent=1)
        j = {"id": f"p9_{name}", "gpus": 1, "mem": 72000 + 7000 * W, "kind": "decode", "deps": [], "cards": [0, 1, 2, 6],
             "left": n, "cmd": f"CUDA_VISIBLE_DEVICES=$GPU0 PYTHONPATH=.:third_party/Megatron-LM {PYM} -m "
                               f"experiments.uniss_qwen3_omni_v1.evaluation.gstar_multi_v6 --thinker {tp} {f}"}
        if needs:
            j["needs"] = [needs]
        return j
    sub = "data/online_v6_20261031/windows_test_ref_en2zh_stream_sub.jsonl"
    for L in (1, 2):
        jobs.append(multi(f"en2zh_S2dl_lag{L}_sub", "S2DL", sub, "en2zh", "lag", ["--lag-ms", str(1000 * L)]))
    jobs.append(multi("ali_S2dlZ_seed2_d1", "S2DLZ2", REF["ali"], "zh2en", "wait"))
    jobs.append(multi("ali_S2dlZ_LA2_d1", "S2DLZ", REF["ali"], "zh2en", "la2"))
    t3 = f"{EV}/thinker_gstar_s2dl_seed20261102r/thinker_merged"
    j = multi("en2zh_S2dl_seed3_d1", None, REF["en2zh"], "en2zh", "wait", needs=f"{t3}/sd_proj.pt", tpath=t3)
    j["cards"] = list(range(8))
    jobs.append(j)
    py = f"CUDA_VISIBLE_DEVICES=$GPU0 PYTHONPATH=. {PYM} reports/gstar_si_v7/qe_gemba.py"
    jobs.append({"id": "p9_qe_all", "gpus": 1, "mem": 90000, "kind": "decode", "deps": [], "cards": [0, 1, 2, 6],
                 "cmd": f"{py} --max-gib 100 --batch 64 --out {OUT}/qe/ali_ref_gemba_p9.jsonl > {OUT}/qe/p9.log 2>&1"})
    info = {j["id"]: j.pop("left", None) for j in jobs}
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

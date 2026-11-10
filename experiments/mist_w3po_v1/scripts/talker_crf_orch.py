#!/usr/bin/env python
"""Talker schemes PL-EM / alignment-CRF talker (reports/mist_w3po_v1/talker_crf_prereg_v1.zh-CN.md; user go
2026-11-10 22:35 UTC): one detached orchestrator per stage, on crf_pair_dev_orch.PairOrch (placement, gpu_launch.lock,
<= 135 GiB per card, talker pair gap, headroom stop), with two additions (YieldOrch):

  * the no-checkpoint workload (scripts/gpu_load/train_mist_workload.py) counts as free memory; right before one of our
    jobs starts on a card, that card's workload processes (worker trees, matched on the program field) are stopped by
    PID and their memory is waited out (the guard keep_cards_busy.sh / idle loops re-arm it once the card idles again);
  * <run>/exclude_cards (optional, read every tick: card indices) keeps jobs off cards a training run holds.

Stages (--stage):
  traindec  the v2' finalpost decode (crf_pair_dev_orch.dec_cmd verbatim) of the train subset windows
            (windows_zh2en_c2g06_train16_v1, 8 shards by meeting) -> merged hyp.jsonl (score_run.merge; no scoring on
            train) -> talker prep exactly as crf_pair_dev_orch.speech's sprep (+ drop_long_streams 400)

  sample    candidate sampling (si_infer --cand-speeds 0,2 --cand-k 4 --dump-cands --cand-cover on A''s talker
            settings c1+b1+s2, T1c, default seed): dev5 (A''s talker input on the 5 precheck meetings) and the train
            subset, 8 shards each; 4 CTC scorer servers as jobs (ports SCORER_PORT0..+3; each talker shard waits for
            its port), stopped (stop file) once every other job is over; dev5's own output (= the A' rule path) goes
            through the speech chain (speechonly_part -> 2 ASR sub-shards -> asr_merge -> asr_hyp -> score_run --asr)
            in decode/plcrf_dev5_aprule (A''s dev5 hyp lines)

    python talker_crf_orch.py --stage traindec --run logs/mist_w3po_v1/talker_crf_v1/traindec [--dry]
"""
from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import crf_pair_dev_orch as cp  # noqa: E402
import finalpost_dev_orch as fo  # noqa: E402
from finalpost_dev_orch import DEC, ENV, GUARD, M, PY, PYE, REPO, SPD  # noqa: E402

WIN_TRAIN = "eval_outputs/mist_w3po_v1/data/windows_zh2en_c2g06_train16_v1/windows_train_online.jsonl"
TRAIN_NAME = "train16_crf_v2p_s3000_finalpost_c2g06"
CKPT = f"{cp.TRAIN}/crf_v2p_20261110T075131Z/ckpt/{cp.STEP}"
WORKLOAD = "train_mist_workload.py"


def workload_pids() -> dict[int, list[int]]:
    """card -> PIDs of no-checkpoint workload processes holding memory there (nvidia-smi compute apps whose command
    line runs train_mist_workload.py)."""
    q = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"], capture_output=True,
                       text=True, timeout=60).stdout
    u2c = {u.strip(): int(i) for i, u in (l.split(",") for l in q.strip().split("\n") if l.strip())}
    a = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,gpu_uuid", "--format=csv,noheader"],
                       capture_output=True, text=True, timeout=60).stdout
    out: dict[int, list[int]] = {}
    for l in a.strip().split("\n"):
        p = [x.strip() for x in l.split(",")]
        if len(p) != 2 or not p[0].isdigit() or p[1] not in u2c:
            continue
        try:
            cmd = open(f"/proc/{p[0]}/cmdline", "rb").read().replace(b"\0", b" ").decode(errors="replace")
        except OSError:
            continue
        if WORKLOAD in cmd:
            out.setdefault(u2c[p[1]], []).append(int(p[0]))
    return out


def tree(pid: int) -> list[int]:
    """pid and its descendants (ps fields, no pattern match)."""
    rows = subprocess.run(["ps", "-eo", "pid=,ppid="], capture_output=True, text=True).stdout.split("\n")
    par = {}
    for r in rows:
        f = r.split()
        if len(f) == 2:
            par[int(f[0])] = int(f[1])
    keep, changed = {pid}, True
    while changed:
        changed = False
        for p, q in par.items():
            if p not in keep and q in keep:
                keep.add(p)
                changed = True
    return sorted(keep)


class YieldOrch(cp.PairOrch):
    def snapshot(self) -> dict:
        snap = super().snapshot()
        wl = workload_pids()
        self.wl = wl
        if wl:   # the workload's memory counts as free (it is stopped right before our job starts on that card)
            a = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                               capture_output=True, text=True, timeout=60).stdout
            mem = {}
            for l in a.strip().split("\n"):
                p = [x.strip() for x in l.split(",")]
                if len(p) == 2 and p[0].isdigit() and p[1].replace(".", "").isdigit():
                    mem[int(p[0])] = float(p[1]) / 1024
            for c, pids in wl.items():
                w = sum(mem.get(p, 0.0) for p in pids)
                snap["cards"][c]["used"] -= w
                snap["other"][c] -= w
                snap["other_plan"][c] = max(0.0, snap["other_plan"][c] - w)
        ex = self.excluded()
        for c in ex:   # a training run's cards: no room for anything
            snap["other_plan"][c] = snap["cards"][c]["total"]
        return snap

    def excluded(self) -> set[int]:
        f = self.run / "exclude_cards"
        if not f.exists():
            return set()
        return {int(x) for x in f.read_text().split() if x.strip().isdigit()}

    def status(self, snap):
        """every tick: once every job but the scorer servers is over, the stop file ends the scorers."""
        sc = [j for j in self.jobs if j.id in SCORER_IDS]
        if sc and all(j.state in ("ok", "fail", "skip") for j in self.jobs if j.id not in SCORER_IDS):
            f = self.run / "scorers_stop"
            if not f.exists():
                f.write_text("1\n")
                self.log("all non-scorer jobs over: scorers_stop")
        return super().status(snap)

    def start(self, j, card):
        if card is not None:
            pids = workload_pids().get(card, [])
            if pids:
                victims = sorted({q for p in pids for q in tree(p)})
                self.log(f"card {card}: stop no-checkpoint workload {pids} (tree {victims}) for {j.id}")
                for p in victims:
                    try:
                        os.kill(p, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                for _ in range(60):
                    time.sleep(1)
                    if not workload_pids().get(card):
                        break
                for p in workload_pids().get(card, []):
                    try:
                        os.kill(p, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                time.sleep(3)
        return super().start(j, card)


def build_traindec() -> list[fo.Job]:
    jobs: list[fo.Job] = []

    def add(*a, **k):
        jobs.append(fo.Job(*a, **k))
        jobs[-1].order = len(jobs)

    cp.WIN = WIN_TRAIN
    D, SP = f"{DEC}/{TRAIN_NAME}", f"{SPD}/{TRAIN_NAME}"
    for k in fo.SHARDS_BIG_FIRST:
        def cmd(att, k=k):
            return (cp.dec_cmd(CKPT, f"{D}/hyp_s{k}_a{att}.jsonl", "${PREV:+--resume-from $PREV} ",
                               f"mkdir -p {D} && PREV=$(ls {D}/hyp_s{k}_a*.jsonl 2>/dev/null); ")
                    + f"--shard {k} --num-shards 8 --shard-by meeting {GUARD} > {D}/infer_s{k}_a{att}.log 2>&1 "
                      f"&& grep -q W3PO_ONLINE_DONE {D}/infer_s{k}_a{att}.log")
        add(f"{TRAIN_NAME}__dec{k}", "dec", 1, [], cmd, retries=2)
    add(f"{TRAIN_NAME}__merge", "cpu", 1, [f"{TRAIN_NAME}__dec{k}" for k in range(8)],
        lambda att: ENV + f"{PY} -c \"import json; from pathlib import Path; "
                          f"from experiments.mist_w3po_v1.evaluation.score_run import merge; "
                          f"print(merge(Path('{D}'), [json.loads(l) for l in open('{WIN_TRAIN}')]))\" "
                          f"> {D}/merge.log 2>&1")
    add(f"{TRAIN_NAME}__sprep", "cpu", 1, [f"{TRAIN_NAME}__merge"],
        lambda att: ENV + f"mkdir -p {SP}/talker && GSTAR_DATA=data/ami_multispk_si_v1 {PYE} -m "
                          f"{M}.gstar_talker_v6 prep --hyp {D}/hyp.jsonl --windows {WIN_TRAIN} --causal --blocks "
                          f"--preset lowlat --session zh_mix_v8 --diar nemotron_lowlat_zh8 --sep ' ' --out "
                          f"{SP}/talker_in_all > {SP}/prep.log 2>&1 && python3 reports/gstar_si_v7/drop_long_streams.py "
                          f"{SP}/talker_in_all {SP}/talker_in 400 >> {SP}/prep.log 2>&1")
    return jobs


A_PRIME = "dev_crf_v2p_s3000_finalpost_s8_c2g06"
WIN_DEV = "eval_outputs/mist_w3po_v1/data/windows_zh2en_c2g06_v1/windows_dev_online.jsonl"
WIN_DEV5 = "eval_outputs/mist_w3po_v1/data/windows_zh2en_c2g06_v1_dev5/windows_dev_online.jsonl"
DEV5 = "plcrf_dev5_cands"            # speech dir: talker_in (A''s, 5 meetings), talker (the sampling run)
DEV5_DEC = "plcrf_dev5_aprule"       # decode dir: A''s hyp lines of the 5 meetings + the A' rule path's speech score
SCORER_PORT0 = 18771
N_SCORERS = 4
SAMPLE_FLAGS = ("--speed-policy queue --history-ban 8 --max-model-len 32768 --gpu-mem 0.1 --speed-shift 2 "
                "--select-k 2 --cand-speeds 0,2 --cand-k 4 --dump-cands --cand-cover")


def wait_port(port: int) -> str:
    return f"for i in $(seq 1 180); do (echo > /dev/tcp/127.0.0.1/{port}) 2>/dev/null && break; sleep 5; done; "


def talk_cmd(model: str, sp: str, out: str, k: int, flags: str, port: int | None, seed=None, tin: str = "talker_in") -> str:
    """crf_pair_dev_orch.speech's talker command (same model-independent flags) with this scheme's flags."""
    sd = f" --seed {seed}" if seed else ""
    url = f" --scorer-url http://127.0.0.1:{port}/score" if port else ""
    return (ENV + (wait_port(port) if port else "") + f"VLLM_USE_FLASHINFER_SAMPLER=0 OMP_NUM_THREADS=16 {fo.PYV} -m "
            f"{M}.si_infer --model {model} --direction zh2en --window 1 --output {sp}/{out} --text-policy file "
            f"--blocks-file {sp}/{tin}/blocks.jsonl --items-file {sp}/{tin}/items.jsonl {flags}{url} "
            f"--shard {k} --num-shards 8{sd} > {sp}/{out}_talk{k}.log 2>&1")


def asr_chain(add, name: str, D: str, SP: str, out: str, prio: int, talk_ids: list[str], win: str) -> None:
    """crf_pair_dev_orch.speech after the talkers, verbatim commands, for talker dir SP/out and decode dir D."""
    T = f"{SP}/{out}"
    parts = []
    for k in range(8):
        add(f"{name}__asrprep{k}", "cpu", prio, [talk_ids[k]],
            lambda att, k=k: ENV + f"[ -s {T}/part_{k}.jsonl ] && {PYE} experiments/mist_w3po_v1/evaluation/"
                                   f"speechonly_part.py {T} {k} > {T}_asrprep{k}.log 2>&1")
        for s_ in range(2):
            parts.append(f"{name}__asrp{k}_{s_}")
            add(parts[-1], "asr", prio, [f"{name}__asrprep{k}"],
                lambda att, k=k, s_=s_: ENV + f"{PYE} -m evaluation.asr_transcribe --input {T}/results_speechonly.t{k}.jsonl "
                                              f"--output {T}/asr_speech.t{k}s{s_}.jsonl --device cuda:0 --batch-size 8 "
                                              f"--resume --num-shards 2 --shard-index {s_} >> {T}_asr_t{k}s{s_}.log 2>&1",
                retries=3)
    add(f"{name}__asrprep", "cpu", prio, talk_ids,
        lambda att: ENV + f"for k in 0 1 2 3 4 5 6 7; do [ -s {T}/part_$k.jsonl ] || MISSING=1; done; "
                          f"[ -z \"$MISSING\" ] && cat {T}/part_*.jsonl > {T}/results.jsonl && "
                          f"{PYE} scripts/gpu_queue/p80_speechonly.py {T} > {T}_asrprep.log 2>&1")
    add(f"{name}__asr", "cpu", prio, parts + [f"{name}__asrprep"],
        lambda att: ENV + f"{PY} -m experiments.mist_w3po_v1.evaluation.asr_merge {T} 16 "
                          f"'asr_speech.t*s*.jsonl' > {T}_asr_merge.log 2>&1")
    add(f"{name}__sscore", "cpu", prio, [f"{name}__asr"],
        lambda att: ENV + f"{PY} -m experiments.mist_w3po_v1.evaluation.asr_hyp {D}/hyp.jsonl "
                          f"{T}/asr_speech_merged.jsonl {D}/hyp_asr.jsonl && {PY} -m "
                          f"experiments.mist_w3po_v1.evaluation.score_run --run {D} --windows {win} --ref-windows "
                          f"{fo.REF_WIN} --set zh2en --asr {D}/hyp_asr.jsonl > {D}/score_asr.log 2>&1")


def scorer_jobs(add, run: str) -> list[str]:
    ids = []
    for i in range(N_SCORERS):
        port = SCORER_PORT0 + i
        ids.append(f"scorer{i}")
        add(ids[-1], "asr", 0, [],
            lambda att, port=port: ENV + f"{PY} -m experiments.mist_w3po_v1.evaluation.si_ctc_scorer --port {port} "
                                         f"> {run}/scorer_{port}.a{att}.log 2>&1 & SP=$!; "
                                         f"while [ ! -e {run}/scorers_stop ] && kill -0 $SP 2>/dev/null; do sleep 15; done; "
                                         f"kill $SP 2>/dev/null; wait $SP; [ -e {run}/scorers_stop ]", retries=3)
    return ids


def build_sample(run: str = "") -> list[fo.Job]:
    jobs: list[fo.Job] = []

    def add(*a, **k):
        jobs.append(fo.Job(*a, **k))
        jobs[-1].order = len(jobs)

    scorer_jobs(add, run)
    D5, S5 = f"{DEC}/{DEV5_DEC}", f"{SPD}/{DEV5}"
    add("dev5_hyp", "cpu", 1, [],
        lambda att: ENV + f"mkdir -p {D5} && {PY} -c \"import json; W={{json.loads(l)['id'] for l in open('{WIN_DEV5}')}}; "
                          f"rows=[l for l in open('{DEC}/{A_PRIME}/hyp.jsonl') if json.loads(l)['id'] in W]; "
                          f"assert len(rows)==len(W); open('{D5}/hyp.jsonl','x').writelines(rows)\" > {D5}/hyp.log 2>&1")
    talks = []
    for k in range(8):
        talks.append(f"dev5__talk{k}")
        add(talks[-1], "talk", 1, [],
            lambda att, k=k: talk_cmd(T1C_PATH, S5, "talker", k, SAMPLE_FLAGS, SCORER_PORT0 + k % N_SCORERS), retries=5)
    asr_chain(add, "dev5", D5, S5, "talker", 1, talks, WIN_DEV5)
    jobs[[j.id for j in jobs].index("dev5__sscore")].deps.append("dev5_hyp")
    ST = f"{SPD}/{TRAIN_NAME}"
    for k in range(8):
        add(f"train__talk{k}", "talk", 2, [],
            lambda att, k=k: talk_cmd(T1C_PATH, ST, "talker_cands", k, SAMPLE_FLAGS, SCORER_PORT0 + k % N_SCORERS),
            retries=5)
    return jobs


T1C_PATH = fo.T1C
STAGES = {"traindec": (lambda run="": build_traindec(), lambda: [f"{DEC}/{TRAIN_NAME}", f"{SPD}/{TRAIN_NAME}"]),
          "sample": (build_sample, lambda: [f"{DEC}/{DEV5_DEC}", f"{SPD}/{DEV5}/talker",
                                            f"{SPD}/{TRAIN_NAME}/talker_cands"])}
SCORER_IDS = {f"scorer{i}" for i in range(N_SCORERS)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", required=True, choices=sorted(STAGES))
    ap.add_argument("--run", required=True)
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()
    os.chdir(REPO)
    cp.CANDS = {}   # PairOrch.loop: no text-score events in these job lists
    build, dirs_of = STAGES[a.stage]
    jobs = build(str(Path(a.run)))
    ids = [j.id for j in jobs]
    assert len(set(ids)) == len(ids)
    for j in jobs:
        for d in j.deps:
            assert d in ids, (j.id, d)
    dirs = dirs_of()
    if a.dry:
        for j in jobs:
            print(j.prio, j.kind, j.id, "deps", len(j.deps), "|", j.cmd(1)[len(ENV):][:600])
        print(len(jobs), "jobs; existing new dirs:", [d for d in dirs if Path(d).exists()] or "none")
        return
    run = Path(a.run)
    if a.resume:
        o = YieldOrch(run, jobs)
        o.adopt()
    else:
        if run.exists():
            raise SystemExit(f"拒绝覆盖已有输出: {run}")
        ex = [d for d in dirs if Path(d).exists()]
        if ex:
            raise SystemExit(f"拒绝覆盖已有输出: {ex}")
        run.mkdir(parents=True)
        o = YieldOrch(run, jobs)
    o.spread = True
    o.log(f"stage {a.stage}; {len(jobs)} jobs")
    try:
        o.loop()
    except Exception as e:  # noqa: BLE001
        import traceback
        o.log("CRASH " + traceback.format_exc().replace("\n", " | "))
        o.alert(f"ORCHESTRATOR CRASH {e!r}; running jobs continue (their rc files land in state/)")
        raise


if __name__ == "__main__":
    main()

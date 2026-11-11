#!/usr/bin/env python
"""talker_crf_prereg_v3 (reports/mist_w3po_v1/talker_crf_prereg_v3.zh-CN.md; user go 2026-11-11 ~07:30 UTC), E2': every
GPU / CPU step on the EVAL cards (4-7, the B-slot eval pool; --exclude 0 1 2 3 is written to <run>/exclude_cards before
the first tick) in ONE detached orchestrator, talker_crf_v2_orch's job shapes verbatim with E2''s names
(talker_crf_orch.YieldOrch: the no-checkpoint workload is stopped by PID right before a job of ours starts on its
card; <= 135 GiB a card; vLLM talker starts <= 1 per GPU pair per 60 s). Own CTC scorer ports 18801-18804 (v1
18771-, v2 18791-), stopped once every other job is over.

  precheck 3  dev5, v1's sampling-run candidates: select-meeting --budget-ref stream_viterbi --drop-1ch 0.5 --sel-stream
              <v1 sel_lattice.json> -> render (BiCodec) -> speech chain in decode/plcrf3_dev5_meeting -> latency
              (speech_latency_lit, current code) of the A' rule path, the per-stream lattice path and the E2' path;
              gap decomposition and the one-seed gate (vs the A' rule path) of the E2' path -> summary2 (same GO line)
  eval plemm2 waits for logs/mist_w3po_v1/talker_crf_v3/export_plemm2 (train_failed_plemm2 ends the wait with a
              failure), then dev, A''s text, both TTS seeds, E2''s inference flags exactly (talker_crf_v2_orch
              FLAGS["plemm"]) -> latency, gap, UTMOS / SIM, gate vs A', paired gate vs PL-EM c1+b1+s2, and (descriptive,
              once talker_crf_v2's orchestrator is DONE and E2's two score files exist) the paired gate vs E2
  summary     eval_summary_v3 -> talker_crf/eval/summary_v4.{md,json}

    setsid nohup python talker_crf_v3_orch.py --run logs/mist_w3po_v1/talker_crf_v3/orch --exclude 0 1 2 3 \
        > logs/mist_w3po_v1/talker_crf_v3/orch.out 2>&1 < /dev/null &
    python talker_crf_v3_orch.py --run <dir> --dry        # print the jobs, launch nothing
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import crf_pair_dev_orch as cp  # noqa: E402
import finalpost_dev_orch as fo  # noqa: E402
import talker_crf_orch as TC  # noqa: E402
import talker_crf_v2_orch as V2  # noqa: E402  (sets TC.SCORER_PORT0 = 18791 on import; overridden below)
from finalpost_dev_orch import DEC, ENV, PY, REPO, SPD  # noqa: E402

TC.SCORER_PORT0 = 18801   # v3's own scorer ports (scorer_jobs / talk commands read the module global at call time)
L3 = "logs/mist_w3po_v1/talker_crf_v3"
V2_DONE = f"{V2.L2}/orch/events/DONE"
PRE = TC.PRE                                                   # v1 precheck (sel_lattice.json, gap_*.json; read only)
PRE3 = "eval_outputs/mist_w3po_v1/talker_crf/precheck3_dev5"
MEET = "plcrf3_dev5_meeting"
A = TC.A_PRIME
EV = TC.EVAL_OUT
QUAL = TC.QUAL
SCHEME = "plemm2"
FLAGS = V2.FLAGS["plemm"]   # E2''s inference = E2's (prereg v3 §1)
PLEM = V2.PLEM
PLEMM = [f"{DEC}/{A}__plemm_c1b1s2", f"{DEC}/{A}__plemm_c1b1s2_tts2"]   # E2 (descriptive pairing)
SMALL_WAIT_S = 25   # 2026-11-11 08:45 (user: launches serialise on gpu_launch.lock): small jobs release it after 25 s


def names() -> list[str]:
    return [f"{A}__{SCHEME}_c1b1s2{s}" for s in ("", "_tts2")]


def build_v3(run: str = "") -> list[fo.Job]:
    jobs: list[fo.Job] = []

    def add(*a, **k):
        jobs.append(fo.Job(*a, **k))
        jobs[-1].order = len(jobs)

    TC.scorer_jobs(add, run)
    # ---- precheck 3 (E2', dev5) ----
    S5, D5 = f"{SPD}/{TC.DEV5}", f"{DEC}/{TC.DEV5_DEC}"
    D, SP = f"{DEC}/{MEET}", f"{SPD}/{MEET}"
    add("pre3_select", "cpu", 1, [],
        lambda att: ENV + f"mkdir -p {PRE3} && {PY} -m experiments.mist_w3po_v1.talker.precheck_plcrf select-meeting "
                          f"--cands '{S5}/talker/cands_*.jsonl' --win {TC.WIN_DEV} --sel-stream {PRE}/sel_lattice.json "
                          f"--budget-ref stream_viterbi --drop-1ch 0.5 --out {PRE3} > {PRE3}/select.log 2>&1")
    add("render_meeting3", "asr", 1, ["pre3_select"],
        lambda att: ENV + f"mkdir -p {SP} && {fo.PYV} -m experiments.mist_w3po_v1.talker.render_selection --run {S5}/talker "
                          f"--sel {PRE3}/sel_meeting.json --out {SP}/talker > {SP}/render.a{att}.log 2>&1", retries=1)
    add("meeting3_hyp", "cpu", 1, ["pre3_select"],
        lambda att: ENV + f"mkdir -p {D} && cp -n {D5}/hyp.jsonl {D5}/score_text.json {D}/")
    TC.asr_chain(add, "meeting3", D, SP, "talker", 1, ["render_meeting3"] * 8, TC.WIN_DEV5)
    jobs[[j.id for j in jobs].index("meeting3__sscore")].deps.append("meeting3_hyp")
    paths = {"aprule": (D5, S5), "lattice": (f"{DEC}/plcrf_dev5_lattice", f"{SPD}/plcrf_dev5_lattice"), "meeting": (D, SP)}
    for key, (Dk, SPk) in paths.items():
        add(f"pre3_lat_{key}", "cpu", 1, ["meeting3__sscore"] if key == "meeting" else ["pre3_select"],
            lambda att, Dk=Dk, SPk=SPk, key=key: ENV + f"{PY} -m experiments.mist_w3po_v1.evaluation.speech_latency_lit "
                                                        f"--run {Dk} --talker {SPk}/talker --win {TC.WIN_DEV5} "
                                                        f"--out {PRE3}/lat_{key}.json > {PRE3}/lat_{key}.log 2>&1")
    add("pre3_gap_meeting", "cpu", 1, ["meeting3__sscore"],
        lambda att: ENV + f"{PY} -m experiments.mist_w3po_v1.evaluation.speech_gap_decomp --run {D} --speech {SP} "
                          f"--win {TC.WIN_DEV5} --out {PRE3}/gap_meeting > {PRE3}/gap_meeting.log 2>&1")
    add("pre3_gate_meeting", "cpu", 1, ["meeting3__sscore"],
        lambda att: ENV + f"{PY} -m experiments.mist_w3po_v1.evaluation.gate_2seed --cand {D} --ref {D5} "
                          f"--out {PRE3}/gate_meeting.json > {PRE3}/gate_meeting.log 2>&1")
    add("pre3_summary", "cpu", 1, ["pre3_lat_aprule", "pre3_lat_lattice", "pre3_lat_meeting", "pre3_gap_meeting",
                                   "pre3_gate_meeting"],
        lambda att: ENV + f"{PY} -m experiments.mist_w3po_v1.talker.precheck_plcrf summary2 --out {PRE3} "
                          f"--dec-aprule {D5} --lat-aprule {PRE3}/lat_aprule.json --gap-aprule {PRE}/gap_aprule.json "
                          f"--dec-lattice {paths['lattice'][0]} --lat-lattice {PRE3}/lat_lattice.json "
                          f"--gap-lattice {PRE}/gap_lattice.json --dec-meeting {D} --lat-meeting {PRE3}/lat_meeting.json "
                          f"--gap-meeting {PRE3}/gap_meeting.json --gate-meeting {PRE3}/gate_meeting.json "
                          f"> {PRE3}/summary2.log 2>&1")
    # ---- the E2' dev eval ----
    tin = f"{SPD}/{A}/talker_in"
    ef = f"{L3}/export_{SCHEME}"
    add(f"wait_{SCHEME}", "cpu", 1, [],
        lambda att: f"until [ -s {ef} ] || [ -e {L3}/train_failed_{SCHEME} ]; do sleep 60; done; "
                    f"[ -s {ef} ] && [ -e \"$(cat {ef})/uniss/model.safetensors\" ]")
    model = f"$(cat {ef})/uniss"
    O = f"{EV}/{SCHEME}"
    for name, sfx, seed in zip(names(), ("", "_tts2"), (None, fo.SEED2)):
        Dn, SPn = f"{DEC}/{name}", f"{SPD}/{name}"
        add(f"{name}__prep", "cpu", 2, [f"wait_{SCHEME}"],
            lambda att, Dn=Dn, SPn=SPn, sfx=sfx: ENV + f"mkdir -p {Dn} {SPn} && cp -n {DEC}/{A}{sfx}/hyp.jsonl "
                                                       f"{DEC}/{A}{sfx}/score_text.json {Dn}/")
        talks = []
        for k in range(8):
            talks.append(f"{name}__talk{k}")
            add(talks[-1], "talk", 2, [f"{name}__prep"],
                lambda att, k=k, SPn=SPn, seed=seed: TC.talk_cmd(
                    model, SPn, "talker", k, f"{TC.EVAL_COMMON} {FLAGS}", TC.SCORER_PORT0 + k % TC.N_SCORERS, seed, tin),
                retries=5)
        TC.asr_chain(add, name, Dn, SPn, "talker", 2, talks, TC.WIN_DEV)
        add(f"{name}__lat", "cpu", 2, [f"{name}__sscore"],
            lambda att, Dn=Dn, SPn=SPn, name=name: ENV + f"mkdir -p {O} && {PY} -m experiments.mist_w3po_v1.evaluation."
                                                         f"speech_latency_lit --run {Dn} --talker {SPn}/talker "
                                                         f"--win {TC.WIN_DEV} --out {O}/lat_{name}.json > {O}/lat_{name}.log 2>&1")
        add(f"{name}__gap", "cpu", 2, [f"{name}__sscore"],
            lambda att, Dn=Dn, SPn=SPn, name=name: ENV + f"mkdir -p {O} && {PY} -m experiments.mist_w3po_v1.evaluation."
                                                         f"speech_gap_decomp --run {Dn} --speech {SPn} "
                                                         f"--win {TC.WIN_DEV} --out {O}/gap_{name} > {O}/gap_{name}.log 2>&1")
        add(f"{name}__qual", "asr", 2, [f"{name}__asrprep"],
            lambda att, SPn=SPn, name=name: ENV + f"{PY} -m experiments.mist_w3po_v1.talker.talker_quality --talker "
                                                  f"{SPn}/talker --items {tin}/items.jsonl --out {QUAL}/{name}/quality.json "
                                                  f"> {SPn}/quality.a{att}.log 2>&1")
    n1, n2 = names()
    add(f"gate_{SCHEME}", "cpu", 2, [f"{n1}__sscore", f"{n2}__sscore"],
        lambda att: ENV + f"mkdir -p {O} && {PY} -m experiments.mist_w3po_v1.evaluation.gate_2seed "
                          f"--cand {DEC}/{n1} {DEC}/{n2} --ref {DEC}/{A} {DEC}/{A}_tts2 "
                          f"--out {O}/gate_c1b1s2.json > {O}/gate_c1b1s2.log 2>&1")
    add(f"gate_vs_plem_{SCHEME}", "cpu", 2, [f"{n1}__sscore", f"{n2}__sscore"],
        lambda att: ENV + f"mkdir -p {O} && {PY} -m experiments.mist_w3po_v1.evaluation.gate_2seed "
                          f"--cand {DEC}/{n1} {DEC}/{n2} --ref {PLEM[0]} {PLEM[1]} "
                          f"--out {O}/gate_vs_plem_c1b1s2.json > {O}/gate_vs_plem_c1b1s2.log 2>&1")
    add(f"gate_vs_plemm_{SCHEME}", "cpu", 2, [f"{n1}__sscore", f"{n2}__sscore"],   # descriptive: vs E2
        lambda att: f"until [ -e {V2_DONE} ]; do sleep 60; done; "
                    + ENV + f"mkdir -p {O} && if [ -s {PLEMM[0]}/score_asr.json ] && [ -s {PLEMM[1]}/score_asr.json ]; then "
                            f"{PY} -m experiments.mist_w3po_v1.evaluation.gate_2seed --cand {DEC}/{n1} {DEC}/{n2} "
                            f"--ref {PLEMM[0]} {PLEMM[1]} --out {O}/gate_vs_plemm_c1b1s2.json > {O}/gate_vs_plemm_c1b1s2.log 2>&1; "
                            f"else echo 'E2 score files missing: no pairing vs E2' > {O}/gate_vs_plemm_c1b1s2.log; fi")
    for j in jobs:   # 2026-11-11 08:45: far below the 12 GiB visibility threshold (scorer 2.4, UTMOS ~3 GiB, render)
        j.small = j.id.startswith(("scorer", "render_")) or j.id.endswith("__qual")
    fin = [j.id for j in jobs if j.id.startswith(("gate_", "pre3_summary")) or j.id.endswith(("__lat", "__gap", "__qual"))]
    add("summary_v4", "cpu", 3, fin,
        lambda att: f"until [ -e {V2_DONE} ]; do sleep 60; done; "
                    + ENV + f"{PY} -m experiments.mist_w3po_v1.talker.eval_summary_v3 --out {EV}/summary_v4 "
                            f"> {L3}/summary_v4.a{att}.log 2>&1")
    return jobs


def new_dirs() -> list[str]:
    return ([PRE3, f"{DEC}/{MEET}", f"{SPD}/{MEET}", f"{EV}/{SCHEME}", f"{EV}/summary_v4.json"]
            + [f"{DEC}/{n}" for n in names()] + [f"{SPD}/{n}" for n in names()] + [f"{QUAL}/{n}" for n in names()])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True)
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--exclude", nargs="*", default=[], help="cards no job may use (written to <run>/exclude_cards)")
    a = ap.parse_args()
    os.chdir(REPO)
    cp.CANDS = {}   # PairOrch.loop: no text-score events in this job list
    jobs = build_v3(str(Path(a.run)))
    ids = [j.id for j in jobs]
    assert len(set(ids)) == len(ids)
    for j in jobs:
        for d in j.deps:
            assert d in ids, (j.id, d)
    if a.dry:
        for j in jobs:
            print(j.prio, j.kind, j.id, "deps", len(j.deps), "|", j.cmd(1)[len(ENV):][:700] if j.cmd(1).startswith(ENV) else j.cmd(1)[:700])
        print(len(jobs), "jobs; existing new dirs:", [d for d in new_dirs() if Path(d).exists()] or "none")
        return
    run = Path(a.run)
    if a.resume:
        o = TC.YieldOrch(run, jobs)
        o.adopt()
    else:
        if run.exists():
            raise SystemExit(f"拒绝覆盖已有输出: {run}")
        ex = [d for d in new_dirs() if Path(d).exists()]
        if ex:
            raise SystemExit(f"拒绝覆盖已有输出: {ex}")
        run.mkdir(parents=True)
        (run / "exclude_cards").write_text(" ".join(a.exclude) + "\n")
        o = TC.YieldOrch(run, jobs)
    o.spread = True
    o.small_wait_s = SMALL_WAIT_S
    o.log(f"talker_crf v3 (E2'); {len(jobs)} jobs; excluded cards {sorted(o.excluded())}; scorer ports "
          f"{TC.SCORER_PORT0}-{TC.SCORER_PORT0 + TC.N_SCORERS - 1}; small jobs hold gpu_launch.lock <= {SMALL_WAIT_S} s")
    try:
        o.loop()
    except Exception as e:  # noqa: BLE001
        import traceback
        o.log("CRASH " + traceback.format_exc().replace("\n", " | "))
        o.alert(f"ORCHESTRATOR CRASH {e!r}; running jobs continue (their rc files land in state/)")
        raise


if __name__ == "__main__":
    main()

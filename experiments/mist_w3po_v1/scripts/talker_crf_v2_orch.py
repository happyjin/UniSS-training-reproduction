#!/usr/bin/env python
"""talker_crf_prereg_v2 (reports/mist_w3po_v1/talker_crf_prereg_v2.zh-CN.md; user go 2026-11-11 ~05:50 UTC): every GPU /
CPU step on the EVAL cards (4-7; --exclude 0 1 2 3 is written to <run>/exclude_cards before the first tick, the
training launcher start_talker_crf_v2.sh holds 0-3) in ONE detached orchestrator (talker_crf_orch.YieldOrch: the
no-checkpoint workload is stopped by PID right before a job of ours starts on its card; <= 135 GiB a card; vLLM talker
starts <= 1 per GPU pair per 60 s; one orchestrator, so the stagger holds across the two experiments). Own CTC scorer
ports 18791-18794 (v1 used 18771-18774), shared by both evals, stopped once every other job is over.

  precheck 2  (experiment 2, dev5 candidates of v1's sampling run) select-meeting -> render the meeting-lattice Viterbi
              path (BiCodec) -> the speech chain (speechonly_part -> 2 ASR sub-shards -> asr_merge -> asr_hyp ->
              score_run --asr) in decode/plcrf2_dev5_meeting -> latency (speech_latency_lit, current code) of the A'
              rule path, the per-stream lattice path (v1's renders) and the meeting path; gap decomposition and gate
              (one seed, vs the A' rule path) of the meeting path -> summary2 (GO line)
  eval bon    (experiment 1) waits for logs/mist_w3po_v1/talker_crf_v2/export_bon (written by the training launcher on
              TALKER_CRF_DONE; train_failed_bon ends the wait with a failure), then dev, A''s text, both TTS seeds,
              PL-EM's inference flags exactly (c1+b1+s2: --speed-policy queue --speed-shift 2 --select-k 2,
              --backlog-token) -> latency, gap, UTMOS / SIM, gate vs A', paired gate vs PL-EM c1+b1+s2
  eval plemm  (experiment 2) the same with --backlog-scope meeting --backlog-win <dev windows> (shared-channel backlog
              token; whole meetings per shard)
  summary     eval_summary_v2 -> talker_crf/eval/summary_v3.{md,json}

    setsid nohup python talker_crf_v2_orch.py --run logs/mist_w3po_v1/talker_crf_v2/orch --exclude 0 1 2 3 \
        > logs/mist_w3po_v1/talker_crf_v2/orch.out 2>&1 < /dev/null &
    python talker_crf_v2_orch.py --run <dir> --dry        # print the jobs, launch nothing
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
from finalpost_dev_orch import DEC, ENV, PY, REPO, SPD  # noqa: E402

TC.SCORER_PORT0 = 18791   # v2's own scorer ports (scorer_jobs / build_* read the module global at call time)
L2 = "logs/mist_w3po_v1/talker_crf_v2"
PRE = TC.PRE                                                   # v1 precheck (sel_lattice.json, gap_*.json; read only)
PRE2 = "eval_outputs/mist_w3po_v1/talker_crf/precheck2_dev5"
MEET = "plcrf2_dev5_meeting"
A = TC.A_PRIME
EV = TC.EVAL_OUT
QUAL = TC.QUAL
FLAGS = {"bon": "--backlog-token --select-k 2",
         "plemm": f"--backlog-token --select-k 2 --backlog-scope meeting --backlog-win {TC.WIN_DEV}"}
PLEM = [f"{DEC}/{A}__plem_c1b1s2", f"{DEC}/{A}__plem_c1b1s2_tts2"]


def names(scheme: str) -> list[str]:
    return [f"{A}__{scheme}_c1b1s2{s}" for s in ("", "_tts2")]


def build_v2(run: str = "") -> list[fo.Job]:
    jobs: list[fo.Job] = []

    def add(*a, **k):
        jobs.append(fo.Job(*a, **k))
        jobs[-1].order = len(jobs)

    TC.scorer_jobs(add, run)
    # ---- precheck 2 (experiment 2, dev5) ----
    S5, D5 = f"{SPD}/{TC.DEV5}", f"{DEC}/{TC.DEV5_DEC}"
    D, SP = f"{DEC}/{MEET}", f"{SPD}/{MEET}"
    add("pre2_select", "cpu", 1, [],
        lambda att: ENV + f"mkdir -p {PRE2} && {PY} -m experiments.mist_w3po_v1.talker.precheck_plcrf select-meeting "
                          f"--cands '{S5}/talker/cands_*.jsonl' --win {TC.WIN_DEV} --sel-stream {PRE}/sel_lattice.json "
                          f"--out {PRE2} > {PRE2}/select.log 2>&1")
    add("render_meeting", "asr", 1, ["pre2_select"],
        lambda att: ENV + f"mkdir -p {SP} && {fo.PYV} -m experiments.mist_w3po_v1.talker.render_selection --run {S5}/talker "
                          f"--sel {PRE2}/sel_meeting.json --out {SP}/talker > {SP}/render.a{att}.log 2>&1", retries=1)
    add("meeting_hyp", "cpu", 1, ["pre2_select"],
        lambda att: ENV + f"mkdir -p {D} && cp -n {D5}/hyp.jsonl {D5}/score_text.json {D}/")
    TC.asr_chain(add, "meeting", D, SP, "talker", 1, ["render_meeting"] * 8, TC.WIN_DEV5)
    jobs[[j.id for j in jobs].index("meeting__sscore")].deps.append("meeting_hyp")
    paths = {"aprule": (D5, S5), "lattice": (f"{DEC}/plcrf_dev5_lattice", f"{SPD}/plcrf_dev5_lattice"), "meeting": (D, SP)}
    for key, (Dk, SPk) in paths.items():
        add(f"pre2_lat_{key}", "cpu", 1, ["meeting__sscore"] if key == "meeting" else ["pre2_select"],
            lambda att, Dk=Dk, SPk=SPk, key=key: ENV + f"{PY} -m experiments.mist_w3po_v1.evaluation.speech_latency_lit "
                                                        f"--run {Dk} --talker {SPk}/talker --win {TC.WIN_DEV5} "
                                                        f"--out {PRE2}/lat_{key}.json > {PRE2}/lat_{key}.log 2>&1")
    add("pre2_gap_meeting", "cpu", 1, ["meeting__sscore"],
        lambda att: ENV + f"{PY} -m experiments.mist_w3po_v1.evaluation.speech_gap_decomp --run {D} --speech {SP} "
                          f"--win {TC.WIN_DEV5} --out {PRE2}/gap_meeting > {PRE2}/gap_meeting.log 2>&1")
    add("pre2_gate_meeting", "cpu", 1, ["meeting__sscore"],
        lambda att: ENV + f"{PY} -m experiments.mist_w3po_v1.evaluation.gate_2seed --cand {D} --ref {D5} "
                          f"--out {PRE2}/gate_meeting.json > {PRE2}/gate_meeting.log 2>&1")
    add("pre2_summary", "cpu", 1, ["pre2_lat_aprule", "pre2_lat_lattice", "pre2_lat_meeting", "pre2_gap_meeting",
                                   "pre2_gate_meeting"],
        lambda att: ENV + f"{PY} -m experiments.mist_w3po_v1.talker.precheck_plcrf summary2 --out {PRE2} "
                          f"--dec-aprule {D5} --lat-aprule {PRE2}/lat_aprule.json --gap-aprule {PRE}/gap_aprule.json "
                          f"--dec-lattice {paths['lattice'][0]} --lat-lattice {PRE2}/lat_lattice.json "
                          f"--gap-lattice {PRE}/gap_lattice.json --dec-meeting {D} --lat-meeting {PRE2}/lat_meeting.json "
                          f"--gap-meeting {PRE2}/gap_meeting.json --gate-meeting {PRE2}/gate_meeting.json "
                          f"> {PRE2}/summary2.log 2>&1")
    # ---- the two dev evals ----
    tin = f"{SPD}/{A}/talker_in"
    for scheme in ("bon", "plemm"):
        ef = f"{L2}/export_{scheme}"
        add(f"wait_{scheme}", "cpu", 1, [],
            lambda att, ef=ef, scheme=scheme: f"until [ -s {ef} ] || [ -e {L2}/train_failed_{scheme} ]; do sleep 60; done; "
                                              f"[ -s {ef} ] && [ -e \"$(cat {ef})/uniss/model.safetensors\" ]")
        model = f"$(cat {ef})/uniss"
        O = f"{EV}/{scheme}"
        for name, sfx, seed in zip(names(scheme), ("", "_tts2"), (None, fo.SEED2)):
            Dn, SPn = f"{DEC}/{name}", f"{SPD}/{name}"
            add(f"{name}__prep", "cpu", 2, [f"wait_{scheme}"],
                lambda att, Dn=Dn, SPn=SPn, sfx=sfx: ENV + f"mkdir -p {Dn} {SPn} && cp -n {DEC}/{A}{sfx}/hyp.jsonl "
                                                           f"{DEC}/{A}{sfx}/score_text.json {Dn}/")
            talks = []
            for k in range(8):
                talks.append(f"{name}__talk{k}")
                add(talks[-1], "talk", 2, [f"{name}__prep"],
                    lambda att, k=k, SPn=SPn, seed=seed, scheme=scheme, model=model: TC.talk_cmd(
                        model, SPn, "talker", k, f"{TC.EVAL_COMMON} {FLAGS[scheme]}", TC.SCORER_PORT0 + k % TC.N_SCORERS,
                        seed, tin), retries=5)
            TC.asr_chain(add, name, Dn, SPn, "talker", 2, talks, TC.WIN_DEV)
            add(f"{name}__lat", "cpu", 2, [f"{name}__sscore"],
                lambda att, Dn=Dn, SPn=SPn, name=name, O=O: ENV + f"mkdir -p {O} && {PY} -m experiments.mist_w3po_v1."
                                                                  f"evaluation.speech_latency_lit --run {Dn} --talker {SPn}/talker "
                                                                  f"--win {TC.WIN_DEV} --out {O}/lat_{name}.json > {O}/lat_{name}.log 2>&1")
            add(f"{name}__gap", "cpu", 2, [f"{name}__sscore"],
                lambda att, Dn=Dn, SPn=SPn, name=name, O=O: ENV + f"mkdir -p {O} && {PY} -m experiments.mist_w3po_v1."
                                                                  f"evaluation.speech_gap_decomp --run {Dn} --speech {SPn} "
                                                                  f"--win {TC.WIN_DEV} --out {O}/gap_{name} > {O}/gap_{name}.log 2>&1")
            add(f"{name}__qual", "asr", 2, [f"{name}__asrprep"],
                lambda att, SPn=SPn, name=name: ENV + f"{PY} -m experiments.mist_w3po_v1.talker.talker_quality --talker "
                                                      f"{SPn}/talker --items {tin}/items.jsonl --out {QUAL}/{name}/quality.json "
                                                      f"> {SPn}/quality.a{att}.log 2>&1")
        n1, n2 = names(scheme)
        add(f"gate_{scheme}", "cpu", 2, [f"{n1}__sscore", f"{n2}__sscore"],
            lambda att, n1=n1, n2=n2, O=O: ENV + f"mkdir -p {O} && {PY} -m experiments.mist_w3po_v1.evaluation.gate_2seed "
                                                 f"--cand {DEC}/{n1} {DEC}/{n2} --ref {DEC}/{A} {DEC}/{A}_tts2 "
                                                 f"--out {O}/gate_c1b1s2.json > {O}/gate_c1b1s2.log 2>&1")
        add(f"gate_vs_plem_{scheme}", "cpu", 2, [f"{n1}__sscore", f"{n2}__sscore"],
            lambda att, n1=n1, n2=n2, O=O: ENV + f"mkdir -p {O} && {PY} -m experiments.mist_w3po_v1.evaluation.gate_2seed "
                                                 f"--cand {DEC}/{n1} {DEC}/{n2} --ref {PLEM[0]} {PLEM[1]} "
                                                 f"--out {O}/gate_vs_plem_c1b1s2.json > {O}/gate_vs_plem_c1b1s2.log 2>&1")
    fin = [j.id for j in jobs if j.id.startswith(("gate_", "pre2_summary")) or j.id.endswith(("__lat", "__gap", "__qual"))]
    add("summary_v3", "cpu", 3, fin,
        lambda att: ENV + f"{PY} -m experiments.mist_w3po_v1.talker.eval_summary_v2 --out {EV}/summary_v3 "
                          f"> {L2}/summary_v3.a{att}.log 2>&1")
    return jobs


def new_dirs() -> list[str]:
    return ([PRE2, f"{DEC}/{MEET}", f"{SPD}/{MEET}", f"{EV}/bon", f"{EV}/plemm", f"{EV}/summary_v3.json"]
            + [f"{DEC}/{n}" for s in ("bon", "plemm") for n in names(s)]
            + [f"{SPD}/{n}" for s in ("bon", "plemm") for n in names(s)]
            + [f"{QUAL}/{n}" for s in ("bon", "plemm") for n in names(s)])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True)
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--exclude", nargs="*", default=[], help="cards no job may use (written to <run>/exclude_cards)")
    a = ap.parse_args()
    os.chdir(REPO)
    cp.CANDS = {}   # PairOrch.loop: no text-score events in this job list
    jobs = build_v2(str(Path(a.run)))
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
    o.log(f"talker_crf v2; {len(jobs)} jobs; excluded cards {sorted(o.excluded())}")
    try:
        o.loop()
    except Exception as e:  # noqa: BLE001
        import traceback
        o.log("CRASH " + traceback.format_exc().replace("\n", " | "))
        o.alert(f"ORCHESTRATOR CRASH {e!r}; running jobs continue (their rc files land in state/)")
        raise


if __name__ == "__main__":
    main()

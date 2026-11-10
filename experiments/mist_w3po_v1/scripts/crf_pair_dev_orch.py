"""Dev eval of two train_crf_sft checkpoints (C0 / C0-fp, step_03000, 2026-11-10) under the final-read causal P protocol
(reports/mist_w3po_v1/finalpost_dev_test.zh-CN.md §1-§3): one detached orchestrator for every GPU / CPU step.
Inference / scoring only. A new file on top of finalpost_dev_orch.py (imported, not changed): the same Job / Orch
machinery, gpu_launch.lock rule, measured footprints (NEED: decode 68, talker 24, ASR 17.5 GiB; PLAN_MARGIN 12 GiB;
<= 135 GiB per card), talker starts <= 1 per GPU pair per 60 s, headroom stop; only the job list and the text flags
differ.

Per candidate X (all NEW dirs; a fresh start refuses existing ones):
  decode/X      crf.decode_crf_ckpt_finalpost --crf-ckpt <run>/ckpt/step_03000 --final-read-post close_detect + the
                exact reference config (queue_v2/jobs/032_dev_v8_ctx0_guard_s8_00.json: v8 merged Thinker,
                windows_dev_online, --policy wait --lag-ms 2000 --context 0 --batch 16 --max-gib 120, 8 shards by
                meeting, --loop-guard-k 6 --loop-guard-maxp 12 --budget-tps 25 --budget-slack 50), then score_run;
                diag_order_attr --only A (check A, CPU) into crf/c0pair_dev_diagA/<key>
  speech        finalpost_dev_orch's chain (prep -> 8 talkers -> 2 ASR sub-shards per shard -> asr_merge -> asr_hyp +
                score_run --asr) for seed 1 (decode/X) and seed 2 (decode/X_tts2, talker --seed 20261109, hyp.jsonl /
                score_text.json copied with cp -n, as mk_tts_seed2_jobs)
Priority: both decodes 1 (C0's 8 shards first, big first, then C0-fp's), diag A 1, speech 2 (C0 first) -> C0's speech
stacks next to C0-fp's decode. Events: <run>/events/{text_done_<key>,text_done,alert_<n>,DONE}.

    setsid nohup python crf_pair_dev_orch.py --run logs/mist_w3po_v1/c0pair_dev > /dev/null 2>&1 < /dev/null &
    python crf_pair_dev_orch.py --run <dir> --dry                 # print the jobs, launch nothing
    python crf_pair_dev_orch.py --run <dir> --smoke [--dry]       # 2 decode jobs (one per candidate, --limit 2) into
                                                                  # crf/c0pair_smoke_<run name>/<key>
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import finalpost_dev_orch as fo  # noqa: E402

from finalpost_dev_orch import DEC, ENV, GUARD, Job, M, PY, PYE, PYV, REF_WIN, REPO, SEED2, SHARDS_BIG_FIRST, SPD, \
    T1C, THINKER, TICK_S, WIN, now  # noqa: E402

TRAIN = "eval_outputs/mist_w3po_v1/crf/train"
STEP = "step_03000"
CANDS = {   # key -> (decode dir name, checkpoint)
    "c0": ("dev_crf_c0_s3000_finalpost_s8", f"{TRAIN}/crf_c0_20261110T022501Z/ckpt/{STEP}"),
    "c0fp": ("dev_crf_c0fp_s3000_finalpost_s8", f"{TRAIN}/crf_c0fp_20261110T022551Z/ckpt/{STEP}"),
}
DIAG = "eval_outputs/mist_w3po_v1/crf/c0pair_dev_diagA"
MOD = "experiments.mist_w3po_v1.crf.decode_crf_ckpt_finalpost"


def dec_cmd(ck: str, out: str, extra: str, prev: str = "") -> str:
    return (ENV + f"{prev}{PY} -m {MOD} --crf-ckpt {ck} --final-read-post close_detect --thinker {THINKER} "
            f"--windows {WIN} --out {out} {extra}--policy wait --lag-ms 2000 --context 0 --dir zh2en --batch 16 "
            f"--max-gib 120 ")


def speech(add, name: str, base: str, seed, prio: int, base_dep: list[str]):
    """finalpost_dev_orch.build_jobs.speech, verbatim commands."""
    D, SP = f"{DEC}/{name}", f"{SPD}/{name}"
    pre = (f"mkdir -p {D} && cp -n {DEC}/{base}/hyp.jsonl {DEC}/{base}/score_text.json {D}/ && "
           if name != base else "")
    add(f"{name}__sprep", "cpu", prio, base_dep,
        lambda att: ENV + pre + f"mkdir -p {SP}/talker && GSTAR_DATA=data/ami_multispk_si_v1 {PYE} -m "
                    f"{M}.gstar_talker_v6 prep --hyp {D}/hyp.jsonl --windows {WIN} --causal --blocks --preset lowlat "
                    f"--session zh_mix_v8 --diar nemotron_lowlat_zh8 --sep ' ' --out {SP}/talker_in_all > "
                    f"{SP}/prep.log 2>&1 && python3 reports/gstar_si_v7/drop_long_streams.py {SP}/talker_in_all "
                    f"{SP}/talker_in 400 >> {SP}/prep.log 2>&1")
    sd = f" --seed {seed}" if seed else ""
    parts = []
    for k in range(8):
        add(f"{name}__talk{k}", "talk", prio, [f"{name}__sprep"],
            lambda att, k=k: ENV + f"VLLM_USE_FLASHINFER_SAMPLER=0 OMP_NUM_THREADS=16 {PYV} -m {M}.si_infer "
                                   f"--model {T1C} --direction zh2en --window 1 --output {SP}/talker --text-policy file "
                                   f"--blocks-file {SP}/talker_in/blocks.jsonl --items-file {SP}/talker_in/items.jsonl "
                                   f"--speed-policy fixed --history-ban 8 --max-model-len 32768 --gpu-mem 0.1 "
                                   f"--shard {k} --num-shards 8{sd} > {SP}/talk{k}.log 2>&1", retries=5)
        add(f"{name}__asrprep{k}", "cpu", prio, [f"{name}__talk{k}"],
            lambda att, k=k: ENV + f"[ -s {SP}/talker/part_{k}.jsonl ] && {PYE} experiments/mist_w3po_v1/evaluation/"
                                   f"speechonly_part.py {SP}/talker {k} > {SP}/asrprep{k}.log 2>&1")
        for s in range(2):
            parts.append(f"{name}__asrp{k}_{s}")
            add(parts[-1], "asr", prio, [f"{name}__asrprep{k}"],
                lambda att, k=k, s=s: ENV + f"{PYE} -m evaluation.asr_transcribe --input "
                                            f"{SP}/talker/results_speechonly.t{k}.jsonl --output "
                                            f"{SP}/talker/asr_speech.t{k}s{s}.jsonl --device cuda:0 --batch-size 8 "
                                            f"--resume --num-shards 2 --shard-index {s} >> {SP}/asr_t{k}s{s}.log 2>&1",
                retries=3)
    add(f"{name}__asrprep", "cpu", prio, [f"{name}__talk{k}" for k in range(8)],
        lambda att: ENV + f"for k in 0 1 2 3 4 5 6 7; do [ -s {SP}/talker/part_$k.jsonl ] || MISSING=1; done; "
                          f"[ -z \"$MISSING\" ] && cat {SP}/talker/part_*.jsonl > {SP}/talker/results.jsonl && "
                          f"{PYE} scripts/gpu_queue/p80_speechonly.py {SP}/talker > {SP}/asrprep.log 2>&1")
    add(f"{name}__asr", "cpu", prio, parts + [f"{name}__asrprep"],
        lambda att: ENV + f"{PY} -m experiments.mist_w3po_v1.evaluation.asr_merge {SP}/talker 16 "
                          f"'asr_speech.t*s*.jsonl' > {SP}/asr_merge.log 2>&1")
    add(f"{name}__sscore", "cpu", prio, [f"{name}__asr"],
        lambda att: ENV + f"{PY} -m experiments.mist_w3po_v1.evaluation.asr_hyp {D}/hyp.jsonl "
                          f"{SP}/talker/asr_speech_merged.jsonl {D}/hyp_asr.jsonl && {PY} -m "
                          f"experiments.mist_w3po_v1.evaluation.score_run --run {D} --windows {WIN} --ref-windows "
                          f"{REF_WIN} --set zh2en --asr {D}/hyp_asr.jsonl > {D}/score_asr.log 2>&1")


def build_jobs() -> list[Job]:
    jobs: list[Job] = []

    def add(*a, **k):
        jobs.append(Job(*a, **k))
        jobs[-1].order = len(jobs)

    for key, (name, ck) in CANDS.items():
        D = f"{DEC}/{name}"
        for k in SHARDS_BIG_FIRST:
            def cmd(att, k=k, D=D, ck=ck):
                return (dec_cmd(ck, f"{D}/hyp_s{k}_a{att}.jsonl", "${PREV:+--resume-from $PREV} ",
                                f"mkdir -p {D} && PREV=$(ls {D}/hyp_s{k}_a*.jsonl 2>/dev/null); ")
                        + f"--shard {k} --num-shards 8 --shard-by meeting {GUARD} > {D}/infer_s{k}_a{att}.log 2>&1 "
                          f"&& grep -q W3PO_ONLINE_DONE {D}/infer_s{k}_a{att}.log")
            add(f"{name}__dec{k}", "dec", 1, [], cmd, retries=2)
    for key, (name, ck) in CANDS.items():
        D = f"{DEC}/{name}"
        add(f"{name}__score", "cpu", 1, [f"{name}__dec{k}" for k in range(8)],
            lambda att, D=D: ENV + f"[ -e {D}/score_text.json ] || {PY} -m experiments.mist_w3po_v1.evaluation.score_run "
                                   f"--run {D} --windows {WIN} --ref-windows {REF_WIN} --set zh2en > {D}/score.log 2>&1")
        add(f"{name}__diagA", "cpu", 1, [f"{name}__score"],
            lambda att, D=D, key=key: ENV + f"{PY} -m experiments.mist_w3po_v1.crf.diag_order_attr --decode {D} "
                                             f"--out {DIAG}/{key} --only A > {DIAG}_{key}.log 2>&1")
    for key, (name, ck) in CANDS.items():
        speech(add, name, name, None, 2, [f"{name}__score"])
        speech(add, name + "_tts2", name, SEED2, 2, [f"{name}__score"])
    return jobs


def build_smoke(run: Path) -> tuple[list[Job], list[str]]:
    jobs: list[Job] = []
    out = f"eval_outputs/mist_w3po_v1/crf/c0pair_smoke_{run.name}"
    for key, (name, ck) in CANDS.items():
        D = f"{out}/{key}"
        jobs.append(Job(f"smoke_{key}", "dec", 1, [],
                        lambda att, D=D, ck=ck: dec_cmd(ck, f"{D}/hyp.jsonl", "--limit 2 ", f"mkdir -p {D} && ")
                        + f"{GUARD} > {D}/infer.log 2>&1 && grep -q W3PO_ONLINE_DONE {D}/infer.log"))
        jobs[-1].order = len(jobs)
    return jobs, [out]


def new_dirs() -> list[str]:
    names = [n for n, _ in CANDS.values()]
    return ([f"{DEC}/{n}{s}" for n in names for s in ("", "_tts2")] + [f"{SPD}/{n}{s}" for n in names for s in ("", "_tts2")]
            + [DIAG] + [f"{DIAG}_{k}.log" for k in CANDS])


class PairOrch(fo.Orch):
    def loop(self):
        """finalpost_dev_orch.Orch.loop with the text flags of this job list (text_done_<key>, text_done)."""
        self.log(f"orchestrator pid {os.getpid()} start, {len(self.jobs)} jobs")
        scores = {key: f"{name}__score" for key, (name, _) in CANDS.items() if f"{name}__score" in self.by_id}
        while True:
            snap = self.snapshot()
            self.poll(snap["sid_of"])
            self.headroom(snap)
            for j in self.jobs:
                if j.state == "wait" and self.deps_state(j) == "fail":
                    j.state = "skip"
                    self.alert(f"SKIPPED {j.id}: a dependency failed")
            stop = (self.run / "STOP").exists()
            ready = [j for j in self.jobs if j.state == "wait" and self.deps_state(j) == "ok" and now() >= j.not_before]
            if not stop:
                n_cpu = sum(1 for j in self.jobs if j.state == "run" and not j.gpu)
                for j in [x for x in ready if not x.gpu][: max(0, fo.MAX_CPU - n_cpu)]:
                    self.start(j, None)
                plan = self.plan(snap, [x for x in ready if x.gpu])
                if plan:
                    self.launch_gpu(plan)
            for key, sid in scores.items():
                fl = self.run / "events" / f"text_done_{key}"
                if not fl.exists() and self.by_id[sid].state == "ok":
                    fl.write_text("1\n")
            fl = self.run / "events" / "text_done"
            if scores and not fl.exists() and all(self.by_id[s].state == "ok" for s in scores.values()):
                fl.write_text("1\n")
            self.status(self.snapshot())
            active = [j for j in self.jobs if j.state == "run" or (j.state == "wait" and not stop)]
            if not active:
                bad = [j.id for j in self.jobs if j.state in ("fail", "skip")]
                (self.run / "events" / "DONE").write_text(json.dumps({"failed_or_skipped": bad,
                                                                       "elapsed_min": round((now() - self.t0) / 60, 1)}) + "\n")
                self.log(f"DONE; failed/skipped {bad}")
                return
            time.sleep(TICK_S)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True)
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="2 decode jobs with --limit 2 (one per candidate)")
    ap.add_argument("--resume", action="store_true", help="continue a crashed orchestrator's run dir (adopt its jobs)")
    a = ap.parse_args()
    os.chdir(REPO)
    run = Path(a.run)
    jobs, dirs = build_smoke(run) if a.smoke else (build_jobs(), new_dirs())
    assert len({j.id for j in jobs}) == len(jobs)
    for j in jobs:
        for d in j.deps:
            assert d in {x.id for x in jobs}, (j.id, d)
    if a.dry:
        for j in jobs:
            print(j.prio, j.kind, j.id, "deps", len(j.deps), "|", j.cmd(1)[len(ENV):][:400])
        print(len(jobs), "jobs; new dirs:", [d for d in dirs if Path(d).exists()] or "none exist")
        return
    if a.resume:
        if not (run / "state").is_dir():
            raise SystemExit(f"--resume: no state in {run}")
        o = PairOrch(run, jobs)
        o.adopt()
    else:
        if run.exists():
            raise SystemExit(f"拒绝覆盖已有输出: {run}")
        ex = [d for d in dirs if Path(d).exists()]
        if ex:
            raise SystemExit(f"拒绝覆盖已有输出: {ex}")
        missing = [f"{ck}/{f}" for _, ck in CANDS.values() for f in ("DONE", "lora.pt", "extra.pt")
                   if not Path(f"{ck}/{f}").exists()]
        if missing:
            raise SystemExit(f"checkpoint files missing: {missing}")
        run.mkdir(parents=True)
        o = PairOrch(run, jobs)
    Path(fo.LOCK).parent.mkdir(parents=True, exist_ok=True)
    try:
        o.loop()
    except Exception as e:  # noqa: BLE001
        import traceback
        o.log("CRASH " + traceback.format_exc().replace("\n", " | "))
        o.alert(f"ORCHESTRATOR CRASH {e!r}; running jobs continue (their rc files land in state/)")
        raise


if __name__ == "__main__":
    main()

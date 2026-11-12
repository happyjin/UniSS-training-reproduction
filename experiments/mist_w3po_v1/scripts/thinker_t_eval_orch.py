#!/usr/bin/env python
"""Dev evaluation of the Thinker T experiments (reports/mist_w3po_v1/thinker_crf_t_prereg_v1.zh-CN.md: T1/T2 评测,
T3 评测, T1-ctrl): one detached orchestrator per candidate on talker_crf_orch.YieldOrch (crf_pair_dev_orch placement,
<= 135 GiB per card, talker pair gap, headroom stop; the no-checkpoint workload counts as free memory and is stopped by
PID right before one of our jobs takes its card; <run>/exclude_cards = cards a training holds, read every tick) with the
shared pending ledger (logs/mist_w3po_v1/locks/pending.jsonl, owner "thinker_t").

Candidate = (name, checkpoint). Jobs, A''s dev chain (c2_dev_orch / crf_pair_dev_orch) with only the listed changes:
  dec0..7   crf_pair_dev_orch.dec_cmd (decode_crf_ckpt_finalpost --final-read-post close_detect, v8 merged, policy wait,
            lag 2000, context 0, batch 16, 8 shards by meeting, A''s loop guard) on the c2g06 dev windows, with
            "--window <s>" inserted before --policy wait (default 0.5: reads every 0.5 s)
  score     score_run (text tcpBLEU all / ov)
  speech    (unless --talker none) both TTS seeds, crf_pair_dev_orch.speech's chain (prep -> 8 talkers -> ASR -> score)
            --talker c1b1s2: A''s talker (T1c) with speech_var_orch.talker_cmd's c1+b1+s2 flags
            --talker plem:   the PL-EM export with talker_crf_orch.eval_settings("plem")["c1b1s2"] flags
                             (EVAL_COMMON + --backlog-token --select-k 2) = talker_crf_results_v1's A' + PL-EM setting
            4 CTC scorer servers (b1) as jobs (ports --scorer-port0 ..+3), stopped once every other job is over
  lat       speech_latency_lit --win <c2g06 windows> per seed (per-stream speech LAAL, text LAAL, EndOffset, 1-channel)
  gate      gate_2seed vs each --gate-ref pair (seed 1 dir, seed 2 dir)

    python thinker_t_eval_orch.py --run <new run dir> --name <decode dir name> --ckpt <ckpt dir> \\
        [--window 0.5] [--talker none|c1b1s2|plem] [--gate-ref <d1> <d2>]... [--exclude 4 5 6 7] [--dry]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import crf_pair_dev_orch as cp  # noqa: E402
import finalpost_dev_orch as fo  # noqa: E402
import speech_var_orch as sv  # noqa: E402
import talker_crf_orch as tc  # noqa: E402
from finalpost_dev_orch import DEC, ENV, GUARD, PY, REPO, SPD  # noqa: E402

WIN = "eval_outputs/mist_w3po_v1/data/windows_zh2en_c2g06_v1/windows_dev_online.jsonl"
PLEM = "eval_outputs/mist_w3po_v1/talker_crf/talker_plem_20261111T001759Z/export"
LEDGER = "logs/mist_w3po_v1/locks/pending.jsonl"
OWNER = "thinker_t"
SCORER_PORT0 = 18801
N_SCORERS = 4
PLAIN = False   # --plain-thinker: decode the official v8 merged Thinker without a CRF checkpoint (default off = unchanged)


ONE_DEC = Path("logs/mist_w3po_v1/t_eval/one_decode_cards")   # --cap135: cards back to one decode after repeated OOMs
OOM_LOG = Path("logs/mist_w3po_v1/t_eval/oom_events.jsonl")


def apply_cap135(o, dec_after: list[str]) -> None:
    """--cap135 (user 2026-11-11 ~15:50 UTC: "up to 135 GiB per card is fine"): two decodes per card.
      cap      min(135, total - 5.4) = 135 GiB on the 140.4 GiB cards (was total - 12 = 128.4); headroom stop at 3 GiB
               free (was 8); a decode is planned at 64 GiB (measured 61-67; was 68)
      scorers  our own idle CTC scorer servers count at their real use + 0.5 GiB instead of the ASR footprint (17.5)
      stagger  no decode on a card where any decode (ledger, any owner) started in the last 60 s; 60 s between our own
      OOM      a decode that fails with a CUDA OOM is retried without using up a retry; logged to oom_events.jsonl; a card
               with >= 2 OOMs goes to one_decode_cards (no second decode there while one runs)
      order    --dec-after <run> <name>: our decodes wait until every <name>__dec<k> of that run has started
      per card at most 2 of our own decodes on one card (Orch.plan allowed 1)
    The ledger accounting is unchanged."""
    import time as _t

    fo.PLAN_MARGIN, fo.HEADROOM = 5.4, 3.0
    fo.NEED["dec"] = 64.0
    fo.LAUNCH_GAP_S["dec"] = 60
    orig_free, orig_plan, orig_finish = o.free_plan, o.plan, o.finish

    def free_plan(snap):
        res = orig_free(snap)
        for j in o.jobs:
            if j.state == "run" and j.gpu and j.id.startswith("scorer") and j.card in res:
                act = snap["per_job"].get(j.id, 0.0)
                if act < fo.NEED["asr"]:
                    res[j.card] += fo.NEED["asr"] - act - 0.5
        return res

    def dec_started(run: str, name: str) -> bool:
        st = Path(run) / "state"
        return all(any(st.glob(f"{name}__dec{k}.a*.pid")) for k in range(8))

    def plan_core(snap, ready):
        """finalpost_dev_orch.Orch.plan with at most 2 of our decodes per card (was 1: dec_on); otherwise the same."""
        free = o.free_plan(snap)
        dec_n = {c: sum(1 for j in o.jobs if j.state == "run" and j.kind == "dec" and j.card == c) for c in free}
        reserved, out, pair_used, blocked = set(), [], set(), False
        t = _t.time()
        for j in sorted(ready, key=lambda x: (x.prio, x.order)):
            need = fo.NEED[j.kind]
            cands = [c for c in free if c not in reserved and free[c] >= need and not (j.kind == "dec" and dec_n[c] >= 2)]
            if j.kind == "dec" and blocked:
                continue
            if j.kind == "talk":
                ok = [c for c in cands if c // 2 not in pair_used and t - o.pair_last[c // 2] >= fo.TALK_PAIR_GAP_S]
                if cands and not ok:
                    reserved.add(min(cands, key=lambda c: free[c] - need))
                    blocked = True
                    continue
                cands = ok
            if not cands:
                blocked = True
                continue
            c = max(cands, key=lambda c: free[c]) if j.kind == "dec" else min(cands, key=lambda c: free[c] - need)
            out.append((j, c))
            free[c] -= need
            if j.kind == "dec":
                dec_n[c] += 1
            if j.kind == "talk":
                pair_used.add(c // 2)
            if len(out) >= 8:
                break
        return out

    def plan(snap, ready):
        dec = o.dec_active(ready) if hasattr(o, "dec_active") else True
        out = plan_core(snap, ready) if (dec or not getattr(o, "spread", False)) else orig_plan(snap, ready)
        if not any(j.kind == "dec" for j, _ in out):
            return out
        t = _t.time()
        recent = {r.get("card") for r in fo.ledger_pending(LEDGER) if "__dec" in str(r.get("id"))
                  and t - float(r.get("t", 0)) < 60}
        one = {int(x) for x in ONE_DEC.read_text().split()} if ONE_DEC.exists() else set()
        wait_other = bool(dec_after) and not dec_started(*dec_after)
        keep = []
        for j, c in out:
            if j.kind == "dec":
                if wait_other or c in recent or (c in one and snap["cards"][c]["used"] >= 40.0):
                    continue
                recent.add(c)
            keep.append((j, c))
        return keep

    def finish(j, rc):
        oom = False
        if rc != 0 and j.kind == "dec":
            nm, k = j.id.rsplit("__dec", 1)
            for lf in (o.run / "logs" / f"{j.id}.a{j.attempt}.log", Path(DEC) / nm / f"infer_s{k}_a{j.attempt}.log"):
                try:
                    if lf.exists() and "out of memory" in lf.read_text(errors="ignore")[-20000:].lower() \
                            and lf.stat().st_mtime >= (j.t_start or 0) - 5:
                        oom = True
                        break
                except OSError:
                    pass
        if oom:
            rec = {"utc": _t.strftime("%Y-%m-%dT%H:%M:%SZ", _t.gmtime()), "run": str(o.run), "job": j.id, "card": j.card,
                   "attempt": j.attempt}
            with open(OOM_LOG, "a") as fh:
                fh.write(json.dumps(rec) + "\n")
            n = sum(1 for l in open(OOM_LOG) if json.loads(l).get("card") == j.card)
            o.alert(f"OOM {j.id} a{j.attempt} card {j.card} (card total {n}); retried without using up a retry")
            j.retries += 1
            if n >= 2:
                one = set(ONE_DEC.read_text().split()) if ONE_DEC.exists() else set()
                one.add(str(j.card))
                ONE_DEC.write_text(" ".join(sorted(one)) + "\n")
                o.alert(f"card {j.card}: {n} OOMs -> one decode at a time there (one_decode_cards)")
        orig_finish(j, rc)

    o.free_plan, o.plan, o.finish = free_plan, plan, finish
    o.log(f"--cap135: cap min(135, total - 5.4), headroom 3, decode 64 GiB, scorers at real use, 60 s decode stagger, "
          f"OOM retry; dec-after {dec_after or None} (user ~15:50 UTC: up to 135 GiB per card is fine)")


def decode_with(cmd: str, extra: str) -> str:
    """c2_dev_orch.decode_with: extra online_decode args right before --policy wait."""
    assert cmd.count(" --policy wait ") == 1, cmd
    return cmd.replace(" --policy wait ", f" {extra} --policy wait ", 1)


def build_jobs(run: str, name: str, ckpt: str, window: float, talker: str, gate_refs: list[list[str]],
               speech_from: str = "", shards: int = 8, shard_by: str = "meeting") -> list[fo.Job]:
    """speech_from: an existing decode dir name whose hyp.jsonl / score_text.json are copied (cp -n) into <name> and
    <name>_tts2; no decode / score jobs (another talker on the same text, e.g. T2's text with the PL-EM talker)."""
    cp.WIN = WIN   # read by dec_cmd / speech at call time
    jobs: list[fo.Job] = []

    def add(*a, **k):
        jobs.append(fo.Job(*a, **k))
        jobs[-1].order = len(jobs)

    D = f"{DEC}/{name}"
    if speech_from:
        add(f"{name}__score", "cpu", 1, [],
            lambda att: ENV + f"mkdir -p {D} && cp -n {DEC}/{speech_from}/hyp.jsonl {DEC}/{speech_from}/score_text.json "
                              f"{D}/ && [ -s {D}/hyp.jsonl ] && [ -s {D}/score_text.json ]")
        return speech_jobs(jobs, add, run, name, talker, gate_refs)
    wx = f"--window {window:g}"
    order = cp.SHARDS_BIG_FIRST if (shards, shard_by) == (8, "meeting") else list(range(shards))
    for k in order:
        def cmd(att, k=k):
            c = (cp.dec_cmd(ckpt, f"{D}/hyp_s{k}_a{att}.jsonl", "${PREV:+--resume-from $PREV} ",
                            f"mkdir -p {D} && PREV=$(ls {D}/hyp_s{k}_a*.jsonl 2>/dev/null); ")
                 + f"--shard {k} --num-shards {shards} --shard-by {shard_by} {GUARD} > {D}/infer_s{k}_a{att}.log 2>&1 "
                   f"&& grep -q W3PO_ONLINE_DONE {D}/infer_s{k}_a{att}.log")
            if PLAIN:   # finalpost_dev_orch.decode's command: online_decode_finalpost, no --crf-ckpt
                old = f"{PY} -m {cp.MOD} --crf-ckpt {ckpt} --final-read-post close_detect "
                assert old in c, c
                c = c.replace(old, f"{PY} -m experiments.mist_w3po_v1.evaluation.online_decode_finalpost "
                                   f"--final-read-post close_detect ", 1)
            return decode_with(c, wx) if window != 1.0 else c
        add(f"{name}__dec{k}", "dec", 1, [], cmd, retries=2)
    add(f"{name}__score", "cpu", 1, [f"{name}__dec{k}" for k in range(shards)],
        lambda att: ENV + f"[ -e {D}/score_text.json ] || {PY} -m experiments.mist_w3po_v1.evaluation.score_run "
                          f"--run {D} --windows {WIN} --ref-windows {fo.REF_WIN} --set zh2en > {D}/score.log 2>&1")
    if talker == "none":   # text only: + the unit-level text latency (speech_latency_lit --text-only)
        O = f"eval_outputs/mist_w3po_v1/crf/t_eval/{name}"
        add(f"{name}__textlat", "cpu", 1, [f"{name}__score"],
            lambda att: ENV + f"mkdir -p {O} && {PY} -m experiments.mist_w3po_v1.evaluation.speech_latency_lit --run {D} "
                              f"--text-only --win {WIN} --out {O}/textlat_{name}.json > {O}/textlat_{name}.log 2>&1")
        return jobs
    return speech_jobs(jobs, add, run, name, talker, gate_refs)


def speech_jobs(jobs: list, add, run: str, name: str, talker: str, gate_refs: list[list[str]]) -> list[fo.Job]:
    """scorers, both seeds' speech chains (after <name>__score), latency and gates."""
    tc.SCORER_PORT0, tc.N_SCORERS = SCORER_PORT0, N_SCORERS
    tc.scorer_jobs(add, run)
    names = [name, name + "_tts2"]
    n0 = len(jobs)
    cp.speech(add, names[0], name, None, 2, [f"{name}__score"])
    cp.speech(add, names[1], name, fo.SEED2, 2, [f"{name}__score"])
    for j in jobs[n0:]:
        if j.kind != "talk":
            continue
        nm, k = j.id.rsplit("__talk", 1)
        k = int(k)
        seed = fo.SEED2 if nm.endswith("_tts2") else None
        port = SCORER_PORT0 + k % N_SCORERS
        if talker == "c1b1s2":
            sv.SCORER_PORT0 = SCORER_PORT0
            j.cmd = (lambda att, f=j.cmd, k=k, port=port: ENV + tc.wait_port(port)
                     + sv.talker_cmd(f(att), "c1+b1+s2", k, N_SCORERS)[len(ENV):])
        elif talker == "plem":
            flags, _ = tc.eval_settings("plem", PLEM, 0.0)["c1b1s2"]
            SP = f"{SPD}/{nm}"
            j.cmd = (lambda att, SP=SP, k=k, port=port, seed=seed, flags=flags: tc.talk_cmd(
                f"{PLEM}/uniss", SP, "talker", k, f"{tc.EVAL_COMMON} {flags}", port, seed, f"{SP}/talker_in"))
        else:
            raise ValueError(talker)
    O = f"eval_outputs/mist_w3po_v1/crf/t_eval/{name}"
    for nm in names:
        Dn, SPn = f"{DEC}/{nm}", f"{SPD}/{nm}"
        add(f"{nm}__lat", "cpu", 2, [f"{nm}__sscore"],
            lambda att, Dn=Dn, SPn=SPn, nm=nm: ENV + f"mkdir -p {O} && {PY} -m experiments.mist_w3po_v1.evaluation."
                                                     f"speech_latency_lit --run {Dn} --talker {SPn}/talker --win {WIN} "
                                                     f"--out {O}/lat_{nm}.json > {O}/lat_{nm}.log 2>&1")
    for i, (r1, r2) in enumerate(gate_refs):
        add(f"{name}__gate{i}", "cpu", 2, [f"{nm}__sscore" for nm in names],
            lambda att, r1=r1, r2=r2, i=i: ENV + f"mkdir -p {O} && {PY} -m experiments.mist_w3po_v1.evaluation.gate_2seed "
                                                 f"--cand {DEC}/{names[0]} {DEC}/{names[1]} --ref {DEC}/{r1} {DEC}/{r2} "
                                                 f"--out {O}/gate_{i}_{r1}.json > {O}/gate_{i}.log 2>&1")
    return jobs


def new_dirs(name: str, talker: str) -> list[str]:
    out = [f"{DEC}/{name}"]
    if talker != "none":
        out += [f"{DEC}/{name}_tts2", f"{SPD}/{name}", f"{SPD}/{name}_tts2"]
    return out


def main():
    global SCORER_PORT0, WIN, PLAIN
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--window", type=float, default=0.5)
    ap.add_argument("--talker", choices=["none", "c1b1s2", "plem"], default="c1b1s2")
    ap.add_argument("--gate-ref", nargs=2, action="append", default=[])
    ap.add_argument("--exclude", nargs="*", default=[])
    ap.add_argument("--win", default=WIN, help="online windows (default: the c2g06 dev windows)")
    ap.add_argument("--shards", type=int, default=8)
    ap.add_argument("--shard-by", choices=["meeting", "window"], default="meeting")
    ap.add_argument("--speech-from", default="", help="existing decode dir name: speech only on its text")
    ap.add_argument("--scorer-port0", type=int, default=SCORER_PORT0)
    ap.add_argument("--plain-thinker", action="store_true",
                    help="official v8 merged Thinker, no CRF checkpoint (--ckpt -); default off = unchanged")
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--cap135", action="store_true", help="apply_cap135 (two decodes per card; default off = unchanged)")
    ap.add_argument("--dec-after", nargs=2, default=[], metavar=("RUN", "NAME"),
                    help="--cap135: our decodes wait until every NAME__dec<k> of RUN has started")
    a = ap.parse_args()
    os.chdir(REPO)
    run = Path(a.run)
    SCORER_PORT0, WIN = a.scorer_port0, a.win
    PLAIN = a.plain_thinker
    assert not PLAIN or a.ckpt == "-", "--plain-thinker needs --ckpt -"
    jobs = build_jobs(str(run), a.name, a.ckpt, a.window, a.talker, a.gate_ref, a.speech_from, a.shards, a.shard_by)
    assert len({j.id for j in jobs}) == len(jobs)
    if a.dry:
        for j in jobs:
            print(j.prio, j.kind, j.id, "deps", len(j.deps), "|", j.cmd(1)[len(ENV):][:700])
        print(len(jobs), "jobs")
        return
    for f in (("DONE", "lora.pt", "extra.pt") if not (a.speech_from or PLAIN) else ()):
        if not Path(a.ckpt, f).exists():
            raise SystemExit(f"checkpoint file missing: {a.ckpt}/{f}")
    if a.resume:
        o = tc.YieldOrch(run, jobs)
        o.adopt()
    else:
        if run.exists():
            raise SystemExit(f"拒绝覆盖已有输出: {run}")
        if a.speech_from and not Path(DEC, a.speech_from, "score_text.json").exists():
            raise SystemExit(f"--speech-from {a.speech_from}: no score_text.json yet")
        ex = [d for d in new_dirs(a.name, a.talker) if Path(d).exists()]
        if ex:
            raise SystemExit(f"拒绝覆盖已有输出: {ex}")
        run.mkdir(parents=True)
        (run / "exclude_cards").write_text(" ".join(a.exclude) + "\n")
        o = tc.YieldOrch(run, jobs)
    o.spread = True
    o.ledger, o.ledger_owner = LEDGER, f"{OWNER}_{run.name}"
    if a.cap135:
        apply_cap135(o, list(a.dec_after))   # one owner per orchestrator: they see each other's pending starts
    o.log(f"thinker T eval {a.name}: ckpt {a.ckpt}; window {a.window}; talker {a.talker}; gate refs {a.gate_ref}; "
          f"excluded {sorted(o.excluded())}; ledger {LEDGER} owner {o.ledger_owner}")
    try:
        o.loop()
    except Exception as e:  # noqa: BLE001
        import traceback
        o.log("CRASH " + traceback.format_exc().replace("\n", " | "))
        o.alert(f"ORCHESTRATOR CRASH {e!r}; running jobs continue")
        raise


if __name__ == "__main__":
    main()

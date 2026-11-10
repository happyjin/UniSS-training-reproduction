"""Unit tests for crf_pair_dev_orch.PairOrch.plan --spread (CPU, no process is started; a made-up nvidia-smi snapshot):

    python experiments/mist_w3po_v1/scripts/test_crf_pair_dev_orch.py

  off        spread False -> exactly finalpost_dev_orch.Orch.plan (best-fit: the ASR jobs go to the busiest card that fits)
  dec        spread True but a decode is running / waiting -> exactly Orch.plan
  spread     spread True, no decode: worst-fit, the empty cards 6 / 7 take the jobs in turn
  pair gap   spread True: at most one talker per GPU pair per round and none on a pair started < 60 s ago
  cap        spread True: a card whose plan room (min(135, total - 12) GiB - ours) is below the need is never used
  cands      --cands: default c0pair = the old module values (job ids / commands / dirs unchanged after a round trip
             through v2p); v2p = C0-fp's 82 jobs with only the decode name, checkpoint, diag A dir and key swapped
The orchestrator's run dir (state / logs / events) goes to a new tempfile.mkdtemp dir; nothing is deleted.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import crf_pair_dev_orch as po  # noqa: E402
import finalpost_dev_orch as fo  # noqa: E402

TOTAL = 143771 / 1024


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name, flush=True)
    assert cond, name


def setup(load: dict[int, list[str]], ready_kinds: list[str], spread: bool, dec_wait: bool = False):
    """load: card -> kinds of our running jobs; returns (orch, snapshot, ready jobs)."""
    jobs, n = [], 0

    def job(kind):
        nonlocal n
        n += 1
        j = fo.Job(f"j{n}_{kind}", kind, 2, [], lambda att: "true")
        j.order = n
        jobs.append(j)
        return j

    per_job = {}
    used = {c: 0.0 for c in range(8)}
    for c, kinds in load.items():
        for k in kinds:
            j = job(k)
            j.state, j.card, j.t_start = "run", c, 0.0
            per_job[j.id] = fo.NEED[k]
            used[c] += fo.NEED[k]
    ready = [job(k) for k in ready_kinds]
    if dec_wait:
        j = job("dec")
        j.not_before = fo.now() + 600
    o = po.PairOrch(Path(tempfile.mkdtemp(prefix="test_cpdo_")) / "run", jobs)
    o.spread = spread
    snap = {"cards": {c: {"used": used[c], "total": TOTAL} for c in range(8)}, "per_job": per_job,
            "ours": dict(used), "other": {c: 0.0 for c in range(8)}, "other_plan": {c: 0.0 for c in range(8)}}
    return o, snap, ready


def ids(plan):
    return [(j.id, c) for j, c in plan]


def main():
    busy = {c: ["talk", "asr", "asr", "asr"] for c in range(6)}   # 76.5 GiB on cards 0-5, 6 / 7 empty

    # --- off: identical to finalpost_dev_orch, best-fit
    o, snap, ready = setup(busy, ["asr"] * 4, spread=False)
    p = o.plan(snap, ready)
    check("off: plan == Orch.plan", ids(p) == ids(fo.Orch.plan(o, snap, ready)))
    check("off: best-fit puts ASR on the busy cards (the 07:04 UTC incident)", all(c < 6 for _, c in p))

    # --- spread with a decode running / waiting: unchanged
    o, snap, ready = setup({**busy, 7: ["dec"]}, ["asr"] * 4, spread=True)
    check("dec running: plan == Orch.plan", ids(o.plan(snap, ready)) == ids(fo.Orch.plan(o, snap, ready)))
    o, snap, ready = setup(busy, ["asr"] * 4, spread=True, dec_wait=True)
    check("dec waiting: plan == Orch.plan", ids(o.plan(snap, ready)) == ids(fo.Orch.plan(o, snap, ready)))

    # --- spread: worst-fit
    o, snap, ready = setup(busy, ["asr"] * 4, spread=True)
    p = o.plan(snap, ready)
    check("spread: the 4 ASR jobs go to the empty cards 6 / 7 in turn", [c for _, c in p] == [6, 7, 6, 7])
    o, snap, ready = setup(busy, ["asr"] * 12, spread=True)
    p = o.plan(snap, ready)
    check("spread: at most 8 per round", len(p) == 8)
    room = fo.Orch.free_plan(o, snap)
    after = {c: room[c] - sum(fo.NEED[j.kind] for j, cc in p if cc == c) for c in range(8)}
    check("spread: every card keeps plan room >= 0 (<= min(135, total - 12) GiB)", all(v >= 0 for v in after.values()))

    # --- talker pair gap
    o, snap, ready = setup(busy, ["talk", "talk", "asr"], spread=True)
    t = fo.now()
    o.pair_last = {0: t, 1: t, 2: t, 3: 0.0}
    p = o.plan(snap, ready)
    talks = [c for j, c in p if j.kind == "talk"]
    check("pair gap: one talker, on pair 3", talks == [6])
    check("pair gap: the 2nd talker reserves card 7 (freest), the ASR takes the freest unreserved card 6",
          [(j.kind, c) for j, c in p] == [("talk", 6), ("asr", 6)])
    o.pair_last = {p_: t for p_ in range(4)}
    p = o.plan(snap, ready)
    check("pair gap: no talker on a pair started < 60 s ago", not [1 for j, _ in p if j.kind == "talk"])

    # --- cap
    full = {c: ["talk", "talk", "talk", "asr", "asr"] for c in range(8)}   # 107 GiB each: room 128.4 - 107 = 21.4
    o, snap, ready = setup(full, ["talk", "asr", "asr"], spread=True)
    p = o.plan(snap, ready)
    check("cap: no talker (24 > 21.4 GiB room), one ASR per card at most", not [1 for j, _ in p if j.kind == "talk"]
          and len({c for _, c in p}) == len(p) == 2)
    # --- cands
    def dump():
        return [(j.id, j.kind, j.prio, tuple(j.deps), j.retries, j.cmd(1), j.cmd(2)) for j in po.build_jobs()]
    base = dump()
    base_dirs = po.new_dirs()
    base_smoke = [(j.id, j.cmd(1)) for j in po.build_smoke(Path("x/run1"))[0]]
    check("cands: default set is the module's original CANDS / DIAG / smoke prefix",
          po.CAND_SETS["c0pair"] == (po.CANDS, po.DIAG, po.SMOKE_PREFIX) and len(base) == 164
          and po.DIAG == "eval_outputs/mist_w3po_v1/crf/c0pair_dev_diagA"
          and base_smoke[0][1].count("crf/c0pair_smoke_run1/c0/hyp.jsonl") == 1)
    po.select("v2p")
    v2p = dump()
    v2p_dirs = po.new_dirs()
    v2p_smoke = [(j.id, j.cmd(1)) for j in po.build_smoke(Path("x/run1"))[0]]
    po.select("c0pair")
    check("cands: c0pair after a v2p round trip == before (ids, kinds, deps, commands, dirs, smoke)",
          dump() == base and po.new_dirs() == base_dirs
          and [(j.id, j.cmd(1)) for j in po.build_smoke(Path("x/run1"))[0]] == base_smoke)
    name, ck = po.CAND_SETS["v2p"][0]["v2p"]
    fpn, fpck = po.CANDS["c0fp"]

    def sub(x):
        if isinstance(x, tuple):
            return tuple(sub(y) for y in x)
        if not isinstance(x, str):
            return x
        return (x.replace(fpn, name).replace(fpck, ck).replace("crf/c0pair_dev_diagA/c0fp", "crf/v2p_dev_diagA/v2p")
                .replace("crf/c0pair_dev_diagA_c0fp.log", "crf/v2p_dev_diagA_v2p.log"))
    want = [sub(t) for t in base if t[0].startswith(fpn + "_")]
    check("cands: v2p jobs == C0-fp's jobs with name / ckpt / diag A swapped (82 jobs, same order)",
          len(v2p) == 82 and v2p == want)
    check("cands: v2p checkpoint is crf_v2p_20261110T075131Z step_03000; nothing points at C0 / C0-fp",
          ck == "eval_outputs/mist_w3po_v1/crf/train/crf_v2p_20261110T075131Z/ckpt/step_03000"
          and not any("crf_c0" in str(t) or "c0pair" in str(t) for t in v2p))
    check("cands: v2p new dirs / smoke dir are its own",
          v2p_dirs == [f"{fo.DEC}/{name}", f"{fo.DEC}/{name}_tts2", f"{fo.SPD}/{name}", f"{fo.SPD}/{name}_tts2",
                       "eval_outputs/mist_w3po_v1/crf/v2p_dev_diagA", "eval_outputs/mist_w3po_v1/crf/v2p_dev_diagA_v2p.log"]
          and len(v2p_smoke) == 1 and "crf/v2p_smoke_run1/v2p/hyp.jsonl" in v2p_smoke[0][1])
    print("ALL PASS", flush=True)


if __name__ == "__main__":
    main()

"""Unit tests for crf_pair_dev_orch.PairOrch.plan --spread (CPU, no process is started; a made-up nvidia-smi snapshot):

    python experiments/mist_w3po_v1/scripts/test_crf_pair_dev_orch.py

  off        spread False -> exactly finalpost_dev_orch.Orch.plan (best-fit: the ASR jobs go to the busiest card that fits)
  dec        spread True but a decode is running / waiting -> exactly Orch.plan
  spread     spread True, no decode: worst-fit, the empty cards 6 / 7 take the jobs in turn
  pair gap   spread True: at most one talker per GPU pair per round and none on a pair started < 60 s ago
  cap        spread True: a card whose plan room (min(135, total - 12) GiB - ours) is below the need is never used
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
    print("ALL PASS", flush=True)


if __name__ == "__main__":
    main()

"""Tests of D1 (reports/mist_w3po_v1/thinker_dagger_d1_prereg_v1.zh-CN.md §4.1 R1-R3), CPU only:

  R1  roll-in driver: the read-by-read batch driver (simulate_rounds) == the literal per-window loop (simulate) with
      one prefix query per boundary, under causal fake scorers, on real T4 / T5 training windows; rho~ monotone per
      speaker, the final read writes the rest; never-wait -> everything at read 0, always-wait -> everything at the
      final read; decide() = the first <wait>
  R2  rows: with rho = dart_aug.perturbed_reads(e + delay) the D1 requests equal aug_requests' field for field; with
      learner trajectories that run ahead of the labels: rows exactly at the reads with an overdue item, H u W* =
      H u {e <= r}, W* = the unwritten due items in kappa order, keep filters reads; a row never has an item ahead
      (kappa-prefix histories: ahead and overdue exclude each other)
  R3  trainer step (latent_write_train.run_step, tiny model, world 1): d1 None == no argument == {} bit for bit; with
      D1 rows the base rows are unchanged, loss = base loss + loss_aug, the D1 rows' loss = a grad pass over them
      alone, prefetch 1 == 0; with DART and D1 together both row sets are appended
  M   manifest pieces: cap_rows deterministic, uniform subset of the right size; load_plan = the trainer's plan

    PYTHONPATH=.:third_party/Megatron-LM CUDA_VISIBLE_DEVICES= python -m experiments.mist_w3po_v1.crf.test_d1
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

if not torch.cuda.is_available():
    torch.accelerator.current_accelerator = lambda *a, **k: None

from experiments.mist_w3po_v1.crf import d1_rollin as R
from experiments.mist_w3po_v1.crf import dart_aug as DA
from experiments.mist_w3po_v1.crf import latent_write_crf as LW
from experiments.mist_w3po_v1.crf import latent_write_train as LWT
from experiments.mist_w3po_v1.crf import precheck_serial as S
from experiments.mist_w3po_v1.crf import test_latent_write_crf as TL
from experiments.mist_w3po_v1.crf import test_serial_v2p_train as TV
from experiments.mist_w3po_v1.crf import test_train_crf_sft as TT
from experiments.mist_w3po_v1.crf import train_crf_sft as T

torch.set_num_threads(8)
C = "eval_outputs/mist_w3po_v1/crf"
SCHEDS = [f"{C}/sched_piece_t0_g1000_v1", f"{C}/sched_t5_pieces4_w10_v1"]


def real_recs(sched: str, n: int) -> list[dict]:
    out = []
    for f in sorted(Path(sched).glob("shard_*.jsonl")):
        for l in open(f, encoding="utf-8"):
            r = json.loads(l)
            if r["n_tok"] > 0 and r["n_read"] >= 3 and len(r["chunks"]) >= 2:
                out.append(r)
            if len(out) >= n:
                return out
    return out


def fake(p_wait: float, salt: str):
    """a causal fake model: the argmax at position p is a hash of (window, read, committed[:p])."""
    def arg(rec, r, prefix):
        h = hashlib.sha1(f"{salt}|{rec['id']}|{r}|{prefix}".encode()).digest()
        return R.WAIT_ID if h[0] / 256.0 < p_wait else 1000 + h[1]

    def scorer(rec, r, ids, pos):
        return [arg(rec, r, tuple(ids[:p])) for p in pos]

    def batch(todo):
        return [scorer(rec, r, ids, pos) for rec, r, ids, pos in todo]
    return scorer, batch


def test_r1():
    assert R.decide([5, 7, R.WAIT_ID, 9]) == 2 and R.decide([R.WAIT_ID]) == 0 and R.decide([3, 4]) == 2
    n = 0
    for sched in SCHEDS:
        recs = real_recs(sched, 150)
        for p_wait, salt in ((0.3, "a"), (0.6, "b"), (0.0, "c"), (1.0, "d")):
            sc, bt = fake(p_wait, salt)
            lit = [R.simulate(rec, lambda rec_, r, ids, pos: [sc(rec_, r, ids[:p], [p])[0] for p in pos])
                   for rec in recs]
            bat = R.simulate_rounds(recs, bt)
            for rec, a, b in zip(recs, lit, bat):
                assert a == b, rec["id"]
                DA.check_rho(rec, a["rho"])
                assert sum(a["written"]) == len(rec["chunks"]) and len(a["written"]) == rec["n_read"]
                if p_wait == 0.0:
                    assert set(a["rho"]) == {0}
                if p_wait == 1.0:
                    assert set(a["rho"]) == {rec["n_read"] - 1}
                n += 1
    print(f"  R1 roll-in: batch driver == per-boundary prefix loop on {n} (window, scorer) cases (T4 + T5 data); "
          f"monotone, final read writes the rest, never / always wait limits")


def test_r2():
    rng = np.random.default_rng(3)
    eq = rows = ahead_rows = 0
    for sched in SCHEDS:
        for rec in real_recs(sched, 200):
            e = S.chunk_reads(rec)
            delay = [int(x) if rng.random() < 0.4 else 0 for x in rng.integers(1, 4, len(e))]
            rho = DA.perturbed_reads(rec, delay)
            a, b = DA.aug_requests(rec, delay), DA.aug_requests_rho(rec, rho)
            assert len(a) == len(b)
            for x, y in zip(a, b):
                assert {k: v for k, v in y.items() if k != "ahead"} == x and y["ahead"] == 0
            eq += len(a)
            _, bt = fake(0.35, "r2")
            traj = R.simulate_rounds([rec], bt)[0]["rho"]
            n = len(e)
            got = DA.aug_requests_rho(rec, traj)
            want_r = [r for r in range(rec["n_read"])
                      if any(traj[i] > r - 1 and e[i] <= r - 1 for i in range(n))]
            assert [q["r"] for q in got] == want_r, rec["id"]
            for q in got:
                r = q["r"]
                H = {i for i in range(n) if traj[i] <= r - 1}
                assert set(q["hist"]) == H and set(q["items"]) == {i for i in range(n) if i not in H and e[i] <= r}
                assert q["items"] == [c for c in R.kap_all(rec) if c in set(q["items"])]
                assert q["forced_ids"] == S.ids_of(rec, q["items"]) and q["hist_ids"] == S.ids_of(rec, q["hist"])
                ahead_rows += q["ahead"] > 0
            rows += len(got)
            if got:
                kp = [got[0]["r"]]
                assert [q["r"] for q in DA.aug_requests_rho(rec, traj, keep=kp)] == kp
    # the roll-in's history is a prefix of the kappa order, which sorts by e first: a history holding an item ahead
    # of its label (e > r - 1) holds every item due earlier, so no row (an overdue read) ever has an ahead item
    assert rows > 50 and ahead_rows == 0, (rows, ahead_rows)
    print(f"  R2 rows: rho = perturbed reads -> aug_requests field for field ({eq} rows); learner trajectories: "
          f"{rows} rows exactly at the overdue reads (none with an item ahead: kappa-prefix histories), H u W*, kappa "
          f"order, keep")


def _run(m, wins, D, pf=False, rows_only=None, **kw):
    ctx = {**TV.vctx(perturb=0.2, budget=1500), "final_full_post": True}
    TT.zero(m)
    if rows_only is not None:
        lg = T.grad_pass(m, rows_only, ctx, 3)
        return None, lg, TT.grads(m)
    v = LWT.run_step(m, wins, D, ctx, 3, LW.TAU, 0, 1, prefetch=pf, **kw)
    ctx["prefetched_batches"] = v["batches"]
    lg = T.grad_pass(m, v["rows"], ctx, 3)
    return v, lg, TT.grads(m)


def test_r3():
    base = TV.small_choice_windows(3, max_paths=60)
    wins = [T.make_window(w.idx, TL.widen(w.rec), True) for w in base]
    T.set_reach(wins, 128)
    m = TT.tiny_model(seed=41)
    bias = torch.zeros(151936)
    bias[T.WAIT_ID] = 9.0
    bias[T.EOS_ID] = 6.0
    m.logits_at = lambda h: TT.ThinkerGStarModel.logits_at(m, h) + bias
    D = T.step_totals(wins)[0]
    v0, l0, g0 = _run(m, wins, D)
    for kw in ({"d1": None}, {"d1": {}}, {"mode": "t2", "d1": None}):
        v, lg, g = _run(m, wins, D, **kw)
        assert len(v["rows"]) == len(v0["rows"]) and lg["loss_main"] == l0["loss_main"], kw
        assert all(torch.equal(g0[k], g[k]) for k in g0), kw
    _, bt = fake(0.7, "r3")
    d1 = {}
    for w in wins:
        rho = R.simulate_rounds([w.rec], bt)[0]["rho"]
        d1[w.idx] = (rho, None)
    v1, l1, g1 = _run(m, wins, D, d1=d1)
    v1p, l1p, g1p = _run(m, wins, D, pf=True, d1=d1)
    assert l1["loss_main"] == l1p["loss_main"] and all(torch.equal(g1[k], g1p[k]) for k in g1)
    n0 = len(v0["rows"])
    for a, b in zip(v0["rows"], v1["rows"][:n0]):
        assert a.key == b.key and np.array_equal(a.c_tok, b.c_tok) and np.array_equal(a.c_end, b.c_end)
    extra = v1["rows"][n0:]
    assert len(extra) == v1["stats"]["d1_rows"] > 0 and all(rw.aug and rw.key[0] == "d1" for rw in extra)
    tol = 1e-5 * max(1.0, abs(l1["loss_main"]))
    assert abs(l1["loss_main"] - (l0["loss_main"] + l1["loss_aug"])) < tol
    _, la, _ = _run(m, wins, D, rows_only=extra)
    assert abs(la["loss_main"] - l1["loss_aug"]) < tol
    draws = {w.idx: [2] * len(w.rec["chunks"]) for w in wins}
    v2, l2, _ = _run(m, wins, D, mode="t2", dart=draws, d1=d1)
    kinds = [rw.key[0] for rw in v2["rows"][n0:]]
    assert kinds.count("dart") == v2["stats"]["dart_rows"] > 0 and kinds.count("d1") == v2["stats"]["d1_rows"]
    assert math.isfinite(l2["loss_main"])
    print(f"  R3 step: d1 None == absent == {{}} bit for bit; {len(extra)} D1 rows appended, base rows unchanged, loss "
          f"{l1['loss_main']:.6f} = {l0['loss_main']:.6f} + aug {l1['loss_aug']:.6f} (= their own pass); prefetch "
          f"1 == 0; DART + D1 together: {kinds.count('dart')} + {kinds.count('d1')} rows")


def test_manifest_parts():
    rows = [(i, r) for i in range(10) for r in range(6)]
    a = R.cap_rows(7, rows, 300)
    assert a == rows                                   # 60 rows <= floor(0.25 x 300) = 75: all kept
    b, c = R.cap_rows(7, rows, 40), R.cap_rows(7, rows, 40)
    assert b == c and len(b) == 10 and set(b) <= set(rows) and b == sorted(b, key=rows.index)
    assert R.cap_rows(8, rows, 40) != b
    for sched in SCHEDS:
        recs, plan = R.load_plan(sched)
        wins = T.load_windows(sched)
        assert [r["id"] for r in recs] == [w.rec["id"] for w in wins]
        assert plan[:50] == T.plan_steps([w.R for w in wins], 20261109, 50, 64)
    print("  M manifest: cap_rows deterministic per step, uniform subset of floor(0.25 x read rows); load_plan = the "
          "trainer's windows and plan (T4, T5 data)")


def main():
    t0 = time.time()
    tests = [test_manifest_parts, test_r1, test_r2, test_r3]
    for t in tests:
        t1 = time.time()
        t()
        print(f"PASS {t.__name__} ({time.time() - t1:.1f}s)", flush=True)
    print(f"ALL {len(tests)} PASSED ({time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()

"""Tests of crf/dart_aug.py (T4) and of the --dart path of latent_write_train.run_step / train_crf_sft.grad_pass, CPU:

  (a) draws: deterministic per (step, window); perturbed share ~ 0.5, item share ~ 0.3, k uniform on {1, 2, 3}
  (b) rows on real T1/T2 piece windows (sched_piece_t0_v2) under random delays: rho~ >= e, <= R - 1, non-decreasing
      along each speaker; H_r a per-speaker prefix and a subset of {e <= r - 1}; a row exactly where an item is overdue;
      H_r u W*_r = {e <= r} (a per-speaker prefix), both in kap order, ids = the items' S tokens; zero delays -> no row;
      in_t2 == (H_r is a state of T2's lattice at read r - 1) via latent_write_crf.lattice
  (c) coefficients: the label-path row's (h / D per target token, 1 / D at the end), seq = history ⊕ target, e_r
  (d) trainer step (tiny model, world 1): dart None == dart {} bit for bit (rows, loss, gradients); with draws, the
      T2 rows are unchanged, loss_main(with) = loss_main(without) + loss_aug and the gradients add (float tolerance),
      loss_aug == the loss of a grad pass over the augmentation rows alone; --prefetch 1 == 0 bit for bit
  (e) manifest: load_manifest accepts the draws of the plan and rejects a changed draw / a foreign plan
  (f) 1.0 s sub-grid labels on real T0 train windows: the sub-grid is a subset of the 0.5 s reads with the same final
      read; e_raw equals a literal loop over the definition; where a_j(R) > 0 the sub-grid write time is never later
      than the first 1.0 s read at or after the 0.5 s write time (the sub-grid condition is weaker)

    PYTHONPATH=.:third_party/Megatron-LM CUDA_VISIBLE_DEVICES= python -m experiments.mist_w3po_v1.crf.test_dart_aug
"""
from __future__ import annotations

import json
import math
import tempfile
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

if not torch.cuda.is_available():    # CPU-only run (CUDA_VISIBLE_DEVICES=), as test_latent_write_crf
    torch.accelerator.current_accelerator = lambda *a, **k: None

from experiments.mist_w3po_v1.crf import dart_aug as DA
from experiments.mist_w3po_v1.crf import latent_write_crf as LW
from experiments.mist_w3po_v1.crf import latent_write_train as LWT
from experiments.mist_w3po_v1.crf import precheck_serial as S
from experiments.mist_w3po_v1.crf import precheck_serial_v2 as V
from experiments.mist_w3po_v1.crf import serial_v2p_train as V2T
from experiments.mist_w3po_v1.crf import test_latent_write_crf as TL
from experiments.mist_w3po_v1.crf import test_serial_v2p_train as TV
from experiments.mist_w3po_v1.crf import test_train_crf_sft as TT
from experiments.mist_w3po_v1.crf import train_crf_sft as T

torch.set_num_threads(8)
PIECE2 = "eval_outputs/mist_w3po_v1/crf/sched_piece_t0_v2"
T0_ROOT = "eval_outputs/mist_w3po_v1/crf/t0_agree_train"


def piece2_recs(n: int, max_tok: int = 400) -> list[dict]:
    out = []
    for f in sorted(Path(PIECE2).glob("shard_*.jsonl")):
        for l in open(f, encoding="utf-8"):
            r = json.loads(l)
            if r["chunks"] and r["chunks"][0].get("label_src") == "t0" and 0 < r["n_tok"] <= max_tok:
                out.append(r)
            if len(out) >= n:
                return out
    return out


# ---------------------------------------------------------------- (a)
def test_draws():
    assert DA.draw_delays(7, 12, 345) == DA.draw_delays(7, 12, 345)
    pert = items = hit = 0
    ks = Counter()
    for s in range(400):
        for i in range(50):
            d = DA.draw_delays(6, s, i)
            if d is None:
                continue
            pert += 1
            items += len(d)
            hit += sum(x > 0 for x in d)
            ks.update(x for x in d if x > 0)
    share, ph = pert / 20000, hit / items
    assert abs(share - DA.FRAC) < 0.015 and abs(ph - DA.P_ITEM) < 0.01, (share, ph)
    assert set(ks) == {1, 2, 3} and max(ks.values()) / min(ks.values()) < 1.06, ks
    print(f"  (a) draws deterministic; perturbed {share:.3f}, item hit {ph:.3f}, k {dict(sorted(ks.items()))}")


# ---------------------------------------------------------------- (b), (c)
def check_window(rec: dict, delay: list[int]) -> tuple[int, int]:
    e = S.chunk_reads(rec)
    R, n = rec["n_read"], len(rec["chunks"])
    rt = DA.perturbed_reads(rec, delay)
    orders = DA.speaker_orders(rec, e)
    for it in orders.values():
        assert all(rt[a] <= rt[b] for a, b in zip(it, it[1:]))
    assert all(e[i] <= rt[i] <= R - 1 for i in range(n))
    qs = DA.aug_requests(rec, delay)
    by_r = {q["r"]: q for q in qs}
    lat = LW.lattice(rec, "t2")
    off = 0
    for r in range(R):
        H = {i for i in range(n) if rt[i] <= r - 1}
        due_prev = {i for i in range(n) if e[i] <= r - 1}
        assert H <= due_prev
        for it in orders.values():     # per-speaker prefix
            m = sum(1 for c in it if c in H)
            assert set(it[:m]) == H & set(it)
        if H == due_prev:
            assert r not in by_r
            continue
        q = by_r[r]
        assert set(q["hist"]) == H and q["hist"] == V.kap(rec, e, H)
        Wst = {i for i in range(n) if e[i] <= r} - H
        assert set(q["items"]) == Wst and q["items"] == V.kap(rec, e, Wst)
        assert q["hist_ids"] == [t for c in q["hist"] for t in rec["S_ids"][rec["chunks"][c]["tok0"]: rec["chunks"][c]["tok1"]]]
        assert q["forced_ids"] == S.ids_of(rec, q["items"]) and q["forced_ids"]
        assert sorted(q["lags"]) == sorted(r - e[i] for i in due_prev - H) and min(q["lags"]) >= 1
        sets_prev = lat["sets"][r - 1] if r > 0 else [frozenset()]
        assert q["in_t2"] == (frozenset(H) in sets_prev), (rec["id"], r)
        off += not q["in_t2"]
    assert len(by_r) == len(qs)
    assert not DA.aug_requests(rec, [0] * n)
    return len(qs), off


def test_rows_real():
    recs = piece2_recs(300)
    assert len(recs) >= 200, len(recs)
    nrow = noff = nw = 0
    for k, rec in enumerate(recs):
        dl = DA.draw_delays(len(rec["chunks"]), 7, k, frac=1.0)
        a, b = check_window(rec, dl)
        nrow += a
        noff += b
        nw += a > 0
        big = [3] * len(rec["chunks"])          # every item late by 3 reads
        check_window(rec, big)
    assert nrow > 100
    # (c) rows and coefficients
    wins = [T.make_window(i, r, True) for i, r in enumerate(recs[:40])]
    T.set_reach(wins, 128)
    D = T.step_totals(wins)[0]
    draws = {w.idx: DA.draw_delays(len(w.rec["chunks"]), 3, w.idx, frac=1.0) for w in wins}
    rows = DA.aug_rows(wins, draws, D)
    want = sum(len(DA.aug_requests(w.rec, draws[w.idx])) for w in wins)
    assert len(rows) == want > 0
    for rw in rows:
        q = next(x for x in DA.aug_requests(rw.win.rec, draws[rw.win.idx]) if x["r"] == rw.r)
        h0, m = len(q["hist_ids"]), len(q["forced_ids"])
        assert rw.seq == q["hist_ids"] + q["forced_ids"] and rw.K == h0 + m and rw.aug
        assert np.array_equal(rw.pos, np.arange(h0, h0 + m + 1))
        assert rw.end == (T.EOS_ID if rw.r == rw.win.R - 1 else T.WAIT_ID)
        hw = V2T.token_weights(rw.win, q["forced_ids"])
        assert np.array_equal(rw.c_tok[:m], hw / D) and rw.c_tok[m] == 0.0
        assert rw.c_end[m] == 1.0 / D and not rw.c_end[:m].any() and not rw.c_state.any()
        assert not rw.c_lab_tok.any() and not rw.c_lab_end.any()
    print(f"  (b)(c) {len(recs)} real piece windows: {nrow} rows from random draws ({nw} windows with rows, "
          f"{noff} states off T2's lattice); invariants, kap order, ids, in_t2 vs the lattice, coefficients hold")


# ---------------------------------------------------------------- (d)
def run(m, wins, D, dart, pf, rows_only=None):
    ctx = {**TV.vctx(perturb=0.2, budget=1500), "final_full_post": True}
    TT.zero(m)
    if rows_only is not None:
        lg = T.grad_pass(m, rows_only, ctx, 3)
        return None, lg, TT.grads(m)
    v = LWT.run_step(m, wins, D, ctx, 3, LW.TAU, 0, 1, prefetch=pf, **({} if dart == "none" else {"dart": dart}))
    ctx["prefetched_batches"] = v["batches"]
    lg = T.grad_pass(m, v["rows"], ctx, 3)
    return v, lg, TT.grads(m)


def test_step():
    base = TV.small_choice_windows(3, max_paths=60)
    wins = [T.make_window(w.idx, TL.widen(w.rec), True) for w in base]
    T.set_reach(wins, 128)
    m = TT.tiny_model(seed=29)
    bias = torch.zeros(151936)
    bias[T.WAIT_ID] = 9.0
    bias[T.EOS_ID] = 6.0
    m.logits_at = lambda h: TT.ThinkerGStarModel.logits_at(m, h) + bias
    D = T.step_totals(wins)[0]
    v0, l0, g0 = run(m, wins, D, "none", False)
    v1, l1, g1 = run(m, wins, D, {}, False)
    assert len(v0["rows"]) == len(v1["rows"]) and l0["loss_main"] == l1["loss_main"] and l1["loss_aug"] == 0.0
    for k in g0:
        assert torch.equal(g0[k], g1[k]), k
    draws = {w.idx: [2] * len(w.rec["chunks"]) for w in wins}
    v2, l2, g2 = run(m, wins, D, draws, False)
    v3, l3, g3 = run(m, wins, D, draws, True)
    assert l2["loss_main"] == l3["loss_main"]
    for k in g2:
        assert torch.equal(g2[k], g3[k]), k
    aug = [rw for rw in v2["rows"] if rw.aug]
    n0 = len(v0["rows"])
    assert len(aug) == v2["stats"]["dart_rows"] > 0 and len(v2["rows"]) == n0 + len(aug)
    for a, b in zip(v0["rows"], v2["rows"][:n0]):     # T2's rows: same structure and coefficients
        assert a.key == b.key and a.r == b.r and np.array_equal(a.c_tok, b.c_tok) and np.array_equal(a.c_end, b.c_end)
    _, la, ga = run(m, wins, D, None, False, rows_only=aug)
    tol = 1e-5 * max(1.0, abs(l2["loss_main"]))
    assert abs(l2["loss_main"] - (l0["loss_main"] + l2["loss_aug"])) < tol, (l2["loss_main"], l0["loss_main"], l2["loss_aug"])
    assert abs(la["loss_main"] - l2["loss_aug"]) < tol and abs(la["loss_aug"] - la["loss_main"]) < tol
    worst = 0.0
    for k in g2:
        d = (g2[k] - g0[k] - ga[k]).abs().max().item()
        s = max(1e-6, g2[k].abs().max().item())
        worst = max(worst, d / s)
    assert worst < 1e-4, worst
    print(f"  (d) step: dart None == {{}} bit for bit; with draws {len(aug)} extra rows, T2 rows unchanged, loss "
          f"{l2['loss_main']:.6f} = {l0['loss_main']:.6f} + aug {l2['loss_aug']:.6f}; gradients add (max rel "
          f"{worst:.1e}); prefetch 1 == 0 bit for bit")


# ---------------------------------------------------------------- (e)
def test_manifest():
    recs = piece2_recs(60)
    wins = [T.make_window(i, r, True) for i, r in enumerate(recs)]
    plan = T.plan_steps([w.R for w in wins], 5, 20, 64)
    with tempfile.TemporaryDirectory() as td:
        sched = str(Path(td) / "sched")
        man = Path(td) / "man"
        man.mkdir()
        (man / "summary.json").write_text(json.dumps({"sched": sched, "plan": {"seed": 5, "steps": 20,
                                                                                "rows_per_step": 64}}))
        lines = []
        for s, ids in enumerate(plan):
            for i in ids:
                d = DA.draw_delays(len(wins[i].rec["chunks"]), s, i)
                if d is not None:
                    lines.append({"step": s, "idx": i, "id": wins[i].rec["id"], "delay": d})
        (man / "dart.jsonl").write_text("".join(json.dumps(x) + "\n" for x in lines))
        got = DA.load_manifest(str(man), wins, plan, sched, 5, 20, 64)
        assert sum(len(v) for v in got.values()) == len(lines) > 0
        for bad_args in ((sched, 6, 20, 64), (sched + "x", 5, 20, 64)):
            try:
                DA.load_manifest(str(man), wins, plan, *bad_args)
                raise AssertionError("foreign plan accepted")
            except SystemExit:
                pass
        x = dict(lines[0])
        x["delay"] = [0 if v else 1 for v in x["delay"]]
        (man / "dart.jsonl").write_text("".join(json.dumps(y) + "\n" for y in [x] + lines[1:]))
        try:
            DA.load_manifest(str(man), wins, plan, sched, 5, 20, 64)
            raise AssertionError("changed draw accepted")
        except SystemExit:
            pass
    print(f"  (e) manifest: {len(lines)} draws of a 20-step plan accepted; changed draw / foreign plan rejected")


# ---------------------------------------------------------------- (f)
def test_subgrid():
    from experiments.mist_w3po_v1.crf import t0_agree_labels as T0

    sc = {}
    for l in open(Path(T0_ROOT) / "score_0of8" / "scores.jsonl", encoding="utf-8"):
        r = json.loads(l)
        sc[r["id"]] = r
        if len(sc) >= 300:
            break
    n = later = 0
    for l in open(Path(T0_ROOT) / "cpu" / "targets.jsonl", encoding="utf-8"):
        tg = json.loads(l)
        if tg["id"] not in sc:
            continue
        a = np.asarray(sc[tg["id"]]["a"], np.int64)
        tg1, lb = DA.subgrid_labels(tg, a)
        t0, t1 = tg["t_ms"], lb["t_ms"]
        assert set(t1) <= set(t0) and t1[-1] == t0[-1] and all(t % 1000 == 0 for t in t1[:-1])
        sub = [t0.index(t) for t in t1]
        As = a[sub]
        R = len(t1)
        u1 = T0.prop_reads(tg1)
        e0, _ = T0.earliest_stable(a, T0.prop_reads(tg))
        for j in range(As.shape[1]):
            if As[-1, j] == 0:
                assert lb["e_raw"][j] == u1[j] and lb["zero"][j] == 1
                continue
            lit = min(r for r in range(R) if all(As[r2, j] >= As[-1, j] for r2 in range(r, R)))
            assert lb["e_raw"][j] == lit, (tg["id"], j)
            first1 = next(k for k, t in enumerate(t1) if t >= t0[e0[j]])
            assert lit <= first1
            later += lit < first1
        assert lb["e"] == [int(x) for x in np.asarray(lb["e"])]
        n += 1
    assert n >= 250, n
    print(f"  (f) sub-grid labels on {n} T0 train windows: literal definition, subset grid; {later} pieces earlier than "
          f"the 0.5 s time rounded up to the 1.0 s grid")


# ---------------------------------------------------------------- (g) T5 lattice mode
def test_t5_mode():
    rng = np.random.default_rng(11)
    worst, cnt = 0.0, 0
    for _ in range(60):
        rec = TL.synthetic(rng, int(rng.integers(2, 6)), int(rng.integers(1, 4)), int(rng.integers(1, 4)))
        e, h = LW.bounds(rec, "t5")
        R = rec["n_read"]
        assert h == [min(R - 1, max(c["u_read"], x)) for c, x in zip(rec["chunks"], e)]
        pl = LW.plan_window(rec, LW.lattice(rec, "t5"), mode="t5")
        npath = math.exp(V.forward_backward([np.where(x, 0.0, -np.inf) for x in pl["edge"]], 1.0)["logZ"])
        if npath > 3000:
            continue
        wv, _ = TL.check_brute(rec, "t5")
        worst = max(worst, wv)
        cnt += 1
    assert cnt >= 20, cnt
    base = TV.small_choice_windows(3, max_paths=60)
    wins = [T.make_window(w.idx, TL.widen(w.rec), True) for w in base]
    T.set_reach(wins, 128)
    m = TT.tiny_model(seed=31)
    D = T.step_totals(wins)[0]
    ctx = {**TV.vctx(perturb=0.2, budget=1500), "final_full_post": True}
    TT.zero(m)
    v = LWT.run_step(m, wins, D, ctx, 3, LW.TAU, 0, 1, mode="t5", dart={w.idx: [1] * len(w.rec["chunks"]) for w in wins})
    lg = T.grad_pass(m, v["rows"], ctx, 3)
    assert math.isfinite(lg["loss_main"]) and v["stats"]["dart_rows"] >= 0
    print(f"  (g) t5 bounds = min(R - 1, max(u, e)); brute force on {cnt} synthetic windows (max |d| {worst:.1e}); "
          f"t5 + dart trainer step finite (loss {lg['loss_main']:.4f}, {v['stats']['dart_rows']} aug rows)")


def main():
    t0 = time.time()
    tests = [test_draws, test_rows_real, test_manifest, test_subgrid, test_step, test_t5_mode]
    for t in tests:
        t1 = time.time()
        t()
        print(f"PASS {t.__name__} ({time.time() - t1:.1f}s)", flush=True)
    print(f"ALL {len(tests)} PASSED ({time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()

"""Tests of crf/joint_speech_crf.py and crf/joint_speech_train.py (T3), CPU only:

  (a) brute force: every (rho assignment in [e_i, h_i], non-decreasing per speaker) x per-read order (canon / newfirst
      where real) x per (read, speaker) block rate class, with the playback simulated per path WITHOUT the lattice
      (queue end on the absolute grid, rounded when a block is added), v2''s dummy LM: log Z, the per-sequence weights
      (cend), E_q[l] equal the augmented chain's to 1e-9 (tau 0.5 / 1, lam 0 / 0.3 / 1.5; synthetic multi-speaker
      records with a 2-class talker model; real AMI windows widened as in T2's test)
  (b) the null talker (one class, D = 0, pi = 1) with the playback terms zeroed = T2's posterior (log Z, cend, E[Lat])
  (c) A5: a tiny state cap forces the coarser grid; the coarse chain still equals the brute force on that grid
  (d) lam: E_q[l](lam) non-increasing; the solve hits the target (rel 1e-6) when binding; T1's target from the masked
      chain equals the brute force restricted to T1's rho sets
  (e) the trainer step (joint_speech_train.run_step -> grad_pass, world 1, tiny model): finite loss, prefetch 1 = 0
      bit for bit, sum of the coefficients x D = sum h + R

    PYTHONPATH=.:third_party/Megatron-LM CUDA_VISIBLE_DEVICES= python -m experiments.mist_w3po_v1.crf.test_joint_speech_crf
"""
from __future__ import annotations

import itertools
import math
import time
from collections import defaultdict

import numpy as np
import torch

if not torch.cuda.is_available():    # CPU-only run: transformers' model modules probe the accelerator at import
    torch.accelerator.current_accelerator = lambda *a, **k: None

from experiments.mist_w3po_v1.crf import joint_speech_crf as JS
from experiments.mist_w3po_v1.crf import joint_speech_train as JST
from experiments.mist_w3po_v1.crf import latent_write_crf as LW
from experiments.mist_w3po_v1.crf import precheck_serial as S
from experiments.mist_w3po_v1.crf import precheck_serial_v2 as V
from experiments.mist_w3po_v1.crf import serial_v2p_train as V2T
from experiments.mist_w3po_v1.crf import test_latent_write_crf as TL
from experiments.mist_w3po_v1.crf import test_serial_v2p_train as TV
from experiments.mist_w3po_v1.crf import test_train_crf_sft as TT
from experiments.mist_w3po_v1.crf import train_crf_sft as T

torch.set_num_threads(8)


def toy_talker() -> JS.TalkerModel:
    pi = [[0.8, 0.2], [0.7, 0.3], [0.5, 0.5], [0.4, 0.6], [0.3, 0.7], [0.3, 0.7], [0.2, 0.8], [0.1, 0.9]]
    return JS.TalkerModel({"feature": "words", "offs": [0, 2], "dur": {"0": [330.0, 210.0], "2": [270.0, 150.0]},
                           "pi": pi})


def text_of(rec: dict) -> dict:
    """synthetic records have 'w w w' texts; give them distinct word counts so D differs."""
    return rec


# ---------------------------------------------------------------- brute force
def brute(rec: dict, tm: JS.TalkerModel, tau: float, lam: float, delta: int = JS.DELTA_MS, zero_play: bool = False,
          rho_sets=None) -> dict:
    e, h = LW.bounds(rec, "t2")
    R, n = rec["n_read"], len(rec["chunks"])
    ch, t_ms = rec["chunks"], rec["t_ms"]
    spk = sorted({c["slot"] for c in ch})
    by = defaultdict(list)
    for c in V.kap(rec, e, range(n)):
        by[ch[c]["slot"]].append(c)
    pairs = [(x, y) for it in by.values() for x, y in zip(it, it[1:])]
    keys, scores, lats = [], [], []
    ranges = [range(e[c], h[c] + 1) for c in range(n)] if rho_sets is None else rho_sets
    for rho in itertools.product(*ranges):
        if any(rho[x] > rho[y] for x, y in pairs):
            continue
        per_read, blocks = [], []
        for r in range(R):
            hist = V.kap(rec, e, [c for c in range(n) if rho[c] <= r - 1])
            Wr = V.kap(rec, e, [c for c in range(n) if rho[c] == r])
            opts = [Wr]
            carried = [c for c in Wr if e[c] <= r - 1]
            if carried:
                ps = {ch[c]["slot"] for c in carried}
                N = [c for c in Wr if e[c] == r and ch[c]["slot"] not in ps]
                if N:
                    opts.append(N + V.kap(rec, e, [c for c in Wr if c not in set(N)]))
            per_read.append([(r, tuple(S.ids_of(rec, hist)), tuple(S.ids_of(rec, o))) for o in opts])
            blocks.append([[c for c in Wr if ch[c]["slot"] == k] for k in spk])
        nb = [(r, x) for r in range(R) for x in range(len(spk)) if blocks[r][x]]
        for rates in itertools.product(range(len(tm.offs)), repeat=len(nb)):
            rate = dict(zip(nb, rates))
            b = [0] * len(spk)
            lp, l_ = 0.0, 0.0
            for r in range(R):
                dt = t_ms[r] - (t_ms[r - 1] if r > 0 else 0)
                bp = [max(0, x - dt) for x in b]
                b = list(bp)
                for x in range(len(spk)):
                    cs = blocks[r][x]
                    if not cs:
                        continue
                    words = sum(LW.n_words(ch[c]) for c in cs)
                    d = float(tm.dur(tm.n(" ".join(ch[c]["text"] for c in cs)))[rate[(r, x)]])
                    l_ += sum(LW.n_words(ch[c]) * (t_ms[r] - ch[c]["src_end_ms"]) / 1000.0 for c in cs)
                    if not zero_play:
                        l_ += words * bp[x] / 1000.0 + (words + 1) / 2.0 * d / 1000.0
                    lp += float(tm.logpi[min(JS.N_BIN - 1, bp[x] // JS.BIN_MS), rate[(r, x)]])
                    if d > 0:
                        end = t_ms[r] + bp[x] + d
                        b[x] = int(min(JS.CAP_MS, max(0, int(np.rint(end / delta)) * delta - t_ms[r])))
            for combo in itertools.product(*per_read):
                sc = 0.0
                for (r, hi, wi) in combo:
                    tl, ec = V.dummy_scores(r, r == R - 1, list(hi), list(wi), "new")
                    sc += float(tl.sum() + ec[-1])
                keys.append(combo)
                scores.append(tau * sc + lp - lam * l_)
                lats.append(l_)
    scores = np.asarray(scores)
    logZ = V.lse(scores)
    q = np.exp(scores - logZ)
    Wb = defaultdict(float)
    for combo, qq in zip(keys, q):
        for k in combo:
            Wb[k] += qq
    return {"logZ": logZ, "W": Wb, "Eq_l": float((q * np.asarray(lats)).sum()), "n_paths": len(keys)}


def chain_vs_brute(rec, tm, taus=(0.5, 1.0), lams=(0.0, 0.3, 1.5), kcap=JS.K_CAP) -> tuple[float, int, int]:
    pl = LW.plan_window(rec, LW.lattice(rec, "t2"))
    res = TL.all_scored(pl)
    w = TL.W_(0, rec)
    worst, n, delta = 0.0, 0, None
    for tau in taus:
        reads = LW.assemble(pl, LW.full_res(w, pl, res))
        Eo = LW.edge_terms(reads, tau)
        aug = JS.augment_capped(rec, pl, Eo, tm, kcap=kcap)
        delta = aug["delta"]
        m1 = JS.t1_mask(aug, pl)
        parts = {"reads": reads, "Eo": Eo, "aug": aug, "edge": pl["edge"], "sum_words": 1.0,
                 "target": JS.forward_backward(aug, 0.0, m1)["Eq_l"]}
        for lam in lams:
            post = JS.posterior_from_parts(w, pl, parts, tau, lam)
            bf = brute(rec, tm, tau, lam, delta=delta)
            d = [abs(post["logZ"] - bf["logZ"]), abs(post["Eq_l"] - bf["Eq_l"])]
            Wt = TL.lattice_weights(pl, post)
            ks = set(Wt) | set(bf["W"])
            d.append(max(abs(Wt.get(k, 0.0) - bf["W"].get(k, 0.0)) for k in ks))
            miss = [k for k in ks if bf["W"].get(k, 0.0) > 1e-12 and k not in Wt]
            assert not miss, (rec["id"], len(miss))
            worst = max(worst, max(d))
            n = bf["n_paths"]
        # T1 target = the brute force restricted to T1's rho sets {e, min(e + 1, R - 1)}
        e, _ = LW.bounds(rec, "t2")
        R = rec["n_read"]
        bf1 = brute(rec, tm, tau, 0.0, delta=delta, rho_sets=[range(x, min(x + 1, R - 1) + 1) for x in e])
        worst = max(worst, abs(parts["target"] - bf1["Eq_l"]))
    return worst, n, delta


def records(rng, n: int, max_paths: int = 600) -> list[dict]:
    out = []
    while len(out) < n:
        rec = TL.synthetic(rng, int(rng.integers(2, 6)), int(rng.integers(1, 3)), int(rng.integers(1, 3)))
        for c in rec["chunks"]:
            c["text"] = " ".join(["word"] * c["n_words"])
        pl = LW.plan_window(rec, LW.lattice(rec, "t2"))
        npath = math.exp(V.forward_backward([np.where(x, 0.0, -np.inf) for x in pl["edge"]], 1.0)["logZ"])
        if npath > max_paths:
            continue
        out.append(rec)
    return out


# ---------------------------------------------------------------- (a)
def test_brute_force():
    rng = np.random.default_rng(11)
    tm = toy_talker()
    worst, paths, multi = 0.0, [], 0
    for rec in records(rng, 30, max_paths=150):
        wv, n, _ = chain_vs_brute(rec, tm)
        worst = max(worst, wv)
        paths.append(n)
        multi += len({c["slot"] for c in rec["chunks"]}) > 1
    nreal = 0
    for r in TV.recs0()[:400]:
        if not (0 < r["n_tok"] <= 100 and r["n_read"] <= 5 and len(r["chunks"]) <= 4):
            continue
        rec = TL.widen(r)
        pl = LW.plan_window(rec)
        npath = math.exp(V.forward_backward([np.where(x, 0.0, -np.inf) for x in pl["edge"]], 1.0)["logZ"])
        if npath > 60 or npath < 3:
            continue
        wv, n, _ = chain_vs_brute(rec, tm, taus=(0.5,), lams=(0.0, 0.7))
        worst = max(worst, wv)
        paths.append(n)
        nreal += 1
        if nreal >= 10:
            break
    assert worst < 1e-9 and multi >= 10 and nreal >= 5, (worst, multi, nreal)
    print(f"  (a) brute force over (rho x order x rate): 30 synthetic ({multi} multi-speaker) + {nreal} widened real "
          f"windows, up to {max(paths)} paths; log Z, per-sequence weights, E_q[l], T1 target max |d| {worst:.1e}")


# ---------------------------------------------------------------- (b)
def test_null_equals_t2():
    rng = np.random.default_rng(3)
    worst = 0.0
    recs = records(rng, 25, max_paths=3000)
    recs += [TL.widen(r) for r in TV.recs0()[:300] if 0 < r["n_tok"] <= 200 and r["n_read"] >= 3][:25]
    for rec in recs:
        pl = LW.plan_window(rec, LW.lattice(rec, "t2"))
        res = TL.all_scored(pl)
        w = TL.W_(0, rec)
        for lam in (0.0, 0.4):
            p2 = LW.posterior(w, pl, res, 0.5, lam)
            p3 = JS.posterior(w, pl, res, JS.TalkerModel.null(), 0.5, lam, zero_play=True)
            d = [abs(p2["logZ"] - p3["logZ"]), abs(p2["Eq_lat"] - p3["Eq_l"]), abs(p2["exp_nf"] - p3["exp_nf"]),
                 abs(p2["exp_defer"] - p3["exp_defer"])]
            d += [float(np.max(np.abs(a - b))) if len(a) else 0.0 for a, b in zip(p2["cend"], p3["cend"])]
            worst = max(worst, max(d))
            assert p3["aug_states_max"] == max(len(s) for s in pl["lat"]["states"])
        p2t = LW.window_parts(w, pl, res, 0.5)["target"]
        p3t = JS.window_parts(w, pl, res, JS.TalkerModel.null(), 0.5, zero_play=True)["target"]
        worst = max(worst, abs(p2t - p3t))
    assert worst < 1e-10, worst
    print(f"  (b) null talker + zero playback == T2 (log Z, cend, E[Lat], exp_nf / defer, T1 target) on {len(recs)} "
          f"windows: max |d| {worst:.1e}")


# ---------------------------------------------------------------- (c)
def test_cap_coarsens():
    rng = np.random.default_rng(8)
    tm = toy_talker()
    fired, worst = 0, 0.0
    for rec in records(rng, 40, max_paths=150):
        pl = LW.plan_window(rec, LW.lattice(rec, "t2"))
        reads = LW.assemble(pl, LW.full_res(TL.W_(0, rec), pl, TL.all_scored(pl)))
        a = JS.augment(rec, pl, LW.edge_terms(reads, 0.5), tm)
        cap = max(a["states"]) - 1
        if cap < 2:
            continue
        wv, _, delta = chain_vs_brute(rec, tm, taus=(0.5,), lams=(0.0, 0.5), kcap=cap)
        assert delta > JS.DELTA_MS
        worst = max(worst, wv)
        fired += 1
        if fired >= 8:
            break
    assert fired >= 4 and worst < 1e-9, (fired, worst)
    print(f"  (c) A5 cap: {fired} windows forced onto a coarser grid; still exact vs brute force on it ({worst:.1e})")


# ---------------------------------------------------------------- (d)
def test_lambda():
    rng = np.random.default_rng(21)
    tm = toy_talker()
    parts_all = []
    for rec in records(rng, 12, max_paths=3000):
        pl = LW.plan_window(rec, LW.lattice(rec, "t2"))
        res = TL.all_scored(pl)
        w = TL.W_(0, rec)
        pa = JS.window_parts(w, pl, res, tm, 0.5)
        ls = [0.0, 0.1, 0.3, 1.0, 3.0, 10.0]
        el = [JS.forward_backward(pa["aug"], x)["Eq_l"] for x in ls]
        assert all(a >= b - 1e-9 for a, b in zip(el, el[1:])), el
        parts_all.append(pa)
    lam, info = JS.solve_lambda(parts_all)
    got = sum(JS.forward_backward(p["aug"], lam)["Eq_l"] for p in parts_all)
    if info["binding"]:
        assert abs(got - info["target"]) < 1e-6 * max(1.0, abs(info["target"])), (got, info)
    else:
        assert lam == 0.0
    # a lower target binds
    for p in parts_all:
        p["target"] = 0.9 * JS.forward_backward(p["aug"], 0.0)["Eq_l"] + 0.1 * JS.forward_backward(p["aug"], 50.0)[
            "Eq_l"]
    lam2, info2 = JS.solve_lambda(parts_all)
    got2 = sum(JS.forward_backward(p["aug"], lam2)["Eq_l"] for p in parts_all)
    assert info2["binding"] and info2["bracketed"] and abs(got2 - info2["target"]) < 1e-6 * abs(info2["target"])
    print(f"  (d) lam: E_q[l] non-increasing on 12 windows; T1-target solve lam {lam:.4f} (binding {info['binding']}); "
          f"a 10%-tighter target: lam {lam2:.4f}, E_q[l] = target to {abs(got2 - info2['target']):.1e}")


# ---------------------------------------------------------------- (e)
def test_run_step():
    base = TV.small_choice_windows(3, max_paths=60)
    wins = [T.make_window(w.idx, TL.widen(w.rec), True) for w in base]
    T.set_reach(wins, 128)
    m = TT.tiny_model(seed=23)
    bias = torch.zeros(151936)
    bias[T.WAIT_ID] = 9.0
    bias[T.EOS_ID] = 6.0
    m.logits_at = lambda h: TT.ThinkerGStarModel.logits_at(m, h) + bias
    D = T.step_totals(wins)[0]
    out = []
    for pf in (False, True):
        ctx = {**TV.vctx(perturb=0.2, budget=1500), "final_full_post": True, "talker_model": toy_talker()}
        TT.zero(m)
        v = JST.run_step(m, wins, D, ctx, 3, JS.TAU, 0, 1, prefetch=pf)
        ctx["prefetched_batches"] = v["batches"]
        lg = T.grad_pass(m, v["rows"], ctx, 3)
        V2T.reduce_logs(v, lg, "cpu")
        rec = V2T.metrics_fields(v, lg, {})
        out.append((v, lg, TT.grads(m), rec))
    (v0, l0, g0, r0), (v1, l1, g1, r1) = out
    assert math.isfinite(l0["loss_main"]) and l0["loss_main"] == l1["loss_main"]
    for k in g0:
        assert torch.equal(g0[k], g1[k]), k
    tot = sum(float(rw.c_tok.sum() + rw.c_end.sum()) for rw in v0["rows"]) * D
    want = sum(float(w.h.sum() + w.R) for w in wins)
    assert abs(tot - want) < 1e-9 * want, (tot, want)
    st = v0["stats"]
    need = {"lam", "lam_bracketed", "Eq_speech_lat_T3", "Eq_speech_lat_T1_target", "sum_words", "exp_nf",
            "aug_states_max", "aug_edges", "aug_coarsened", "sft_loss", "rows_pass1", "rows_pass2"}
    assert need <= set(r0), need - set(r0)
    if st["lam_binding"]:
        assert abs(st["Eq_speech_lat_T3"] - st["Eq_speech_lat_T1_target"]) < 1e-6 * max(
            1.0, abs(st["Eq_speech_lat_T1_target"]))
    print(f"  (e) step path (tiny model, widened real windows, world 1): loss {l0['loss_main']:.6f} finite, prefetch "
          f"1 = 0 bit for bit; sum coef x D = sum h + R ({want:.0f}); lam {st['lam']:.4f} (binding "
          f"{st['lam_binding']}), E[l]/word T3 {st['Eq_speech_lat_T3']:.3f} vs T1 target "
          f"{st['Eq_speech_lat_T1_target']:.3f} (lam 0 {st['Eq_speech_lat_lam0']:.3f}); aug states max "
          f"{st['aug_states_max']}, edges {st['aug_edges']}")


def main():
    t0 = time.time()
    tests = [test_null_equals_t2, test_brute_force, test_cap_coarsens, test_lambda, test_run_step]
    for t in tests:
        t1 = time.time()
        t()
        print(f"PASS {t.__name__} ({time.time() - t1:.1f}s)", flush=True)
    print(f"ALL {len(tests)} PASSED ({time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()

"""Tests of crf/serial_v2p_train.py (train_crf_sft.py --objective serial_v2p), CPU only: real sched_v1 AMI-train windows
(audio, posteriors, tokenizer, prompts), the tiny random causal LM of test_train_crf_sft in place of v8.

  (1) posterior: the trainer's per-sequence weights (pass 1 -> forward-backward -> cend) and its state / deferral /
      order marginals equal (a) the precheck's own forward (VP.forward_p on VP.assemble_p of the precheck's plan with
      forced reads skipped) and (b) a brute-force enumeration over (rho assignment x per-read order) built without the
      lattice (VP.brute_paths_p / brute_force_p), with v2's dummy LM on many real windows (exact, 1e-9) and with the
      tiny model's pass-1 log-probs on windows with real order choices (each brute-force sequence scored alone)
  (2) pass 2 = the EM surrogate: the trainer's gradient (shared-prefix rows, summed coefficients) equals autograd of
      -(1/D) sum_seq W(seq) [sum_t h_t log p + log p(e_r)] over the brute-force sequences scored one by one; and at
      tau = 1, bar off, h = 1, D = 1 it equals -grad log Z (Fisher identity, log Z by brute force on differentiable psi)
  (3) collapse: when the lattice is the label path (lattice states patched to the label; and real windows whose v2'
      lattice is a single path) the rows are C0's rows and the loss / gradients are bit-identical to --objective heuristic
  (4) packing: different micro-batch budgets and rank splits give the same posterior weights and gradients (fp32)
  (5) weights: per window sum of all coefficients x D = sum h + R (= C0's per-window weight, so D is C0's); every read's
      edge-order mass sums to 1
  (6) --fast-moe-bwd: the patched grouped_mm fallback (no frozen expert-weight gradient) gives bit-identical forward
      outputs and input gradients

    PYTHONPATH=.:third_party/Megatron-LM python -m experiments.mist_w3po_v1.crf.test_serial_v2p_train
"""
from __future__ import annotations

import json
import math
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from experiments.mist_w3po_v1.crf import precheck_serial as S
from experiments.mist_w3po_v1.crf import precheck_serial_v2 as V
from experiments.mist_w3po_v1.crf import precheck_serial_v2p as VP
from experiments.mist_w3po_v1.crf import serial_v2p_train as V2T
from experiments.mist_w3po_v1.crf import test_train_crf_sft as TT
from experiments.mist_w3po_v1.crf import train_crf_sft as T

torch.set_num_threads(8)
_C = {}


def vctx(perturb: float = 0.0, budget: int = 4000, bar: bool = True, per_row: bool = True) -> dict:
    """test_train_crf_sft.ctx_of plus --feat-per-row's cache (the setting of the v2' run)."""
    c = TT.ctx_of(perturb=perturb, budget=budget, bar=bar)
    if per_row:
        c["feat_cache"] = {}
    return c


def recs0() -> list[dict]:
    if "recs" not in _C:
        _C["recs"] = [json.loads(l) for l in open(Path(T.SCHED) / "shard_000.jsonl", encoding="utf-8")]
    return _C["recs"]


def real_choice(rec: dict) -> bool:
    pl = VP.plan_window_p(rec, skip_forced=False)
    return any(cx["nf"] for rd in pl["reads"] for cx in rd["ctx"])


def n_paths(rec: dict) -> float:
    pl = VP.plan_window_p(rec, skip_forced=False)
    return VP.n_paths_lattice(VP.assemble_p(pl, V.dummy_res(pl)))


def small_choice_windows(n: int, max_tok: int = 80, max_paths: int = 40, max_r: int = 5, hier: bool = True):
    """n windows of shard_000 with a real order choice, <= max_paths (rho x order) paths, short S."""
    out = []
    for r in recs0():
        if r["n_tok"] == 0 or r["n_tok"] > max_tok or r["n_read"] > max_r:
            continue
        if real_choice(r) and n_paths(r) <= max_paths:
            out.append(r)
        if len(out) >= n:
            break
    assert len(out) == n, len(out)
    wins = [T.make_window(i, r, hier) for i, r in enumerate(out)]
    T.set_reach(wins, 128)
    return wins


def seq_key(r: int, hist, written) -> tuple:
    return (r, tuple(int(x) for x in hist), tuple(int(x) for x in written))


def trainer_weights(w: T.Window, pl: dict, post: dict) -> dict:
    """(read, hist ids, written ids) -> the trainer's weight (cend summed over the cells of that sequence)."""
    W = defaultdict(float)
    for k, q in enumerate(pl["reqs"]):
        for b, c in enumerate(post["cend"][k]):
            if c != 0.0 or b == len(q["forced_ids"]):
                W[seq_key(q["r"], q["hist_ids"], q["forced_ids"][:b])] += c
    return W


def brute(rec: dict, psi_fn, tau: float):
    """(brute_force_p result, {seq key: weight}) over every (rho x order) path."""
    lat = V.lattice(rec)
    paths, scores = VP.brute_paths_p(rec, lat, psi_fn)
    bf = VP.brute_force_p(rec, lat, paths, scores, tau)
    rho, R, n = lat["rho"], lat["R"], len(rec["chunks"])
    W = defaultdict(float)
    keys = []
    for (a, os_), q in zip(paths, bf["q"]):
        ks = []
        for r in range(R):
            hist = V.kap(rec, rho, [c for c in range(n) if a[c] <= r - 1])
            Wr = [c for c in range(n) if a[c] == r]
            order = dict(VP.brute_orders(rec, rho, r, Wr))[os_[r]]
            k = seq_key(r, S.ids_of(rec, hist), S.ids_of(rec, order))
            W[k] += q
            ks.append(k)
        keys.append(ks)
    return lat, paths, scores, bf, W, keys


def cmp_marginals(lat, post: dict, ref: dict, tol: float, what: str) -> float:
    """state / deferral / order marginals of the trainer's posterior vs a reference (forward_p fb or brute force)."""
    fb = post["fb"]
    d = []
    for r in range(lat["R"]):
        if "state_marg" in ref:
            d += [abs(fb["marg"][r][j] - ref["state_marg"][r].get(s, 0.0)) for j, s in enumerate(lat["states"][r])]
        else:
            d += list(np.abs(np.asarray(fb["marg"][r]) - np.asarray(ref["marg"][r])))
    dm = V.deferral_marginals(lat, fb)
    rd = ref["defer"] if "defer" in ref else V.deferral_marginals(lat, ref)
    d += [abs(dm[c] - rd[c]) for c in rd]
    worst = max(d) if d else 0.0
    assert worst < tol, f"{what}: marginals differ by {worst:.3e}"
    return worst


def cmp_weights(Wt: dict, Wb: dict, tol: float, what: str) -> float:
    keys = set(Wt) | set(Wb)
    worst = max(abs(Wt.get(k, 0.0) - Wb.get(k, 0.0)) for k in keys)
    big = [k for k in keys if Wb.get(k, 0.0) > 1e-12 and k not in Wt]
    assert not big, f"{what}: brute-force sequences missing from the trainer's rows: {len(big)}"
    assert worst < tol, f"{what}: per-sequence weights differ by {worst:.3e}"
    return worst


def order_marg(post: dict) -> list:
    out = []
    for rd in post["reads"]:
        C, Nf = VP.mats(rd)
        out.append(Nf is not None and not np.isnan(Nf).all())
    return out


# ---------------------------------------------------------------- (1a) dummy LM, many windows, exact
def test_posterior_dummy_exact():
    recs = [r for r in recs0()[:1500] if r["n_tok"] > 0]
    n_ch, n_bf, worst = 0, 0, 0.0
    for i, rec in enumerate(recs):
        w = T.make_window(i, rec, True)
        pl = V2T.plan(w)
        res = {(w.idx, k): v for k, v, q in zip(range(len(pl["reqs"])), V.dummy_res(pl), pl["reqs"]) if q["scored"]}
        post = V2T.posterior(w, pl, res, V2T.TAU)
        # (a) the precheck's own plan (forced reads skipped) and forward_p on the same dummy scores
        pp = VP.plan_window_p(rec, skip_forced=True)
        fr = VP.forward_p(VP.assemble_p(pp, V.dummy_res(pp)), V2T.TAU)
        assert abs(fr["logZ"] - post["logZ"]) < 1e-9, (rec["id"], fr["logZ"], post["logZ"])
        worst = max(worst, cmp_marginals(pl["lat"], post, fr, 1e-9, rec["id"]))
        for o_t, o_p in zip(fr["order"], VP.forward_p(post["reads"], V2T.TAU)["order"]):
            assert (o_t is None) == (o_p is None) and (o_t is None or abs(o_t["q_nf"] - o_p["q_nf"]) < 1e-12)
        has = any(order_marg(post))
        n_ch += has
        # (b) brute force (rho x order), small windows: marginals and per-sequence weights
        if (has or n_bf < 40) and VP.n_paths_lattice(post["reads"]) <= 3000:
            dm = {}

            def psi_fn(r, final, hist, written):
                tl, ec = V.dummy_scores(r, final, hist, written)
                return float(tl.sum() + ec[-1])
            lat, paths, scores, bf, Wb, _ = brute(rec, psi_fn, V2T.TAU)
            worst = max(worst, cmp_marginals(lat, post, bf, 1e-9, rec["id"] + " brute"))
            for r in range(lat["R"]):
                o = fr["order"][r]
                if o is not None:
                    assert abs(o["q_nf"] - bf["order"][r]["q_nf"]) < 1e-9
            worst = max(worst, cmp_weights(trainer_weights(w, pl, post), Wb, 1e-9, rec["id"]))
            n_bf += 1
            del dm
    assert n_ch >= 50 and n_bf >= 60, (n_ch, n_bf)
    print(f"  (1a) dummy LM, {len(recs)} real windows ({n_ch} with a real order choice): trainer posterior = precheck "
          f"forward_p; {n_bf} windows = brute force (marginals, per-sequence weights), max |d| {worst:.1e}")


# ---------------------------------------------------------------- tiny-model helpers
def score_rows_alone(m, keys: list[tuple], wins_by_r: dict, ctx: dict, bar: bool, grad: bool = False):
    """log-probs of brute-force sequences, each its own row: {key: (tl [m], ec [m + 1])} (torch, grad if asked)."""
    rows = []
    for key in keys:
        w, r = wins_by_r[key]
        _, hist, wr = key
        seq = list(hist) + list(wr)
        rw = T.Row(w, r, len(seq), np.arange(len(hist), len(seq) + 1), seq=seq, end=w.end_ids[r], key=key)
        rows.append(rw)
    out = {}
    for mb in T.micro_batches(rows, ctx["tok_budget"], ctx["n_prompt0"]):
        b = T.collate(mb, ctx["audio"], ctx["proc"], ctx["prompter"], False, ctx["perturb"], ctx["seed"], 0, "score",
                      feat_cache=ctx.get("feat_cache"))
        with torch.set_grad_enabled(grad):
            lt, le, lm = m.pos_logprobs(b, ctx["lp_chunk"], masked_final=bar)
        k = 0
        for rw in mb:
            n = len(rw.pos)
            if bar and rw.r == rw.win.R - 1:
                a, e = lm[0][k: k + n], lm[1][k: k + n]
            else:
                a, e = lt[k: k + n], le[k: k + n]
            out[rw.key] = (a[: n - 1], e)
            k += n
    return out


def pass1(m, wins, ctx, budget=None, nw=1):
    """the trainer's pass 1 (split over nw simulated ranks, each with its own packing) -> plans, res."""
    c = dict(ctx) if budget is None else {**ctx, "tok_budget": budget}
    plans = {w.idx: V2T.plan(w) for w in wins}
    res = {}
    for part in V2T.split(V2T.pass1_rows(wins, plans), nw, c["n_prompt0"]):
        r_, _ = V2T.score_batches(m, V2T.collate_all(part, c, 0, "score", False), c)
        res.update(r_)
    return plans, res


def brute_model(w, ctx, m, tau, bar):
    """brute force with the tiny model's psi (each distinct sequence scored alone, no grad)."""
    rec = w.rec
    R = rec["n_read"]
    seen = {}

    def psi_fn(r, final, hist, written):
        k = seq_key(r, hist, written)
        seen.setdefault(k, None)
        return 0.0
    VP.brute_paths_p(rec, V.lattice(rec), psi_fn)          # collect the keys
    sc = score_rows_alone(m, list(seen), {k: (w, k[0]) for k in seen}, ctx, bar)
    psi = {k: float(v[0].sum() + v[1][-1]) for k, v in sc.items()}
    del R
    return brute(rec, lambda r, final, hist, written: psi[seq_key(r, hist, written)], tau)


# ---------------------------------------------------------------- (1b) tiny model
def test_posterior_tiny_model():
    wins = small_choice_windows(3)
    ctx = vctx(perturb=0.0, budget=4000, bar=True)
    m = TT.tiny_model(seed=11)
    bias = torch.zeros(151936)
    bias[T.WAIT_ID] = 9.0          # p(<wait>) not tiny, so the order / deferral posterior is not degenerate
    bias[T.EOS_ID] = 6.0
    m.logits_at = lambda h: TT.ThinkerGStarModel.logits_at(m, h) + bias
    plans, res = pass1(m, wins, ctx)
    worst_m, worst_w, q_nf = 0.0, 0.0, []
    for w in wins:
        post = V2T.posterior(w, plans[w.idx], res, V2T.TAU)
        pp = VP.plan_window_p(w.rec, skip_forced=True)
        mp = {(q["r"], tuple(q["hist_ids"]), tuple(q["forced_ids"])): k for k, q in enumerate(plans[w.idx]["reqs"])}
        assert len(mp) == len(plans[w.idx]["reqs"])
        rp = [res[(w.idx, mp[(q["r"], tuple(q["hist_ids"]), tuple(q["forced_ids"]))])] for q in pp["reqs"]]
        fr = VP.forward_p(VP.assemble_p(pp, rp), V2T.TAU)
        assert abs(fr["logZ"] - post["logZ"]) < 1e-9
        worst_m = max(worst_m, cmp_marginals(pl_lat(plans, w), post, fr, 1e-9, "tiny vs forward_p"))
        lat, paths, scores, bf, Wb, _ = brute_model(w, ctx, m, V2T.TAU, True)
        worst_m = max(worst_m, cmp_marginals(lat, post, bf, 1e-4, "tiny vs brute"))
        worst_w = max(worst_w, cmp_weights(trainer_weights(w, plans[w.idx], post), Wb, 1e-4, "tiny weights"))
        q_nf.append(post["exp_nf"])
    assert max(q_nf) > 1e-3, q_nf          # the order latent carries mass in this test
    print(f"  (1b) tiny model, {len(wins)} windows with real order choices: trainer pass-1 posterior = precheck "
          f"forward_p (1e-9) and = brute force over (rho x order) with every sequence scored alone (marginals "
          f"{worst_m:.1e}, per-sequence weights {worst_w:.1e}); E[newfirst reads] {[round(x, 3) for x in q_nf]}")


def pl_lat(plans, w):
    return plans[w.idx]["lat"]


# ---------------------------------------------------------------- (2) EM surrogate and Fisher identity
def test_grad_equals_em_surrogate():
    for hier, tau, bar, fisher in ((True, V2T.TAU, True, False), (False, 1.0, False, True)):
        wins = small_choice_windows(2, hier=hier)
        ctx = vctx(perturb=0.2, budget=4000, bar=bar)
        m = TT.tiny_model(seed=13)
        bias = torch.zeros(151936)
        bias[T.WAIT_ID] = 9.0
        bias[T.EOS_ID] = 6.0
        m.logits_at = lambda h, m=m: TT.ThinkerGStarModel.logits_at(m, h) + bias
        D = 1.0 if fisher else T.step_totals(wins)[0]
        # trainer: pass 1 -> posterior -> pass-2 rows -> grad_pass
        plans, res = pass1(m, wins, ctx)
        posts = {w.idx: V2T.posterior(w, plans[w.idx], res, tau) for w in wins}
        rows = V2T.pass2_rows(wins, plans, posts, D)
        TT.zero(m)
        lg = T.grad_pass(m, rows, ctx, 0)
        g_tr = TT.grads(m)
        # reference: brute-force sequences scored one by one with grad
        TT.zero(m)
        total, logZs = 0.0, []
        for w in wins:
            lat, paths, scores, bf, Wb, keys = brute_model(w, ctx, m, tau, bar)
            ks = sorted(Wb)
            sc = score_rows_alone(m, ks, {k: (w, k[0]) for k in ks}, ctx, False, grad=True)
            if fisher:      # -log Z over the paths with differentiable psi (plain at every read: bar off)
                ps = torch.stack([sum(sc[k][0].sum() + sc[k][1][-1] for k in kk) for kk in keys]).double()
                logZ = torch.logsumexp(tau * ps, 0)
                (-logZ / D).backward()
                logZs.append(float(logZ))
            else:
                loss = 0.0
                for k in ks:
                    h = V2T.token_weights(w, k[2])
                    tl, ec = sc[k]
                    ll = (torch.as_tensor(h, dtype=torch.float32) * tl).sum() + ec[-1]
                    loss = loss - Wb[k] / D * ll
                loss.backward()
                total += float(loss)
        g_ref = TT.grads(m)
        err = TT.cmp_grads(g_tr, g_ref, 1e-4, "fisher" if fisher else "EM surrogate")
        if not fisher:
            assert abs(lg["loss_main"] - total) / abs(total) < 1e-4, (lg["loss_main"], total)
            print(f"  (2) pass-2 gradient = autograd of the EM surrogate over the brute-force sequences (tau 0.5, bar, "
                  f"h, D = C0's {D:.0f}): loss {lg['loss_main']:.6f} vs {total:.6f}, grads max rel {err:.1e}")
        else:
            print(f"  (2) Fisher identity (tau 1, bar off, h = 1, D = 1): pass-2 gradient = -grad log Z (brute force, "
                  f"log Z {[round(x, 2) for x in logZs]}), grads max rel {err:.1e}")


# ---------------------------------------------------------------- (3) collapse to the label path
def heuristic_vs_v2p(wins, ctx, m, what):
    D, A = T.step_totals(wins)
    c0 = T.train_rows(wins, {w.idx: T.heur_weights(w) for w in wins}, D, A, 0.0)
    plans = {w.idx: V2T.plan(w) for w in wins}
    assert not V2T.pass1_rows(wins, plans), "collapsed lattice: nothing to score"
    posts = {w.idx: V2T.posterior(w, plans[w.idx], {}, V2T.TAU) for w in wins}
    assert all(abs(p["q_label"] - 1.0) < 1e-15 for p in posts.values())
    rows = V2T.pass2_rows(wins, plans, posts, D)
    assert len(rows) == len(c0)
    for a, b in zip(c0, rows):
        assert a.win is b.win and a.r == b.r and a.K == b.K and np.array_equal(a.pos, b.pos)
        assert b.seq == a.win.S[: a.K]
        assert np.array_equal(a.c_tok, b.c_tok) and np.array_equal(a.c_end, b.c_end), what
        assert np.array_equal(a.c_state, b.c_state)
        assert np.array_equal(b.c_lab_tok, b.c_tok) and np.array_equal(b.c_lab_end, b.c_end)
    out = []
    for rs in (c0, rows):
        TT.zero(m)
        lg = T.grad_pass(m, rs, ctx, 0)
        out.append((lg, TT.grads(m)))
    (l0, g0), (l1, g1) = out
    assert l0["loss_main"] == l1["loss_main"], (l0["loss_main"], l1["loss_main"])
    assert l1["loss_label"] == l1["loss_main"] or abs(l1["loss_label"] - l1["loss_main"]) < 1e-6
    for k in g0:
        assert torch.equal(g0[k], g1[k]), (what, k)
    return l0["loss_main"]


def test_collapse_equals_heuristic():
    m = TT.tiny_model(seed=17)
    # (i) the lattice patched to the label path on windows that do have choices; both feature modes (the batched call
    # of C0, and --feat-per-row), each compared with --objective heuristic in the same mode
    wins = small_choice_windows(2)
    orig = V.read_states
    try:
        V.read_states = lambda rec, rho, Br, final: [tuple(V.kap(rec, rho, Br))]
        l_i = heuristic_vs_v2p(wins, vctx(perturb=0.2, per_row=False), m, "patched, batched features")
        heuristic_vs_v2p(wins, vctx(perturb=0.2, per_row=True), m, "patched, per-row features")
    finally:
        V.read_states = orig
    ctx = vctx(perturb=0.2, per_row=True)
    # (ii) real windows whose v2' lattice is a single path
    one = []
    for i, r in enumerate(recs0()):
        if r["n_tok"] and 2 <= r["n_read"] <= 4 and r["n_tok"] <= 40 and n_paths(r) == 1:
            one.append(T.make_window(i, r, True))
        if len(one) >= 2:
            break
    T.set_reach(one, 128)
    l_ii = heuristic_vs_v2p(one, ctx, m, "single-path windows")
    print(f"  (3) collapsed lattice (patched to the label, batched and per-row features; and {len(one)} real "
          f"single-path windows): rows = C0's rows, loss ({l_i:.6f}, {l_ii:.6f}) and gradients bit-identical to "
          f"--objective heuristic")


# ---------------------------------------------------------------- (4) packing invariance
def test_packing_invariance():
    wins = small_choice_windows(3, max_paths=60)
    ctx = vctx(perturb=0.2, budget=100000)
    m = TT.tiny_model(seed=19)
    D = T.step_totals(wins)[0]
    ref_post, ref_g = None, None
    worst_c, worst_g = 0.0, 0.0
    for budget, nw in ((100000, 1), (700, 1), (100000, 3), (900, 8)):
        plans, res = pass1(m, wins, ctx, budget, nw)
        posts = {w.idx: V2T.posterior(w, plans[w.idx], res, V2T.TAU) for w in wins}
        rows = V2T.pass2_rows(wins, plans, posts, D)
        TT.zero(m)
        mbs = 0
        for part in V2T.split(rows, nw, ctx["n_prompt0"]):
            mbs += T.grad_pass(m, part, {**ctx, "tok_budget": budget}, 0)["mbs"]
        g = TT.grads(m)
        cend = np.concatenate([np.concatenate(p["cend"]) for p in posts.values()])
        if ref_post is None:
            ref_post, ref_g = cend, g
            continue
        worst_c = max(worst_c, float(np.abs(cend - ref_post).max()))
        worst_g = max(worst_g, TT.cmp_grads(g, ref_g, 1e-4, f"budget {budget} ranks {nw} ({mbs} micro-batches)"))
    assert worst_c < 1e-5, worst_c
    # the same with C0's batched features: the inputs themselves depend on the packing (last frame of shorter rows)
    c2 = vctx(perturb=0.2, budget=100000, per_row=False)
    gs = []
    for budget in (100000, 700):
        plans, res = pass1(m, wins, c2, budget, 1)
        posts = {w.idx: V2T.posterior(w, plans[w.idx], res, V2T.TAU) for w in wins}
        TT.zero(m)
        T.grad_pass(m, V2T.pass2_rows(wins, plans, posts, D), {**c2, "tok_budget": budget}, 0)
        gs.append(TT.grads(m))
    d_b = max(float((gs[0][k] - gs[1][k]).abs().max()) / max(float(gs[1][k].abs().max()), 1e-12) for k in gs[0])
    print(f"  (4) packing with --feat-per-row (budget 1e5 / 700 / 900, 1 / 3 / 8 simulated ranks): posterior weights "
          f"max |d| {worst_c:.1e}, gradients max rel {worst_g:.1e} (fp32); with C0's batched features the same "
          f"repacking moves the gradients by {d_b:.1e} (input change, the reason for --feat-per-row)")


# ---------------------------------------------------------------- (5) weight sums
def test_weight_sums():
    n, n_ch = 0, 0
    for i, rec in enumerate(recs0()[:800]):
        if rec["n_tok"] == 0:
            continue
        w = T.make_window(i, rec, True)
        pl = V2T.plan(w)
        res = {(w.idx, k): v for k, (v, q) in enumerate(zip(V.dummy_res(pl), pl["reqs"])) if q["scored"]}
        post = V2T.posterior(w, pl, res, V2T.TAU)
        rows = V2T.pass2_rows([w], {w.idx: pl}, {w.idx: post}, 1.0)
        tot = sum(float(rw.c_tok.sum() + rw.c_end.sum()) for rw in rows)
        assert abs(tot - (w.h.sum() + w.R)) < 1e-9 * max(1.0, tot), (rec["id"], tot, w.h.sum() + w.R)
        lab = sum(float(rw.c_lab_tok.sum() + rw.c_lab_end.sum()) for rw in rows)
        assert abs(lab - (w.h.sum() + w.R)) < 1e-9 * max(1.0, lab)
        n += 1
        n_ch += any(order_marg(post))
    assert n_ch > 10
    print(f"  (5) {n} windows ({n_ch} with order choices): sum of pass-2 coefficients x D = sum h + R (C0's weight); "
          f"label coefficients the same; every read's edge-order mass = 1 (checked inside posterior)")


# ---------------------------------------------------------------- (6) fast MoE backward patch
def test_fast_moe_bwd():
    from transformers.integrations import moe

    g = torch.Generator().manual_seed(0)
    E_, d_in, d_out = 6, 16, 12
    counts = [3, 0, 5, 1, 0, 7]
    offs = torch.as_tensor(np.cumsum(counts), dtype=torch.int32)
    x = torch.randn(int(offs[-1]), d_in, generator=g)
    wt = torch.randn(E_, d_in, d_out, generator=g)
    gy = torch.randn(int(offs[-1]), d_out, generator=g)
    xa = x.clone().requires_grad_(True)
    ya = torch.ops.transformers.grouped_mm_fallback(xa, wt, offs)
    ya.backward(gy)
    xb = x.clone().requires_grad_(True)
    yb = V2T.FastGroupedMM.apply(xb, wt, offs)
    yb.backward(gy)
    assert torch.equal(ya, yb) and torch.equal(xa.grad, xb.grad)
    wc = wt.clone().requires_grad_(True)        # a trainable weight still gets its gradient
    xc = x.clone().requires_grad_(True)
    yc = V2T.FastGroupedMM.apply(xc, wc, offs)
    yc.backward(gy)
    wd = wt.clone().requires_grad_(True)
    xd = x.clone().requires_grad_(True)
    torch.ops.transformers.grouped_mm_fallback(xd, wd, offs).backward(gy)
    assert torch.equal(wc.grad, wd.grad) and torch.equal(xc.grad, xd.grad)
    # the patch routes moe._grouped_mm's fallback through it (and is undone)
    V2T.patch_fast_moe_bwd(True)
    try:
        xe = x.clone().requires_grad_(True)
        ye = moe._grouped_mm(xe, wt, offs)
        ye.backward(gy)
        assert torch.equal(ye, ya) and torch.equal(xe.grad, xa.grad)
    finally:
        V2T.patch_fast_moe_bwd(False)
    assert moe._grouped_mm is V2T._ORIG_GROUPED_MM
    print("  (6) --fast-moe-bwd: forward and input gradients bit-identical to transformers' grouped_mm fallback; "
          "the frozen expert weights get no gradient computed; a trainable weight still gets the same gradient")


def main():
    t0 = time.time()
    tests = [test_fast_moe_bwd, test_weight_sums, test_posterior_dummy_exact, test_collapse_equals_heuristic,
             test_posterior_tiny_model, test_grad_equals_em_surrogate, test_packing_invariance]
    for t in tests:
        t1 = time.time()
        t()
        print(f"PASS {t.__name__} ({time.time() - t1:.1f}s)", flush=True)
    print(f"ALL {len(tests)} PASSED ({time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()

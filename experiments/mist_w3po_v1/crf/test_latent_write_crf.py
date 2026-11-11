"""Tests of crf/latent_write_crf.py and crf/latent_write_train.py (T2), CPU only:

  (a) brute force: every assignment rho_i in [e_i, h_i] (non-decreasing along each speaker's order) x every per-read
      order (canon / newfirst where real), built without the lattice; v2''s dummy LM (a function of the whole prefix)
      tau psi - lam Lat: log Z, per-sequence weights (the trainer's cend), E_q[Lat] equal the lattice's (1e-9), for
      lam 0 / 0.3 / 2 and tau 0.5 / 1, on synthetic multi-speaker records and on real AMI-train windows widened to
      [rho - 1, rho + 1] (and on the T1/T2 piece data when it exists)
  (b) collapsed == v2': with h = min(R - 1, e + 1) the plan equals VP.plan_window_p field for field and the lam = 0
      posterior equals serial_v2p_train.posterior bit for bit (cend ==, log Z ==, q_label, exp_nf) on real sched_v1
      windows (and the piece data when it exists)
  (c) label path == C0: with h = e the rows are C0's rows and the loss / gradients are bit-identical to
      --objective heuristic (tiny model, real windows, per-row and batched features)
  (d) lam: E_q[Lat](lam) non-increasing; the solve hits the target (rel 1e-6); 0 when not binding; capped and flagged
      when the target is below the earliest path; T1's target from T2's sub-lattice == E[Lat] of the v2' lattice's own
      posterior on the same scores
  (e) the trainer step (latent_write_train.run_step -> grad_pass, world 1, tiny model, --final-read-full-post 1): finite
      loss, sum of the coefficients x D = sum h + R per step, lam > 0 binding, --prefetch 1 = 0 bit for bit

    PYTHONPATH=.:third_party/Megatron-LM CUDA_VISIBLE_DEVICES= python -m experiments.mist_w3po_v1.crf.test_latent_write_crf
"""
from __future__ import annotations

import itertools
import json
import math
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

if not torch.cuda.is_available():    # CPU-only run (CUDA_VISIBLE_DEVICES=): transformers' model modules probe the
    torch.accelerator.current_accelerator = lambda *a, **k: None   # accelerator at import; report "none" (test only)

from experiments.mist_w3po_v1.crf import latent_write_crf as LW
from experiments.mist_w3po_v1.crf import latent_write_train as LWT
from experiments.mist_w3po_v1.crf import precheck_serial as S
from experiments.mist_w3po_v1.crf import precheck_serial_v2 as V
from experiments.mist_w3po_v1.crf import precheck_serial_v2p as VP
from experiments.mist_w3po_v1.crf import serial_v2p_train as V2T
from experiments.mist_w3po_v1.crf import test_serial_v2p_train as TV
from experiments.mist_w3po_v1.crf import test_train_crf_sft as TT
from experiments.mist_w3po_v1.crf import train_crf_sft as T

torch.set_num_threads(8)
PIECE = "eval_outputs/mist_w3po_v1/crf/sched_piece_t0_v1"
NEW0, SPK0 = S.NEW0, S.SPK0


# ---------------------------------------------------------------- records
def synthetic(rng: np.random.Generator, n_read: int, n_spk: int, n_per: int, wide: int = 3) -> dict:
    """A random T2-format record: per speaker n_per pieces in time order (speakers overlap), e = a running max of a
    noisy proportional read, u = the first read >= source end + 1 s; S in (e, source end, source start) order."""
    dur = 500 * n_read - int(rng.integers(0, 400))
    t_ms = [min(500 * (r + 1), dur) for r in range(n_read)]
    raw = []
    for k in range(1, n_spk + 1):
        t = int(rng.integers(0, dur // 3))
        e_run = 0
        for p in range(n_per):
            a = min(t, dur - 50)
            b = min(dur, a + int(rng.integers(100, 900)))
            t = b + int(rng.integers(0, 200))
            e = min(n_read - 1, max(e_run, math.ceil(b / 500) - 1 + int(rng.integers(-wide, 2))))
            e = max(e, e_run, 0)
            e_run = e
            u = min(n_read - 1, next((r for r, x in enumerate(t_ms) if x >= b + 1000), n_read - 1))
            nw = int(rng.integers(1, 5))
            raw.append({"slot": k, "utt": k * 10, "k": p, "src_start_ms": a, "src_end_ms": b, "e_read": e,
                        "u_read": u, "n_words": nw, "text": " ".join(["w"] * nw)})
    raw.sort(key=lambda c: (c["e_read"], c["src_end_ms"], c["src_start_ms"], c["utt"], c["k"]))
    ids, chunks = [], []
    for c in raw:
        toks = [SPK0 + c["slot"] - 1, NEW0 + c["src_start_ms"] // 80] + [1000 + int(x) for x in
                                                                         rng.integers(0, 50, c["n_words"])] + \
               [NEW0 + c["src_end_ms"] // 80]
        c.update(tok0=len(ids), tok1=len(ids) + len(toks), commit_ms=c["src_end_ms"] + 1000, piece=f"p{len(chunks)}",
                 q_start=c["src_start_ms"] // 80, q_end=c["src_end_ms"] // 80)
        ids += toks
        chunks.append(c)
    beta = [len(ids) if r == n_read - 1 else sum(c["tok1"] - c["tok0"] for c in chunks if c["e_read"] <= r)
            for r in range(n_read)]
    return {"id": f"syn{int(rng.integers(1 << 30))}", "chunks": chunks, "S_ids": ids, "n_tok": len(ids),
            "n_read": n_read, "t_ms": t_ms, "beta_heur": beta, "dur_ms": dur}


def widen(rec: dict, back: int = 1) -> dict:
    """A real sched_v1 window as a T2 record: e = max(0, rho - back), u = rho (so h = min(R - 1, rho + 1)); S keeps its
    order (a monotone map of rho), beta recomputed for the e label path."""
    r2 = json.loads(json.dumps(rec))
    rho = S.chunk_reads(rec)
    R = rec["n_read"]
    for c, x in zip(r2["chunks"], rho):
        c["e_read"] = max(0, x - back)
        c["u_read"] = x
        c["n_words"] = len(c["text"].split())
    r2["beta_heur"] = [rec["n_tok"] if r == R - 1 else sum(c["tok1"] - c["tok0"] for c in r2["chunks"]
                                                           if c["e_read"] <= r) for r in range(R)]
    assert S.chunk_reads(r2) == [c["e_read"] for c in r2["chunks"]]
    return r2


def piece_recs(n: int, max_tok: int = 200) -> list[dict]:
    out = []
    fs = sorted(Path(PIECE).glob("shard_*.jsonl"))
    if not fs:
        return out
    for l in open(fs[0], encoding="utf-8"):
        r = json.loads(l)
        if 0 < r["n_tok"] <= max_tok:
            out.append(r)
        if len(out) >= n:
            break
    return out


class W_:
    """the minimum of T.Window the posterior functions read."""

    def __init__(self, idx, rec):
        self.idx, self.rec = idx, rec


def dummy_full(pl: dict) -> list:
    R = pl["lat"]["R"]
    return [V.dummy_scores(q["r"], q["r"] == R - 1, q["hist_ids"], q["forced_ids"], q["proto"]) for q in pl["reqs"]]


def all_scored(pl: dict) -> dict:
    for q in pl["reqs"]:
        q["scored"] = True
    return {(0, k): v for k, v in enumerate(dummy_full(pl))}


# ---------------------------------------------------------------- (a) brute force
def brute(rec: dict, mode: str, tau: float, lam: float) -> dict:
    e, h = LW.bounds(rec, mode)
    R, n = rec["n_read"], len(rec["chunks"])
    ch = rec["chunks"]
    by = defaultdict(list)
    for c in V.kap(rec, e, range(n)):
        by[ch[c]["slot"]].append(c)
    pairs = [(x, y) for it in by.values() for x, y in zip(it, it[1:])]
    keys, scores, lats = [], [], []
    for rho in itertools.product(*[range(e[c], h[c] + 1) for c in range(n)]):
        if any(rho[x] > rho[y] for x, y in pairs):
            continue
        per_read = []
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
        lat = sum(LW.n_words(ch[c]) * (rec["t_ms"][rho[c]] - ch[c]["src_end_ms"]) / 1000.0 for c in range(n))
        for combo in itertools.product(*per_read):
            sc = 0.0
            for (r, hi, wi) in combo:
                tl, ec = V.dummy_scores(r, r == R - 1, list(hi), list(wi), "new")
                sc += float(tl.sum() + ec[-1])
            keys.append(combo)
            scores.append(tau * sc - lam * lat)
            lats.append(lat)
    scores = np.asarray(scores)
    logZ = V.lse(scores)
    q = np.exp(scores - logZ)
    Wb = defaultdict(float)
    for combo, qq in zip(keys, q):
        for k in combo:
            Wb[k] += qq
    return {"logZ": logZ, "W": Wb, "Eq_lat": float((q * np.asarray(lats)).sum()), "n_paths": len(keys)}


def lattice_weights(pl: dict, post: dict) -> dict:
    Wt = defaultdict(float)
    for k, q in enumerate(pl["reqs"]):
        for b, c in enumerate(post["cend"][k]):
            if c != 0.0 or b == len(q["forced_ids"]):
                Wt[(q["r"], tuple(int(x) for x in q["hist_ids"]), tuple(int(x) for x in q["forced_ids"][:b]))] += c
    return Wt


def check_brute(rec: dict, mode: str, taus=(0.5, 1.0), lams=(0.0, 0.3, 2.0)) -> tuple[float, int]:
    worst = 0.0
    pl = LW.plan_window(rec, LW.lattice(rec, mode), mode=mode)
    res = all_scored(pl)
    w = W_(0, rec)
    n = 0
    for tau in taus:
        parts = LW.window_parts(w, pl, res, tau)
        for lam in lams:
            post = LW.posterior_from_parts(w, pl, parts, tau, lam)
            bf = brute(rec, mode, tau, lam)
            # lattice log Z is that of tau psi - lam Lat: same as brute
            d = [abs(post["logZ"] - bf["logZ"]), abs(post["Eq_lat"] - bf["Eq_lat"])]
            Wt = lattice_weights(pl, post)
            ks = set(Wt) | set(bf["W"])
            d.append(max(abs(Wt.get(k, 0.0) - bf["W"].get(k, 0.0)) for k in ks))
            miss = [k for k in ks if bf["W"].get(k, 0.0) > 1e-12 and k not in Wt]
            assert not miss, (rec["id"], len(miss))
            worst = max(worst, max(d))
            n = bf["n_paths"]
    assert worst < 1e-9, (rec["id"], mode, worst)
    return worst, n


def test_brute_force():
    rng = np.random.default_rng(5)
    worst, cnt, paths = 0.0, 0, []
    for _ in range(40):
        rec = synthetic(rng, int(rng.integers(2, 6)), int(rng.integers(1, 4)), int(rng.integers(1, 4)))
        lat = LW.lattice(rec, "t2")
        pl = LW.plan_window(rec, lat)
        npath = math.exp(V.forward_backward([np.where(x, 0.0, -np.inf) for x in pl["edge"]], 1.0)["logZ"])
        if npath > 3000:
            continue
        for mode in ("t2", "v2p"):
            wv, n = check_brute(rec, mode)
            worst = max(worst, wv)
            paths.append(n)
        cnt += 1
    nreal = 0
    for i, r in enumerate(TV.recs0()[:400]):
        if not (0 < r["n_tok"] <= 120 and r["n_read"] <= 6 and len(r["chunks"]) <= 6):
            continue
        rec = widen(r)
        pl = LW.plan_window(rec)
        npath = math.exp(V.forward_backward([np.where(x, 0.0, -np.inf) for x in pl["edge"]], 1.0)["logZ"])
        if npath > 2000 or npath < 3:
            continue
        wv, n = check_brute(rec, "t2", taus=(0.5,), lams=(0.0, 0.7))
        worst = max(worst, wv)
        nreal += 1
        if nreal >= 25:
            break
    npc = 0
    for rec in piece_recs(200):
        pl = LW.plan_window(rec)
        npath = math.exp(V.forward_backward([np.where(x, 0.0, -np.inf) for x in pl["edge"]], 1.0)["logZ"])
        if npath > 2000 or npath < 3:
            continue
        wv, _ = check_brute(rec, "t2", taus=(0.5,), lams=(0.0, 0.7))
        worst = max(worst, wv)
        npc += 1
        if npc >= 15:
            break
    assert cnt >= 20 and nreal >= 10, (cnt, nreal)
    print(f"  (a) brute force over (rho x order): {cnt} synthetic records (t2 and v2p bounds, tau 0.5/1, lam 0/0.3/2; "
          f"up to {max(paths)} paths), {nreal} widened real windows, {npc} piece-data windows: log Z, per-sequence "
          f"weights, E_q[Lat] max |d| {worst:.1e}")


# ---------------------------------------------------------------- (b) collapsed == v2'
def test_collapsed_equals_v2p():
    n, nreal, worst = 0, 0, 0
    recs = [r for r in TV.recs0()[:1500] if r["n_tok"] > 0]
    pcs = piece_recs(300)
    for i, rec in enumerate(recs + pcs):
        a = VP.plan_window_p(rec, V.lattice(rec), skip_forced=False)
        b = LW.plan_window(rec, LW.lattice(rec, "v2p"), mode="v2p")
        assert a["reqs"] == b["reqs"], rec["id"]
        assert a["reads"] == b["reads"], rec["id"]
        w = T.make_window(i, rec, True)
        pa, pb = V2T.plan(w), LWT.plan(w, "v2p")
        assert pa["reqs"] == pb["reqs"] and pa["reads"] == pb["reads"]
        res = {(w.idx, k): v for k, (v, q) in enumerate(zip(V.dummy_res(pa), pa["reqs"])) if q["scored"]}
        x, y = V2T.posterior(w, pa, res, V2T.TAU), LW.posterior(w, pb, res, V2T.TAU, 0.0)
        assert x["logZ"] == y["logZ"] and x["exp_nf"] == y["exp_nf"] and x["n_real"] == y["n_real"], rec["id"]
        assert x["q_label"] == y["q_label"], rec["id"]
        assert abs(x["exp_defer"] - y["exp_defer"]) < 1e-12, (rec["id"], x["exp_defer"], y["exp_defer"])
        for c1, c2 in zip(x["cend"], y["cend"]):
            assert np.array_equal(c1, c2), rec["id"]
        n += 1
        nreal += x["n_real"] > 0
    print(f"  (b) collapsed (h = min(R - 1, e + 1)): {len(recs)} sched_v1 + {len(pcs)} piece windows ({nreal} with real "
          f"order choices): plan == VP.plan_window_p field for field; lam = 0 posterior == serial_v2p_train.posterior "
          f"bit for bit (log Z, cend, q_label, exp_nf)")


# ---------------------------------------------------------------- (c) label path == C0
def label_vs_c0(wins, ctx, m, what):
    D, A = T.step_totals(wins)
    c0 = T.train_rows(wins, {w.idx: T.heur_weights(w) for w in wins}, D, A, 0.0)
    plans = {w.idx: LWT.plan(w, "label") for w in wins}
    assert not V2T.pass1_rows(wins, plans), "label lattice: nothing to score"
    posts = {w.idx: LW.posterior(w, plans[w.idx], {}, LW.TAU, 0.0) for w in wins}
    assert all(abs(p["q_label"] - 1.0) < 1e-15 for p in posts.values())
    rows = V2T.pass2_rows(wins, plans, posts, D)
    assert len(rows) == len(c0)
    for a, b in zip(c0, rows):
        assert a.win is b.win and a.r == b.r and a.K == b.K and np.array_equal(a.pos, b.pos)
        assert b.seq == a.win.S[: a.K]
        assert np.array_equal(a.c_tok, b.c_tok) and np.array_equal(a.c_end, b.c_end), what
    out = []
    for rs in (c0, rows):
        TT.zero(m)
        lg = T.grad_pass(m, rs, ctx, 0)
        out.append((lg, TT.grads(m)))
    (l0, g0), (l1, g1) = out
    assert l0["loss_main"] == l1["loss_main"], (l0["loss_main"], l1["loss_main"])
    for k in g0:
        assert torch.equal(g0[k], g1[k]), (what, k)
    return l0["loss_main"]


def test_label_equals_c0():
    m = TT.tiny_model(seed=17)
    wins = TV.small_choice_windows(2)
    l1 = label_vs_c0(wins, TV.vctx(perturb=0.2, per_row=False), m, "batched features")
    l2 = label_vs_c0(wins, TV.vctx(perturb=0.2, per_row=True), m, "per-row features")
    wd = [T.make_window(w.idx, widen(w.rec), True) for w in wins]
    T.set_reach(wd, 128)
    l3 = label_vs_c0(wd, TV.vctx(perturb=0.2, per_row=True), m, "widened labels")
    print(f"  (c) label path (h = e): rows = C0's rows, loss ({l1:.6f}, {l2:.6f}, widened {l3:.6f}) and gradients "
          f"bit-identical to --objective heuristic (batched and per-row features)")


# ---------------------------------------------------------------- (d) lambda
def test_lambda():
    rng = np.random.default_rng(9)
    parts, t1chk = [], 0.0
    recs = [widen(r) for r in TV.recs0()[:300] if 0 < r["n_tok"] <= 200 and r["n_read"] >= 3][:24]
    for i, rec in enumerate(recs):
        w = T.make_window(i, rec, True)
        pl = LWT.plan(w)
        res = {(w.idx, k): v for k, (v, q) in enumerate(zip(dummy_full(pl), pl["reqs"])) if q["scored"]}
        p = LW.window_parts(w, pl, res, LW.TAU)
        parts.append(p)
        # T1 target from T2's sub-lattice == E[Lat] of the v2' lattice's own posterior (same dummy scores)
        pv = LWT.plan(w, "v2p")
        rv = {(w.idx, k): v for k, (v, q) in enumerate(zip(dummy_full(pv), pv["reqs"])) if q["scored"]}
        post_v = LW.posterior(w, pv, rv, LW.TAU, 0.0)
        t1chk = max(t1chk, abs(post_v["Eq_lat"] - p["target"]))
        assert p["target"] >= p["lat_label"] - 1e-9
    assert t1chk < 1e-9, t1chk
    grid = [0.0, 0.01, 0.1, 0.3, 1.0, 3.0, 10.0, 100.0]
    fs = [sum(LW.exp_lat(LW.with_lam(p["Eo"], p["L"], g), p["L"])[0] for p in parts) for g in grid]
    assert all(a >= b - 1e-9 for a, b in zip(fs, fs[1:])), fs
    lam, info = LW.solve_lambda(parts)
    got = sum(LW.exp_lat(LW.with_lam(p["Eo"], p["L"], lam), p["L"])[0] for p in parts)
    if info["binding"]:
        assert info["bracketed"] and abs(got - info["target"]) <= 1e-6 * abs(info["target"]) + 1e-6, (got, info)
    # not binding: raise every target above E_{q_0}
    loose = [dict(p, target=p["target"] + 100.0) for p in parts]
    l0, i0 = LW.solve_lambda(loose)
    assert l0 == 0.0 and not i0["binding"]
    # binding with a target between the earliest path and E_{q_0}: hits it
    mid = [dict(p, target=0.5 * (p["lat_label"] + LW.exp_lat(p["Eo"], p["L"])[0])) for p in parts]
    lm, im = LW.solve_lambda(mid)
    gm = sum(LW.exp_lat(LW.with_lam(p["Eo"], p["L"], lm), p["L"])[0] for p in mid)
    assert im["binding"] and im["bracketed"] and lm > 0 and abs(gm - im["target"]) <= 1e-6 * abs(im["target"]) + 1e-6
    # impossible: below the earliest path -> capped and flagged
    imp = [dict(p, target=p["lat_label"] - 5.0) for p in parts]
    lc, ic = LW.solve_lambda(imp)
    assert lc == LW.LAM_MAX and ic["capped"] and not ic["bracketed"]
    del rng
    print(f"  (d) lam: E_q[Lat] non-increasing on a grid ({fs[0]:.2f} -> {fs[-1]:.2f}); T1 target from T2's sub-lattice "
          f"== the v2' lattice's own E[Lat] (max |d| {t1chk:.1e}); dummy-score solve lam {lam:.4f} (binding "
          f"{info['binding']}); a mid target is hit (lam {lm:.4f}, rel err < 1e-6); non-binding -> 0; impossible -> "
          f"capped {LW.LAM_MAX:g}, flagged")


# ---------------------------------------------------------------- (e) trainer step
def test_run_step():
    base = TV.small_choice_windows(3, max_paths=60)
    wins = [T.make_window(w.idx, widen(w.rec), True) for w in base]
    T.set_reach(wins, 128)
    m = TT.tiny_model(seed=23)
    bias = torch.zeros(151936)
    bias[T.WAIT_ID] = 9.0
    bias[T.EOS_ID] = 6.0
    m.logits_at = lambda h: TT.ThinkerGStarModel.logits_at(m, h) + bias
    D = T.step_totals(wins)[0]
    out = []
    for pf in (False, True):
        ctx = {**TV.vctx(perturb=0.2, budget=1500), "final_full_post": True}
        TT.zero(m)
        v = LWT.run_step(m, wins, D, ctx, 3, LW.TAU, 0, 1, prefetch=pf)
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
    need = {"lam", "lam_bracketed", "Eq_lat_T2", "Eq_lat_T1_target", "lat_label", "sum_words", "q_label", "exp_nf",
            "sft_loss", "rows_pass1", "rows_pass2"}
    assert need <= set(r0), need - set(r0)
    if st["lam_binding"]:
        assert abs(st["Eq_lat_T2"] - st["Eq_lat_T1_target"]) < 1e-6 * max(1.0, abs(st["Eq_lat_T1_target"]))
    assert st["lat_label"] <= st["Eq_lat_T1_target"] + 1e-9
    print(f"  (e) step path (tiny model, widened real windows, world 1): loss {l0['loss_main']:.6f} finite, prefetch "
          f"1 = 0 bit for bit; sum coef x D = sum h + R ({want:.0f}); lam {st['lam']:.4f} (binding "
          f"{st['lam_binding']}), E[Lat]/word T2 {st['Eq_lat_T2']:.3f} vs T1 target {st['Eq_lat_T1_target']:.3f} "
          f"(label {st['lat_label']:.3f}, lam 0 {st['Eq_lat_lam0']:.3f}); pass-1 rows {st['rows_pass1']}, pass-2 "
          f"{st['rows_pass2']}")


def main():
    t0 = time.time()
    tests = [test_lambda, test_collapsed_equals_v2p, test_brute_force, test_label_equals_c0, test_run_step]
    for t in tests:
        t1 = time.time()
        t()
        print(f"PASS {t.__name__} ({time.time() - t1:.1f}s)", flush=True)
    print(f"ALL {len(tests)} PASSED ({time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()

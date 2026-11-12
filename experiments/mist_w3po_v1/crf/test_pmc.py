"""Tests of PMC (reports/mist_w3po_v1/thinker_pmc_prereg_v1.zh-CN.md §2 P1, P2, P5): latent_write_crf / latent_write_train
mode "pmc", CPU only.

  P1 brute force: every assignment z_i in [l_i, h_i] = [max(0, e_i - 1), min(R - 1, e_i + 1)] (non-decreasing along each
     speaker's order) x every per-read order (canon / newfirst where real), built without the lattice, on v2''s dummy
     LM: (a) q_lam (tau psi - lam Lat): log Z, per-sequence weights (cend), E_q[Lat], E[#early chunks] equal the
     lattice's (1e-9); (b) q^pol (tau phi - lam Lat, phi = sum_{j<k} log(1 - p_end(b_j)) + log p_end(b_k) computed here
     from the dummy ec of each path's own write sequence): per-sequence weights, E[Lat], E[#early]; lam 0 / 0.3 / 2,
     tau 0.5 / 1; synthetic multi-speaker records and real VE1 windows (sched_t5_merge3_w10_v1); the budget == the
     e path's Lat, which is a path of the lattice
  P2 default modes bit for bit: t2 / t5 / v2p / label on 200 real VE1 windows: plan (reqs, reads, L, edge, t1),
     lattice, window_parts (target, lat_label), posterior (log Z, q_label, Eq_lat, exp_nf, cend) of the new module ==
     those of the module before PMC (git show <BASE>:latent_write_crf.py), with dummy scores, lam 0 and 0.37
  P5 the trainer step (latent_write_train.run_step mode pmc -> grad_pass, tiny model, world 1, --final-read-full-post 1):
     finite loss, --prefetch 1 == 0 bit for bit; sum c_tok x D = 2 sum h (q_lam and q^pol each write every token once),
     sum c_end x D = sum R; the coefficients == V2T.set_coefs + h suf(cend_pol) / D; a binding lam holds E_q[Lat] at the
     e-path budget; --lw-mode pmc parses (train_crf_sft.build_args)

    PYTHONPATH=.:third_party/Megatron-LM CUDA_VISIBLE_DEVICES= python -m experiments.mist_w3po_v1.crf.test_pmc
"""
from __future__ import annotations

import importlib.util
import itertools
import json
import math
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

if not torch.cuda.is_available():    # CPU-only run: transformers' model modules probe the accelerator at import
    torch.accelerator.current_accelerator = lambda *a, **k: None

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
VE1 = "eval_outputs/mist_w3po_v1/crf/sched_t5_merge3_w10_v1"
BASE = "46cc50e4"     # the commit of the PMC pre-registration: latent_write_crf.py before PMC


def ve1_recs(n: int, shard: int = 0) -> list[dict]:
    out = []
    for l in open(Path(VE1) / f"shard_{shard:03d}.jsonl", encoding="utf-8"):
        r = json.loads(l)
        if r["n_tok"] > 0:
            out.append(r)
        if len(out) >= n:
            break
    return out


def n_paths(pl: dict) -> float:
    return math.exp(V.forward_backward([np.where(x, 0.0, -np.inf) for x in pl["edge"]], 1.0)["logZ"])


# ---------------------------------------------------------------- P1 brute force
def _l1m(x: float) -> float:
    return math.log1p(-math.exp(min(x, -1e-15)))


def brute_pmc(rec: dict, tau: float, lam: float, pot: str) -> dict:
    e, h = LW.bounds(rec, "pmc")
    lo = [max(0, x - 1) for x in e]
    R, n = rec["n_read"], len(rec["chunks"])
    ch = rec["chunks"]
    by = defaultdict(list)
    for c in V.kap(rec, e, range(n)):
        by[ch[c]["slot"]].append(c)
    pairs = [(x, y) for it in by.values() for x, y in zip(it, it[1:])]
    keys, scores, lats, early = [], [], [], []
    for z in itertools.product(*[range(lo[c], h[c] + 1) for c in range(n)]):
        if any(z[x] > z[y] for x, y in pairs):
            continue
        per_read = []
        for r in range(R):
            hist = V.kap(rec, e, [c for c in range(n) if z[c] <= r - 1])
            Wr = V.kap(rec, e, [c for c in range(n) if z[c] == r])
            opts = [Wr]
            carried = [c for c in Wr if e[c] <= r - 1]
            if carried:
                ps = {ch[c]["slot"] for c in carried}
                N = [c for c in Wr if e[c] == r and ch[c]["slot"] not in ps]
                if N:
                    opts.append(N + V.kap(rec, e, [c for c in Wr if c not in set(N)]))
            per_read.append([(r, tuple(S.ids_of(rec, hist)), tuple(S.ids_of(rec, o)),
                              tuple(int(x) for x in np.cumsum([0] + [ch[c]["tok1"] - ch[c]["tok0"] for c in o])))
                             for o in opts])
        lat = sum(LW.n_words(ch[c]) * (rec["t_ms"][z[c]] - ch[c]["src_end_ms"]) / 1000.0 for c in range(n))
        ne = sum(1 for c in range(n) if z[c] < e[c])
        for combo in itertools.product(*per_read):
            sc = 0.0
            for (r, hi, wi, offs) in combo:
                tl, ec = V.dummy_scores(r, r == R - 1, list(hi), list(wi), "new")
                if pot == "psi":
                    sc += float(tl.sum() + ec[-1])
                else:
                    sc += sum(_l1m(float(ec[b])) for b in offs[:-1]) + float(ec[offs[-1]])
            keys.append(tuple((r, hi, wi) for (r, hi, wi, _o) in combo))
            scores.append(tau * sc - lam * lat)
            lats.append(lat)
            early.append(ne)
    scores = np.asarray(scores)
    logZ = V.lse(scores)
    q = np.exp(scores - logZ)
    Wb = defaultdict(float)
    for combo, qq in zip(keys, q):
        for k in combo:
            Wb[k] += qq
    e_lat = sum(LW.n_words(ch[c]) * (rec["t_ms"][e[c]] - ch[c]["src_end_ms"]) / 1000.0 for c in range(n))
    return {"logZ": logZ, "W": Wb, "Eq_lat": float((q * np.asarray(lats)).sum()),
            "early": float((q * np.asarray(early)).sum()), "n_paths": len(keys), "e_lat": e_lat}


def _wdiff(pl: dict, post: dict, bf: dict, rid: str) -> float:
    Wt = TL.lattice_weights(pl, post)
    ks = set(Wt) | set(bf["W"])
    miss = [k for k in ks if bf["W"].get(k, 0.0) > 1e-12 and k not in Wt]
    assert not miss, (rid, len(miss))
    return max(abs(Wt.get(k, 0.0) - bf["W"].get(k, 0.0)) for k in ks)


def check_pmc(rec: dict, taus=(0.5, 1.0), lams=(0.0, 0.3, 2.0)) -> tuple[float, int]:
    worst = 0.0
    lat = LW.lattice(rec, "pmc")
    pl = LW.plan_window(rec, lat, mode="pmc")
    res = TL.all_scored(pl)
    w = TL.W_(0, rec)
    n = 0
    for tau in taus:
        parts = LW.window_parts(w, pl, res, tau)
        pp = LW.pol_parts(w, pl, res, parts, tau)
        for lam in lams:
            post = LW.posterior_from_parts(w, pl, parts, tau, lam)
            pol = LW.posterior_from_parts(w, pl, pp, tau, lam)
            bf = brute_pmc(rec, tau, lam, "psi")
            bp = brute_pmc(rec, tau, lam, "phi")
            assert abs(parts["target"] - bf["e_lat"]) < 1e-9, (rec["id"], parts["target"], bf["e_lat"])
            d = [abs(post["logZ"] - bf["logZ"]), abs(post["Eq_lat"] - bf["Eq_lat"]), _wdiff(pl, post, bf, rec["id"]),
                 abs(LW.exp_early(pl, post["fb"]) - bf["early"]),
                 abs(pol["Eq_lat"] - bp["Eq_lat"]), _wdiff(pl, pol, bp, rec["id"]),
                 abs(LW.exp_early(pl, pol["fb"]) - bp["early"])]
            worst = max(worst, max(d))
            n = bf["n_paths"]
    assert worst < 1e-9, (rec["id"], worst)
    return worst, n


def test_p1_brute_force():
    rng = np.random.default_rng(11)
    worst, cnt, paths, early_seen = 0.0, 0, [], False
    for _ in range(60):
        rec = TL.synthetic(rng, int(rng.integers(2, 6)), int(rng.integers(1, 4)), int(rng.integers(1, 4)))
        pl = LW.plan_window(rec, LW.lattice(rec, "pmc"), mode="pmc")
        if n_paths(pl) > 3000:
            continue
        wv, n = check_pmc(rec)
        worst = max(worst, wv)
        paths.append(n)
        cnt += 1
        early_seen |= any(c["e_read"] > 0 for c in rec["chunks"])
    nreal, rp = 0, []
    for rec in ve1_recs(3000):
        if not (rec["n_tok"] <= 160 and rec["n_read"] <= 7 and len(rec["chunks"]) <= 6):
            continue
        pl = LW.plan_window(rec, LW.lattice(rec, "pmc"), mode="pmc")
        npath = n_paths(pl)
        if npath > 2000 or npath < 3:
            continue
        wv, n = check_pmc(rec, taus=(0.5,), lams=(0.0, 0.7))
        worst = max(worst, wv)
        rp.append(n)
        nreal += 1
        if nreal >= 30:
            break
    assert cnt >= 30 and nreal >= 15 and early_seen, (cnt, nreal)
    print(f"  P1 brute force over (z in [e-1, e+1] x order): {cnt} synthetic records (up to {max(paths)} paths; tau "
          f"0.5/1, lam 0/0.3/2) + {nreal} real VE1 windows (up to {max(rp)} paths; tau 0.5, lam 0/0.7): q_lam log Z, "
          f"cend, E[Lat], E[#early] and q^pol cend, E[Lat], E[#early] max |d| {worst:.1e}; budget == e-path Lat")


# ---------------------------------------------------------------- P2 default modes bit for bit
def old_module():
    src = subprocess.check_output(["git", "show", f"{BASE}:experiments/mist_w3po_v1/crf/latent_write_crf.py"])
    assert b'"pmc"' not in src
    spec = importlib.util.spec_from_loader("latent_write_crf_before_pmc", loader=None)
    mod = importlib.util.module_from_spec(spec)
    exec(compile(src, "latent_write_crf_before_pmc.py", "exec"), mod.__dict__)
    sys.modules["latent_write_crf_before_pmc"] = mod
    return mod


def _plan_eq(a: dict, b: dict, rid: str) -> None:
    assert a["reqs"] == b["reqs"] and a["reads"] == b["reads"], rid
    assert a["W"] == b["W"], rid
    for x, y in zip(a["L"], b["L"]):
        assert np.array_equal(x, y), rid
    for x, y in zip(a["edge"], b["edge"]):
        assert np.array_equal(x, y), rid
    for (p1, c1), (p2, c2) in zip(a["t1"], b["t1"]):
        assert np.array_equal(p1, p2) and np.array_equal(c1, c2), rid
    la, lb = a["lat"], dict(b["lat"])
    assert lb.pop("lo") == lb["rho"], rid
    assert la == lb, rid


def test_p2_default_bit_identical():
    old = old_module()
    recs = [r for r in ve1_recs(2000) if r["n_read"] <= 12][:200]
    assert len(recs) == 200
    cnt = defaultdict(int)
    for i, rec in enumerate(recs):
        w = T.make_window(i, rec, True)
        for mode in ("t2", "t5", "v2p", "label"):
            assert old.bounds(rec, mode) == LW.bounds(rec, mode)
            pa = old.plan_window(rec, old.lattice(rec, mode), skip_forced=False, mode=mode)
            pb = LWT.plan(w, mode)
            forced = {rd["r"] for rd in pa["reads"] if rd["forced"]}
            for q in pa["reqs"]:
                q["scored"] = q["r"] not in forced
            _plan_eq(pa, pb, rec["id"])
            res = {(w.idx, k): v for k, (v, q) in enumerate(zip(V.dummy_res(pb), pb["reqs"])) if q["scored"]}
            xa, xb = old.window_parts(w, pa, res, LW.TAU), LW.window_parts(w, pb, res, LW.TAU)
            assert xa["target"] == xb["target"] and xa["lat_label"] == xb["lat_label"], (rec["id"], mode)
            for lam in (0.0, 0.37):
                ya = old.posterior_from_parts(w, pa, xa, LW.TAU, lam)
                yb = LW.posterior_from_parts(w, pb, xb, LW.TAU, lam)
                for k in ("logZ", "q_label", "Eq_lat", "exp_nf", "n_real", "exp_defer"):
                    assert ya[k] == yb[k], (rec["id"], mode, k)
                for c1, c2 in zip(ya["cend"], yb["cend"]):
                    assert np.array_equal(c1, c2), (rec["id"], mode)
            cnt[mode] += 1
    print(f"  P2 default modes: {dict(cnt)} real VE1 windows: plan, lattice, target / lat_label, posterior (log Z, "
          f"q_label, E[Lat], exp_nf, cend; lam 0 / 0.37) == the module before PMC ({BASE}) bit for bit")


# ---------------------------------------------------------------- P5 trainer step
def test_p5_run_step():
    a = T.build_args(["--objective", "latent_write", "--lw-mode", "pmc", "--out", "/nonexistent/x"])
    assert a.lw_mode == "pmc"
    base = TV.small_choice_windows(3, max_paths=60)
    wins = [T.make_window(w.idx, TL.widen(w.rec), True) for w in base]
    T.set_reach(wins, 128)
    m = TT.tiny_model(seed=29)
    bias = torch.zeros(151936)
    bias[T.WAIT_ID] = 9.0
    bias[T.EOS_ID] = 6.0
    m.logits_at = lambda h: TT.ThinkerGStarModel.logits_at(m, h) + bias
    D = T.step_totals(wins)[0]
    out = []
    for pf in (False, True):
        ctx = {**TV.vctx(perturb=0.2, budget=1500), "final_full_post": True}
        TT.zero(m)
        v = LWT.run_step(m, wins, D, ctx, 3, LW.TAU, 0, 1, prefetch=pf, mode="pmc")
        ctx["prefetched_batches"] = v["batches"]
        lg = T.grad_pass(m, v["rows"], ctx, 3)
        V2T.reduce_logs(v, lg, "cpu")
        rec = V2T.metrics_fields(v, lg, {})
        out.append((v, lg, TT.grads(m), rec))
    (v0, l0, g0, r0), (v1, l1, g1, r1) = out
    assert math.isfinite(l0["loss_main"]) and l0["loss_main"] == l1["loss_main"]
    for k in g0:
        assert torch.equal(g0[k], g1[k]), k
    tok = sum(float(rw.c_tok.sum()) for rw in v0["rows"]) * D
    end = sum(float(rw.c_end.sum()) for rw in v0["rows"]) * D
    sh = sum(float(w.h.sum()) for w in wins)
    sr = sum(float(w.R) for w in wins)
    assert abs(tok - 2.0 * sh) < 1e-9 * sh and abs(end - sr) < 1e-9 * sr, (tok, 2 * sh, end, sr)
    st = v0["stats"]
    need = {"lam", "lam_bracketed", "Eq_lat_T2", "Eq_lat_T1_target", "Eq_lat_pol", "lat_e", "exp_early_q",
            "exp_early_pol", "n_chunks"}
    assert need <= set(st), need - set(st)
    assert abs(st["lat_e"] - st["Eq_lat_T1_target"]) < 1e-9, st
    if st["lam_binding"]:
        assert abs(st["Eq_lat_T2"] - st["lat_e"]) < 1e-6 * max(1.0, abs(st["lat_e"])), st
    # the coefficients == V2T.set_coefs (q_lam) + h suf(cend_pol) / D, recomputed from scratch on the same scores
    plans = {w.idx: LWT.plan(w, "pmc") for w in wins}
    res = {(w.idx, k): v for w in wins for k, (v, q) in enumerate(zip(TL.dummy_full(plans[w.idx]),
                                                                       plans[w.idx]["reqs"])) if q["scored"]}
    parts = {w.idx: LW.window_parts(w, plans[w.idx], res, LW.TAU) for w in wins}
    lam, _ = LW.solve_lambda(list(parts.values()))
    posts = {w.idx: LW.posterior_from_parts(w, plans[w.idx], parts[w.idx], LW.TAU, lam) for w in wins}
    pols = {w.idx: LW.posterior_from_parts(w, plans[w.idx], LW.pol_parts(w, plans[w.idx], res, parts[w.idx]),
                                           LW.TAU, lam) for w in wins}
    ra = V2T.pass2_rows(wins, plans, None, D)
    rb = V2T.pass2_rows(wins, plans, None, D)
    V2T.set_coefs(ra, plans, posts, D)
    LWT.set_coefs_pmc(rb, plans, posts, pols, D)
    worst = 0.0
    for x, y in zip(ra, rb):
        widx, k = x.key
        q = plans[widx]["reqs"][k]
        mm = len(q["forced_ids"])
        h = V2T.token_weights(x.win, q["forced_ids"])
        cp = pols[widx]["cend"][k]
        want = x.c_tok.copy()
        want[:mm] += h * np.array([cp[t + 1:].sum() for t in range(mm)]) / D
        worst = max(worst, float(np.abs(y.c_tok - want).max()))
        assert np.array_equal(x.c_end, y.c_end)
    assert worst < 1e-12, worst
    print(f"  P5 step path (tiny model, widened real windows, world 1, mode pmc): loss {l0['loss_main']:.6f} finite, "
          f"prefetch 1 = 0 bit for bit; sum c_tok x D = 2 sum h ({2 * sh:.0f}), sum c_end x D = sum R ({sr:.0f}); "
          f"coefficients == set_coefs + h suf(cend_pol) / D (max |d| {worst:.1e}); lam {st['lam']:.4f} (binding "
          f"{st['lam_binding']}), E[Lat]/word q {st['Eq_lat_T2']:.3f}, q^pol {st['Eq_lat_pol']:.3f}, budget e path "
          f"{st['lat_e']:.3f}; E[#early] q {st['exp_early_q']:.2f}, q^pol {st['exp_early_pol']:.2f} of "
          f"{st['n_chunks']} chunks; --lw-mode pmc parses")


def main():
    t0 = time.time()
    tests = [test_p1_brute_force, test_p2_default_bit_identical, test_p5_run_step]
    for t in tests:
        t1 = time.time()
        t()
        print(f"PASS {t.__name__} ({time.time() - t1:.1f}s)", flush=True)
    print(f"ALL {len(tests)} PASSED ({time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()

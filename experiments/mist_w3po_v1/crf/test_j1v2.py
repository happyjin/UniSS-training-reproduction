"""PC1 of J1-v2 (reports/mist_w3po_v1/thinker_joint_crf_v2_prereg_v1.zh-CN.md §3), CPU only:

  (1) brute force in lattice mode v2p (VE1's): every rho in {e, e + 1} (non-decreasing per speaker) x per-read order x
      per-(read, speaker) rate class, playback simulated per path without the lattice: log Z, per-sequence weights,
      E_q[l], E_q[blocks] equal the augmented chain's to 1e-9 (test_j1.chain_vs_brute with mode v2p)
  (2) the shift budget: window_parts_shift's E0 = the brute-force lam = 0 E[l] and l_e = the brute force restricted to
      the label path (every chunk at e_i), 1e-9; budget_shift = max(E0 - delta W, l_e) by definition; solve_lambda_to
      hits T (relative residual <= 1e-6) when binding, lam = 0 when T >= E0, and with T = sum of T1 targets returns
      solve_lambda's lam / info bit for bit; under the label mask E_q[blocks] = the label path's (read, speaker) groups
  (3) defaults: joint_speech_train.run_step's default call and budget "t1" equal the module before J1-v2 (git show of
      the parent commit, loaded as a separate module) bit for bit (rows, coefficients, stats, loss, gradients), for
      T3's default call and J1's (t5 + DART); budget "shift" (v2p + DART): finite loss, the J1-v2 log fields, the
      step's E_q[l] = budget_T x words
  (4) train_crf_sft's arguments: --js-budget shift only with --objective joint_speech, --js-delta only with shift,
      defaults t1 / 0.134

    PYTHONPATH=.:third_party/Megatron-LM CUDA_VISIBLE_DEVICES= python -m experiments.mist_w3po_v1.crf.test_j1v2
"""
from __future__ import annotations

import importlib.util
import math
import subprocess
import sys
import time

import numpy as np
import torch

if not torch.cuda.is_available():
    torch.accelerator.current_accelerator = lambda *a, **k: None

from experiments.mist_w3po_v1.crf import joint_speech_crf as JS
from experiments.mist_w3po_v1.crf import joint_speech_train as JST
from experiments.mist_w3po_v1.crf import latent_write_crf as LW
from experiments.mist_w3po_v1.crf import precheck_serial as S
from experiments.mist_w3po_v1.crf import test_j1 as TJ1
from experiments.mist_w3po_v1.crf import test_joint_speech_crf as TJ
from experiments.mist_w3po_v1.crf import test_latent_write_crf as TL
from experiments.mist_w3po_v1.crf import test_serial_v2p_train as TV
from experiments.mist_w3po_v1.crf import test_train_crf_sft as TT
from experiments.mist_w3po_v1.crf import train_crf_sft as T

torch.set_num_threads(8)
BASE = "625a1926"   # the J1-v2 pre-registration commit: joint_speech_* before the J1-v2 code


def test_brute_v2p():
    tm = TJ.toy_talker()
    rng = np.random.default_rng(21)
    worst, paths, multi = 0.0, [], 0
    recs = TJ1.records(rng, 30, "v2p")
    for rec in recs:
        wv, n = TJ1.chain_vs_brute(rec, tm, "v2p")
        worst = max(worst, wv)
        paths.append(n)
        multi += len({c["slot"] for c in rec["chunks"]}) > 1
    assert worst < 1e-9 and multi >= 4, (worst, multi)
    print(f"  (1) v2p brute force (rho x order x rate): log Z, weights, E_q[l], E_q[blocks] max |d| {worst:.1e} on "
          f"{len(recs)} windows (up to {max(paths)} paths, {multi} multi-speaker)")
    return worst


def test_budget():
    tm = TJ.toy_talker()
    rng = np.random.default_rng(23)
    recs = TJ1.records(rng, 24, "v2p")
    worst, parts = 0.0, []
    for rec in recs:
        pl = LW.plan_window(rec, LW.lattice(rec, "v2p"), mode="v2p")
        res = TL.all_scored(pl)
        w = TL.W_(0, rec)
        p = JS.window_parts_shift(w, pl, res, tm, 0.5)
        e, _ = LW.bounds(rec, "v2p")
        d = p["aug"]["delta"]
        b0 = TJ1.brute(rec, tm, 0.5, 0.0, "v2p", delta=d)
        be = TJ1.brute(rec, tm, 0.5, 0.0, "v2p", delta=d, rho_sets=[[x] for x in e])
        worst = max(worst, abs(p["E0"] - b0["Eq_l"]), abs(p["l_e"] - be["Eq_l"]), abs(p["target"] - p["E0"]))
        # under the label mask: one Thinker path, E[blocks] = its (read, speaker) groups
        fb = JS.forward_backward(p["aug"], 0.0, JS.label_mask(p["aug"], pl))
        eb = 0.0
        for r, rd in enumerate(p["aug"]["reads"]):
            qe = np.bincount(rd["te"], weights=fb["q"][r], minlength=rd["nprev"] * rd["ncur"]).reshape(rd["nprev"],
                                                                                                      rd["ncur"])
            eb += float((qe * JS.edge_blocks(rec, pl, r, rd["nprev"], rd["ncur"])).sum())
        want = len({(r, c["slot"]) for r, c in zip(S.chunk_reads(rec), rec["chunks"])})
        worst = max(worst, abs(eb - want))
        parts.append(p)
    assert worst < 1e-9, worst
    # budget_shift by definition; binding / non-binding / T1-target identity
    for delta in (0.0, 0.134, 0.5, 50.0):
        T_, bi = JS.budget_shift(parts, delta)
        e0 = sum(p["E0"] for p in parts)
        W = sum(p["sum_words"] for p in parts)
        le = sum(p["l_e"] for p in parts)
        assert T_ == max(e0 - delta * W, le) and bi["floor_active"] == (e0 - delta * W < le)
        lam, info = JS.solve_lambda_to(parts, T_)
        got = sum(JS.forward_backward(p["aug"], lam)["Eq_l"] for p in parts)
        if T_ < e0:
            assert info["binding"] and lam > 0 and not info["capped"]
            assert abs(got - T_) <= 1e-6 * abs(T_), (delta, got, T_)
        else:
            assert lam == 0.0
    l1, i1 = JS.solve_lambda(parts)
    l2, i2 = JS.solve_lambda_to(parts, sum(p["target"] for p in parts))
    assert l1 == l2 and i1 == i2
    _, bf = JS.budget_shift(parts, 50.0)
    assert bf["floor_active"]
    print(f"  (2) budget: E0 and l_e = brute force (lam 0; label path only) and the label mask's E[blocks] = the label "
          f"groups, max |d| {worst:.1e}; T = max(E0 - delta W, l_e); solve_lambda_to hits T (rel <= 1e-6) / lam 0 when "
          f"not binding / == solve_lambda at the T1 target; floor active at large delta")
    return worst


def old_modules():
    out = {}
    for name in ("joint_speech_crf", "joint_speech_train"):
        src = subprocess.check_output(["git", "show", f"{BASE}:experiments/mist_w3po_v1/crf/{name}.py"])
        assert b"budget_shift" not in src and b'"shift"' not in src
        spec = importlib.util.spec_from_loader(f"{name}_before_j1v2", loader=None)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[f"{name}_before_j1v2"] = mod
        if name == "joint_speech_train":
            src = src.replace(b"from experiments.mist_w3po_v1.crf import joint_speech_crf as JS",
                              b"import joint_speech_crf_before_j1v2 as JS")
        exec(compile(src, f"{name}_before_j1v2.py", "exec"), mod.__dict__)
        out[name] = mod
    return out


def _run(mod, m, wins, D, pf=False, **kw):
    ctx = {**TV.vctx(perturb=0.2, budget=1500), "final_full_post": True, "talker_model": TJ.toy_talker()}
    TT.zero(m)
    v = mod.run_step(m, wins, D, ctx, 3, JS.TAU, 0, 1, prefetch=pf, **kw)
    ctx["prefetched_batches"] = v["batches"]
    lg = T.grad_pass(m, v["rows"], ctx, 3)
    return v, lg, TT.grads(m)


def test_defaults_and_shift():
    old = old_modules()["joint_speech_train"]
    base = TV.small_choice_windows(3, max_paths=60)
    wins = [T.make_window(w.idx, TL.widen(w.rec), True) for w in base]
    T.set_reach(wins, 128)
    m = TT.tiny_model(seed=47)
    bias = torch.zeros(151936)
    bias[T.WAIT_ID] = 9.0
    bias[T.EOS_ID] = 6.0
    m.logits_at = lambda h: TT.ThinkerGStarModel.logits_at(m, h) + bias
    D = T.step_totals(wins)[0]
    draws = {w.idx: [2] * len(w.rec["chunks"]) for w in wins}
    for kw_old, kw_new in (({}, {}), ({}, {"budget": "t1"}), ({"mode": "t5", "dart": draws},
                                                              {"mode": "t5", "dart": draws, "budget": "t1"})):
        a, la, ga = _run(old, m, wins, D, **kw_old)
        b, lb, gb = _run(JST, m, wins, D, **kw_new)
        TJ1._same_rows(a["rows"], b["rows"])
        nt = lambda d: {k: x for k, x in d.items() if not k.startswith("t_")}   # wall-clock fields excluded
        assert nt(a["stats"]) == nt(b["stats"]) and la["loss_main"] == lb["loss_main"], kw_new
        assert all(torch.equal(ga[k], gb[k]) for k in ga), kw_new
    v, lg, _ = _run(JST, m, wins, D, mode="v2p", dart={w.idx: [1] * len(w.rec["chunks"]) for w in wins},
                    budget="shift")
    vp, lgp, _ = _run(JST, m, wins, D, pf=True, mode="v2p", dart={w.idx: [1] * len(w.rec["chunks"]) for w in wins},
                      budget="shift")
    st = v["stats"]
    assert math.isfinite(lg["loss_main"]) and lg["loss_main"] == lgp["loss_main"]
    assert {"js_budget", "budget_T", "Eq_speech_lat_e_path", "budget_shifted", "budget_floor_active", "js_delta",
            "exp_blocks", "dart_rows"} <= set(st) and st["lw_mode"] == "v2p"
    if st["lam_binding"]:
        assert abs(st["Eq_speech_lat_T3"] - st["budget_T"]) <= 1e-6 * st["budget_T"]
    print(f"  (3) run_step: default / budget t1 == the module before J1-v2 bit for bit (T3 default and J1 t5 + DART: "
          f"rows, stats, loss, grads); shift (v2p + DART): loss {lg['loss_main']:.4f}, lam {st['lam']:.4f} "
          f"(binding {st['lam_binding']}), E_q[l] / word {st['Eq_speech_lat_T3']:.4f} vs T {st['budget_T']:.4f}, "
          f"floor {st['budget_floor_active']}; prefetch 1 == 0")


def test_args():
    common = ["--out", "x", "--sched", "s"]
    a = T.build_args(["--objective", "joint_speech", "--lw-mode", "v2p", "--js-budget", "shift"] + common)
    assert a.js_budget == "shift" and a.js_delta == 0.134 and a.lw_mode == "v2p"
    b = T.build_args(["--objective", "latent_write"] + common)
    assert b.js_budget == "t1" and b.js_delta == 0.134
    for bad in (["--objective", "latent_write", "--js-budget", "shift"],
                ["--objective", "joint_speech", "--js-delta", "0.2"]):
        try:
            T.build_args(bad + common)
        except SystemExit:
            continue
        raise AssertionError(bad)
    print("  (4) args: --js-budget shift / --js-delta 0.134 parsed; refused off joint_speech or without shift")


def main():
    t0 = time.time()
    for t in (test_args, test_budget, test_brute_v2p, test_defaults_and_shift):
        t1 = time.time()
        t()
        print(f"PASS {t.__name__} ({time.time() - t1:.1f}s)", flush=True)
    print(f"ALL PASSED ({time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()

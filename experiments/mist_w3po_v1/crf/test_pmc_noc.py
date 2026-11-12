"""Tests of PMC-noC (reports/mist_w3po_v1/thinker_pmc_noc_prereg_v1.zh-CN.md §2): the PMC content-term weight
(latent_write_train.set_coefs_pmc(w_pol), run_step(pmc_w), train_crf_sft --pmc-content-weight), CPU only.

  N1 arguments: default 1.0; --lw-mode pmc --pmc-content-weight 0 -> 0.0; a non-1 weight in another mode / objective,
     a negative weight and a non-1 weight with --d1 are refused
  N2 coefficients on widened sched_v1 windows (test_latent_write_crf.widen) and real VE1 windows (sched_t5_merge3_w10_v1, mode pmc, dummy
     scores, lam* of the step): w_pol = 0 == V2T.set_coefs bit for bit (c_tok and c_end); no w_pol == w_pol = 1.0 bit
     for bit; w_pol = 0.5 == set_coefs + 0.5 h suf(cend_pol) / D (1e-12)
  N3 the trainer step (run_step mode pmc -> grad_pass, tiny model, world 1, as test_pmc P5): pmc_w = 0: finite loss,
     sum c_tok x D = sum h (VE1's M step: each token once), sum c_end x D = sum R, lam*, q_lam's E[Lat], q^pol's stats
     == those of pmc_w = 1 in the same step bit for bit, stats["pmc_w"] == 0; no pmc_w == pmc_w = 1.0 (loss, grads,
     coefficients) bit for bit and has no "pmc_w" field

    PYTHONPATH=.:third_party/Megatron-LM CUDA_VISIBLE_DEVICES= python -m experiments.mist_w3po_v1.crf.test_pmc_noc
"""
from __future__ import annotations

import math
import time

import numpy as np
import torch

if not torch.cuda.is_available():    # CPU-only run: transformers' model modules probe the accelerator at import
    torch.accelerator.current_accelerator = lambda *a, **k: None

from experiments.mist_w3po_v1.crf import latent_write_crf as LW
from experiments.mist_w3po_v1.crf import latent_write_train as LWT
from experiments.mist_w3po_v1.crf import serial_v2p_train as V2T
from experiments.mist_w3po_v1.crf import test_latent_write_crf as TL
from experiments.mist_w3po_v1.crf import test_pmc as TP
from experiments.mist_w3po_v1.crf import test_serial_v2p_train as TV
from experiments.mist_w3po_v1.crf import test_train_crf_sft as TT
from experiments.mist_w3po_v1.crf import train_crf_sft as T

torch.set_num_threads(8)


# ---------------------------------------------------------------- N1 arguments
def _refused(argv: list[str]) -> bool:
    try:
        T.build_args(argv)
    except SystemExit:
        return True
    return False


def test_n1_args():
    base = ["--objective", "latent_write", "--out", "/nonexistent/x"]
    assert T.build_args(base + ["--lw-mode", "pmc"]).pmc_content_weight == 1.0
    assert T.build_args(base).pmc_content_weight == 1.0
    a = T.build_args(base + ["--lw-mode", "pmc", "--pmc-content-weight", "0"])
    assert a.pmc_content_weight == 0.0 and a.lw_mode == "pmc"
    assert T.build_args(base + ["--lw-mode", "v2p", "--pmc-content-weight", "1"]).pmc_content_weight == 1.0
    assert _refused(base + ["--lw-mode", "v2p", "--pmc-content-weight", "0"])
    assert _refused(base + ["--pmc-content-weight", "0"])
    assert _refused(["--objective", "joint_speech", "--lw-mode", "pmc", "--pmc-content-weight", "0", "--out", "/x"])
    assert _refused(base + ["--lw-mode", "pmc", "--pmc-content-weight", "-0.5"])
    assert _refused(base + ["--lw-mode", "pmc", "--pmc-content-weight", "0", "--d1", "/nonexistent/d1"])
    print("  N1 --pmc-content-weight: default 1.0; pmc + 0 -> 0.0; refused in v2p / t2 / joint_speech, negative, "
          "with --d1")


# ---------------------------------------------------------------- N2 coefficients
def _step_objects(wins: list) -> tuple:
    plans = {w.idx: LWT.plan(w, "pmc") for w in wins}
    res = {(w.idx, k): v for w in wins for k, (v, q) in enumerate(zip(TL.dummy_full(plans[w.idx]),
                                                                       plans[w.idx]["reqs"])) if q["scored"]}
    parts = {w.idx: LW.window_parts(w, plans[w.idx], res, LW.TAU) for w in wins}
    lam, _ = LW.solve_lambda(list(parts.values()))
    posts = {w.idx: LW.posterior_from_parts(w, plans[w.idx], parts[w.idx], LW.TAU, lam) for w in wins}
    pols = {w.idx: LW.posterior_from_parts(w, plans[w.idx], LW.pol_parts(w, plans[w.idx], res, parts[w.idx]),
                                           LW.TAU, lam) for w in wins}
    D = float(sum(w.h.sum() + w.R for w in wins if w.feasible))
    return plans, posts, pols, D, lam


def check_coefs(wins: list) -> tuple[float, int, float]:
    """-> (max |w=0.5 - want|, rows, PMC-term share of the content weight at w = 1)"""
    plans, posts, pols, D, lam = _step_objects(wins)
    rows = {k: V2T.pass2_rows(wins, plans, None, D) for k in ("set", "w0", "dflt", "w1", "half")}
    V2T.set_coefs(rows["set"], plans, posts, D)
    LWT.set_coefs_pmc(rows["w0"], plans, posts, pols, D, 0.0)
    LWT.set_coefs_pmc(rows["dflt"], plans, posts, pols, D)
    LWT.set_coefs_pmc(rows["w1"], plans, posts, pols, D, 1.0)
    LWT.set_coefs_pmc(rows["half"], plans, posts, pols, D, 0.5)
    worst, tot_q, tot_p = 0.0, 0.0, 0.0
    for a, b, c, d, e in zip(rows["set"], rows["w0"], rows["dflt"], rows["w1"], rows["half"]):
        assert a.key == b.key == c.key == d.key == e.key
        assert np.array_equal(a.c_tok, b.c_tok) and np.array_equal(a.c_end, b.c_end), a.key
        assert np.array_equal(c.c_tok, d.c_tok) and np.array_equal(c.c_end, d.c_end), a.key
        assert np.array_equal(a.c_end, e.c_end), a.key
        widx, k = a.key
        q = plans[widx]["reqs"][k]
        mm = len(q["forced_ids"])
        h = V2T.token_weights(a.win, q["forced_ids"])
        cp = pols[widx]["cend"][k]
        add = h * np.array([cp[t + 1:].sum() for t in range(mm)]) / D
        want = a.c_tok.copy()
        want[:mm] += 0.5 * add
        worst = max(worst, float(np.abs(e.c_tok - want).max()))
        tot_q += float(a.c_tok.sum())
        tot_p += float(add.sum())
    assert worst < 1e-12, worst
    return worst, len(rows["set"]), tot_p / max(tot_q + tot_p, 1e-12)


def test_n2_coefs():
    worst, nrow, nsyn = 0.0, 0, 0
    wide = [TL.widen(r) for r in TV.recs0()[:400] if 0 < r["n_tok"] <= 200 and r["n_read"] >= 3][:40]
    for i in range(0, len(wide), 4):   # widened sched_v1 windows (test_latent_write_crf's T2-format records), 4 per step
        ws = [T.make_window(nsyn + j, rec, True) for j, rec in enumerate(wide[i:i + 4])]
        wv, n, _ = check_coefs(ws)
        worst, nrow, nsyn = max(worst, wv), nrow + n, nsyn + len(ws)
    real = []
    for rec in TP.ve1_recs(400):
        if rec["n_tok"] <= 200 and rec["n_read"] <= 8:
            w = T.make_window(len(real), rec, True)
            if w.feasible:
                real.append(w)
        if len(real) >= 40:
            break
    shares = []
    for i in range(0, len(real), 8):
        wv, n, sh = check_coefs(real[i:i + 8])
        worst, nrow = max(worst, wv), nrow + n
        shares.append(sh)
    assert nsyn >= 20 and len(real) >= 30, (nsyn, len(real))
    print(f"  N2 coefficients: {nsyn} widened sched_v1 windows + {len(real)} real VE1 windows ({nrow} rows): w_pol 0 == "
          f"V2T.set_coefs bit for bit; default == 1.0 bit for bit; 0.5 == set_coefs + 0.5 h suf(cend_pol) / D (max |d| "
          f"{worst:.1e}); PMC term share of the content weight at w = 1 (real, dummy scores) "
          f"{np.mean(shares):.3f}")


# ---------------------------------------------------------------- N3 trainer step
def _run(m, wins, D, pf, **kw):
    ctx = {**TV.vctx(perturb=0.2, budget=1500), "final_full_post": True}
    TT.zero(m)
    v = LWT.run_step(m, wins, D, ctx, 3, LW.TAU, 0, 1, prefetch=pf, mode="pmc", **kw)
    ctx["prefetched_batches"] = v["batches"]
    lg = T.grad_pass(m, v["rows"], ctx, 3)
    V2T.reduce_logs(v, lg, "cpu")
    return v, lg, TT.grads(m)


def test_n3_run_step():
    base = TV.small_choice_windows(3, max_paths=60)
    wins = [T.make_window(w.idx, TL.widen(w.rec), True) for w in base]
    T.set_reach(wins, 128)
    m = TT.tiny_model(seed=29)
    bias = torch.zeros(151936)
    bias[T.WAIT_ID] = 9.0
    bias[T.EOS_ID] = 6.0
    m.logits_at = lambda h: TT.ThinkerGStarModel.logits_at(m, h) + bias
    D = T.step_totals(wins)[0]
    v0, l0, g0 = _run(m, wins, D, False, pmc_w=0.0)
    v1, l1, g1 = _run(m, wins, D, False, pmc_w=1.0)
    vd, ld, gd = _run(m, wins, D, True)
    assert math.isfinite(l0["loss_main"]) and math.isfinite(l1["loss_main"])
    sh = sum(float(w.h.sum()) for w in wins)
    sr = sum(float(w.R) for w in wins)
    for v, k in ((v0, 1.0), (v1, 2.0)):
        tok = sum(float(rw.c_tok.sum()) for rw in v["rows"]) * D
        end = sum(float(rw.c_end.sum()) for rw in v["rows"]) * D
        assert abs(tok - k * sh) < 1e-9 * sh and abs(end - sr) < 1e-9 * sr, (k, tok, sh, end, sr)
    s0, s1, sd = v0["stats"], v1["stats"], vd["stats"]
    for key in ("lam", "lam_binding", "Eq_lat_T2", "Eq_lat_T1_target", "Eq_lat_lam0", "lat_label", "q_label",
                "Eq_lat_pol", "lat_e", "exp_early_q", "exp_early_pol", "n_chunks", "logZ_mean"):
        assert s0[key] == s1[key] == sd[key], (key, s0[key], s1[key], sd[key])
    assert s0["pmc_w"] == 0.0 and "pmc_w" not in s1 and "pmc_w" not in sd
    assert l1["loss_main"] == ld["loss_main"]
    for k in g1:
        assert torch.equal(g1[k], gd[k]), k
    for a, b in zip(v1["rows"], vd["rows"]):
        assert np.array_equal(a.c_tok, b.c_tok) and np.array_equal(a.c_end, b.c_end)
    assert l0["loss_main"] != l1["loss_main"]
    print(f"  N3 step path (tiny model, mode pmc): pmc_w 0: loss {l0['loss_main']:.6f} finite, sum c_tok x D = sum h "
          f"({sh:.0f}), sum c_end x D = sum R ({sr:.0f}); lam {s0['lam']:.4f}, E[Lat] {s0['Eq_lat_T2']:.4f}, q^pol stats "
          f"== pmc_w 1 bit for bit; stats pmc_w 0.0; no pmc_w == 1.0 (loss {l1['loss_main']:.6f}, grads, coefficients, "
          f"prefetch 1 vs 0) bit for bit")


def main():
    t0 = time.time()
    tests = [test_n1_args, test_n2_coefs, test_n3_run_step]
    for t in tests:
        t1 = time.time()
        t()
        print(f"PASS {t.__name__} ({time.time() - t1:.1f}s)", flush=True)
    print(f"ALL {len(tests)} PASSED ({time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()

"""Tests of plan B's latency-budget posterior (crf/serial_v2p_lat.py), CPU, dummy LM scores on real sched_v1 windows:
  (1) ratio 1 -> serial_v2p_train.posterior bit-identical (cend, logZ, q_label, exp_defer)
  (2) ratio 0.5 -> E_q[Lat] = 0.5 E_q0[Lat] (rel 1e-6) on every window with E_q0[Lat] > 0; per-read mass 1;
      E_q[Lat] = sum_c q(defer c) cost(c) (linearity of the state costs)
  (3) brute force over every (rho x order) path with scores - lam Lat / tau: state / deferral / order marginals equal
  (4) install(): the module attributes are replaced and run_step's stats carry the lat_* fields (fake orig)

    PYTHONPATH=.:third_party/Megatron-LM python -m experiments.mist_w3po_v1.crf.test_serial_v2p_lat
"""
from __future__ import annotations

import numpy as np

from experiments.mist_w3po_v1.crf import precheck_serial_v2 as V
from experiments.mist_w3po_v1.crf import precheck_serial_v2p as VP
from experiments.mist_w3po_v1.crf import serial_v2p_lat as L
from experiments.mist_w3po_v1.crf import serial_v2p_train as V2T
from experiments.mist_w3po_v1.crf import test_serial_v2p_train as TS
from experiments.mist_w3po_v1.crf import train_crf_sft as T

ORIG_POST = V2T.posterior


def dummy(w):
    pl = V2T.plan(w)
    res = {(w.idx, k): v for k, v, q in zip(range(len(pl["reqs"])), V.dummy_res(pl), pl["reqs"]) if q["scored"]}
    return pl, res


def wins(n: int):
    out = []
    for i, rec in enumerate(TS.recs0()):
        if rec["n_tok"] > 0:
            out.append(T.make_window(i, rec, True))
        if len(out) >= n:
            break
    return out


def test_ratio_one_identical():
    for w in wins(300):
        pl, res = dummy(w)
        a = ORIG_POST(w, pl, res, V2T.TAU)
        b = L.posterior_lat(w, pl, res, V2T.TAU, 1.0)
        assert b["lam"] == 0.0 and a["logZ"] == b["logZ"] and a["q_label"] == b["q_label"], w.rec["id"]
        assert a["exp_defer"] == b["exp_defer"] and a["exp_nf"] == b["exp_nf"], w.rec["id"]
        for x, y in zip(a["cend"], b["cend"]):
            assert np.array_equal(x, y), w.rec["id"]
    print("  (1) ratio 1: 300 windows bit-identical to serial_v2p_train.posterior")


def test_budget_hit():
    n_b, worst = 0, 0.0
    for w in wins(400):
        pl, res = dummy(w)
        p = L.posterior_lat(w, pl, res, V2T.TAU, 0.5)
        cc = L.chunk_costs(w.rec, pl["lat"])
        dm = V.deferral_marginals(pl["lat"], p["fb"])
        lin = sum(dm[c] * cc[c] for c in dm)
        assert abs(lin - p["lat"]) < 1e-9 * max(1.0, p["lat"]), (w.rec["id"], lin, p["lat"])
        if p["lat0"] > 1e-9:
            n_b += 1
            rel = abs(p["lat"] - 0.5 * p["lat0"]) / p["lat0"]
            worst = max(worst, rel)
            assert rel < 1e-6 and p["lam"] > 0, (w.rec["id"], p["lat0"], p["lat"], p["lam"])
        else:
            assert p["lam"] == 0.0
    assert n_b >= 50, n_b
    print(f"  (2) ratio 0.5: {n_b} windows with deferral cost hit the budget (max rel {worst:.1e}); linearity exact")


def test_brute():
    n, worst = 0, 0.0
    for i, rec in enumerate(TS.recs0()[:3000]):
        if rec["n_tok"] == 0 or rec["n_tok"] > 120 or rec["n_read"] > 6:
            continue
        w = T.make_window(i, rec, True)
        pl, res = dummy(w)
        if VP.n_paths_lattice(VP.assemble_p(pl, [res.get((w.idx, k), (np.zeros(len(q["forced_ids"])),
                                                                          np.zeros(len(q["forced_ids"]) + 1)))
                                                  for k, q in enumerate(pl["reqs"])])) > 3000:
            continue
        post = L.posterior_lat(w, pl, res, V2T.TAU, 0.5)
        if post["lam"] == 0.0:
            continue

        def psi_fn(r, final, hist, written):
            tl, ec = V.dummy_scores(r, final, hist, written)
            return float(tl.sum() + ec[-1])
        lat = V.lattice(rec)
        paths, scores = VP.brute_paths_p(rec, lat, psi_fn)
        cc = L.chunk_costs(rec, lat)
        lat_p = np.array([sum(cc[c] for c in cc if a[c] == lat["rho"][c] + 1) for a, _ in paths])
        bf = VP.brute_force_p(rec, lat, paths, scores - post["lam"] * lat_p / V2T.TAU, V2T.TAU)
        worst = max(worst, TS.cmp_marginals(lat, post, bf, 1e-9, rec["id"] + " brute"))
        assert abs(float(np.dot(bf["q"], lat_p)) - post["lat"]) < 1e-9
        n += 1
        if n >= 60:
            break
    assert n >= 30, n
    print(f"  (3) brute force: {n} windows with lam > 0, marginals max |d| {worst:.1e}")


def test_install():
    so, sp = V2T.run_step, V2T.posterior
    try:
        def fake_step(*a, **k):
            L.STEP.extend([{"lam": 0.5, "lat0": 2.0, "lat": 1.0}, {"lam": 0.0, "lat0": 0.0, "lat": 0.0}])
            return {"stats": {"q_label": 0.6}}
        V2T.run_step = fake_step
        L.install(0.5)
        assert V2T.posterior is not sp
        st = V2T.run_step()["stats"]
        assert st["lat_ratio"] == 0.5 and st["lat_lam_mean"] == 0.25 and st["lat_E0"] == 2.0 and st["lat_E"] == 1.0
        assert st["lat_bound"] == 1 and st["q_label"] == 0.6
        try:
            L.install(1.5)
            raise AssertionError("ratio 1.5 accepted")
        except ValueError:
            pass
    finally:
        V2T.run_step, V2T.posterior = so, sp
    print("  (4) install: attributes replaced, step stats carry lat_*")


if __name__ == "__main__":
    for f in (test_ratio_one_identical, test_budget_hit, test_brute, test_install):
        f()
    print("ALL OK")

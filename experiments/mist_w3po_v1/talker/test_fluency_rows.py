"""Checks of fluency_rows (v11 P2): exact forward-backward with the fluency dual, monotonicity, dual solution."""
import numpy as np

from experiments.mist_w3po_v1.talker import fluency_rows as FR
from experiments.mist_w3po_v1.talker import playback_lattice as PL


def _streams(seed=0, n=6):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        N = int(rng.integers(2, 5))
        a = np.cumsum(rng.uniform(300, 2500, N))
        dur = [rng.uniform(400, 3000, 3) for _ in range(N)]
        feat = [rng.normal(-2, 1, 3) for _ in range(N)]
        st = PL.Stream(a, dur, feat, rng.integers(1, 8, N))
        st.__class__ = FR.FStream
        out.append(st)
    return out


def test_exact_vs_brute_force_with_mu():
    for st in _streams(1):
        st.mu = 0.7
        logz, marg, el, _ = st.forward_backward(0.3)
        bz, bm, _, bel = PL.brute_force(st, 0.3)
        assert abs(logz - bz) < 1e-8 and abs(el - bel) < 1e-8
        for m, b in zip(marg, bm):
            assert np.allclose(m.sum(0), b, atol=1e-10)


def test_mu_zero_is_plem():
    st = _streams(2)[0]
    st.mu = 0.0
    a = st.forward_backward(0.4)
    b = PL.Stream.forward_backward(st, 0.4)
    assert abs(a[0] - b[0]) < 1e-12


def test_monotone_and_solve():
    ss = _streams(3, 20)
    l0, d0 = FR.corpus(ss, 0.0, 0.0)
    l1, d1 = FR.corpus(ss, 0.0, 1.0)
    assert d1 >= d0 - 1e-12
    l2, _ = FR.corpus(ss, 1.0, 0.0)
    assert l2 <= l0 + 1e-12
    budget, smin = l0 - 0.05, d0 + 0.01
    sol = FR.solve(ss, budget, smin)
    el, ed = FR.corpus(ss, sol["lambda"], sol["mu"])
    if sol["feasible"]:
        assert el <= budget + 2e-3 and ed >= smin - 2e-4

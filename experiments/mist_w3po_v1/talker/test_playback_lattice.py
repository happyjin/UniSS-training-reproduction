"""P2 (exactness): the playback lattice's forward-backward / Viterbi against brute-force enumeration (tol 1e-9)."""
import numpy as np

from experiments.mist_w3po_v1.talker.playback_lattice import Stream, brute_force, corpus_latency, quant, solve_lambda


def rand_stream(rng, N, C):
    a = np.cumsum(rng.integers(0, 3000, size=N)).astype(float)
    dur = [rng.integers(200, 4000, size=C) for _ in range(N)]
    feat = [rng.normal(size=C) for _ in range(N)]
    words = rng.integers(1, 12, size=N)
    return Stream(a, dur, feat, words)


def test_quant():
    assert quant(0) == 0 and quant(1) == 1 and quant(80) == 1 and quant(81) == 2 and quant(1e9) == 100


def test_exact_vs_brute():
    rng = np.random.default_rng(0)
    for trial in range(60):
        N, C = int(rng.integers(1, 5)), int(rng.integers(1, 5))
        st = rand_stream(rng, N, C)
        lam = float(rng.uniform(0, 3))
        logz, marg, el, beta = st.forward_backward(lam)
        bz, bm, bp, bel = brute_force(st, lam)
        assert abs(logz - bz) < 1e-9, (logz, bz)
        for n in range(N):
            assert np.allclose(marg[n].sum(0), bm[n], atol=1e-9)
        assert abs(el - bel) < 1e-9
        vp = st.viterbi(lam)
        ps = [st.psi(n, lam) for n in range(N)]
        sc = lambda p: sum(ps[n][s, c] for n, (s, c) in enumerate(zip(st.path_states(p), p)))
        assert abs(sc(vp) - sc(bp)) < 1e-9
        # conditional at the first block (state 0) = the block-0 marginal
        assert np.allclose(st.conditional(0, 0, lam, beta), bm[0], atol=1e-9)


def test_lambda_bisection():
    rng = np.random.default_rng(1)
    streams = [rand_stream(rng, 3, 4) for _ in range(20)]
    e0 = corpus_latency(streams, 0.0)
    lam, e = solve_lambda(streams, e0 - 0.2)
    assert lam > 0 and abs(e - (e0 - 0.2)) < 1e-6
    assert solve_lambda(streams, e0 + 1)[0] == 0.0


if __name__ == "__main__":
    test_quant()
    test_exact_vs_brute()
    test_lambda_bisection()
    print("ok")

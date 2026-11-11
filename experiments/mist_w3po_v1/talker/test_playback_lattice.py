"""P2 (exactness): the playback lattice's forward-backward / Viterbi against brute-force enumeration (tol 1e-9)."""
import numpy as np

from experiments.mist_w3po_v1.talker.playback_lattice import (Stream, brute_force, corpus_latency, quant, quant_nearest,
                                                              solve_lambda)


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


def rand_meeting(rng, n_spk, n_blk, C, rounding="nearest", cap=32000):
    """a tiny meeting (talker_crf_prereg_v2 experiment 2): n_spk streams' blocks merged FIFO by availability into one
    chain on the shared channel."""
    L = []
    for k in range(n_spk):
        a = np.cumsum(rng.integers(0, 2500, size=n_blk)).astype(float) + rng.integers(0, 1000)
        L += [(float(x), f"S{k}", i) for i, x in enumerate(a)]
    L.sort()
    dur = [rng.integers(200, 4000, size=C) for _ in L]
    feat = [rng.normal(size=C) for _ in L]
    words = rng.integers(1, 12, size=len(L))
    return L, Stream([x[0] for x in L], dur, feat, words, cap=cap, rounding=rounding)


def test_meeting_chain_exact_vs_brute():
    """P2 for the meeting chain (nearest rounding, 32 s cap): forward-backward / Viterbi / conditional vs enumeration."""
    rng = np.random.default_rng(2)
    for trial in range(40):
        L, st = rand_meeting(rng, int(rng.integers(2, 4)), int(rng.integers(1, 3)), int(rng.integers(1, 4)))
        if len(L) > 6:
            continue
        lam = float(rng.uniform(0, 3))
        logz, marg, el, beta = st.forward_backward(lam)
        bz, bm, bp, bel = brute_force(st, lam)
        assert abs(logz - bz) < 1e-9 and abs(el - bel) < 1e-9
        for n in range(len(L)):
            assert np.allclose(marg[n].sum(0), bm[n], atol=1e-9)
        vp = st.viterbi(lam)
        ps = [st.psi(n, lam) for n in range(len(L))]
        sc = lambda p: sum(ps[n][s, c] for n, (s, c) in enumerate(zip(st.path_states(p), p)))
        assert abs(sc(vp) - sc(bp)) < 1e-9
        assert np.allclose(st.conditional(0, 0, lam, beta), bm[0], atol=1e-9)


def test_nearest_rounding_and_default():
    assert quant_nearest(39) == 0 and quant_nearest(40) == 1 and quant_nearest(119) == 1 and quant_nearest(1e9) == 100
    assert quant_nearest(1e9, cap=32000) == 400
    rng = np.random.default_rng(3)
    st = rand_stream(rng, 4, 3)
    st2 = Stream(st.a, st.dur, st.feat, st.words, rounding="ceil")
    assert all((x == y).all() for x, y in zip(st.nxt[:-1], st2.nxt[:-1]))   # default = ceil, unchanged


def test_meeting_chain_tracks_shared_channel():
    """with a 1 ms grid (integer ms inputs) the chain's states on any path are the exact shared-channel backlogs."""
    from experiments.mist_w3po_v1.talker.shared_channel import SharedChannel

    rng = np.random.default_rng(4)
    L = sorted((float(rng.integers(0, 20000)), f"S{k}", i) for k in range(3) for i in range(4))
    dur = [rng.integers(1, 200, size=2) * 20 for _ in L]
    st = Stream([x[0] for x in L], dur, [np.zeros(2)] * len(L), [3] * len(L), delta=1, cap=10 ** 6, rounding="nearest")
    for _ in range(5):
        path = [int(x) for x in rng.integers(0, 2, size=len(L))]
        ch, exact = SharedChannel(), []
        for (a, _, _), d, c in zip(L, dur, path):
            exact.append(ch.backlog(a))
            ch.play(a, d[c])
        assert st.path_states(path) == [int(x) for x in exact]


if __name__ == "__main__":
    test_quant()
    test_exact_vs_brute()
    test_lambda_bisection()
    test_meeting_chain_exact_vs_brute()
    test_nearest_rounding_and_default()
    test_meeting_chain_tracks_shared_channel()
    print("ok")

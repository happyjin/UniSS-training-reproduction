"""Brute-force checks of experiments/mist_w3po_v1/crf/turn_crf.py on tiny instances (few slots, T <= 6, small L):
log Z, frame marginals, Viterbi, the right-censored prefix version and log Z_y for the semi-CRF and the linear chain,
plus autograd (marginals == d log Z / d potentials). pytest-compatible; also runnable directly:

    PYTHONPATH=. python -m experiments.mist_w3po_v1.crf.test_turn_crf
"""
from __future__ import annotations

import itertools
import math

import torch

from experiments.mist_w3po_v1.crf import turn_crf as tc

TOL = 1e-8


def _compositions(T, L):
    if T == 0:
        yield ()
        return
    for q in range(1, min(L, T) + 1):
        for rest in _compositions(T - q, L):
            yield (q,) + rest


def brute_semi(u, tr, start_c, end_c, d_c, ds_c, censored=False):
    """All maximal segmentations: list of (score, frame labels, [(q, c), ...])."""
    T, C = u.shape
    L = d_c.shape[0]
    out = []
    for comp in _compositions(T, L):
        bounds = list(itertools.accumulate((0,) + comp))
        for labs in itertools.product(range(C), repeat=len(comp)):
            if any(labs[m] == labs[m - 1] for m in range(1, len(labs))):
                continue
            s = float(start_c[labs[0]])
            y = []
            for m, (q, c) in enumerate(zip(comp, labs)):
                i, j = bounds[m], bounds[m + 1]
                last = m == len(comp) - 1
                s += float(u[i:j, c].sum()) + float((ds_c if (censored and last) else d_c)[q - 1, c])
                if m:
                    s += float(tr[i, labs[m - 1], c])
                y += [c] * q
            if not censored:
                s += float(end_c[labs[-1]])
            out.append((s, tuple(y), list(zip(comp, labs))))
    return out


def brute_chain(u, tr, start_c, end_c, censored=False):
    T, C = u.shape
    out = []
    for y in itertools.product(range(C), repeat=T):
        s = float(start_c[y[0]]) + sum(float(u[t, y[t]]) for t in range(T))
        s += sum(float(tr[t, y[t - 1], y[t]]) for t in range(1, T))
        if not censored:
            s += float(end_c[y[-1]])
        out.append((s, tuple(y), None))
    return out


def _summ(items, T, C):
    sc = torch.tensor([s for s, _, _ in items], dtype=torch.float64)
    logZ = torch.logsumexp(sc, 0)
    w = torch.exp(sc - logZ)
    mu = torch.zeros(T, C, dtype=torch.float64)
    for wi, (_, y, _) in zip(w, items):
        for t, c in enumerate(y):
            mu[t, c] += wi
    best = int(sc.argmax())
    return float(logZ), mu, items[best][1], float(sc[best]), w


def _logZ_y(items, y):
    sc = [s for s, yy, _ in items if yy == tuple(y)]
    return float(torch.logsumexp(torch.tensor(sc, dtype=torch.float64), 0)) if sc else -math.inf


def _rand_model(kind, K, L=None, Q=None, seed=0, scale=0.7):
    torch.manual_seed(seed)
    m = tc.SemiTurnCRF(K=K, L=L, Q=Q) if kind == "semi" else tc.ChainTurnCRF(K=K)
    with torch.no_grad():
        for p in m.parameters():
            p.add_(scale * torch.randn_like(p))
    return m


def _rand_P(T, K, seed):
    g = torch.Generator().manual_seed(seed)
    P = torch.rand(T, K, generator=g, dtype=torch.float64)
    P[P < 0.15] = 0.0            # exercise the clipping (float16 zeros in the real data)
    return P


def _valid_y(items, seed):
    g = torch.Generator().manual_seed(seed)
    return items[int(torch.randint(len(items), (1,), generator=g))][1]


def _check_semi_case(K, T, L, Q, seed):
    m = _rand_model("semi", K, L=L, Q=Q, seed=seed)
    P = _rand_P(T, K, seed + 100)
    p = m.potentials(P)
    C = m.space.C
    for censored in (False, True):
        items = brute_semi(p["u"].detach(), p["tr"].detach(), p["start_c"].detach(), p["end_c"].detach(),
                           p["d_c"].detach(), p["ds_c"].detach(), censored)
        logZ, mu, yb, sb, _ = _summ(items, T, C)
        lz, mu2 = m.marginals(P, censored=censored)
        assert abs(float(lz) - logZ) < TOL, (censored, float(lz), logZ)
        assert torch.allclose(mu2, mu, atol=1e-9), (censored, (mu2 - mu).abs().max())
        yv, sv = m.viterbi(P, censored=censored)
        assert abs(sv - sb) < TOL and tuple(yv.tolist()) == yb, (censored, sv, sb)
        # log Z_y: explicit DP and batched DP vs enumeration
        y = torch.tensor(_valid_y(items, seed + 7))
        lzy_b = _logZ_y(items, y.tolist())
        q = {k: v.detach() for k, v in p.items()}
        lzy, _ = tc.semi_forward_backward(q["u"], q["tr"], q["start_c"], q["end_c"], q["d_c"], q["ds_c"], censored, y)
        assert abs(float(lzy) - lzy_b) < TOL, (float(lzy), lzy_b)
    # batched closed + censored prefix at every tau, constrained and unconstrained, in one pass
    one = torch.ones(1, dtype=torch.long)
    for tau in range(1, T + 1):
        pre = m.potentials(P[:tau])
        it_c = brute_semi(pre["u"].detach(), pre["tr"].detach(), pre["start_c"].detach(), pre["end_c"].detach(),
                          pre["d_c"].detach(), pre["ds_c"].detach(), censored=True)
        y_full = torch.tensor(_valid_y(brute_semi(p["u"].detach(), p["tr"].detach(), p["start_c"].detach(),
                                                  p["end_c"].detach(), p["d_c"].detach(), p["ds_c"].detach()),
                                       seed + tau))
        lz, lzc = m.log_partitions(P[None], one * T, one * tau)
        assert abs(float(lzc) - _summ(it_c, tau, C)[0]) < TOL, (tau, float(lzc))
        lzy, lzyc = m.log_partitions(P[None], one * T, one * tau, y_full[None])
        assert abs(float(lzyc) - _logZ_y(it_c, y_full[:tau].tolist())) < TOL
    full_items = brute_semi(p["u"].detach(), p["tr"].detach(), p["start_c"].detach(), p["end_c"].detach(),
                            p["d_c"].detach(), p["ds_c"].detach())
    assert abs(float(lz) - _summ(full_items, T, C)[0]) < TOL
    assert abs(float(lzy) - _logZ_y(full_items, y_full.tolist())) < TOL


def test_semi_bruteforce():
    _check_semi_case(K=3, T=5, L=3, Q=2, seed=0)      # L < T: the max-length constraint is active
    _check_semi_case(K=3, T=6, L=6, Q=3, seed=1)      # geometric tail beyond Q
    _check_semi_case(K=4, T=4, L=2, Q=2, seed=2)


def test_semi_reference_with_long_run_is_impossible():
    m = _rand_model("semi", 3, L=2, Q=2, seed=3)
    P = _rand_P(5, 3, 9)
    y = torch.tensor([1, 1, 1, 0, 0])               # run of 3 > L = 2
    lz, lzy = m.log_partitions(P[None], torch.tensor([5]), None, None)[0], \
        m.log_partitions(P[None], torch.tensor([5]), None, y[None])[0]
    assert float(lz - lzy) > 1000.0


def _check_chain_case(K, T, seed):
    m = _rand_model("chain", K, seed=seed)
    P = _rand_P(T, K, seed + 50)
    p = m.potentials(P)
    C = m.space.C
    q = {k: v.detach() for k, v in p.items()}
    for censored in (False, True):
        items = brute_chain(q["u"], q["tr"], q["start_c"], q["end_c"], censored)
        logZ, mu, yb, sb, _ = _summ(items, T, C)
        lz, mu2 = m.marginals(P, censored=censored)
        assert abs(float(lz) - logZ) < TOL and torch.allclose(mu2, mu, atol=1e-9)
        yv, sv = m.viterbi(P, censored=censored)
        assert abs(sv - sb) < TOL and tuple(yv.tolist()) == yb
        y = torch.tensor(_valid_y(items, seed + 3))
        lzy, _ = tc.chain_forward_backward(q["u"], q["tr"], q["start_c"], q["end_c"], censored, y)
        assert abs(float(lzy) - _logZ_y(items, y.tolist())) < TOL
    one = torch.ones(1, dtype=torch.long)
    for tau in range(1, T + 1):
        pre = {k: v.detach() for k, v in m.potentials(P[:tau]).items()}
        it_c = brute_chain(pre["u"], pre["tr"], pre["start_c"], pre["end_c"], censored=True)
        y = torch.tensor(_valid_y(brute_chain(q["u"], q["tr"], q["start_c"], q["end_c"]), seed + tau))
        _, lzc = m.log_partitions(P[None], one * T, one * tau)
        _, lzyc = m.log_partitions(P[None], one * T, one * tau, y[None])
        assert abs(float(lzc) - _summ(it_c, tau, C)[0]) < TOL
        assert abs(float(lzyc) - _logZ_y(it_c, y[:tau].tolist())) < TOL


def test_chain_bruteforce():
    _check_chain_case(K=3, T=5, seed=10)
    _check_chain_case(K=2, T=6, seed=11)


def test_autograd_marginals():
    """mu_t^c == d log Z / d u_t^c (closed and censored), and E[#segments of length q with label c] == d log Z / d d_c."""
    for kind in ("semi", "chain"):
        m = _rand_model(kind, 3, L=4, Q=2, seed=20) if kind == "semi" else _rand_model(kind, 3, seed=20)
        P = _rand_P(6, 3, 21)
        p = {k: v.detach().clone().requires_grad_(True) for k, v in m.potentials(P).items()}
        T = P.shape[0]
        lengths, tau = torch.tensor([T]), torch.tensor([T])
        for censored in (False, True):
            if kind == "semi":
                lz, lzc = tc.semi_logZ(p["u"][None], p["tr"][None], p["start_c"], p["end_c"], p["d_c"], p["ds_c"],
                                       lengths, tau)
                q = {k: v.detach() for k, v in p.items()}
                _, mu = tc.semi_forward_backward(q["u"], q["tr"], q["start_c"], q["end_c"], q["d_c"], q["ds_c"],
                                                 censored)
            else:
                lz, lzc = tc.chain_logZ(p["u"][None], p["tr"][None], p["start_c"], p["end_c"], lengths, tau)
                q = {k: v.detach() for k, v in p.items()}
                _, mu = tc.chain_forward_backward(q["u"], q["tr"], q["start_c"], q["end_c"], censored)
            target = lzc if censored else lz
            g = torch.autograd.grad(target.sum(), [p["u"]] + ([p["d_c"]] if kind == "semi" else []))
            assert torch.allclose(g[0], mu, atol=1e-9), (kind, censored, (g[0] - mu).abs().max())
            if kind == "semi" and not censored:
                items = brute_semi(q["u"], q["tr"], q["start_c"], q["end_c"], q["d_c"], q["ds_c"])
                _, _, _, _, w = _summ(items, T, m.space.C)
                cnt = torch.zeros_like(q["d_c"])
                for wi, (_, _, segs) in zip(w, items):
                    for qq, c in segs:
                        cnt[qq - 1, c] += wi
                assert torch.allclose(g[1], cnt, atol=1e-9)


def test_padded_batch_matches_single():
    for kind in ("semi", "chain"):
        m = _rand_model(kind, 3, L=4, Q=2, seed=30) if kind == "semi" else _rand_model(kind, 3, seed=30)
        Ps = [_rand_P(6, 3, 31), _rand_P(4, 3, 32)]
        Pb = torch.zeros(2, 6, 3, dtype=torch.float64)
        Pb[0], Pb[1, :4] = Ps[0], Ps[1]
        y = torch.tensor([[1, 1, 0, 2, 2, 4], [3, 0, 0, 1, -1, -1]])
        lengths, tau = torch.tensor([6, 4]), torch.tensor([3, 2])
        nll, full, pref = tc.window_nll(m, Pb, lengths, y, tau)
        nll2, full2, pref2 = tc.window_nll(m, Pb, lengths, y, tau, constrained="dp")
        assert torch.allclose(full, full2, atol=TOL) and torch.allclose(pref, pref2, atol=TOL)
        for b in range(2):
            T = int(lengths[b])
            lz, lzc = m.log_partitions(Ps[b][None], lengths[b:b + 1], tau[b:b + 1])
            lzy, lzyc = m.log_partitions(Ps[b][None], lengths[b:b + 1], tau[b:b + 1], y[b:b + 1, :T])
            assert abs(float((lz - lzy) / T) - float(full[b])) < TOL
            assert abs(float((lzc - lzyc) / tau[b]) - float(pref[b])) < TOL
            assert float(full[b]) >= -TOL and float(pref[b]) >= -TOL


def test_path_score_equals_constrained_dp():
    """Direct reference-path score (training) == y-constrained DP (== enumeration, above), padded batch, every tau."""
    g = torch.Generator().manual_seed(60)
    for kind in ("semi", "chain"):
        m = _rand_model(kind, 3, L=8, Q=3, seed=61) if kind == "semi" else _rand_model(kind, 3, seed=61)
        T = 8
        P = torch.rand(3, T, 3, generator=g, dtype=torch.float64)
        lengths = torch.tensor([8, 5, 1])
        y = torch.randint(0, m.space.C, (3, T), generator=g)
        y[0, 2:6] = y[0, 1]                                    # a long run
        for b in range(3):
            y[b, int(lengths[b]):] = -1
        for t0 in range(1, T + 1):
            tau = torch.minimum(torch.full((3,), t0), lengths)
            p = m.potentials(P)
            if kind == "semi":
                a = tc.semi_path_score(p["u"], p["tr"], p["start_c"], p["end_c"], p["d_c"], p["ds_c"], y, lengths, tau)
            else:
                a = tc.chain_path_score(p["u"], p["tr"], p["start_c"], p["end_c"], y, lengths, tau)
            b_ = m.log_partitions(P, lengths, tau, y)
            assert torch.allclose(a[0], b_[0], atol=1e-8) and torch.allclose(a[1], b_[1], atol=1e-8), (kind, t0)


def test_chain_equals_semi_with_geometric_durations():
    K, T = 3, 6
    semi = _rand_model("semi", K, L=T, Q=T, seed=40)
    chain = tc.ChainTurnCRF(K=K)
    with torch.no_grad():
        for n in ("a", "b", "A", "start", "end", "w"):
            getattr(chain, n).copy_(getattr(semi, n))
        chain.A_self.copy_(torch.tensor([0.3, -0.4, 0.8], dtype=torch.float64))
        semi.d_tab.copy_(chain.A_self[:, None] * torch.arange(T, dtype=torch.float64))   # d_q = (q - 1) * s
    P = _rand_P(T, K, 41)
    lz1, mu1 = semi.marginals(P)
    lz2, mu2 = chain.marginals(P)
    assert abs(float(lz1 - lz2)) < TOL and torch.allclose(mu1, mu2, atol=1e-9)


def test_init_is_independent_P():
    """a = 1, b = 0, flat durations, zero transitions -> mu_t^c = independent-P label probability renormalised over C."""
    K, T = 3, 6
    P = _rand_P(T, K, 50)
    for m in (tc.SemiTurnCRF(K=K, L=T, Q=2), tc.ChainTurnCRF(K=K)):
        _, mu = m.marginals(P)
        ell = tc.content_ell(P, m.space)
        assert torch.allclose(mu, torch.softmax(ell, -1), atol=1e-9)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for f in tests:
        f()
        print(f"PASS {f.__name__}", flush=True)
    print(f"all {len(tests)} tests passed")

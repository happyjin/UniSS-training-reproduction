"""Unit tests for align_crf: semi-Markov forward vs brute-force enumeration (Z_cover, Z_all, E[d_m])."""
import itertools
import math

import torch

from experiments.mist_w3po_v1.talker import align_crf as A


def compositions(T, k, D):
    """ordered k-tuples of durations in [1, D] summing to T."""
    if k == 0:
        return [()] if T == 0 else []
    out = []
    for d in range(1, min(D, T) + 1):
        out += [(d,) + r for r in compositions(T - d, k - 1, D)]
    return out


def brute(e, T, M, lens, D):
    om, mu = A.duration_prior(torch.tensor([T]), lens[None], torch.ones(1, M, dtype=torch.bool), D)
    om = om[0]
    def score(ds):
        t, s = 0, 0.0
        for m, d in enumerate(ds):
            s += float(e[t: t + d, m].sum()) + float(om[d - 1, m])
            t += d
        return s
    full = [(score(ds), ds) for ds in compositions(T, M, D)]
    allz = []
    for k in range(1, M + 1):
        allz += [score(ds) for ds in compositions(T, k, D)]
    lse = lambda xs: max(xs) + math.log(sum(math.exp(x - max(xs)) for x in xs))
    zc = lse([s for s, _ in full])
    ed = [sum(math.exp(s - zc) * ds[m] for s, ds in full) for m in range(M)]
    return zc, lse(allz), ed


def test_vs_brute():
    g = torch.Generator().manual_seed(0)
    for T, M, D in [(5, 2, 4), (6, 3, 6), (7, 3, 3), (4, 4, 2), (8, 2, 8)]:
        e = torch.randn(T, M, generator=g, dtype=torch.float64)
        lens = torch.randint(1, 5, (M,), generator=g)
        zc, za = A.forward(e[None], torch.tensor([T]), torch.tensor([M]), lens[None], D)
        bzc, bza, bed = brute(e, T, M, lens, D)
        assert abs(float(zc) - bzc) < 1e-9 and abs(float(za) - bza) < 1e-9, (T, M, float(zc), bzc, float(za), bza)
        s = torch.zeros(1, M, dtype=torch.float64, requires_grad=True)
        zc2, _ = A.forward(e[None], torch.tensor([T]), torch.tensor([M]), lens[None], D, s)
        ed = torch.autograd.grad(zc2.sum(), s)[0][0]
        assert all(abs(float(a) - b) < 1e-9 for a, b in zip(ed, bed))


def test_batch_padding_and_losses():
    g = torch.Generator().manual_seed(1)
    rows = [(6, 2), (9, 3), (3, 3)]
    Tm, Mm = 9, 3
    e = torch.randn(len(rows), Tm, Mm, generator=g, dtype=torch.float64, requires_grad=True)
    T = torch.tensor([r[0] for r in rows]); M = torch.tensor([r[1] for r in rows])
    lens = torch.randint(1, 4, (len(rows), Mm), generator=g)
    zc, za = A.forward(e, T, M, lens, 5)
    for i, (t, m) in enumerate(rows):
        bzc, bza, _ = brute(e[i, :t, :m].detach(), t, m, lens[i, :m], 5)
        assert abs(float(zc[i]) - bzc) < 1e-9 and abs(float(za[i]) - bza) < 1e-9
    al, prl, p = A.losses(e, T, M, lens, torch.tensor([1, 0, 1]), pr=True, D=5)
    assert torch.all(al >= 0) and torch.all((p >= 0) & (p <= 1)) and prl[1] == 0
    (al.sum() + prl.sum()).backward()
    assert e.grad is not None and torch.isfinite(e.grad).all()
    # infeasible (T < M): no loss for label 1, P = 0
    al2, _, p2 = A.losses(torch.zeros(1, 2, 3, dtype=torch.float64), torch.tensor([2]), torch.tensor([3]),
                          torch.ones(1, 3, dtype=torch.long), torch.tensor([1]), pr=False)
    assert float(al2) == 0 and float(p2) == 0


if __name__ == "__main__":
    test_vs_brute()
    test_batch_padding_and_losses()
    print("ok")

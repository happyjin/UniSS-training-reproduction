"""Tests of crf/schedule_lattice.py (scheme 1', crf_novel_ideas.zh-CN.md §10-§11), CPU only:

  (a) brute-force enumeration on random tiny instances (N <= 5, R <= 4, lo/hi/W constraints, -inf entries, padded
      batches): log Z, w_tok, w_end, E[Lat], Viterbi, FFBS empirical frequencies, infeasibility detection
  (b) autograd: d logZ / d tok_lp = w_tok, d logZ / d end_lp = w_end, d logZ / d lam = -E[Lat]
  (c) HMT oracle (Zhang & Feng, ICLR 2023, Eqs. 16-18, re-derived from the paper's formulas, not from its code):
      HMT's forward algorithm over token states, mapped onto the lattice
        tok_lp[k, i] = log c_{i,k} + log p(y_i | k),  end_lp[l, i] = log(1 - c_{i,l}) on token i's states, 0 elsewhere,
        end_lp[., N] = 0,  lo/hi = token i's state range
      log Z = HMT's log p(y | x), and the lattice's w_tok = HMT's state posteriors p(z_i = k | x, y)
  (d) the global lam* bisection hits its target

    PYTHONPATH=. python -m experiments.mist_w3po_v1.crf.test_schedule_lattice      (or pytest on this file)
"""
from __future__ import annotations

import itertools
import math

import numpy as np
import torch

from experiments.mist_w3po_v1.crf import schedule_lattice as sl

DT = torch.float64
NEG_INF = float("-inf")


# ---------------------------------------------------------------- brute force
def random_instance(rng: np.random.Generator, N: int, R: int, W: int | None = None, p_ninf: float = 0.05) -> dict:
    tok = -rng.exponential(1.5, (R, N))
    end = np.log(rng.uniform(0.02, 1.0, (R, N + 1)))
    tok[rng.uniform(size=(R, N)) < p_ninf] = NEG_INF
    end[rng.uniform(size=(R, N + 1)) < p_ninf] = NEG_INF
    lo = rng.integers(0, R, N)
    hi = np.array([rng.integers(l, R) if rng.uniform() < 0.7 else R - 1 for l in lo])
    lag = rng.uniform(-1.0, 3.0, (R, N))
    return {"tok_lp": tok, "end_lp": end, "lo": lo, "hi": hi, "lag": lag,
            "W": int(W if W is not None else rng.integers(1, N + 2)), "lam": float(rng.uniform(-2, 2))}


def brute(inst: dict) -> dict | None:
    tok, end, lo, hi, lag, W, lam = (inst[k] for k in ("tok_lp", "end_lp", "lo", "hi", "lag", "W", "lam"))
    R, N = end.shape[0], end.shape[1] - 1
    scheds, scores, lats = [], [], []
    for st in itertools.combinations_with_replacement(range(N + 1), R - 1):
        st = list(st) + [N]
        prev, ok, sc, lt = 0, True, 0.0, 0.0
        for r, s in enumerate(st):
            if s - prev > W:
                ok = False
                break
            for j in range(prev, s):
                if not (lo[j] <= r <= hi[j]) or tok[r, j] == NEG_INF:
                    ok = False
                sc += tok[r, j] - lam / max(N, 1) * lag[r, j]
                lt += lag[r, j]
            sc += end[r, s]
            prev = s
        if ok and sc > -1e300:
            scheds.append(tuple(st))
            scores.append(sc)
            lats.append(lt / max(N, 1))
    if not scheds:
        return None
    sc = np.array(scores)
    logZ = np.logaddexp.reduce(sc)
    p = np.exp(sc - logZ)
    w_tok = np.zeros((R, N))
    w_end = np.zeros((R, N + 1))
    for pi, st in zip(p, scheds):
        prev = 0
        for r, s in enumerate(st):
            w_tok[r, prev:s] += pi
            w_end[r, s] += pi
            prev = s
    best = int(np.argmax(sc))
    return {"logZ": logZ, "w_tok": w_tok, "w_end": w_end, "exp_lat": float((p * np.array(lats)).sum()),
            "scheds": scheds, "p": p, "map": scheds[best], "map_score": sc[best],
            "map_unique": (np.sort(sc)[-2] < sc[best] - 1e-9) if len(sc) > 1 else True}


def lat_of(inst: dict) -> sl.Lattice:
    return sl.Lattice.single(inst["tok_lp"], inst["end_lp"], inst["lo"], inst["hi"], inst["lag"], W=inst["W"])


def check_against_brute(inst: dict, lat: sl.Lattice, i: int, mg: dict, vit, bf: dict | None) -> None:
    R, N = inst["end_lp"].shape[0], inst["end_lp"].shape[1] - 1
    if bf is None:
        assert not bool(mg["feasible"][i]), "brute force found no schedule but the lattice is feasible"
        return
    assert bool(mg["feasible"][i])
    assert abs(float(mg["logZ"][i]) - bf["logZ"]) < 1e-9, (float(mg["logZ"][i]), bf["logZ"])
    assert abs(float(mg["logZ_back"][i]) - bf["logZ"]) < 1e-9
    np.testing.assert_allclose(mg["w_tok"][i, :R, :N].numpy(), bf["w_tok"], atol=1e-10)
    np.testing.assert_allclose(mg["w_end"][i, :R, : N + 1].numpy(), bf["w_end"], atol=1e-10)
    assert abs(float(mg["exp_lat"][i]) - bf["exp_lat"]) < 1e-10
    states, score = vit
    assert abs(float(score[i]) - bf["map_score"]) < 1e-9
    if bf["map_unique"]:
        assert tuple(states[i, :R].tolist()) == bf["map"], (states[i, :R].tolist(), bf["map"])
    # every token committed once, one end state per read
    if N:
        np.testing.assert_allclose(mg["w_tok"][i, :R, :N].sum(0).numpy(), np.ones(N), atol=1e-10)
    np.testing.assert_allclose(mg["w_end"][i, :R].sum(-1).numpy(), np.ones(R), atol=1e-10)


def test_brute_force_single():
    rng = np.random.default_rng(0)
    n_feas = n_inf = 0
    for _ in range(600):
        inst = random_instance(rng, int(rng.integers(1, 6)), int(rng.integers(1, 5)))
        lat = lat_of(inst)
        mg = sl.marginals(lat, inst["lam"])
        vit = sl.viterbi(lat, inst["lam"])
        bf = brute(inst)
        check_against_brute(inst, lat, 0, mg, vit, bf)
        n_feas += bf is not None
        n_inf += bf is None
    assert n_feas > 200 and n_inf > 20, (n_feas, n_inf)


def test_brute_force_padded_batch():
    rng = np.random.default_rng(1)
    for _ in range(40):
        insts = [random_instance(rng, int(rng.integers(0, 6)), int(rng.integers(1, 5)), W=W) for W in [5] * 6]
        for x in insts:
            x["lam"] = 0.7
        lat = sl.pack(insts, W=5)
        mg = sl.marginals(lat, 0.7)
        vit = sl.viterbi(lat, 0.7)
        for i, x in enumerate(insts):
            check_against_brute(x, lat, i, mg, vit, brute(x))


def test_ffbs_frequencies():
    rng = np.random.default_rng(2)
    done = 0
    gen = torch.Generator().manual_seed(3)
    while done < 6:
        inst = random_instance(rng, int(rng.integers(2, 6)), int(rng.integers(2, 5)), p_ninf=0.0)
        bf = brute(inst)
        if bf is None or len(bf["scheds"]) < 3:
            continue
        n = 40000
        smp = sl.sample(lat_of(inst), inst["lam"], n=n, generator=gen)[0]
        R = inst["end_lp"].shape[0]
        idx = {s: k for k, s in enumerate(bf["scheds"])}
        cnt = np.zeros(len(idx))
        for row in smp[:, :R].tolist():
            cnt[idx[tuple(row)]] += 1     # KeyError = an inadmissible sample
        freq = cnt / n
        tol = 5 * np.sqrt(bf["p"] * (1 - bf["p"]) / n) + 1e-3
        assert np.all(np.abs(freq - bf["p"]) <= tol), (freq, bf["p"])
        done += 1


def test_blocks_ok_and_schedule_score():
    rng = np.random.default_rng(4)
    for _ in range(200):
        inst = random_instance(rng, int(rng.integers(1, 6)), int(rng.integers(1, 5)))
        bf = brute(inst)
        if bf is None:
            continue
        lat = lat_of(inst)
        st = torch.tensor([bf["map"]])
        assert bool(sl.blocks_ok(lat, st)[0])
        assert abs(float(sl.schedule_score(lat, st, inst["lam"])[0]) - bf["map_score"]) < 1e-9


# ---------------------------------------------------------------- autograd
def test_autograd_marginals():
    rng = np.random.default_rng(5)
    checked = 0
    while checked < 60:
        insts = [random_instance(rng, int(rng.integers(1, 7)), int(rng.integers(1, 6)), W=4) for _ in range(3)]
        toks = [torch.tensor(x["tok_lp"], dtype=DT, requires_grad=True) for x in insts]
        ends = [torch.tensor(x["end_lp"], dtype=DT, requires_grad=True) for x in insts]
        lam = torch.tensor(0.37, dtype=DT, requires_grad=True)
        lat = sl.pack([{"tok_lp": t, "end_lp": e, "lo": x["lo"], "hi": x["hi"], "lag": x["lag"]}
                       for t, e, x in zip(toks, ends, insts)], W=4)
        _, logZ = sl.forward(lat, lam)
        mg = sl.marginals(lat, 0.37)
        feas = mg["feasible"]
        if not bool(feas.any()):
            continue
        (logZ * feas.to(DT)).sum().backward()
        dlam = 0.0
        for i, (t, e) in enumerate(zip(toks, ends)):
            R, N = t.shape
            if not bool(feas[i]):
                continue
            g_t = torch.nan_to_num(t.grad, nan=0.0)
            np.testing.assert_allclose(g_t.numpy(), mg["w_tok"][i, :R, :N].numpy(), atol=1e-10)
            np.testing.assert_allclose(e.grad.numpy(), mg["w_end"][i, :R, : N + 1].numpy(), atol=1e-10)
            dlam -= float(mg["exp_lat"][i])
            checked += 1
        assert abs(float(lam.grad) - dlam) < 1e-9, (float(lam.grad), dlam)


# ---------------------------------------------------------------- HMT oracle
def hmt_log_marginal(emit: torch.Tensor, c: torch.Tensor, L: int) -> torch.Tensor:
    """HMT (ICLR 2023) Eqs. 16-18: log p(y | x) = log sum_z prod_i p(y_i | x<=t_{i,z_i}, y<i, z_i) p(z_i | z_{i-1}),
    t_{i,k} = L + (i-1) + (k-1) (1-based; no |x| cap here), and
    p(z_i = k | z_{i-1} = k') = c_{i,k} prod_{l: t_{i-1,k'} <= t_{i,l} < t_{i,k}} (1 - c_{i,l}) if t_{i,k} >= t_{i-1,k'}
    else 0, with c_{i,K} = 1; token 1 starts from its first state (prod over l < k).
    emit [N, K] = log p(y_i | ., z_i = k), c [N, K] (0-based i, k)."""
    N, K = emit.shape
    t = lambda i, k: L + i + k   # noqa: E731   0-based i, k: t_{i+1, k+1}
    lc, l1c = torch.log(c), torch.log1p(-c)

    def trans(i, kp, k):
        if t(i, k) < t(i - 1, kp):
            return None
        s = lc[i, k]
        for l in range(K):
            if t(i - 1, kp) <= t(i, l) < t(i, k):
                s = s + l1c[i, l]
        return s

    a = [emit[0, k] + lc[0, k] + sum((l1c[0, l] for l in range(k)), torch.zeros((), dtype=DT)) for k in range(K)]
    for i in range(1, N):
        nxt = []
        for k in range(K):
            terms = [a[kp] + tr for kp in range(K) if (tr := trans(i, kp, k)) is not None]
            nxt.append(emit[i, k] + torch.logsumexp(torch.stack(terms), 0))
        a = nxt
    return torch.logsumexp(torch.stack(a), 0)


def hmt_to_lattice(emit: np.ndarray, c: np.ndarray, L: int) -> dict:
    """Reads t = L-1 .. L+N+K-2 (one read before the first state), read r <-> t = L - 1 + r."""
    N, K = emit.shape
    t0 = L - 1
    R = N + K
    tok = np.full((R, N), NEG_INF)
    end = np.zeros((R, N + 1))
    with np.errstate(divide="ignore"):
        for r in range(R):
            t = t0 + r
            for i in range(N):
                k = t - L - i
                if 0 <= k < K:
                    tok[r, i] = math.log(c[i, k]) + emit[i, k]
                    end[r, i] = math.log1p(-c[i, k]) if c[i, k] < 1 else NEG_INF
    lo = np.array([L + i - t0 for i in range(N)])
    hi = np.array([L + i + K - 1 - t0 for i in range(N)])
    return {"tok_lp": tok, "end_lp": end, "lo": lo, "hi": hi, "lag": np.zeros((R, N)), "W": N, "lam": 0.0}


def test_hmt_oracle():
    rng = np.random.default_rng(6)
    for _ in range(150):
        N, K, L = int(rng.integers(1, 7)), int(rng.integers(1, 5)), int(rng.integers(1, 4))
        emit = torch.tensor(np.log(rng.uniform(0.01, 1.0, (N, K))), dtype=DT, requires_grad=True)
        c_np = rng.uniform(0.02, 0.98, (N, K))
        c_np[:, K - 1] = 1.0
        logp = hmt_log_marginal(emit, torch.tensor(c_np, dtype=DT), L)
        logp.backward()
        inst = hmt_to_lattice(emit.detach().numpy(), c_np, L)
        lat = lat_of(inst)
        mg = sl.marginals(lat, 0.0)
        assert bool(mg["feasible"][0])
        assert abs(float(mg["logZ"][0]) - float(logp)) < 1e-10, (float(mg["logZ"][0]), float(logp))
        # HMT state posteriors p(z_i = k | x, y) = d log p / d emit[i, k] = lattice w_tok at read t_{i,k}
        post = emit.grad.numpy()
        for i in range(N):
            for k in range(K):
                r = (L + i + k) - (L - 1)
                assert abs(float(mg["w_tok"][0, r, i]) - post[i, k]) < 1e-10
        if N <= 4 and inst["end_lp"].shape[0] <= 6:   # and the brute force agrees too
            bf = brute(inst)
            assert abs(bf["logZ"] - float(logp)) < 1e-10


def test_hmt_normalised():
    """Sanity of the oracle itself: with emissions = 0 (prob 1) HMT's transitions sum to 1, so log p = 0."""
    rng = np.random.default_rng(7)
    for _ in range(30):
        N, K, L = int(rng.integers(1, 6)), int(rng.integers(1, 5)), int(rng.integers(1, 3))
        c = rng.uniform(0.05, 0.95, (N, K))
        c[:, K - 1] = 1.0
        assert abs(float(hmt_log_marginal(torch.zeros((N, K), dtype=DT), torch.tensor(c, dtype=DT), L))) < 1e-12


# ---------------------------------------------------------------- lam* bisection
def test_solve_lambda():
    rng = np.random.default_rng(8)
    insts = []
    while len(insts) < 40:
        x = random_instance(rng, int(rng.integers(2, 7)), int(rng.integers(2, 6)), W=6, p_ninf=0.0)
        if brute(x) is not None:
            insts.append(x)
    lats = [sl.pack(insts[:20], W=6), sl.pack(insts[20:], W=6)]
    e_lo, e_hi = sl.expected_lat(lats, 20.0), sl.expected_lat(lats, -20.0)
    assert e_lo < e_hi
    target = 0.3 * e_lo + 0.7 * e_hi
    out = sl.solve_lambda(lats, target)
    assert out["bracketed"] and abs(out["exp_lat"] - target) < 1e-6, out
    # E[Lat] is non-increasing in lam
    vals = [sl.expected_lat(lats, x) for x in (-3.0, -1.0, 0.0, 1.0, 3.0)]
    assert all(a >= b - 1e-12 for a, b in zip(vals, vals[1:])), vals


def test_empty_stream():
    lat = sl.Lattice.single(np.zeros((3, 0)), np.log(np.full((3, 1), 0.5)), np.zeros(0, int), np.zeros(0, int),
                            np.zeros((3, 0)))
    mg = sl.marginals(lat, 1.0)
    assert bool(mg["feasible"][0]) and abs(float(mg["logZ"][0]) - 3 * math.log(0.5)) < 1e-12


if __name__ == "__main__":
    import time

    tests = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    t0 = time.time()
    for name, fn in tests:
        t1 = time.time()
        fn()
        print(f"PASS {name} ({time.time() - t1:.1f}s)", flush=True)
    print(f"ALL {len(tests)} TESTS PASSED ({time.time() - t0:.1f}s)")

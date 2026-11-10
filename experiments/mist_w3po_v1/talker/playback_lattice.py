"""PL-EM's playback lattice (reports/mist_w3po_v1/talker_crf_prereg_v1.zh-CN.md §1.1; talker_crf_ideas.zh-CN.md §2.2).

One output channel (a window x speaker stream) with blocks n = 0..N-1, each available at a_n (ms, the online protocol of
speech_latency_lit.py) and holding candidates c with duration d_c (ms) and features (lpbar = talker log-prob per token,
f = -min(CTC NLL per character, 10)). State = backlog b_n (ms) on an 80 ms grid capped at B = 8 s, b_0 = 0; the
(max, +) transition b_{n+1} = ceil_D(max(0, a_n + b_n + d_c - a_{n+1})); edge potential

    psi_n(b, c) = tau * lpbar_c + beta * f_c - lam * l_n(b, c),   l_n(b, c) = W_n * b / 1000 + d_c / 1000 * (W_n + 1) / 2

(l: the block's words spread evenly over its playback; the sum of their spoken times after a_n, seconds x words).
Exact sum-product forward-backward in log space (marginals per block / state / candidate, log Z, E_q[sum l]), Viterbi,
the conditional q(c | b_n = s), and the dual bisection of lam to a corpus budget of E_q[sum l] / sum W.
"""
from __future__ import annotations

import math

import numpy as np

DELTA_MS = 80
B_MS = 8000
TAU = 0.5
BETA = 5.0
NLL_CAP = 10.0


def n_states(delta: int = DELTA_MS, cap: int = B_MS) -> int:
    return cap // delta + 1


def quant(x_ms: float, delta: int = DELTA_MS, cap: int = B_MS) -> int:
    """state index of a backlog: ceil to the grid, capped."""
    return int(min(cap // delta, math.ceil(max(0.0, x_ms) / delta - 1e-9)))


class Stream:
    """a: availability (ms) per block; dur[n]: candidate durations (ms); feat[n]: tau*lpbar + beta*f per candidate;
    words[n]: W_n."""

    def __init__(self, a, dur, feat, words, delta: int = DELTA_MS, cap: int = B_MS):
        self.a = [float(x) for x in a]
        self.dur = [np.asarray(d, dtype=np.float64) for d in dur]
        self.feat = [np.asarray(f, dtype=np.float64) for f in feat]
        self.words = [max(1, int(w)) for w in words]
        self.delta, self.cap = delta, cap
        self.S = n_states(delta, cap)
        sv = np.arange(self.S, dtype=np.float64) * delta
        self.lat, self.nxt = [], []
        for n in range(len(self.a)):
            W = self.words[n]
            self.lat.append(W * sv[:, None] / 1000.0 + self.dur[n][None, :] / 1000.0 * (W + 1) / 2.0)   # S x C
            if n + 1 < len(self.a):
                x = self.a[n] + sv[:, None] + self.dur[n][None, :] - self.a[n + 1]
                self.nxt.append(np.minimum(cap // delta, np.ceil(np.maximum(0.0, x) / delta - 1e-9)).astype(np.int64))
            else:
                self.nxt.append(None)

    def psi(self, n: int, lam: float) -> np.ndarray:
        return self.feat[n][None, :] - lam * self.lat[n]

    def forward_backward(self, lam: float):
        """-> (log Z, marginals list of S x C arrays, E_q[sum l], beta messages)."""
        N, S = len(self.a), self.S
        alpha = [np.full(S, -np.inf)]
        alpha[0][0] = 0.0
        psis = [self.psi(n, lam) for n in range(N)]
        for n in range(N - 1):
            v = alpha[n][:, None] + psis[n]
            m = np.max(v)
            acc = np.zeros(S)
            np.add.at(acc, self.nxt[n].ravel(), np.exp(v - m).ravel())
            with np.errstate(divide="ignore"):
                alpha.append(np.log(acc) + m)
        beta = [None] * N
        beta[N - 1] = _lse(psis[N - 1], axis=1)
        for n in range(N - 2, -1, -1):
            beta[n] = _lse(psis[n] + beta[n + 1][self.nxt[n]], axis=1)
        logz = _lse(alpha[0] + beta[0], axis=0)
        marg, el = [], 0.0
        for n in range(N):
            tail = beta[n + 1][self.nxt[n]] if n + 1 < N else 0.0
            with np.errstate(invalid="ignore"):
                q = np.exp(alpha[n][:, None] + psis[n] + tail - logz)
            q = np.nan_to_num(q)
            marg.append(q)
            el += float((q * self.lat[n]).sum())
        return float(logz), marg, el, beta

    def viterbi(self, lam: float) -> list[int]:
        N, S = len(self.a), self.S
        psis = [self.psi(n, lam) for n in range(N)]
        delta = np.full(S, -np.inf)
        delta[0] = 0.0
        back = []
        for n in range(N - 1):
            v = delta[:, None] + psis[n]
            nd = np.full(S, -np.inf)
            arg = np.full((S, 2), -1, dtype=np.int64)
            flat_v, flat_n = v.ravel(), self.nxt[n].ravel()
            order = np.argsort(-flat_v, kind="stable")
            seen = np.zeros(S, dtype=bool)
            for i in order:
                if not np.isfinite(flat_v[i]):
                    break
                t = flat_n[i]
                if not seen[t]:
                    seen[t] = True
                    nd[t] = flat_v[i]
                    arg[t] = divmod(i, v.shape[1])
            back.append(arg)
            delta = nd
        v = delta[:, None] + psis[N - 1]
        s, c = divmod(int(np.argmax(v)), v.shape[1])
        path = [c]
        for n in range(N - 2, -1, -1):
            s, c = back[n][s]
            path.append(int(c))
        return path[::-1]

    def path_states(self, path: list[int]) -> list[int]:
        s, out = 0, []
        for n, c in enumerate(path):
            out.append(s)
            if n + 1 < len(self.a):
                s = int(self.nxt[n][s, c])
        return out

    def path_latency(self, path: list[int]) -> float:
        return float(sum(self.lat[n][s, c] for n, (s, c) in enumerate(zip(self.path_states(path), path))))

    def conditional(self, n: int, s: int, lam: float, beta=None) -> np.ndarray:
        """q(c | b_n = s): psi_n(s, c) + beta_{n+1}(next(s, c)), normalised over c."""
        if beta is None:
            beta = self.forward_backward(lam)[3]
        v = self.psi(n, lam)[s] + (beta[n + 1][self.nxt[n][s]] if n + 1 < len(self.a) else 0.0)
        v = v - _lse(v, axis=0)
        return np.exp(v)


def _lse(x, axis):
    m = np.max(x, axis=axis, keepdims=True)
    m = np.where(np.isfinite(m), m, 0.0)
    with np.errstate(divide="ignore"):
        return np.squeeze(np.log(np.sum(np.exp(x - m), axis=axis, keepdims=True)) + m, axis=axis)


def corpus_latency(streams: list[Stream], lam: float) -> float:
    """E_q[sum l] / sum W over the corpus (seconds)."""
    num = sum(st.forward_backward(lam)[2] for st in streams)
    return num / max(1, sum(sum(st.words) for st in streams))


def solve_lambda(streams: list[Stream], budget: float, lam_max: float = 50.0, iters: int = 40) -> tuple[float, float]:
    """the dual bisection: the smallest lam >= 0 with corpus_latency(lam) <= budget (0 when already met, lam_max when
    out of reach). Returns (lam, corpus_latency(lam))."""
    e0 = corpus_latency(streams, 0.0)
    if e0 <= budget:
        return 0.0, e0
    hi_e = corpus_latency(streams, lam_max)
    if hi_e > budget:
        return lam_max, hi_e
    lo, hi = 0.0, lam_max
    for _ in range(iters):
        mid = (lo + hi) / 2
        if corpus_latency(streams, mid) > budget:
            lo = mid
        else:
            hi = mid
    return hi, corpus_latency(streams, hi)


def brute_force(st: Stream, lam: float):
    """every path enumerated: (log Z, per-block candidate marginals, best path, E[sum l]) -- the P2 check."""
    import itertools

    N = len(st.a)
    psis = [st.psi(n, lam) for n in range(N)]
    scores, paths, lats = [], [], []
    for path in itertools.product(*[range(len(d)) for d in st.dur]):
        ss = st.path_states(list(path))
        scores.append(sum(psis[n][s, c] for n, (s, c) in enumerate(zip(ss, path))))
        lats.append(st.path_latency(list(path)))
        paths.append(path)
    scores = np.asarray(scores)
    logz = float(_lse(scores, axis=0))
    p = np.exp(scores - logz)
    marg = [np.zeros(len(d)) for d in st.dur]
    for w, path in zip(p, paths):
        for n, c in enumerate(path):
            marg[n][c] += w
    return logz, marg, list(paths[int(np.argmax(scores))]), float((p * np.asarray(lats)).sum())

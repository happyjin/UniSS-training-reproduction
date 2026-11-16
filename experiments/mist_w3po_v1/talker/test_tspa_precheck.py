"""TS-PA starvation potential: forward-backward against brute-force enumeration with the pause indicator recomputed
from raw times (independent of TSStream.pz). python -m experiments.mist_w3po_v1.talker.test_tspa_precheck"""
import itertools

import numpy as np

from experiments.mist_w3po_v1.talker import tspa_precheck_p1 as T


def brute(st, lam, kappa, a, dur, feat, words, texts):
    scores, wps, lats = [], [], []
    for path in itertools.product(*[range(len(d)) for d in dur]):
        ss = st.path_states(list(path))
        sc = wp = la = 0.0
        for n, (s, c) in enumerate(zip(ss, path)):
            W = max(1, words[n])
            l = W * s * 80 / 1000.0 + dur[n][c] / 1000.0 * (W + 1) / 2.0
            pz = 0.0
            if n + 1 < len(a):
                pz = float(a[n + 1] - (a[n] + 80 * s + dur[n][c]) >= 500)
            om = 0.25 if T.sent_end(texts[n]) else 1.0
            sc += feat[n][c] - lam * l - kappa * om * pz
            wp += om * pz
            la += l
        scores.append(sc)
        wps.append(wp)
        lats.append(la)
    s = np.array(scores)
    m = s.max()
    logz = m + np.log(np.exp(s - m).sum())
    p = np.exp(s - logz)
    return logz, float(p @ np.array(wps)), float(p @ np.array(lats))


def test_against_brute(trials=60):
    rng = np.random.default_rng(0)
    for t in range(trials):
        N = int(rng.integers(1, 5))
        a = np.cumsum(rng.uniform(0, 3000, N)).tolist()
        dur = [rng.integers(5, 150, int(rng.integers(1, 4))).astype(float) * 20 for _ in range(N)]
        feat = [rng.normal(-3, 1, len(d)) for d in dur]
        words = rng.integers(1, 9, N).tolist()
        texts = [rng.choice(["ok so", "fine.", "好的，", "是的。", "what?"]) for _ in range(N)]
        rec = {"a": a, "dur": [d.tolist() for d in dur], "feat": [f.tolist() for f in feat], "words": words,
               "texts": texts, "f": [f.tolist() for f in feat], "lp": [f.tolist() for f in feat], "chosen": [0] * N}
        st = T.make_stream(rec)
        lam, kappa = float(rng.uniform(0, 2)), float(rng.uniform(0, 5))
        st.kappa = kappa
        logz, marg, el, _ = st.forward_backward(lam)
        wp = sum(st.om[n] * float((q * st.pz[n]).sum()) for n, q in enumerate(marg))
        bz, bwp, bl = brute(st, lam, kappa, a, dur, feat, words, texts)
        assert abs(logz - bz) < 1e-9, (t, logz, bz)
        assert abs(wp - bwp) < 1e-9, (t, wp, bwp)
        assert abs(el - bl) < 1e-9, (t, el, bl)
        s = T._sums(st, lam, kappa)
        assert abs(s[1] - bwp) < 1e-9
    print(f"test_against_brute: {trials}/{trials} ok")


def test_sent_end():
    assert T.sent_end("Yes.") and T.sent_end("好的。") and T.sent_end("真的吗？”") and T.sent_end("what?)")
    assert not T.sent_end("so,") and not T.sent_end("好的，") and not T.sent_end("")
    print("test_sent_end: ok")


if __name__ == "__main__":
    test_sent_end()
    test_against_brute()


def test_replay_vs_enumeration(trials=40):
    """tspa_causal_replay: the exact forward expectation equals the sum over paths with probability prod pi2."""
    from experiments.mist_w3po_v1.talker import tspa_causal_replay as CR

    rng = np.random.default_rng(1)
    for t in range(trials):
        N = int(rng.integers(1, 5))
        a = np.cumsum(rng.uniform(0, 3000, N)).tolist()
        dur = [rng.integers(5, 150, int(rng.integers(1, 4))).astype(float) * 20 for _ in range(N)]
        f = [np.round(rng.normal(-1, 0.5, len(d)), 1) for d in dur]          # rounding makes ties
        words = rng.integers(1, 9, N).tolist()
        texts = [rng.choice(["ok so", "fine.", "好的，", "是的。"]) for _ in range(N)]
        rec = {"a": a, "dur": [d.tolist() for d in dur], "feat": [x.tolist() for x in f], "words": words,
               "texts": texts, "f": [x.tolist() for x in f], "lp": [x.tolist() for x in f], "chosen": [0] * N}
        st = T.make_stream(rec)
        st.fast = [rng.integers(0, 2, len(d)).astype(float) for d in dur]
        st.se = [float(T.sent_end(x)) for x in texts]
        th, mu, sd = rng.normal(0, 1, 12), np.zeros(12), np.ones(12)
        for bo2 in (True, False):
            acc = CR.replay_stream(st, th, mu, sd, bo2)
            el = pz = 0.0
            for path in itertools.product(*[range(len(d)) for d in dur]):
                ss = st.path_states(list(path))
                pr = 1.0
                for n, (s, c) in enumerate(zip(ss, path)):
                    p = CR.policy(st, n, 0.5 * CR.bin_s(s), th, mu, sd)
                    p = CR.best_of_2(p, st.f[n]) if bo2 else p
                    pr *= p[c]
                el += pr * sum(st.lat[n][s, c] for n, (s, c) in enumerate(zip(ss, path)))
                pz += pr * sum(st.pz[n][s, c] for n, (s, c) in enumerate(zip(ss, path)))
            assert abs(acc[0] - el) < 1e-9 and abs(acc[2] - pz) < 1e-9, (t, bo2, acc[0], el, acc[2], pz)
        p = CR.policy(st, 0, 0.5, th, mu, sd)
        assert abs(CR.best_of_2(p, st.f[0]).sum() - 1) < 1e-12
    print(f"test_replay_vs_enumeration: {trials}/{trials} ok")


if __name__ == "__main__":
    test_replay_vs_enumeration()

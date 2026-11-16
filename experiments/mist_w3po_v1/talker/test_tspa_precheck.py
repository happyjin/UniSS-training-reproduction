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

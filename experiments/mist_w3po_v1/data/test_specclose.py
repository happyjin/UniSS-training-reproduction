"""CPU tests of data/specclose.py (SpecClose prereg §2 P1). PYTHONPATH=. python -m experiments.mist_w3po_v1.data.test_specclose"""
from __future__ import annotations

import numpy as np

from experiments.mist_w3po_v1.data import specclose as S
from experiments.mist_w3po_v1.data.build_windows_zh2en import online_windows


def rand_act(n: int, seed: int) -> np.ndarray:
    r = np.random.default_rng(seed)
    a = np.zeros(n, dtype=bool)
    i = 0
    while i < n:
        on = int(r.integers(1, 60))
        a[i: i + on] = True
        i += on + int(r.choice([1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 20, 40]))
    return a


def test_simulate_matches_online_windows():
    for seed in range(30):
        act = rand_act(4000, seed)
        post = np.zeros((len(act), 8), dtype=np.float32)
        post[act, seed % 8] = 0.9
        ref = online_windows(post, 0.6)
        got = [(w["st"], w["en"], w["j"]) for w in S.simulate(act)]
        assert got == ref
        never = [(w["st"], w["en"], w["j"], w["kind"]) for w in S.simulate(act, lambda f, u: False)]
        assert never == [(w["st"], w["en"], w["j"], w["kind"]) for w in S.simulate(act)]


def test_chunk_times():
    for f in range(400):
        assert S.t_chunk(f) <= f * S.FR + S.LA + 1e-9
        assert S.t_chunk(f) >= f * S.FR + 0.40 - 1e-9
        if f % 9 == 0:
            assert abs(S.t_chunk(f) - (f * S.FR + S.LA)) < 1e-9
        assert S.vis_end(S.t_chunk(f)) == S.chunk_end(f)
        assert S.vis_end(S.t_chunk(f) - 1e-3) == S.chunk_end(f) - 9


def test_spec_close_sees_all_active_frames():
    for seed in range(20):
        act = rand_act(3000, 100 + seed)
        for w in S.simulate(act, lambda f, u: True):
            if w["kind"] != "spec":
                continue
            f = w["j"]
            assert not act[f] and act[f - 1] and w["en"] == f
            assert w["tau"] >= S.t_chunk(f) - 1e-9 and w["tau"] >= (f + S.GAP) * S.FR - 1e-9
            assert S.vis_end(w["tau"]) >= w["en"] - 1          # every active frame's posterior is out
            assert S.vis_end(w["tau"]) < f + S.GAP - 1          # and the gap is NOT fully known (else no call)
            assert w["tau"] < S.t_chunk(f + S.GAP - 1) + 1e-9   # never later than the chunk close


def test_calls_only_when_unknown_frames():
    act = rand_act(5000, 7)
    seen = []
    S.simulate(act, lambda f, u: seen.append((f, u)) or False)
    for f, u in seen:
        assert 1 <= u <= S.GAP - 1 and u == f + S.GAP - 1 - S.chunk_end(f)
        assert not act[f + 1: S.chunk_end(f) + 1].any()


def test_candidate_labels():
    act = np.zeros(200, dtype=bool)
    act[10:30] = True          # pause at f = 30 (30 % 9 = 3 -> chunk_end 35, u = 37 - 35 = 2)
    act[36] = True             # resumes at f + 6: in the unknown frames 36, 37
    act[37:50] = True
    c = {(f, u): (y, real) for f, u, y, real in S.candidates(act, augment=True)}
    assert c[(30, 2)] == (True, True)
    assert (30, 1) not in c                     # frame 36 is among the first 6 (P-known if u = 1): no call
    assert c[(30, 3)] == (True, False) and c[(30, 7)] == (True, False)
    # a final pause: nothing after
    act2 = np.zeros(200, dtype=bool)
    act2[10:30] = True
    c2 = {(f, u): (y, real) for f, u, y, real in S.candidates(act2, augment=True)}
    assert all(not y for (f, u), (y, r) in c2.items() if f == 30) and c2[(30, 2)][1]


def test_window_row_cut_covers_active_frames():
    for seed in range(10):
        act = rand_act(3000, 300 + seed)
        sess = len(act) * S.FR + 1.0
        for w in S.simulate(act, lambda f, u: (f % 2) == 0):
            r = S.window_row(w, "M", sess)
            if w["kind"] in ("gap", "spec"):
                assert r["final_post_cut_ms"] >= min(r["dur_ms"], (w["en"] - r["f0"]) * S.FR_MS)
            base_close = w["j"] * S.FR + S.LA if w["kind"] != "spec" else None
            if base_close is not None and w["kind"] != "end":
                assert r["close_detect_ms"] <= round((base_close - r["start_s"]) * 1000) + 1
            assert r["open_detect_ms"] <= round((w["st"] * S.FR + S.LA - r["start_s"]) * 1000) + 1


def test_theta_for():
    p = np.array([0.01, 0.02, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.97, 0.99] * 10)
    y = np.ones_like(p)
    th = S.theta_for(p, y, 0.10)
    assert (p[y == 1] < th).mean() <= 0.10
    assert S.theta_for(p, y, 0.0) == p.min()


def main() -> None:
    import time
    t0 = time.time()
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t1 = time.time()
        t()
        print(f"PASS {t.__name__} ({time.time() - t1:.1f}s)", flush=True)
    print(f"ALL {len(tests)} PASSED ({time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()

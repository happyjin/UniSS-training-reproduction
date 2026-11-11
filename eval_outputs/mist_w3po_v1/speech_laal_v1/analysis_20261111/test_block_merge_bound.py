"""CPU tests for block_merge_bound.py (grouping rules, member durations, oracle DP vs brute force)."""
import itertools
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import block_merge_bound as B  # noqa: E402


def test_group_hold_read_level():
    # reads at 1000 (1+2 words), 2000 (1 word), 3000 (4 words); window 4000
    wr, nw = [1000, 1000, 2000, 3000], [1, 2, 1, 4]
    assert B.group_hold(wr, nw, 4000, 0, True) == [([0, 1], 1000), ([2], 2000), ([3], 3000)]
    assert B.group_hold(wr, nw, 4000, 3, True) == [([0, 1], 1000), ([2, 3], 3000)]
    assert B.group_hold(wr, nw, 4000, 5, True) == [([0, 1, 2, 3], 3000)]
    # held to the end of the stream -> released at the final read (dur_ms)
    assert B.group_hold(wr, nw, 4000, 9, True) == [([0, 1, 2, 3], 4000)]


def test_group_hold_literal_and_final_read():
    wr, nw = [1000, 1000, 2000, 4000], [1, 2, 1, 1]
    assert B.group_hold(wr, nw, 4000, 3, False) == [([0, 1], 1000), ([2, 3], 4000)]
    # a final-read block closes the held group whatever its size
    assert B.group_hold([1000, 4000], [1, 1], 4000, 6, False) == [([0, 1], 4000)]
    assert B.group_hold([1000, 1000], [4, 4], 4000, 3, False) == [([0], 1000), ([1], 1000)]


def test_member_durs():
    assert B.member_durs([1.0, 1.0], [3, 3], 0.0) == [1.0, 1.0]
    assert B.member_durs([1.0, 1.0], [3, 3], 0.4) == [1.0, 0.6]
    assert B.member_durs([1.0, 0.3], [3, 2], 0.4) == [1.0, 2 * B.FLOOR_S_PER_WORD]


def _cost(G, wr, nw, du, av, c):
    E, cost = -1e18, 0.0
    for mem, rel in G:
        t = max(av(rel), E)
        for k, dk in zip(mem, B.member_durs([du[i] for i in mem], [nw[i] for i in mem], c)):
            cost += nw[k] * t + dk * (nw[k] + 1) / 2 if nw[k] else 0.0
            t += dk
        E = t
    return cost


def _brute(wr, nw, du, av, c):
    n, best = len(wr), None
    for cuts in itertools.product([0, 1], repeat=n - 1):
        G, cur = [], [0]
        for i, cut in enumerate(cuts, start=1):
            if cut:
                G.append(cur)
                cur = [i]
            else:
                cur.append(i)
        G.append(cur)
        G = [(m, max(wr[i] for i in m)) for m in G]
        v = _cost(G, wr, nw, du, av, c)
        if best is None or v < best[0] - 1e-9:
            best = (v, G)
    return best


def test_oracle_matches_brute_force():
    rng = random.Random(0)
    for _ in range(200):
        n = rng.randint(1, 7)
        wr = sorted(rng.choice([500, 1000, 1500, 2000, 3000]) for _ in range(n))
        nw = [rng.randint(0, 5) for _ in range(n)]
        du = [0.3 + 0.25 * k + rng.random() * 0.3 for k in nw]
        c = rng.choice([0.0, 0.2, 0.45])
        av = lambda rel: max(rel, 700) / 1000  # noqa: E731
        G = B.group_oracle(wr, nw, du, av, c)
        assert sorted(i for m, _ in G for i in m) == list(range(n))
        bv, _ = _brute(wr, nw, du, av, c)
        assert abs(_cost(G, wr, nw, du, av, c) - bv) < 1e-6


def test_oracle_c0_is_identity_cost():
    # with no overhead saved, merging can only delay: the oracle's cost equals the unmerged cost
    rng = random.Random(1)
    for _ in range(100):
        n = rng.randint(1, 6)
        wr = sorted(rng.choice([500, 1000, 2000]) for _ in range(n))
        nw = [rng.randint(1, 4) for _ in range(n)]
        du = [0.4 + 0.2 * k for k in nw]
        av = lambda rel: rel / 1000  # noqa: E731
        ident = [([i], wr[i]) for i in range(n)]
        G = B.group_oracle(wr, nw, du, av, 0.0)
        assert abs(_cost(G, wr, nw, du, av, 0.0) - _cost(ident, wr, nw, du, av, 0.0)) < 1e-9


if __name__ == "__main__":
    for name, f in list(globals().items()):
        if name.startswith("test_"):
            f()
            print("ok", name)

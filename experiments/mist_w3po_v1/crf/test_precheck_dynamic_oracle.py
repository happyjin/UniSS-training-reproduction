"""CPU tests of crf/precheck_dynamic_oracle.py (CRF dynamic oracle precheck): the generalised order rule, the residual
lattice vs brute force (seeds disjoint from the O3 run) and its negative tests, a hand-computed oracle, the label-path
special case (the residual lattice at a label state == v2''s lattice from the next read), the coverage rules on a toy
meeting, the trace rebuild on a toy trace, and the scoring positions. pytest is not installed:

    PYTHONPATH=.:third_party/Megatron-LM python -m experiments.mist_w3po_v1.crf.test_precheck_dynamic_oracle
"""
from __future__ import annotations

import math
from collections import Counter

import numpy as np

from experiments.mist_w3po_v1.crf import precheck_serial as S
from experiments.mist_w3po_v1.crf import precheck_serial_v2 as V
from experiments.mist_w3po_v1.crf import precheck_serial_v2p as VP
from experiments.mist_w3po_v1.crf import precheck_dynamic_oracle as X


def test_nf_gen_reduces_to_v2p():
    rng = np.random.default_rng(7)
    n = 0
    for _ in range(80):
        rec = X.synthetic_residual(rng, int(rng.integers(2, 5)), int(rng.integers(1, 7)), int(rng.integers(1, 4)))
        rec["overdue"] = []
        lat = V.lattice(rec)
        for r in range(lat["R"]):
            for i in range(len(V.prev_states(lat, r))):
                _, pend = V.context(rec, lat, r, i)
                for D in lat["states"][r]:
                    assert X.nf_gen(rec, lat["rho"], pend, D) == VP.newfirst(rec, lat["rho"], pend, D)
                    n += 1
    assert n > 100


def test_nf_gen_overdue_at_read0():
    rec = hand_rec()
    lat = V.lattice(rec)
    st = lat["states"][0]
    full = next(D for D in st if len(D) == 2)
    assert X.nf_gen(rec, lat["rho"], [], full) == ((1,), (0,))          # the due-now other-speaker chunk first
    assert VP.newfirst(rec, lat["rho"], [], full) is None               # v2' has no choice at the residual read 0
    assert X.nf_gen(rec, lat["rho"], [], (0,)) is None and X.nf_gen(rec, lat["rho"], [], (1,)) is None


def test_residual_brute_force_and_negatives():
    st = Counter()
    for seed in range(100, 260):                                         # O3 uses seeds 20261110..
        x = X._o3_syn(seed)
        if x["main"].get("skipped"):
            continue
        st["checked"] += 1
        st["pass"] += x["main"]["pass"]
        st["real0"] += x["main"]["real_choice_read0"]
        if x["N1"] is not None:
            st["n1"] += 1
            st["n1_caught"] += x["N1"]["pass"] == 0
        if x["N2"] is not None:
            st["n2"] += 1
            st["n2_caught"] += x["N2"]["pass"] == 0
    assert st["pass"] == st["checked"] > 100, st
    assert st["real0"] > 3 and st["n1"] == st["n1_caught"] > 3 and st["n2"] == st["n2_caught"] > 20, st


def hand_rec(own=(5, 6, 7)):
    """residual read 0 (non-final) with an overdue chunk 0 of slot 1 and a due-now chunk 1 of slot 2; final read 1."""
    ids0 = [S.SPK0 + 0, S.NEW0 + 1, 1000, S.NEW0 + 3]
    ids1 = [S.SPK0 + 1, S.NEW0 + 2, 1001, 1002, S.NEW0 + 9]
    ch = [{"utt": 0, "orig": 0, "rid": "u0", "slot": 1, "tok0": 0, "tok1": 4, "src_start_ms": 80, "src_end_ms": 240,
           "commit_ms": 1240, "rt": 0, "rho_lab": -1, "piece": "a"},
          {"utt": 1, "orig": 1, "rid": "u1", "slot": 2, "tok0": 4, "tok1": 9, "src_start_ms": 160, "src_end_ms": 720,
           "commit_ms": 1720, "rt": 0, "rho_lab": 0, "piece": "b"}]
    return {"id": "hand", "chunks": ch, "S_ids": ids0 + ids1, "n_read": 2, "t_ms": [2000, 2600], "dur_ms": 2600,
            "beta_heur": [9, 9], "n_tok": 9, "units": [{"utt": 0, "a_ms": 80}, {"utt": 1, "a_ms": 160}], "overdue": [0],
            "own": list(own)}


def test_hand_oracle():
    rec = hand_rec()
    own = rec["own"]
    i0, i1 = rec["S_ids"][:4], rec["S_ids"][4:]
    psi = lambda r, hist, w: V.dummy_psi(r, r == 1, own + hist, w)   # noqa: E731
    paths = {                                                          # (first action, score) of the 5 paths
        "wait": [("wait", psi(0, [], []) + psi(1, [], i0 + i1))],
        "c0": [("spk1", psi(0, [], i0) + psi(1, i0, i1))],
        "c1": [("spk2", psi(0, [], i1) + psi(1, i1, i0))],
        "both": [("spk1", psi(0, [], i0 + i1) + psi(1, i0 + i1, [])), ("spk2", psi(0, [], i1 + i0) + psi(1, i0 + i1, []))],
    }
    for tau in (0.5, 1.0):
        allp = [x for v in paths.values() for x in v]
        Z = math.log(sum(math.exp(tau * s) for _, s in allp))
        q = Counter()
        for a, s in allp:
            q[a] += math.exp(tau * s - Z)
        plan = X.plan_state(rec, skip_forced=False)
        reads = VP.assemble_p(plan, V.dummy_res(plan))
        o = X.oracle(rec, plan, reads, tau)
        assert abs(o["logZ"] - Z) < 1e-12
        for a in X.ACTS:
            assert abs(o["q"][a] - q.get(a, 0.0)) < 1e-12, (a, o["q"][a], q.get(a))
        assert abs(sum(o["q"].values()) - 1) < 1e-12


def test_label_path_special_case():
    """at a label state (end of read i on the label path) nothing is pending: residual read 0 is forced (q(wait) = 1)
    and every later residual request is a v2' request of the window with the same history and write set."""
    n = 0
    for rec in S.sample("dev")[:40]:
        lat = V.lattice(rec)
        plan_v = VP.plan_window_p(rec, lat, skip_forced=False)
        reqv = {(q["r"], tuple(q["hist_ids"]), tuple(q["forced_ids"])) for q in plan_v["reqs"]}
        beta = [0] + list(rec["beta_heur"])
        items = [{"rid": f"c{c}", "slot": ch["slot"], "ids": S.ids_of(rec, [c]), "piece": ch.get("piece", ""),
                  "src_start_ms": ch["src_start_ms"], "src_end_ms": ch["src_end_ms"], "commit_ms": ch["commit_ms"],
                  "rho": lat["rho"][c]} for c, ch in enumerate(rec["chunks"])]
        for i in range(lat["R"] - 1):
            unc = [c for c in range(len(items)) if items[c]["rho"] > i]
            res = X.residual_rec(rec["id"], items, rec["t_ms"], i, unc, rec["S_ids"][: beta[i + 1]])
            assert not res["overdue"]
            plan = X.plan_state(res, skip_forced=False)
            assert plan["lat"]["states"][0] == [()]
            for q in plan["reqs"]:
                if q["r"] == 0:
                    continue
                assert (q["r"] + i, tuple(q["hist_ids"]), tuple(q["forced_ids"])) in reqv
                n += 1
            free = X.plan_state(res, skip_forced=True)
            if free["reqs"]:
                o = X.oracle(res, free, VP.assemble_p(free, V.dummy_res(free)))
                assert abs(o["q"]["wait"] - 1) < 1e-12
    assert n > 200


def test_coverage_rules_toy():
    st = {"sc": {"u1": {"spk": "A", "a": 1.0, "b": 1.5, "ref": "Yeah."},
                 "u2": {"spk": "B", "a": 0.5, "b": 4.0, "ref": "I think we should go with the yellow one."},
                 "u3": {"spk": "A", "a": 6.0, "b": 9.0, "ref": "Then the remote needs a big button."}},
          "us": [("u1", "A", 1.0, 1.5, False), ("u2", "B", 0.5, 4.0, False), ("u3", "A", 6.0, 9.0, False)],
          "mp": {1: "A", 2: "B"}, "unit_of": {"u1": "u1", "u2": "u2", "u3": "u3"}}
    x0 = {"spk": 2, "a": 1.1, "b": 1.4, "text": "Yeah."}                                    # u1 under B's slot
    x1 = {"spk": 2, "a": 0.6, "b": 3.9, "text": "I think we should choose the yellow one"}  # u2, right label
    x2 = {"spk": 2, "a": 6.1, "b": 8.8, "text": "Then the remote control needs a large button"}  # u3 mislabelled
    segs = [x0, x1, x2]
    st["got"] = {"u2": [(x1, x1["text"]), (x0, x0["text"])], "u3": []}
    cov = X.meeting_tables(st, segs)
    assert cov["u1"] == {"a": [0], "b": [], "c": []}                 # short: text rule under any label only
    assert cov["u2"]["b"] == [0, 1] and cov["u2"]["c"] == [1]        # x0 inside u2: credit yes, content guard no
    assert cov["u3"] == {"a": [], "b": [], "c": [2]}                 # long mislabelled: time + content
    assert X.word_overlap(["yeah"], ["i", "think"]) is False


def test_positions():
    pf, pr = X.positions(20, 3, 2)
    assert pf == [16, 17, 18, 19] and pr == [14, 15]
    pf, pr = X.positions(10, 0, 0)
    assert pf == [9] and pr == []


def test_rebuild_toy():
    tk = X.tok()
    raw0 = "<|spk1|><|t2|>Well,<|t6|>"
    raw1 = "<|spk2|><|t6|>I'm Abigail"
    raw2 = " Crawley.<|t30|>"
    h = {"id": "toy", "ctx": "", "raw": raw0 + raw1 + raw2,
         "read_log": [{"t": 1000, "new": raw0}, {"t": 2000, "new": raw1}, {"t": 2400, "new": raw2}]}
    from experiments.mist_w3po_v1.core.engine import Parser
    p = Parser(tk)
    blocks = []
    for rd in h["read_log"]:
        blocks += p.feed(tk(rd["new"], add_special_tokens=False)["input_ids"], rd["t"])
    p.close(2.4, 2400)
    h["segments"], h["blocks"] = p.segments, blocks
    tr = X.rebuild(h)
    assert tr["seg_state"] == ["out", "text", "out"] and [len(x) > 0 for x in tr["ids"]] == [True] * 3
    h2 = dict(h, segments=h["segments"][:1])
    try:
        X.rebuild(h2)
        raise RuntimeError("a wrong segment list must fail")
    except AssertionError:
        pass


TESTS = [test_nf_gen_reduces_to_v2p, test_nf_gen_overdue_at_read0, test_hand_oracle, test_positions,
         test_coverage_rules_toy, test_rebuild_toy, test_label_path_special_case, test_residual_brute_force_and_negatives]

if __name__ == "__main__":
    import time
    t0 = time.time()
    for f in TESTS:
        t = time.time()
        f()
        print(f"PASS {f.__name__} ({time.time() - t:.1f}s)", flush=True)
    print(f"ALL {len(TESTS)} TESTS PASSED ({time.time() - t0:.1f}s)")

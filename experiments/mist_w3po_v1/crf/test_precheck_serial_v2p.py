"""CPU tests of crf/precheck_serial_v2p.py (SerialCRF v2'): the newfirst order, the canonical plan unchanged from v2,
the newfirst forwards and their after-N boundary, the order-marginalised forward (hand example, both temperings),
S3' (b) brute force over (rho x orders) on synthetic and real windows, the negative controls, the label path (no
order choice, == S), the S1' classes and the trace-order check, and the analysis on dummy scores. pytest is not
installed:

    PYTHONPATH=.:third_party/Megatron-LM python -m experiments.mist_w3po_v1.crf.test_precheck_serial_v2p
"""
from __future__ import annotations

import math

import numpy as np

from experiments.mist_w3po_v1.crf import precheck_serial as S
from experiments.mist_w3po_v1.crf import precheck_serial_v2 as V
from experiments.mist_w3po_v1.crf import precheck_serial_v2p as X


def choice_windows(split: str, n: int, max_def: int = 10) -> list[dict]:
    out = []
    for rec in S.sample(split):
        lat = V.lattice(rec)
        if sum(1 for r in lat["rho"] if r < lat["R"] - 1) > max_def:
            continue
        if any(cx["nf"] for rd in X.plan_window_p(rec, lat)["reads"] for cx in rd["ctx"]):
            out.append(rec)
        if len(out) >= n:
            break
    return out


def test_newfirst_definition():
    n_choice = 0
    for rec in S.sample("dev")[:150] + S.sample("train")[:100]:
        lat = V.lattice(rec)
        rho = lat["rho"]
        for r in range(lat["R"]):
            for i in range(len(V.prev_states(lat, r))):
                hist, pend = V.context(rec, lat, r, i)
                for D in lat["states"][r]:
                    nf = X.newfirst(rec, rho, pend, D)
                    canon = V.kap(rec, rho, list(pend) + list(D))
                    if nf is None:
                        assert not pend or all(X.slot(rec, c) in {X.slot(rec, p) for p in pend} for c in D)
                        continue
                    n_choice += 1
                    N, rest = nf
                    seq = list(N) + list(rest)
                    assert sorted(seq) == sorted(canon) and seq != canon                # same items, a new order
                    assert not {X.slot(rec, c) for c in N} & {X.slot(rec, c) for c in pend}
                    assert list(rest) == V.kap(rec, rho, rest) and list(rest[: len(pend)]) == list(pend)
                    for k in {X.slot(rec, c) for c in seq}:                              # per-speaker label order kept
                        assert [c for c in seq if X.slot(rec, c) == k] == [c for c in canon if X.slot(rec, c) == k]
                    assert [list(w) for _, w in X.brute_orders(rec, rho, r, seq)] == [canon, seq]
    assert n_choice > 20


def test_canonical_plan_unchanged_from_v2():
    for rec in S.sample("dev")[:60]:
        lat = V.lattice(rec)
        a = V.plan_window(rec, lat, old_final=False)
        b = X.plan_window_p(rec, lat)
        assert b["reqs"][: len(a["reqs"])] == a["reqs"]
        assert all(q.get("kind") == "nf" for q in b["reqs"][len(a["reqs"]):])
        for ra, rb in zip(a["reads"], b["reads"]):
            for ca, cb in zip(ra["ctx"], rb["ctx"]):
                assert ca["cells"] == cb["cells"] and ca["pend"] == cb["pend"]


def test_label_path_no_choice():
    for split, n in (("dev", 300), ("train", 150)):
        recs = S.sample(split)[:n]
        V.check_label_or_die(recs)
        for rec in recs:
            lat = V.lattice(rec)
            for r in range(lat["R"]):
                hist, pend = V.context(rec, lat, r, 0)
                assert not pend and X.newfirst(rec, lat["rho"], pend, lat["states"][r][0]) is None


def test_nf_cells_and_after_N_boundary():
    for rec in choice_windows("dev", 4):
        lat = V.lattice(rec)
        plan = X.plan_window_p(rec, lat)
        reads = X.assemble_p(plan, V.dummy_res(plan))
        n = 0
        for o, rd in zip(reads, plan["reads"]):
            for cx in rd["ctx"]:
                h = S.ids_of(rec, cx["hist"])
                for j, D in enumerate(lat["states"][rd["r"]]):
                    nf = X.newfirst(rec, lat["rho"], cx["pend"], D)
                    assert (nf is None) == (o["psi_nf"] is None or o["psi_nf"][cx["i"]][j] is None)
                    if nf is None:
                        continue
                    n += 1
                    seq = S.ids_of(rec, list(nf[0]) + list(nf[1]))
                    assert abs(o["psi_nf"][cx["i"]][j] - V.dummy_psi(rd["r"], rd["final"], h, seq)) < 1e-12
                    tl, ec = V.dummy_scores(rd["r"], rd["final"], h, S.ids_of(rec, nf[0]))
                    c = next(c for c in o["nf_ctx"] if c["i"] == cx["i"] and tuple(c["N"]) == tuple(nf[0]))
                    assert abs(c["e_lp"] - ec[-1]) < 1e-12 and c["m"] == cx["pend"][0] and j in c["js"]
            if rd["ctx"]:
                q, b = rd["ctx"][0]["cells"]["new"][0]
                lab = V.dummy_scores(rd["r"], rd["final"], S.ids_of(rec, rd["ctx"][0]["hist"]),
                                     S.ids_of(rec, lat["states"][rd["r"]][0]))[1][-1]
                assert abs(o["label_e_lp"] - lab) < 1e-12
        assert n > 0


def test_forward_hand():
    reads = [{"psi": np.log([[0.6, 0.2]]).tolist(), "psi_nf": None, "final": False},
             {"psi": np.log([[0.5], [0.3]]).tolist(), "psi_nf": [[None], [math.log(0.4)]], "final": True}]
    for tau in (0.5, 1.0):
        w = np.array([(0.6 * 0.5) ** tau, (0.2 * 0.3) ** tau, (0.2 * 0.4) ** tau])
        fb = X.forward_p(reads, tau, "joint")
        assert abs(fb["logZ"] - math.log(w.sum())) < 1e-12
        assert fb["order"][0] is None
        assert abs(fb["order"][1]["q_nf"] - w[2] / w.sum()) < 1e-12
        assert abs(fb["order"][1]["q_real"] - (w[1] + w[2]) / w.sum()) < 1e-12
        assert np.allclose(fb["marg"][0], [w[0] / w.sum(), (w[1] + w[2]) / w.sum()])
        z = 0.3 ** tau + 0.14 ** tau
        fl = X.forward_p(reads, tau, "lse_temper")
        assert abs(fl["logZ"] - math.log(z)) < 1e-12
        assert abs(fl["order"][1]["q_real"] - 0.14 ** tau / z) < 1e-12
        assert abs(fl["order"][1]["q_nf"] - 0.14 ** tau / z * 0.4 / 0.7) < 1e-12
    assert abs(X.n_paths_lattice(reads) - 3) < 1e-9


def test_S3b_synthetic_bruteforce():
    rng = np.random.default_rng(11)
    n_choice = 0
    for _ in range(200):
        rec = V.synthetic_rec(rng, int(rng.integers(1, 6)), int(rng.integers(1, 9)), int(rng.integers(1, 4)))
        c = X.check_brute_p(rec)
        if c["skipped"]:
            continue
        assert c["stats"]["pass"] == 1, (rec["id"], c)
        n_choice += c["real_choice_reads"] > 0
    assert n_choice > 20


def test_S3b_real_windows_bruteforce():
    recs = choice_windows("dev", 5) + choice_windows("train", 3)
    assert len(recs) == 8
    for rec in recs:
        c = X.check_brute_p(rec)
        assert not c["skipped"] and c["stats"]["pass"] == 1 and c["max_abs_diff"] < 1e-9, (rec["id"], c)
        assert c["real_choice_reads"] > 0


def test_negative_controls():
    neg = X.negative_controls()
    assert neg["all_caught"] == 1 and neg["windows"] >= 6, neg


def test_s1p_class_and_trace_order():
    assert X.s1p_class(False, False, False, False, False, 0) == "NOT_DEFERRAL"
    assert X.s1p_class(True, True, True, False, False, 2) == "U_WRITTEN_NEXT_SAME_LABEL"
    assert X.s1p_class(True, True, False, True, True, 2) == "U_WRITTEN_NEXT_OTHER_LABEL"
    assert X.s1p_class(True, False, False, True, False, 1) == "U_WRITTEN_LATER_OTHER_LABEL"
    assert X.s1p_class(True, False, False, True, True, 0) == "U_WRITTEN_LATER_SAME_LABEL"
    assert X.s1p_class(True, False, False, False, False, 0) == "NOTHING_ELSE_WRITTEN_AT_NEXT"
    assert X.s1p_class(True, False, False, False, False, 1) == "CHAIN"
    H = {"a": {"raw": "xy", "read_log": [{"t": 1, "new": "x"}, {"t": 2, "new": "y"}],
               "segments": [{"write_ms": 1}, {"write_ms": 2}]},
         "b": {"raw": "xy", "read_log": [{"t": 1, "new": "y"}], "segments": [{"write_ms": 2}, {"write_ms": 1}]}}
    st = X.trace_order_check(H)
    assert st == {"windows": 2, "raw_eq_concat_reads": 1, "segments_in_emission_order": 1}
    rows = [{"cls": "ORDER_OMISSION", "le05": True, "deferral": True, "why2": "CHAIN", "chain": True, "next_final": False,
             "u_written_in_other_window": False, "n_other_at_next": 2},
            {"cls": "MISLABEL", "le05": False, "deferral": True, "why2": "U_WRITTEN_LATER_OTHER_LABEL", "chain": False},
            {"cls": "MISLABEL", "le05": False, "deferral": False, "why2": "NOT_DEFERRAL", "chain": False}]
    sm = X.s1p_summary(rows)["all_ov50_le1_missed"]
    assert sm["n"] == 3 and sm["chain"] == 1 and abs(sm["share"] - 1 / 3) < 1e-12 and sm["deferral"] == 2


def test_analysis_on_dummy_scores():
    per = []
    for rec in choice_windows("dev", 6):
        plan = X.plan_window_p(rec)
        row = {"id": rec["id"], "R": plan["lat"]["R"], "n_fw": len(plan["reqs"]),
               "states": [[list(s) for s in st] for st in plan["lat"]["states"]],
               "reads": X.assemble_p(plan, V.dummy_res(plan))}
        w = X.window_analysis(rec, row)
        assert w["orders"] and w["ctxs"]
        for o in w["orders"]:
            for k, v in o.items():
                if k.startswith("q_nf"):
                    assert -1e-12 <= v <= o[k.replace("q_nf", "q_real")] + 1e-12 <= 1 + 1e-12
        per.append(w)
    rep = X.split_report(per)
    s = rep[f"S2a[joint,tau={X.TAU}]"]
    assert s["n"] == rep["real_choice_reads"] > 0 and 0 <= s["mean"] <= 1
    assert rep["S2b"]["all"]["contexts"] == rep["nf_contexts"] > 0


if __name__ == "__main__":
    import time

    tests = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    t0 = time.time()
    for name, fn in tests:
        t1 = time.time()
        fn()
        print(f"PASS {name} ({time.time() - t1:.1f}s)", flush=True)
    print(f"ALL {len(tests)} TESTS PASSED ({time.time() - t0:.1f}s)")

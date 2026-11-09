"""CPU tests of crf/diag_recovery.py (openable items, the D3 pairs and off-label histories, the pre-registered decision
functions, the content-free D2 / D3 quantities, D1 counts); the scoring path is crf/precheck_serial.score_reqs (tested
with a fake engine in test_precheck_serial) and is checked again by the fresh-vs-stored control in the smoke.

    PYTHONPATH=.:third_party/Megatron-LM python -m experiments.mist_w3po_v1.crf.test_diag_recovery
"""
from __future__ import annotations

import math

import numpy as np

from experiments.mist_w3po_v1.crf import diag_recovery as D
from experiments.mist_w3po_v1.crf import precheck_serial as S
from experiments.mist_w3po_v1.crf.test_precheck_serial import fake_rec


def test_module_references_resolve():
    import re
    from pathlib import Path

    src = Path(D.__file__).read_text()
    for name in set(re.findall(r"\bS\.([A-Za-z_]\w*)", src)):
        assert hasattr(S, name), name
    for name in set(re.findall(r"\bS\.P\.([A-Za-z_]\w*)", src)):
        assert hasattr(S.P, name), name


def test_openable():
    rec = fake_rec([0, 1, 1, 2])
    assert D.openable(rec, [0, 1, 2, 3]) == [0, 1, 3]
    assert D.openable(rec, [2, 3]) == [2, 3] and D.openable(rec, [1, 2]) == [1]


def test_pairs_on_dev_sample():
    n = 0
    for rec in S.sample("dev"):
        rho = S.chunk_reads(rec)
        beta = [0] + list(rec["beta_heur"])
        B = S.read_items(rec, rho)
        for p in D.d3_pairs(rec):
            n += 1
            r, f, m = p["r"], p["f"], p["m"]
            assert r < rec["n_read"] - 1 and f in B[r] and m in B[r] and f != m
            assert rec["chunks"][f]["utt"] != rec["chunks"][m]["utt"]
            assert 2 <= len(B[r]) <= S.CAP and S.n_orders(rec, B[r]) >= 2
            assert p["hist_on"] == rec["S_ids"][: beta[r]]                       # on-path history = S prefix
            assert p["hist_off"] == p["hist_on"] + S.ids_of(rec, [f])             # f committed at r goes last
            miss = {rec["chunks"][c]["utt"] for c in B[r] if c != f}
            assert all(rec["chunks"][c]["utt"] not in miss for c in p["B1"])
            assert p["B1"] == [c for c in B[r + 1] if c in p["B1"]]                # label order kept
            op = D.opener_ids(rec, m)
            assert op == rec["S_ids"][rec["chunks"][m]["tok0"]: rec["chunks"][m]["tok0"] + 2]
            reqs = D.pair_reqs(rec, p)
            assert reqs[0][1] == r and reqs[1][1] == r + 1 and reqs[1][3] == op
            assert len(reqs) == (3 if p["B1"] else 2)
    assert n > 50


def test_decisions():
    assert D.h_rec([0.1, 0.2, 0.25], [0.8, 0.8, 0.8])["verdict"] == "SUPPORTED"
    assert D.h_rec([0.1, 0.2, 0.25], [0.4, 0.45, 0.5])["verdict"] == "AMBIGUOUS"     # gap 0.25 < 0.3
    assert D.h_rec([0.6, 0.7, 0.9], [0.9, 0.9, 0.9])["verdict"] == "REJECTED"
    assert D.h_rec([0.3, 0.4, 0.5], [0.9, 0.9, 0.9])["verdict"] == "AMBIGUOUS"
    assert D.h_rec([], [])["verdict"] == "NO_DATA"
    assert D.h_final(0.6, [0.5, 0.6], [0.1, 0.2])["verdict"] == "SUPPORTED"
    assert D.h_final(0.4, [0.5, 0.6], [0.1, 0.2])["verdict"] == "NOT_SUPPORTED"
    assert D.h_final(0.6, [0.1], [0.5])["verdict"] == "NOT_SUPPORTED"


def synthetic_row():
    lo = math.log
    # label [0, 1]; chunk 0 short ov50 (slot 1, 4 tokens), chunk 1 (slot 2, 5 tokens)
    row = {"id": "x", "r": 1, "final": False, "label": [0, 1], "capped": False, "n_pi": 2,
           "items": [{"chunk": 0, "slot": 1, "n_tok": 4, "dur_s": 0.5, "ov50": True},
                     {"chunk": 1, "slot": 2, "n_tok": 5, "dur_s": 3.0, "ov50": True}],
           "orders": [{"pi": [0, 1], "tok_lp": [lo(0.4), lo(0.5), -1, -1, lo(0.6), lo(0.5), -1, -1, -1],
                       "end_at_bounds": [lo(0.01), lo(0.2), lo(0.8)]},
                      {"pi": [1, 0], "tok_lp": [lo(0.5), lo(0.8), -1, -1, -1, lo(0.3), lo(0.5), -1, -1],
                       "end_at_bounds": [lo(0.01), lo(0.6), lo(0.7)]}]}
    return row


def test_d2_read():
    x = D.d2_read(synthetic_row())
    assert x["f_star"] == 1 and not x["f_star_is_label"]                        # 0.5 * 0.8 > 0.4 * 0.5
    assert abs(x["after_fstar"]["p_open"] - 0.15) < 1e-12 and abs(x["after_fstar"]["p_e"] - 0.6) < 1e-12
    assert x["after_fstar"]["e_gt_open"] and x["after_fstar"]["rem_short_ov"]
    assert abs(x["after_label"]["p_open"] - 0.3) < 1e-12 and abs(x["after_label"]["p_e"] - 0.2) < 1e-12
    assert not x["after_label"]["e_gt_open"] and not x["after_label"]["rem_short_ov"]
    ctrl = D.stored_control([synthetic_row()])
    assert abs(math.exp(ctrl[("x", 1, 0, 1)]) - 0.3) < 1e-12 and abs(math.exp(ctrl[("x", 1, 1, 0)]) - 0.15) < 1e-12
    cs = D.stored_control([synthetic_row()], spk_only=True)
    assert abs(math.exp(cs[("x", 1, 0, 1)]) - 0.6) < 1e-12 and abs(math.exp(cs[("x", 1, 1, 0)]) - 0.3) < 1e-12
    s = D.d2_summary([x])
    assert s["pi2|nonfinal|f=f*|all"]["n"] == 1 and s["pi2|nonfinal|f=f*|all"]["share_p_e_gt_p_open"] == 1.0


def test_pair_result():
    rec = fake_rec([0, 1, 2])                                                    # 4 tokens per chunk
    p = {"id": "fake", "r": 0, "f": 0, "m": 1, "label_first": 0, "n_items": 2, "hist_on": [], "hist_off": [9],
         "B1": [2], "B1_dropped": 0, "r1_final": True, "m_dur_s": 0.5, "m_ov50": True, "m_short_ov": True}
    tl0 = np.array([-1, -1, -1, -1, -0.2, -0.3])
    tl1 = np.array([-0.5, -0.1])
    tl2 = np.array([-1, -1, -1, -1, -2.0, -0.4])
    res = [(tl0, np.zeros(7)), (tl1, np.array([-3.0, 0, 0])), (tl2, np.array([0, 0, 0, 0, -0.7, 0, 0]))]
    out = D.pair_result(rec, p, res)
    assert abs(out["control_fresh_lp"] + 0.5) < 1e-12 and abs(out["R_first_lp"] + 0.6) < 1e-12
    assert out["e_first_lp"] == -3.0 and abs(out["R_end_lp"] + 2.4) < 1e-12 and out["e_end_lp"] == -0.7
    assert out["control_fresh_spk_lp"] == -0.2 and out["R_first_spk_lp"] == -0.5 and out["R_end_spk_lp"] == -2.0
    out2 = D.pair_result(rec, {**p, "B1": []}, res[:2])
    assert out2["R_end_lp"] == out2["R_first_lp"] and out2["e_end_lp"] == out2["e_first_lp"]
    pairs = [{**out, "f_is_fstar": True}, {**out2, "r": 1, "f_is_fstar": False}]
    ctrl = {("fake", 0, 0, 1): -0.5, ("fake", 1, 0, 1): -0.5}
    sm = D.d3_summary(pairs, ctrl, {k: -0.2 for k in ctrl})
    assert sm["all"]["pairs"] == 2 and sm["f_is_fstar"]["pairs"] == 1
    assert sm["all"]["H_rec"]["n"] == 2 and sm["all"]["H_rec_spk_only(secondary)"]["n"] == 2


def test_d1_counts():
    rep, units = D.d1()
    oo = rep["ORDER_OMISSION_ov50_le1"]
    assert oo["n"] == 151 and oo["online_with_window"] == 151 and oo["ref_found"] == 151
    assert rep["all_ov50_le1_misses"]["n"] == 295
    assert rep["D1_primary"] == oo["due_final_online"] / 151


if __name__ == "__main__":
    import time

    tests = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    t0 = time.time()
    for name, fn in tests:
        t1 = time.time()
        fn()
        print(f"PASS {name} ({time.time() - t1:.1f}s)", flush=True)
    print(f"ALL {len(tests)} TESTS PASSED ({time.time() - t0:.1f}s)")

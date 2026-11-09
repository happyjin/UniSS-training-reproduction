"""CPU tests of crf/rescore_openers.py (item construction, forced sequences, class collapse, the decision rule) on the
real precheck sample of sched_v1; the GPU scoring path is checked by a --limit smoke run.

    PYTHONPATH=.:third_party/Megatron-LM python -m experiments.mist_w3po_v1.crf.test_rescore_openers
"""
from __future__ import annotations

import numpy as np

from experiments.mist_w3po_v1.crf import diag_short_units as D
from experiments.mist_w3po_v1.crf import precheck_sched as P
from experiments.mist_w3po_v1.crf import rescore_openers as O


def test_items_and_forced():
    recs = P.load_records(P.SCHED, 500, 20261109)
    items = O.unit_items(recs, 1.0)
    byid = {r["id"]: r for r in recs}
    n_units = sum(1 for r in recs for u in r["units"] if (u["b_ms"] - u["a_ms"]) / 1000 <= 1.0)
    assert len({(it["id"], it["utt"]) for it in items}) == n_units == 393
    for it in items:
        rec = byid[it["id"]]
        S = rec["S_ids"]
        g = O.forced_G(rec, it)
        assert g[0] == O.spk_id(it["slot"]) and O.NEW0 <= g[1] < O.SPK0 and O.NEW0 <= g[-1] < O.SPK0
        assert all(rec["unit"][j] == it["utt"] for j in range(it["j0"], it["tok1"]))
        assert it["j0"] == min(j for j in range(rec["n_tok"]) if rec["unit"][j] == it["utt"])
        rh = D.heur_reads(rec)
        assert ("final" in it["reads"]) == (it["r"] == rec["n_read"] - 1)
        assert ("heur" in it["reads"]) == (it["r"] == rh[it["j0"]])
        if it["prev_slot"] is not None:
            assert S[[c["tok0"] for c in rec["chunks"] if c["tok1"] == it["j0"]][0]] == O.spk_id(it["prev_slot"])
        for k2 in range(1, 9):
            if k2 == it["slot"]:
                continue
            a = O.forced_A(rec, it, k2)
            assert a[0] == O.spk_id(k2) and a[1:] == g[1:]
            c = O.pending_chunk(rec, it["j0"], k2)
            if c is not None:
                b = O.forced_B(rec, c)
                assert b[0] == O.spk_id(k2) and c["tok0"] > it["j0"]
                assert all(not (x["slot"] == k2 and it["j0"] < x["tok0"] < c["tok0"]) for x in rec["chunks"])
    # each unit has one item per distinct read (final, heur)
    per = {}
    for it in items:
        per.setdefault((it["id"], it["utt"]), []).extend(it["reads"])
    assert all(sorted(v) == ["final", "heur"] for v in per.values())


def test_collapse_and_classes():
    rng = np.random.default_rng(0)
    x = rng.normal(size=151936).astype(np.float32)
    lp = x - np.log(np.exp(x.astype(np.float64)).sum())
    m = O.collapse(lp)
    assert abs(sum(m.values()) - 1.0) < 1e-6 and set(m) == set(O.CLASSES)
    assert O.argmax_class(O.spk_id(3)) == "spk3" and O.argmax_class(O.WAIT_ID) == "wait"
    assert O.argmax_class(O.EOS_ID) == "eos" and O.argmax_class(O.NEW0 + 5) == "time" and O.argmax_class(100) == "text"


def test_rule():
    base = {"slot": 2}
    assert O.classify({**base, "argmax_class": "spk2"}) == "GOLD"
    assert O.classify({**base, "argmax_class": "spk1", "lpA": -1.0, "lpB": None}) == "ATTRIBUTION_NOPENDING"
    assert O.classify({**base, "argmax_class": "spk1", "lpA": -1.0, "lpB": -1.5}) == "ATTRIBUTION"
    assert O.classify({**base, "argmax_class": "spk1", "lpA": -1.5, "lpB": -1.0}) == "ORDER"
    assert O.classify({**base, "argmax_class": "spk1", "lpA": -1.2, "lpB": -1.0}) == "AMBIGUOUS"
    assert O.classify({**base, "argmax_class": "time"}) == "OTHER_CONT"
    assert O.classify({**base, "argmax_class": "text"}) == "OTHER_CONT"
    assert O.classify({**base, "argmax_class": "eos"}) == "END_WAIT"


if __name__ == "__main__":
    import time

    tests = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    t0 = time.time()
    for name, fn in tests:
        t1 = time.time()
        fn()
        print(f"PASS {name} ({time.time() - t1:.1f}s)", flush=True)
    print(f"ALL {len(tests)} TESTS PASSED ({time.time() - t0:.1f}s)")

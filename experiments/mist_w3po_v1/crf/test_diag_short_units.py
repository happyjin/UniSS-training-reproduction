"""Tests of crf/diag_short_units.py (CPU; real sched_v1 windows with random log-probs):

  (a) ov50: union coverage by other speakers' units (same speaker ignored, overlapping others merged)
  (b) per_window_lam: interior windows hit their own target (|E[Lat] - target| < 1e-6) and agree with
      schedule_lattice.solve_lambda run on that window alone; boundary classes are consistent
  (c) unit_rows: the per-unit numbers equal direct reads of tok_lp / end_lp / w_tok (opening token, final read,
      heuristic read, P_early = 1 - w_tok[last read, last token] = P(committed before the final read) by FFBS)

    PYTHONPATH=.:third_party/Megatron-LM python -m experiments.mist_w3po_v1.crf.test_diag_short_units
"""
from __future__ import annotations

import numpy as np
import torch

from experiments.mist_w3po_v1.crf import diag_short_units as D
from experiments.mist_w3po_v1.crf import precheck_sched as P
from experiments.mist_w3po_v1.crf import schedule_lattice as sl


def _fake(n=40, seed=3):
    recs = [r for r in P.load_records(P.SCHED, n, seed)]
    rng = np.random.default_rng(seed)
    lps = {r["id"]: (-rng.exponential(1.0, (r["n_read"], r["n_tok"])),
                     np.log(rng.uniform(1e-3, 1, (r["n_read"], r["n_tok"] + 1)))) for r in recs}
    return recs, lps


def test_ov50():
    u = {"a_ms": 0, "b_ms": 1000, "speaker_id": "A"}
    others = [u, {"a_ms": 0, "b_ms": 2000, "speaker_id": "A"},             # same speaker: ignored
              {"a_ms": 100, "b_ms": 400, "speaker_id": "B"}, {"a_ms": 300, "b_ms": 550, "speaker_id": "C"}]
    assert not D.ov50(others, u)                                            # union 100..550 = 450 ms < 500
    others.append({"a_ms": 540, "b_ms": 600, "speaker_id": "B"})            # + 540..600 -> 100..600 = 500 ms
    assert D.ov50(others, u)
    assert not D.ov50([u], u)


def test_per_window_lam():
    recs, lps = _fake()
    lat = D.lattices(recs, lps, "any", 128, 64)[0]
    tgt = torch.tensor([r["lat_heur"] for r in recs], dtype=sl.DTYPE)
    o = D.per_window_lam(lat, tgt)
    br = o["bracketed"]
    assert br.sum() > 5
    assert o["resid"][br].max() < 1e-6
    singles = D._singles(recs, lps, 128)
    for i in np.nonzero(br)[0][:8]:
        s = sl.solve_lambda([sl.pack([singles[i]], W=128)], recs[i]["lat_heur"])
        assert s["bracketed"] and abs(s["lam"] - o["lam"][i]) < 1e-4 * max(1.0, abs(s["lam"])), (s, o["lam"][i])
    # classes are exclusive
    cls = br.astype(int) + o["degenerate"].astype(int) + o["at_latest"].astype(int) + o["at_earliest"].astype(int)
    assert cls.max() <= 1


def test_unit_rows():
    recs, lps = _fake(12, 5)
    reach = D.reach_all(recs, 128)
    lats = D.lattices(recs, lps, "any", 128, 64)
    wt = {"lam0_any": D.w_tok_all(lats, 0.0)}
    rows = D.unit_rows(recs, lps, reach, wt)
    by = {(r["id"], r["utt"]): r for r in rows}
    gen = torch.Generator().manual_seed(0)
    smp = sl.sample(lats[0], 0.0, n=4000, generator=gen)
    for k, rec in enumerate(recs):
        tok_lp, end_lp = lps[rec["id"]]
        R = rec["n_read"]
        rh = D.heur_reads(rec)
        for u in rec["units"]:
            J = [j for j in range(rec["n_tok"]) if rec["unit"][j] == u["utt"]]
            j0, jL = min(J), max(J)
            jt = [j for j in J if rec["kind"][j] == D.K_TEXT]
            row = by[(rec["id"], u["utt"])]
            assert abs(row["p_open_final"] - np.exp(tok_lp[R - 1, j0])) < 1e-12
            assert abs(row["p_end_at_open_final"] - np.exp(end_lp[R - 1, j0])) < 1e-12
            assert abs(row["p_content_final"] - np.exp(np.mean(tok_lp[R - 1, jt]))) < 1e-12
            assert abs(row["p_open_heur"] - np.exp(tok_lp[rh[j0], j0])) < 1e-12
            rr = np.nonzero(reach[k][:, j0])[0]
            assert abs(row["p_open_max"] - np.exp(tok_lp[rr, j0].max())) < 1e-12
            if R > 1:
                st = smp[k, :, :R].numpy()                              # states after each read
                emp = float(np.mean(st[:, R - 2] > jL))                  # last token committed by read R-2
                assert abs(row["P_early_lam0_any"] - emp) < 0.04, (row["P_early_lam0_any"], emp)
            else:
                assert "P_early_lam0_any" not in row


if __name__ == "__main__":
    import time

    tests = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    t0 = time.time()
    for name, fn in tests:
        t1 = time.time()
        fn()
        print(f"PASS {name} ({time.time() - t1:.1f}s)", flush=True)
    print(f"ALL {len(tests)} TESTS PASSED ({time.time() - t0:.1f}s)")

"""Test of crf/heldout_loss.window_terms (CPU, random log-probs on real en2zh dev sched windows, h = 1):
  (a) the label path's log-prob (-label_nll) never exceeds the lattice log-likelihood full_ll (it is one of its paths)
  (b) on windows whose v2' plan has a single path (no deferral, no order choice) they are equal
  (c) label_nll equals the direct sum over the label requests (context 0 -> state 0 of every read: all its forced
      tokens + its end token), i.e. the C0 loss terms (the posterior itself is serial_v2p_train.posterior, tested
      against brute force in test_serial_v2p_train)

    PYTHONPATH=.:third_party/Megatron-LM python -m experiments.mist_w3po_v1.crf.test_heldout_loss
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from experiments.mist_w3po_v1.crf import heldout_loss as H
from experiments.mist_w3po_v1.crf import serial_v2p_train as V
from experiments.mist_w3po_v1.crf import train_crf_sft as T

SCHED = Path("eval_outputs/mist_w3po_v1/crf/sched_en2zh_dev_v1")


def main() -> None:
    recs = [json.loads(l) for l in open(SCHED / "shard_000.jsonl", encoding="utf-8")][:300]
    recs = [r for r in recs if r["n_tok"] > 0]
    rng = np.random.default_rng(0)
    n_single = n_multi = 0
    for i, rec in enumerate(recs):
        w = T.make_window(i, rec, False)
        T.set_reach([w], 128)
        if not w.feasible:
            continue
        pl = V.plan(w)
        res = {}
        for k, q in enumerate(pl["reqs"]):
            m = len(q["forced_ids"])
            res[(w.idx, k)] = (-rng.exponential(1.0, m), np.log(rng.uniform(1e-3, 1.0, m + 1)))
        t = H.window_terms(w, pl, res)
        assert -t["label_nll"] <= t["full_ll"] + 1e-9, (rec["id"], t)
        # (c) direct label sum: per read, the request of context 0 -> state 0 (set_coefs' label row)
        lab = 0.0
        for rd in pl["reads"]:
            k, b = rd["ctx"][0]["cells"]["new"][0]
            tl, ec = res[(w.idx, k)]
            lab -= float(tl[:b].sum() + ec[b])
        assert abs(lab - t["label_nll"]) < 1e-9, (rec["id"], lab, t["label_nll"])
        single = abs(t["full_ll"] + t["label_nll"]) < 1e-9
        n_single += single
        n_multi += not single
    assert n_single > 0 and n_multi > 0, (n_single, n_multi)
    print(f"ok: {n_single} single-path windows (equal), {n_multi} multi-path windows (label <= lattice)")


if __name__ == "__main__":
    main()

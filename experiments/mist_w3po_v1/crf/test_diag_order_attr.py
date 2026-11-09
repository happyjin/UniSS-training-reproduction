"""CPU tests of crf/diag_order_attr.py (text match, due read, prompt-P evidence, cause partition) on toy inputs.

    PYTHONPATH=.:third_party/Megatron-LM python -m experiments.mist_w3po_v1.crf.test_diag_order_attr
"""
from __future__ import annotations

import numpy as np

from experiments.mist_w3po_v1.crf import diag_order_attr as X


def test_text_match():
    assert X.norm("Yeah. Mm-hmm, OK!") == ["yeah", "mm", "hmm", "ok"]
    r = X.norm("Yeah.")
    assert X.edge_match(X.norm("Yeah."), r)
    assert X.edge_match(X.norm("Yeah, so we need the remote."), r)
    assert X.edge_match(X.norm("we need the remote, yeah"), r)
    assert not X.edge_match(X.norm("so yeah we need it"), r)
    assert X.contains(X.norm("so yeah we need it"), r)
    assert not X.edge_match([], r) and not X.edge_match(["a"], [])


def test_due_read():
    reads = [1000, 2000, 3000, 3400]
    # unit ends at 0.5 s of the window -> commit at 1.5 s -> read 2000
    assert X.due_read(reads, 10.0, 10.5) == 1
    assert X.due_read(reads, 10.0, 11.0) == 1          # exactly 2.0 s
    assert X.due_read(reads, 10.0, 12.9) == 3          # past the end -> the last read
    assert X.due_read([700], 0.0, 0.2) == 0


def test_evidence_and_cause():
    P = np.zeros((25, 8), np.float32)                  # 2.0 s window
    P[5:10, 2] = 0.9                                   # unit 0.4-0.8 s on column 2 (slot 3)
    e = X.evidence(P, 400, 800, 2000, k=1, k2=3)      # read at 2.0 s: frames >= 12 zeroed, unit visible
    assert e["ev"] == "K2_COL" and e["vis_frac"] == 1.0 and e["p_k2"] > 0.8 and e["p_gold"] == 0.0
    e = X.evidence(P, 400, 800, 1200, k=1, k2=3)      # read at 1.2 s: frames >= 2 zeroed -> unseen
    assert e["ev"] == "NONE" and e["ev_sub"] == "UNSEEN" and e["vis_frac"] == 0.0 and e["ev_full"] == "K2_COL"
    e = X.evidence(P, 1200, 1600, 2000, k=1, k2=3)    # silent frames
    assert e["ev"] == "NONE" and e["ev_sub"] == "SILENT"
    e = X.evidence(P, 400, 800, 2000, k=3, k2=1)
    assert e["ev"] == "GOLD_COL"
    assert X.cause_of("K2_COL", "WIN_K2") == "LABEL_NOISE"
    assert X.cause_of("K2_COL", "WIN_AGREE") == "WRONG_EVIDENCE"
    assert X.cause_of("NONE", "WIN_K2") == "NO_EVIDENCE"
    assert X.cause_of("GOLD_COL", "WIN_AGREE") == "IGNORES_EVIDENCE"


def test_window_map():
    P = np.zeros((25, 8), np.float32)
    P[0:10, 0] = 0.9                                   # speaker A on column 0
    P[12:20, 4] = 0.9                                  # speaker B on column 4
    win = {"utts": [{"speaker_id": "A", "begin_s": 0.0, "end_s": 0.8}, {"speaker_id": "B", "begin_s": 0.96, "end_s": 1.6}]}
    sess = {"A": 1, "B": 2}
    m = X.window_map(win, P, sess, "B")
    assert m["win_slot"] == 5 and m["covered"] and m["n_spk_win"] == 2
    m = X.window_map(win, P, sess, "A")
    assert m["win_slot"] == 1


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"{name} ok", flush=True)
    print("ALL OK")

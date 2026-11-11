"""Tests of crf/build_piece_sched.py (CPU): on real T1/T2 plan windows (the T0 train targets) with random per-speaker
monotone labels, S_prop labels, and a single-read label:
  (a) the record: S = the pieces sorted by (e_read, src_end, src_start); tok spans; chunk_reads == e_read;
      beta_heur; kap == S order == precheck_serial.kappa; the label path's v2' rows == C0's rows (check_record)
  (b) the record loads in the trainer (make_window), the v2' lattice exists, the label path is a lattice path
  (c) a non-monotone label within a speaker is caught by check_record

    PYTHONPATH=.:third_party/Megatron-LM CUDA_VISIBLE_DEVICES= python -m experiments.mist_w3po_v1.crf.test_build_piece_sched
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np

from experiments.mist_w3po_v1.crf import build_piece_sched as BP
from experiments.mist_w3po_v1.crf import t0_agree_labels as T0

TARGETS = "eval_outputs/mist_w3po_v1/crf/t0_agree_train/cpu/targets.jsonl"


def windows(n: int = 60):
    src = Path(T0.SRC_TRAIN)
    tgs = [json.loads(l) for l in itertools.islice(open(TARGETS, encoding="utf-8"), n)]
    ids = {t["id"] for t in tgs}
    W = {}
    for l in open(src / "windows_train_ref.jsonl", encoding="utf-8"):
        w = json.loads(l)
        if w["id"] in ids:
            W[w["id"]] = w
    return [(W[t["id"]], t) for t in tgs]


def rand_labels(tg: dict, rng) -> list[int]:
    R = len(tg["t_ms"])
    raw = rng.integers(0, R, len(tg["pieces"]))
    from experiments.mist_w3po_v1.crf import precheck_b2_earliest_write as B

    return B.cummax_by_speaker(raw, [p["slot"] for p in tg["pieces"]])


def test_records():
    from experiments.mist_w3po_v1.crf import precheck_serial as S
    from experiments.mist_w3po_v1.crf import precheck_serial_v2 as V
    from experiments.mist_w3po_v1.crf import train_crf_sft as T
    from experiments.mist_w3po_v1.crf.build_sched_windows import tokenizer

    tok = tokenizer()
    rng = np.random.default_rng(0)
    n_multi = 0
    for w, tg in windows():
        u = T0.prop_reads(tg)
        R = len(tg["t_ms"])
        for e in (rand_labels(tg, rng), u, [R - 1] * len(u)):
            J = len(e)
            extra = {"t0_e_raw": [None] * J, "t0_zero": [None] * J, "label_src": ["test"] * J}
            rec = BP.sched_record(w, tg, e, u, extra, tok)
            ch = rec["chunks"]
            keys = [(c["e_read"], c["src_end_ms"], c["src_start_ms"], c["utt"], c["k"]) for c in ch]
            assert keys == sorted(keys)
            assert sorted(c["piece"] for c in ch) == sorted(p["piece"] for p in tg["pieces"])
            assert rec["S"] == "".join(c["piece"] for c in ch)
            for c in ch:
                assert tok(c["piece"], add_special_tokens=False)["input_ids"] == rec["S_ids"][c["tok0"]: c["tok1"]]
                assert c["commit_ms"] == c["src_end_ms"] + 1000
            assert S.chunk_reads(rec) == [c["e_read"] for c in ch]
            assert rec["beta_heur"][-1] == rec["n_tok"] and len(rec["beta_heur"]) == R
            c0 = BP.check_record(rec)
            assert c0["ok"], rec["id"]
            win = T.make_window(0, rec, True)
            assert win.R == R and win.N == rec["n_tok"]
            lat = V.lattice(rec)
            assert lat["rho"] == [c["e_read"] for c in ch]
            assert all(len(s) >= 1 for s in lat["states"])
            n_multi += len({c["slot"] for c in ch}) > 1
    assert n_multi > 10


def test_non_monotone_caught():
    from experiments.mist_w3po_v1.crf.build_sched_windows import tokenizer

    tok = tokenizer()
    for w, tg in windows(200):
        R = len(tg["t_ms"])
        slots = [p["slot"] for p in tg["pieces"]]
        js = [j for j in range(len(slots)) if slots.count(slots[j]) >= 2]
        if R < 3 or not js:
            continue
        s = slots[js[0]]
        same = [j for j in range(len(slots)) if slots[j] == s]
        e = [0] * len(slots)
        e[same[0]] = R - 1                     # the first piece of the speaker after its second piece
        u = T0.prop_reads(tg)
        extra = {"t0_e_raw": [None] * len(e), "t0_zero": [None] * len(e), "label_src": ["test"] * len(e)}
        rec = BP.sched_record(w, tg, e, u, extra, tok)
        assert not BP.check_record(rec)["ok"]
        return
    raise AssertionError("no window to test")


def main() -> None:
    tests = [test_records, test_non_monotone_caught]
    for t in tests:
        t()
        print("PASS", t.__name__, flush=True)
    print(f"ALL {len(tests)} PASSED")


if __name__ == "__main__":
    main()

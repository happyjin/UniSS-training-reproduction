"""Tests of crf/t0_agree_labels.py (CPU only):

  (a) earliest_stable: the first read from which a_j stays >= a_j(R); a non-monotone curve; a_j(R) = 0 -> u_j
      (literal: the first read); R = 1
  (b) agree_counts over the text spans; text_spans on real dev5 targets (header tokens excluded, structure asserted)
  (c) make_item: dev = B-2's items; train final read = posteriors cut at t // 80 with full=True, non-final unchanged
  (d) train targets: B-2's builder on real windows_train_ref windows reproduces sched_v1's punctuation chunks; pieces
      <= 4 words in kappa order; prop_reads = S_prop's reads
  (e) plan_windows = the windows train_crf_sft.plan_steps touches with the 0.5 s read counts
  (f) t0_events = B-2's S_lik path (running max per speaker, online clamps) on a toy window

    PYTHONPATH=.:third_party/Megatron-LM CUDA_VISIBLE_DEVICES= python -m experiments.mist_w3po_v1.crf.test_t0_agree_labels
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np

from experiments.mist_w3po_v1.crf import precheck_b2_earliest_write as B
from experiments.mist_w3po_v1.crf import t0_agree_labels as T0

DEV_TARGETS = "eval_outputs/mist_w3po_v1/crf/b2_earliest_write_dev5/cpu/targets.jsonl"


def test_earliest_stable():
    a = np.array([[0, 1, 0, 2],
                  [1, 0, 0, 2],
                  [2, 2, 0, 1],
                  [1, 2, 0, 2],
                  [2, 2, 0, 2]])
    e, zero = T0.earliest_stable(a, u=[3, 3, 4, 2])
    # piece 0: a = 0,1,2,1,2 -> stable at >= 2 only from read 4; piece 1: 1,0,2,2,2 -> from read 2
    # piece 2: never agrees -> u = 4; piece 3: 2,2,1,2,2 -> from read 3
    assert e.tolist() == [4, 2, 4, 3] and zero.tolist() == [False, False, True, False]
    e_lit, _ = T0.earliest_stable(a, literal=True)
    assert e_lit.tolist() == [4, 2, 0, 3]
    # brute force on random curves
    rng = np.random.default_rng(0)
    for _ in range(300):
        R, J = int(rng.integers(1, 7)), int(rng.integers(1, 5))
        a = rng.integers(0, 4, (R, J))
        u = rng.integers(0, R, J).tolist()
        e, zero = T0.earliest_stable(a, u)
        for j in range(J):
            if a[-1, j] == 0:
                assert e[j] == u[j] and zero[j]
            else:
                ref = min(r for r in range(R) if all(a[x, j] >= a[-1, j] for x in range(r, R)))
                assert e[j] == ref
    assert T0.earliest_stable(np.array([[0, 3]]), [0, 0])[0].tolist() == [0, 0]


def test_spans_and_counts():
    recs = [json.loads(l) for l in itertools.islice(open(DEV_TARGETS, encoding="utf-8"), 40)]
    n_pc = 0
    for rec in recs:
        sp = T0.text_spans(rec)
        assert len(sp) == len(rec["pieces"])
        assert sum(b - a for a, b in sp) == len(rec["S_ids"]) - 3 * len(rec["pieces"])
        for (a, b), p in zip(sp, rec["pieces"]):
            assert a == p["tok0"] + 2 and b == p["tok1"] - 1 and b > a
        n_pc += len(sp)
        hit = np.zeros((3, len(rec["S_ids"])), bool)
        hit[1, sp[0][0]] = True
        hit[2, :] = True
        cnt = T0.agree_counts(hit, sp)
        assert cnt[0].sum() == 0 and cnt[1, 0] == 1 and cnt[1, 1:].sum() == 0
        assert cnt[2].tolist() == [b - a for a, b in sp]
    assert n_pc > 40
    # a broken piece (no start-time token) is caught
    bad = json.loads(json.dumps(recs[0]))
    bad["S_ids"][bad["pieces"][0]["tok0"] + 1] = 100
    try:
        T0.text_spans(bad)
        raise AssertionError("structure check missed")
    except AssertionError as ex:
        assert "structure check missed" not in str(ex)


def test_make_item():
    rec = json.loads(next(open(DEV_TARGETS, encoding="utf-8")))
    R = len(rec["t_ms"])
    wav = np.zeros(rec["dur_ms"] * 16, np.float32)
    post = np.ones((rec["dur_ms"] // 80 + 1, 8), np.float32)
    for r in range(R):
        d0, d1 = B.make_item(rec, r, wav, post), T0.make_item(rec, r, wav, post, "dev")
        assert d0.t_ms == d1.t_ms and d0.final == d1.final and d0.full == d1.full and d0.akey == d1.akey
        assert np.array_equal(d0.post, d1.post)
    tr = dict(rec, close_detect_ms=None, open_detect_ms=None)
    it = T0.make_item(tr, R - 1, wav, post, "train")
    t = tr["t_ms"][-1]
    assert it.final and it.full and it.t_ms == t
    assert (it.post[: t // 80] == 1).all() and (it.post[t // 80:] == 0).all()
    assert (post == 1).all()                                  # the window's posteriors are not modified
    if R > 1:
        it0 = T0.make_item(tr, 0, wav, post, "train")
        assert not it0.final and not it0.full and it0.post is post


def test_train_targets():
    from experiments.mist_w3po_v1.crf.build_sched_windows import tokenizer

    src = Path(T0.SRC_TRAIN)
    smap = json.loads((src / "slot_map.json").read_text(encoding="utf-8"))["lowlat"]
    want = ["EN2001a_zh_0000", "EN2001a_zh_0005", "EN2001a_zh_0040"]
    W = {}
    for l in open(src / "windows_train_ref.jsonl", encoding="utf-8"):
        w = json.loads(l)
        if w["id"] in want:
            W[w["id"]] = w
        if len(W) == len(want):
            break
    sched = {}
    for l in open(Path(T0.SCHED_V1) / "shard_000.jsonl", encoding="utf-8"):
        r = json.loads(l)
        if r["id"] in want:
            sched[r["id"]] = r
    tok = tokenizer()
    for i in want:
        rec = T0.train_target(W[i], smap[W[i]["meeting"]], tok, sched[i])   # asserts the chunks equal sched_v1's
        assert rec["close_detect_ms"] is None and rec["joint_tok_equal"]
        assert all(p["n_words"] <= 4 for p in rec["pieces"])
        keys = [(p["commit_ms"], p["src_start_ms"]) for p in rec["pieces"]]
        assert keys == sorted(keys)
        assert rec["t_ms"] == B.read_times(rec["dur_ms"], 500)
        u = T0.prop_reads(rec)
        assert u == [B.first_read(rec["t_ms"], p["src_end_ms"] + 1000) for p in rec["pieces"]]
        T0.text_spans(rec)
        # a perturbed sched record is caught
        bad = json.loads(json.dumps(sched[i]))
        bad["chunks"][0]["src_end_ms"] += 1
        try:
            T0.train_target(W[i], smap[W[i]["meeting"]], tok, bad)
            raise RuntimeError("missed")
        except AssertionError:
            pass


def test_plan_windows():
    from experiments.mist_w3po_v1.crf.train_crf_sft import plan_steps

    rng = np.random.default_rng(1)
    durs = {f"w{i:04d}": int(rng.integers(300, 15000)) for i in range(400)}
    used = T0.plan_windows(durs, 7, 30, 64)
    ids = sorted(durs)
    Rs = [len(B.read_times(durs[i], 500)) for i in ids]
    plan = plan_steps(Rs, 7, 30, 64)
    assert used == sorted({ids[i] for s in plan for i in s})
    assert sum(Rs[ids.index(i)] for i in used) >= 30 * 32


def test_t0_events():
    rec = {"t_ms": [500, 1000, 1500, 2000, 2300], "dur_ms": 2300, "open_detect_ms": 800, "close_detect_ms": 3100,
           "pieces": [{"utt": 0, "n_words": 4, "slot": 1}, {"utt": 1, "n_words": 2, "slot": 2},
                      {"utt": 0, "n_words": 3, "slot": 1}, {"utt": 1, "n_words": 1, "slot": 2}]}
    ev = T0.t0_events(rec, np.array([2, 0, 1, 4]))
    assert [x["read"] for x in ev] == [2, 0, 2, 4]
    assert [x["c"] for x in ev] == [1500, 800, 1500, 3100]
    assert [x["final"] for x in ev] == [False, False, False, True]


def main() -> None:
    tests = [test_earliest_stable, test_spans_and_counts, test_make_item, test_train_targets, test_plan_windows,
             test_t0_events]
    for t in tests:
        t()
        print("PASS", t.__name__, flush=True)
    print(f"ALL {len(tests)} PASSED")


if __name__ == "__main__":
    main()

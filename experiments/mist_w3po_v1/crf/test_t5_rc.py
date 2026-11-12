"""Tests of t5_evidence_labels --granularity pieces4rc (RC5, reports/mist_w3po_v1/thinker_rc5_prereg_v1.zh-CN.md §1), CPU:

  (a) coalesce_same_read on synthetic utterances: adjacent pieces with the same label read merge, different reads stay
      apart; a piece without an EN_WORD token joins the next piece (at the end: the previous one); an utterance without
      any token is untouched; every word keeps its label read; texts and spans concatenate exactly
  (b) defaults unchanged: on real train windows (the first windows of the T5 build order), window_chunks with pieces4 /
      merge3 reproduce the chunk texts and label reads of the existing sched_t5_pieces4_w10_v1 / sched_t5_merge3_w10_v1
      records (the new code path is not taken)
  (c) pieces4rc on the same windows: the per-word label reads equal pieces4's, the word sequence per utterance is the
      same, no two adjacent pieces of an utterance share a label read, and each merged piece's source span is the
      contiguous union of its parts (start of the first, end of the last)

    PYTHONPATH=.:third_party/Megatron-LM CUDA_VISIBLE_DEVICES= python -m experiments.mist_w3po_v1.crf.test_t5_rc
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from experiments.mist_w3po_v1.crf import precheck_b2_earliest_write as B
from experiments.mist_w3po_v1.crf import t5_evidence_labels as T5

C = Path("eval_outputs/mist_w3po_v1/crf")
N_WIN = 300


def test_coalesce_synthetic():
    t_ms = [1000, 2000, 3000, 3500]
    # utterance 0: pieces a(2 tok, tau 900 -> read 0) b(1 tok, tau 950 -> 0) c(no token) d(2 tok, tau 2100 -> 2)
    #              e(1 tok, tau 2900 -> 2) f(1 tok, tau 3400 -> 3)
    cs = ["a1 a2", "b1", "--", "d1 d2", "e1", "f1"]
    spans = [[0, 1], [2], [], [3, 4], [5], [6]]
    tau = {(0, 0): 500, (0, 1): 900, (0, 2): 950, (0, 3): 1500, (0, 4): 2100, (0, 5): 2900, (0, 6): 3400}
    x = (0, {"ref": "x"}, cs, spans, None, None, None)
    mi, u, cs2, sp2, *_ = T5.coalesce_same_read(x, tau, t_ms)
    assert cs2 == ["a1 a2 b1", "-- d1 d2 e1", "f1"], cs2
    assert sp2 == [[0, 1, 2], [3, 4, 5], [6]], sp2
    e_word = {j: B.first_read(t_ms, tau[(0, sp[-1])]) for sp in sp2 for j in sp}
    e_old = {j: B.first_read(t_ms, tau[(0, sp[-1])]) for sp in spans if sp for j in sp}
    assert e_word == e_old
    # trailing token-free piece joins the previous one
    x2 = (0, {}, ["a1", "b1", "!!"], [[0], [1], []], None, None, None)
    out = T5.coalesce_same_read(x2, {(0, 0): 100, (0, 1): 1500}, t_ms)
    assert out[2] == ["a1", "b1 !!"] and out[3] == [[0], [1]], out[2:4]
    # no token at all: untouched
    x3 = (0, {}, ["--", "!!"], [[], []], None, None, None)
    assert T5.coalesce_same_read(x3, {}, t_ms) is x3
    # all in one read -> one piece
    x4 = (0, {}, ["a b c d", "e f g h", "i"], [[0, 1, 2, 3], [4, 5, 6, 7], [8]], None, None, None)
    tau4 = {(0, j): 3100 + j for j in range(9)}
    assert T5.coalesce_same_read(x4, tau4, t_ms)[2] == ["a b c d e f g h i"]


def _windows(sched_dir: Path, n: int):
    recs = {}
    for f in sorted(sched_dir.glob("shard_*.jsonl")):
        for l in open(f, encoding="utf-8"):
            r = json.loads(l)
            recs[r["id"]] = r
            if len(recs) >= n:
                return recs
    return recs


def _real_setup():
    W = {w["id"]: w for w in map(json.loads, open(T5.TRAIN_WIN, encoding="utf-8"))}
    align = T5.load_align(str(C / "t5_align_v1" / "train"))
    smap = json.loads((T5.DATA / "e2e_gstar_zh2en_v1" / "slot_map.json").read_text(encoding="utf-8"))["lowlat"]
    return W, align, smap


def _chunks(w, align, smap, gran):
    t_ms = B.read_times(w["dur_ms"], 1000)
    return T5.window_chunks(w["utts"], w["start_s"], w["dur_ms"], t_ms, smap[w["meeting"]], align, gran), t_ms


def _word_reads(ch):
    out = defaultdict(list)
    for c in sorted(ch, key=lambda c: (c["utt"], c["k"])):
        out[c["utt"]] += [(wd, c["e"]) for wd in c["text"].split()]
    return dict(out)


def test_defaults_and_rc_on_real_windows():
    W, align, smap = _real_setup()
    n_merge = n_pieces = 0
    for gran, d in (("pieces4", "sched_t5_pieces4_w10_v1"), ("merge3", "sched_t5_merge3_w10_v1")):
        recs = _windows(C / d, N_WIN)
        assert len(recs) == N_WIN
        for wid, r in recs.items():
            ch, _ = _chunks(W[wid], align, smap, gran)
            got = sorted((c["utt"], c["k"], c["text"], c["e"]) for c in ch)
            exp = sorted((c["utt"], c["k"], c["text"], c["e_read"]) for c in r["chunks"])
            assert got == exp, (gran, wid)
    recs = _windows(C / "sched_t5_pieces4_w10_v1", N_WIN)
    for wid in recs:
        p4, t_ms = _chunks(W[wid], align, smap, "pieces4")
        rc, _ = _chunks(W[wid], align, smap, "pieces4rc")
        assert _word_reads(p4) == _word_reads(rc), wid
        n_pieces += len(p4)
        n_merge += len(p4) - len(rc)
        by = defaultdict(list)
        for c in rc:
            by[c["utt"]].append(c)
        for cs in by.values():
            cs.sort(key=lambda c: c["k"])
            assert all(a["e"] != b["e"] for a, b in zip(cs, cs[1:])), wid
            assert all(a["src_end_ms"] == b["src_start_ms"] for a, b in zip(cs, cs[1:])), wid
        # merged span = union of the parts: first part's start, last part's end
        byp = defaultdict(list)
        for c in p4:
            byp[c["utt"]].append(c)
        for ui, cs in by.items():
            ps = sorted(byp[ui], key=lambda c: c["k"])
            i = 0
            for c in cs:
                nw = len(c["text"].split())
                parts = []
                while nw > 0:
                    parts.append(ps[i])
                    nw -= len(ps[i]["text"].split())
                    i += 1
                assert nw == 0 and " ".join(p["text"] for p in parts) == c["text"], wid
                assert c["src_start_ms"] == parts[0]["src_start_ms"] and c["src_end_ms"] == parts[-1]["src_end_ms"], wid
                assert all(p["e"] == c["e"] for p in parts if p["n_words"]), wid
    assert n_merge > 0
    print(f"  real windows: {N_WIN}, pieces4 {n_pieces} -> pieces4rc {n_pieces - n_merge}")


if __name__ == "__main__":
    for name, f in list(globals().items()):
        if name.startswith("test_") and callable(f):
            f()
            print(f"{name}: ok")
    print("ALL_OK")

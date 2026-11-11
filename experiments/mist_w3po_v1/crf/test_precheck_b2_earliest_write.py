"""Tests of crf/precheck_b2_earliest_write.py (CPU only):

  (a) earliest: the largest-jump argmax with the first read's jump = -inf, ties -> the earlier read, R = 1; the literal
      reading (first jump +inf) is degenerate (every piece at the first read)
  (b) cummax_by_speaker: monotone within each speaker, in kappa order, speakers independent
  (c) online clamps: >= open detect; a final-read write >= close detect; read schedules and first_read
  (d) pieces: <= 4 words, the words of each chunk partitioned in order, character-proportional source ends, the last
      piece of a chunk = utt_chunks' chunk end
  (e) members: the S0-3 rule (straddle / no_speech / gate / kept, window-relative clamped times, sorted)
  (f) label-path LAAL on a toy window: the three schedules' write times and LAAL by hand, and the in-utterance test
  (g) members() reproduces build_windows_zh2en's windows_train_diar membership (window ids and record ids) on the
      first 2 meetings of windows_zh2en_c2g06_train16_v1 (metadata only; no audio, no model)

    PYTHONPATH=.:third_party/Megatron-LM CUDA_VISIBLE_DEVICES= python -m experiments.mist_w3po_v1.crf.test_precheck_b2_earliest_write
"""
from __future__ import annotations

import json
import math
from collections import defaultdict

import numpy as np

from experiments.mist_w3po_v1.crf import precheck_b2_earliest_write as B


def test_earliest():
    s = np.array([[-5.0, -5.0, -5.0],
                  [-1.0, -4.0, -5.0],
                  [-0.5, -1.0, -4.9],
                  [-0.4, -0.9, -1.0]])
    assert B.earliest(s).tolist() == [1, 2, 3]
    # the first read is never the argmax, even when it is the best read
    s2 = np.array([[-0.1], [-3.0], [-2.0]])
    assert B.earliest(s2).tolist() == [2]
    # ties -> the earlier read
    s3 = np.array([[-3.0], [-2.0], [-1.0]])
    assert B.earliest(s3).tolist() == [1]
    # one read only
    assert B.earliest(np.array([[-2.0, -1.0]])).tolist() == [0, 0]
    # the literal reading of the pre-registration (s(r0 - 1) = -inf -> first jump +inf) is degenerate
    assert B.earliest(s, first_jump=math.inf).tolist() == [0, 0, 0]


def test_cummax():
    assert B.cummax_by_speaker([3, 1, 2, 0, 4], [1, 2, 1, 2, 2]) == [3, 1, 3, 1, 4]
    assert B.cummax_by_speaker([], []) == []


def test_online_and_reads():
    w = {"open_detect_ms": 1240, "close_detect_ms": 4520, "dur_ms": 3140}
    assert B.online(500, w) == 1240
    assert B.online(2000, w) == 2000
    assert B.online(3140, w) == 4520
    assert B.read_times(3140, 500) == [500, 1000, 1500, 2000, 2500, 3000, 3140]
    assert B.read_times(3000, 1000) == [1000, 2000, 3000]
    assert B.read_times(300, 500) == [300]
    T = B.read_times(3140, 500)
    assert B.first_read(T, 1000) == 1 and B.first_read(T, 1001) == 2 and B.first_read(T, 9999) == len(T) - 1


def test_pieces():
    from experiments.uniss_qwen3_omni_v1.data_clean.build_ami_ms_rows import utt_chunks

    ref = "so we can go to the next item, okay. yes"
    u = {"begin_time": 10.2, "end_time": 13.4, "zh": ref}
    ch, pc = B.utt_pieces(u, 10.0)
    assert [c["text"] for c in ch] == ["so we can go to the next item,", "okay.", "yes"]
    assert [p["text"] for p in pc] == ["so we can go", "to the next item,", "okay.", "yes"]
    assert [p["chunk"] for p in pc] == [0, 0, 1, 2]
    assert all(p["n_words"] <= 4 for p in pc)
    assert " ".join(p["text"] for p in pc).split() == ref.split()
    # chars without spaces / punctuation: so we can go = 9, to the next item = 13, okay = 4, yes = 3 -> 29
    assert pc[0]["src_start_ms"] == 200 and pc[0]["src_end_ms"] == round((10.2 + 9 / 29 * 3.2 - 10.0) * 1000)
    assert pc[1]["src_start_ms"] == pc[0]["src_end_ms"]
    ref_ch = utt_chunks(dict(u), [], 1000, 10.0)
    assert pc[1]["src_end_ms"] == ref_ch[0]["src_end_ms"] and pc[2]["src_end_ms"] == ref_ch[1]["src_end_ms"]
    assert pc[3]["src_end_ms"] == ref_ch[2]["src_end_ms"] == 3400
    assert [c["commit_ms"] for c in ch] == [c["src_end_ms"] + 1000 for c in ch]


def _talk():
    return [{"record_id": "a", "speaker_id": "S1", "zh_begin": 10.5, "zh_end": 12.0, "ref": "hello there", "_bad": None},
            {"record_id": "b", "speaker_id": "S2", "zh_begin": 11.0, "zh_end": 11.8, "ref": "yes", "_bad": None},
            {"record_id": "c", "speaker_id": "S1", "zh_begin": 13.9, "zh_end": 15.0, "ref": "no", "_bad": None},
            {"record_id": "d", "speaker_id": "S2", "zh_begin": 20.0, "zh_end": 21.0, "ref": "x", "_bad": "cer"}]


def test_members():
    talk = _talk()
    mem, why = B.members({"start_s": 10.0, "dur_ms": 3000}, talk)
    assert why is None and [u["record_id"] for u in mem] == ["a", "b"]
    assert mem[0]["begin_s"] == 0.5 and mem[0]["end_s"] == 2.0 and mem[1]["begin_s"] == 1.0
    # c is 1.0 / 1.1 = 0.909 inside [10, 14.9]: kept, its end clamped to the window end
    mem, why = B.members({"start_s": 10.0, "dur_ms": 4900}, talk)
    assert why is None and [u["record_id"] for u in mem] == ["a", "b", "c"] and mem[2]["end_s"] == 4.9
    # c only 0.8 / 1.1 inside [10, 14.7]: straddle
    assert B.members({"start_s": 10.0, "dur_ms": 4700}, talk) == (None, "straddle")
    assert B.members({"start_s": 16.0, "dur_ms": 2000}, talk) == (None, "no_speech")
    assert B.members({"start_s": 19.5, "dur_ms": 2000}, talk) == (None, "gate")


def _toy_window():
    w = {"id": "M_on_0000", "meeting": "M", "start_s": 10.0, "dur_ms": 4000, "audio": "x.wav", "post": {"lowlat": "x.npy"},
         "open_detect_ms": 1240, "close_detect_ms": 5000}
    mem = [{"record_id": "u1", "speaker_id": "S1", "begin_s": 0.2, "end_s": 3.4,
            "ref": "one two three four five six seven eight", "hesitation": False}]
    return B.build_window(w, mem, {"S1": 1})


def test_label_laal_toy():
    rec = _toy_window()
    # chars 3+3+5+4 = 15 of 32: piece 1 ends at 0.2 + 15/32 * 3.2 = 1.7 s, piece 2 at 3.4 s
    assert [(p["text"], p["src_end_ms"], p["commit_ms"]) for p in rec["pieces"]] == [
        ("one two three four", 1700, 2700), ("five six seven eight", 3400, 4000)]
    assert rec["pieces"][0]["piece"] == "<|spk1|><|t2|>one two three four<|t21|>"
    assert len(rec["chunks"]) == 1 and rec["chunks"][0]["commit_ms"] == 4000
    assert rec["t_cur"] == [1000, 2000, 3000, 4000] and rec["t_ms"] == [500 * k for k in range(1, 9)]
    cur = B.schedule(rec, "S_cur")
    assert [(e["read"], e["final"], e["c"]) for e in cur] == [(3, True, 5000)]
    prop = B.schedule(rec, "S_prop")
    assert [(e["t"], e["final"], e["c"]) for e in prop] == [(3000, False, 3000), (4000, True, 5000)]
    # s: piece 1 jumps most at read index 2 (t = 1.5 s), piece 2 at the final read; a first-read write (t = 0.5 s)
    # would be clamped to the open detect (1.24 s)
    s = np.full((8, 2), -6.0)
    s[2:, 0] = -1.0
    s[7, 1] = -0.5
    e = B.earliest(s)
    assert e.tolist() == [2, 7]
    lik = B.schedule(rec, "S_lik", e)
    assert [(e_["t"], e_["final"], e_["c"]) for e_ in lik] == [(1500, False, 1500), (4000, True, 5000)]
    unit = {"a": 10.2, "b": 13.4, "ref": rec["members"][0]["ref"], "utts": ["u1"], "reg": "1spk"}
    got = {}
    for nm, ev in (("S_cur", cur), ("S_prop", prop), ("S_lik", lik)):
        words_of = defaultdict(list)
        for x in ev:
            words_of[rec["members"][x["utt"]]["record_id"]].append((x["n_words"], rec["start_s"] + x["c"] / 1000))
        r = B.unit_laal({"u1": unit}, words_of)
        assert r["missing"] == []
        got[nm] = r["laal"]["u1"][0]
    # X = 3.2 s, Y = 8 words, X / Y = 0.4
    #   S_cur  every word at 15.0 s: d = 4.8 >= X at once -> 4.8
    #   S_prop d = 2.8 x 4, 4.8 x 4: tau = 4 -> (2.8 + 2.4 + 2.0 + 1.6 + 3.2) / 5 = 2.4
    #   S_lik  d = 1.3 x 4, 4.8 x 4: tau = 4 -> (1.3 + 0.9 + 0.5 + 0.1 + 3.2) / 5 = 1.2
    assert abs(got["S_cur"] - 4.8) < 1e-9 and abs(got["S_prop"] - 2.4) < 1e-9 and abs(got["S_lik"] - 1.2) < 1e-9
    # incomplete unit: an utterance without words is reported, not averaged
    r = B.unit_laal({"u1": unit, "u2": {**unit, "utts": ["u2"]}}, {"u1": [(8, 15.0)]})
    assert list(r["laal"]) == ["u1"] and r["missing"] == ["u2"]
    # GO line 2 on the toy: the non-last piece is written at 1.5 s < the utterance end 3.4 s; under S_prop at 3.0 s
    utt_end = round(rec["members"][0]["end_s"] * 1000)
    assert lik[0]["c"] < utt_end and prop[0]["c"] < utt_end and not rec["pieces"][0]["last"] and rec["pieces"][1]["last"]


def test_piece_scores_and_kappa():
    w = {"id": "M_on_0001", "meeting": "M", "start_s": 0.0, "dur_ms": 6000, "audio": "x.wav", "post": {"lowlat": "x.npy"},
         "open_detect_ms": 1240, "close_detect_ms": 7000}
    mem = [{"record_id": "u1", "speaker_id": "S1", "begin_s": 0.2, "end_s": 4.0, "ref": "a b c d e f g h i",
            "hesitation": False},
           {"record_id": "u2", "speaker_id": "S2", "begin_s": 1.0, "end_s": 2.0, "ref": "yes indeed", "hesitation": False}]
    rec = B.build_window(w, mem, {"S1": 1, "S2": 2})
    keys = [(p["commit_ms"], p["src_start_ms"]) for p in rec["pieces"]]
    assert keys == sorted(keys)
    assert [p["utt"] for p in rec["pieces"]] == [0, 1, 0, 0]          # u2's piece (ends 2.0 s) sorts between u1's
    for p, (t0, t1) in zip(rec["pieces"], [(0, 3), (3, 5), (5, 9), (9, 10)]):
        p["tok0"], p["tok1"] = t0, t1
    tl = np.arange(20, dtype=np.float64).reshape(2, 10)
    s = B.piece_scores(tl, rec["pieces"])
    assert s.shape == (2, 4) and s[0].tolist() == [1.0, 3.5, 6.5, 9.0] and s[1].tolist() == [11.0, 13.5, 16.5, 19.0]
    # S_lik's cumulative max is per speaker: u2 is not held back by u1
    ev = B.schedule(rec, "S_lik", [5, 1, 3, 9])
    assert [x["read"] for x in ev] == [5, 1, 5, 9]


def test_members_match_builder():
    root = "eval_outputs/mist_w3po_v1/data/windows_zh2en_c2g06_train16_v1"
    online = [json.loads(l) for l in open(f"{root}/windows_train_online.jsonl", encoding="utf-8")]
    diar = {d["id"]: d for d in map(json.loads, open(f"{root}/windows_train_diar.jsonl", encoding="utf-8"))}
    meetings = sorted({w["meeting"] for w in online})[:2]
    en = {r["record_id"]: r["en"] for r in map(json.loads, open(B.EN_NORM, encoding="utf-8"))}
    n = 0
    for m in meetings:
        talk = B.talk_utts(m, en_norm=en, asr={})
        for w in (w for w in online if w["meeting"] == m):
            mem, why = B.members(w, talk)
            d = diar.get(w["id"])
            assert (d is None) == (mem is None), (w["id"], why)
            if d is not None:
                assert [u["record_id"] for u in mem] == [u["record_id"] for u in d["utts"]], w["id"]
                assert all(abs(a["begin_s"] - b["begin_s"]) <= 0.0015 and abs(a["end_s"] - b["end_s"]) <= 0.0015
                           and a["ref"] == b["ref"] for a, b in zip(mem, d["utts"])), w["id"]
                n += 1
    assert n > 100, n


if __name__ == "__main__":
    import time

    tests = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    t0 = time.time()
    for name, fn in tests:
        t1 = time.time()
        fn()
        print(f"PASS {name} ({time.time() - t1:.1f}s)", flush=True)
    print(f"ALL {len(tests)} TESTS PASSED ({time.time() - t0:.1f}s)")

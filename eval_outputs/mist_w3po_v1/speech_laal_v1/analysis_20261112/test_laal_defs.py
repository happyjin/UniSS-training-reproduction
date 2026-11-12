"""CPU tests for laal_defs.py on toy examples (no data read).

    /opt/dlami/nvme/jasonleeeli/conda_envs/qwen3omni/bin/python \
        eval_outputs/mist_w3po_v1/speech_laal_v1/analysis_20261112/test_laal_defs.py
"""
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import laal_defs as L  # noqa: E402
from experiments.uniss_qwen3_omni_v1.evaluation.si_metrics import laal as si_laal  # noqa: E402


# --- SimulEval 1.1.4, simuleval/evaluator/scorers/latency_scorer.py: LatencyScorer.get_delays_lengths and
# LAALScorer.compute copied verbatim (default use_ref_len=True, computation_aware=False), on a stub Instance whose
# delays / source_length / reference_length are what instance.py's SpeechOutputInstance + SpeechInputInstance hold.
class _Ins:
    def __init__(self, delays, source_length, reference, latency_unit="word"):
        self.delays, self.source_length, self.reference, self.latency_unit = delays, source_length, reference, latency_unit

    @property
    def reference_length(self) -> int:   # instance.py Instance.reference_length (word / char)
        if self.latency_unit == "word":
            return len(self.reference.split(" "))
        elif self.latency_unit == "char":
            return len(self.reference.strip())
        raise NotImplementedError


class _SimulEvalLAAL:
    use_ref_len = True
    timestamp_type = "delays"

    def get_delays_lengths(self, ins):
        delays = getattr(ins, self.timestamp_type, None)
        assert delays
        if not self.use_ref_len or ins.reference is None:
            tgt_len = len(delays)
        else:
            tgt_len = ins.reference_length
        src_len = ins.source_length
        return delays, src_len, tgt_len

    def compute(self, ins):
        delays, source_length, target_length = self.get_delays_lengths(ins)
        if delays[0] > source_length:
            return delays[0]

        LAAL = 0
        gamma = max(len(delays), target_length) / source_length
        tau = 0
        for t_minus_1, d in enumerate(delays):
            LAAL += d - t_minus_1 / gamma
            tau = t_minus_1 + 1

            if d >= source_length:
                break
        LAAL /= tau
        return LAAL


def test_d0_hand_example():
    # 3 segments read at 1000 / 2000 / 4000 ms of a 3000 ms source, 4 reference words: gamma = 4 / 3000,
    # tau = 3 (4000 >= 3000): (1000 + (2000 - 750) + (4000 - 1500)) / 3
    v = L.laal_simuleval([1000.0, 2000.0, 4000.0], 3000.0, 4)
    assert abs(v - 4750 / 3) < 1e-9
    assert v == _SimulEvalLAAL().compute(_Ins([1000.0, 2000.0, 4000.0], 3000.0, "a b c d"))
    # first delay past the source end: LAAL = d_1
    assert L.laal_simuleval([3500.0, 3600.0], 3000.0, 4) == 3500.0
    # exactly at the end: the loop takes it (tau = 1), same value
    assert L.laal_simuleval([3000.0, 3600.0], 3000.0, 4) == 3000.0
    # more segments than reference words: gamma uses the segment count
    assert abs(L.laal_simuleval([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 3000.0, 2) - (-(0 + 500 + 1000 + 1500 + 2000 + 2500) / 6)) < 1e-9


def test_d0_matches_simuleval_randomised():
    rng = random.Random(20261112)
    sc = _SimulEvalLAAL()
    for _ in range(2000):
        n = rng.randint(1, 12)
        src = rng.uniform(300, 12000)
        delays = sorted(rng.uniform(0, src * 1.3) for _ in range(n))
        ref = " ".join("w" for _ in range(rng.randint(1, 30)))
        assert L.laal_simuleval(delays, src, len(ref.split(" "))) == sc.compute(_Ins(delays, src, ref))
        zh = "".join("字" for _ in range(rng.randint(1, 30))) + "。"
        assert L.laal_simuleval(delays, src, len(zh.strip())) == sc.compute(_Ins(delays, src, zh, "char"))


def test_unitwise_formula_is_one_formula():
    # speech_latency_lit.laal (multi-speaker D1 / D2) == si_metrics.laal (the paper's D1) on sorted delays
    rng = random.Random(7)
    for _ in range(2000):
        n = rng.randint(1, 25)
        X = rng.uniform(0.3, 12.0)
        d = sorted(rng.uniform(-0.5, X * 1.4) for _ in range(n))
        T = rng.randint(1, 30)
        assert abs(L.laal_unitwise(d, X, T) - si_laal(d, X, T)) < 1e-9


def _toy_unit():
    # unit [10 s, 14 s], 6 reference words; two blocks of two words each
    b1 = {"write": 11.0, "avail": 12.0, "on_start": 12.0, "on_end": 13.0, "rec_start": 11.0, "rec_end": 12.0, "text": "w1 w2"}
    b2 = {"write": 13.5, "avail": 15.0, "on_start": 15.0, "on_end": 16.0, "rec_start": 13.5, "rec_end": 14.5, "text": "w3 w4"}
    s1 = {"a": 10.2, "text": "w1 w2", "commit": 12.0, "raw_commit": 11.0, "blk": b1}
    s2 = {"a": 12.5, "text": "w3 w4", "commit": 15.0, "raw_commit": 13.5, "blk": b2}
    u = {"a": 10.0, "b": 14.0, "ref": "r1 r2 r3 r4 r5 r6"}
    return [(s1, "w1 w2"), (s2, "w3 w4")], u


def test_unit_defs_toy():
    pw, u = _toy_unit()
    r = L.unit_defs(pw, u)
    # D0: segments emitted at 2 s and 5 s (online), X = 4, |Y*| = 6 -> gamma = 1.5: (2 + (5 - 1/1.5)) / 2
    assert abs(r["D0"] - (2 + 5 - 1 / 1.5) / 2) < 1e-9
    # D0_raw: 1 s and 3.5 s, none past X -> tau = 2
    assert abs(r["D0_raw"] - (1 + 3.5 - 1 / 1.5) / 2) < 1e-9
    # D1: words at 2, 2, 5, 5; X / |Y| = 4 / 6; tau = 3 (5 >= 4): (2 + (2 - 2/3) + (5 - 4/3)) / 3
    assert abs(r["D1"] - (2 + 2 - 2 / 3 + 5 - 4 / 3) / 3) < 1e-9
    # D2: words spoken by 2.5, 3.0, 5.5, 6.0: (2.5 + (3 - 2/3) + (5.5 - 4/3)) / 3
    assert abs(r["D2"] - (2.5 + 3 - 2 / 3 + 5.5 - 4 / 3) / 3) < 1e-9
    # recorded timeline (no detector): words at 11, 11, 13.5, 13.5 (-10); spoken by 1.5, 2, 4, 4.5
    assert abs(r["D1_rec"] - (1 + 1 - 2 / 3 + 3.5 - 4 / 3 + 3.5 - 2) / 4) < 1e-9
    assert abs(r["D2_rec"] - (1.5 + 2 - 2 / 3 + 4 - 4 / 3) / 3) < 1e-9
    # text: D0_text = D0 here (one segment per block); text_laal: words at 2, 2, 5, 5 = D1
    assert abs(r["D0_text"] - r["D0"]) < 1e-12 and abs(r["text_laal"] - r["D1"]) < 1e-12
    assert r["n_seg"] == 2 and r["n_words"] == 4
    # D0w: words at their block's online commit (no queue wait here, = D1); D0w_raw at the raw write (= D1_rec)
    assert abs(r["D0w"] - r["D1"]) < 1e-12 and abs(r["D0w_raw"] - r["D1_rec"]) < 1e-12
    assert abs(r["text_laal_raw"] - r["D0w_raw"]) < 1e-12


def test_unit_defs_queue():
    # block 2 is committed at 12.5 s but block 1 plays until 13 s: D1 / D2 see the queue, D0 / D0w do not
    pw, u = _toy_unit()
    b2 = pw[1][0]["blk"]
    b2.update({"avail": 12.5, "on_start": 13.0, "on_end": 14.0})
    pw[1][0]["commit"] = 12.5
    r = L.unit_defs(pw, u)
    assert abs(r["D0"] - (2 + 2.5 - 1 / 1.5) / 2) < 1e-9                 # 2 and 2.5, none >= 4: tau = 2
    assert abs(r["D0w"] - (2 + 2 - 2 / 3 + 2.5 - 4 / 3 + 2.5 - 2) / 4) < 1e-9
    assert abs(r["D1"] - (2 + 2 - 2 / 3 + 3 - 4 / 3 + 3 - 2) / 4) < 1e-9


def test_unit_defs_split_block_counts_once():
    # one segment split into two pieces of the same unit: ONE D0 segment, all its words in D1 / D2
    b = {"write": 11.0, "avail": 11.0, "on_start": 11.0, "on_end": 13.0, "rec_start": 11.0, "rec_end": 13.0,
         "text": "w1 w2 w3 w4"}
    s = {"a": 10.0, "text": "w1 w2 w3 w4", "commit": 11.0, "raw_commit": 11.0, "blk": b}
    u = {"a": 10.0, "b": 12.0, "ref": "r1 r2 r3 r4"}
    r = L.unit_defs([(s, "w1 w2"), (s, "w3 w4")], u)
    assert r["n_seg"] == 1 and r["D0"] == 1.0
    assert r["n_words"] == 4
    # D2: the 4 words spoken by 11.5, 12, 12.5, 13 (-10); X / |Y| = 0.5; tau = 2 (2 >= 2): (1.5 + (2 - 0.5)) / 2
    assert abs(r["D2"] - 1.5) < 1e-9


def test_text_only_unit():
    pw, u = _toy_unit()
    for x, _ in pw:
        x["blk"] = None
    r = L.unit_defs(pw, u, speech=False)
    assert "D0" not in r and abs(r["D0_text"] - (2 + 5 - 1 / 1.5) / 2) < 1e-9
    assert abs(r["D0_text_raw"] - (1 + 3.5 - 1 / 1.5) / 2) < 1e-9


def test_sentence_defs_toy():
    r = {"tgt_lang": "eng", "source_duration_ms": 3000, "translation_ref": "a b c d",
         "blocks": [{"write_ms": 1000, "start_ms": 1000, "dur_ms": 1000, "text": "w1 w2"},
                    {"write_ms": 3000, "start_ms": 3000, "dur_ms": 500, "text": "w3."}]}
    o = L.sentence_defs(r)
    assert abs(o["D0"] - (1000 + 3000 - 750) / 2) < 1e-9                    # 2 segments, gamma = 4 / 3000
    assert abs(o["D1"] - (1000 + 1000 - 750 + 3000 - 1500) / 3) < 1e-9      # 3 words at 1000, 1000, 3000
    assert abs(o["D2"] - (1500 + 2000 - 750 + 3500 - 1500) / 3) < 1e-9      # spoken by 1500, 2000, 3500
    assert o["D1"] == o["D1_sl"] or abs(o["D1"] - o["D1_sl"]) < 1e-9
    # Chinese target: units = CJK characters; D0_char counts the raw reference (punctuation included)
    z = {"tgt_lang": "cmn", "source_duration_ms": 2000, "translation_ref": "你好。",
         "blocks": [{"write_ms": 500, "start_ms": 600, "dur_ms": 400, "text": "你好"}]}
    o = L.sentence_defs(z)
    assert o["T"] == 2 and abs(o["D0"] - 500) < 1e-9 and abs(o["D0_char"] - 500) < 1e-9
    assert abs(o["D1"] - (600 + 600 - 1000) / 2) < 1e-9
    assert abs(o["D0w"] - (500 + 500 - 1000) / 2) < 1e-9                    # units at the write time
    # granularity: 900 ms of speech in 320 ms chunks -> 3 segments at 500 ms; gamma = max(3, 2) / 2000
    z["blocks"][0]["dur_ms"] = 900
    assert abs(L.sentence_d0_chunks(z, 320) - (500 + 500 - 2000 / 3 + 500 - 4000 / 3) / 3) < 1e-9
    assert abs(L.sentence_d0_chunks(z, 1000) - 500) < 1e-9
    # a write past the source end is read as the source end (SimulEval cannot read further)
    z["blocks"][0]["write_ms"] = 2600
    assert L.sentence_defs(z)["D0"] == 2000


if __name__ == "__main__":
    n = 0
    for k, f in list(globals().items()):
        if k.startswith("test_") and callable(f):
            f()
            n += 1
            print("ok", k)
    print(f"{n} tests passed")

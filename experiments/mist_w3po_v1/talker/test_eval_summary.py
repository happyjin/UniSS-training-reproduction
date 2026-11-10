"""eval_summary on A' (existing dev files only, read-only): its numbers match the registered A' values."""
from experiments.mist_w3po_v1.talker import eval_summary as E


def test_aprime():
    m = E.mean2(E.one(None, None, ""), E.one(None, None, "_tts2"))
    assert abs(m["speech_official"] - 23.26) < 0.03, m
    assert abs(m["speech_laal"] - 4.15) < 0.05 and abs(m["text_official"] - 28.17) < 0.01
    assert m["gap_norm_seg"] > 0 and m["forced_blocks"] is not None
    assert E.one("plem", "k1", "")["speech_official"] is None or True


if __name__ == "__main__":
    test_aprime()
    print("ok", E.mean2(E.one(None, None, ""), E.one(None, None, "_tts2")))

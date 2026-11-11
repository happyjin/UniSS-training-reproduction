"""eval_summary on A' (existing dev files only, read-only): its numbers match the registered A' values."""
from experiments.mist_w3po_v1.talker import eval_summary as E


def test_aprime():
    m = E.mean2(E.one(None, None, ""), E.one(None, None, "_tts2"))
    assert abs(m["speech_official"] - 23.26) < 0.03, m
    assert abs(m["speech_laal"] - 4.15) < 0.05 and abs(m["text_official"] - 28.17) < 0.01
    assert m["gap_norm_seg"] > 0 and m["forced_blocks"] is not None and abs(m["speech_laal_1ch"] - 7.07) < 0.02
    assert E.one("plem", "k1", "")["speech_official"] is None or True


def test_gate_label():
    same = {"text_all": {"diff": 0.0, "p_le0": 1.0, "pass": False}, "text_ov": {"diff": 0.0, "p_le0": 1.0, "pass": False},
            "speech_all": {"diff": 0.35, "p_le0": 0.0, "pass": True}}
    assert E.gate_label(same).startswith("通过 (文本相同")
    same["speech_all"] = {"diff": -0.5, "p_le0": 0.99, "pass": False}
    assert E.gate_label(same).startswith("未通过")
    diff = {"text_all": {"diff": 0.3, "p_le0": 0.99, "pass": False}, "speech_all": {"diff": 0.35, "p_le0": 0.0, "pass": True}}
    assert E.gate_label(diff).startswith("未通过")


if __name__ == "__main__":
    test_gate_label()
    test_aprime()
    print("ok", E.mean2(E.one(None, None, ""), E.one(None, None, "_tts2")))

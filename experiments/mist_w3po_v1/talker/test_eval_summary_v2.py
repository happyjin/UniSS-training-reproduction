"""eval_summary_v2 on the existing dev files (read-only): A' and PL-EM rows match the v1 table; the new systems are
empty until their runs exist; the vs-PL-EM differences of PL-EM itself would be 0."""
from experiments.mist_w3po_v1.talker import eval_summary_v2 as E2


def test_build():
    res = E2.build()
    a, p = res["A'"]["mean"], res["PL-EM c1+b1+s2"]["mean"]
    assert abs(a["speech_official"] - 23.26) < 0.03 and abs(p["speech_official"] - 23.62) < 0.03
    assert abs(p["speech_laal_1ch"] - 5.64) < 0.02 and res["PL-EM c1+b1+s2"]["gate"]["speech_all"]["pass"]
    assert all(v == 0 for v in E2.diff(p, p).values() if v is not None)
    md = E2.markdown(res)
    assert md[0].startswith("| 系统 |") and any(l.startswith("| 实验一 BoN-SFT") for l in md)


if __name__ == "__main__":
    test_build()
    print("ok")

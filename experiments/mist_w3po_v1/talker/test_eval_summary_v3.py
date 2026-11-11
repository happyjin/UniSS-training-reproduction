"""eval_summary_v3 on the existing dev files (read-only): the v2 rows are unchanged; E2' is empty until its run exists;
its vs-E2 block renders with missing values."""
from experiments.mist_w3po_v1.talker import eval_summary_v2 as E2
from experiments.mist_w3po_v1.talker import eval_summary_v3 as E3


def test_build():
    r2, r3 = E2.build(), E3.build()
    for k in r2:
        assert r3[k]["mean"] == r2[k]["mean"] and r3[k]["gate"] == r2[k]["gate"]
    assert E3.E2P[0] in r3 and "vs_plemm" in r3[E3.E2P[0]] and "vs_plem" in r3[E3.E2P[0]]
    md = E3.markdown(r3)
    assert any(l.startswith(f"| {E3.E2P[0]} |") for l in md) and any("E2′ 对 E2" in l for l in md)


if __name__ == "__main__":
    test_build()
    print("ok")

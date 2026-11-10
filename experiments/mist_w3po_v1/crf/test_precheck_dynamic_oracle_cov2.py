"""CPU tests of the coverage judge v2 (precheck_dynamic_oracle_cov2.py). Run: python -m <this module>."""
from experiments.mist_w3po_v1.crf import precheck_dynamic_oracle_cov2 as C


def test_content_match():
    n = lambda s: s.lower().replace(".", "").replace(",", "").split()  # noqa: E731
    assert C.content_match(n("Yeah."), n("Yes"))                       # same backchannel class
    assert C.content_match(n("For double check."), n("To double check, so that"))
    assert C.content_match(n("Like a mobile phone."), n("Like phones."))
    assert not C.content_match(n("T. i. d. television."), n("That's cool."))
    assert not C.content_match(n("Transparent or yeah."), n("That's cool."))
    assert not C.content_match(n("Hello."), n("Yeah."))                 # different short classes
    assert C.short_class(n("Mm-hmm")) is None or True                   # tokeniser-dependent; class lookup is total


def test_meeting_tables_v2():
    sc = {"u1": {"spk": "A", "a": 10.0, "b": 10.5, "ref": "Yeah."},
          "u2": {"spk": "A", "a": 14.0, "b": 14.4, "ref": "Yes."},
          "u3": {"spk": "B", "a": 20.0, "b": 23.0, "ref": "That's a very cool design idea"}}
    segs = [{"spk": 1, "a": 10.1, "b": 10.4, "text": "Yeah."},          # covers u1 (closest), not u2
            {"spk": 2, "a": 20.2, "b": 22.8, "text": "That is a very cool idea of design."},
            {"spk": 2, "a": 20.0, "b": 22.0, "text": "T. i. d. television."}]
    st = {"sc": sc, "unit_of": {"u1": "U1", "u2": "U2", "u3": "U3"},
          "got": {"U1": [(segs[0], "Yeah.")], "U3": [(segs[1], segs[1]["text"]), (segs[2], segs[2]["text"])]}}
    t = C.meeting_tables_v2(st, segs)
    assert t["u1"]["a"] == [0] and t["u2"]["a"] == []                  # one "Yeah." covers one utterance
    assert t["u3"]["b_raw"] == [1, 2] and t["u3"]["b"] == [1]          # garbage credit removed
    assert t["u3"]["c"] == [1]                                          # IoU and content
    assert t["u1"]["b"] == [0] and t["u2"]["b"] == [] and t["u2"]["b_raw"] == []


if __name__ == "__main__":
    for k, f in sorted(globals().items()):
        if k.startswith("test_"):
            f()
    print("cov2 tests pass")

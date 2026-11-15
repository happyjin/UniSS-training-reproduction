"""Unit tests for wallclock_v11 / timed_reads / timed_talker (CPU, fakes only)."""
import io
import json

import pytest

from experiments.mist_w3po_v1.evaluation import timed_reads as TR
from experiments.mist_w3po_v1.evaluation import timed_talker as TT
from experiments.mist_w3po_v1.evaluation import wallclock_v11 as WC


def test_read_due_clamps():
    w = {"start_s": 10.0, "dur_ms": 3000, "open_detect_ms": 1500, "close_detect_ms": 4200}
    assert WC.read_due(w, 1000) == pytest.approx(11.5)
    assert WC.read_due(w, 2000) == pytest.approx(12.0)
    assert WC.read_due(w, 3000) == pytest.approx(14.2)


def test_thinker_queue_backlog():
    reads = [{"key": ("a", t), "due": t / 1000, "ms": 1500.0} for t in (1000, 2000, 3000)]
    q = WC.thinker_queue(reads)
    assert q["commit"][("a", 1000)] == pytest.approx(2.5)
    assert q["commit"][("a", 2000)] == pytest.approx(4.0)
    assert q["commit"][("a", 3000)] == pytest.approx(5.5)
    s = q["summary"]
    assert s["rtf"] == pytest.approx(4.5 / 2.0)
    assert s["over_cadence"] == 1.0
    assert s["queue_wait_s"]["max"] == pytest.approx(1.0)


def blk(item, r, rel, dur, slot="S1", m="M"):
    return {"item": item, "r": r, "meeting": m, "slot": slot, "rel_ca": rel + 0.5, "rel_cf": rel, "dur_s": dur,
            "src_start": rel - 1, "src_end": rel}


def test_replay_layouts_and_dependency():
    bs = [blk("w__S1", 0, 1.0, 2.0, "S1"), blk("w__S2", 0, 1.0, 2.0, "S2"), blk("w__S1", 1, 1.2, 1.0, "S1")]
    cost = {("w__S1", 0): 1.0, ("w__S2", 0): 1.0, ("w__S1", 1): 0.5}
    ps = {(b["item"], b["r"]): b for b in WC.replay(bs, cost, "per-speaker")}
    # worker FIFO: S1/0 ready 2.5, S2/0 ready 3.5, S1/1 (released 1.7, waits for the worker) ready 4.0
    assert ps[("w__S1", 0)]["ready_ca"] == pytest.approx(2.5)
    assert ps[("w__S2", 0)]["ready_ca"] == pytest.approx(3.5)
    assert ps[("w__S1", 1)]["ready_ca"] == pytest.approx(4.0)
    assert ps[("w__S1", 1)]["start_ca"] == pytest.approx(4.5)   # after S1/0 ends (2.5 + 2.0)
    assert ps[("w__S2", 0)]["start_cf"] == pytest.approx(1.0)   # own channel
    sg = {(b["item"], b["r"]): b for b in WC.replay(bs, cost, "single")}
    assert sg[("w__S2", 0)]["start_cf"] == pytest.approx(3.0)   # one channel: after S1/0
    s = WC.summarise(WC.replay(bs, cost, "per-speaker"))
    assert s["ca_minus_cf_end"]["n"] == 3


def test_talker_costs_shares():
    timing = [{"kind": "gen", "r": 0, "owner": "a", "ms": 1000}, {"kind": "gen", "r": 0, "owner": "b", "ms": 3000},
              {"kind": "score", "r": 0, "n": 2, "ms": 400}, {"kind": "round", "r": 0, "n": 2, "ms": 4500},
              {"kind": "gen", "r": 0, "owner": None, "ms": 200}]
    blocks = [{"item": "a", "r": 0, "dur_s": 1.0}, {"item": "b", "r": 0, "dur_s": 3.0}]
    c = WC.talker_costs(timing, blocks, front_s=2.0, decode_s=4.0)
    assert c[("a", 0)] == pytest.approx(1.0 + 0.3 + 1.0 + 1.0)
    assert c[("b", 0)] == pytest.approx(3.0 + 0.3 + 1.0 + 3.0)


def test_timing_line():
    assert WC.parse_timing_line("x\ntiming front_end 12.5s  talker 30.0s  decode 4.0s  sentences 3\n") == (12.5, 4.0)


class Item:
    def __init__(self, k):
        self.akey = k


class Out:
    def __init__(self, n):
        self.ids = list(range(n))


def test_timed_read_records():
    fh = io.StringIO()
    clock = iter([0.0, 0.25])
    called = []
    f = TR.make_timed_read(lambda self, items, **k: [Out(3) for _ in items], fh, lambda: None,
                           clock=lambda: next(clock), reserve=lambda: called.append(1))
    outs = f(None, [Item("w|1000")], max_new=5)
    d = json.loads(fh.getvalue())
    assert d["akeys"] == ["w|1000"] and d["ms"] == pytest.approx(250.0) and d["n_new"] == [3]
    assert len(outs) == 1 and called == [1]


def test_split_args():
    o, rest = TR.split(["--target", "m", "--reads-out", "x.jsonl", "--keep-free-gib", "14", "--", "--batch", "1"])
    assert o == {"target": "m", "reads_out": "x.jsonl", "keep_free_gib": 14.0} and rest == ["--batch", "1"]


class Res:
    def __init__(self, n):
        self.outputs = [type("C", (), {"token_ids": list(range(n))})() for _ in range(2)]


def test_generate_split_and_round_tags():
    recs, calls = [], []

    def orig(self, ps, sp, *a, **k):
        calls.append((len(ps), sp))
        return [Res(5)]

    ctx = {}
    g = TT.wrap_generate(orig, ctx, recs.append, clock=iter([0, 1, 1, 3]).__next__)
    rnd = TT.wrap_round(lambda r, items: g(None, ["p1", "p2"], ["s1", "s2"]), ctx, recs.append,
                        clock=iter([0, 5]).__next__)
    out = rnd(0, [{"id": "a", "blocks": [1]}, {"id": "b", "blocks": [1]}, {"id": "c", "blocks": []}])
    assert len(out) == 2 and calls == [(1, ["s1"]), (1, ["s2"])]
    gens = [d for d in recs if d["kind"] == "gen"]
    assert [d["owner"] for d in gens] == ["a", "b"] and [d["ms"] for d in gens] == [1000.0, 2000.0]
    assert recs[-1] == {"kind": "round", "r": 0, "n": 2, "ms": 5000.0}

"""Unit tests for precheck_plcrf select (synthetic dump) and render_selection.timeline; CPU."""
import argparse
import json
import tempfile
from pathlib import Path

from experiments.mist_w3po_v1.talker import precheck_plcrf as P
from experiments.mist_w3po_v1.talker.render_selection import timeline
from experiments.mist_w3po_v1.talker.test_talker_rows import dump_item


def test_timeline():
    bl = [{"write_ms": 1000}, {"write_ms": 1200}, {"write_ms": 5000}]
    assert timeline(bl, [500, 300, 100]) == [(1000, 500), (1500, 300), (5000, 100)]


def test_select():
    with tempfile.TemporaryDirectory() as d:
        items = []
        for i in range(5):
            it = dump_item()
            it["id"] = f"M_on_000{i}__S1"
            items.append(it)
        (Path(d) / "cands_0.jsonl").write_text("".join(json.dumps(x) + "\n" for x in items))
        w = [{"id": f"M_on_000{i}", "open_detect_ms": 900, "close_detect_ms": 1600, "dur_ms": 1500} for i in range(5)]
        (Path(d) / "win.jsonl").write_text("".join(json.dumps(x) + "\n" for x in w))
        P.select(argparse.Namespace(cands=f"{d}/cands_*.jsonl", win=f"{d}/win.jsonl", out=f"{d}/o"))
        sl = json.load(open(f"{d}/o/sel_lattice.json"))
        sc = json.load(open(f"{d}/o/sel_cover.json"))
        st = json.load(open(f"{d}/o/select_stats.json"))
        assert set(sl) == set(sc) == {x["id"] for x in items}
        assert all(len(v) == 2 for v in sl.values()) and sc["M_on_0000__S1"] == [1, 0]
        assert st["viterbi_latency_s"] <= st["aprule_latency_s"] + 1e-9 or st["lambda"] == 0
        assert "p3_go" in st


if __name__ == "__main__":
    test_timeline()
    test_select()
    print("ok")

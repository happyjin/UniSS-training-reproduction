"""Unit test for check_shared_bins: a synthetic meeting-scope run whose tokens follow the shared channel matches; a
corrupted token is counted as a mismatch."""
import json
import tempfile
from pathlib import Path

from experiments.mist_w3po_v1.talker import check_shared_bins as C
from experiments.mist_w3po_v1.talker import shared_channel as SC


def test_check():
    W = {"M_on_0000": {"id": "M_on_0000", "meeting": "M", "start_s": 10.0, "open_detect_ms": 900, "close_detect_ms": 1600,
                       "dur_ms": 1500},
         "M_on_0001": {"id": "M_on_0001", "meeting": "M", "start_s": 11.0, "open_detect_ms": 300, "close_detect_ms": 900,
                       "dur_ms": 800}}
    items = [{"id": "M_on_0000__S1", "blocks": [{"write_ms": 1000, "codes": 150}, {"write_ms": 1200, "codes": 40}]},
             {"id": "M_on_0000__S2", "blocks": [{"write_ms": 1100, "codes": 60}]},
             {"id": "M_on_0001__S1", "blocks": [{"write_ms": 400, "codes": 20}]}]
    durs = {(n, i): b["codes"] * 20 for n, it in enumerate(items) for i, b in enumerate(it["blocks"])}
    bl = SC.shared_backlogs(items, W, durs)
    for (n, i), v in bl.items():
        b = items[n]["blocks"][i]
        b.update(dur_ms=b["codes"] * 20, backlog_bin=SC.backlog_bin(v), backlog_ms_shared=round(v, 1))
    assert max(bl.values()) > 500   # the shared channel queues
    with tempfile.TemporaryDirectory() as d:
        (Path(d) / "win.jsonl").write_text("".join(json.dumps(w) + "\n" for w in W.values()))
        (Path(d) / "part_0.jsonl").write_text("".join(json.dumps(it) + "\n" for it in items[:2]))
        (Path(d) / "part_1.jsonl").write_text("".join(json.dumps(it) + "\n" for it in items[2:]))
        r = C.check(d, f"{d}/win.jsonl")
        assert r["blocks"] == 4 and r["codes_x20"]["bin_mismatch"] == 0 and r["dur_ms"]["bin_match"] == 4
        assert r["codes_x20"]["max_abs_logged_minus_recomputed_ms"] < 0.1
        items[2]["blocks"][0]["backlog_bin"] = (items[2]["blocks"][0]["backlog_bin"] + 1) % 8
        (Path(d) / "part_1.jsonl").write_text("".join(json.dumps(it) + "\n" for it in items[2:]))
        assert C.check(d, f"{d}/win.jsonl")["codes_x20"]["bin_mismatch"] == 1


if __name__ == "__main__":
    test_check()
    print("ok")

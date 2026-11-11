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


def test_select_meeting_and_summary2():
    """talker_crf_prereg_v2 experiment 2: the meeting-lattice Viterbi selection and the GO line."""
    from experiments.mist_w3po_v1.talker.test_talker_rows import meeting_items

    with tempfile.TemporaryDirectory() as d:
        items, W = meeting_items()
        (Path(d) / "cands_0.jsonl").write_text("".join(json.dumps(x) + "\n" for x in items))
        (Path(d) / "win.jsonl").write_text("".join(json.dumps({"id": k, **v}) + "\n" for k, v in W.items()))
        ss = {it["id"]: [b["chosen"] for b in it["blocks"]] for it in items}
        (Path(d) / "ss.json").write_text(json.dumps(ss))
        P.select_meeting(argparse.Namespace(cands=f"{d}/cands_*.jsonl", win=f"{d}/win.jsonl", out=f"{d}/o",
                                            sel_stream=f"{d}/ss.json"))
        sel = json.load(open(f"{d}/o/sel_meeting.json"))
        st = json.load(open(f"{d}/o/select_meeting_stats.json"))
        assert {k: len(v) for k, v in sel.items()} == {it["id"]: len(it["blocks"]) for it in items}
        assert abs(st["stream_lattice_latency_1ch_s"] - st["aprule_latency_1ch_s"]) < 1e-9   # ss = the A' path here
        assert st["viterbi_latency_1ch_s"] <= st["aprule_latency_1ch_s"] + 1e-9 or st["lambda"] == 0
        # summary2's GO line on made-up numbers
        rows = {"aprule": (23.4, 7.8), "lattice": (23.8, 5.5), "meeting": (23.3, 4.9)}
        args = {"out": d, "gate_meeting": ""}
        for k, (sp, l1) in rows.items():
            D = Path(d) / k
            D.mkdir()
            (D / "score_asr.json").write_text(json.dumps({"tcpBLEU_all": sp}))
            lat = {"unit": {"speech_laal_online_1ch": {"all": {"mean": l1}}, "speech_end_online_1ch": {"all": {"mean": l1}},
                            "speech_laal_online": {"all": {"mean": 4.0}}, "speech_end_online": {"all": {"mean": 3.0}}}}
            gap = {"scores": {"asr_seg": {"repo": {"all": 28.0}}, "text_seg": {"raw": {"all": 28.4}, "repo": {"all": 29.7}}},
                   "fidelity": {"wer": 0.05}}
            (D / "lat.json").write_text(json.dumps(lat))
            (D / "gap.json").write_text(json.dumps(gap))
            args.update({f"dec_{k}": str(D), f"lat_{k}": str(D / "lat.json"), f"gap_{k}": str(D / "gap.json")})
        P.summary2(argparse.Namespace(**args))
        res = json.load(open(f"{d}/summary2.json"))
        assert res["exp2_go"] is True and abs(res["meeting_minus"]["lattice"]["speech_laal_1ch"] + 0.6) < 1e-9


def test_select_meeting_budget_ref():
    """talker_crf_prereg_v3 (E2'): --budget-ref stream_viterbi takes the --sel-stream path's proxy - --drop-1ch."""
    from experiments.mist_w3po_v1.talker import talker_rows as TR
    from experiments.mist_w3po_v1.talker.test_talker_rows import meeting_items

    with tempfile.TemporaryDirectory() as d:
        items, W = meeting_items()
        (Path(d) / "cands_0.jsonl").write_text("".join(json.dumps(x) + "\n" for x in items))
        (Path(d) / "win.jsonl").write_text("".join(json.dumps({"id": k, **v}) + "\n" for k, v in W.items()))
        ss = {it["id"]: [0 for _ in it["blocks"]] for it in items}
        (Path(d) / "ss.json").write_text(json.dumps(ss))
        P.select_meeting(argparse.Namespace(cands=f"{d}/cands_*.jsonl", win=f"{d}/win.jsonl", out=f"{d}/o",
                                            sel_stream=f"{d}/ss.json", budget_ref="stream_viterbi", drop_1ch=0.5))
        st = json.load(open(f"{d}/o/select_meeting_stats.json"))
        assert st["budget_ref"] == "stream_viterbi" and st["ref_latency_1ch_s"] == st["stream_lattice_latency_1ch_s"]
        assert abs(st["budget_s"] - round(st["ref_latency_1ch_s"] - 0.5, 4)) < 1e-9
        _, _, lam, st2 = TR.meeting_lattice(items, W, drop=0.5, ref_paths=ss)
        assert lam == st["lambda"]
        # defaults (no budget_ref / drop_1ch attributes, as v2's orchestrator calls it) are unchanged
        P.select_meeting(argparse.Namespace(cands=f"{d}/cands_*.jsonl", win=f"{d}/win.jsonl", out=f"{d}/o2",
                                            sel_stream=f"{d}/ss.json"))
        P.select_meeting(argparse.Namespace(cands=f"{d}/cands_*.jsonl", win=f"{d}/win.jsonl", out=f"{d}/o3",
                                            sel_stream=f"{d}/ss.json", budget_ref="aprule", drop_1ch=TR.BUDGET_DROP_1CH_S))
        for f in ("sel_meeting.json", "select_meeting_stats.json"):
            assert (Path(d) / "o2" / f).read_text() == (Path(d) / "o3" / f).read_text()
        assert "budget_ref" not in json.load(open(f"{d}/o2/select_meeting_stats.json"))


if __name__ == "__main__":
    test_timeline()
    test_select()
    test_select_meeting_and_summary2()
    test_select_meeting_budget_ref()
    print("ok")

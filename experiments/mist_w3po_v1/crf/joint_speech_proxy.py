"""J1's CPU speech-latency proxy (reports/mist_w3po_v1/thinker_joint_crf_prereg_v1.zh-CN.md §3 rule 2, §5 PC2, §6 line 4).

A dev decode's talker is simulated with J1's talker model (T3's crf/t3_talker_model_v1.json: D(B, v) = a_v chars + c_v,
pi(v | backlog bin)) instead of synthesising speech:
  blocks     per window x speaker stream, the non-empty hyp blocks the talker prep would play (gstar_talker_v6 prep
             --blocks: per speaker, ordered by (write_ms, start_s)), optionally after the read-level merge
             (crf/merge_read_segments.py) -- the J1 / D1 talker input
  durations  the model's expected duration sum_v pi(v | bin(b')) D(B, v) at the model's own backlog b' (online
             availability, queue end on the absolute 80 ms grid): T3's A1+A2+A4 construction (joint_speech_checks)
  metrics    speech_latency_lit, unchanged, on (the decode or its merged copy, the simulated talker dir)
PC2 (model_durations_on_talker): the same model durations put on an ACTUAL talker run's blocks, against the actual run.

    python -m experiments.mist_w3po_v1.crf.joint_speech_proxy --run <decode dir> --win <windows> --merge 0|1 \
        --out <new json> [--talker-model <json>]
Refuses to overwrite; the scratch dir is <out>.work (new).
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from experiments.mist_w3po_v1.crf import joint_speech_checks as JC
from experiments.mist_w3po_v1.crf import joint_speech_crf as JS
from experiments.mist_w3po_v1.crf import merge_read_segments as MR

TALKER_MODEL = "eval_outputs/mist_w3po_v1/crf/t3_talker_model_v1.json"


def stream_blocks(h: dict, w: dict) -> dict:
    """{spk: [{"text", "write_ms", "start_s"}]} as the talker prep builds them (non-empty hyp blocks, or the segments
    when the record has no blocks; per speaker ordered by (write_ms, start_s), stable)."""
    by = defaultdict(list)
    src = h.get("blocks") or h.get("segments", [])
    for b in src:
        if b["text"].strip():
            by[b["spk"]].append({"text": b["text"], "write_ms": int(b.get("write_ms", w["dur_ms"])),
                                 "start_s": b.get("start_s") or 0.0})
    for k in by:
        by[k].sort(key=lambda b: (b["write_ms"], b["start_s"]))
    return by


def simulate(blocks: list[dict], w: dict, tm: JS.TalkerModel) -> list[dict]:
    """per block of one stream: dur_ms from the model at its own backlog (grid queue end), start_ms = when it starts
    on the online queue (window-relative)."""
    out, end = [], None
    for b in blocks:
        a = JC.online(w, b["write_ms"])
        bp = 0.0 if end is None else max(0.0, end - a)
        d = JC.exp_dur(tm, b["text"], bp)
        e2 = JC.grid_end(a, bp, d)
        out.append({"write_ms": b["write_ms"], "text": b["text"], "start_ms": a + bp, "dur_ms": max(0.0, e2 - (a + bp))})
        end = e2
    return out


def proxy_rows(hyps: list[dict], wins: dict, tm: JS.TalkerModel) -> list[dict]:
    rows = []
    for h in hyps:
        w = wins.get(h["id"])
        if w is None:
            continue
        for k, bl in sorted(stream_blocks(h, w).items()):
            B = simulate(bl, w, tm)
            rows.append({"id": f"{h['id']}__S{k}", "error": None, "blocks": B,
                         "start_offset_ms": B[0]["start_ms"],
                         "end_offset_ms": B[-1]["start_ms"] + B[-1]["dur_ms"] - w["dur_ms"]})
    return rows


def model_durations_on_talker(talker_dir: str, wins: dict, tm: JS.TalkerModel) -> list[dict]:
    """PC2: an actual talker run's results with every block's dur_ms replaced by the model's (T3's check_a124)."""
    rows = []
    for l in open(Path(talker_dir) / "results.jsonl", encoding="utf-8"):
        r = json.loads(l)
        if not r.get("error") and r["blocks"]:
            w = wins[r["id"].split("__S")[0]]
            end = None
            for b in r["blocks"]:
                a = JC.online(w, b["write_ms"])
                bp = 0.0 if end is None else max(0.0, end - a)
                d = JC.exp_dur(tm, b["text"], bp)
                e2 = JC.grid_end(a, bp, d)
                b["dur_ms"] = max(0.0, e2 - (a + bp))
                end = e2
        rows.append(r)
    return rows


def write_rows(rows: list[dict], d: Path) -> None:
    d.mkdir(parents=True)
    with open(d / "results.jsonl", "x", encoding="utf-8") as fh:
        fh.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)


def lat(run: str, talker: str, win: str, out: str) -> dict:
    """speech_latency_lit.main on (run, talker) -> its json (speech LAAL etc.)."""
    from experiments.mist_w3po_v1.evaluation import speech_latency_lit as SL

    argv = sys.argv
    sys.argv = ["speech_latency_lit", "--run", run, "--talker", talker, "--out", out, "--win", win]
    try:
        SL.main()
    finally:
        sys.argv = argv
    return json.loads(Path(out).read_text())


def summary(res: dict, rows: list[dict]) -> dict:
    u = res["unit"]
    nb = [len(b["text"].split()) for r in rows for b in r.get("blocks", []) if b["text"].strip()]
    return {"speech_laal": u["speech_laal_online"]["all"]["mean"], "speech_laal_ov": u["speech_laal_online"]["ov"]["mean"],
            "speech_laal_1spk": u["speech_laal_online"]["1spk"]["mean"], "text_laal": u["text_laal_online"]["all"]["mean"],
            "end_offset": u["speech_end_online"]["all"]["mean"],
            "speech_laal_1ch": u.get("speech_laal_online_1ch", {}).get("all", {}).get("mean"),
            "queue_wait": res["unit_end_parts_online"]["queue_wait"]["mean"], "units": u["speech_laal_online"]["all"]["n"],
            "blocks": len(nb), "words_per_block": round(float(np.mean(nb)), 3) if nb else None,
            "segment_block_mismatch": res["segment_block_mismatch"]}


def run_proxy(run: str, win: str, merge: bool, out: str, tm_path: str = TALKER_MODEL) -> dict:
    """the proxy of one decode; scratch in <out>.work (new). Returns the summary (also written to <out>)."""
    o = Path(out)
    if o.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {o}")
    work = Path(str(o) + ".work")
    work.mkdir(parents=True)
    tm = JS.TalkerModel.load(tm_path)
    wins = {w["id"]: w for w in map(json.loads, open(win, encoding="utf-8"))}
    hyps = [json.loads(l) for l in open(Path(run) / "hyp.jsonl", encoding="utf-8")]
    run_used = run
    if merge:
        hyps = [MR.merge_hyp(h)[0] for h in hyps]
        rd = work / "decode_rl"
        rd.mkdir()
        with open(rd / "hyp.jsonl", "x", encoding="utf-8") as fh:
            fh.writelines(json.dumps(h, ensure_ascii=False) + "\n" for h in hyps)
        run_used = str(rd)
    rows = proxy_rows(hyps, wins, tm)
    write_rows(rows, work / "talker")
    res = lat(run_used, str(work / "talker"), win, str(work / "lat.json"))
    s = {"run": run, "win": win, "merge": bool(merge), "talker_model": tm_path, **summary(res, rows)}
    with open(o, "x", encoding="utf-8") as fh:
        json.dump(s, fh, indent=1)
    return s


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--win", required=True)
    ap.add_argument("--merge", type=int, choices=[0, 1], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--talker-model", default=TALKER_MODEL)
    a = ap.parse_args()
    print(json.dumps(run_proxy(a.run, a.win, bool(a.merge), a.out, a.talker_model)))


if __name__ == "__main__":
    main()

"""Prechecks of the two talker schemes on the dev5 sampling run (reports/mist_w3po_v1/talker_crf_prereg_v1.zh-CN.md
§1.2, §2.2). CPU only.

select   from the run's cands_*.jsonl: the PL-EM lattice (budget = the A' rule path's corpus latency - 0.3 s, lambda by
         the dual bisection, Viterbi path per stream) -> sel_lattice.json; the scheme-2 cover pick per block ->
         sel_cover.json; P3 (block-mean KL of the candidate marginal to uniform) and lattice stats -> select_stats.json.
summary  the speech scores / latencies / gap decompositions of the three paths (A' rule = the run itself, lattice, cover)
         -> GO / NOT GO per the registered lines -> summary.json.

    python -m experiments.mist_w3po_v1.talker.precheck_plcrf select --cands '<talker>/cands_*.jsonl' --win <windows> --out <dir>
    python -m experiments.mist_w3po_v1.talker.precheck_plcrf summary --out <dir> --dec-aprule D --dec-lattice D --dec-cover D \
        --lat-aprule J --lat-lattice J --lat-cover J --gap-aprule J --gap-lattice J --gap-cover J --gate-lattice J --gate-cover J

talker_crf_prereg_v2 experiment 2 (one shared output channel per meeting), additive:
select-meeting  the meeting lattice (talker_rows.meeting_lattice: one chain per meeting, FIFO, shared backlog, cap 32 s,
                nearest rounding, budget = the A' rule path's shared-channel corpus latency - 1.0 s) -> Viterbi path per
                meeting -> sel_meeting.json (id -> candidate per block), select_meeting_stats.json (with the A' path's
                and, given --sel-stream, the per-stream lattice path's proxy latency on the meeting lattice)
summary2        A' rule path vs per-stream lattice path vs meeting-lattice path: 1-channel speech LAAL (FIFO; primary),
                per-stream speech LAAL, EndOffset (both), speech official / normalised segment-level, WER -> GO iff
                1ch LAAL(meeting) <= 1ch LAAL(per-stream lattice) - 0.5 s and speech official(meeting) >= A' - 0.2
                -> summary2.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.mist_w3po_v1.talker import talker_rows as TR


def select(a) -> None:
    out = Path(a.out)
    if (out / "sel_lattice.json").exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    out.mkdir(parents=True, exist_ok=True)
    items = TR.load_items(a.cands)
    W = TR.windows(a.win)
    streams, lam, st = TR.lattice(items, W)
    sel_l, sel_c, kl_u, ent, changed, vit_lat = {}, {}, [], [], 0, 0.0
    for it, s in zip(items, streams):
        _, marg, _, _ = s.forward_backward(lam)
        for r, b in enumerate(it["blocks"]):
            qm = marg[r].sum(0)
            qm = qm / qm.sum()
            kl_u.append(float(np.sum(qm * (np.log(np.maximum(qm, 1e-300)) + np.log(len(qm))))))
            ent.append(float(-np.sum(qm * np.log(np.maximum(qm, 1e-300)))))
        path = s.viterbi(lam)
        vit_lat += s.path_latency(path)
        sel_l[it["id"]] = path
        changed += sum(p != b["chosen"] for p, b in zip(path, it["blocks"]))
        sel_c[it["id"]] = [TR.cover_pick(b) if TR.cover_pick(b) is not None else b["chosen"] for b in it["blocks"]]
    words = st["words"]
    nb = sum(len(it["blocks"]) for it in items)
    cov_full = [sum(1 for c in b["cands"] if c["o"] == 0 and c["cover"] is not None and c["cover"] >= 1.0) > 0
                for it in items for b in it["blocks"]]
    stats = {**st, "viterbi_latency_s": round(vit_lat / words, 4), "blocks": nb,
             "lattice_changed_blocks": changed, "cover_changed_blocks":
                 sum(c != b["chosen"] for it in items for c, b in zip(sel_c[it["id"]], it["blocks"])),
             "blocks_with_full_cover_cand": int(sum(cov_full)),
             "p3_kl_uniform_mean": round(float(np.mean(kl_u)), 4), "posterior_entropy_mean": round(float(np.mean(ent)), 4),
             "p3_go": bool(np.mean(kl_u) >= 0.1)}
    (out / "sel_lattice.json").write_text(json.dumps(sel_l) + "\n")
    (out / "sel_cover.json").write_text(json.dumps(sel_c) + "\n")
    (out / "select_stats.json").write_text(json.dumps(stats, indent=1) + "\n")
    print(json.dumps(stats))


def summary(a) -> None:
    out = Path(a.out) / "summary.json"
    if out.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    rows = {}
    for k in ("aprule", "lattice", "cover"):
        sa = json.load(open(Path(getattr(a, f"dec_{k}")) / "score_asr.json"))
        lat = json.load(open(getattr(a, f"lat_{k}")))
        gap = json.load(open(getattr(a, f"gap_{k}")))
        rows[k] = {"speech_official": _all(sa), "speech_norm_seg": gap["scores"]["asr_seg"]["repo"]["all"],
                   "text_official": gap["scores"]["text_seg"]["raw"]["all"], "text_norm_seg": gap["scores"]["text_seg"]["repo"]["all"],
                   "speech_laal": _laal(lat), "end_offset": _endoff(lat), "talker_wer": gap["fidelity"]["wer"]}
    ref = rows["aprule"]
    d = {k: {m: round(rows[k][m] - ref[m], 3) for m in ("speech_official", "speech_norm_seg", "speech_laal", "end_offset")}
         for k in ("lattice", "cover")}
    gl = json.load(open(a.gate_lattice)) if a.gate_lattice else {}
    gc = json.load(open(a.gate_cover)) if a.gate_cover else {}
    dl, dc = d["lattice"], d["cover"]
    p1 = ((dl["speech_laal"] <= -0.25 and dl["speech_official"] >= -0.2)
          or (dl["speech_laal"] <= 0.05 and dl["speech_norm_seg"] >= 0.4))
    s2 = dc["speech_norm_seg"] >= 0.3 and dc["speech_laal"] <= 0.05
    res = {"rows": rows, "diff_vs_aprule": d, "gate_lattice": gl, "gate_cover": gc, "P1_go": bool(p1), "scheme2_go": bool(s2)}
    out.write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res))


def select_meeting(a) -> None:
    out = Path(a.out)
    if (out / "sel_meeting.json").exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    out.mkdir(parents=True, exist_ok=True)
    items = TR.load_items(a.cands)
    W = TR.windows(a.win)
    chains, streams, lam, st = TR.meeting_lattice(items, W)
    sel = {it["id"]: [None] * len(it["blocks"]) for it in items}
    vit_lat, changed = 0.0, 0
    for ch, s in zip(chains.values(), streams):
        path = s.viterbi(lam)
        vit_lat += s.path_latency(path)
        for (n, r), c in zip(ch, path):
            sel[items[n]["id"]][r] = int(c)
            changed += int(c) != items[n]["blocks"][r]["chosen"]
    assert all(c is not None for v in sel.values() for c in v)
    words = st["words"]
    stats = {**st, "viterbi_latency_1ch_s": round(vit_lat / words, 4), "blocks": sum(len(c) for c in chains.values()),
             "meeting_changed_blocks": changed}
    if a.sel_stream:   # the per-stream lattice path (v1 precheck) on the meeting lattice's proxy
        ss = json.load(open(a.sel_stream))
        stats["stream_lattice_latency_1ch_s"] = round(sum(
            s.path_latency([ss[items[n]["id"]][r] for n, r in ch]) for ch, s in zip(chains.values(), streams)) / words, 4)
        stats["blocks_differing_from_stream_lattice"] = sum(
            int(sel[k][r] != ss[k][r]) for k in sel for r in range(len(sel[k])))
    (out / "sel_meeting.json").write_text(json.dumps(sel) + "\n")
    (out / "select_meeting_stats.json").write_text(json.dumps(stats, indent=1) + "\n")
    print(json.dumps(stats))


GO_1CH_DROP_S = 0.5
GO_SPEECH_TOL = 0.2


def summary2(a) -> None:
    out = Path(a.out) / "summary2.json"
    if out.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    rows = {}
    for k in ("aprule", "lattice", "meeting"):
        sa = json.load(open(Path(getattr(a, f"dec_{k}")) / "score_asr.json"))
        lat = json.load(open(getattr(a, f"lat_{k}")))
        gap = json.load(open(getattr(a, f"gap_{k}")))
        rows[k] = {"speech_official": _all(sa), "speech_norm_seg": gap["scores"]["asr_seg"]["repo"]["all"],
                   "text_official": gap["scores"]["text_seg"]["raw"]["all"], "text_norm_seg": gap["scores"]["text_seg"]["repo"]["all"],
                   "speech_laal_1ch": lat["unit"]["speech_laal_online_1ch"]["all"]["mean"],
                   "end_offset_1ch": lat["unit"]["speech_end_online_1ch"]["all"]["mean"],
                   "speech_laal": _laal(lat), "end_offset": _endoff(lat), "talker_wer": gap["fidelity"]["wer"]}
    ms = ("speech_official", "speech_norm_seg", "speech_laal_1ch", "end_offset_1ch", "speech_laal", "end_offset", "talker_wer")
    d = {ref: {m: round(rows["meeting"][m] - rows[ref][m], 4) for m in ms} for ref in ("aprule", "lattice")}
    go = (rows["meeting"]["speech_laal_1ch"] <= rows["lattice"]["speech_laal_1ch"] - GO_1CH_DROP_S
          and rows["meeting"]["speech_official"] >= rows["aprule"]["speech_official"] - GO_SPEECH_TOL)
    res = {"rows": rows, "meeting_minus": d, "gate_meeting_vs_aprule": json.load(open(a.gate_meeting)) if a.gate_meeting else {},
           "go_line": f"1ch LAAL(meeting) <= 1ch LAAL(lattice) - {GO_1CH_DROP_S} and speech official(meeting) >= "
                      f"A' - {GO_SPEECH_TOL}", "exp2_go": bool(go)}
    out.write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res))


def _all(sa):
    for k in ("tcpBLEU_all", "all"):
        if k in sa:
            return sa[k]
    return sa["headline"]["tcpBLEU_all"]


def _laal(lat):
    """speech_latency_lit: per-unit speech LAAL on the online queue, all units."""
    return lat["unit"]["speech_laal_online"]["all"]["mean"]


def _endoff(lat):
    return lat["unit"]["speech_end_online"]["all"]["mean"]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("select")
    s.add_argument("--cands", required=True)
    s.add_argument("--win", required=True)
    s.add_argument("--out", required=True)
    m = sub.add_parser("summary")
    m.add_argument("--out", required=True)
    for k in ("aprule", "lattice", "cover"):
        m.add_argument(f"--dec-{k}", required=True)
        m.add_argument(f"--lat-{k}", required=True)
        m.add_argument(f"--gap-{k}", required=True)
    m.add_argument("--gate-lattice", default="")
    m.add_argument("--gate-cover", default="")
    s2 = sub.add_parser("select-meeting")
    s2.add_argument("--cands", required=True)
    s2.add_argument("--win", required=True)
    s2.add_argument("--out", required=True)
    s2.add_argument("--sel-stream", default="", help="sel_lattice.json of the per-stream lattice (descriptive)")
    m2 = sub.add_parser("summary2")
    m2.add_argument("--out", required=True)
    for k in ("aprule", "lattice", "meeting"):
        m2.add_argument(f"--dec-{k}", required=True)
        m2.add_argument(f"--lat-{k}", required=True)
        m2.add_argument(f"--gap-{k}", required=True)
    m2.add_argument("--gate-meeting", default="")
    a = ap.parse_args()
    {"select": select, "summary": summary, "select-meeting": select_meeting, "summary2": summary2}[a.cmd](a)


if __name__ == "__main__":
    main()

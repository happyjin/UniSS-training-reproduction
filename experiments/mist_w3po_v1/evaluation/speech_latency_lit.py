"""Speech latency of a MIST dev talker run under the literature's definitions (CPU; reports/mist_w3po_v1/
text_to_speech_gap_plan.zh-CN.md §3).

Talker blocks carry window-relative ms: write_ms (the Thinker's commit), start_ms / dur_ms (where the talker laid the
block: start = max(write_ms, previous block's end) -- SimulEval's playback queue, no compute time). Variants:
  recorded     the talker's own timeline (write_ms NOT clamped to the window detector)
  online       each block becomes available at the online-protocol commit time of overlap_score (>= open_detect_ms;
               a final-read block >= close_detect_ms), then the same per-stream queue
  online_1ch   online, and all streams of a meeting share ONE output channel (one interpreter voice, Hibiki / Seamless
               style): a block starts at max(available, end of whatever was playing)
Reference points:
  stream       per window x speaker stream: StartOffset = first speech start - window start; EndOffset = last speech
               end - window end (SimulEval with the window as the source; = the talker's start/end_offset_ms)
  unit         per scored reference unit (utterances of a speaker < 1 s apart, as tcpBLEU): the blocks of the text
               segments the official assignment gives it; StartOffset = first block start - unit start, EndOffset =
               last block end - unit end (Hibiki's End Offset is per utterance: end of the last output word - end of
               the last source word)
Per unit also LAAL (Papi et al. 2022; metric_headroom's text form): text_laal_online (word delay = online commit -
unit start) and speech_laal_online (word delay = when the word has been spoken on the online queue - unit start, a
block's words spread evenly over its playback). EndOffset (unit, online) is split into: model lag (raw commit - unit end), detector wait (online commit - raw commit),
queue wait (start - online commit), and the last block's duration. Means (the literature's aggregate, SimulEval
mean()) and p50 / p90. No computation-aware number: the decode is batched (see the plan).
Writes <out> (json; refuses to overwrite). Dev only.

    python -m experiments.mist_w3po_v1.evaluation.speech_latency_lit --run <decode dir> --talker <talker dir> --out <json> [--win <windows jsonl>]
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from experiments.mist_w3po_v1.evaluation import metric_headroom as M


def q(a):
    a = np.asarray(a, dtype=float)
    if not len(a):
        return None
    return {"mean": round(float(a.mean()), 3), "p50": round(float(np.median(a)), 3), "p90": round(float(np.percentile(a, 90)), 3),
            "n": int(len(a))}


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True)
    ap.add_argument("--talker", required=True)
    ap.add_argument("--out", required=True)
    # the online windows the run was decoded on (default = metric_headroom.WIN, unchanged; C2 runs pass their own,
    # e.g. windows_zh2en_c2g05_v1). The reference windows (scoring) stay metric_headroom.REF_WIN.
    ap.add_argument("--win", default=M.WIN)
    return ap


def laal(d: list, u: dict) -> float:
    """LAAL (Papi et al. 2022) of one scored reference unit: d = per output word delay from the unit start (s, any
    order), |X| = unit length, |Y| = max(output words, reference words), tau = first word at or after the unit end."""
    d = sorted(d)
    X = max(u["b"] - u["a"], 1e-3)
    Y = max(len(d), len(u["ref"].split()))
    tau = next((i for i, di in enumerate(d) if di >= X), len(d) - 1)
    return float(np.mean([d[i] - i * X / Y for i in range(tau + 1)]))


def speech_word_delays(pw: list, a: float) -> list:
    """per output word of the unit's assigned pieces: when it has been SPOKEN (online queue timeline, no compute
    time) - unit start. A block's words are spread evenly over its playback [on_start, on_end] (no word timestamps);
    a piece (the part of a segment the assignment gave this unit) takes its words' positions inside the block."""
    out = []
    for x, t in pw:
        b = x.get("blk")
        if b is None:
            continue
        bw, tw = b["text"].split(), t.split()
        if not bw or not tw:
            continue
        k0 = len(x["text"][: max(x["text"].find(t.strip()), 0)].split())
        for j in range(len(tw)):
            k = min(k0 + j, len(bw) - 1)
            out.append(b["on_start"] + (k + 1) / len(bw) * (b["on_end"] - b["on_start"]) - a)
    return out


def main() -> None:
    args = build_parser().parse_args()
    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    from meeting_score import universe
    W = [json.loads(l) for l in open(M.REPO / args.win, encoding="utf-8")]
    wins = {w["id"]: w for w in W}
    H = {h["id"]: h for h in map(json.loads, open(Path(args.run) / "hyp.jsonl", encoding="utf-8"))}
    RW = [json.loads(l) for l in open(M.REPO / M.REF_WIN, encoding="utf-8")]
    U = universe(M.DR, {w["meeting"] for w in W})
    sc = M.scored_utts(W, RW)
    rows = {}
    for l in open(Path(args.talker) / "results.jsonl", encoding="utf-8"):
        r = json.loads(l)
        if not r.get("error"):
            rows[r["id"]] = r

    def online(w, wm):
        c = max(wm, w["open_detect_ms"])
        if wm >= w["dur_ms"]:
            c = max(c, w["close_detect_ms"])
        return c

    # per block absolute times (s) under the variants
    blk = {}   # (stream id, i) -> dict
    chk = {"recorded_start_matches": 0, "recorded_end_matches": 0, "streams": 0}
    stream = defaultdict(lambda: defaultdict(list))
    for k, r in rows.items():
        wid = k.split("__S")[0]
        w = wins[wid]
        t0 = w["start_s"]
        prev = None
        bl = r["blocks"]
        for i, b in enumerate(bl):
            av = online(w, b["write_ms"])
            st = av if prev is None else max(av, prev)
            prev = st + b["dur_ms"]
            blk[(k, i)] = {"meeting": w["meeting"], "write": t0 + b["write_ms"] / 1000, "avail": t0 + av / 1000,
                           "rec_start": t0 + b["start_ms"] / 1000, "rec_end": t0 + (b["start_ms"] + b["dur_ms"]) / 1000,
                           "on_start": t0 + st / 1000, "on_end": t0 + prev / 1000, "dur": b["dur_ms"] / 1000,
                           "final_read": b["write_ms"] >= w["dur_ms"], "text": b["text"]}
        if not bl:
            continue
        chk["streams"] += 1
        chk["recorded_start_matches"] += bl[0]["start_ms"] == r["start_offset_ms"]
        chk["recorded_end_matches"] += (bl[-1]["start_ms"] + bl[-1]["dur_ms"] - w["dur_ms"]) == r["end_offset_ms"]
        B = [blk[(k, i)] for i in range(len(bl))]
        we = t0 + w["dur_ms"] / 1000
        stream["recorded"]["start"].append(B[0]["rec_start"] - t0)
        stream["recorded"]["end"].append(B[-1]["rec_end"] - we)
        stream["online"]["start"].append(B[0]["on_start"] - t0)
        stream["online"]["end"].append(B[-1]["on_end"] - we)
    # one output channel per meeting
    bym = defaultdict(list)
    for kk, b in blk.items():
        bym[b["meeting"]].append((b["avail"], kk))
    for m, L in bym.items():
        prev = -1e9
        # a stream's own blocks stay in order: sort by availability, ties by stream order
        for av, kk in sorted(L, key=lambda x: (x[0], x[1][0], x[1][1])):
            b = blk[kk]
            b["ch_start"] = max(av, prev)
            prev = b["ch_start"] + b["dur"]
            b["ch_end"] = prev
    for k, r in rows.items():
        bl = r["blocks"]
        if not bl:
            continue
        w = wins[k.split("__S")[0]]
        B = [blk[(k, i)] for i in range(len(bl))]
        stream["online_1ch"]["start"].append(min(b["ch_start"] for b in B) - w["start_s"])
        stream["online_1ch"]["end"].append(max(b["ch_end"] for b in B) - w["start_s"] - w["dur_ms"] / 1000)

    # unit level: text segments -> blocks (by order among the speaker's non-empty segments), then the assignment
    segs = defaultdict(list)
    miss = 0
    for w in W:
        h = H.get(w["id"])
        if h is None:
            continue
        idx = defaultdict(int)
        for s in h["segments"]:
            if not s["text"].strip():
                continue
            k = f"{w['id']}__S{s['spk']}"
            i = idx[s["spk"]]
            idx[s["spk"]] += 1
            b = blk.get((k, i))
            if b is None or b["text"].strip() != s["text"].strip():
                miss += 1
                b = None
            c = online(w, s["write_ms"])
            segs[w["meeting"]].append({"spk": s["spk"], "a": w["start_s"] + s["start_s"],
                                       "b": w["start_s"] + max(s["end_s"], s["start_s"] + 0.08), "text": s["text"].strip(),
                                       "commit": w["start_s"] + c / 1000, "raw_commit": w["start_s"] + s["write_ms"] / 1000,
                                       "blk": b})
    U_ = defaultdict(lambda: defaultdict(list))
    parts = defaultdict(list)
    for m in sorted(segs):
        A = M.assign(m, segs[m], sc, U[m])
        for uid, u in A["units"].items():
            ss = [x for x, _ in A["got"].get(uid, [])]
            if not ss:
                continue
            reg = u["reg"]
            U_[reg]["text_end_online"].append(max(x["commit"] for x in ss) - u["b"])
            U_[reg]["text_end_raw"].append(max(x["raw_commit"] for x in ss) - u["b"])
            U_[reg]["text_start_online"].append(min(x["commit"] for x in ss) - u["a"])
            bs = [x["blk"] for x in ss if x["blk"] is not None]
            if not bs:
                continue
            for var, s0, s1 in (("recorded", "rec_start", "rec_end"), ("online", "on_start", "on_end"), ("online_1ch", "ch_start", "ch_end")):
                U_[reg][f"speech_start_{var}"].append(min(b[s0] for b in bs) - u["a"])
                U_[reg][f"speech_end_{var}"].append(max(b[s1] for b in bs) - u["b"])
            pw = [(x, t) for x, t in sorted(A["got"].get(uid, []), key=lambda xt: xt[0]["a"])]
            td = [x["commit"] - u["a"] for x, t in pw for _ in t.split()]
            if td:
                U_[reg]["text_laal_online"].append(laal(td, u))
            sd = speech_word_delays(pw, u["a"])
            if sd:
                U_[reg]["speech_laal_online"].append(laal(sd, u))
            last = max(bs, key=lambda b: b["on_end"])
            parts["model_lag"].append(last["write"] - u["b"])
            parts["detector_wait"].append(last["avail"] - last["write"])
            parts["queue_wait"].append(last["on_start"] - last["avail"])
            parts["last_block_dur"].append(last["dur"])
            parts["final_read_last"].append(float(last["final_read"]))
    # output load: talker speech seconds against the meeting's source time and its annotated speech (union)
    load = {}
    for m in sorted(bym):
        sp = sum(blk[kk]["dur"] for _, kk in bym[m])
        iv = sorted((u[2], u[3]) for u in U[m])
        un, c0, c1 = 0.0, None, None
        for x0, x1 in iv:
            if c1 is None or x0 > c1:
                un += (c1 - c0) if c1 is not None else 0.0
                c0, c1 = x0, x1
            else:
                c1 = max(c1, x1)
        un += (c1 - c0) if c1 is not None else 0.0
        span = max(blk[kk]["on_end"] for _, kk in bym[m]) - min(blk[kk]["avail"] for _, kk in bym[m])
        load[m] = {"speech_s": round(sp, 1), "src_speech_union_s": round(un, 1), "span_s": round(span, 1),
                   "speech_over_src_union": round(sp / max(un, 1e-6), 3)}
    tot_sp = sum(v["speech_s"] for v in load.values())
    tot_un = sum(v["src_speech_union_s"] for v in load.values())
    durs = [b["dur"] for b in blk.values()]
    res_load = {"speech_s": round(tot_sp, 1), "src_speech_union_s": round(tot_un, 1),
                "speech_over_src_union": round(tot_sp / max(tot_un, 1e-6), 3), "per_meeting": load,
                "block_dur": q(durs)}
    res = {"run": args.run, "talker": args.talker, "check": chk, "segment_block_mismatch": miss,
           "stream": {v: {k: q(x) for k, x in d.items()} for v, d in stream.items()},
           "unit": {}, "unit_end_parts_online": {k: q(v) for k, v in parts.items()}, "load": res_load}
    for key in sorted({k for d in U_.values() for k in d}):
        res["unit"][key] = {"all": q(U_["ov"][key] + U_["1spk"][key]), "ov": q(U_["ov"][key]), "1spk": q(U_["1spk"][key])}
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "x", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1, ensure_ascii=False)
    print(json.dumps({"check": chk, "miss": miss, "stream": res["stream"], "parts": res["unit_end_parts_online"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()

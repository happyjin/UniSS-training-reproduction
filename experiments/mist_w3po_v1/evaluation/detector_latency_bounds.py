"""Window-detector latency: CPU upper bounds on an existing dev talker run (detector method design v4b, 2026-11-12;
reports/mist_w3po_v1/thinker_detector_*_prereg_v1.zh-CN.md). Read-only on the run; dev only; no GPU.

speech_latency_lit stamps every block with the online-protocol commit c = max(write_ms, open_detect_ms), and a
final-read block (write_ms >= dur_ms) with c >= close_detect_ms. The window jsonl builds both detection times with the
lowlat preset's WORST-CASE look-ahead (every 80 ms posterior frame known 1.04 s after its start) and C2's close rule
(gap 0.6 s). This tool re-times the SAME text and the SAME talker blocks (durations, per-stream queue) under other
detection times and reports per-stream speech LAAL / text LAAL / EndOffset per variant:

  base       the window jsonl as is (reproduces speech_latency_lit's speech_laal_online)
  a_noclose  final read stamped at the window's audio end (dur_ms): no close-detection wait (bound, needs P there)
  b_noopen   no open clamp (c = write_ms for non-final reads)
  c_both     a + b
  d_g03      close rule gap 0.3 s on the SAME windows (close - 4 frames; bound: gap 0.3 also splits windows)
  e_chunk    exact chunk emission of the lowlat streaming preset (NeMo streaming_feat_loader: chunks of 9 frames
             from frame 0, 4 frames right context): frame f is known at T(f) = (9 floor(f/9) + 13) * 80 ms, so the
             open is known at T(st) and the close at T(j - 1) (j = the frame index the gap rule stops at; the
             decision needs frames <= j - 1). T(f) <= f * 80 + 1040 always (equality when f = 0 mod 9): the window
             jsonl's convention is this rule's worst case. Same windows, same posteriors, and every in-window frame
             is still visible at the (earlier) final read, so the decode is unchanged -- checked per window
  f_chunk_g03  e_chunk with the close at T(j03 - 1), j03 = the gap-0.3 stop frame (same windows; bound)
  g_eou      final read at the window's audio end + 0.02 s (the Thinker hears 0.32 s of trailing silence itself, no
             diarizer look-ahead) and the open of e_chunk (bound of a Thinker end-of-window decision)
  h1_spec    speculative close with an ORACLE judge: the final read at min(T(j - 1), max(T(en), en * 80 ms + gap)):
             the posterior of the first silent frame en is out (every active frame of the window is visible) and the
             audio of the whole gap has arrived, so a judge that hears the gap can confirm the close; e_chunk open
  h2_spec    as h1 without waiting for T(en): min(T(j - 1), en * 80 ms + gap) (the last active frames' posteriors
             may still be unknown: attribution risk counted)

Attribution / window risk per variant (posterior proxies; the decode is NOT re-run):
  visible frames at the final read: base = the finalpost rule (frames < min(dur, close_detect - 1.04 s)); other
  variants = frames known at their final stamp (1.04 s rule, or T(f) for the chunk variants)
  seg_unsupported   final-read segments (and words) whose slot is active (> 0.5) in a visible frame of its span
                    (+- 0.16 s) under base but in none under the variant
  tail_new_slot     windows whose invisible in-window tail has an active slot never active in the visible part
  decode_changes    windows whose final read would see fewer in-window frames than base (its text may change)
Window-boundary changes of the gap rule (d / f): gap-0.6 windows with an internal silence run that gap 0.3 cuts
(>= 4 silent frames, < 8), and their final-read words.

    python -m experiments.mist_w3po_v1.evaluation.detector_latency_bounds --run <decode dir> --talker <talker dir> \
        --win <windows jsonl> --out <json>
Writes <out> (refuses to overwrite).
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from experiments.mist_w3po_v1.evaluation import metric_headroom as M
from experiments.mist_w3po_v1.evaluation.speech_latency_lit import laal, q, speech_word_delays

FR = 0.08            # s per posterior frame
LA = 1.04            # lowlat worst-case look-ahead (s)
CHUNK, RC = 9, 4     # lowlat preset (dump_nemotron_posteriors.PRESETS["lowlat"]): chunk_len, chunk_right_context
GAP06, GAP03, CAP = round(0.6 / FR), round(0.3 / FR), round(20.0 / FR)
SPAN_TOL = 2         # frames of slack around a segment's span for the support check
VARIANTS = ("base", "a_noclose", "b_noopen", "c_both", "d_g03", "e_chunk", "f_chunk_g03", "g_eou", "h1_spec", "h2_spec")


def t_chunk(f: int) -> float:
    """absolute time (s) at which posterior frame f (80 ms, session index) is emitted by the lowlat streaming preset"""
    return (CHUNK * (f // CHUNK) + CHUNK + RC) * FR


def t_worst(f: int) -> float:
    return f * FR + LA


def online_windows(post: np.ndarray, gap: int) -> list:
    """build_windows_zh2en.online_windows (gap in frames): (first active, last active + 1, stop frame j)"""
    act = (post > 0.5).any(1)
    out, i, n = [], 0, len(act)
    while i < n:
        if not act[i]:
            i += 1
            continue
        st, last, j = i, i, i
        while j < n and j - st < CAP and j - last <= gap:
            if act[j]:
                last = j
            j += 1
        out.append((st, last + 1, j))
        i = last + 1
    return out


def window_times(w: dict, st: int, en: int, j: int, n: int, sess_s: float, act: np.ndarray) -> dict:
    """per window: absolute detection times (s) under each variant + the facts the risk checks need."""
    ws = w["start_s"]
    we = ws + w["dur_ms"] / 1000
    open_b, close_b = ws + w["open_detect_ms"] / 1000, ws + w["close_detect_ms"] / 1000
    gap_closed = j < n and j - (en - 1) > GAP06
    sess_end = j >= n
    # the gap-0.3 stop frame on the same window (first run of GAP03 silent frames after the LAST active frame)
    j03 = (en - 1) + GAP03 + 1
    open_c = t_chunk(st)
    close_c = sess_s if sess_end else min(sess_s, t_chunk(j - 1))
    close_c03 = close_c if not gap_closed else min(sess_s, t_chunk(j03 - 1))
    close_w03 = close_b if not gap_closed else close_b - (GAP06 - GAP03) * FR
    # internal silences that gap 0.3 would cut (>= GAP03 silent frames, inside [st, en))
    cuts, run = [], 0
    for f in range(st, en):
        if act[f]:
            if run >= GAP03:
                cuts.append(f)
            run = 0
        else:
            run += 1
    tm = {
        "base": (open_b, close_b),
        "a_noclose": (open_b, max(we, open_b)),
        "b_noopen": (ws, close_b),
        "c_both": (ws, we),
        "d_g03": (open_b, close_w03),
        "e_chunk": (open_c, close_c),
        "f_chunk_g03": (open_c, close_c03),
        "g_eou": (open_c, max(we + 0.02, open_c)),
        "h1_spec": (open_c, close_c if not gap_closed else max(open_c, min(close_c, max(t_chunk(en), en * FR + GAP06 * FR)))),
        "h2_spec": (open_c, close_c if not gap_closed else max(open_c, min(close_c, en * FR + GAP06 * FR))),
    }
    return {"ws": ws, "we": we, "st": st, "en": en, "j": j, "times": tm, "gap_closed": gap_closed,
            "g03_cuts": cuts, "sess_end": sess_end,
            "chunk_rule": {v: v in ("e_chunk", "f_chunk_g03", "g_eou", "h1_spec", "h2_spec") for v in VARIANTS}}


def visible_until(wt: dict, v: str) -> float:
    """absolute time (s) up to which the final read's posteriors are visible: frames f with avail(f) <= the stamp;
    returned as the end of the last visible frame region (frames < k visible -> k * FR)."""
    ws, we = wt["ws"], wt["we"]
    if v == "base":   # online_decode_finalpost.final_cut_ms: max(t - LA, min(t, close - LA)), t = dur
        cut = max(we - LA, min(we, wt["times"]["base"][1] - LA))
        return cut
    stamp = wt["times"][v][1]
    if wt["chunk_rule"][v]:
        c = (stamp / FR - CHUNK - RC)            # frames f with 9 floor(f/9) + 13 <= stamp / FR
        k = (math.floor(c / CHUNK + 1e-9) + 1) * CHUNK if c >= 0 else 0
        return min(we, k * FR)
    return min(we, max(0.0, stamp - LA))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True)
    ap.add_argument("--talker", required=True)
    ap.add_argument("--win", required=True)
    ap.add_argument("--diar", default="data/ami_multispk_si_v1/diar/nemotron_lowlat_zh8")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    import torch
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

    # windows recomputed from the session posteriors and checked against the jsonl
    WT, POST = {}, {}
    chk = defaultdict(int)
    bym = defaultdict(list)
    for w in W:
        bym[w["meeting"]].append(w)
    for m, L in bym.items():
        post = torch.load(M.REPO / args.diar / f"{m}.post.pt", weights_only=False)["post80"].float().numpy()
        act = (post > 0.5).any(1)
        POST[m] = post
        wl = online_windows(post, GAP06)
        sess_s = max(w["start_s"] + w["dur_ms"] / 1000 for w in L)   # only used to cap close times
        for w in L:
            k = int(w["id"].rsplit("_", 1)[1])
            st, en, j = wl[k]
            ok_open = abs((st * FR + LA - w["start_s"]) * 1000 - w["open_detect_ms"]) <= 1
            ok_close = j >= len(post) or abs((j * FR + LA - w["start_s"]) * 1000 - w["close_detect_ms"]) <= 1 or \
                abs((sess_s - w["start_s"]) * 1000 - w["close_detect_ms"]) <= 1
            chk["windows"] += 1
            chk["open_match"] += ok_open
            chk["close_match"] += ok_close
            wt = window_times(w, st, en, j, len(post), max(sess_s, w["start_s"] + w["close_detect_ms"] / 1000), act)
            # chunk rule never later than the jsonl's worst case
            assert wt["times"]["e_chunk"][0] <= wt["times"]["base"][0] + 1e-6, w["id"]
            assert wt["times"]["e_chunk"][1] <= wt["times"]["base"][1] + 1e-6, w["id"]
            WT[w["id"]] = wt

    def stamp(v, w, wm):
        wt = WT[w["id"]]
        o, c = wt["times"][v]
        t = w["start_s"] + wm / 1000
        a = max(t, o)
        if wm >= w["dur_ms"]:
            a = max(a, c)
        return a

    # segments, unit assignment (independent of the stamps)
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
            r = rows.get(k)
            bi = i if (r is not None and i < len(r["blocks"]) and r["blocks"][i]["text"].strip() == s["text"].strip()) else None
            if bi is None:
                miss += 1
            segs[w["meeting"]].append({"spk": s["spk"], "a": w["start_s"] + s["start_s"],
                                       "b": w["start_s"] + max(s["end_s"], s["start_s"] + 0.08), "text": s["text"].strip(),
                                       "wm": s["write_ms"], "w": w, "bk": (k, bi) if bi is not None else None,
                                       "s0": s["start_s"], "s1": s["end_s"]})
    A = {m: M.assign(m, segs[m], sc, U[m]) for m in sorted(segs)}

    res = {"run": args.run, "talker": args.talker, "win": args.win, "check": dict(chk), "segment_block_mismatch": miss,
           "variants": {}}
    for v in VARIANTS:
        # per-stream queue under the variant's availability
        blk = {}
        for k, r in rows.items():
            w = wins[k.split("__S")[0]]
            prev = None
            for i, b in enumerate(r["blocks"]):
                av = stamp(v, w, b["write_ms"])
                st = av if prev is None else max(av, prev)
                prev = st + b["dur_ms"] / 1000
                blk[(k, i)] = {"on_start": st, "on_end": prev, "text": b["text"], "avail": av,
                               "final": b["write_ms"] >= w["dur_ms"]}
        UT = defaultdict(lambda: defaultdict(list))
        for m in sorted(segs):
            for x in segs[m]:
                x["commit"] = stamp(v, x["w"], x["wm"])
                x["blk"] = blk.get(x["bk"]) if x["bk"] is not None else None
            for uid, u in A[m]["units"].items():
                pw = sorted(A[m]["got"].get(uid, []), key=lambda xt: xt[0]["a"])
                if not pw:
                    continue
                bs = [x["blk"] for x, _ in pw if x["blk"] is not None]
                if not bs:
                    continue
                reg = u["reg"]
                td = [x["commit"] - u["a"] for x, t in pw for _ in t.split()]
                if td:
                    UT[reg]["text_laal"].append(laal(td, u))
                sd = speech_word_delays(pw, u["a"])
                if sd:
                    UT[reg]["speech_laal"].append(laal(sd, u))
                UT[reg]["speech_end"].append(max(b["on_end"] for b in bs) - u["b"])
        vr = {key: {"all": q(UT["ov"][key] + UT["1spk"][key]), "ov": q(UT["ov"][key]), "1spk": q(UT["1spk"][key])}
              for key in ("speech_laal", "text_laal", "speech_end")}
        # risk proxies (final-read segments)
        rk = defaultdict(int)
        for wid, wt in WT.items():
            w = wins[wid]
            h = H.get(wid)
            post = POST[w["meeting"]]
            vis_b, vis_v = visible_until(wt, "base"), visible_until(wt, v)
            fb, fv = int(round(vis_b / FR)), int(round(vis_v / FR))
            fin = [s for s in (h["segments"] if h else []) if s["text"].strip() and s["write_ms"] >= w["dur_ms"]]
            rk["final_segs"] += len(fin)
            rk["final_words"] += sum(len(s["text"].split()) for s in fin)
            if fv < fb:
                rk["decode_changes"] += 1
            f_end = int(math.ceil(wt["we"] / FR))
            seen = set(np.nonzero((post[int(wt["ws"] / FR): max(fv, int(wt["ws"] / FR))] > 0.5).any(0))[0].tolist())
            tail = set(np.nonzero((post[max(fv, int(wt["ws"] / FR)): f_end] > 0.5).any(0))[0].tolist()) if fv < f_end else set()
            if tail - seen:
                rk["tail_new_slot_windows"] += 1
                rk["tail_new_slot_final_words"] += sum(len(s["text"].split()) for s in fin)
            for s in fin:
                a0 = int((w["start_s"] + s["start_s"]) / FR) - SPAN_TOL
                a1 = int(math.ceil((w["start_s"] + s["end_s"]) / FR)) + SPAN_TOL
                sl = s["spk"] - 1
                if not (0 <= sl < post.shape[1]):
                    continue
                sup_b = bool((post[max(a0, 0): min(a1, fb), sl] > 0.5).any()) if min(a1, fb) > max(a0, 0) else False
                sup_v = bool((post[max(a0, 0): min(a1, fv), sl] > 0.5).any()) if min(a1, fv) > max(a0, 0) else False
                rk["seg_supported_base"] += sup_b
                if sup_b and not sup_v:
                    rk["seg_unsupported"] += 1
                    rk["word_unsupported"] += len(s["text"].split())
            if v in ("d_g03", "f_chunk_g03") and wt["g03_cuts"]:
                rk["g03_split_windows"] += 1
                rk["g03_split_final_words"] += sum(len(s["text"].split()) for s in fin)
        nb = sum(1 for b in blk.values() if b["final"])
        vr["risk"] = dict(rk)
        vr["final_blocks"] = nb
        res["variants"][v] = vr
        print(v, json.dumps({"speech_laal": vr["speech_laal"]["all"], "text_laal": vr["text_laal"]["all"]["mean"],
                             "speech_end": vr["speech_end"]["all"]["mean"], "risk": dict(rk)}), flush=True)
    # detection-time summaries (s, window-relative waits)
    det = {}
    for v in VARIANTS:
        det[v] = {"open_after_start": q([wt["times"][v][0] - wt["ws"] for wt in WT.values()]),
                  "close_after_audio_end": q([wt["times"][v][1] - wt["we"] for wt in WT.values()])}
    res["detection"] = det
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "x", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()

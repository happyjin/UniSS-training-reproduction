"""Pause flush (speech-gap plan, after C2; user go 2026-11-10 ~19:10 UTC), CPU, dev only: an online windows jsonl
(build_windows_zh2en.py, e.g. the C2 gap-0.6 dev windows) -> the same rows with the flush schedule of a pause hold.

The online system cannot tell a pause inside a window from the pause that will close it (gap s of silence). With a
hold of h frames (80 ms each), whenever speech is followed by h silent frames (act = any diarizer posterior > 0.5,
the window builder's rule) the system knows it at (pause onset + h) * 80 ms + LOOKAHEAD (1.04 s, a lowlat frame's
delay) and flushes: a read with <wait> barred, so the Thinker writes everything it can (online_decode --flush-reads).
  flush_ms         window-relative times of the pauses INSIDE the window (speech resumes before the window's last
                   active frame) whose flush time is before the window end (dur_ms); online_decode adds these reads
  close_detect_ms  min(close detection, the flush time of the window's final pause): that flush is the window's final
                   read (the window does close there, so nothing after it could change what the system knows then);
                   the finalpost rule then shows P up to min(dur, close - 1.04 s), the frames known at that time, and
                   every scorer stamps final-read output at this time (overlap_score / metric_headroom /
                   speech_latency_lit read close_detect_ms). A window closed by the 20 s cap (speech goes on) keeps
                   its close detection.
  close_detect_orig_ms, flush_hold_ms   kept for reference
The windows are recomputed from the session posteriors with online_windows(post, --gap) and every row is checked
against them (start, duration, close detection); audio and posterior files are the input rows' own.

    python -m experiments.mist_w3po_v1.data.build_flush_windows --win <windows jsonl> --gap 0.6 --hold-frames 4 \
        --out <new windows jsonl>
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from experiments.mist_w3po_v1.data.build_windows_zh2en import FRAME, LOOKAHEAD, online_windows

FR_MS, LA_MS = round(FRAME * 1000), round(LOOKAHEAD * 1000)


def flush_rows(rows: list[dict], post: np.ndarray, sess_ms: int, gap: float, h: int) -> tuple[list[dict], dict]:
    """rows: one meeting's windows (ids <meeting>_on_<k>); post: its session posteriors [F80, slots]."""
    act = (post > 0.5).any(1)
    n = len(act)
    win = online_windows(post, gap)
    out, st_ = [], defaultdict(int)
    for r in rows:
        k = int(r["id"].rsplit("_on_", 1)[1])
        st, en, j = win[k]
        ws = max(0, st * FR_MS - 200)
        we = min(sess_ms, en * FR_MS + 300)
        close = min(sess_ms, j * FR_MS + LA_MS) if j < n else sess_ms
        assert abs(ws - round(r["start_s"] * 1000)) <= 1 and abs((we - ws) - r["dur_ms"]) <= 1, (r["id"], ws, we)
        assert abs((close - ws) - r["close_detect_ms"]) <= 1, (r["id"], close - ws, r["close_detect_ms"])
        dur = r["dur_ms"]
        fl = []
        for p in range(st + 1, en):
            if act[p - 1] and not act[p: p + h].any() and p + h < en:
                t = (p + h) * FR_MS + LA_MS - ws
                if t < dur:
                    fl.append(t)
                else:
                    st_["inside_after_end"] += 1
        c = r["close_detect_ms"]
        if en + h <= n and not act[en: en + h].any():
            c = min(c, (en + h) * FR_MS + LA_MS - ws)
            st_["final_flush"] += 1
        else:
            st_["no_final_pause"] += 1
        assert c >= dur, (r["id"], c, dur)
        o = dict(r)
        o.update({"close_detect_ms": int(c), "close_detect_orig_ms": r["close_detect_ms"], "flush_ms": fl,
                  "flush_hold_ms": h * FR_MS})
        out.append(o)
        st_["windows"] += 1
        st_["flush_reads"] += len(fl)
        st_["close_saved_ms"] += r["close_detect_ms"] - c
    return out, dict(st_)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--win", required=True)
    ap.add_argument("--gap", type=float, required=True, help="the gap the input windows were built with")
    ap.add_argument("--hold-frames", type=int, default=4)
    ap.add_argument("--data", default="data/ami_multispk_si_v1")
    ap.add_argument("--diar", default="nemotron_lowlat_zh8")
    ap.add_argument("--mix", default="zh_mix_v8")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    import soundfile as sf
    import torch
    rows = [json.loads(l) for l in open(a.win, encoding="utf-8")]
    by_m = defaultdict(list)
    for r in rows:
        by_m[r["meeting"]].append(r)
    res, tot = {}, defaultdict(int)
    for m, rs in sorted(by_m.items()):
        post = torch.load(Path(a.data) / "diar" / a.diar / f"{m}.post.pt", weights_only=False)["post80"].float().numpy()
        info = sf.info(str(Path(a.data) / "session_audio" / a.mix / f"{m}.wav"))   # the builder's session length
        sess_ms = round(info.frames / info.samplerate * 1000)
        o, s = flush_rows(rs, post, sess_ms, a.gap, a.hold_frames)
        for r in o:
            res[r["id"]] = r
        for k, v in s.items():
            tot[k] += v
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "x", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(res[r["id"]]) + "\n")
    tot = dict(tot)
    tot["close_saved_ms_mean"] = round(tot.pop("close_saved_ms") / max(tot["windows"], 1), 1)
    tot.update({"gap": a.gap, "hold_frames": a.hold_frames, "win": a.win})
    json.dump(tot, open(str(out) + ".summary.json", "x"), indent=1)
    print(json.dumps(tot), flush=True)


if __name__ == "__main__":
    main()

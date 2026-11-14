"""E7 build (CPU; reports/mist_w3po_v1/e7_seamless_multispk_prereg.zh-CN.md §3-§4): run_e7.py's per-window Seamless
output -> our run formats, so the existing scorers / latency tools read it unchanged.

  <dec>/<name>/hyp_seamless.jsonl        per window {id, segments [{spk, start_s, end_s, text, write_ms}], ...}
                                         (score_run merges it into hyp.jsonl)
  <dec>/<name>_1spk/hyp_seamless.jsonl   the same segments, every one labelled speaker 1 (lower bound, text only)
  <spd>/<name>/talker/results.jsonl      one row per (window, labelled speaker) stream with speech: blocks [{write_ms,
                                         start_ms, dur_ms, text}] (start = max(write, end of the stream's previous
                                         block)), speech_path (the stream's blocks back to back, for ASR), audio_path
                                         (the timeline wav)
  <dec>/<name>/build_e7.json             counts

Segments (prereg §3): one per emitted speech segment, text = the words it speaks, write_ms = text_ms (source read when
its last token left the text decoder), source span [previous segment's text_ms, its text_ms] (window-relative; the
first from 0); the unspoken tail text (if any) is one more text-only segment. An empty-text speech segment's audio is
appended to the window's previous speech block (prepended to the next one if it is the first).
Speaker (prereg §4): reports/gstar_si_v8/overlap_score.relabel_causal, called as overlap_score.score(relabel=True) calls
it: the window's post.lowlat posterior, commit = the online-protocol commit (>= open_detect_ms; >= close_detect_ms
when written after the whole window was read); 0 -> "unk".

    PYTHONPATH=. $PY experiments/s2st_baselines_v1/seamless_multispk/build_e7.py --raw <run_e7 out dir> \
        --windows <online windows> --name jtest_<set>_seamless_default --tgt eng|cmn --src cmn|eng
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "reports/gstar_si_v8"))
sys.path.insert(0, str(REPO / "reports/gstar_si_v6"))
DEC = REPO / "eval_outputs/mist_w3po_v1/decode"
SPD = REPO / "eval_outputs/mist_w3po_v1/speech"
SR = 16000


def online(w: dict, wm: int) -> int:
    """overlap_score.score's online commit (ms, window-relative)."""
    c = max(wm, w["open_detect_ms"])
    if wm >= w["dur_ms"]:
        c = max(c, w["close_detect_ms"])
    return c


def window_segments(w: dict, r: dict, stats: Counter) -> list[dict]:
    """[(text, text_ms, block or None)] after the empty-text merge; block = {write_ms, dur_ms, wavs}."""
    sp = [{"text": s["text"].strip(), "text_ms": s["text_ms"],
           "blk": {"write_ms": s["write_ms"], "dur_ms": s["dur_ms"], "wavs": [s["wav"]]}} for s in r["segs"]]
    out: list[dict] = []
    carry = None   # leading empty-text blocks, prepended to the next block
    for s in sp:
        if not s["text"]:
            stats["empty_text_speech_segments"] += 1
            if out:
                out[-1]["blk"]["dur_ms"] += s["blk"]["dur_ms"]
                out[-1]["blk"]["wavs"] += s["blk"]["wavs"]
            elif carry is None:
                carry = s["blk"]
            else:
                carry["dur_ms"] += s["blk"]["dur_ms"]
                carry["wavs"] += s["blk"]["wavs"]
            continue
        if carry is not None:
            s["blk"]["dur_ms"] += carry["dur_ms"]
            s["blk"]["wavs"] = carry["wavs"] + s["blk"]["wavs"]
            carry = None
        out.append(s)
    if carry is not None:   # a window whose speech has no text at all: kept as one text-less block
        stats["textless_windows_with_speech"] += 1
        out.append({"text": "", "text_ms": carry["write_ms"], "blk": carry})
    if r.get("unspoken") and r["unspoken"]["text"].strip():
        stats["unspoken_segments"] += 1
        out.append({"text": r["unspoken"]["text"].strip(), "text_ms": r["unspoken"]["text_ms"], "blk": None})
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", required=True)
    ap.add_argument("--windows", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--tgt", choices=["eng", "cmn"], required=True)
    ap.add_argument("--src", choices=["eng", "cmn"], required=True)
    ap.add_argument("--dec-root", default=str(DEC), help="smoke tests only")
    ap.add_argument("--spd-root", default=str(SPD), help="smoke tests only")
    a = ap.parse_args()
    from overlap_score import relabel_causal

    dec, spd = Path(a.dec_root), Path(a.spd_root)
    D, D1, T = dec / a.name, dec / f"{a.name}_1spk", spd / a.name / "talker"
    for p in (D, D1, spd / a.name):
        if p.exists():
            raise SystemExit(f"拒绝覆盖已有输出: {p}")
    W = [json.loads(l) for l in open(a.windows, encoding="utf-8")]
    R = {}
    for f in sorted(Path(a.raw).glob("part_*.jsonl")):
        for l in open(f, encoding="utf-8"):
            r = json.loads(l)
            R[r["id"]] = r
    miss = [w["id"] for w in W if w["id"] not in R]
    if miss:
        raise SystemExit(f"{len(miss)} of {len(W)} windows missing in {a.raw}, e.g. {miss[:3]}")
    for p in (D, D1, T / "wav"):
        p.mkdir(parents=True)
    stats = Counter()
    sep = " " if a.tgt == "eng" else ""
    H, H1, rows = [], [], []
    for w in W:
        r = R[w["id"]]
        stats["windows"] += 1
        stats["error_windows"] += bool(r.get("error"))
        stats["restarts"] += r.get("restarts", 0)
        stats["raw_speech_segments"] += len(r["segs"])
        segs = window_segments(w, r, stats)
        pp = (w.get("post") or {}).get("lowlat")
        post = np.load(pp) if pp and Path(pp).exists() else None
        stats["windows_without_post"] += post is None
        prev = 0
        hs = []
        for s in segs:
            t0, t1 = prev / 1000, s["text_ms"] / 1000
            prev = max(prev, s["text_ms"])
            seg = {"a": w["start_s"] + t0, "b": w["start_s"] + max(t1, t0 + 0.08)}
            c = w["start_s"] + online(w, s["text_ms"]) / 1000
            k = relabel_causal(seg, post, w["start_s"], c)
            spk = k if k else "unk"
            stats["unk_segments"] += spk == "unk"
            hs.append({"spk": spk, "start_s": round(t0, 3), "end_s": round(t1, 3), "text": s["text"],
                       "write_ms": int(s["text_ms"]), "blk": s["blk"]})
        stats["segments"] += sum(1 for s in hs if s["text"])
        tr = sep.join(s["text"] for s in hs if s["text"])
        H.append({"id": w["id"], "translation": tr, "system": "SeamlessStreaming official defaults (E7)",
                  "segments": [{k_: v for k_, v in s.items() if k_ != "blk"} for s in hs]})
        H1.append({"id": w["id"], "translation": tr, "system": "SeamlessStreaming official defaults (E7), one speaker",
                   "segments": [dict({k_: v for k_, v in s.items() if k_ != "blk"}, spk=1) for s in hs]})
        # talker streams: (window, speaker) in emission order, queued per stream
        by = defaultdict(list)
        for s in hs:
            if s["blk"] is not None:
                by[s["spk"]].append(s)
        for spk, ss in by.items():
            sid = f"{w['id']}__S{spk}"
            blocks, wavs, end_prev = [], [], 0
            for s in ss:
                b = s["blk"]
                x = np.concatenate([sf.read(p, dtype="float32")[0] for p in b["wavs"]])
                dur = int(round(len(x) / 16))
                st = max(b["write_ms"], end_prev)
                end_prev = st + dur
                blocks.append({"write_ms": int(b["write_ms"]), "start_ms": int(st), "dur_ms": dur, "text": s["text"],
                               "text_ms": int(s["write_ms"])})
                wavs.append(x)
            speech = np.concatenate(wavs)
            total = int(max(w["dur_ms"], end_prev)) + 300
            tl = np.zeros(total * 16, dtype=np.float32)
            for b, x in zip(blocks, wavs):
                a0 = b["start_ms"] * 16
                tl[a0: a0 + len(x)] += x[: len(tl) - a0]
            sp_path, tl_path = T / "wav" / f"{sid}_speech.wav", T / "wav" / f"{sid}_simul.wav"
            sf.write(sp_path, speech, SR)
            sf.write(tl_path, tl, SR)
            stats["streams"] += 1
            stats["blocks"] += len(blocks)
            stats["speech_s"] += len(speech) / SR
            rows.append({"id": sid, "mode": "si", "name": sid, "src_lang": a.src, "tgt_lang": a.tgt,
                         "transcription_ref": "", "translation_ref": "",
                         "translated_text": sep.join(b["text"] for b in blocks), "audio_path": str(tl_path),
                         "speech_path": str(sp_path), "audio_duration_seconds": round(len(tl) / SR, 3),
                         "source_audio_path": w["audio"], "source_duration_ms": int(w["dur_ms"]), "error": None,
                         "start_offset_ms": blocks[0]["start_ms"], "end_offset_ms": int(end_prev - w["dur_ms"]),
                         "blocks": blocks, "system": "SeamlessStreaming official defaults (E7)"})
    for p, L in ((D / "hyp_seamless.jsonl", H), (D1 / "hyp_seamless.jsonl", H1), (T / "results.jsonl", rows)):
        with open(p, "x", encoding="utf-8") as fh:
            fh.writelines(json.dumps(x, ensure_ascii=False) + "\n" for x in L)
    stats["speech_s"] = round(stats["speech_s"], 1)
    spk_count = Counter(str(s["spk"]) for h in H for s in h["segments"])
    res = {"name": a.name, "raw": a.raw, "windows_file": a.windows, "stats": dict(stats), "spk_segments": dict(spk_count)}
    with open(D / "build_e7.json", "x", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1, ensure_ascii=False)
    print(json.dumps(res, ensure_ascii=False))


if __name__ == "__main__":
    main()

"""E8 build (CPU): run_e8.py's per-window X-Translator output -> our run formats, so the existing scorers / latency
tools read it unchanged. A copy of experiments/s2st_baselines_v1/seamless_multispk/build_e7.py (not changed); only the
segment rules differ (reports/mist_w3po_v1/multispk_s2st_baselines_prereg_v2.zh-CN.md §1 S2 = RealSI registration §2
for clock / write times / queue / TTS failures, E7 §3-§4 for source spans, online clamping, relabelling, formats).

  <dec>/<name>/hyp_xtranslator.jsonl     per window {id, segments [{spk, start_s, end_s, text, write_ms}], ...}
                                         (score_run merges it into hyp.jsonl)
  <dec>/<name>_1spk/hyp_xtranslator.jsonl the same segments, every one labelled speaker 1 (lower bound, text only)
  <spd>/<name>/talker/results.jsonl      one row per (window, labelled speaker) stream with speech: blocks [{write_ms,
                                         start_ms, dur_ms, text}] (start = max(write, end of the stream's previous
                                         block)), speech_path (the stream's blocks back to back, for ASR), audio_path
                                         (the timeline wav)
  <dec>/<name>/build_e8.json             counts

Segments: one per X-Translator translation item, in emission order. text = the LMT output (Chinese: whitespace next to
a CJK character removed, E7 §3); write_ms = text time = the source ms received when its ASR final was emitted (MT and
TTS run at once, computation not counted). An item whose translation is empty (or whose MT failed) has no text and no
block (RealSI §2: no sound in the original system); it is dropped and counted. An item with text but no audio (X-Voice
error) is a text-only segment (`unspoken`, RealSI §2 / E7 §3). Source span (E7 §3): [previous segment's text time, its
text time] (window-relative; the first from 0). Speaker (E7 §4): reports/gstar_si_v8/overlap_score.relabel_causal on
the window's post.lowlat posterior at the online-protocol commit; 0 -> "unk". X-Translator's own CAM++ speaker ids
and source-span estimates are kept in the raw rows only.

    PYTHONPATH=. $PY experiments/s2st_baselines_v1/xtranslator_multispk/build_e8.py --raw <run_e8 out dir> \
        --windows <online windows> --name jtest_<set>_xtranslator_default --tgt eng|cmn --src cmn|eng
"""
from __future__ import annotations

import argparse
import json
import re
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
SYSTEM = "X-Translator xt-default (E8)"
ZH_SP = re.compile(r"\s+(?=[一-鿿　-〿＀-￯])|(?<=[一-鿿　-〿＀-￯])\s+")   # run_e7.zh_nospace


def online(w: dict, wm: int) -> int:
    """overlap_score.score's online commit (ms, window-relative)."""
    c = max(wm, w["open_detect_ms"])
    if wm >= w["dur_ms"]:
        c = max(c, w["close_detect_ms"])
    return c


def window_segments(r: dict, tgt: str, stats: Counter) -> list[dict]:
    """[{text, text_ms, blk (None = text only)}] in emission order; blk = {write_ms, dur_ms, wavs}."""
    out = []
    for s in r["segs"]:
        stats["raw_items"] += 1
        txt = (s.get("text") or "").strip()
        if tgt == "cmn":
            txt = ZH_SP.sub("", txt)
        stats["mt_fail_items"] += bool(s.get("mt_error"))
        stats["tts_fail_items"] += bool(s.get("tts_error"))
        stats["tts_timeout_items"] += "Timeout" in (s.get("tts_error") or "")
        if not txt:
            stats["empty_translation_items"] += 1
            stats["empty_translation_items_with_audio"] += bool(s.get("wav"))
            continue
        if s.get("wav"):
            blk = {"write_ms": int(s["write_ms"]), "dur_ms": int(s["dur_ms"]), "wavs": [s["wav"]]}
        else:
            stats["unspoken_segments"] += 1
            blk = None
        out.append({"text": txt, "text_ms": int(s["write_ms"]), "blk": blk})
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
        c = r.get("counters") or {}
        stats["boundary_flushes"] += c.get("boundary_flushes", 0)
        stats["speaker_events"] += c.get("speaker_events", 0)
        segs = window_segments(r, a.tgt, stats)
        stats["empty_windows"] += not segs
        stats["windows_without_speech"] += not any(s["blk"] for s in segs)
        pp = (w.get("post") or {}).get("lowlat")
        post = np.load(pp) if pp and Path(pp).exists() else None
        stats["windows_without_post"] += post is None
        prev = 0
        hs = []
        for s in segs:
            t0, t1 = prev / 1000, s["text_ms"] / 1000
            prev = max(prev, s["text_ms"])
            seg = {"a": w["start_s"] + t0, "b": w["start_s"] + max(t1, t0 + 0.08)}
            cm = w["start_s"] + online(w, s["text_ms"]) / 1000
            k = relabel_causal(seg, post, w["start_s"], cm)
            spk = k if k else "unk"
            stats["unk_segments"] += spk == "unk"
            hs.append({"spk": spk, "start_s": round(t0, 3), "end_s": round(t1, 3), "text": s["text"],
                       "write_ms": int(s["text_ms"]), "blk": s["blk"]})
        stats["segments"] += len(hs)
        tr = sep.join(s["text"] for s in hs)
        H.append({"id": w["id"], "translation": tr, "system": SYSTEM,
                  "segments": [{k_: v for k_, v in s.items() if k_ != "blk"} for s in hs]})
        H1.append({"id": w["id"], "translation": tr, "system": SYSTEM + ", one speaker",
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
                         "blocks": blocks, "system": SYSTEM})
    for p, L in ((D / "hyp_xtranslator.jsonl", H), (D1 / "hyp_xtranslator.jsonl", H1), (T / "results.jsonl", rows)):
        with open(p, "x", encoding="utf-8") as fh:
            fh.writelines(json.dumps(x, ensure_ascii=False) + "\n" for x in L)
    stats["speech_s"] = round(stats["speech_s"], 1)
    spk_count = Counter(str(s["spk"]) for h in H for s in h["segments"])
    res = {"name": a.name, "raw": a.raw, "windows_file": a.windows, "stats": dict(stats), "spk_segments": dict(spk_count)}
    with open(D / "build_e8.json", "x", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1, ensure_ascii=False)
    print(json.dumps(res, ensure_ascii=False))


if __name__ == "__main__":
    main()

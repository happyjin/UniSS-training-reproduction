"""E10 build (CPU): run_e10.py's per-window output -> the formats E7/E8/E9 use, so the existing scorers and latency
tools read it unchanged (a copy of experiments/s2st_baselines_v1/xtranslator_multispk/build_e8.py; only the segment
source and the speaker rule differ).

  <dec>/<name>/hyp_liveinterpret.jsonl   per window {id, translation, segments [{spk, start_s, end_s, text, write_ms}]}
  <dec>/<name>_1spk/hyp_liveinterpret.jsonl  every segment labelled speaker 1 (lower bound, text only)
  <spd>/<name>/talker/results.jsonl      one row per (window, speaker) stream with speech: blocks [{write_ms, start_ms,
                                         dur_ms, text}] (start = max(write, end of the stream's previous block)),
                                         speech_path (the stream's blocks back to back), audio_path (timeline wav)
  <dec>/<name>/build_e10.json            counts

Segments: one per Talker piece in emission order (run_e10.py splits every WRITE at its speaker tags). text_ms = the
piece's write time (source consumed by its step; computation not counted). Source span (E7 §3): [previous segment's
text time, its text time]. Speaker:
  --attr own      the model's own tag (E10); a piece written before any tag -> "unk"
  --attr relabel  reports/gstar_si_v8/overlap_score.relabel_causal on the window's post.lowlat at the online-protocol
                  commit (E7/E8 rule; E10-0, and E10's appendix variant)
A piece with text but no speech is a text-only (`unspoken`) segment.

    PYTHONPATH=/opt/dlami/nvme/jasonleeeli/projects/UniSS $PY e10_multispk/build_e10.py --raw <run_e10 out> \
        --windows <online windows> --name jtest_<set>_e10 --tgt eng|cmn --src cmn|eng --attr own|relabel
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

REPO = Path("/opt/dlami/nvme/jasonleeeli/projects/UniSS")
sys.path.insert(0, str(REPO / "reports/gstar_si_v8"))
sys.path.insert(0, str(REPO / "reports/gstar_si_v6"))
DEC = REPO / "eval_outputs/mist_w3po_v1/decode"
SPD = REPO / "eval_outputs/mist_w3po_v1/speech"
SR = 16000
ZH_SP = re.compile(r"\s+(?=[一-鿿　-〿＀-￯])|(?<=[一-鿿　-〿＀-￯])\s+")   # run_e7.zh_nospace


def online(w: dict, wm: int) -> int:
    """overlap_score.score's online commit (ms, window-relative) = build_e8.online."""
    c = max(wm, w["open_detect_ms"])
    if wm >= w["dur_ms"]:
        c = max(c, w["close_detect_ms"])
    return c


def window_segments(r: dict, tgt: str, stats: Counter) -> list[dict]:
    out = []
    for s in r.get("segs") or []:
        stats["raw_pieces"] += 1
        txt = (s.get("text") or "").strip()
        if tgt == "cmn":
            txt = ZH_SP.sub("", txt)
        if not txt:
            stats["empty_text_pieces"] += 1
            stats["empty_text_pieces_with_audio"] += bool(s.get("wav"))
            continue
        if s.get("wav"):
            blk = {"write_ms": int(s["write_ms"]), "dur_ms": int(s["dur_ms"]), "wavs": [s["wav"]]}
        else:
            stats["unspoken_segments"] += 1
            blk = None
        out.append({"text": txt, "text_ms": int(s["write_ms"]), "blk": blk, "spk": s.get("spk")})
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", required=True)
    ap.add_argument("--windows", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--tgt", choices=["eng", "cmn"], required=True)
    ap.add_argument("--src", choices=["eng", "cmn"], required=True)
    ap.add_argument("--attr", choices=["own", "relabel"], required=True)
    ap.add_argument("--system", default="LiveInterpret S5 (E10)")
    ap.add_argument("--dec-root", default=str(DEC), help="smoke tests only")
    ap.add_argument("--spd-root", default=str(SPD), help="smoke tests only")
    ap.add_argument("--allow-missing", action="store_true", help="smoke tests only")
    a = ap.parse_args()
    from overlap_score import relabel_causal

    dec, spd = Path(a.dec_root).resolve(), Path(a.spd_root).resolve()
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
    if miss and not a.allow_missing:
        raise SystemExit(f"{len(miss)} of {len(W)} windows missing in {a.raw}, e.g. {miss[:3]}")
    W = [w for w in W if w["id"] in R]
    modes = {R[w["id"]].get("mode") for w in W}
    if a.attr == "own" and modes != {"e10"}:
        raise SystemExit(f"--attr own needs E10 output (modes {modes})")
    for p in (D, D1, T / "wav"):
        p.mkdir(parents=True)
    stats = Counter()
    sep = " " if a.tgt == "eng" else ""
    H, H1, rows = [], [], []
    for w in W:
        r = R[w["id"]]
        stats["windows"] += 1
        stats["error_windows"] += bool(r.get("error"))
        stats[f"stop_{r.get('stop')}"] += 1
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
            if a.attr == "own":
                spk = s["spk"] if isinstance(s["spk"], int) else "unk"
            else:
                seg = {"a": w["start_s"] + t0, "b": w["start_s"] + max(t1, t0 + 0.08)}
                cm = w["start_s"] + online(w, s["text_ms"]) / 1000
                k = relabel_causal(seg, post, w["start_s"], cm)
                spk = k if k else "unk"
            stats["unk_segments"] += spk == "unk"
            hs.append({"spk": spk, "start_s": round(t0, 3), "end_s": round(t1, 3), "text": s["text"],
                       "write_ms": int(s["text_ms"]), "blk": s["blk"]})
        stats["segments"] += len(hs)
        tr = sep.join(s["text"] for s in hs)
        H.append({"id": w["id"], "translation": tr, "system": a.system,
                  "segments": [{k_: v for k_, v in s.items() if k_ != "blk"} for s in hs]})
        H1.append({"id": w["id"], "translation": tr, "system": a.system + ", one speaker",
                   "segments": [dict({k_: v for k_, v in s.items() if k_ != "blk"}, spk=1) for s in hs]})
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
                         "blocks": blocks, "system": a.system})
    for p, L in ((D / "hyp_liveinterpret.jsonl", H), (D1 / "hyp_liveinterpret.jsonl", H1),
                 (T / "results.jsonl", rows)):
        with open(p, "x", encoding="utf-8") as fh:
            fh.writelines(json.dumps(x, ensure_ascii=False) + "\n" for x in L)
    stats["speech_s"] = round(stats["speech_s"], 1)
    spk_count = Counter(str(s["spk"]) for h in H for s in h["segments"])
    res = {"name": a.name, "raw": a.raw, "windows_file": a.windows, "attr": a.attr, "stats": dict(stats),
           "spk_segments": dict(spk_count), "missing_windows": len(miss)}
    with open(D / "build_e10.json", "x", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1, ensure_ascii=False)
    print(json.dumps(res, ensure_ascii=False))


if __name__ == "__main__":
    main()

"""Paper v6 copy of gstar_talker.py: the data root is $GSTAR_DATA (default data/ami_multispk_si_v1), so that
AliMeeting (data/alimeeting_si_v1, session near_mix) can be spoken too; nothing else changes.

G-STAR-SI speech (docs/multi_speaker_si_plan_v7/v8): every speaker slot is spoken in that speaker's own voice by the
UniSS talker, then the streams are laid back on the meeting's clock.

  voices  per meeting and slot, the speaker's reference: the stretches where Nemotron-3 hears that slot alone
          (p > 0.5 and every other slot < 0.5, runs >= 0.4 s), up to 8 s, from the whole session (offline) or only
          before the slot's first commit (--causal, streaming)
  prep    hypotheses (gstar_si_stream.py) -> si_infer --items-file / --blocks-file: one stream per window and slot,
          its segments in order; offline every block is written at the window's end, streaming at its commit time
  stitch  si_infer's per-stream speech -> one channel: a segment starts at its source start (offline, dubbing) or
          at its commit (streaming), never before the previous one ends (0.2 s later on a change of speaker);
          per window: translation.wav, stereo (left: the meeting, right: the translation); per meeting excerpt
          (--excerpt-s): the reviewed windows of its first N seconds on one timeline; transcript.jsonl (who, when,
          what)

    python -m experiments.uniss_qwen3_omni_v1.evaluation.gstar_talker prep --hyp ... --windows ... --out ...
"""

from __future__ import annotations

import argparse
import os
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

DATA = Path(os.environ.get("GSTAR_DATA", "data/ami_multispk_si_v1"))   # v6: GSTAR_DATA=data/alimeeting_si_v1 for AliMeeting


def load(p):
    return [json.loads(l) for l in open(p, encoding="utf-8")]


def session_of(args, m: str):
    """The meeting's posteriors (args.diar, default nemotron_<preset>) and audio (args.session: ihm_mix or a Chinese mix)."""
    post = torch.load(DATA / "diar" / (args.diar or f"nemotron_{args.preset}") / f"{m}.post.pt",
                      weights_only=False)["post80"].float().numpy()
    name = f"{m}.Mix-Headset.wav" if args.session == "ihm_mix" else f"{m}.wav"
    x, sr = sf.read(str(DATA / "session_audio" / args.session / name), dtype="float32", always_2d=True)
    return post, x.mean(1), sr


def solo_runs(post: np.ndarray, k: int, until_frame: int | None = None) -> list[tuple[int, int]]:
    """Runs (>= 0.4 s) of frames where slot k alone is active, optionally only before until_frame."""
    alone = (post[:, k] > 0.5) & (np.delete(post, k, axis=1).max(1) < 0.5)
    if until_frame is not None:
        alone = alone[: max(0, until_frame)]
    runs, i = [], 0
    while i < len(alone):
        if alone[i]:
            j = i
            while j < len(alone) and alone[j]:
                j += 1
            if (j - i) * 0.08 >= 0.4:
                runs.append((i, j))
            i = j
        else:
            i += 1
    return runs


def voices(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    meetings = sorted({w["meeting"] for w in load(args.windows)})
    n = 0
    for m in meetings:
        post, x, sr = session_of(args, m)
        for k in range(post.shape[1]):
            if (out / f"{m}_S{k + 1}.wav").exists():
                continue
            runs = solo_runs(post, k)
            pieces, tot = [], 0.0
            for a, b in runs:
                seg = x[int(a * 0.08 * sr): int(b * 0.08 * sr)]
                pieces.append(seg)
                tot += len(seg) / sr
                if tot >= args.max_s:
                    break
            if tot >= 1.0:
                sf.write(str(out / f"{m}_S{k + 1}.wav"), np.concatenate(pieces)[: int(args.max_s * sr)], sr)
                n += 1
    print(json.dumps({"voices": n, "out": str(out)}))


def prep(args):
    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    (out / "voice_fallback").mkdir(parents=True)
    W = {w["id"]: w for w in load(args.windows)}
    vdir = Path(args.voices) if args.voices else None
    sess = {}
    n = 0
    with open(out / "items.jsonl", "x", encoding="utf-8") as fi, open(out / "blocks.jsonl", "x", encoding="utf-8") as fb:
        for h in load(args.hyp):
            w = W[h["id"]]
            by = defaultdict(list)
            if args.blocks and h.get("blocks"):   # streaming: every read's commitment is spoken when it is committed
                seg_end = {(s["spk"], s["start_s"]): s["end_s"] for s in h["segments"]}
                for b in h["blocks"]:
                    if b["text"].strip():
                        by[b["spk"]].append({"text": b["text"], "write_ms": b["write_ms"], "start_s": b["start_s"] or 0.0,
                                             "end_s": seg_end.get((b["spk"], b["start_s"]), w["dur_ms"] / 1000)})
                for k in by:
                    by[k].sort(key=lambda b: (b["write_ms"], b["start_s"]))
            else:
                for s in h["segments"]:
                    if s["text"].strip():
                        by[s["spk"]].append(s)
            for k, segs in sorted(by.items()):
                iid = f"{w['id']}__S{k}"
                v = vdir / f"{w['meeting']}_S{k}.wav" if vdir else Path("/nonexistent")
                if args.causal:   # only what has been heard when the slot's first block is committed
                    if w["meeting"] not in sess:
                        sess.clear()
                        sess[w["meeting"]] = session_of(args, w["meeting"])
                    post, x, sr = sess[w["meeting"]]
                    cut = w["start_s"] + int(segs[0].get("write_ms", w["dur_ms"])) / 1000
                    pieces, tot = [], 0.0
                    for a, b in reversed(solo_runs(post, k - 1, int(cut / 0.08))):   # the most recent first
                        pieces.insert(0, x[int(a * 0.08 * sr): int(b * 0.08 * sr)])
                        tot += (b - a) * 0.08
                        if tot >= 8.0:
                            break
                    v = out / "voice_causal" / f"{iid}.wav"
                    v.parent.mkdir(exist_ok=True)
                    if tot >= 1.0:
                        sf.write(str(v), np.concatenate(pieces)[-int(8.0 * sr):], sr)
                    else:   # too little heard alone: the window's audio up to the first commit
                        xw, srw = sf.read(w["audio"], dtype="float32")
                        sf.write(str(v), xw[: max(srw, int((cut - w["start_s"]) * srw))], srw)
                elif not v.exists():   # no clean stretch of this slot in the session: its own segments in the window
                    x, sr = sf.read(w["audio"], dtype="float32")
                    v = out / "voice_fallback" / f"{iid}.wav"
                    sf.write(str(v), np.concatenate([x[int(s["start_s"] * sr): int(s["end_s"] * sr)] for s in segs] +
                                                    [np.zeros(1, np.float32)]), sr)
                blocks = [{"write_ms": int(s.get("write_ms", w["dur_ms"])), "text": s["text"], "final": False,
                           "start_s": s["start_s"], "end_s": s["end_s"]} for s in segs]
                blocks[-1]["final"] = True
                fi.write(json.dumps({"id": iid, "audio": w["audio"], "dur_ms": w["dur_ms"],
                                     "translation": args.sep.join(s["text"] for s in segs), "voice_audio": str(v)},
                                    ensure_ascii=False) + "\n")
                fb.write(json.dumps({"id": iid, "blocks": blocks}, ensure_ascii=False) + "\n")
                n += 1
    print(json.dumps({"streams": n, "out": str(out)}))


def place(segs, gap_ms=200):
    """segs: dicts with at_ms (earliest start), spk, audio -> start_ms each, one channel, no overlap."""
    t_end, last = 0, None
    for s in segs:
        s["start_ms"] = max(s["at_ms"], t_end + (gap_ms if last not in (None, s["spk"]) else 0))
        t_end, last = s["start_ms"] + len(s["audio"]) // 16, s["spk"]
    return segs


def stitch(args):
    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    (out / "windows").mkdir(parents=True)
    W = {w["id"]: w for w in load(args.windows)}
    blocks_in = {r["id"]: r["blocks"] for r in load(Path(args.prep) / "blocks.jsonl")}
    per_window = defaultdict(list)
    for r in load(args.results):
        wid, spk = r["id"].rsplit("__S", 1)
        speech, sr = sf.read(r["speech_path"], dtype="float32")
        for b, src in zip(r["blocks"], blocks_in[r["id"]]):
            a = b["speech_at_ms"] * 16
            per_window[wid].append({"spk": int(spk), "text": b["text"], "src_start_s": src["start_s"],
                                    "src_end_s": src["end_s"], "write_ms": b["write_ms"],
                                    "at_ms": int(src["start_s"] * 1000) if args.mode == "offline" else b["write_ms"],
                                    "audio": speech[a: a + b["dur_ms"] * 16]})
    with open(out / "transcript.jsonl", "x", encoding="utf-8") as ft:
        for wid, segs in sorted(per_window.items()):
            w = W[wid]
            segs = place(sorted(segs, key=lambda s: (s["at_ms"], s["src_start_s"])))
            total = max([w["dur_ms"]] + [s["start_ms"] + len(s["audio"]) // 16 for s in segs]) + 300
            right = np.zeros(total * 16, np.float32)
            for s in segs:
                right[s["start_ms"] * 16: s["start_ms"] * 16 + len(s["audio"])] += s["audio"]
            src, _ = sf.read(w["audio"], dtype="float32")
            left = np.zeros_like(right)
            left[: len(src)] = src[: len(right)]
            sf.write(str(out / "windows" / f"{wid}.wav"), right, 16000)
            sf.write(str(out / "windows" / f"{wid}_stereo.wav"), np.stack([0.8 * left, right], 1), 16000)
            ft.write(json.dumps({"id": wid, "meeting": w["meeting"], "window_start_s": w["start_s"],
                                 "segments": [{"spk": s["spk"], "src_start_s": round(w["start_s"] + s["src_start_s"], 2),
                                               "src_end_s": round(w["start_s"] + s["src_end_s"], 2),
                                               "play_start_s": round(w["start_s"] + s["start_ms"] / 1000, 2),
                                               "dur_s": round(len(s["audio"]) / 16000, 2), "text": s["text"]}
                                              for s in segs]}, ensure_ascii=False) + "\n")
    if args.excerpt_s > 0:   # one listening excerpt per meeting: its first excerpt_s seconds on the meeting clock
        (out / "excerpts").mkdir()
        trans = load(out / "transcript.jsonl")
        by_m = defaultdict(list)
        for t in trans:
            by_m[t["meeting"]].append(t)
        for m, ts in sorted(by_m.items()):
            t0 = min(t["window_start_s"] for t in ts)
            t1 = t0 + args.excerpt_s
            sess, sr = sf.read(str(DATA / f"session_audio/ihm_mix/{m}.Mix-Headset.wav"), dtype="float32",
                               start=int(t0 * 16000), stop=int(t1 * 16000))
            right = np.zeros(int((args.excerpt_s + 10) * 16000), np.float32)
            segs = []
            for t in ts:
                if t["window_start_s"] >= t1:
                    continue
                wav, _ = sf.read(str(out / "windows" / f"{t['id']}.wav"), dtype="float32")
                a = int((t["window_start_s"] - t0) * 16000)
                n = min(len(wav), len(right) - a)
                right[a: a + n] += wav[:n]
                segs += t["segments"]
            left = np.zeros_like(right)
            left[: len(sess)] = sess
            sf.write(str(out / "excerpts" / f"{m}_stereo.wav"), np.stack([0.8 * left, right], 1), 16000)
            sf.write(str(out / "excerpts" / f"{m}_translation.wav"), right, 16000)
            (out / "excerpts" / f"{m}_transcript.json").write_text(json.dumps(
                {"meeting": m, "start_s": t0, "segments": sorted(segs, key=lambda s: s["play_start_s"])},
                ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({"windows": len(per_window), "out": str(out)}))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("voices")
    a.add_argument("--windows", required=True)
    a.add_argument("--out", required=True)
    a.add_argument("--preset", default="offline")
    a.add_argument("--diar", default="", help="diar/<this> (default nemotron_<preset>)")
    a.add_argument("--session", default="ihm_mix", help="session_audio/<this>: ihm_mix or a Chinese mix")
    a.add_argument("--max-s", type=float, default=8.0)
    a = sub.add_parser("prep")
    a.add_argument("--hyp", required=True)
    a.add_argument("--windows", required=True)
    a.add_argument("--voices", default="")
    a.add_argument("--causal", action="store_true", help="voice prompts from solo frames before the slot's first commit")
    a.add_argument("--preset", default="lowlat")
    a.add_argument("--diar", default="")
    a.add_argument("--session", default="ihm_mix")
    a.add_argument("--blocks", action="store_true",
                   help="streaming: speak each read's committed block at its commit time (not whole segments)")
    a.add_argument("--sep", default="", help="joins a slot's segments (\" \" for English)")
    a.add_argument("--out", required=True)
    a = sub.add_parser("stitch")
    a.add_argument("--results", required=True)
    a.add_argument("--prep", required=True)
    a.add_argument("--windows", required=True)
    a.add_argument("--mode", choices=["offline", "stream"], default="offline")
    a.add_argument("--excerpt-s", type=float, default=0.0)
    a.add_argument("--out", required=True)
    args = ap.parse_args()
    {"voices": voices, "prep": prep, "stitch": stitch}[args.cmd](args)


if __name__ == "__main__":
    main()

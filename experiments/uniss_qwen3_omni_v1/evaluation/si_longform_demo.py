"""Long-form listening demo for the sentence-level simultaneous system: cut a long recording at pauses, run each piece
through si_thinker_stream / si_infer as items, and lay the pieces' speech back on the recording's timeline.

  segment   energy-based cut at pauses (>= --min-pause ms below the floor), pieces between --min-s and --max-s
            seconds; writes the pieces' wavs and an items jsonl {id, audio, dur_ms, translation, offset_ms}
  stitch    each piece's speech blocks (si_infer *_speech.wav) placed at the moment each was committed on the recording's
            timeline; a block whose speech would overlap the one before waits for it (the listener hears one voice);
            writes translation.wav, stereo.wav (source left, translation right), mix.wav, and transcript.json

    python si_longform_demo.py segment --audio long.wav --direction zh2en --out <dir>
    python si_longform_demo.py stitch --items <dir>/items.jsonl --infer <si_infer output>/realsi_<dir> --blocks <blocks jsonl> --out <dir>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf

SR = 16000


def write_ogg(path: Path, x: np.ndarray, sr: int = SR) -> None:
    """Vorbis in blocks: libsndfile's encoder crashes on one large write."""
    with sf.SoundFile(path, "w", sr, 1 if x.ndim == 1 else x.shape[1], format="OGG", subtype="VORBIS") as fh:
        for i in range(0, len(x), 32768):
            fh.write(x[i: i + 32768])


def cmd_segment(args) -> None:
    out = Path(args.out)
    if (out / "items.jsonl").exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out / 'items.jsonl'}")
    (out / "pieces").mkdir(parents=True, exist_ok=True)
    wav, sr = sf.read(args.audio, dtype="float32", always_2d=True)
    assert sr == SR, f"{args.audio}: {sr} Hz, expected {SR}"
    wav = wav.mean(axis=1)
    if args.limit_s:
        wav = wav[: int(args.limit_s * SR)]
    hop = 320   # 20 ms frames
    f = wav[: len(wav) // hop * hop].reshape(-1, hop)
    db = 10 * np.log10((f ** 2).mean(1) + 1e-10)
    floor = np.percentile(db, 20) + args.margin_db
    quiet = db < floor
    min_run = args.min_pause // 20
    cuts, run = [], 0
    for i, q in enumerate(quiet):   # the middle of every pause long enough
        run = run + 1 if q else 0
        if run == min_run:
            cuts.append(i - min_run // 2)
    pieces, start = [], 0
    for c in cuts + [len(db)]:
        c_ms = c * 20
        if c_ms - start >= args.min_s * 1000 or c == len(db):
            while c_ms - start > args.max_s * 1000:   # too long without a pause: cut at the quietest frame in range
                lo, hi = int(start + args.min_s * 1000) // 20, int(start + args.max_s * 1000) // 20
                q = int(lo + np.argmin(db[lo:hi]))
                pieces.append((start, q * 20))
                start = q * 20
            if c_ms > start:
                pieces.append((start, c_ms))
            start = c_ms
    stem = Path(args.audio).stem
    with open(out / "items.jsonl", "x", encoding="utf-8") as fh:
        for k, (a, b) in enumerate(pieces):
            piece = wav[a * 16: b * 16]
            if len(piece) < 0.5 * SR:
                continue
            p = out / "pieces" / f"{stem}_{k:03d}.wav"
            sf.write(p, piece, SR)
            fh.write(json.dumps({"id": f"{stem}_{k:03d}", "audio": str(p.resolve()), "dur_ms": int(len(piece) / 16),
                                 "translation": "", "offset_ms": int(a)}, ensure_ascii=False) + "\n")
    print(json.dumps({"audio": args.audio, "seconds": round(len(wav) / SR, 1), "pieces": len(pieces),
                      "mean_s": round(float(np.mean([b - a for a, b in pieces])) / 1000, 2)}), flush=True)


def cmd_voices(args) -> None:
    """A causal speaker memory: each piece's voice reference is the recording heard up to its first commit, the
    last --window-s seconds of it (BiCodec wants 6 s; a short clip it tiles, and the speaker drifts piece to piece)."""
    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    items = [json.loads(l) for l in open(args.items, encoding="utf-8")]
    blocks = {json.loads(l)["id"]: json.loads(l) for l in open(args.blocks, encoding="utf-8")}
    wav, sr = sf.read(args.audio, dtype="float32", always_2d=True)
    wav = wav.mean(axis=1)
    vdir = out.parent / "voices"
    vdir.mkdir(parents=True, exist_ok=True)
    lens = []
    with open(out, "x", encoding="utf-8") as fh:
        for it in items:
            b = blocks.get(it["id"], {}).get("blocks") or [{"write_ms": it["dur_ms"]}]
            heard = it["offset_ms"] + max(int(b[0]["write_ms"]), 1000)
            a = max(0, heard - int(args.window_s * 1000))
            p = vdir / f"{it['id']}.wav"
            sf.write(p, wav[a * 16: heard * 16], SR)
            lens.append((heard - a) / 1000)
            fh.write(json.dumps({**it, "voice_audio": str(p.resolve())}, ensure_ascii=False) + "\n")
    print(json.dumps({"out": str(out), "pieces": len(items), "ref_s_median": round(float(np.median(lens)), 2),
                      "ref_under_6s": int(sum(l < 6 for l in lens))}), flush=True)


def cmd_stitch(args) -> None:
    out = Path(args.out)
    if (out / "translation.wav").exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out / 'translation.wav'}")
    items = [json.loads(l) for l in open(args.items, encoding="utf-8")]
    infer = Path(args.infer)
    res = {json.loads(l)["id"]: json.loads(l) for l in open(infer / "results.jsonl", encoding="utf-8")}
    blocks = {json.loads(l)["id"]: json.loads(l) for l in open(args.blocks, encoding="utf-8")} if args.blocks else {}
    src_total = max(it["offset_ms"] + it["dur_ms"] for it in items)
    source = np.zeros(int(src_total * 16) + SR, dtype=np.float32)
    for it in items:
        x, _ = sf.read(it["audio"], dtype="float32")
        a = it["offset_ms"] * 16
        source[a: a + len(x)] = x
    spoken = 0
    for it in items:   # room for every piece's speech even if they all queue
        r = res.get(it["id"])
        if r and not r.get("error"):
            spoken += int(sf.info(r["audio_path"]).duration * SR) + SR
    track = np.zeros(len(source) + spoken, dtype=np.float32)
    busy_until, lags, transcript = 0, [], []
    for it in items:
        r = res.get(it["id"])
        if not r or r.get("error"):
            continue
        y, ysr = sf.read(r["speech_path"], dtype="float32")   # the piece's speech alone, block after block
        if ysr != SR:
            idx = np.round(np.arange(int(len(y) * SR / ysr)) * ysr / SR).astype(int).clip(0, len(y) - 1)
            y = y[idx]
        first_place, shifts = None, []
        for b in r["blocks"]:   # each block when it was committed, or as soon as the voice is free
            seg = y[int(b["speech_at_ms"] * 16): int((b["speech_at_ms"] + b["dur_ms"]) * 16)]
            if not len(seg):
                continue
            due = (it["offset_ms"] + b["write_ms"]) * 16
            place = max(due, busy_until)
            behind = (place - due) / SR
            if args.catchup_max > 1 and behind > 0.5:   # speak faster while behind, as an interpreter catches up
                import librosa

                seg = librosa.effects.time_stretch(seg, rate=min(args.catchup_max, 1 + args.catchup_per_s * behind))
            track[place: place + len(seg)] += seg
            busy_until = place + len(seg)
            shifts.append((place - due) / SR)
            first_place = place if first_place is None else first_place
        if first_place is None:
            continue
        lags += shifts
        transcript.append({"id": it["id"], "start_s": round(it["offset_ms"] / 1000, 2), "speech_at_s": round(first_place / SR, 2),
                           "queued_s": round(max(shifts), 2), "translation": r.get("translated_text", ""),
                           "blocks": [(round(b["write_ms"] / 1000, 2), b["text"]) for b in blocks.get(it["id"], {}).get("blocks", [])]})
    n = max(len(source), busy_until)
    track, source = track[:n], np.pad(source, (0, max(0, n - len(source))))[:n]
    peak = max(1e-6, float(np.abs(track).max()))
    track = track / peak * 0.9 if peak > 0.9 else track
    out.mkdir(parents=True, exist_ok=True)
    sf.write(out / "translation.wav", track, SR)
    sf.write(out / "stereo.wav", np.stack([source, track], axis=1), SR)
    sf.write(out / "mix.wav", np.clip(0.35 * source + track, -1, 1), SR)
    for name in ("translation", "stereo", "mix"):   # compact copies for the web page
        write_ogg(out / f"{name}.ogg", sf.read(out / f"{name}.wav", dtype="float32")[0])
    (out / "transcript.json").write_text(json.dumps(transcript, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({"out": str(out), "seconds": round(n / SR, 1), "pieces": len(transcript),
                      "queued_mean_s": round(float(np.mean(lags)), 2) if lags else None}), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("segment")
    s.add_argument("--audio", required=True)
    s.add_argument("--direction", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--min-pause", type=int, default=300)
    s.add_argument("--min-s", type=float, default=3.0)
    s.add_argument("--max-s", type=float, default=15.0)
    s.add_argument("--margin-db", type=float, default=6.0)
    s.add_argument("--limit-s", type=float, default=0.0)
    t = sub.add_parser("stitch")
    t.add_argument("--items", required=True)
    t.add_argument("--infer", required=True)
    t.add_argument("--blocks", default="")
    t.add_argument("--out", required=True)
    t.add_argument("--catchup-max", type=float, default=1.0,
                   help="when a block starts late, time-stretch it up to this rate (1: off), pitch unchanged")
    t.add_argument("--catchup-per-s", type=float, default=0.05, help="rate gained per second behind")
    v = sub.add_parser("voices")
    v.add_argument("--items", required=True)
    v.add_argument("--blocks", required=True)
    v.add_argument("--audio", required=True, help="the whole recording")
    v.add_argument("--out", required=True, help="items jsonl with voice_audio")
    v.add_argument("--window-s", type=float, default=8.0)
    args = ap.parse_args()
    {"segment": cmd_segment, "stitch": cmd_stitch, "voices": cmd_voices}[args.cmd](args)


if __name__ == "__main__":
    main()

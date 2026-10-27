"""Fluency plan v2: the listener's timeline of whole talks, and a bounded playback tempo (docs/m2_fluency_plan_v2.zh-CN.md).

An si_infer run speaks each segment (a RealSI sentence, an ACL piece) on its own clock, starting with an empty queue.
Here the segments go back onto their talk: a block is committed at the segment's offset in the talk plus its write time,
and one talker plays the blocks in order, each at the later of its commit and the end of the block before it -- the
queue now carries over from one segment to the next.

--trim cuts each block's edge silence beyond 60 ms (about 5% of the block time) before it is played.
--tempo plays each block at a rate chosen when it starts, from what the player knows then (causal):
  * blocks already committed and waiting behind it: faster, 1.05 + 0.05 per second waiting, at most --tempo-max (1.15)
  * it ends its segment (the source sentence is over): at least --tempo-final (1.10)
  * nothing waiting and the segment goes on: --tempo-slow (0.92; the talker stretches while the Thinker is still listening)
  * the rate moves at most 0.1 from one block to the next; WSOLA keeps the pitch.

  talk mode     per talk and pooled: queue wait (start - commit), time with > 2 s of committed speech unplayed, drift
                (queue wait against talk time, s/min), speech still going after the talk's source ends, continuous LAAL
                (every unit at its block's start on the talk clock, minus the segment offset), and pauses measured on
                the stitched talk audio (VAD as si_fluency_stats)
  --write-run   segment mode with --tempo: a copy of the run whose blocks and wavs are the re-timed ones (results.jsonl,
                <id>_simul.wav / _speech.wav), for si_fluency_compare and the ASR

    python -m experiments.uniss_qwen3_omni_v1.evaluation.si_talk_timeline --run <realsi_dir|acl dir> --items realsi:en2zh|acl \
        [--tempo] --out <json> [--wav-dir <dir>] [--write-run <dir>]
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf

from experiments.uniss_qwen3_omni_v1.evaluation.si_fluency_stats import n_units, vad_pauses
from experiments.uniss_qwen3_omni_v1.evaluation.si_metrics import laal, units

SR = 16000


def wsola(x: np.ndarray, r: float) -> np.ndarray:
    """x played r times faster (r > 1: shorter), pitch kept: waveform-similarity overlap-add with 20 ms frames, a
    periodic Hann window at 50% overlap (constant sum) and a normalised cross-correlation search of +-5 ms. At rate 1 it
    costs UTMOS 0.05 on the Talker's speech; the first version (30 ms, symmetric window, raw correlation) cost 0.24."""
    if abs(r - 1.0) < 1e-3 or len(x) < 2000:
        return x
    L, Ha, tol = 320, 160, 80
    w = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(L) / L)
    n_out = int(round(len(x) / r))
    xp = np.pad(x, (tol, L + tol + int(Ha * r) + Ha + 1))
    n_frames = n_out // Ha + 1
    y, ws = np.zeros(n_frames * Ha + L), np.zeros(n_frames * Ha + L)
    delta = 0
    for k in range(n_frames):
        a = int(round(k * Ha * r)) + tol + delta
        seg = xp[a: a + L]
        if len(seg) < L:
            break
        y[k * Ha: k * Ha + L] += seg * w
        ws[k * Ha: k * Ha + L] += w
        nat = xp[a + Ha: a + Ha + L]   # what would follow this frame naturally
        nxt = int(round((k + 1) * Ha * r)) + tol
        region = xp[nxt - tol: nxt + tol + L]
        if len(nat) < L or len(region) < 2 * tol + L:
            break
        c = np.correlate(region, nat, "valid") / np.sqrt(np.convolve(region ** 2, np.ones(L), "valid") + 1e-9)
        delta = int(np.argmax(c)) - tol
    return (y / np.maximum(ws, 1e-3))[:n_out].astype(np.float32)


def load(run: Path, items: str):
    rows = [json.loads(l) for l in open(run / "results.jsonl", encoding="utf-8")]
    if items == "acl":
        meta = {r["id"]: (r["talk"], r["offset_ms"]) for r in map(json.loads, open("data/acl6060_dev_v1/items.jsonl"))}
    else:
        from experiments.uniss_qwen3_omni_v1.evaluation.si_infer import realsi_items
        meta = {it["id"]: (Path(it["wav"]).stem, it["start"]) for it in realsi_items(items.split(":")[1])}
    return rows, meta


def trim_edges(a: np.ndarray, keep: float = 0.06) -> np.ndarray:
    """A block without the silence before its first and after its last sound, beyond keep seconds."""
    n = len(a) // 160
    if n < 3:
        return a
    db = 10 * np.log10(np.mean(a[: n * 160].reshape(n, 160) ** 2, 1) + 1e-12)
    idx = np.flatnonzero(db > np.percentile(db, 90) - 35)
    if not len(idx):
        return a
    k = int(keep * 100)
    return a[max(0, idx[0] - k) * 160: min(n, idx[-1] + 1 + k) * 160]


def schedule(segs: list[dict], tempo: bool, fast_max: float = 1.15, slow: float = 0.92, final: float = 1.10) -> list[dict]:
    """segs: [{offset, blocks:[{commit_ms (talk clock), dur_ms, audio, final}]}] -> played blocks in order."""
    blocks = sorted((b for s in segs for b in s["blocks"]), key=lambda b: (b["commit"], b["k"]))
    out, free, prev_r = [], -1e9, 1.0
    for i, b in enumerate(blocks):
        start = max(b["commit"], free)
        r = 1.0
        if tempo:
            waiting = sum(x["dur"] for x in blocks[i + 1:] if x["commit"] <= start) / 1000
            if waiting > 0:
                r = min(fast_max, 1.05 + 0.05 * waiting)
            elif not b["final"]:
                r = slow
            if b["final"]:
                r = max(r, final)
            r = float(np.clip(r, prev_r - 0.1, prev_r + 0.1))
            prev_r = r
        audio = wsola(b["audio"], r) if tempo else b["audio"]
        d = len(audio) / SR * 1000
        out.append({**b, "start": start, "play_dur": d, "rate": r, "audio": audio})
        free = start + d
    return out


def segments(row: dict, offset: int, k0: int, trim: bool = False) -> dict:
    x, sr = sf.read(row["audio_path"], dtype="float32", always_2d=True)
    x = x.mean(1)
    assert sr == SR
    bl = []
    for k, b in enumerate(row["blocks"]):
        a = x[int(b["start_ms"] * 16): int((b["start_ms"] + b["dur_ms"]) * 16)]
        if trim:
            a = trim_edges(a)
        bl.append({"seg": row["id"], "k": k0 + k, "commit": offset + b["write_ms"], "dur": len(a) / 16, "audio": a,
                   "final": b["final"], "text": b["text"], "offset": offset})
    return {"offset": offset, "blocks": bl, "dur_ms": row["source_duration_ms"]}


def talk_metrics(played: list[dict], talk_end_ms: float, wav: Path | None) -> dict:
    waits = np.array([(b["start"] - b["commit"]) / 1000 for b in played])
    commits = np.array([b["commit"] / 1000 for b in played])
    t0, t1 = commits.min(), max((b["start"] + b["play_dur"]) / 1000 for b in played)
    grid = np.arange(t0, t1, 0.1)
    st = np.array([b["start"] / 1000 for b in played])
    en = st + np.array([b["play_dur"] / 1000 for b in played])
    dur = en - st
    backlog = np.array([np.sum(np.where(commits <= t, np.where(t < st, dur, np.clip(en - t, 0, None)), 0)) for t in grid])
    drift = float(np.polyfit(commits / 60, waits, 1)[0]) if len(played) > 2 else 0.0
    res = {"blocks": len(played), "wait": waits, "backlog_gt2": float(np.mean(backlog > 2.0)) if len(grid) else 0.0,
           "backlog_max": float(backlog.max()) if len(grid) else 0.0, "drift_s_per_min": drift,
           "after_talk_s": float(t1 - talk_end_ms / 1000), "span_s": float(t1 - t0)}
    if wav is not None:   # the stitched talk, for its pauses
        n = int(t1 * SR) + SR
        y = np.zeros(n, dtype=np.float32)
        for b in played:
            a = int(b["start"] * 16)
            y[a: a + len(b["audio"])] += b["audio"][: max(0, n - a)]
        wav.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(wav), np.clip(y, -1, 1), SR)
        pauses, first, last, sounding = vad_pauses(str(wav))
        p = np.array([b - a for a, b in pauses]) if pauses else np.zeros(0)
        res.update(sounding_s=sounding, pause_span_s=last - first, n_pause_ge1=int(np.sum(p >= 1.0)),
                   n_pause_ge2=int(np.sum(p >= 2.0)), pause_s_ge1=float(np.sum(p[p >= 1.0])))
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True, help="si_infer run dir (results.jsonl with blocks, *_simul.wav)")
    ap.add_argument("--items", required=True, help="realsi:en2zh | realsi:zh2en | acl")
    ap.add_argument("--tempo", action="store_true")
    ap.add_argument("--tempo-max", type=float, default=1.15, help="--tempo: the fastest rate")
    ap.add_argument("--tempo-slow", type=float, default=0.92, help="--tempo: the rate with nothing waiting (1: never slower)")
    ap.add_argument("--tempo-final", type=float, default=1.10, help="--tempo: the least rate of a segment's last block")
    ap.add_argument("--trim", action="store_true", help="cut each block's edge silence beyond 60 ms before playing it")
    ap.add_argument("--out", required=True, help="talk-mode summary json")
    ap.add_argument("--wav-dir", default="", help="write the stitched talks here (and measure their pauses)")
    ap.add_argument("--write-run", default="", help="segment mode: write the re-timed run here (needs --tempo)")
    args = ap.parse_args()
    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"refusing to overwrite {out}")
    rows, meta = load(Path(args.run), args.items)
    tgt = rows[0]["tgt_lang"]

    if args.write_run:   # segment by segment, each on its own clock, re-timed
        wr = Path(args.write_run)
        if wr.exists():
            raise SystemExit(f"refusing to overwrite {wr}")
        (wr / "wav").mkdir(parents=True)
        with open(wr / "results.jsonl", "x", encoding="utf-8") as fh:
            for row in rows:
                if not row.get("blocks"):
                    fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                    continue
                played = schedule([segments(row, 0, 0, args.trim)], args.tempo, args.tempo_max, args.tempo_slow, args.tempo_final)
                n = int(max(b["start"] + b["play_dur"] for b in played) * 16) + int(0.3 * SR)
                y = np.zeros(n, dtype=np.float32)
                for b in played:
                    a = int(b["start"] * 16)
                    y[a: a + len(b["audio"])] += b["audio"]
                sim, sp = wr / "wav" / f"{row['id']}_simul.wav", wr / "wav" / f"{row['id']}_speech.wav"
                sf.write(str(sim), np.clip(y, -1, 1), SR)
                sf.write(str(sp), np.concatenate([b["audio"] for b in played]), SR)
                blocks = [{**ob, "start_ms": int(round(b["start"])), "dur_ms": int(round(b["play_dur"])), "rate": round(b["rate"], 3)}
                          for ob, b in zip(row["blocks"], sorted(played, key=lambda b: b["k"]))]
                fh.write(json.dumps({**row, "blocks": blocks, "audio_path": str(sim.resolve()), "speech_path": str(sp.resolve()),
                                     "audio_duration_seconds": round(n / SR, 3)}, ensure_ascii=False) + "\n")

    talks = defaultdict(list)
    for row in rows:
        if row.get("blocks") and row["id"] in meta:
            talks[meta[row["id"]][0]].append(row)
    per, laals = {}, []
    for talk, rs in sorted(talks.items()):
        rs.sort(key=lambda r: meta[r["id"]][1])
        segs, k0 = [], 0
        for r in rs:
            segs.append(segments(r, meta[r["id"]][1], k0, args.trim))
            k0 += len(r["blocks"])
        played = schedule(segs, args.tempo, args.tempo_max, args.tempo_slow, args.tempo_final)
        end = max(meta[r["id"]][1] + r["source_duration_ms"] for r in rs)
        per[talk] = talk_metrics(played, end, Path(args.wav_dir) / f"{talk}.wav" if args.wav_dir else None)
        by_seg = defaultdict(list)
        for b in played:
            by_seg[b["seg"]].append(b)
        for r in rs:
            ds = [b["start"] - b["offset"] for b in sorted(by_seg[r["id"]], key=lambda b: b["k"]) for _ in units(b["text"], tgt)]
            ref = r.get("translation_ref") or ""
            if ds:
                laals.append(laal(ds, r["source_duration_ms"], len(units(ref, tgt)) if ref else 0))
    W = np.concatenate([p["wait"] for p in per.values()])
    U = sum(n_units(r.get("translated_text", ""), tgt) for r in rows)
    summ = {
        "talks": len(per), "blocks": int(len(W)),
        "wait_s_p50_p90_max": [round(float(np.median(W)), 2), round(float(np.percentile(W, 90)), 2), round(float(W.max()), 2)],
        "time_backlog_gt2s": round(float(np.mean([p["backlog_gt2"] for p in per.values()])), 3),
        "backlog_max_s_p50": round(float(np.median([p["backlog_max"] for p in per.values()])), 2),
        "drift_s_per_min_p50": round(float(np.median([p["drift_s_per_min"] for p in per.values()])), 3),
        "after_talk_s_p50": round(float(np.median([p["after_talk_s"] for p in per.values()])), 2),
        "continuous_laal_s": round(float(np.mean(laals)) / 1000, 3),
    }
    if args.wav_dir:
        snd = sum(p["sounding_s"] for p in per.values())
        span = sum(p["pause_span_s"] for p in per.values())
        summ.update(silence_ratio=round(1 - snd / span, 3), pauses_ge1_per_min_speech=round(60 * sum(p["n_pause_ge1"] for p in per.values()) / snd, 2),
                    pauses_ge2_per_min_speech=round(60 * sum(p["n_pause_ge2"] for p in per.values()) / snd, 2),
                    pause_s_ge1_per_100u=round(100 * sum(p["pause_s_ge1"] for p in per.values()) / max(U, 1), 2))
    summ["per_talk"] = {t: {k: (round(v, 3) if isinstance(v, float) else v) for k, v in p.items() if k != "wait"} for t, p in per.items()}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summ, ensure_ascii=False, indent=1))
    print(json.dumps({k: v for k, v in summ.items() if k != "per_talk"}), flush=True)


if __name__ == "__main__":
    main()

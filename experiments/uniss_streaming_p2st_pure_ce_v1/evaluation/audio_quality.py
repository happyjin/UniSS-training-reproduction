#!/usr/bin/env python3
"""Audio-quality metrics for streaming translation output.

The metrics the m3 run published -- voiced fraction, gap count and length,
high-band energy share, peak, internal silence, seam energy jumps -- were
computed by an ad-hoc script that was never committed, so their exact
thresholds are unrecoverable.  Rather than guess them, this module defines them
once and is applied to *both* runs' wav files.  A comparison computed by one
piece of code is exact whether or not it reproduces the original definition,
and ``--validate`` prints this module's numbers on m3's own files next to the
numbers m3 published so the reader can see how close the definitions are.

Definitions, all on the right channel (the translation) at 16 kHz:

voiced          20 ms frames whose RMS exceeds ``floor``, which is 2% of the
                file's 99th-percentile frame RMS but never below 1e-3.  A
                relative floor is necessary because the runs differ in peak by
                more than a factor of two.
gap             a run of unvoiced frames at least 200 ms long that lies between
                two voiced frames.  Leading and trailing silence is excluded:
                it is latency and truncation, measured separately.
internal
silence         unvoiced frames between the first and last voiced frame, as a
                fraction of that span.  This is the quantity that reached 12.7%
                on m3's long audio and is what "choppy" means numerically.
high band       energy at or above 4 kHz as a share of total energy, over
                voiced frames only.  Normal speech sits at 5-15%; m3 measured
                30.6% on its worst sample, which is the electric-artefact
                signature.
seam jumps      per thousand frames, the count of adjacent voiced-frame pairs
                whose RMS ratio exceeds 4x in either direction.  Fragment
                boundaries show up here.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf

SAMPLE_RATE = 16000
FRAME_MS = 20
FRAME = SAMPLE_RATE * FRAME_MS // 1000
GAP_MS = 200
HIGH_BAND_HZ = 4000
JUMP_RATIO = 4.0


def _frames(x: np.ndarray) -> np.ndarray:
    n = len(x) // FRAME
    if n <= 0:
        return np.zeros((0, FRAME), dtype=np.float32)
    return x[: n * FRAME].reshape(n, FRAME)


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    start = None
    for i, v in enumerate(mask):
        if v and start is None:
            start = i
        elif not v and start is not None:
            out.append((start, i))
            start = None
    if start is not None:
        out.append((start, len(mask)))
    return out


def measure(x: np.ndarray) -> dict:
    x = np.asarray(x, dtype=np.float32)
    fr = _frames(x)
    if not len(fr):
        return {"frames": 0}
    rms = np.sqrt(np.mean(fr.astype(np.float64) ** 2, axis=1))
    floor = max(1e-3, 0.02 * float(np.percentile(rms, 99)))
    voiced = rms > floor
    nv = int(voiced.sum())
    out: dict = {
        "seconds": len(x) / SAMPLE_RATE,
        "frames": int(len(fr)),
        "peak": float(np.max(np.abs(x))) if len(x) else 0.0,
        "rms_overall": float(np.sqrt(np.mean(x.astype(np.float64) ** 2))),
        "voiced_frames": nv,
        "voiced_fraction": nv / len(fr),
        "voiced_floor": floor,
    }
    if nv == 0:
        return out
    first, last = int(np.argmax(voiced)), int(len(voiced) - 1 - np.argmax(voiced[::-1]))
    span = last - first + 1
    out["leading_silence_ms"] = first * FRAME_MS
    out["trailing_silence_ms"] = (len(fr) - 1 - last) * FRAME_MS
    out["internal_silence_fraction"] = float((span - voiced[first : last + 1].sum()) / span)
    gaps = [
        (b - a) * FRAME_MS
        for a, b in _runs(~voiced)
        if a > first and b <= last + 1 and (b - a) * FRAME_MS >= GAP_MS
    ]
    segs = [(b - a) * FRAME_MS for a, b in _runs(voiced)]
    out["gaps"] = len(gaps)
    out["gap_ms_median"] = float(np.median(gaps)) if gaps else 0.0
    out["gap_ms_total"] = float(sum(gaps))
    out["voiced_segments"] = len(segs)
    out["voiced_segment_ms_median"] = float(np.median(segs))
    win = np.hanning(FRAME).astype(np.float32)
    spec = np.abs(np.fft.rfft(fr[voiced] * win, axis=1)) ** 2
    freq = np.fft.rfftfreq(FRAME, 1.0 / SAMPLE_RATE)
    total = float(spec.sum())
    out["high_band_fraction"] = float(spec[:, freq >= HIGH_BAND_HZ].sum() / total) if total else 0.0
    idx = np.flatnonzero(voiced)
    pair = [
        (idx[i], idx[i + 1])
        for i in range(len(idx) - 1)
        if idx[i + 1] == idx[i] + 1
    ]
    jumps = sum(
        1
        for a, b in pair
        if max(rms[a], rms[b]) > JUMP_RATIO * max(min(rms[a], rms[b]), 1e-12)
    )
    out["seam_jumps_per_1k_frames"] = 1000.0 * jumps / len(fr)
    return out


def measure_file(path: Path) -> dict:
    data, rate = sf.read(str(path), dtype="float32")
    if int(rate) != SAMPLE_RATE:
        raise ValueError(f"{path} is {rate} Hz, expected {SAMPLE_RATE}")
    if data.ndim == 2 and data.shape[1] >= 2:
        out = {"channels": int(data.shape[1])}
        out["source"] = measure(data[:, 0])
        out["translation"] = measure(data[:, 1])
        return out
    return {"channels": 1, "translation": measure(data if data.ndim == 1 else data[:, 0])}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wav", action="append", dest="wavs", default=[])
    parser.add_argument("--glob", action="append", dest="globs", default=[])
    parser.add_argument("--label", default="run")
    parser.add_argument("--output")
    args = parser.parse_args()
    paths: list[Path] = [Path(p) for p in args.wavs]
    for pattern in args.globs:
        paths.extend(sorted(Path().glob(pattern)))
    rows = {}
    for path in paths:
        rows[path.name] = measure_file(path)
    print(
        "%-46s %7s %7s %6s %6s %8s %7s %8s %8s"
        % ("file", "有声%", "内静%", "空隙", "空隙中位", "段中位ms", ">4kHz", "峰值", "突变/千帧")
    )
    for name, row in rows.items():
        t = row["translation"]
        if not t.get("voiced_frames"):
            print("%-46s %7s" % (name[:46], "无声"))
            continue
        print(
            "%-46s %6.1f%% %6.1f%% %6d %8.0f %8.0f %6.1f%% %8.3f %8.1f"
            % (
                name[:46],
                100 * t["voiced_fraction"],
                100 * t["internal_silence_fraction"],
                t["gaps"],
                t["gap_ms_median"],
                t["voiced_segment_ms_median"],
                100 * t["high_band_fraction"],
                t["peak"],
                t["seam_jumps_per_1k_frames"],
            )
        )
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(
            json.dumps({"label": args.label, "files": rows}, indent=1, sort_keys=True) + "\n"
        )
        print("wrote", args.output)


if __name__ == "__main__":
    main()

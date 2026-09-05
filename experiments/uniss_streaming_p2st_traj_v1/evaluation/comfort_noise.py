"""Fill the timeline's gaps with room tone instead of digital zero.

Why
---
SimulS2ST-Omni can emit a decoded chunk of silence on every wait step
(``--enable-wait-silence-decode`` in their S2ST agent), so their output is a
continuous stream.  Ours places fragments on a timeline and leaves the space
between them as ``np.zeros``.  Measured on one placed file at iter 200 scale
4.0: of 462 gap frames, **340 are exactly zero**, and their VoiceBox silence
sits at about 5.2e-4 RMS.

Digital zero is not what a pause sounds like.  A telephone system inserts
comfort noise for exactly this reason: an abrupt drop to a numerically silent
floor is heard as the line dropping, while a low room tone is heard as someone
not talking.  18% of our timeline is that drop.

What this does and does not change
----------------------------------
It fills **only** frames that are already silent, at the level the *source*
recording's own pauses sit at, and it fades in and out across the gap edges so
the join is not itself a step.  It never touches a voiced sample, so it cannot
change a word, and ASR-BLEU is unaffected by construction -- which is the
point: this is the one remaining inference-side change whose downside is zero.

It is a post-process on audio that already exists, so it needs no GPU and no
re-inference.  That also means it is honest about what it is: cosmetic.  It
does not make a truncated fragment whole, and it does not translate anything
the model failed to translate.

Level
-----
The first design matched the *source* recording's own noise floor.  That was
wrong on this data and the measurement said so immediately: seven of the eight
longform sources report a floor of exactly ``0.00e+00``.  LibriSpeech and
Emilia are cleaned corpora whose pauses are digitally zeroed, so there is no
ambience to match.

The target is instead the **translation's own** pause level -- the median RMS
of frames that are quiet but not numerically silent, which is what BiCodec
puts between the words it does speak.  Matching that makes an inserted gap
indistinguishable from a pause the model produced itself, which is a stronger
property than matching the input's room, and it degrades gracefully: a decoder
that emits true zeros everywhere falls back to ``--fixed-level``, whose
default is the 5.2e-4 measured from SimulS2ST-Omni's VoiceBox silence.

A floor and a ceiling guard the degenerate cases -- a signal with no
measurable pause at all, and one so noisy that matching it would be worse than
silence.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import numpy as np
import soundfile as sf

SAMPLE_RATE = 16_000
FRAME_MS = 15
FRAME = SAMPLE_RATE * FRAME_MS // 1000
FLOOR_FRACTION = 0.02
# Below this the fill would be inaudible and pointless; above it the fill
# would be noticeable as hiss in its own right.  Both are RMS in full scale.
MIN_LEVEL = 1.0e-4
MAX_LEVEL = 3.0e-3
FADE_MS = 10
FADE = SAMPLE_RATE * FADE_MS // 1000


def _frame_rms(audio: np.ndarray) -> np.ndarray:
    frames = len(audio) // FRAME
    if frames == 0:
        return np.zeros(0)
    block = audio[: frames * FRAME].reshape(frames, FRAME)
    return np.sqrt(np.mean(block.astype(np.float64) ** 2, axis=1))


# SimulS2ST-Omni's VoiceBox silence, measured at 5.18e-4 RMS.  Used only when
# the signal offers no pause level of its own.
FALLBACK_LEVEL = 5.2e-4


def noise_floor(audio: np.ndarray, *, ignore_zeros: bool = False) -> float:
    """The level the signal sits at when nobody is speaking.

    ``ignore_zeros`` skips numerically silent frames.  It is what to use on a
    generated channel, where a zero frame means "the decoder emitted nothing
    here", not "the room is this quiet" -- averaging those in drags the target
    to zero and the fill never happens.
    """
    rms = _frame_rms(np.asarray(audio, dtype=np.float32))
    if not len(rms):
        return 0.0
    threshold = FLOOR_FRACTION * float(np.percentile(rms, 99))
    quiet = rms[rms <= max(threshold, 1e-12)]
    if ignore_zeros:
        quiet = quiet[quiet > 0.0]
    if not len(quiet):
        return 0.0
    return float(np.median(quiet))


def _shaped_noise(count: int, level: float, seed: int) -> np.ndarray:
    """Low-passed noise at ``level`` RMS.

    White noise at a room-tone level reads as tape hiss because its energy is
    flat to 8 kHz, where real room tone is not.  One pole of smoothing is
    enough to move it out of the way; this is a filler, not a synthesiser.
    """
    if count <= 0 or level <= 0:
        return np.zeros(max(0, count), dtype=np.float32)
    rng = np.random.default_rng(seed)
    raw = rng.standard_normal(count + 4)
    smoothed = np.empty_like(raw)
    smoothed[0] = raw[0]
    alpha = 0.35
    for index in range(1, len(raw)):
        smoothed[index] = alpha * raw[index] + (1.0 - alpha) * smoothed[index - 1]
    out = smoothed[4:]
    current = float(np.sqrt(np.mean(out**2)))
    if current <= 0:
        return np.zeros(count, dtype=np.float32)
    return (out * (level / current)).astype(np.float32)


def _silent_runs(audio: np.ndarray) -> list[tuple[int, int]]:
    """Sample ranges the translation channel is silent over."""
    rms = _frame_rms(audio)
    if not len(rms):
        return []
    threshold = FLOOR_FRACTION * float(np.percentile(rms, 99))
    quiet = rms <= max(threshold, 1e-12)
    runs: list[tuple[int, int]] = []
    start = None
    for index, value in enumerate(quiet):
        if value and start is None:
            start = index
        elif not value and start is not None:
            runs.append((start * FRAME, index * FRAME))
            start = None
    if start is not None:
        runs.append((start * FRAME, len(audio)))
    return runs


def fill(
    translation: np.ndarray, level: float, *, seed: int = 0
) -> tuple[np.ndarray, dict[str, float]]:
    """Return the translation with its silent runs replaced by room tone."""
    audio = np.array(translation, dtype=np.float32, copy=True)
    level = float(min(max(level, 0.0), MAX_LEVEL))
    stats = {"level": level, "runs": 0, "filled_samples": 0}
    if level < MIN_LEVEL:
        return audio, stats
    for start, end in _silent_runs(audio):
        length = end - start
        if length <= 2 * FADE:
            continue
        noise = _shaped_noise(length, level, seed + start)
        ramp = np.linspace(0.0, 1.0, FADE, dtype=np.float32)
        noise[:FADE] *= ramp
        noise[-FADE:] *= ramp[::-1]
        # Add rather than replace: whatever decoder tail is already there is
        # part of the signal, and overwriting it would be a new discontinuity.
        audio[start:end] += noise
        stats["runs"] += 1
        stats["filled_samples"] += length
    return audio, stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-arm", required=True, help="new arm directory name")
    parser.add_argument("--label", default="comfort")
    parser.add_argument("--seed", type=int, default=20260905)
    parser.add_argument(
        "--level-source",
        default="translation",
        choices=("translation", "source", "fixed"),
        help=(
            "where the fill level comes from.  'translation' matches the "
            "model's own pauses and is the default; 'source' matches the "
            "input recording, which is zero on cleaned corpora"
        ),
    )
    parser.add_argument("--fixed-level", type=float, default=FALLBACK_LEVEL)
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    out_root = manifest_path.parent.parent / args.output_arm
    rows: list[dict[str, object]] = []
    samples: list[dict] = []
    for sample in data["samples"]:
        stereo_path = manifest_path.parent / "stereo" / f"{sample['sample_id']}.wav"
        placed_path = Path(str(sample["translation_placed"]))
        if not stereo_path.exists() or not placed_path.exists():
            continue
        stereo, rate = sf.read(str(stereo_path), dtype="float32")
        if int(rate) != SAMPLE_RATE:
            raise ValueError(f"{stereo_path} is {rate} Hz")
        source, translation = stereo[:, 0], stereo[:, 1]
        if args.level_source == "source":
            level = noise_floor(source)
        elif args.level_source == "fixed":
            level = float(args.fixed_level)
        else:
            level = noise_floor(translation, ignore_zeros=True)
        if level < MIN_LEVEL:
            level = float(args.fixed_level)
        filled, stats = fill(translation, level, seed=args.seed)

        for target_dir, channels in (
            ("stereo", np.stack([source, filled], axis=1)),
            ("translation_placed", filled),
        ):
            target = out_root / target_dir / f"{sample['sample_id']}.wav"
            target.parent.mkdir(parents=True, exist_ok=True)
            sf.write(str(target), channels, SAMPLE_RATE, subtype="PCM_16")

        new_sample = dict(sample)
        new_sample["translation_placed"] = str(
            out_root / "translation_placed" / f"{sample['sample_id']}.wav"
        )
        samples.append(new_sample)
        rows.append(
            {
                "sample_id": sample["sample_id"],
                "measured_level": level,
                "level_source": args.level_source,
                "applied_level": stats["level"],
                "runs_filled": stats["runs"],
                "filled_seconds": stats["filled_samples"] / SAMPLE_RATE,
            }
        )
    (out_root / "MANIFEST.json").write_text(
        json.dumps({"samples": samples}, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
    )
    report = {
        "schema_version": "uniss_streaming_p2st_comfort_noise_v1",
        "source_manifest": str(manifest_path.resolve()),
        "label": args.label,
        "level_source": args.level_source,
        "median_level": (
            statistics.median(float(r["measured_level"]) for r in rows)
            if rows
            else None
        ),
        "rows": rows,
    }
    (out_root / "COMFORT_NOISE.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"{args.label}: {len(rows)} files, median level "
        f"{report['median_level']:.3e} from {args.level_source}, "
        f"filled {sum(float(r['filled_seconds']) for r in rows):.1f}s"
    )
    print(f"-> {out_root}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Place each speech fragment where it was actually emitted, not at t=0.

`build_stereo_demos` writes the translation into `stereo[:len(translation), 1]`,
and the mono translation it reads is `np.concatenate(chunks)` -- every fragment
glued back to back from zero.  The emission schedule is discarded.

On `emilia_zh_0003980703` that is a 6.2 second error: the model emits 13,160 ms
of speech spread across a 19,380 ms source, and the demo compresses all of it
into the first 13.16 seconds.  Listening to that, the translation appears to run
ahead of the source and to finish long before it -- which is what a listener
reported, and which the model does not actually do.  Checked against the gold
alignment, the model translates only source it has already heard: at event 7,
1920 ms in, it says "safety" for 安全, and at event 9, 2560 ms in, "to satisfy"
for 来满足.

This renders honestly.  Fragment k becomes available at its event's
`source_end_ms` and starts playing at the later of that time and the end of
fragment k-1, because a single speaker cannot play two fragments at once.  Gaps
where the system had nothing ready stay silent, which is what a real listener
would hear.
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
import soundfile as sf

SAMPLE_RATE = 16000
SAMPLES_PER_TOKEN = SAMPLE_RATE // 50  # BiCodec semantic is 50 tokens/second


def fragment_lengths(events: list[dict]) -> list[tuple[int, int]]:
    """(source_end_ms, token_count) for events that produced speech."""

    out = []
    for event in events:
        value = event.get("semantic_tokens", 0)
        count = len(value) if isinstance(value, (list, tuple)) else int(value)
        if count:
            out.append((int(event["source_end_ms"]), count))
    return out


def place_on_timeline(
    translation: np.ndarray, schedule: list[tuple[int, int]], total_samples: int
) -> tuple[np.ndarray, dict]:
    placed = np.zeros(max(total_samples, 1), dtype=np.float32)
    cursor_in = 0
    cursor_out = 0
    late_ms = []
    for source_end_ms, count in schedule:
        piece = translation[cursor_in : cursor_in + count * SAMPLES_PER_TOKEN]
        cursor_in += count * SAMPLES_PER_TOKEN
        if not len(piece):
            continue
        earliest = int(round(source_end_ms * SAMPLE_RATE / 1000.0))
        start = max(earliest, cursor_out)
        late_ms.append(1000.0 * (start - earliest) / SAMPLE_RATE)
        end = start + len(piece)
        if end > len(placed):
            placed = np.concatenate([placed, np.zeros(end - len(placed), dtype=np.float32)])
        placed[start:end] += piece
        cursor_out = end
    stats = {
        "fragments": len(schedule),
        "placed_seconds": cursor_out / SAMPLE_RATE,
        "concatenated_seconds": len(translation) / SAMPLE_RATE,
        "queueing_delay_ms_mean": float(np.mean(late_ms)) if late_ms else 0.0,
        "queueing_delay_ms_max": float(np.max(late_ms)) if late_ms else 0.0,
        "unused_translation_samples": max(0, len(translation) - cursor_in),
    }
    return placed, stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, help="probe/gate run root")
    parser.add_argument("--selection", required=True)
    parser.add_argument("--sample-id", action="append", dest="sample_ids", default=[])
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()

    selection = json.loads(open(args.selection, encoding="utf-8").read())
    records = {r["sample_id"]: r for r in selection["records"]}
    events_by_sample: dict[str, list[dict]] = {}
    for path in sorted(glob.glob(os.path.join(args.run, "workers", "*.json"))):
        for sample in json.load(open(path, encoding="utf-8")).get("samples", []):
            free = sample.get("e_s2s_free")
            if free:
                events_by_sample[sample["sample_id"]] = free.get("events", [])
    translations = {}
    for path in glob.glob(os.path.join(args.run, "audio", "**", "*.wav"), recursive=True):
        translations[os.path.basename(path)[:-4]] = path

    os.makedirs(args.output_dir, exist_ok=True)
    rows = []
    for sample_id in args.sample_ids or sorted(events_by_sample):
        record = records.get(sample_id)
        events = events_by_sample.get(sample_id)
        audio_path = translations.get(sample_id)
        if not (record and events and audio_path):
            rows.append({"sample_id": sample_id, "status": "missing"})
            continue
        source, rate = sf.read(record["source_audio"], dtype="float32")
        if source.ndim > 1:
            source = source.mean(axis=1)
        if rate != SAMPLE_RATE:
            raise SystemExit(f"{sample_id}: source is {rate} Hz, expected {SAMPLE_RATE}")
        translation, _ = sf.read(audio_path, dtype="float32")
        if translation.ndim > 1:
            translation = translation.mean(axis=1)
        placed, stats = place_on_timeline(
            translation, fragment_lengths(events), len(source)
        )
        peak = float(np.max(np.abs(placed))) if len(placed) else 0.0
        if peak > 0.95:
            placed = placed * (0.95 / peak)
        length = max(len(source), len(placed))
        stereo = np.zeros((length, 2), dtype=np.float32)
        stereo[: len(source), 0] = source
        stereo[: len(placed), 1] = placed
        destination = os.path.join(args.output_dir, f"{sample_id}__timeline__stereo.wav")
        sf.write(destination, stereo, SAMPLE_RATE, subtype="PCM_16")
        row = {
            "sample_id": sample_id,
            "status": "written",
            "stereo": destination,
            "source_seconds": len(source) / SAMPLE_RATE,
            **stats,
        }
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False))
    with open(args.manifest, "w", encoding="utf-8") as handle:
        json.dump({"schema_version": "uniss_timeline_stereo_v1", "rows": rows}, handle, indent=1)
    print(f"MANIFEST={args.manifest}")


if __name__ == "__main__":
    main()

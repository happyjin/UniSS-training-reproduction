"""Render the streaming output under a playout buffer, so it can be heard.

``playout_buffer`` computes the schedule and scores it from timings alone; this
writes the audio that schedule implies.  The fragments are recovered by slicing
``translation_concat`` -- which is the fragments pasted back to back -- at the
cumulative ``durations``, so not a sample of the model's output is altered.
Only when each fragment is heard changes, and it can only ever move later:
``schedule`` treats each fragment's source boundary as a hard lower bound, so
nothing is ever played before the audio that justified it had arrived.

The point of hearing it is that the buffer can only touch gaps *between*
fragments.  A hole inside a fragment is content the model generated and no
schedule can remove it, so the buffered audio is the ceiling of what scheduling
alone can achieve.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import numpy as np
import soundfile as sf

from experiments.uniss_streaming_p2st_traj_v1.evaluation.playout_buffer import (
    SAMPLE_RATE,
    render,
    schedule,
    silence_ratio_from_schedule,
)


def slice_fragments(audio: np.ndarray, durations: list[float]) -> list[np.ndarray]:
    """Cut the packed translation back into its fragments."""
    out: list[np.ndarray] = []
    cursor = 0
    for duration in durations:
        length = int(round(float(duration) * SAMPLE_RATE / 1000))
        end = min(cursor + length, len(audio))
        out.append(audio[cursor:end])
        cursor = end
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--buffer-ms", type=float, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--report", default="")
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    samples = json.loads(Path(args.manifest).read_text(encoding="utf-8"))["samples"]
    rows: list[dict[str, object]] = []
    for sample in samples:
        durations = [float(v) for v in sample.get("durations", [])]
        delays = [float(v) for v in sample.get("delays", [])]
        if not durations or len(durations) != len(delays):
            continue
        audio, rate = sf.read(str(sample["translation_concat"]), dtype="float32")
        if audio.ndim == 2:
            audio = audio[:, -1]
        if int(rate) != SAMPLE_RATE:
            raise ValueError(f"{sample['translation_concat']} is {rate} Hz")
        fragments = slice_fragments(audio, durations)
        starts = schedule(delays, durations, float(args.buffer_ms))
        total = starts[-1] + durations[-1]
        rendered = render(fragments, starts, total)
        path = out / f"{sample['sample_id']}.wav"
        sf.write(str(path), rendered, SAMPLE_RATE, subtype="PCM_16")
        rows.append(
            {
                "sample_id": sample["sample_id"],
                "buffer_ms": float(args.buffer_ms),
                "onset_ms": starts[0],
                "onset_cost_ms": starts[0] - delays[0],
                "span_ms": total - starts[0],
                "silence_ratio": silence_ratio_from_schedule(starts, durations),
                # how far past the end of the source the last word now lands
                "end_offset_ms": total - float(sample.get("source_duration_ms") or 0.0),
                "audio": str(path),
            }
        )
        print(f"  {sample['sample_id']:<26} onset {starts[0]:>7.0f} ms "
              f"(+{starts[0] - delays[0]:.0f})  silence {rows[-1]['silence_ratio']:.3f}",
              flush=True)
    if not rows:
        raise SystemExit("no sample carried the per-fragment timings")
    mean = lambda key: statistics.mean(float(r[key]) for r in rows)
    print(f"buffer {args.buffer_ms:.0f} ms over {len(rows)} samples: "
          f"onset {mean('onset_ms'):.0f} ms (+{mean('onset_cost_ms'):.0f}), "
          f"silence {mean('silence_ratio'):.3f}, "
          f"ends {mean('end_offset_ms'):.0f} ms after the source")
    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(
            json.dumps({"buffer_ms": args.buffer_ms, "samples": rows}, indent=1),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()

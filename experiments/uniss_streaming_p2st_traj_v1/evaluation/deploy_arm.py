"""Render a whole RealSI arm through the settled deployment stack.

The rollout arms are raw: fragments are placed on the real timeline with the
cuts exactly where the model stopped, and nothing smooths them.  The settled
configuration never ships that audio -- it snaps each cut back to the quietest
nearby frame, tapers the edges that border a gap, fills the gaps with comfort
noise and plays out behind a one second buffer.

That distinction decides how a checkpoint should be judged.  A model that
speaks in more, shorter fragments trades silence for cuts, and on raw audio the
cuts are heard in full; under the deployment stack they are snapped to silence
and tapered.  Measuring only the raw arm would charge the model for a defect
the renderer removes.

The output is a sibling arm directory whose manifest is the original with
``translation_placed`` pointing at the rendered audio, so the existing ASR
chain scores it with no change.  ``translation_concat`` is left alone, which
keeps it as a content control: the renderer cannot alter what was said.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import numpy as np
import soundfile as sf

from experiments.uniss_streaming_p2st_traj_v1.evaluation.deploy_render import (
    SAMPLE_RATE,
    deploy,
)
from experiments.uniss_streaming_p2st_traj_v1.evaluation.snap_cuts import snap


def render_sample(
    packed: np.ndarray,
    delays: list[float],
    durations: list[float],
    *,
    window_ms: float,
    buffer_ms: float,
    taper_ms: float,
    gap_ms: float,
    tone: bool,
) -> tuple[np.ndarray, dict]:
    """Snap the cuts, then place, taper, fill and buffer.

    The order is load-bearing and was settled by measurement: snapping first
    means the taper acts on a boundary that already sits in a quiet frame.
    """
    snapped, moved = snap(durations, packed, window_ms=window_ms)
    audio, stats = deploy(
        packed, delays, snapped,
        buffer_ms=buffer_ms, taper_ms=taper_ms, gap_ms=gap_ms, tone=tone,
    )
    stats["cuts_moved"] = float(sum(1 for m in moved if m > 0.0))
    stats["cuts"] = float(len(moved))
    return audio, stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--arm-dir", required=True, help="where the sibling arm goes")
    parser.add_argument("--window-ms", type=float, default=200.0)
    parser.add_argument("--buffer-ms", type=float, default=1000.0)
    parser.add_argument("--taper-ms", type=float, default=40.0)
    parser.add_argument("--gap-ms", type=float, default=120.0)
    parser.add_argument("--no-tone", action="store_true")
    args = parser.parse_args()

    arm = Path(args.arm_dir)
    audio_dir = arm / "translation_placed"
    audio_dir.mkdir(parents=True, exist_ok=True)
    samples = json.loads(Path(args.manifest).read_text(encoding="utf-8"))["samples"]

    rendered, stats = [], []
    for sample in samples:
        durations = [float(v) for v in sample.get("durations", [])]
        delays = [float(v) for v in sample.get("delays", [])]
        if not durations or len(durations) != len(delays):
            continue
        packed, rate = sf.read(str(sample["translation_concat"]), dtype="float32")
        if packed.ndim == 2:
            packed = packed[:, -1]
        if int(rate) != SAMPLE_RATE:
            raise ValueError(f"{sample['translation_concat']} is {rate} Hz")
        audio, row = render_sample(
            packed, delays, durations,
            window_ms=args.window_ms, buffer_ms=args.buffer_ms,
            taper_ms=args.taper_ms, gap_ms=args.gap_ms, tone=not args.no_tone,
        )
        path = audio_dir / f"{sample['sample_id']}.wav"
        sf.write(str(path), audio, SAMPLE_RATE, subtype="PCM_16")
        rendered.append({**sample, "translation_placed": str(path)})
        stats.append(row)

    if not rendered:
        raise SystemExit("no sample carried the per-fragment timings")
    (arm / "MANIFEST.json").write_text(
        json.dumps({"samples": rendered}, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    mean = lambda k: statistics.mean(float(r[k]) for r in stats)
    moved = sum(r["cuts_moved"] for r in stats)
    cuts = sum(r["cuts"] for r in stats)
    print(
        f"{len(rendered)} samples  onset {mean('onset_ms'):.0f} ms  "
        f"edges tapered {mean('edges_tapered'):.2f}/sample  "
        f"cuts snapped {moved:.0f}/{cuts:.0f} ({moved / max(1.0, cuts):.1%})"
    )
    print(f"-> {arm}")


if __name__ == "__main__":
    main()

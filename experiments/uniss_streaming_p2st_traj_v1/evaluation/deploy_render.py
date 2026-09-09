"""The deployable rendering: schedule, then release, then fill.

This composes the three inference-side changes that survived measurement on 777
utterances with paired intervals, in the only order that works:

1. **Playout buffer.**  Fragments are held and paid out continuously, never
   earlier than the source boundary that justified them.  Paired on 777, a
   500 ms buffer drops the hole share 18.6% -> 15.7% and 1000 ms -> 13.4%,
   intervals [2.67%, 3.19%] and [4.79%, 5.65%] on the difference.  This is the
   only step that changes gap *durations*.
2. **Edge release.**  A 40 ms raised cosine on fragments a gap still follows,
   and the mirror attack after.  40.2% of boundaries stop inside a phone, so
   without this the speech ends at full amplitude and the gap is heard as a
   cut.  It must run on the *buffered* schedule: the buffer removes most gaps,
   and tapering a join the buffer has closed would dig a hole into a phone
   that is now continuous.
3. **Room tone.**  Fills what silence remains at the translation's own pause
   level, taking exactly-zero gaps from 39% to 0% -- gold and offline are both
   at 0%.

Nothing here changes a semantic code, so the words are the model's own and
ASR-BLEU moves only through intelligibility, not through content.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import numpy as np
import soundfile as sf

from experiments.uniss_streaming_p2st_traj_v1.evaluation import comfort_noise as cn
from experiments.uniss_streaming_p2st_traj_v1.evaluation.buffered_demo import (
    slice_fragments,
)
from experiments.uniss_streaming_p2st_traj_v1.evaluation.edge_taper import taper_edges
from experiments.uniss_streaming_p2st_traj_v1.evaluation.playout_buffer import (
    SAMPLE_RATE,
    render,
    schedule,
)


def deploy(
    packed: np.ndarray,
    delays: list[float],
    durations: list[float],
    *,
    buffer_ms: float,
    taper_ms: float = 40.0,
    gap_ms: float = 120.0,
    tone: bool = True,
    seed: int = 20260909,
) -> tuple[np.ndarray, dict[str, float]]:
    """Render one utterance through the three stages, in order."""
    starts = schedule(delays, durations, float(buffer_ms))
    fragments = slice_fragments(packed, durations)
    total = starts[-1] + durations[-1]
    audio = render(fragments, starts, total)
    audio, tapered = taper_edges(
        audio, starts, durations, taper_ms=taper_ms, gap_ms=gap_ms
    )
    stats = {"onset_ms": starts[0], "span_ms": total - starts[0],
             "edges_tapered": float(tapered), "fill_level": 0.0}
    if tone:
        level = cn.noise_floor(audio, ignore_zeros=True)
        audio, fill = cn.fill(audio, level, seed=seed)
        stats["fill_level"] = float(fill["level"])
        stats["filled_runs"] = float(fill["runs"])
    return audio, stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--buffer-ms", type=float, default=1000.0)
    parser.add_argument("--taper-ms", type=float, default=40.0)
    parser.add_argument("--gap-ms", type=float, default=120.0)
    parser.add_argument("--no-tone", action="store_true")
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
        packed, rate = sf.read(str(sample["translation_concat"]), dtype="float32")
        if packed.ndim == 2:
            packed = packed[:, -1]
        if int(rate) != SAMPLE_RATE:
            raise ValueError(f"{sample['translation_concat']} is {rate} Hz")
        audio, stats = deploy(
            packed, delays, durations,
            buffer_ms=args.buffer_ms, taper_ms=args.taper_ms,
            gap_ms=args.gap_ms, tone=not args.no_tone,
        )
        path = out / f"{sample['sample_id']}.wav"
        sf.write(str(path), audio, SAMPLE_RATE, subtype="PCM_16")
        rows.append({"sample_id": sample["sample_id"], **stats, "audio": str(path)})
    if not rows:
        raise SystemExit("no sample carried the per-fragment timings")
    mean = lambda k: statistics.mean(float(r[k]) for r in rows)
    print(f"buffer {args.buffer_ms:.0f} ms, taper {args.taper_ms:.0f} ms, "
          f"tone {'off' if args.no_tone else 'on'}: {len(rows)} files, "
          f"onset {mean('onset_ms'):.0f} ms, {mean('edges_tapered'):.1f} edges tapered, "
          f"fill level {mean('fill_level'):.2e}")
    print(f"-> {out}")
    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(json.dumps(
            {"buffer_ms": args.buffer_ms, "taper_ms": args.taper_ms,
             "tone": not args.no_tone, "samples": rows}, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()

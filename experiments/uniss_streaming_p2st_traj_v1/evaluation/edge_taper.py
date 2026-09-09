"""Give a fragment that stops mid-phone a release, so the gap after it is heard
as a pause rather than a cut.

Why comfort noise alone was not enough
--------------------------------------
Filling the gaps with room tone took the exactly-zero share from 39% to 0% and
brought the gap level to 8.5e-04 against gold's 1.1e-03, but the step *into*
the gap stayed at 10x where gold and offline sit at 4-5x.  The reason is in the
measurement that started this: 40.2% of fragment boundaries land inside a phone
with full voice on both sides, so the speech does not decay at the boundary --
it stops at full amplitude.  Noise in the gap does not change that edge.

What this adds
--------------
A short raised-cosine taper on the last few milliseconds of a fragment that is
followed by a gap, and the mirror image on the first few of the next fragment.
A phone cut in half then gets a release and an attack, which is what the ear
uses to parse a pause.

Scope and cost
--------------
Only fragments with a gap after them are tapered -- where the next fragment
follows contiguously the audio is left exactly as it was, because there the
phone is already continuous and a taper would put a dip into the middle of it.
The taper is 40 ms by default, a fraction of a phone, and it only ever reduces
amplitude, so it cannot introduce a word.  It does touch voiced samples, unlike
comfort noise, so ASR-BLEU has to be measured rather than assumed.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import numpy as np
import soundfile as sf

SAMPLE_RATE = 16_000
DEFAULT_TAPER_MS = 40.0
DEFAULT_GAP_MS = 120.0


def raised_cosine(count: int, *, rising: bool) -> np.ndarray:
    """Half a raised cosine, 0->1 rising or 1->0 falling."""
    if count <= 0:
        return np.zeros(0, dtype=np.float32)
    t = np.linspace(0.0, 1.0, count, endpoint=True)
    ramp = 0.5 - 0.5 * np.cos(np.pi * t)
    return (ramp if rising else ramp[::-1]).astype(np.float32)


def taper_edges(
    audio: np.ndarray,
    starts: list[float],
    durations: list[float],
    *,
    taper_ms: float = DEFAULT_TAPER_MS,
    gap_ms: float = DEFAULT_GAP_MS,
) -> tuple[np.ndarray, int]:
    """Fade out fragment ends and fade in fragment starts across real gaps."""
    out = audio.copy()
    n = min(len(starts), len(durations))
    width = int(round(taper_ms * SAMPLE_RATE / 1000))
    touched = 0
    for k in range(n):
        end_ms = starts[k] + durations[k]
        # The utterance's own first attack and last release are file edges,
        # not gaps, so they are left alone: 0.0 reads as "no gap here".
        gap_after = (starts[k + 1] - end_ms) if k + 1 < n else 0.0
        if gap_after >= gap_ms:
            stop = int(round(end_ms * SAMPLE_RATE / 1000))
            begin = max(0, stop - width)
            if stop > begin and stop <= len(out):
                out[begin:stop] *= raised_cosine(stop - begin, rising=False)
                touched += 1
        gap_before = (starts[k] - (starts[k - 1] + durations[k - 1])) if k > 0 else 0.0
        if gap_before >= gap_ms:
            begin = int(round(starts[k] * SAMPLE_RATE / 1000))
            stop = min(len(out), begin + width)
            if stop > begin:
                out[begin:stop] *= raised_cosine(stop - begin, rising=True)
                touched += 1
    return out, touched


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="the arm to taper")
    parser.add_argument("--audio-dir", default="", help="override where the placed wavs are")
    parser.add_argument("--output-arm", required=True)
    parser.add_argument("--taper-ms", type=float, default=DEFAULT_TAPER_MS)
    parser.add_argument("--gap-ms", type=float, default=DEFAULT_GAP_MS)
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    out_root = manifest_path.parent.parent / args.output_arm
    (out_root / "translation_placed").mkdir(parents=True, exist_ok=True)
    from experiments.uniss_streaming_p2st_traj_v1.evaluation.playout_buffer import schedule

    rows = []
    for sample in data["samples"]:
        durations = [float(v) for v in sample.get("durations", [])]
        delays = [float(v) for v in sample.get("delays", [])]
        if not durations or len(durations) != len(delays):
            continue
        src = (
            Path(args.audio_dir) / f"{sample['sample_id']}.wav"
            if args.audio_dir
            else Path(str(sample["translation_placed"]))
        )
        audio, rate = sf.read(str(src), dtype="float32")
        if audio.ndim == 2:
            audio = audio[:, -1]
        if int(rate) != SAMPLE_RATE:
            raise ValueError(f"{src} is {rate} Hz")
        starts = schedule(delays, durations, 0.0)
        tapered, touched = taper_edges(
            audio, starts, durations, taper_ms=args.taper_ms, gap_ms=args.gap_ms
        )
        path = out_root / "translation_placed" / f"{sample['sample_id']}.wav"
        sf.write(str(path), tapered, SAMPLE_RATE, subtype="PCM_16")
        rows.append({"sample_id": sample["sample_id"], "edges_tapered": touched,
                     "audio": str(path)})
        sample = dict(sample); sample["translation_placed"] = str(path)
    (out_root / "MANIFEST.json").write_text(
        json.dumps({"samples": [
            {**s, "translation_placed": str(out_root / "translation_placed" / f"{s['sample_id']}.wav")}
            for s in data["samples"]]}, ensure_ascii=False, indent=1), encoding="utf-8")
    (out_root / "EDGE_TAPER.json").write_text(
        json.dumps({"taper_ms": args.taper_ms, "gap_ms": args.gap_ms, "samples": rows},
                   indent=1), encoding="utf-8")
    print(f"tapered {statistics.mean(r['edges_tapered'] for r in rows):.1f} edges per file "
          f"over {len(rows)} files, {args.taper_ms:.0f} ms raised cosine")
    print(f"-> {out_root}")


if __name__ == "__main__":
    main()

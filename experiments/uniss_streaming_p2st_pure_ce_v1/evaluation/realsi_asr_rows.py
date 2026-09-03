#!/usr/bin/env python3
"""Turn a rollout manifest into the rows evaluation/asr_transcribe.py consumes.

Two things this exists to get right.

The audio must be **mono**.  ``asr_transcribe.load_audio_array`` reads with
``always_2d=True`` and then means over channels, so handing it the stereo
listening demo would mix the source speech into the ASR input and inflate
ASR-BLEU.  The rollout writes separate mono files for exactly this reason and
this builder points at them.

The ``mode`` field carries the arm and the audio variant, because
``compute_grouped_bleu`` groups on ``(mode, src_lang, tgt_lang)`` and
``asr_transcribe`` keys its resume set on ``(id, mode)``.  Encoding the arm
there means one ASR pass can cover several arms and the BLEU report separates
them without any extra code.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

VARIANTS = {
    # SimulEval's own convention: the stream as a listener hears it, with
    # silence where the system had nothing ready.
    "placed": "translation_placed",
    # Fragments back to back.  If BLEU differs from `placed` by more than half
    # a point, that gap is ASR sensitivity to inserted silence, not quality.
    "concat": "translation_concat",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    # default=None, not "placed": argparse's append action would otherwise try
    # to append to the string default and raise AttributeError.
    parser.add_argument(
        "--variant", choices=sorted(VARIANTS), action="append", dest="variants",
        default=None,
    )
    args = parser.parse_args()
    variants = args.variants or ["placed"]

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    rows = manifest["samples"]
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with out.open("w", encoding="utf-8") as handle:
        for row in rows:
            for variant in variants:
                audio = Path(row[VARIANTS[variant]])
                if not audio.is_file():
                    raise FileNotFoundError(audio)
                seconds = row[
                    "placed_seconds" if variant == "placed" else "concat_seconds"
                ]
                handle.write(
                    json.dumps(
                        {
                            "id": row["sample_id"],
                            "mode": f"realsi_{row['arm']}_{variant}",
                            "src_lang": row["src_lang"],
                            "tgt_lang": row["tgt_lang"],
                            "audio_path": str(audio.resolve()),
                            "audio_duration_seconds": float(seconds),
                            "translation_ref": row["translation_reference"],
                            "arm": row["arm"],
                            "read_stride": row["read_stride"],
                            "read_step_ms": row["read_step_ms"],
                            "direction": row["direction"],
                            "target_hypothesis": row["target_hypothesis"],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                written += 1
    print(f"{written} rows -> {out}")


if __name__ == "__main__":
    main()

"""Verify an isolated corrected English ASR run before reporting BLEU."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from evaluation.asr_transcribe import WHISPER_ASR_PROTOCOL
from evaluation.io_utils import iter_jsonl, write_json
from training.constants_uniss import normalize_language


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--asr", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    expected = {
        (str(row["id"]), str(row["mode"]))
        for row in iter_jsonl(args.input)
        if row.get("audio_path")
        and not row.get("error")
        and normalize_language(str(row["tgt_lang"])) == "eng"
    }
    seen: set[tuple[str, str]] = set()
    suspicious: list[dict[str, object]] = []
    wrong_protocol: list[dict[str, object]] = []
    for row in iter_jsonl(args.asr):
        key = (str(row["id"]), str(row["mode"]))
        if key in seen:
            raise RuntimeError(f"duplicate corrected ASR row: {key}")
        seen.add(key)
        if row.get("asr_protocol") != WHISPER_ASR_PROTOCOL or not row.get(
            "asr_attention_mask"
        ):
            wrong_protocol.append({"id": key[0], "mode": key[1]})
        duration = float(row.get("audio_duration_seconds") or 0.0)
        word_count = len(str(row.get("asr_text") or "").split())
        if duration > 0 and word_count > max(64, math.ceil(duration * 12.0)):
            suspicious.append(
                {
                    "id": key[0],
                    "mode": key[1],
                    "duration_seconds": duration,
                    "word_count": word_count,
                }
            )
    missing = sorted(expected - seen)
    extra = sorted(seen - expected)
    report = {
        "expected": len(expected),
        "observed": len(seen),
        "missing_count": len(missing),
        "extra_count": len(extra),
        "suspicious_count": len(suspicious),
        "wrong_protocol_count": len(wrong_protocol),
        "protocol": WHISPER_ASR_PROTOCOL,
        "complete": not missing and not extra and not suspicious and not wrong_protocol,
        "missing_preview": missing[:20],
        "extra_preview": extra[:20],
        "suspicious_preview": suspicious[:20],
        "wrong_protocol_preview": wrong_protocol[:20],
    }
    write_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["complete"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

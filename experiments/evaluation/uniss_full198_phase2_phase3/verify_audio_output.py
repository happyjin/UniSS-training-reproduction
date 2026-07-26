"""Strictly verify a smoke/listening audio evaluation directory."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import soundfile as sf

from evaluation.io_utils import iter_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--expected-modes", nargs="+", required=True)
    args = parser.parse_args()

    manifest = list(iter_jsonl(args.manifest))
    results = list(iter_jsonl(args.results))
    expected_count = len(manifest) * len(args.expected_modes)
    if len(results) != expected_count:
        raise ValueError(f"result count={len(results)}, expected={expected_count}")
    keys = [(str(row["id"]), str(row["mode"])) for row in results]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate id/mode result keys")
    if Counter(mode for _, mode in keys) != Counter({mode: len(manifest) for mode in args.expected_modes}):
        raise ValueError("mode counts do not match the manifest")

    failures = []
    for row in results:
        if row.get("error") or row.get("source_audio_error") or row.get("reference_audio_error"):
            failures.append({"id": row.get("id"), "mode": row.get("mode"), "reason": "recorded_error"})
            continue
        if int(row.get("semantic_token_count", 0)) <= 0:
            failures.append({"id": row.get("id"), "mode": row.get("mode"), "reason": "no_semantic_tokens"})
            continue
        for field in ("audio_path", "source_audio_path", "reference_audio_path"):
            path = Path(str(row.get(field)))
            if not path.is_file():
                failures.append({"id": row.get("id"), "mode": row.get("mode"), "reason": f"missing:{field}"})
                break
            info = sf.info(path)
            if info.samplerate != 16000 or info.frames <= 0:
                failures.append({"id": row.get("id"), "mode": row.get("mode"), "reason": f"invalid:{field}"})
                break
    if failures:
        raise ValueError(json.dumps(failures[:20], ensure_ascii=False, indent=2))

    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    if int(summary.get("failed", -1)) != 0 or int(summary.get("no_semantic_tokens", -1)) != 0:
        raise ValueError(f"summary reports failures: {summary}")
    print(json.dumps({"verified": len(results), "expected": expected_count}, indent=2))


if __name__ == "__main__":
    main()

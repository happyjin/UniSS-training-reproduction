"""Verify the structural integrity of an audio evaluation directory.

Full-corpus evaluation may legitimately expose model-level generation failures
(currently ``no_semantic_tokens``).  Those samples must remain in the result
set and be reported as model failures, but they must not abort all downstream
metrics.  Infrastructure failures (source/reference decode failures, corrupt
audio, dummy vocabulary tokens, or any unexpected generated-audio error) stay
fatal.
"""

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
    parser.add_argument(
        "--allow-generated-failures",
        action="store_true",
        help="Allow accounted no_semantic_tokens model failures while keeping infrastructure checks strict.",
    )
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
    allowed_generated_failures = 0
    for row in results:
        generated_error = row.get("error")
        source_error = row.get("source_audio_error")
        reference_error = row.get("reference_audio_error")
        semantic_token_count = int(row.get("semantic_token_count", 0))
        if source_error or reference_error:
            failures.append(
                {
                    "id": row.get("id"),
                    "mode": row.get("mode"),
                    "reason": "source_or_reference_audio_error",
                }
            )
            continue
        if generated_error:
            if args.allow_generated_failures and generated_error == "no_semantic_tokens" and semantic_token_count == 0:
                allowed_generated_failures += 1
            else:
                failures.append(
                    {
                        "id": row.get("id"),
                        "mode": row.get("mode"),
                        "reason": f"generated_audio_error:{generated_error}",
                    }
                )
                continue
        elif semantic_token_count <= 0:
            failures.append({"id": row.get("id"), "mode": row.get("mode"), "reason": "no_semantic_tokens"})
            continue
        if int(row.get("dummy_token_count", 0)) != 0:
            failures.append({"id": row.get("id"), "mode": row.get("mode"), "reason": "dummy_vocab_token"})
            continue
        required_audio_fields = ["source_audio_path", "reference_audio_path"]
        if not generated_error:
            required_audio_fields.insert(0, "audio_path")
        for field in required_audio_fields:
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
    if (
        int(summary.get("failed", -1)) != allowed_generated_failures
        or int(summary.get("no_semantic_tokens", -1)) != allowed_generated_failures
    ):
        raise ValueError(f"summary reports failures: {summary}")
    print(
        json.dumps(
            {
                "verified": len(results),
                "expected": expected_count,
                "generated_audio_successes": len(results) - allowed_generated_failures,
                "allowed_model_failures": allowed_generated_failures,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

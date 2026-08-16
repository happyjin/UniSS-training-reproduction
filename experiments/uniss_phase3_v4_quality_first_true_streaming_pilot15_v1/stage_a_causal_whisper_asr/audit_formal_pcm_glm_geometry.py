#!/usr/bin/env python3
"""Audit every acoustically selected formal Stage A PCM/GLM geometry."""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import soundfile as sf

from experiments.uniss_phase3_v4_quality_first_true_streaming_pilot15_v1.stage_a_causal_whisper_asr.training.dataset import (
    SAMPLE_RATE,
    rotated_acoustic_indices,
)
from experiments.uniss_phase3_v4_quality_first_true_streaming_pilot15_v1.stage_a_causal_whisper_asr.training.objective import (
    terminal_codec_extension_deficit_samples,
)
from experiments.uniss_phase3_v4_quality_first_true_streaming_pilot15_v1.stage00_baseline.shared_causal_frontend import (
    TOKEN_HOP_SAMPLES,
)


SCHEMA = "uniss_stage_a_formal_pcm_glm_geometry_audit_v1"


def _atomic_create(path: Path, value: object) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite Stage A geometry audit: {path}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _inspect(
    entry: tuple[int, int, int, str, str, int, int]
) -> dict[str, object]:
    (
        pack_index,
        epoch,
        acoustic_index,
        sample_id,
        audio,
        sidecar_duration_ms,
        packed_tokens,
    ) = entry
    info = sf.info(audio)
    samples = int(info.frames)
    sample_rate = int(info.samplerate)
    causal_tokens = math.ceil(samples / TOKEN_HOP_SAMPLES)
    deficit = terminal_codec_extension_deficit_samples(
        samples, causal_tokens, packed_tokens
    )
    if packed_tokens == causal_tokens:
        category = "exact"
    elif deficit == 0:
        category = "terminal_extension_exact_hop"
    elif deficit is not None:
        category = "terminal_extension_one_frame_short"
    else:
        category = "invalid"
    duration_ms = round(1000 * samples / sample_rate) if sample_rate else -1
    return {
        "pack_index": pack_index,
        "coverage_epoch": epoch,
        "acoustic_index": acoustic_index,
        "sample_id": sample_id,
        "source_audio": audio,
        "sample_rate": sample_rate,
        "waveform_samples": samples,
        "waveform_mod_80ms": samples % TOKEN_HOP_SAMPLES,
        "duration_ms": duration_ms,
        "sidecar_duration_ms": sidecar_duration_ms,
        "causal_tokens": causal_tokens,
        "packed_glm_tokens": packed_tokens,
        "delta": packed_tokens - causal_tokens,
        "terminal_deficit_samples": deficit,
        "category": category,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--packs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--coverage-epochs", type=int, required=True)
    parser.add_argument("--max-acoustics-per-pack", type=int, default=2)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--progress-interval", type=int, default=1000)
    parser.add_argument("--expected-packs", type=int)
    parser.add_argument("--expected-selected", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.coverage_epochs <= 0 or args.max_acoustics_per_pack <= 0:
        raise ValueError("audit coverage geometry must be positive")
    if args.workers <= 0:
        raise ValueError("audit worker count must be positive")
    started = time.time()
    counts: Counter[str] = Counter()
    delta_counts: Counter[int] = Counter()
    violations: list[dict[str, object]] = []
    pack_count = 0
    selected_count = 0

    def entries():
        nonlocal pack_count, selected_count
        with args.packs.resolve().open("rb") as handle:
            for pack_index, line in enumerate(handle):
                pack = json.loads(line)
                acoustics = list(pack.get("acoustics", []))
                for epoch in range(args.coverage_epochs):
                    selected = rotated_acoustic_indices(
                        len(acoustics),
                        args.max_acoustics_per_pack,
                        epoch,
                        pack_index,
                    )
                    for acoustic_index in selected:
                        selected_count += 1
                        acoustic = acoustics[acoustic_index]
                        yield (
                            pack_index,
                            epoch,
                            acoustic_index,
                            str(acoustic["sample_id"]),
                            str(acoustic["source_audio"]),
                            int(acoustic["source_duration_ms"]),
                            len(acoustic["source_glm"]),
                        )
                pack_count = pack_index + 1
                if args.progress_interval and pack_count % args.progress_interval == 0:
                    print(
                        f"packs={pack_count} selected={selected_count} "
                        f"elapsed_s={time.time() - started:.1f}",
                        flush=True,
                    )

    def consume(result: dict[str, object]) -> None:
        category = str(result["category"])
        counts[category] += 1
        delta_counts[int(result["delta"])] += 1
        duration_ok = (
            int(result["sample_rate"]) == SAMPLE_RATE
            and abs(
                int(result["duration_ms"])
                - int(result["sidecar_duration_ms"])
            )
            <= 20
        )
        if category == "invalid" or not duration_ok:
            if len(violations) < 100:
                violations.append({**result, "duration_ok": duration_ok})

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        pending: list[tuple[int, int, int, str, str, int, int]] = []
        for entry in entries():
            pending.append(entry)
            if len(pending) < args.workers * 32:
                continue
            for result in executor.map(_inspect, pending):
                consume(result)
            pending.clear()
        for result in executor.map(_inspect, pending):
            consume(result)

    expected_ok = (
        (args.expected_packs is None or pack_count == args.expected_packs)
        and (args.expected_selected is None or selected_count == args.expected_selected)
    )
    passed = not violations and expected_ok and sum(counts.values()) == selected_count
    report = {
        "schema_version": SCHEMA,
        "passed": passed,
        "packs": str(args.packs.resolve()),
        "coverage_epochs": args.coverage_epochs,
        "max_acoustics_per_pack": args.max_acoustics_per_pack,
        "pack_count": pack_count,
        "selected_acoustics": selected_count,
        "category_counts": dict(sorted(counts.items())),
        "delta_counts": {str(key): value for key, value in sorted(delta_counts.items())},
        "expected_packs": args.expected_packs,
        "expected_selected": args.expected_selected,
        "expected_geometry_ok": expected_ok,
        "violation_count": len(violations),
        "violation_examples": violations,
        "elapsed_seconds": time.time() - started,
    }
    _atomic_create(args.output, report)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True), flush=True)
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

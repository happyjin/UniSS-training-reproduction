#!/usr/bin/env python3
"""Audit append-only BiCodec semantic coverage on a real pilot15 record."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import torch

from training.generate_unist_eval_audio import iter_manifest_records
from uniss.speech_tokenizer.bicodec.models.bicodec import BiCodec
from uniss.streaming.bicodec_streamer import StreamingBiCodecDecoder


def _atomic_json(path: Path, value: object) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite BiCodec audit: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


class Decoder:
    def __init__(self, checkpoint: Path, device: torch.device) -> None:
        self.device = device
        self.model = BiCodec.load_from_checkpoint(checkpoint).to(device).eval()

    @torch.inference_mode()
    def __call__(self, speaker, semantic) -> np.ndarray:
        global_tensor = torch.tensor([speaker], dtype=torch.long, device=self.device).unsqueeze(1)
        semantic_tensor = torch.tensor([semantic], dtype=torch.long, device=self.device)
        return (
            self.model.detokenize(semantic_tensor, global_tensor)
            .detach()
            .float()
            .reshape(-1)
            .cpu()
            .numpy()
        )


def _scheme(decoder: Decoder, speaker: list[int], semantic: list[int], sizes: list[int]):
    streamer = StreamingBiCodecDecoder(
        decode=decoder,
        left_context_tokens=50,
        holdback_tokens=5,
        overlap_ms=80.0,
    )
    outputs = []
    spans = []
    start = 0
    index = 0
    while start < len(semantic):
        end = min(len(semantic), start + sizes[index % len(sizes)])
        waveform = streamer.push(
            semantic[start:end], speaker_tokens=speaker, is_final=end == len(semantic)
        )
        outputs.append(waveform)
        spans.append([start, end])
        start = end
        index += 1
    joined = np.concatenate(outputs) if outputs else np.zeros(0, dtype=np.float32)
    return joined, spans, streamer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--validation-manifest", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    device = torch.device(args.device)
    record = next(iter_manifest_records(Path(args.validation_manifest).resolve(), limit_records=1))
    speaker = [int(value) for value in record["bicodec_global"]]
    semantic = [int(value) for value in record["target_bicodec"]]
    if len(speaker) != 32 or not semantic:
        raise ValueError("fixed validation record lacks valid BiCodec supervision")
    decoder = Decoder(Path(args.checkpoint).resolve(), device)
    one_shot = decoder(speaker, semantic)
    schemes = {}
    all_passed = True
    for name, sizes in {
        "micro_irregular": [5, 7, 11, 13],
        "phrase_irregular": [16, 24, 31],
        "large_irregular": [50, 73],
    }.items():
        waveform, spans, streamer = _scheme(decoder, speaker, semantic, sizes)
        contiguous = all(spans[index][1] == spans[index + 1][0] for index in range(len(spans) - 1))
        coverage = spans[0][0] == 0 and spans[-1][1] == len(semantic) and contiguous
        finite = bool(np.isfinite(waveform).all())
        length_equal = len(waveform) == len(one_shot) == len(semantic) * 320
        passed = coverage and finite and length_equal and streamer.emitted_samples == len(one_shot)
        all_passed = all_passed and passed
        schemes[name] = {
            "passed": passed,
            "chunk_sizes": sizes,
            "pushes": len(spans),
            "semantic_spans": spans,
            "semantic_gap_count": sum(
                spans[index + 1][0] != spans[index][1] for index in range(len(spans) - 1)
            ),
            "semantic_overlap_count": sum(
                spans[index + 1][0] < spans[index][1] for index in range(len(spans) - 1)
            ),
            "waveform_samples": len(waveform),
            "one_shot_samples": len(one_shot),
            "expected_samples": len(semantic) * 320,
            "finite": finite,
            "emitted_samples_state": streamer.emitted_samples,
        }
    speaker_freeze = StreamingBiCodecDecoder(decode=decoder)
    speaker_freeze.push(semantic[:10], speaker_tokens=speaker, is_final=False)
    rejected_speaker_change = False
    try:
        speaker_freeze.push(semantic[10:20], speaker_tokens=[0] * 32, is_final=False)
    except ValueError:
        rejected_speaker_change = True
    checks = {
        "speaker_has_32_tokens": len(speaker) == 32,
        "all_chunk_schemes_zero_gap_overlap": all_passed,
        "speaker_change_rejected": rejected_speaker_change,
    }
    output = {
        "schema_version": "uniss_stage00_bicodec_streaming_coverage_v1",
        "passed": all(checks.values()),
        "checks": checks,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "validation_manifest": str(Path(args.validation_manifest).resolve()),
        "sample_id": record.get("id"),
        "semantic_tokens": len(semantic),
        "one_shot_samples": len(one_shot),
        "schemes": schemes,
    }
    _atomic_json(Path(args.output_json).resolve(), output)
    print(json.dumps(output, ensure_ascii=False, sort_keys=True))
    if not output["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()


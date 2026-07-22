"""Replay a prepared schedule through the append-only streaming controller."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import soundfile as sf

from uniss.streaming.bicodec_streamer import StreamingBiCodecDecoder, bicodec_decode_function
from uniss.streaming.controller import FrontendStep, StreamingController, WriteResult
from uniss.streaming.policy import PolicyDecision


class ScheduleModelAdapter:
    def __init__(self) -> None:
        self.current_event: dict[str, object] | None = None
        self.source_tokens: list[int] = []
        self.waits = 0

    def append_source(self, glm_tokens: Sequence[int]) -> None:
        self.source_tokens.extend(int(token) for token in glm_tokens)

    def choose_action(self, eligible: bool, is_final: bool) -> PolicyDecision:
        if self.current_event is None:
            raise RuntimeError("current_event is not set")
        if is_final:
            return PolicyDecision.WRITE
        scheduled = str(self.current_event["action"])
        if scheduled == "write" and eligible:
            return PolicyDecision.WRITE
        return PolicyDecision.WAIT

    def commit_wait(self) -> None:
        self.waits += 1

    def generate_write(self, is_final: bool) -> WriteResult:
        del is_final
        if self.current_event is None:
            raise RuntimeError("current_event is not set")
        return WriteResult(
            self.current_event.get("target_text_ids", []),
            self.current_event.get("target_semantic", []),
        )


def synthetic_decode(_speaker_tokens: Sequence[int], semantic_tokens: Sequence[int]) -> np.ndarray:
    values = np.asarray(semantic_tokens, dtype=np.float32)
    if values.size == 0:
        return np.zeros(0, dtype=np.float32)
    values = (values / 8191.0) * 0.2 - 0.1
    return np.repeat(values, 320)


def load_first_schedule(path: Path, record_index: int) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if index == record_index:
                return json.loads(line)
    raise IndexError(f"record index {record_index} is outside {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-wav", required=True)
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--tensorboard-dir", default=None)
    parser.add_argument("--record-index", type=int, default=0)
    parser.add_argument("--decoder", choices=["synthetic", "bicodec"], default="synthetic")
    parser.add_argument("--bicodec-model-dir", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--left-context-tokens", type=int, default=50)
    parser.add_argument("--holdback-tokens", type=int, default=5)
    parser.add_argument("--overlap-ms", type=float, default=80.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    schedule = load_first_schedule(Path(args.input), args.record_index)
    if args.decoder == "bicodec":
        if not args.bicodec_model_dir:
            raise ValueError("--bicodec-model-dir is required for --decoder bicodec")
        import torch

        from uniss.speech_tokenizer.bicodec.bicodec_tokenizer import BiCodecTokenizer

        tokenizer = BiCodecTokenizer(Path(args.bicodec_model_dir), device=torch.device(args.device))
        decode = bicodec_decode_function(tokenizer)
    else:
        decode = synthetic_decode

    codec = StreamingBiCodecDecoder(
        decode,
        left_context_tokens=args.left_context_tokens,
        holdback_tokens=args.holdback_tokens,
        overlap_ms=args.overlap_ms,
    )
    model = ScheduleModelAdapter()
    controller = StreamingController(model=model, codec=codec)
    cumulative_glm: list[int] = []
    audio_chunks: list[np.ndarray] = []
    boundary_jumps: list[float] = []
    first_audio_ms: float | None = None
    previous_last: float | None = None
    events = schedule["events"]
    for event in events:
        model.current_event = event
        cumulative_glm.extend(event["source_glm"])
        action, waveform = controller.process_step(
            FrontendStep(
                cumulative_glm,
                source_count=int(event["source_ctc_count_proxy"]),
                target_supported_count=int(event["target_ctc_count_proxy"]),
            ),
            speaker_tokens=schedule["speaker_tokens"],
            is_final=bool(event["source_is_final"]),
        )
        if waveform.size:
            if first_audio_ms is None:
                first_audio_ms = float(event["source_end_ms"])
            if previous_last is not None:
                boundary_jumps.append(abs(float(waveform[0]) - previous_last))
            previous_last = float(waveform[-1])
            audio_chunks.append(waveform)

    waveform = np.concatenate(audio_chunks) if audio_chunks else np.zeros(0, dtype=np.float32)
    output_wav = Path(args.output_wav)
    output_wav.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_wav, waveform, codec.sample_rate)
    metrics = {
        "record_id": schedule["id"],
        "decoder": args.decoder,
        "wait_count": controller.wait_count,
        "write_count": controller.write_count,
        "first_audio_ms": first_audio_ms,
        "output_samples": len(waveform),
        "output_seconds": len(waveform) / codec.sample_rate,
        "boundary_jump_mean": float(np.mean(boundary_jumps)) if boundary_jumps else 0.0,
        "boundary_jump_max": float(np.max(boundary_jumps)) if boundary_jumps else 0.0,
        "committed_source_tokens": len(controller.prefix_committer.committed),
        "committed_target_tokens": controller.committed_target_tokens,
        "source_revision_events": controller.prefix_committer.revision_events,
    }
    metrics_path = Path(args.metrics)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.tensorboard_dir:
        from torch.utils.tensorboard import SummaryWriter

        with SummaryWriter(args.tensorboard_dir) as writer:
            for name, value in metrics.items():
                if isinstance(value, (int, float)) and value is not None:
                    writer.add_scalar(f"stage5/{name}", value, args.record_index)
            if waveform.size:
                writer.add_audio("stage5/output_audio", waveform, args.record_index, codec.sample_rate)
            writer.flush()
    print(json.dumps(metrics, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()

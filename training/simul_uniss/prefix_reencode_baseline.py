"""Measure real GLM tokenizer prefix stability by cumulative audio re-encoding."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Callable

import torch
import torchaudio

from uniss.streaming.stable_prefix import StablePrefixCommitter, longest_common_prefix


TokenizePrefix = Callable[[torch.Tensor, int], list[int]]


def evaluate_prefixes(
    waveform: torch.Tensor,
    sample_rate: int,
    tokenize_prefix: TokenizePrefix,
    *,
    chunk_ms: int = 640,
    holdback_tokens: int = 2,
) -> dict[str, object]:
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    chunk_samples = max(1, int(round(sample_rate * chunk_ms / 1000.0)))
    committer = StablePrefixCommitter(holdback_tokens=holdback_tokens)
    candidates: list[list[int]] = []
    committed_per_step: list[list[int]] = []
    encode_seconds: list[float] = []
    revised_tokens = 0
    compared_tokens = 0
    first_commit_ms: int | None = None
    total_samples = waveform.shape[-1]
    for chunk_index, end in enumerate(range(chunk_samples, total_samples + chunk_samples, chunk_samples)):
        end = min(end, total_samples)
        started = time.perf_counter()
        candidate = tokenize_prefix(waveform[..., :end], sample_rate)
        encode_seconds.append(time.perf_counter() - started)
        if candidates:
            lcp = longest_common_prefix(candidates[-1], candidate)
            revised_tokens += max(len(candidates[-1]), len(candidate)) - lcp
            compared_tokens += max(len(candidates[-1]), len(candidate))
        is_final = end >= total_samples
        new_tokens = committer.update(candidate, is_final=is_final)
        if new_tokens and first_commit_ms is None:
            first_commit_ms = int(round(end * 1000 / sample_rate))
        candidates.append(candidate)
        committed_per_step.append(new_tokens)
        if is_final:
            break
    duration_seconds = total_samples / sample_rate
    return {
        "chunk_ms": chunk_ms,
        "audio_duration_seconds": duration_seconds,
        "chunks": len(candidates),
        "final_candidate_length": len(candidates[-1]) if candidates else 0,
        "committed_length": len(committer.committed),
        "committed_rollback_events": committer.revision_events,
        "prefix_revision_rate": revised_tokens / max(1, compared_tokens),
        "first_stable_token_ms": first_commit_ms,
        "total_encode_seconds": sum(encode_seconds),
        "computation_aware_rtf": sum(encode_seconds) / max(duration_seconds, 1e-6),
        "encode_seconds_per_chunk": encode_seconds,
        "candidate_lengths": [len(candidate) for candidate in candidates],
        "new_committed_lengths": [len(tokens) for tokens in committed_per_step],
        "final_tokens": committer.committed,
    }


def first_source_audio(manifest_path: Path, record_index: int) -> tuple[str, Path]:
    with manifest_path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if index == record_index:
                item = json.loads(line)
                return str(item["id"]), Path(item["source_audio"])
    raise IndexError(record_index)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--glm-tokenizer", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--tensorboard-dir", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--record-index", type=int, default=0)
    parser.add_argument("--chunk-ms", type=int, default=640)
    parser.add_argument("--holdback-tokens", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from uniss.speech_tokenizer.glm4.glm4_tokenizer import Glm4Tokenizer

    record_id, audio_path = first_source_audio(Path(args.manifest), args.record_index)
    waveform, sample_rate = torchaudio.load(audio_path)
    tokenizer = Glm4Tokenizer(args.glm_tokenizer, device=args.device)

    def tokenize_prefix(prefix: torch.Tensor, sr: int) -> list[int]:
        tokens = tokenizer.tokenize(speech=prefix, sr=sr)
        return [int(token) for token in tokens.squeeze(0).detach().cpu().tolist()]

    metrics = evaluate_prefixes(
        waveform,
        sample_rate,
        tokenize_prefix,
        chunk_ms=args.chunk_ms,
        holdback_tokens=args.holdback_tokens,
    )
    metrics["record_id"] = record_id
    metrics["audio_path"] = str(audio_path)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.tensorboard_dir:
        from torch.utils.tensorboard import SummaryWriter

        with SummaryWriter(args.tensorboard_dir) as writer:
            for name in (
                "prefix_revision_rate",
                "committed_rollback_events",
                "first_stable_token_ms",
                "total_encode_seconds",
                "computation_aware_rtf",
            ):
                value = metrics[name]
                if value is not None:
                    writer.add_scalar(f"stage0_prefix/{name}", value, args.record_index)
            for step, value in enumerate(metrics["candidate_lengths"]):
                writer.add_scalar("stage0_prefix/candidate_length", value, step)
            for step, value in enumerate(metrics["new_committed_lengths"]):
                writer.add_scalar("stage0_prefix/new_committed_length", value, step)
            writer.flush()
    summary = {key: value for key, value in metrics.items() if not isinstance(value, list)}
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()

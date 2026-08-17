"""Strict cached runtime helpers for a trained Stage A v2 checkpoint."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.distributed.checkpoint as dcp

from experiments.uniss_phase3_v4_quality_first_true_streaming_pilot15_v1.stage00_baseline.shared_causal_frontend import (
    BLOCK_SAMPLES,
    SharedCausalWhisperVQFrontend,
)
from experiments.uniss_phase3_v4_quality_first_true_streaming_pilot15_v1.stage_a_causal_whisper_asr.training.frontend import (
    TrainableSharedCausalWhisperVQ,
)
from experiments.uniss_phase3_v4_quality_first_true_streaming_pilot15_v2.stage_a_causal_whisper_asr.training.objective import (
    StageAObjective,
)


@dataclass(frozen=True)
class CachedFrontendResult:
    hidden: torch.Tensor
    quantized: torch.Tensor
    tokens: torch.Tensor
    frames_seen: tuple[int, ...]
    reset_blocks: tuple[int, ...]


def resolve_iteration_checkpoint(path: str | Path) -> Path:
    """Resolve a checkpoint root or an explicit ``iter_XXXXXXX`` directory."""

    value = Path(path).resolve()
    if (value / ".metadata").is_file():
        return value
    latest = value / "latest_checkpointed_iteration.txt"
    if not latest.is_file():
        raise FileNotFoundError(f"checkpoint has no iteration metadata: {value}")
    iteration = int(latest.read_text(encoding="utf-8").strip())
    candidate = value / f"iter_{iteration:07d}"
    if not (candidate / ".metadata").is_file():
        raise FileNotFoundError(f"latest checkpoint iteration is incomplete: {candidate}")
    return candidate


def load_trained_objective(
    checkpoint: str | Path,
    whispervq_model: str | Path,
    device: str | torch.device,
    *,
    dtype: torch.dtype = torch.float32,
) -> tuple[StageAObjective, Path]:
    """Load only the trained Stage A sidecar from a native Megatron checkpoint."""

    resolved = resolve_iteration_checkpoint(checkpoint)
    objective = StageAObjective(
        TrainableSharedCausalWhisperVQ(
            whispervq_model,
            gradient_checkpointing=False,
        ),
        qwen_hidden_size=896,
    ).to(device=torch.device(device), dtype=dtype)
    state = {
        f"stage_a_objective.{name}": value
        for name, value in objective.state_dict().items()
    }
    dcp.load(state, checkpoint_id=str(resolved))
    objective.requires_grad_(False).eval()
    return objective, resolved


def make_cached_frontend(
    objective: StageAObjective,
    device: str | torch.device,
) -> SharedCausalWhisperVQFrontend:
    frontend = SharedCausalWhisperVQFrontend(
        objective.frontend.encoder,
        objective.frontend.mel_filters,
        device=device,
    )
    frontend.requires_grad_(False).eval()
    return frontend


@torch.inference_mode()
def run_cached_frontend(
    frontend: SharedCausalWhisperVQFrontend,
    waveform: np.ndarray,
) -> CachedFrontendResult:
    state = None
    hidden: list[torch.Tensor] = []
    quantized: list[torch.Tensor] = []
    tokens: list[torch.Tensor] = []
    frames_seen: list[int] = []
    reset_blocks: list[int] = []
    for block_index, start in enumerate(range(0, len(waveform), BLOCK_SAMPLES)):
        end = min(len(waveform), start + BLOCK_SAMPLES)
        output = frontend.push(
            waveform[start:end],
            state,
            is_final=end == len(waveform),
        )
        state = output.state
        hidden.append(output.pre_vq_hidden.float().cpu())
        quantized.append(output.quantized_hidden.float().cpu())
        tokens.append(output.token_ids.cpu())
        if output.encoder_reset_before_block:
            reset_blocks.append(block_index)
        frames_seen.append(
            int(state.encoder.frames_seen) if state.encoder is not None else 0
        )
    if state is None or not state.finalized:
        raise RuntimeError("cached frontend did not finalize")
    return CachedFrontendResult(
        hidden=torch.cat(hidden, dim=1),
        quantized=torch.cat(quantized, dim=1),
        tokens=torch.cat(tokens, dim=1),
        frames_seen=tuple(frames_seen),
        reset_blocks=tuple(reset_blocks),
    )


def hidden_metrics(reference: torch.Tensor, actual: torch.Tensor) -> dict[str, Any]:
    if reference.shape != actual.shape:
        return {
            "shape_equal": False,
            "reference_shape": list(reference.shape),
            "actual_shape": list(actual.shape),
            "allclose": False,
        }
    reference = reference.float()
    actual = actual.float()
    absolute = (reference - actual).abs()
    return {
        "shape_equal": True,
        "shape": list(reference.shape),
        "maximum_absolute_error": float(absolute.max().item()),
        "mean_absolute_error": float(absolute.mean().item()),
        "allclose": bool(torch.allclose(reference, actual, rtol=2e-5, atol=2e-6)),
        "rtol": 2e-5,
        "atol": 2e-6,
    }


def token_metrics(reference: torch.Tensor, actual: torch.Tensor) -> dict[str, Any]:
    shape_equal = reference.shape == actual.shape
    matches = int((reference == actual).sum().item()) if shape_equal else 0
    total = int(reference.numel()) if shape_equal else max(reference.numel(), actual.numel())
    return {
        "shape_equal": bool(shape_equal),
        "reference_tokens": int(reference.numel()),
        "actual_tokens": int(actual.numel()),
        "matching_tokens": matches,
        "match_ratio": float(matches / total) if total else 1.0,
        "exact": bool(shape_equal and matches == total),
    }


def cache_growth_is_valid(frames_seen: Sequence[int], reset_blocks: Sequence[int]) -> bool:
    resets = set(int(value) for value in reset_blocks)
    previous = 0
    for block, raw in enumerate(frames_seen):
        current = int(raw)
        expected = 8 if block in resets else previous + 8
        if current != expected:
            return False
        previous = current
    return True


def append_only_commit_audit(events: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Audit the irreversible token ledger used by Stage A streaming ASR."""

    committed: list[int] = []
    snapshots: list[dict[str, Any]] = []
    rollback_count = 0
    for event_index, event in enumerate(events):
        before = tuple(committed)
        committed.extend(int(value) for value in event.get("predicted_tokens", ()))
        preserved = tuple(committed[: len(before)]) == before
        rollback_count += int(not preserved)
        digest = hashlib.sha256(
            np.asarray(committed, dtype="<i8").tobytes()
        ).hexdigest()
        snapshots.append(
            {
                "event_index": event_index,
                "tokens_before": len(before),
                "tokens_after": len(committed),
                "prior_prefix_preserved": preserved,
                "committed_sha256": digest,
            }
        )
    return {
        "events": len(events),
        "committed_tokens": len(committed),
        "rollback_count": rollback_count,
        "rollback_rate": rollback_count / max(1, len(events)),
        "append_only": rollback_count == 0,
        "snapshots": snapshots,
    }


__all__ = [
    "CachedFrontendResult",
    "append_only_commit_audit",
    "cache_growth_is_valid",
    "hidden_metrics",
    "load_trained_objective",
    "make_cached_frontend",
    "resolve_iteration_checkpoint",
    "run_cached_frontend",
    "token_metrics",
]

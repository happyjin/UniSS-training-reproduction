"""Bayesian safe-commit evidence extracted from the latent Stage-B-v3 Student."""

from __future__ import annotations

import math
from pathlib import Path

import torch

from training.simul_uniss.subsecond_v2.stage_b_latent_model import (
    LatentCausalAudioStudent,
    LatentStageBModelConfig,
    load_whispervq_codebook,
)
from training.simul_uniss.subsecond_v1.stage_c import StageCGateConfig


EVIDENCE_SCHEMA = "simul_uniss_stage_c_after_v3_evidence_v1"


def load_latent_student(
    checkpoint: str | Path, device: torch.device
) -> tuple[LatentCausalAudioStudent, dict[str, object]]:
    path = Path(checkpoint).resolve()
    value = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    config = LatentStageBModelConfig.from_dict(value["model_config"])
    codebook = load_whispervq_codebook(
        value["codebook_model"], key=value.get("codebook_key", "codebook.weight")
    )
    student = LatentCausalAudioStudent(config, codebook).to(device).eval()
    missing, unexpected = student.load_state_dict(value["model"], strict=False)
    if missing or unexpected:
        raise ValueError(
            f"latent Student checkpoint mismatch: missing={missing}, unexpected={unexpected}"
        )
    student.requires_grad_(False)
    metadata = {
        "checkpoint": str(path),
        "schema_version": value.get("schema_version"),
        "checkpoint_step": value.get("step"),
        "model_config": config.__dict__,
    }
    return student, metadata


def _tail_geometry(
    lengths: torch.Tensor, time_steps: int, count: int
) -> tuple[torch.Tensor, torch.Tensor]:
    if count <= 0:
        raise ValueError("tail count must be positive")
    positions = torch.arange(count, device=lengths.device).reshape(1, -1)
    clipped = lengths.clamp(min=1, max=time_steps)
    indices = (clipped.reshape(-1, 1) - count + positions).clamp(0, time_steps - 1)
    mask = positions >= (count - clipped.reshape(-1, 1)).clamp_min(0)
    return indices, mask


def _gather_time(value: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    return value.gather(
        1, indices.unsqueeze(-1).expand(-1, -1, value.shape[-1])
    )


@torch.no_grad()
def _codebook_tail_statistics(
    latent: torch.Tensor,
    mask: torch.Tensor,
    codebook: torch.Tensor,
    *,
    temperature: float,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if temperature <= 0.0 or chunk_size <= 0:
        raise ValueError("invalid codebook statistics parameters")
    selected = latent.float()[mask]
    codebook_float = codebook.float()
    codebook_norm = codebook_float.square().sum(dim=1).reshape(1, -1)
    token_parts: list[torch.Tensor] = []
    confidence_parts: list[torch.Tensor] = []
    margin_parts: list[torch.Tensor] = []
    for start in range(0, len(selected), chunk_size):
        current = selected[start : start + chunk_size]
        distances = (
            current.square().sum(dim=1, keepdim=True)
            + codebook_norm
            - 2.0 * current @ codebook_float.T
        ) / current.shape[-1]
        nearest, ids = torch.topk(distances, k=2, dim=1, largest=False, sorted=True)
        gap = (nearest[:, 1] - nearest[:, 0]).clamp_min(0.0)
        token_parts.append(ids[:, 0])
        confidence_parts.append(torch.sigmoid(gap / temperature))
        margin_parts.append((2.0 * torch.sigmoid(gap / temperature) - 1.0).clamp(0.0, 1.0))
    tokens = torch.cat(token_parts) if token_parts else torch.zeros(0, dtype=torch.long, device=latent.device)
    confidence = (
        torch.cat(confidence_parts)
        if confidence_parts
        else torch.zeros(0, device=latent.device)
    )
    margin = (
        torch.cat(margin_parts) if margin_parts else torch.zeros(0, device=latent.device)
    )
    token_grid = torch.zeros(mask.shape, dtype=torch.long, device=latent.device)
    confidence_grid = torch.zeros(mask.shape, dtype=torch.float32, device=latent.device)
    margin_grid = torch.zeros(mask.shape, dtype=torch.float32, device=latent.device)
    token_grid[mask] = tokens
    confidence_grid[mask] = confidence
    margin_grid[mask] = margin
    return token_grid, confidence_grid, margin_grid


def _masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (value * mask.to(value.dtype)).sum(dim=1) / mask.sum(dim=1).clamp_min(1)


@torch.no_grad()
def extract_latent_gate_inputs(
    student: LatentCausalAudioStudent,
    student_output: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    gate_config: StageCGateConfig,
    *,
    tail_token_count: int,
    codebook_temperature: float,
    codebook_chunk_size: int,
) -> dict[str, torch.Tensor]:
    latent = student_output["glm_latent"]
    token_lengths = student_output["token_lengths"]
    token_indices, token_mask = _tail_geometry(
        token_lengths, latent.shape[1], tail_token_count
    )
    tail_latent = _gather_time(latent, token_indices)
    tokens, token_confidence, token_margin = _codebook_tail_statistics(
        tail_latent,
        token_mask,
        student.codebook,
        temperature=codebook_temperature,
        chunk_size=codebook_chunk_size,
    )

    stability = torch.sigmoid(student_output["stability_logits"].float())
    tail_stability = stability.gather(1, token_indices)
    capacity = torch.sigmoid(
        student_output["target_capacity_logits"].float().gather(
            1, (token_lengths - 1).clamp_min(0).reshape(-1, 1)
        ).squeeze(1)
    )

    source_logits = student_output["source_ctc_logits"].float()
    source_indices, source_mask = _tail_geometry(
        student_output["output_lengths"],
        source_logits.shape[1],
        student.config.segment_frames,
    )
    source_tail = _gather_time(source_logits, source_indices)
    source_confidence = torch.exp(
        source_tail.amax(dim=-1) - torch.logsumexp(source_tail, dim=-1)
    )

    pair_mask = token_mask[:, 1:] & token_mask[:, :-1]
    repeated = (tokens[:, 1:] == tokens[:, :-1]).float()
    persistence = _masked_mean(repeated, pair_mask)
    prefix_fraction = (
        batch["utterance_sample_lengths"].float()
        / batch["full_samples"].float().clamp_min(1)
    ).clamp(0.0, 1.0)
    prefix_seconds = (
        batch["utterance_sample_lengths"].float() / 16_000.0 / 8.0
    ).clamp(0.0, 1.5)
    duration_seconds = (batch["full_samples"].float() / 16_000.0 / 12.0).clamp(
        0.0, 2.0
    )
    context = torch.stack(
        (prefix_fraction, prefix_seconds, duration_seconds, batch["direction"].float()),
        dim=-1,
    )
    evidence = torch.stack(
        (
            _masked_mean(token_confidence, token_mask),
            _masked_mean(token_margin, token_mask),
            torch.ones_like(prefix_fraction),
            _masked_mean(token_confidence, token_mask),
            _masked_mean(tail_stability, token_mask),
            _masked_mean(source_confidence, source_mask),
            persistence,
            capacity,
        ),
        dim=-1,
    ).clamp(0.0, 1.0)
    if "safe_label" not in batch:
        raise ValueError("Stage-C-after-v3 requires formal safe_label supervision")
    labels = batch["safe_label"].float()
    return {
        "context": context,
        "evidence": evidence,
        "labels": labels,
        "predicted_top_class": tokens,
        "support_ready": (batch["support_count"] >= gate_config.minimum_commit_tokens).float(),
        "evidence_schema": EVIDENCE_SCHEMA,
    }

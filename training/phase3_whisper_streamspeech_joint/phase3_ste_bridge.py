"""Hard-forward, straight-through bridge into Phase3 GLM embeddings."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class STEBridgeOutput:
    embeddings: torch.Tensor
    hard_embeddings: torch.Tensor
    hard_code_ids: torch.Tensor
    commitment_loss: torch.Tensor
    teacher_ce_loss: torch.Tensor
    teacher_commitment_loss: torch.Tensor
    teacher_agreement: torch.Tensor
    teacher_coverage: torch.Tensor
    code_perplexity: torch.Tensor
    active_code_fraction: torch.Tensor
    hidden_rms: torch.Tensor


class Phase3STEBridge(nn.Module):
    """Quantize Whisper hidden while preserving gradients to the frontend."""

    def __init__(
        self,
        whisper_hidden_size: int,
        qwen_hidden_size: int,
        codebook: torch.Tensor,
        qwen_glm_embeddings: torch.Tensor,
        *,
        surrogate: str = "projection",
        topk: int = 8,
        temperature: float = 0.1,
        gradient_scale: float = 1.0,
        teacher_temperature: float = 0.1,
    ) -> None:
        super().__init__()
        if codebook.ndim != 2 or qwen_glm_embeddings.ndim != 2:
            raise ValueError("codebook and Qwen embeddings must be matrices")
        if codebook.shape[0] != qwen_glm_embeddings.shape[0]:
            raise ValueError("GLM codebook sizes differ")
        if codebook.shape[1] != whisper_hidden_size:
            raise ValueError("Whisper hidden size does not match codebook")
        if qwen_glm_embeddings.shape[1] != qwen_hidden_size:
            raise ValueError("Qwen hidden size does not match embeddings")
        if surrogate not in {"projection", "topk_soft"}:
            raise ValueError(f"unsupported STE surrogate: {surrogate}")
        if topk <= 0 or (surrogate == "topk_soft" and topk > codebook.shape[0]):
            raise ValueError("topk must be in [1, codebook size]")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if not 0 <= gradient_scale <= 1:
            raise ValueError("gradient_scale must be in [0, 1]")
        if teacher_temperature <= 0:
            raise ValueError("teacher_temperature must be positive")
        self.surrogate = surrogate
        self.topk = int(topk)
        self.temperature = float(temperature)
        self.gradient_scale = float(gradient_scale)
        self.teacher_temperature = float(teacher_temperature)
        self.continuous_projection = (
            nn.Linear(whisper_hidden_size, qwen_hidden_size)
            if surrogate == "projection"
            else None
        )
        self.register_buffer("codebook", codebook.detach().float().clone())
        self.register_buffer(
            "qwen_glm_embeddings", qwen_glm_embeddings.detach().float().clone()
        )

    def forward(
        self,
        hidden: torch.Tensor,
        lengths: torch.Tensor | None = None,
        teacher_code_ids: torch.Tensor | None = None,
        teacher_lengths: torch.Tensor | None = None,
    ) -> STEBridgeOutput:
        if hidden.ndim != 3 or hidden.shape[-1] != self.codebook.shape[-1]:
            raise ValueError("hidden has incompatible geometry")
        flat = hidden.float().reshape(-1, hidden.shape[-1])
        distances = (
            flat.square().sum(dim=1, keepdim=True)
            - 2 * flat @ self.codebook.t()
            + self.codebook.square().sum(dim=1).unsqueeze(0)
        )
        ids = distances.argmin(dim=-1).reshape(hidden.shape[:-1])
        hard = F.embedding(ids, self.qwen_glm_embeddings).to(hidden.dtype)
        if self.surrogate == "projection":
            if self.continuous_projection is None:
                raise RuntimeError("projection surrogate is not initialized")
            continuous = self.continuous_projection(hidden)
        else:
            nearest_distances, nearest_ids = distances.topk(
                self.topk, dim=-1, largest=False, sorted=False
            )
            logits = -nearest_distances / (
                hidden.shape[-1] * self.temperature
            )
            weights = logits.softmax(dim=-1)
            nearest_embeddings = F.embedding(
                nearest_ids, self.qwen_glm_embeddings
            )
            continuous = (
                weights.unsqueeze(-1) * nearest_embeddings
            ).sum(dim=-2).reshape(*hidden.shape[:-1], -1).to(hidden.dtype)
        embeddings = hard.detach() + self.gradient_scale * (
            continuous - continuous.detach()
        )
        chosen_code = F.embedding(ids.reshape(-1), self.codebook).reshape_as(hidden)
        squared_error = (hidden.float() - chosen_code.detach()).square()
        if lengths is None:
            commitment = squared_error.mean()
        else:
            if lengths.ndim != 1 or lengths.shape[0] != hidden.shape[0]:
                raise ValueError("lengths must have shape [batch]")
            positions = torch.arange(hidden.shape[1], device=hidden.device)
            valid = positions.unsqueeze(0) < lengths.unsqueeze(1)
            denominator = valid.sum().clamp_min(1) * hidden.shape[-1]
            commitment = (
                squared_error * valid.unsqueeze(-1)
            ).sum() / denominator
        if lengths is None:
            valid_hidden = torch.ones_like(ids, dtype=torch.bool)
        else:
            positions = torch.arange(hidden.shape[1], device=hidden.device)
            valid_hidden = positions.unsqueeze(0) < lengths.unsqueeze(1)
        valid_ids = ids[valid_hidden]
        counts = torch.bincount(valid_ids, minlength=self.codebook.shape[0]).float()
        probabilities = counts / counts.sum().clamp_min(1)
        nonzero = probabilities > 0
        entropy = -(probabilities[nonzero] * probabilities[nonzero].log()).sum()
        code_perplexity = entropy.exp()
        active_code_fraction = nonzero.float().mean()
        hidden_denominator = valid_hidden.sum().clamp_min(1) * hidden.shape[-1]
        hidden_rms = (
            (hidden.float().square() * valid_hidden.unsqueeze(-1)).sum()
            / hidden_denominator
        ).sqrt()

        zero = commitment * 0.0
        teacher_ce = zero
        teacher_commitment = zero
        teacher_agreement = zero.detach()
        teacher_coverage = zero.detach()
        if teacher_code_ids is not None:
            if teacher_lengths is None:
                raise ValueError("teacher_lengths are required with teacher_code_ids")
            if teacher_code_ids.ndim != 2 or teacher_code_ids.shape[0] != hidden.shape[0]:
                raise ValueError("teacher_code_ids must have shape [batch, time]")
            if teacher_lengths.ndim != 1 or teacher_lengths.shape[0] != hidden.shape[0]:
                raise ValueError("teacher_lengths must have shape [batch]")
            positions = torch.arange(hidden.shape[1], device=hidden.device)
            aligned_lengths = torch.minimum(lengths, teacher_lengths) if lengths is not None else teacher_lengths
            valid_teacher = positions.unsqueeze(0) < aligned_lengths.unsqueeze(1)
            teacher = teacher_code_ids[:, : hidden.shape[1]]
            if teacher.shape[1] < hidden.shape[1]:
                teacher = F.pad(teacher, (0, hidden.shape[1] - teacher.shape[1]), value=-1)
            valid_teacher = valid_teacher & (teacher >= 0) & (teacher < self.codebook.shape[0])
            flat_valid = valid_teacher.reshape(-1)
            if bool(flat_valid.any()):
                targets = teacher.reshape(-1)[flat_valid].long()
                teacher_logits = -distances[flat_valid] / (
                    hidden.shape[-1] * self.teacher_temperature
                )
                teacher_ce = F.cross_entropy(teacher_logits, targets)
                teacher_vectors = F.embedding(targets, self.codebook)
                teacher_hidden = flat[flat_valid]
                teacher_commitment = (teacher_hidden - teacher_vectors.detach()).square().mean()
                teacher_agreement = (
                    ids.reshape(-1)[flat_valid] == targets
                ).float().mean()
                teacher_coverage = flat_valid.sum().float() / valid_hidden.sum().clamp_min(1)
        return STEBridgeOutput(
            embeddings=embeddings,
            hard_embeddings=hard,
            hard_code_ids=ids,
            commitment_loss=commitment,
            teacher_ce_loss=teacher_ce,
            teacher_commitment_loss=teacher_commitment,
            teacher_agreement=teacher_agreement,
            teacher_coverage=teacher_coverage,
            code_perplexity=code_perplexity,
            active_code_fraction=active_code_fraction,
            hidden_rms=hidden_rms,
        )

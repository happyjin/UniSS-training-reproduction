"""Loss helpers for S4 full/cropped example mixtures."""

import torch
import torch.nn.functional as F


def _validated_example_weights(targets, example_weights, dtype):
    if targets.ndim != 2:
        raise ValueError(f"targets shape must be [B, L], got {tuple(targets.shape)}")
    batch_size = targets.size(0)
    if example_weights is None:
        return torch.ones(batch_size, device=targets.device, dtype=dtype)
    if not torch.is_tensor(example_weights):
        example_weights = torch.as_tensor(example_weights)
    if example_weights.ndim != 1 or example_weights.numel() != batch_size:
        raise ValueError(
            "example_weights shape must be [B], got "
            f"{tuple(example_weights.shape)} for B={batch_size}"
        )
    weights = example_weights.to(device=targets.device, dtype=dtype)
    if not bool(torch.isfinite(weights).all()):
        raise ValueError("example_weights must be finite")
    if not bool((weights >= 0).all()):
        raise ValueError("example_weights must be nonnegative")
    return weights


def speech_token_weights(targets, example_weights, dtype):
    """Expand per-example weights over valid (non--100) speech targets."""
    weights = _validated_example_weights(targets, example_weights, dtype)
    return weights[:, None] * (targets != -100).to(dtype=dtype)


def weighted_speech_ce(logits, targets, example_weights):
    """Mean speech CE with both numerator and valid-token count example-weighted."""
    if logits.ndim != 3 or targets.ndim != 2 or logits.shape[:2] != targets.shape:
        raise ValueError(
            "logits/targets shape must be [B, L, V] and [B, L], got "
            f"{tuple(logits.shape)} and {tuple(targets.shape)}"
        )
    token_weights = speech_token_weights(targets, example_weights, logits.dtype)
    denominator = token_weights.sum()
    if not bool(denominator.detach() > 0):
        return logits.sum() * 0
    per_token = F.cross_entropy(
        logits.transpose(1, 2), targets, reduction="none", ignore_index=-100
    )
    return (per_token * token_weights).sum() / denominator

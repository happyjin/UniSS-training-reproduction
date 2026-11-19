# Copyright (c) ModelScope Contributors. All rights reserved.
"""Token-accuracy helpers for Qwen-OmniSpeak external training plugin."""
from __future__ import annotations

from typing import Any, Dict, Optional

import torch

_ORIGINAL_COMPUTE_ACC_ATTR = '_qwen_omnispeak_original_compute_acc'
_PATCHED_ATTR = '_qwen_omnispeak_accuracy_patched'


def compute_direct_token_values(
    logits: torch.Tensor,
    labels: torch.Tensor,
    sample_mask: Optional[torch.Tensor] = None,
    ignore_index: int = -100,
) -> Optional[torch.Tensor]:
    """Return flat 0/1 correctness values for directly aligned token labels."""
    if logits is None or labels is None:
        return None
    with torch.no_grad():
        labels = labels.to(logits.device)
        mask = labels != ignore_index
        if sample_mask is not None:
            keep = sample_mask.to(device=logits.device, dtype=torch.bool)
            if keep.numel() == labels.size(0):
                view_shape = [labels.size(0)] + [1] * (labels.dim() - 1)
                mask = mask & keep.view(*view_shape).expand_as(labels)
        if not bool(mask.any()):
            return None
        preds = logits.argmax(dim=-1)
        return ((preds == labels) & mask)[mask].float().detach()


def compute_causal_text_token_values(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100,
) -> Optional[torch.Tensor]:
    """Return flat 0/1 correctness values with causal-LM next-token shift."""
    if logits is None or labels is None or logits.dim() < 3 or labels.dim() < 2:
        return None
    if logits.shape[:2] != labels.shape:
        return None
    return compute_direct_token_values(logits[:, :-1, :], labels[:, 1:], None, ignore_index)


def _get_output_field(outputs: Any, name: str, default=None):
    value = getattr(outputs, name, default)
    if value is not default:
        return value
    if isinstance(outputs, dict):
        return outputs.get(name, default)
    get = getattr(outputs, 'get', None)
    if callable(get):
        return get(name, default)
    return default


def _normalise_metric_values(value: Any) -> Optional[torch.Tensor]:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return None
        return value.detach().float().view(-1)
    if isinstance(value, (list, tuple)):
        if not value:
            return None
        return torch.as_tensor(value, dtype=torch.float32)
    return torch.as_tensor([value], dtype=torch.float32)


def _metrics_from_payload(payload: Any) -> Dict[str, torch.Tensor]:
    if not payload:
        return {}
    specs = payload.values() if isinstance(payload, dict) else payload
    metrics: Dict[str, torch.Tensor] = {}
    for spec in specs:
        if not isinstance(spec, dict):
            continue
        name = spec.get('name')
        if not name:
            continue
        alignment = spec.get('alignment')
        if alignment == 'causal_text':
            values = compute_causal_text_token_values(spec.get('logits'), spec.get('labels'))
        else:
            values = compute_direct_token_values(spec.get('logits'), spec.get('labels'), spec.get('sample_mask'))
        if values is not None:
            metrics[name] = values
    return metrics


def _extract_omnispeak_metrics(outputs: Any) -> Dict[str, torch.Tensor]:
    raw_metrics = _get_output_field(outputs, 'metrics', None)
    if raw_metrics:
        metrics = {}
        for key, value in raw_metrics.items():
            values = _normalise_metric_values(value)
            if values is not None:
                metrics[key] = values
        if metrics:
            return metrics
    payload = _get_output_field(outputs, 'omnispeak_accuracy_payload', None)
    return _metrics_from_payload(payload)


def install_qwen_omnispeak_accuracy_patch() -> None:
    """Install an idempotent Swift trainer patch for OmniSpeak multi-route acc."""
    from swift.trainers.mixin import SwiftMixin

    if getattr(SwiftMixin, _PATCHED_ATTR, False):
        return

    original_compute_acc = SwiftMixin._compute_acc
    setattr(SwiftMixin, _ORIGINAL_COMPUTE_ACC_ATTR, original_compute_acc)

    def _patched_compute_acc(self, outputs, labels, cu_seqlens=None):
        metrics = _extract_omnispeak_metrics(outputs)
        if metrics:
            mode = 'train' if self.model.training else 'eval'
            for key, values in metrics.items():
                self.custom_metrics[mode][key].update(values)
            return
        return original_compute_acc(self, outputs, labels, cu_seqlens=cu_seqlens)

    SwiftMixin._compute_acc = _patched_compute_acc
    setattr(SwiftMixin, _PATCHED_ATTR, True)


# Backwards-compatible alias for older plugin imports.
install_qwen3_omnispeak_accuracy_patch = install_qwen_omnispeak_accuracy_patch

__all__ = [
    'compute_direct_token_values',
    'compute_causal_text_token_values',
    'install_qwen_omnispeak_accuracy_patch',
    'install_qwen3_omnispeak_accuracy_patch',
]

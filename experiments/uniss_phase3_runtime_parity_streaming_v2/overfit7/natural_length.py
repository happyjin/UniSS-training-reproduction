"""Parallel semantic content with an explicit natural-length posterior.

V6 asked every semantic slot to choose between 8192 content classes and an
END class.  That couples two very different decisions and allowed a single
missed END slot to invalidate an otherwise useful block.  V7 predicts content
and length with separate trainable posteriors.  Runtime still uses the model's
argmax length; neither an oracle length nor a forced truncation is used.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from experiments.uniss_phase3_dense_aligned_streaming_pilot15_v1.data.packing import (
    ROLE_SEMANTIC,
)
from experiments.uniss_phase3_runtime_parity_streaming_v2.overfit5.semantic_block import (
    END_CLASS,
)
from experiments.uniss_phase3_runtime_parity_streaming_v2.overfit6.pretrain_overfit6 import (
    UntiedParallelSemanticBlockHead,
)
from experiments.uniss_phase3_true_subsecond_deadline_full198_v1.training.losses import (
    LossTerm,
    values_to_term,
    zero_term,
)
from training import constants_uniss as c


@dataclass(frozen=True)
class NaturalLengthSemanticOutput:
    content_term: LossTerm
    length_term: LossTerm
    token_accuracy: torch.Tensor
    length_accuracy: torch.Tensor
    length_mae: torch.Tensor
    blocks: torch.Tensor


class NaturalLengthParallelSemanticBlockHead(UntiedParallelSemanticBlockHead):
    """Predict semantic content and a 1..N block length independently."""

    def __init__(self, hidden_size: int, *, maximum_semantic_tokens: int = 24) -> None:
        super().__init__(
            hidden_size, maximum_semantic_tokens=maximum_semantic_tokens
        )
        self.length_head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size, bias=False),
            nn.SiLU(),
            nn.Linear(hidden_size, maximum_semantic_tokens),
        )
        for module in self.length_head:
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        for parameter in self.length_head.parameters():
            parameter.uniss_lr_new_heads = True

    def content_logits(self, context: torch.Tensor) -> torch.Tensor:
        # Retain the v6 tensor geometry for checkpoint compatibility, but do
        # not let the obsolete END row compete with semantic content.
        return super().forward(context, torch.empty(0))[
            :, : self.maximum_semantic_tokens, :END_CLASS
        ]

    def length_logits(self, context: torch.Tensor) -> torch.Tensor:
        return self.length_head(context)

    @torch.inference_mode()
    def decode(
        self,
        context: torch.Tensor,
        word_embedding_weight: torch.Tensor,
    ) -> tuple[tuple[int, ...], bool]:
        del word_embedding_weight
        if context.ndim == 1:
            context = context.unsqueeze(0)
        if context.ndim != 2 or context.shape[0] != 1:
            raise ValueError("runtime semantic decode requires exactly one block")
        natural_length = int(self.length_logits(context).float().argmax(dim=-1)[0]) + 1
        choices = self.content_logits(context)[0].float().argmax(dim=-1)
        return tuple(int(value) for value in choices[:natural_length]), True

    def training_output(
        self,
        hidden: torch.Tensor,
        labels: torch.Tensor,
        token_roles: torch.Tensor,
        loss_mask: torch.Tensor,
        word_embedding_weight: torch.Tensor,
    ) -> NaturalLengthSemanticOutput:
        del word_embedding_weight
        active = (token_roles == ROLE_SEMANTIC) & (loss_mask > 0)
        previous = torch.zeros_like(active)
        previous[1:] = active[:-1]
        following = torch.zeros_like(active)
        following[:-1] = active[1:]
        starts = (active & ~previous).nonzero(as_tuple=False).flatten()
        ends = (active & ~following).nonzero(as_tuple=False).flatten()
        anchor = sum(
            (parameter.reshape(-1)[0] * 0.0 for parameter in self.parameters()),
            hidden.sum() * 0.0,
        )
        if not starts.numel():
            zero = anchor.detach().new_zeros(())
            return NaturalLengthSemanticOutput(
                zero_term(anchor), zero_term(anchor), zero, zero, zero, zero
            )
        if starts.numel() != ends.numel():
            raise ValueError("semantic role spans have unmatched boundaries")
        lengths = ends - starts + 1
        if int(lengths.min()) < 1 or int(lengths.max()) > self.maximum_semantic_tokens:
            raise ValueError(
                "semantic block length escaped natural posterior support: "
                f"min={int(lengths.min())} max={int(lengths.max())}"
            )

        positions = torch.arange(
            self.maximum_semantic_tokens, device=hidden.device
        )
        gather = (starts[:, None] + positions[None, :]).clamp_max(labels.numel() - 1)
        targets = labels[gather] - c.BICODEC_SEMANTIC_OFFSET
        target_mask = positions[None, :] < lengths[:, None]
        if bool(((targets[target_mask] < 0) | (targets[target_mask] >= END_CLASS)).any()):
            raise ValueError("parallel semantic content target escaped the codebook")
        targets = torch.where(target_mask, targets, torch.zeros_like(targets))

        context = hidden[starts]
        content_logits = self.content_logits(context)
        content_losses = F.cross_entropy(
            content_logits.float().reshape(-1, END_CLASS),
            targets.reshape(-1),
            reduction="none",
        ).reshape_as(targets)
        content_term = values_to_term(content_losses, target_mask)

        length_targets = lengths - 1
        length_logits = self.length_logits(context)
        length_losses = F.cross_entropy(
            length_logits.float(), length_targets, reduction="none"
        )
        length_term = values_to_term(length_losses, torch.ones_like(length_losses))
        content_prediction = content_logits.float().argmax(dim=-1)
        length_prediction = length_logits.float().argmax(dim=-1) + 1
        return NaturalLengthSemanticOutput(
            content_term=content_term,
            length_term=length_term,
            token_accuracy=(content_prediction == targets)[target_mask].float().mean(),
            length_accuracy=(length_prediction == lengths).float().mean(),
            length_mae=(length_prediction - lengths).abs().float().mean(),
            blocks=starts.new_tensor(float(starts.numel()), dtype=torch.float32),
        )


__all__ = [
    "NaturalLengthParallelSemanticBlockHead",
    "NaturalLengthSemanticOutput",
]

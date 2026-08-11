"""Natural causal microblocks for low-dispatch BiCodec semantic generation.

The failed v11 head predicted as many as 24 semantic units independently from
one ``START_SEMANTIC`` hidden state.  This module shortens the prediction
horizon to four units and conditions every later microblock on semantic units
that have actually entered the main Qwen KV cache.  Inside a microblock a
small causal transition consumes the preceding ground-truth unit during
training and the preceding predicted unit at runtime.

Content logits remain tied to the frozen Phase3 semantic-token embedding
rows.  At initialization slot zero is therefore exactly the frozen Phase3
next-token classifier restricted to the legal BiCodec semantic vocabulary.
There is no oracle length: a learned CONTINUE/END posterior decides whether
another microblock is required, and a learned 1..4 posterior selects only the
length of the final microblock.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from experiments.uniss_phase3_dense_aligned_streaming_pilot15_v1.data.packing import (
    ROLE_SEMANTIC,
)
from experiments.uniss_phase3_true_subsecond_deadline_full198_v1.training.losses import (
    LossTerm,
    values_to_term,
    zero_term,
)
from training import constants_uniss as c


@dataclass(frozen=True)
class MicroblockTargets:
    """Teacher-forced microblocks extracted from shifted causal labels."""

    contexts: torch.Tensor
    targets: torch.Tensor
    content_mask: torch.Tensor
    final_mask: torch.Tensor
    lengths: torch.Tensor
    continue_targets: torch.Tensor


@dataclass(frozen=True)
class MicroblockSemanticOutput:
    content_term: LossTerm
    final_length_term: LossTerm
    continue_term: LossTerm
    token_accuracy: torch.Tensor
    first_slot_accuracy: torch.Tensor
    final_length_accuracy: torch.Tensor
    final_length_mae: torch.Tensor
    continue_accuracy: torch.Tensor
    predicted_continue_fraction: torch.Tensor
    target_continue_fraction: torch.Tensor
    predicted_unique_fraction: torch.Tensor
    blocks: torch.Tensor


def _balanced_example_weights(
    targets: torch.Tensor,
    mask: torch.Tensor,
    *,
    classes: int,
    minimum: float = 0.5,
    maximum: float = 4.0,
) -> torch.Tensor:
    """Return clipped inverse-sqrt local-frequency weights with mean one.

    The clipping is intentionally mild.  It prevents one frequent codec unit
    from dominating the loss without turning rare/noisy units into an
    unstable high-gradient objective.
    """

    active = mask.bool()
    weights = torch.zeros_like(targets, dtype=torch.float32)
    if not bool(active.any()):
        return weights
    selected = targets[active].long()
    counts = torch.bincount(selected, minlength=int(classes)).float()
    inverse = counts[selected].clamp_min(1.0).rsqrt()
    inverse = inverse / inverse.mean().clamp_min(1.0e-8)
    weights[active] = inverse.clamp(float(minimum), float(maximum))
    return weights


def extract_microblock_targets(
    hidden: torch.Tensor,
    labels: torch.Tensor,
    token_roles: torch.Tensor,
    loss_mask: torch.Tensor,
    *,
    block_size: int,
) -> MicroblockTargets | None:
    """Extract contexts at semantic offsets 0, ``block_size``, ...

    Because the training records use ``tokens[:-1]`` and ``labels=tokens[1:]``,
    the first active semantic-label position is the hidden state whose input
    token is ``START_SEMANTIC``.  At offset four the hidden state's input is
    the fourth already-known semantic unit, exactly matching runtime after
    the first four predicted units have entered the persistent KV cache.
    """

    if block_size <= 0:
        raise ValueError("microblock size must be positive")
    if hidden.ndim != 2:
        raise ValueError("flattened hidden states must be [tokens,hidden]")
    if any(value.ndim != 1 for value in (labels, token_roles, loss_mask)):
        raise ValueError("labels, roles and mask must be flattened")
    if not (hidden.shape[0] == labels.numel() == token_roles.numel() == loss_mask.numel()):
        raise ValueError("microblock tensors have inconsistent token counts")

    active = (token_roles == ROLE_SEMANTIC) & (loss_mask > 0)
    previous = torch.zeros_like(active)
    previous[1:] = active[:-1]
    following = torch.zeros_like(active)
    following[:-1] = active[1:]
    span_starts = (active & ~previous).nonzero(as_tuple=False).flatten()
    span_ends = (active & ~following).nonzero(as_tuple=False).flatten()
    if not span_starts.numel():
        return None
    if span_starts.numel() != span_ends.numel():
        raise ValueError("semantic role spans have unmatched boundaries")

    context_positions: list[torch.Tensor] = []
    block_lengths: list[torch.Tensor] = []
    continue_targets: list[torch.Tensor] = []
    for start, end in zip(span_starts, span_ends):
        span_length = int((end - start + 1).item())
        if span_length <= 0:
            raise ValueError("semantic span is empty")
        offsets = torch.arange(0, span_length, block_size, device=hidden.device)
        lengths = torch.minimum(
            offsets.new_full(offsets.shape, block_size),
            offsets.new_full(offsets.shape, span_length) - offsets,
        )
        context_positions.append(start + offsets)
        block_lengths.append(lengths)
        continue_targets.append((offsets + block_size < span_length).long())

    positions = torch.cat(context_positions)
    lengths = torch.cat(block_lengths)
    continuation = torch.cat(continue_targets)
    slots = torch.arange(block_size, device=hidden.device)
    gather = positions[:, None] + slots[None, :]
    if int(gather.max()) >= labels.numel():
        raise ValueError("microblock target position exceeds flattened sequence")
    targets = labels[gather] - c.BICODEC_SEMANTIC_OFFSET
    content_mask = slots[None, :] < lengths[:, None]
    legal = targets[content_mask]
    if bool(((legal < 0) | (legal >= c.BICODEC_SEMANTIC_SIZE)).any()):
        raise ValueError("microblock semantic target escaped the codebook")
    targets = torch.where(content_mask, targets, torch.zeros_like(targets))
    return MicroblockTargets(
        contexts=hidden[positions],
        targets=targets.long(),
        content_mask=content_mask,
        final_mask=continuation == 0,
        lengths=lengths.long(),
        continue_targets=continuation,
    )


class CausalMicroblockSemanticHead(nn.Module):
    """Generate four tied-vocabulary semantic units per Qwen dispatch."""

    def __init__(self, hidden_size: int, *, block_size: int = 4) -> None:
        super().__init__()
        if hidden_size <= 0 or block_size <= 0:
            raise ValueError("microblock geometry must be positive")
        self.hidden_size = int(hidden_size)
        self.block_size = int(block_size)
        self.slot_embeddings = nn.Parameter(torch.zeros(block_size, hidden_size))
        self.content_adapter = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size * 2, bias=False),
            nn.SiLU(),
            nn.Linear(hidden_size * 2, hidden_size, bias=False),
        )
        self.transition_norm = nn.LayerNorm(hidden_size * 2)
        self.transition = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size * 2, bias=False),
            nn.SiLU(),
            nn.Linear(hidden_size * 2, hidden_size, bias=False),
        )
        self.final_length_head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size, bias=False),
            nn.SiLU(),
            nn.Linear(hidden_size, block_size),
        )
        self.continue_head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size, bias=False),
            nn.SiLU(),
            nn.Linear(hidden_size, 2),
        )
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        # Preserve the Phase3 next-token semantic classifier exactly at step
        # zero.  The adapters learn residuals instead of replacing its useful
        # geometry with a random 8192-way classifier.
        nn.init.zeros_(self.content_adapter[-1].weight)
        nn.init.zeros_(self.transition[-1].weight)
        for parameter in self.parameters():
            parameter.uniss_lr_new_heads = True

    @staticmethod
    def semantic_embeddings(word_embedding_weight: torch.Tensor) -> torch.Tensor:
        return word_embedding_weight[
            c.BICODEC_SEMANTIC_OFFSET :
            c.BICODEC_SEMANTIC_OFFSET + c.BICODEC_SEMANTIC_SIZE
        ]

    def content_logits(
        self,
        context: torch.Tensor,
        word_embedding_weight: torch.Tensor,
        *,
        teacher_targets: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return [blocks,slots,8192] causal microblock logits."""

        if context.ndim == 1:
            context = context.unsqueeze(0)
        if context.ndim != 2 or context.shape[-1] != self.hidden_size:
            raise ValueError("microblock context must be [blocks,hidden]")
        if teacher_targets is not None and teacher_targets.shape != (
            context.shape[0], self.block_size
        ):
            raise ValueError("teacher targets do not match microblock geometry")
        classes = self.semantic_embeddings(word_embedding_weight).to(context.dtype)
        state = context
        outputs: list[torch.Tensor] = []
        for slot in range(self.block_size):
            slot_state = state + self.slot_embeddings[slot].to(state.dtype)
            slot_state = slot_state + self.content_adapter(slot_state)
            logits = F.linear(slot_state, classes)
            outputs.append(logits)
            if slot + 1 == self.block_size:
                continue
            if teacher_targets is None:
                selected = logits.float().argmax(dim=-1)
            else:
                selected = teacher_targets[:, slot]
            previous_embedding = F.embedding(selected, classes)
            transition_input = torch.cat((state, previous_embedding), dim=-1)
            state = state + self.transition(self.transition_norm(transition_input))
        return torch.stack(outputs, dim=1)

    def length_logits(self, context: torch.Tensor) -> torch.Tensor:
        return self.final_length_head(context)

    def continuation_logits(self, context: torch.Tensor) -> torch.Tensor:
        return self.continue_head(context)

    @torch.inference_mode()
    def decode(
        self,
        context: torch.Tensor,
        word_embedding_weight: torch.Tensor,
    ) -> tuple[tuple[int, ...], bool]:
        """Decode one natural microblock and return ``(units, continue)``."""

        if context.ndim == 1:
            context = context.unsqueeze(0)
        if context.ndim != 2 or context.shape[0] != 1:
            raise ValueError("runtime microblock decode requires exactly one context")
        content = self.content_logits(context, word_embedding_weight)[0]
        choices = content.float().argmax(dim=-1)
        should_continue = bool(
            self.continuation_logits(context).float().argmax(dim=-1)[0].item()
        )
        length = (
            self.block_size
            if should_continue
            else int(self.length_logits(context).float().argmax(dim=-1)[0]) + 1
        )
        return tuple(int(value) for value in choices[:length]), should_continue

    def training_output(
        self,
        hidden: torch.Tensor,
        labels: torch.Tensor,
        token_roles: torch.Tensor,
        loss_mask: torch.Tensor,
        word_embedding_weight: torch.Tensor,
    ) -> MicroblockSemanticOutput:
        extracted = extract_microblock_targets(
            hidden,
            labels,
            token_roles,
            loss_mask,
            block_size=self.block_size,
        )
        anchor = hidden.sum() * 0.0
        for parameter in self.parameters():
            anchor = anchor + parameter.reshape(-1)[0] * 0.0
        if extracted is None:
            zero = anchor.detach().new_zeros(())
            return MicroblockSemanticOutput(
                zero_term(anchor), zero_term(anchor), zero_term(anchor),
                zero, zero, zero, zero, zero, zero, zero, zero, zero,
            )

        content_logits = self.content_logits(
            extracted.contexts,
            word_embedding_weight,
            teacher_targets=extracted.targets,
        )
        content_losses = F.cross_entropy(
            content_logits.float().reshape(-1, c.BICODEC_SEMANTIC_SIZE),
            extracted.targets.reshape(-1),
            reduction="none",
            label_smoothing=0.01,
        ).reshape_as(extracted.targets)
        content_weights = _balanced_example_weights(
            extracted.targets,
            extracted.content_mask,
            classes=c.BICODEC_SEMANTIC_SIZE,
        )
        content_term = values_to_term(content_losses, content_weights)

        length_logits = self.length_logits(extracted.contexts)
        length_targets = extracted.lengths - 1
        length_losses = F.cross_entropy(
            length_logits.float(), length_targets, reduction="none"
        )
        length_weights = _balanced_example_weights(
            length_targets,
            extracted.final_mask,
            classes=self.block_size,
            minimum=0.75,
            maximum=2.0,
        )
        final_length_term = values_to_term(length_losses, length_weights)

        continuation_logits = self.continuation_logits(extracted.contexts)
        continuation_losses = F.cross_entropy(
            continuation_logits.float(), extracted.continue_targets, reduction="none"
        )
        continuation_mask = torch.ones_like(extracted.continue_targets, dtype=torch.bool)
        continuation_weights = _balanced_example_weights(
            extracted.continue_targets,
            continuation_mask,
            classes=2,
            minimum=0.75,
            maximum=2.0,
        )
        continue_term = values_to_term(continuation_losses, continuation_weights)

        content_prediction = content_logits.float().argmax(dim=-1)
        continuation_prediction = continuation_logits.float().argmax(dim=-1)
        length_prediction = length_logits.float().argmax(dim=-1) + 1
        first_active = extracted.content_mask[:, 0]
        predicted_active = content_prediction[extracted.content_mask]
        unique_fraction = (
            predicted_active.unique().numel() / float(predicted_active.numel())
            if predicted_active.numel()
            else 0.0
        )
        return MicroblockSemanticOutput(
            content_term=content_term,
            final_length_term=final_length_term,
            continue_term=continue_term,
            token_accuracy=(
                content_prediction[extracted.content_mask]
                == extracted.targets[extracted.content_mask]
            ).float().mean(),
            first_slot_accuracy=(
                content_prediction[:, 0][first_active]
                == extracted.targets[:, 0][first_active]
            ).float().mean(),
            final_length_accuracy=(
                length_prediction[extracted.final_mask]
                == extracted.lengths[extracted.final_mask]
            ).float().mean(),
            final_length_mae=(
                length_prediction[extracted.final_mask]
                - extracted.lengths[extracted.final_mask]
            ).abs().float().mean(),
            continue_accuracy=(
                continuation_prediction == extracted.continue_targets
            ).float().mean(),
            predicted_continue_fraction=continuation_prediction.float().mean(),
            target_continue_fraction=extracted.continue_targets.float().mean(),
            predicted_unique_fraction=anchor.detach().new_tensor(unique_fraction),
            blocks=extracted.contexts.new_tensor(
                float(extracted.contexts.shape[0]), dtype=torch.float32
            ),
        )


__all__ = [
    "CausalMicroblockSemanticHead",
    "MicroblockSemanticOutput",
    "MicroblockTargets",
    "_balanced_example_weights",
    "extract_microblock_targets",
]

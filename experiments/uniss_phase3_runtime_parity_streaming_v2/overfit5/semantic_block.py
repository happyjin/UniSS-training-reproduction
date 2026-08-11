"""Natural-length parallel BiCodec semantic-block prediction.

The head consumes the Qwen hidden state whose input token is
``START_SEMANTIC``.  Learned slot queries predict all semantic tokens in one
parallel operation.  ``END_SEMANTIC`` is an ordinary extra class, so runtime
block length is selected by the model rather than supplied by an oracle.
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


END_CLASS = c.BICODEC_SEMANTIC_SIZE


@dataclass(frozen=True)
class SemanticBlockOutput:
    term: LossTerm
    token_accuracy: torch.Tensor
    end_accuracy: torch.Tensor
    length_mae: torch.Tensor
    blocks: torch.Tensor


class ParallelSemanticBlockHead(nn.Module):
    """Predict up to ``maximum_semantic_tokens`` plus natural END in parallel."""

    def __init__(
        self,
        hidden_size: int,
        *,
        maximum_semantic_tokens: int = 24,
        end_loss_weight: float = 4.0,
    ) -> None:
        super().__init__()
        if hidden_size <= 0 or maximum_semantic_tokens <= 0 or end_loss_weight <= 0:
            raise ValueError("semantic block geometry must be positive")
        self.hidden_size = int(hidden_size)
        self.maximum_semantic_tokens = int(maximum_semantic_tokens)
        self.end_loss_weight = float(end_loss_weight)
        self.slot_count = self.maximum_semantic_tokens + 1
        self.context_projection = nn.Linear(hidden_size, hidden_size, bias=False)
        self.slot_embeddings = nn.Embedding(self.slot_count, hidden_size)
        self.hidden_projection = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2, bias=False),
            nn.SiLU(),
            nn.Linear(hidden_size * 2, hidden_size, bias=False),
        )
        self.output_norm = nn.LayerNorm(hidden_size)
        self.class_bias = nn.Parameter(torch.zeros(c.BICODEC_SEMANTIC_SIZE + 1))
        nn.init.normal_(self.slot_embeddings.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.context_projection.weight, mean=0.0, std=0.02)
        for module in self.hidden_projection:
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
        for parameter in self.parameters():
            parameter.uniss_lr_new_heads = True

    @staticmethod
    def class_embeddings(word_embedding_weight: torch.Tensor) -> torch.Tensor:
        semantic = word_embedding_weight[
            c.BICODEC_SEMANTIC_OFFSET :
            c.BICODEC_SEMANTIC_OFFSET + c.BICODEC_SEMANTIC_SIZE
        ]
        terminal = word_embedding_weight[c.TOKEN_END_SEMANTIC].unsqueeze(0)
        return torch.cat((semantic, terminal), dim=0)

    def forward(
        self,
        context: torch.Tensor,
        word_embedding_weight: torch.Tensor,
    ) -> torch.Tensor:
        if context.ndim != 2 or context.shape[-1] != self.hidden_size:
            raise ValueError("semantic block context must be [blocks,hidden]")
        slots = self.slot_embeddings.weight.to(context.dtype).unsqueeze(0)
        hidden = self.context_projection(context).unsqueeze(1) + slots
        hidden = hidden + self.hidden_projection(hidden)
        hidden = self.output_norm(hidden)
        classes = self.class_embeddings(word_embedding_weight).to(hidden.dtype)
        return F.linear(hidden, classes, self.class_bias.to(hidden.dtype))

    @torch.inference_mode()
    def decode(
        self,
        context: torch.Tensor,
        word_embedding_weight: torch.Tensor,
    ) -> tuple[tuple[int, ...], bool]:
        if context.ndim == 1:
            context = context.unsqueeze(0)
        if context.shape[0] != 1:
            raise ValueError("runtime semantic decode requires exactly one block")
        choices = self(context, word_embedding_weight)[0].float().argmax(dim=-1)
        values = [int(value) for value in choices]
        if END_CLASS not in values:
            return tuple(values[: self.maximum_semantic_tokens]), False
        end = values.index(END_CLASS)
        return tuple(values[:end]), True

    def training_output(
        self,
        hidden: torch.Tensor,
        labels: torch.Tensor,
        token_roles: torch.Tensor,
        loss_mask: torch.Tensor,
        word_embedding_weight: torch.Tensor,
    ) -> SemanticBlockOutput:
        active = (token_roles == ROLE_SEMANTIC) & (loss_mask > 0)
        previous = torch.zeros_like(active)
        previous[1:] = active[:-1]
        following = torch.zeros_like(active)
        following[:-1] = active[1:]
        starts = (active & ~previous).nonzero(as_tuple=False).flatten()
        ends = (active & ~following).nonzero(as_tuple=False).flatten()
        anchor = sum((parameter.reshape(-1)[0] * 0.0 for parameter in self.parameters()), hidden.sum() * 0.0)
        if not starts.numel():
            zero = anchor.detach().new_zeros(())
            return SemanticBlockOutput(zero_term(anchor), zero, zero, zero, zero)
        if starts.numel() != ends.numel():
            raise ValueError("semantic role spans have unmatched boundaries")
        lengths = ends - starts + 1
        if int(lengths.max()) > self.maximum_semantic_tokens:
            raise ValueError(
                "semantic block exceeds parallel head capacity: "
                f"{int(lengths.max())} > {self.maximum_semantic_tokens}"
            )
        positions = torch.arange(self.slot_count, device=hidden.device)
        gather = starts[:, None] + positions[None, :]
        safe_gather = gather.clamp_max(labels.numel() - 1)
        targets = labels[safe_gather] - c.BICODEC_SEMANTIC_OFFSET
        target_mask = positions[None, :] <= lengths[:, None]
        targets = torch.where(
            positions[None, :] == lengths[:, None],
            torch.full_like(targets, END_CLASS),
            targets,
        )
        if bool(((targets[target_mask] < 0) | (targets[target_mask] > END_CLASS)).any()):
            raise ValueError("parallel semantic targets escaped the legal class range")
        # Cross entropy validates every target even when its downstream loss
        # weight is zero.  Give padded slots an arbitrary legal class before
        # computing the unreduced values; ``target_mask`` removes them.
        targets = torch.where(target_mask, targets, torch.zeros_like(targets))
        logits = self(hidden[starts], word_embedding_weight)
        losses = F.cross_entropy(
            logits.float().reshape(-1, END_CLASS + 1),
            targets.reshape(-1),
            reduction="none",
        ).reshape_as(targets)
        weights = target_mask.float()
        weights = torch.where(
            positions[None, :] == lengths[:, None],
            self.end_loss_weight * weights,
            weights,
        )
        term = values_to_term(losses, weights)
        predictions = logits.float().argmax(dim=-1)
        semantic_mask = target_mask & (positions[None, :] < lengths[:, None])
        token_accuracy = (
            (predictions == targets)[semantic_mask].float().mean()
            if bool(semantic_mask.any())
            else anchor.detach().new_zeros(())
        )
        end_accuracy = (
            predictions.gather(1, lengths[:, None]).squeeze(1) == END_CLASS
        ).float().mean()
        end_hits = predictions == END_CLASS
        sentinel = torch.full_like(predictions, self.slot_count)
        predicted_lengths = torch.where(end_hits, positions[None, :], sentinel).min(dim=1).values
        length_mae = (predicted_lengths - lengths).abs().float().mean()
        return SemanticBlockOutput(
            term,
            token_accuracy,
            end_accuracy,
            length_mae,
            starts.new_tensor(float(starts.numel()), dtype=torch.float32),
        )


__all__ = [
    "END_CLASS",
    "ParallelSemanticBlockHead",
    "SemanticBlockOutput",
]

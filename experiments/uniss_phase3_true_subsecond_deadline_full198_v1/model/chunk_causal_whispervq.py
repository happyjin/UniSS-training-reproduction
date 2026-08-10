"""Small causal adapter over frozen WhisperVQ pre-VQ hidden states.

The released WhisperVQ encoder remains the teacher. Preprocessing exports
bounded-prefix hidden states. This adapter is the trainable causal correction
layer and exposes an exact append-only cache API used by inference.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class CausalAdapterState:
    layer_inputs: tuple[torch.Tensor, ...]

    def detach(self) -> "CausalAdapterState":
        return CausalAdapterState(tuple(value.detach() for value in self.layer_inputs))


class CausalDepthwiseBlock(nn.Module):
    def __init__(self, hidden_size: int, kernel_size: int, expansion: int, dropout: float) -> None:
        super().__init__()
        if kernel_size < 2:
            raise ValueError("kernel_size must be at least two")
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.norm = nn.LayerNorm(hidden_size)
        self.depthwise = nn.Conv1d(
            hidden_size,
            hidden_size,
            kernel_size,
            groups=hidden_size,
            bias=True,
        )
        self.in_proj = nn.Linear(hidden_size, expansion * hidden_size * 2)
        self.out_proj = nn.Linear(expansion * hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def _transform(self, normalized: torch.Tensor, history: torch.Tensor) -> torch.Tensor:
        combined = torch.cat((history, normalized), dim=1)
        convolved = self.depthwise(combined.transpose(1, 2)).transpose(1, 2)
        gate, value = self.in_proj(F.silu(convolved)).chunk(2, dim=-1)
        update = self.out_proj(F.silu(gate) * value)
        return self.dropout(update)

    def forward_full(self, value: torch.Tensor) -> torch.Tensor:
        normalized = self.norm(value)
        history = normalized.new_zeros(
            normalized.shape[0], self.kernel_size - 1, normalized.shape[-1]
        )
        return value + self._transform(normalized, history)

    def forward_chunk(
        self, value: torch.Tensor, history: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        normalized = self.norm(value)
        if history is None:
            history = normalized.new_zeros(
                normalized.shape[0], self.kernel_size - 1, normalized.shape[-1]
            )
        expected = (normalized.shape[0], self.kernel_size - 1, normalized.shape[-1])
        if tuple(history.shape) != expected:
            raise ValueError(f"invalid causal cache shape {tuple(history.shape)}, expected {expected}")
        output = value + self._transform(normalized, history)
        next_history = torch.cat((history, normalized), dim=1)[:, -(self.kernel_size - 1) :]
        return output, next_history


class ChunkCausalWhisperVQAdapter(nn.Module):
    """Causal residual adapter with exact full/chunk parity."""

    def __init__(
        self,
        hidden_size: int = 1280,
        *,
        layers: int = 4,
        kernel_size: int = 5,
        expansion: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if layers <= 0:
            raise ValueError("layers must be positive")
        self.hidden_size = hidden_size
        self.layers = nn.ModuleList(
            CausalDepthwiseBlock(hidden_size, kernel_size, expansion, dropout)
            for _ in range(layers)
        )
        self.output_norm = nn.LayerNorm(hidden_size)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        if hidden.ndim != 3 or hidden.shape[-1] != self.hidden_size:
            raise ValueError("hidden must be [batch,time,hidden_size]")
        value = hidden
        for layer in self.layers:
            value = layer.forward_full(value)
        return self.output_norm(value)

    def forward_chunk(
        self, hidden: torch.Tensor, state: CausalAdapterState | None = None
    ) -> tuple[torch.Tensor, CausalAdapterState]:
        if hidden.ndim != 3 or hidden.shape[-1] != self.hidden_size:
            raise ValueError("hidden must be [batch,time,hidden_size]")
        histories: Sequence[torch.Tensor | None]
        if state is None:
            histories = (None,) * len(self.layers)
        else:
            if len(state.layer_inputs) != len(self.layers):
                raise ValueError("causal state layer count mismatch")
            histories = state.layer_inputs
        value = hidden
        next_histories = []
        for layer, history in zip(self.layers, histories):
            value, next_history = layer.forward_chunk(value, history)
            next_histories.append(next_history)
        return self.output_norm(value), CausalAdapterState(tuple(next_histories))

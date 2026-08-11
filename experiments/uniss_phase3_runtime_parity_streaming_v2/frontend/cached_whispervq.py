"""Exact block-causal K/V caching for the released WhisperVQ encoder layers.

This module is deliberately isolated from the historical UniSS WhisperVQ
implementation.  It accepts the hidden sequence *after* the two convolution
layers, adds the released absolute position embeddings, and runs the existing
WhisperVQ encoder-layer weights with this attention rule::

    current query -> every completed history block + the complete current block

Consequently, positions inside a 160 ms block are bidirectional, while no
position can observe a later block.  ``forward_full`` is the independent
block-mask reference.  ``forward_chunk`` only evaluates the new block and
caches each layer's projected historical keys and values.

The released checkpoint pools after layer 16 and quantizes immediately after
that pooling operation.  The wrapper therefore returns both pooled pre-VQ
hidden states and nearest-codebook token IDs.  Audio feature extraction and
the causal convolution cache intentionally remain outside this minimal core.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class LayerKVCache:
    """Projected keys and values for all completed blocks at one layer."""

    key: torch.Tensor
    value: torch.Tensor

    @property
    def frames(self) -> int:
        return int(self.key.shape[-2])

    def detach(self) -> "LayerKVCache":
        return LayerKVCache(self.key.detach(), self.value.detach())


@dataclass(frozen=True)
class CachedWhisperVQState:
    """Append-only encoder state for one streaming utterance."""

    layers: tuple[LayerKVCache, ...]
    frames_seen: int
    finalized: bool = False

    def detach(self) -> "CachedWhisperVQState":
        return CachedWhisperVQState(
            tuple(layer.detach() for layer in self.layers),
            self.frames_seen,
            self.finalized,
        )


@dataclass(frozen=True)
class CachedWhisperVQOutput:
    """New output produced by either full or incremental evaluation."""

    pre_vq_hidden: torch.Tensor
    quantized_hidden: torch.Tensor
    token_ids: torch.Tensor
    state: CachedWhisperVQState | None = None


def block_causal_attention_mask(
    *,
    batch_size: int,
    sequence_length: int,
    block_frames: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return an additive mask with bidirectional attention inside each block."""

    if batch_size <= 0 or sequence_length <= 0 or block_frames <= 0:
        raise ValueError("batch, sequence length, and block size must be positive")
    positions = torch.arange(sequence_length, device=device)
    query_blocks = torch.div(positions[:, None], block_frames, rounding_mode="floor")
    key_blocks = torch.div(positions[None, :], block_frames, rounding_mode="floor")
    allowed = key_blocks <= query_blocks
    mask = torch.zeros((sequence_length, sequence_length), device=device, dtype=dtype)
    mask.masked_fill_(~allowed, torch.finfo(dtype).min)
    return mask.reshape(1, 1, sequence_length, sequence_length).expand(
        batch_size, -1, -1, -1
    )


class CachedBlockCausalWhisperVQ(nn.Module):
    """Inference-only, exact cached execution of WhisperVQ encoder layers.

    The generic constructor also supports tiny randomly initialized
    ``WhisperVQEncoderLayer`` instances in CPU tests.  Use
    :meth:`from_whispervq_encoder` to share the released 16-layer weights and
    its 16,384-entry codebook without copying or modifying the old model.
    """

    def __init__(
        self,
        layers: Sequence[nn.Module],
        position_embedding: nn.Embedding,
        codebook: nn.Embedding,
        *,
        chunk_ms: int = 160,
        frame_ms: int = 20,
        right_context_ms: int = 0,
        pooling_kernel_size: int = 4,
        pooling_type: str = "avg",
    ) -> None:
        super().__init__()
        if not layers:
            raise ValueError("at least one WhisperVQ layer is required")
        if chunk_ms <= 0 or frame_ms <= 0 or chunk_ms % frame_ms:
            raise ValueError("chunk_ms must be a positive multiple of frame_ms")
        if right_context_ms != 0:
            raise ValueError("this exact cached prototype only supports right_context_ms=0")
        if pooling_kernel_size <= 0:
            raise ValueError("pooling_kernel_size must be positive")

        block_frames = chunk_ms // frame_ms
        if block_frames % pooling_kernel_size:
            raise ValueError(
                "block frames must be divisible by pooling_kernel_size so that "
                "cached pooling never crosses a committed block boundary"
            )
        if pooling_type not in {"avg", "max"}:
            raise ValueError(f"unsupported pooling type: {pooling_type}")

        hidden_size = int(position_embedding.embedding_dim)
        if int(codebook.embedding_dim) != hidden_size:
            raise ValueError("position embedding and codebook dimensions differ")
        for index, layer in enumerate(layers):
            if int(getattr(layer, "embed_dim", -1)) != hidden_size:
                raise ValueError(f"layer {index} hidden size differs from the codebook")

        # ModuleList retains the exact supplied layer objects.  In the released
        # model path this intentionally shares read-only weights instead of
        # allocating a second ~1.4 GB encoder copy.
        self.layers = nn.ModuleList(list(layers))
        self.position_embedding = position_embedding
        self.codebook = codebook
        self.chunk_ms = int(chunk_ms)
        self.frame_ms = int(frame_ms)
        self.right_context_ms = int(right_context_ms)
        self.block_frames = int(block_frames)
        self.pooling_kernel_size = int(pooling_kernel_size)
        self.pooling_type = pooling_type

    @classmethod
    def from_whispervq_encoder(
        cls,
        encoder: nn.Module,
        *,
        chunk_ms: int = 160,
        frame_ms: int = 20,
        right_context_ms: int = 0,
    ) -> "CachedBlockCausalWhisperVQ":
        """Wrap a loaded released ``WhisperVQEncoder`` without changing it."""

        config = encoder.config
        layer_count = len(encoder.layers)
        pooling_position = int(config.pooling_position)
        quantize_position = int(config.quantize_position)
        if pooling_position != layer_count or quantize_position != layer_count:
            raise ValueError(
                "the minimal core requires pooling and VQ immediately after the "
                f"last encoder layer; got layers={layer_count}, "
                f"pooling_position={pooling_position}, "
                f"quantize_position={quantize_position}"
            )
        if encoder.pooling_layer is None or encoder.codebook is None:
            raise ValueError("WhisperVQ encoder has no pooling layer or codebook")
        return cls(
            encoder.layers,
            encoder.embed_positions,
            encoder.codebook,
            chunk_ms=chunk_ms,
            frame_ms=frame_ms,
            right_context_ms=right_context_ms,
            pooling_kernel_size=int(config.pooling_kernel_size),
            pooling_type=str(config.pooling_type),
        )

    @property
    def hidden_size(self) -> int:
        return int(self.position_embedding.embedding_dim)

    def _validate_input(self, convolved_hidden: torch.Tensor) -> None:
        if convolved_hidden.ndim != 3:
            raise ValueError("convolved_hidden must have shape [batch, frames, hidden]")
        if convolved_hidden.shape[0] <= 0 or convolved_hidden.shape[1] <= 0:
            raise ValueError("batch and frame dimensions must be non-empty")
        if convolved_hidden.shape[2] != self.hidden_size:
            raise ValueError(
                f"expected hidden size {self.hidden_size}, got {convolved_hidden.shape[2]}"
            )

    def _add_positions(self, hidden: torch.Tensor, offset: int) -> torch.Tensor:
        end = offset + hidden.shape[1]
        if end > self.position_embedding.num_embeddings:
            raise ValueError(
                f"position {end} exceeds WhisperVQ maximum "
                f"{self.position_embedding.num_embeddings}"
            )
        positions = self.position_embedding.weight[offset:end].to(
            device=hidden.device, dtype=hidden.dtype
        )
        return hidden + positions

    def _pool(self, hidden: torch.Tensor) -> torch.Tensor:
        value = hidden.transpose(1, 2)
        remainder = value.shape[-1] % self.pooling_kernel_size
        if remainder:
            value = F.pad(value, (0, self.pooling_kernel_size - remainder))
        if self.pooling_type == "avg":
            value = F.avg_pool1d(value, self.pooling_kernel_size)
        else:
            value = F.max_pool1d(value, self.pooling_kernel_size)
        return value.transpose(1, 2).contiguous()

    def _quantize(self, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        codebook = self.codebook.weight.to(device=hidden.device, dtype=hidden.dtype)
        flat = hidden.reshape(-1, hidden.shape[-1])
        distances = (
            flat.square().sum(dim=1, keepdim=True)
            + codebook.square().sum(dim=1).unsqueeze(0)
            - 2.0 * flat @ codebook.t()
        )
        token_ids = distances.argmin(dim=1).reshape(hidden.shape[:-1])
        quantized = F.embedding(token_ids, codebook)
        return quantized, token_ids

    @staticmethod
    def _run_layer_cached(
        layer: nn.Module,
        hidden: torch.Tensor,
        cache: LayerKVCache | None,
    ) -> tuple[torch.Tensor, LayerKVCache]:
        """Run one existing WhisperVQ layer on only the new complete block."""

        residual = hidden
        normalized = layer.self_attn_layer_norm(hidden)
        attention = layer.self_attn
        batch, frames, _ = normalized.shape
        queries = attention._shape(
            attention.q_proj(normalized) * attention.scaling, frames, batch
        )
        current_keys = attention._shape(attention.k_proj(normalized), frames, batch)
        current_values = attention._shape(attention.v_proj(normalized), frames, batch)
        if cache is None:
            keys, values = current_keys, current_values
        else:
            expected_prefix = (batch, attention.num_heads, attention.head_dim)
            if (
                cache.key.shape[0] != expected_prefix[0]
                or cache.key.shape[1] != expected_prefix[1]
                or cache.key.shape[3] != expected_prefix[2]
                or cache.value.shape != cache.key.shape
            ):
                raise ValueError("cached K/V geometry differs from the current block")
            keys = torch.cat((cache.key, current_keys), dim=2)
            values = torch.cat((cache.value, current_values), dim=2)

        weights = torch.matmul(queries, keys.transpose(2, 3))
        probabilities = F.softmax(weights, dim=-1)
        probabilities = F.dropout(
            probabilities, p=float(attention.dropout), training=layer.training
        )
        attended = torch.matmul(probabilities, values)
        attended = attended.transpose(1, 2).reshape(batch, frames, attention.embed_dim)
        attended = attention.out_proj(attended)
        attended = F.dropout(attended, p=float(layer.dropout), training=layer.training)
        hidden = residual + attended

        residual = hidden
        hidden = layer.final_layer_norm(hidden)
        hidden = layer.activation_fn(layer.fc1(hidden))
        hidden = F.dropout(
            hidden, p=float(layer.activation_dropout), training=layer.training
        )
        hidden = layer.fc2(hidden)
        hidden = F.dropout(hidden, p=float(layer.dropout), training=layer.training)
        hidden = residual + hidden
        return hidden, LayerKVCache(keys, values)

    def _require_inference_mode(self) -> None:
        if self.training or any(layer.training for layer in self.layers):
            raise RuntimeError(
                "cached WhisperVQ requires eval() so dropout/layerdrop cannot "
                "break deterministic full-vs-cached parity"
            )

    @torch.inference_mode()
    def forward_full(self, convolved_hidden: torch.Tensor) -> CachedWhisperVQOutput:
        """Independent full-utterance reference using one block-causal mask."""

        self._require_inference_mode()
        self._validate_input(convolved_hidden)
        hidden = self._add_positions(convolved_hidden, 0)
        mask = block_causal_attention_mask(
            batch_size=hidden.shape[0],
            sequence_length=hidden.shape[1],
            block_frames=self.block_frames,
            device=hidden.device,
            dtype=hidden.dtype,
        )
        for layer in self.layers:
            hidden = layer(
                hidden,
                mask,
                layer_head_mask=None,
                output_attentions=False,
            )[0]
        pooled = self._pool(hidden)
        quantized, token_ids = self._quantize(pooled)
        return CachedWhisperVQOutput(pooled, quantized, token_ids)

    @torch.inference_mode()
    def forward_chunk(
        self,
        convolved_hidden: torch.Tensor,
        state: CachedWhisperVQState | None = None,
        *,
        is_final: bool = False,
    ) -> CachedWhisperVQOutput:
        """Append one 160 ms block, or flush one final shorter block.

        A non-final short block is rejected.  Because attention inside a block
        is bidirectional, evaluating half of a block and later appending the
        other half would otherwise revise already committed hidden states.
        """

        self._require_inference_mode()
        self._validate_input(convolved_hidden)
        frames = int(convolved_hidden.shape[1])
        if frames > self.block_frames:
            raise ValueError(f"a chunk may contain at most {self.block_frames} frames")
        if frames != self.block_frames and not is_final:
            raise ValueError("a short block can only be submitted with is_final=True")
        if state is None:
            caches: Sequence[LayerKVCache | None] = (None,) * len(self.layers)
            frames_seen = 0
        else:
            if state.finalized:
                raise ValueError("cannot append to a finalized streaming state")
            if len(state.layers) != len(self.layers):
                raise ValueError("streaming state layer count differs from the model")
            if any(layer.frames != state.frames_seen for layer in state.layers):
                raise ValueError("streaming state K/V lengths are internally inconsistent")
            caches = state.layers
            frames_seen = int(state.frames_seen)

        hidden = self._add_positions(convolved_hidden, frames_seen)
        next_caches: list[LayerKVCache] = []
        for layer, cache in zip(self.layers, caches):
            hidden, next_cache = self._run_layer_cached(layer, hidden, cache)
            next_caches.append(next_cache)

        pooled = self._pool(hidden)
        quantized, token_ids = self._quantize(pooled)
        next_state = CachedWhisperVQState(
            tuple(next_caches), frames_seen + frames, bool(is_final)
        )
        return CachedWhisperVQOutput(pooled, quantized, token_ids, next_state)


__all__ = [
    "CachedBlockCausalWhisperVQ",
    "CachedWhisperVQOutput",
    "CachedWhisperVQState",
    "LayerKVCache",
    "block_causal_attention_mask",
]

"""Model components for the true-subsecond joint model."""

from .chunk_causal_whispervq import CausalAdapterState, ChunkCausalWhisperVQAdapter
from .safe_commit_head import SafeCommitHead
from .support_head import ActionHead, SupportOrdinalHead
from .megatron_lora import (
    AdditiveLoRABranch,
    MegatronLoRAController,
    MegatronLoRASummary,
    inject_native_megatron_lora,
)

__all__ = [
    "ActionHead",
    "CausalAdapterState",
    "ChunkCausalWhisperVQAdapter",
    "SafeCommitHead",
    "SupportOrdinalHead",
    "AdditiveLoRABranch",
    "MegatronLoRAController",
    "MegatronLoRASummary",
    "inject_native_megatron_lora",
]

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from experiments.uniss_phase3_event_rollout_joint_pilot15_v2.evaluation.uncached_backend import (
    UncachedHuggingFaceBackend,
)


class _Model(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = torch.nn.Embedding(32, 4)
        self.config = SimpleNamespace(_attn_implementation="eager")
        self.seen_lengths = []

    def get_input_embeddings(self):
        return self.embedding

    def forward(self, *, inputs_embeds, use_cache, output_hidden_states, return_dict):
        assert not use_cache and return_dict
        self.seen_lengths.append(int(inputs_embeds.shape[1]))
        logits = torch.nn.functional.pad(inputs_embeds, (0, 28))
        hidden = (inputs_embeds,) if output_hidden_states else None
        return SimpleNamespace(logits=logits, hidden_states=hidden, past_key_values=None)


class _Objective(torch.nn.Module):
    pass


def test_uncached_backend_recomputes_the_complete_committed_history() -> None:
    model = _Model()
    backend = UncachedHuggingFaceBackend(model, _Objective(), device="cpu")
    first = backend.append_token_ids([1, 2], past_key_values=None)
    second = backend.append_token_ids(
        [3], past_key_values=first.past_key_values, capture_last_hidden=True
    )
    assert model.seen_lengths == [2, 3]
    assert first.past_key_values is None
    assert second.past_key_values is None
    assert second.last_hidden.shape == (1, 4)


def test_uncached_backend_rejects_static_cache() -> None:
    with pytest.raises(ValueError, match="cannot also use"):
        UncachedHuggingFaceBackend(
            _Model(), _Objective(), device="cpu", use_static_cache=True
        )

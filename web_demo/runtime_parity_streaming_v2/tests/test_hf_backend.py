from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn

from training import constants_uniss as c
from web_demo.runtime_parity_streaming_v2.hf_backend import (
    HuggingFaceKVBackend,
)


class FakeModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(c.VOCAB_SIZE, 4)
        self.calls: list[dict[str, object]] = []

    def get_input_embeddings(self):
        return self.embedding

    def forward(
        self,
        *,
        input_ids=None,
        inputs_embeds=None,
        past_key_values=None,
        use_cache,
        output_hidden_states=False,
        return_dict,
    ):
        assert use_cache and return_dict
        hidden = self.embedding(input_ids) if inputs_embeds is None else inputs_embeds
        cache_length = 0 if past_key_values is None else int(past_key_values)
        next_cache = cache_length + hidden.shape[1]
        self.calls.append(
            {
                "input_ids": None if input_ids is None else input_ids.detach().clone(),
                "inputs_embeds": (
                    None if inputs_embeds is None else inputs_embeds.detach().clone()
                ),
                "incoming_cache": past_key_values,
            }
        )
        return SimpleNamespace(
            past_key_values=next_cache,
            logits=hidden,
            hidden_states=(hidden,) if output_hidden_states else None,
        )


class FakeObjective(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.codebook = nn.Embedding(c.GLM_SEMANTIC_SIZE, 4)
        self.frontend_adapter = nn.Identity()
        self.frontend_projection = nn.Linear(4, 4, bias=False)
        with torch.no_grad():
            self.frontend_projection.weight.copy_(torch.eye(4))


def test_hf_backend_keeps_one_cache_and_matches_tick_reset_frontend() -> None:
    model = FakeModel()
    objective = FakeObjective()
    backend = HuggingFaceKVBackend(model, objective, device="cpu")

    first = backend.append_token_ids(
        [c.TOKEN_START_GLM], past_key_values=None, capture_last_hidden=True
    )
    codes = [3, 9]
    canonical = list(c.encode_glm_semantic(codes))
    second = backend.append_source_codes(
        codes, canonical, past_key_values=first.past_key_values
    )
    third = backend.append_token_ids(
        [c.TOKEN_END_GLM],
        past_key_values=second.past_key_values,
        capture_last_hidden=True,
    )

    assert first.past_key_values == 1
    assert second.past_key_values == 3
    assert third.past_key_values == 4
    assert first.last_hidden is not None and third.last_hidden is not None
    assert model.calls[1]["incoming_cache"] == 1
    assert model.calls[1]["input_ids"] is None
    expected = model.embedding(torch.tensor([canonical]))
    expected = expected + objective.codebook(torch.tensor([codes]))
    torch.testing.assert_close(model.calls[1]["inputs_embeds"], expected)


def test_fused_tick_is_one_forward_with_training_identical_embeddings() -> None:
    model = FakeModel()
    objective = FakeObjective()
    backend = HuggingFaceKVBackend(
        model, objective, device="cpu", fuse_ticks=True
    )
    codes = [3, 9]
    canonical = list(c.encode_glm_semantic(codes))
    result = backend.append_tick(codes, canonical, past_key_values=None)

    assert result.past_key_values == 4
    assert result.last_hidden is not None
    assert len(model.calls) == 1
    embeddings = model.calls[0]["inputs_embeds"]
    assert embeddings is not None
    expected_source = model.embedding(torch.tensor([canonical]))
    expected_source = expected_source + objective.codebook(torch.tensor([codes]))
    torch.testing.assert_close(embeddings[:, 1:-1], expected_source)
    torch.testing.assert_close(
        embeddings[:, :1], model.embedding(torch.tensor([[c.TOKEN_START_GLM]]))
    )
    torch.testing.assert_close(
        embeddings[:, -1:], model.embedding(torch.tensor([[c.TOKEN_END_GLM]]))
    )

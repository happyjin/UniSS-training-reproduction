from __future__ import annotations

import hashlib
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from experiments.uniss_phase3_v4_e2e_simuls2st_pilot15_v1.rollout.io import (
    partition_bounds,
)
from experiments.uniss_phase3_v4_e2e_simuls2st_pilot15_v1.rollout.persistent_runtime import (
    PersistentV1ASRSession,
)
from experiments.uniss_phase3_v4_e2e_simuls2st_pilot15_v1.rollout.schema import (
    V1Rollout,
    V1RolloutEvent,
    validate_rollout,
)
from training import constants_uniss as c


DIGEST = hashlib.sha256(b"rollout").hexdigest()


def _rollout() -> V1Rollout:
    return V1Rollout(
        sample_id="sample-1",
        split="valid",
        src_lang="eng",
        source_manifest_record=0,
        v1_checkpoint_sha256=DIGEST,
        v1_hf_sha256=DIGEST,
        runtime_sha256=DIGEST,
        source_audio_sha256=DIGEST,
        events=(
            V1RolloutEvent(
                event_index=0,
                source_end_ms=160,
                visible_glm_tokens=2,
                generated_tokens=(
                    c.TOKEN_WRITE_GENERATE,
                    c.TOKEN_ENG,
                    c.TOKEN_START_CONTENT,
                    42,
                    c.TOKEN_END_CONTENT,
                ),
                content_tokens=(42,),
                v1_source_delta="hello",
                v1_source_prefix="hello",
                reached_content_stop=True,
                write_structure_valid=True,
                early_eos=False,
                noise_severity="exact",
            ),
            V1RolloutEvent(
                event_index=1,
                source_end_ms=320,
                visible_glm_tokens=2,
                generated_tokens=(),
                content_tokens=(),
                v1_source_delta="",
                v1_source_prefix="hello",
                reached_content_stop=True,
                write_structure_valid=True,
                early_eos=False,
                noise_severity="exact",
            ),
        ),
        final_generated_tokens=(c.TOKEN_EOS,),
        final_reached_eos=True,
        full_text="hello",
        metric="wer",
        errors=0,
        reference_units=1,
        error_rate=0.0,
        empty_events=1,
        early_eos_events=0,
        malformed_write_events=0,
        final_visible_glm_tokens=4,
        elapsed_seconds=0.25,
    )


def test_rollout_schema_round_trip_and_append_only_gate() -> None:
    original = _rollout()
    recovered = V1Rollout.from_mapping(__import__("json").loads(original.to_json()))
    assert recovered == original
    validate_rollout(recovered, expected_events=2)
    bad = V1Rollout.from_mapping(__import__("json").loads(original.to_json()))
    object.__setattr__(bad.events[1], "v1_source_prefix", "rewritten")
    with pytest.raises(ValueError, match="append-only"):
        validate_rollout(bad)


def test_partition_bounds_cover_without_gaps() -> None:
    ranges = [partition_bounds(32, rank, 8) for rank in range(8)]
    assert ranges[0] == (0, 4)
    assert ranges[-1] == (28, 32)
    assert all(left[1] == right[0] for left, right in zip(ranges, ranges[1:]))


class _FakeQwen(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(180_480, 4)
        self.calls: list[tuple[int, torch.Tensor]] = []

    def get_input_embeddings(self):
        return self.embedding

    def forward(self, *, inputs_embeds, past_key_values=None, use_cache=True):
        del use_cache
        previous = int(past_key_values or 0)
        self.calls.append((previous, inputs_embeds.detach().clone()))
        logits = torch.full(
            (1, inputs_embeds.shape[1], 180_480),
            -1000.0,
            dtype=torch.float32,
        )
        logits[:, -1, c.TOKEN_END_CONTENT] = 1.0
        return SimpleNamespace(
            logits=logits,
            past_key_values=previous + int(inputs_embeds.shape[1]),
        )


class _FakeTokenizer:
    def __len__(self) -> int:
        return 180_480


def test_persistent_session_appends_only_new_acoustics_and_tokens() -> None:
    torch.manual_seed(7)
    qwen = _FakeQwen()
    speech = torch.randn(3, 4)
    trajectory = SimpleNamespace(src_lang="eng", speaker_global=tuple(range(32)))
    session = PersistentV1ASRSession(qwen, _FakeTokenizer(), speech, trajectory)
    header_length = len(qwen.calls[0][1][0])
    session.append_source_until(2)
    previous, acoustic_call = qwen.calls[1]
    assert previous == header_length
    assert acoustic_call.shape[1] == 4
    assert torch.equal(acoustic_call[0, 1], speech[0])
    assert torch.equal(acoustic_call[0, 2], speech[1])
    assert session.generate(stop_id=c.TOKEN_END_CONTENT, max_tokens=3) == (
        c.TOKEN_END_CONTENT,
    )
    before_final_chunk = qwen.calls[-1][0] + qwen.calls[-1][1].shape[1]
    session.append_source_until(3)
    previous, final_acoustic_call = qwen.calls[-1]
    assert previous == before_final_chunk
    assert final_acoustic_call.shape[1] == 3
    assert torch.equal(final_acoustic_call[0, 1], speech[2])
    assert session.visible_glm == 3

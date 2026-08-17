from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn
from types import SimpleNamespace

from experiments.uniss_phase3_v4_quality_first_true_streaming_pilot15_v2.stage_a_causal_whisper_asr.build_teacher_cache import (
    combine_acoustic,
)
from experiments.uniss_phase3_v4_quality_first_true_streaming_pilot15_v2.stage_a_causal_whisper_asr.same_prefix_teacher import (
    TeacherRequest,
)
from experiments.uniss_phase3_v4_quality_first_true_streaming_pilot15_v2.stage_a_causal_whisper_asr.training.objective import (
    StageAObjective,
)
from experiments.uniss_phase3_v4_quality_first_true_streaming_pilot15_v2.stage_a_causal_whisper_asr.training.pretrain_stage_a_megatron import (
    checkpoint_contains_stage_a_state,
)
from experiments.uniss_phase3_v4_quality_first_true_streaming_pilot15_v2.stage_a_causal_whisper_asr.training import (
    pretrain_stage_a_megatron as entrypoint,
)


class TinyFrontend(nn.Module):
    hidden_size = 2

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("codes", torch.eye(2))

    @property
    def codebook(self) -> torch.Tensor:
        return self.codes


def test_reference_anchor_preserves_teacher_candidates_without_conflicting_label() -> None:
    request = TeacherRequest(
        prompt_ids=(1,),
        target_ids=(2, 3),
        selected_target_indices=(0, 1),
        student_positions=(4, 5),
        reference_labels=(2, 3),
        visible_glm_tokens=1,
        event_index=0,
    )
    summary = {
        "indices": np.asarray([[9, 2], [3, 8]], dtype=np.int32),
        "probabilities": np.asarray([[0.9, 0.1], [0.2, 0.8]], dtype=np.float16),
        "top1": np.asarray([9, 8], dtype=np.int32),
        "confidence": np.asarray([0.9, 0.8], dtype=np.float16),
    }
    arrays, candidates = combine_acoustic(
        [request],
        [summary],
        require_reference_in_topk=True,
        reference_anchor=0.5,
    )
    assert candidates == 2
    assert len(arrays["positions"]) == 2
    reference_mass = (
        arrays["probabilities"].astype(np.float32)
        * (arrays["indices"] == arrays["labels"][:, None])
    ).sum(axis=1)
    assert bool((reference_mass >= 0.5).all())
    assert np.allclose(
        arrays["probabilities"].astype(np.float32).sum(axis=1), 1.0, atol=1e-3
    )


def test_teacher_kl_requires_real_nonzero_batch_fields() -> None:
    objective = StageAObjective(TinyFrontend(), qwen_hidden_size=2)
    logits = torch.zeros(2, 5, requires_grad=True)
    batch = {
        "teacher_batch": torch.tensor([0]),
        "teacher_positions": torch.tensor([0]),
        "teacher_reference_labels": torch.tensor([1]),
        "teacher_indices": torch.tensor([[1, 2]]),
        "teacher_probabilities": torch.tensor([[0.75, 0.25]]),
        "teacher_mask": torch.tensor([[True, True]]),
    }
    term = objective._teacher_kl(logits, batch, logits, original_seq_length=2)
    assert term.denominator.item() == 1
    assert torch.isfinite(term.mean)
    term.mean.backward()
    assert logits.grad is not None

    with pytest.raises(KeyError, match="teacher batch fields"):
        objective._teacher_kl(logits.detach(), {}, logits.detach(), original_seq_length=2)

    empty = dict(batch)
    empty["teacher_batch"] = torch.empty(0, dtype=torch.long)
    empty["teacher_positions"] = torch.empty(0, dtype=torch.long)
    empty["teacher_reference_labels"] = torch.empty(0, dtype=torch.long)
    empty["teacher_indices"] = torch.empty(0, 2, dtype=torch.long)
    empty["teacher_probabilities"] = torch.empty(0, 2)
    empty["teacher_mask"] = torch.empty(0, 2, dtype=torch.bool)
    with pytest.raises(ValueError, match="denominator is zero"):
        objective._teacher_kl(
            logits.detach(), empty, logits.detach(), original_seq_length=2
        )


def test_stage_a_resume_is_detected_without_accepting_plain_phase3() -> None:
    assert checkpoint_contains_stage_a_state(
        ["decoder.layers.weight", "stage_a_objective.bridge_projection.weight"]
    )
    assert checkpoint_contains_stage_a_state(
        ["stage_a_objective.frontend.encoder.layers.0.weight/shard_0_8"]
    )
    assert not checkpoint_contains_stage_a_state(
        ["embedding.word_embeddings.weight", "decoder.layers.weight"]
    )


def test_dataset_factory_returns_the_teacher_backed_adapter(monkeypatch) -> None:
    calls = []

    def fake_dataset(*args, **kwargs):
        calls.append((args, kwargs))
        return "dataset"

    monkeypatch.setattr(entrypoint, "IndexedStageAPackDataset", fake_dataset)
    args = SimpleNamespace(
        stage_a_train_packs="/tmp/train.jsonl",
        stage_a_train_teacher_cache="/tmp/train-cache.jsonl",
        stage_a_valid_teacher_cache="/tmp/valid-cache.jsonl",
        stage_a_max_acoustics_per_pack=2,
        stage_a_teacher_lru_capacity=16,
        seq_length=4096,
    )
    assert entrypoint._dataset(args, "/tmp/train.jsonl", load_audio=False) == "dataset"
    assert calls[0][1]["teacher_cache_manifest"] == "/tmp/train-cache.jsonl"
    assert calls[0][1]["load_audio"] is False

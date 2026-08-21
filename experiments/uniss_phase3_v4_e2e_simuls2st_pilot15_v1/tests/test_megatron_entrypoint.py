from __future__ import annotations

import pytest
import torch
from torch import nn

from experiments.uniss_phase3_v4_e2e_simuls2st_pilot15_v1.training.pretrain_e2e_megatron import (
    _tag_trainable_qwen_and_freeze_v1,
    e2e_chunk_ms_for_progress,
    validate_family_denominators,
    validate_smoke_scope,
    validate_v1_checkpoint_load_policy,
    validate_v1_checkpoint_key_sets,
)


class _TinyCompound(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(8, 4)
        self.decoder = nn.Linear(4, 4)
        self.stage_a_objective = nn.Sequential(nn.Linear(4, 4), nn.LayerNorm(4))


def test_v1_key_audit_requires_exact_compound_model_keys() -> None:
    checkpoint = {
        "embedding.word_embeddings.weight/shard_0_1",
        "decoder.layers.weight/shard_0_1",
        "stage_a_objective.frontend.weight/shard_0_1",
        "optimizer.state.exp_avg.decoder.layers.weight/shard_0_1",
    }
    current = {
        "embedding.word_embeddings.weight",
        "decoder.layers.weight",
        "stage_a_objective.frontend.weight",
    }
    report = validate_v1_checkpoint_key_sets(checkpoint, current)
    assert report["exact_key_match"] is True
    with pytest.raises(RuntimeError, match="key audit failed"):
        validate_v1_checkpoint_key_sets(
            checkpoint, current - {"stage_a_objective.frontend.weight"}
        )


def test_v1_checkpoint_load_policy_ignores_only_unrequested_checkpoint_state() -> None:
    class Args:
        finetune = True
        no_load_optim = True
        no_load_rng = True
        dist_ckpt_strictness = "raise_unexpected"

    args = Args()
    validate_v1_checkpoint_load_policy(args)
    args.dist_ckpt_strictness = "raise_all"
    with pytest.raises(ValueError, match="raise_unexpected"):
        validate_v1_checkpoint_load_policy(args)
    args.dist_ckpt_strictness = "log_all"
    with pytest.raises(ValueError, match="raise_unexpected"):
        validate_v1_checkpoint_load_policy(args)


def test_v1_frontend_is_frozen_and_absent_from_trainable_partition() -> None:
    model = _TinyCompound()
    counts = _tag_trainable_qwen_and_freeze_v1(model)
    assert counts["qwen"] > 0
    assert counts["qwen_io"] > 0
    assert counts["frozen_stage_a"] > 0
    for name, parameter in model.named_parameters():
        if name.startswith("stage_a_objective."):
            assert parameter.requires_grad is False
        else:
            assert parameter.requires_grad is True


def test_e2e_chunk_curriculum_reaches_deployment_chunk() -> None:
    assert {e2e_chunk_ms_for_progress(0.0, index) for index in range(2)} == {
        1280,
        960,
    }
    assert {e2e_chunk_ms_for_progress(0.20, index) for index in range(2)} == {
        960,
        640,
    }
    assert {e2e_chunk_ms_for_progress(0.50, index) for index in range(2)} == {
        640,
        320,
    }
    assert {e2e_chunk_ms_for_progress(0.90, index) for index in range(2)} == {
        320,
        160,
    }


def test_smoke_scope_cannot_bypass_formal_teacher_and_length_gates() -> None:
    validate_smoke_scope(smoke=True, allow_missing_teachers=True, train_iters=2)
    validate_smoke_scope(smoke=False, allow_missing_teachers=False, train_iters=100)
    validate_smoke_scope(
        smoke=False,
        learning_canary=True,
        allow_missing_teachers=False,
        train_iters=100,
    )
    with pytest.raises(ValueError, match="only in smoke mode"):
        validate_smoke_scope(
            smoke=False, allow_missing_teachers=True, train_iters=2
        )
    with pytest.raises(ValueError, match="one or two updates"):
        validate_smoke_scope(
            smoke=True, allow_missing_teachers=False, train_iters=3
        )
    validate_smoke_scope(
        smoke=True,
        allow_missing_teachers=False,
        train_iters=1,
        smoke_family="streaming_asr_event",
    )
    with pytest.raises(ValueError, match="one-family"):
        validate_smoke_scope(
            smoke=False,
            allow_missing_teachers=False,
            train_iters=1,
            smoke_family="streaming_asr_event",
        )
    with pytest.raises(ValueError, match="10--100"):
        validate_smoke_scope(
            smoke=False,
            learning_canary=True,
            allow_missing_teachers=False,
            train_iters=101,
        )
    with pytest.raises(ValueError, match="mutually exclusive"):
        validate_smoke_scope(
            smoke=True,
            learning_canary=True,
            allow_missing_teachers=False,
            train_iters=2,
        )


def test_active_family_denominators_fail_closed() -> None:
    metrics = {
        "denominator/asr_ce": torch.tensor(11.0),
        "denominator/boundary_ce": torch.tensor(3.0),
        "denominator/v1_asr_kl": torch.tensor(11.0),
    }
    validate_family_denominators("streaming_asr_event", metrics)
    metrics["denominator/v1_asr_kl"] = torch.tensor(0.0)
    with pytest.raises(RuntimeError, match="v1_asr_kl"):
        validate_family_denominators("streaming_asr_event", metrics)
    validate_family_denominators(
        "streaming_asr_event", metrics, allow_missing_teachers=True
    )


@pytest.mark.parametrize(
    "family", ("phase3_quality_replay", "phase3_performance_replay")
)
def test_replay_families_require_runtime_replay_ce_denominator(family: str) -> None:
    metrics = {"denominator/replay_ce": torch.tensor(17.0)}
    validate_family_denominators(family, metrics)
    with pytest.raises(RuntimeError, match="replay_ce"):
        validate_family_denominators(
            family, {"denominator/replay_ce": torch.tensor(0.0)}
        )

from __future__ import annotations

import pytest
import torch
from torch import nn

from experiments.uniss_phase3_v4_e2e_simuls2st_pilot15_v1.training.pretrain_e2e_megatron import (
    _tag_trainable_qwen_and_freeze_v1,
    apply_model_generated_semantic_boundary_rollin,
    apply_symmetric_model_generated_semantic_rollin,
    corrupt_interleaved_semantic_prefixes,
    e2e_chunk_ms_for_progress,
    semantic_boundary_rollin_candidates,
    semantic_boundary_rollin_statistics,
    semantic_rollin_continue_candidates,
    semantic_rollin_continue_statistics,
    validate_family_denominators,
    validate_smoke_scope,
    validate_v1_checkpoint_load_policy,
    validate_v1_checkpoint_key_sets,
)
from experiments.uniss_phase3_v4_e2e_simuls2st_pilot15_v1.training.task_samples import (
    FAMILY_INTERLEAVED,
)
from training import constants_uniss as c


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


def test_semantic_prefix_corruption_changes_only_bounded_fragment_suffix() -> None:
    semantic = c.BICODEC_SEMANTIC_OFFSET
    inputs = torch.tensor(
        [[c.TOKEN_START_SEMANTIC, semantic, semantic + 1, semantic + 2, c.TOKEN_END_SEMANTIC]]
    )
    labels = torch.tensor(
        [[semantic, semantic + 1, semantic + 2, c.TOKEN_END_SEMANTIC, c.TOKEN_EOS]]
    )
    original_labels = labels.clone()
    corrupted, changed, eligible, effective_rate = (
        corrupt_interleaved_semantic_prefixes(
            inputs,
            labels,
            family=FAMILY_INTERLEAVED,
            training=True,
            rate=1.0,
            tail=2,
            ramp_updates=0,
            update=7,
        )
    )
    assert changed == eligible == 2
    assert effective_rate == 1.0
    assert torch.equal(corrupted[:, :2], inputs[:, :2])
    assert torch.all(corrupted[:, 2:4] != inputs[:, 2:4])
    assert corrupted[0, 4] == c.TOKEN_END_SEMANTIC
    assert torch.equal(labels, original_labels)
    repeated, *_ = corrupt_interleaved_semantic_prefixes(
        inputs,
        labels,
        family=FAMILY_INTERLEAVED,
        training=True,
        rate=1.0,
        tail=2,
        ramp_updates=0,
        update=7,
    )
    assert torch.equal(repeated, corrupted)


def test_semantic_prefix_corruption_is_disabled_for_eval_and_other_families() -> None:
    inputs = torch.tensor([[c.BICODEC_SEMANTIC_OFFSET]])
    labels = torch.tensor([[c.TOKEN_END_SEMANTIC]])
    for family, training in ((FAMILY_INTERLEAVED, False), ("incremental_mt_event", True)):
        output, changed, eligible, effective_rate = (
            corrupt_interleaved_semantic_prefixes(
                inputs,
                labels,
                family=family,
                training=training,
                rate=1.0,
                tail=1,
                ramp_updates=0,
                update=0,
            )
        )
        assert output is inputs
        assert (changed, eligible, effective_rate) == (0, 0, 0.0)


def test_semantic_boundary_candidates_match_runtime_restricted_choice() -> None:
    semantic = c.BICODEC_SEMANTIC_OFFSET
    inputs = torch.tensor(
        [
            [c.TOKEN_START_SEMANTIC, semantic, semantic + 1],
            [c.TOKEN_START_SEMANTIC, semantic + 2, semantic + 3],
        ]
    )
    labels = torch.tensor(
        [
            [semantic, semantic + 1, c.TOKEN_END_SEMANTIC],
            [semantic + 2, semantic + 3, c.TOKEN_END_SEMANTIC],
        ]
    )
    original_labels = labels.clone()
    logits = torch.full((inputs.numel(), c.TOKEN_END_SEMANTIC + 1), -20.0)
    first_boundary_prediction = 1
    second_boundary_prediction = inputs.shape[1] + 1
    logits[first_boundary_prediction, c.TOKEN_END_SEMANTIC] = 10.0
    logits[first_boundary_prediction, semantic + 7] = 9.0
    logits[second_boundary_prediction, c.TOKEN_END_SEMANTIC] = 9.0
    logits[second_boundary_prediction, semantic + 8] = 10.0

    candidates = semantic_boundary_rollin_candidates(logits, inputs, labels)
    assert candidates[0, 2].item() == -1
    assert candidates[1, 2].item() == semantic + 8
    assert torch.count_nonzero(candidates >= 0).item() == 1
    assert torch.equal(labels, original_labels)


def test_semantic_boundary_first_runtime_token_cannot_end() -> None:
    semantic = c.BICODEC_SEMANTIC_OFFSET
    inputs = torch.tensor([[c.TOKEN_START_SEMANTIC, semantic]])
    labels = torch.tensor([[semantic, c.TOKEN_END_SEMANTIC]])
    logits = torch.full((2, c.TOKEN_END_SEMANTIC + 1), -20.0)
    logits[0, c.TOKEN_END_SEMANTIC] = 30.0
    logits[0, semantic + 9] = 10.0
    candidates = semantic_boundary_rollin_candidates(logits, inputs, labels)
    assert candidates.tolist() == [[-1, semantic + 9]]


def test_semantic_continue_candidates_target_wrong_early_end_inside_tail() -> None:
    semantic = c.BICODEC_SEMANTIC_OFFSET
    inputs = torch.tensor(
        [[c.TOKEN_START_SEMANTIC, semantic + 1, semantic + 2, semantic + 3, c.TOKEN_END_SEMANTIC]]
    )
    labels = torch.tensor(
        [[semantic + 1, semantic + 2, semantic + 3, c.TOKEN_END_SEMANTIC, c.TOKEN_EOS]]
    )
    logits = torch.full((inputs.numel(), c.TOKEN_END_SEMANTIC + 1), -20.0)
    logits[1, c.TOKEN_END_SEMANTIC] = 10.0
    logits[1, semantic + 7] = 9.0
    candidates = semantic_rollin_continue_candidates(
        logits,
        inputs,
        labels,
        sample_boundaries=[[(0, 5)]],
        tail=3,
    )
    assert candidates.tolist() == [[-1, -1, semantic + 7, -1, -1]]


def test_symmetric_rollin_type_ratio_and_one_per_sample() -> None:
    semantic = c.BICODEC_SEMANTIC_OFFSET
    inputs = torch.arange(12, dtype=torch.long).reshape(2, 6) + semantic
    end_candidates = torch.full_like(inputs, -1)
    continue_candidates = torch.full_like(inputs, -1)
    for row in range(2):
        end_candidates[row, 2] = semantic + 100 + row
        continue_candidates[row, 1] = semantic + 200 + row
        end_candidates[row, 5] = semantic + 300 + row
        continue_candidates[row, 4] = semantic + 400 + row
    boundaries = [[(0, 3), (3, 6)], [(0, 3), (3, 6)]]
    ends = apply_symmetric_model_generated_semantic_rollin(
        inputs,
        end_candidates,
        continue_candidates,
        sample_boundaries=boundaries,
        family=FAMILY_INTERLEAVED,
        training=True,
        rate=1.0,
        ramp_updates=0,
        continue_ratio=0.0,
        update=11,
    )
    continues = apply_symmetric_model_generated_semantic_rollin(
        inputs,
        end_candidates,
        continue_candidates,
        sample_boundaries=boundaries,
        family=FAMILY_INTERLEAVED,
        training=True,
        rate=1.0,
        ramp_updates=0,
        continue_ratio=1.0,
        update=11,
    )
    assert ends.selected_samples == continues.selected_samples == 4
    assert ends.eligible_samples == continues.eligible_samples == 4
    assert ends.selected_end_samples == 4
    assert ends.selected_continue_samples == 0
    assert continues.selected_end_samples == 0
    assert continues.selected_continue_samples == 4
    assert not bool((ends.end_mask & ends.continue_mask).any())
    assert not bool((continues.end_mask & continues.continue_mask).any())
    for result in (ends, continues):
        for row in range(2):
            assert int(result.selected_mask[row, :3].sum()) == 1
            assert int(result.selected_mask[row, 3:].sum()) == 1


def test_symmetric_rollin_hash_tracks_configured_type_ratio_at_scale() -> None:
    semantic = c.BICODEC_SEMANTIC_OFFSET
    samples = 512
    inputs = torch.arange(samples * 2, dtype=torch.long) % 1000 + semantic
    inputs = inputs.reshape(1, -1)
    end_candidates = torch.full_like(inputs, -1)
    continue_candidates = torch.full_like(inputs, -1)
    boundaries = [[]]
    for sample in range(samples):
        start = sample * 2
        boundaries[0].append((start, start + 2))
        continue_candidates[0, start] = semantic + 1001 + sample
        end_candidates[0, start + 1] = semantic + 2001 + sample
    result = apply_symmetric_model_generated_semantic_rollin(
        inputs,
        end_candidates,
        continue_candidates,
        sample_boundaries=boundaries,
        family=FAMILY_INTERLEAVED,
        training=True,
        rate=1.0,
        ramp_updates=0,
        continue_ratio=0.5,
        update=23,
    )
    continue_fraction = result.selected_continue_samples / result.selected_samples
    assert result.selected_samples == samples
    assert 0.45 <= continue_fraction <= 0.55


def test_semantic_boundary_rollin_is_deterministic_and_changes_only_boundaries() -> None:
    semantic = c.BICODEC_SEMANTIC_OFFSET
    inputs = torch.arange(100, dtype=torch.long) + semantic
    candidates = torch.arange(100, dtype=torch.long) + semantic + 100
    expected_inputs = inputs.clone()
    first = apply_model_generated_semantic_boundary_rollin(
        inputs,
        candidates,
        sample_boundaries=[[(index, index + 1) for index in range(100)]],
        family=FAMILY_INTERLEAVED,
        training=True,
        rate=0.5,
        ramp_updates=10,
        update=4,
    )
    second = apply_model_generated_semantic_boundary_rollin(
        inputs,
        candidates,
        sample_boundaries=[[(index, index + 1) for index in range(100)]],
        family=FAMILY_INTERLEAVED,
        training=True,
        rate=0.5,
        ramp_updates=10,
        update=4,
    )
    rolled, mask, selected, eligible, changed, effective_rate = first[:6]
    assert torch.equal(rolled, second[0])
    assert torch.equal(mask, second[1])
    assert (selected, eligible, changed, effective_rate) == (
        second[2],
        second[3],
        second[4],
        second[5],
    )
    assert first[6:] == second[6:]
    assert first[6] == selected
    assert first[7] == eligible
    assert effective_rate == pytest.approx(0.25)
    assert eligible == 100
    assert selected == changed == int(mask.sum())
    assert 0 < selected < eligible
    assert torch.equal(rolled[~mask], inputs[~mask])
    assert torch.equal(rolled[mask], candidates[mask])
    assert torch.equal(inputs, expected_inputs)


def test_semantic_boundary_rollin_is_disabled_for_eval_and_other_families() -> None:
    inputs = torch.tensor([c.BICODEC_SEMANTIC_OFFSET])
    candidates = torch.tensor([c.BICODEC_SEMANTIC_OFFSET + 1])
    for family, training in ((FAMILY_INTERLEAVED, False), ("incremental_mt_event", True)):
        result = (
            apply_model_generated_semantic_boundary_rollin(
                inputs,
                candidates,
                sample_boundaries=[[(0, 1)]],
                family=family,
                training=training,
                rate=1.0,
                ramp_updates=0,
                update=0,
            )
        )
        output, mask, selected, eligible, changed, effective_rate = result[:6]
        assert output is inputs
        assert not bool(mask.any())
        assert (selected, eligible, changed, effective_rate) == (0, 0, 0, 0.0)
        assert result[6:] == (0, 0)


def test_semantic_boundary_rollin_selects_at_most_one_boundary_per_sample() -> None:
    semantic = c.BICODEC_SEMANTIC_OFFSET
    inputs = torch.arange(16, dtype=torch.long).reshape(2, 8) + semantic
    candidates = torch.full_like(inputs, -1)
    for row in range(2):
        candidates[row, 1] = semantic + 101 + row
        candidates[row, 2] = semantic + 201 + row
        candidates[row, 5] = semantic + 301 + row
        candidates[row, 6] = semantic + 401 + row
    result = apply_model_generated_semantic_boundary_rollin(
        inputs,
        candidates,
        sample_boundaries=[[(0, 4), (4, 8)], [(0, 4), (4, 8)]],
        family=FAMILY_INTERLEAVED,
        training=True,
        rate=1.0,
        ramp_updates=0,
        update=7,
    )
    rolled, mask, selected, eligible, changed, effective_rate = result[:6]
    selected_samples, eligible_samples = result[6:]
    assert (selected, eligible, changed) == (4, 8, 4)
    assert (selected_samples, eligible_samples, effective_rate) == (4, 4, 1.0)
    for row in range(2):
        assert int(mask[row, :4].sum()) == 1
        assert int(mask[row, 4:].sum()) == 1
    assert torch.equal(rolled[~mask], inputs[~mask])
    assert torch.equal(rolled[mask], candidates[mask])


def test_semantic_boundary_rollin_diagnostics_use_selected_end_rows() -> None:
    semantic = c.BICODEC_SEMANTIC_OFFSET
    logits = torch.full((2, c.TOKEN_END_SEMANTIC + 1), -10.0)
    logits[1, semantic] = 2.0
    logits[1, c.TOKEN_END_SEMANTIC] = 5.0
    labels = torch.tensor([semantic, c.TOKEN_END_SEMANTIC])
    selected = torch.tensor([False, True])
    ce_sum, margin_sum, count = semantic_boundary_rollin_statistics(
        logits, labels, selected
    )
    expected_ce = torch.nn.functional.cross_entropy(
        logits[1:2], labels[1:2], reduction="sum"
    )
    assert count == 1
    assert torch.allclose(ce_sum, expected_ce)
    assert margin_sum.item() == pytest.approx(3.0)


def test_semantic_continue_rollin_diagnostic_is_target_minus_end() -> None:
    semantic = c.BICODEC_SEMANTIC_OFFSET
    logits = torch.full((2, c.TOKEN_END_SEMANTIC + 1), -10.0)
    logits[1, semantic + 1] = 5.0
    logits[1, c.TOKEN_END_SEMANTIC] = 2.0
    labels = torch.tensor([semantic, semantic + 1])
    margin_sum, count = semantic_rollin_continue_statistics(
        logits, labels, torch.tensor([False, True])
    )
    assert count == 1
    assert margin_sum.item() == pytest.approx(3.0)


def test_smoke_scope_cannot_bypass_formal_teacher_and_length_gates() -> None:
    validate_smoke_scope(smoke=True, allow_missing_teachers=True, train_iters=2)
    validate_smoke_scope(smoke=False, allow_missing_teachers=False, train_iters=100)
    validate_smoke_scope(
        smoke=False,
        learning_canary=True,
        allow_missing_teachers=False,
        train_iters=100,
        phase_stratified_canary=True,
    )
    with pytest.raises(ValueError, match="requires --e2e-learning-canary"):
        validate_smoke_scope(
            smoke=False,
            learning_canary=False,
            allow_missing_teachers=False,
            train_iters=100,
            phase_stratified_canary=True,
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

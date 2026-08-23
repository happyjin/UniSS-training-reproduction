from __future__ import annotations

import torch

from training import constants_uniss as c

from experiments.uniss_phase3_v4_e2e_simuls2st_pilot15_v1.training.cache_reader import (
    TeacherPosterior,
)
from experiments.uniss_phase3_v4_e2e_simuls2st_pilot15_v1.training.objective import (
    E2ELossWeights,
    LogitConsistencyPair,
    SpeakerContinuityPair,
    commit_consistency_kl,
    commit_pairs_from_full_logits,
    compute_e2e_objective,
    distributed_e2e_objective,
    flattened_e2e_objective,
    flattened_rollin_semantic_continue_decision_margin_term,
    flattened_rollin_semantic_continue_margin_term,
    flattened_rollin_semantic_end_terms,
    flattened_semantic_continue_margin_term,
    flattened_semantic_end_margin_term,
    speaker_continuity_loss,
    token_ce_terms,
    token_nll_from_logits,
    topk_teacher_kl,
)
from experiments.uniss_phase3_v4_e2e_simuls2st_pilot15_v1.training.task_samples import (
    LOSS_ASR,
    LOSS_BOUNDARY,
    LOSS_EOS,
    LOSS_MT,
    LOSS_NONE,
    LOSS_REPLAY,
    LOSS_SEMANTIC,
)


def test_token_ce_terms_keep_independent_numerators_and_denominators() -> None:
    nll = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 100.0]])
    kinds = torch.tensor(
        [[LOSS_ASR, LOSS_MT, LOSS_SEMANTIC, LOSS_BOUNDARY, LOSS_EOS, LOSS_REPLAY, LOSS_NONE]]
    )
    terms = token_ce_terms(nll, kinds)
    assert terms["asr_ce"].numerator.item() == 1.0
    assert terms["mt_ce"].loss.item() == 2.0
    assert terms["semantic_ce"].denominator.item() == 1.0
    assert terms["boundary_ce"].loss.item() == 4.0
    assert terms["eos_ce"].loss.item() == 5.0
    assert terms["replay_ce"].loss.item() == 6.0


def test_flattened_semantic_end_ce_is_normalized_independently() -> None:
    vocab = c.TOKEN_END_SEMANTIC + 1
    logits = torch.zeros((2, vocab), requires_grad=True)
    labels = torch.tensor([c.TOKEN_END_SEMANTIC, c.TOKEN_END_CONTENT])
    kinds = torch.tensor([LOSS_BOUNDARY, LOSS_BOUNDARY])
    terms = flattened_e2e_objective(
        logits=logits,
        labels=labels,
        loss_kinds=kinds,
        batch={},
        original_seq_length=2,
    )
    assert terms["boundary_ce"].denominator.item() == 2
    assert terms["content_end_ce"].denominator.item() == 1
    assert terms["semantic_end_ce"].denominator.item() == 1
    assert torch.allclose(
        terms["semantic_end_ce"].loss,
        torch.tensor(float(vocab)).log(),
    )


def test_semantic_end_margin_matches_restricted_greedy_decision() -> None:
    vocab = c.TOKEN_END_SEMANTIC + 1
    logits = torch.zeros((2, vocab), requires_grad=True)
    semantic = c.BICODEC_SEMANTIC_OFFSET
    with torch.no_grad():
        logits[0, semantic] = 3.0
        logits[0, c.TOKEN_END_SEMANTIC] = 2.0
        logits[1, semantic] = 1.0
        logits[1, c.TOKEN_END_SEMANTIC] = 4.0
    labels = torch.tensor([c.TOKEN_END_SEMANTIC, c.TOKEN_END_SEMANTIC])
    kinds = torch.tensor([LOSS_BOUNDARY, LOSS_BOUNDARY])
    term = flattened_semantic_end_margin_term(
        logits, labels, kinds, margin=0.5
    )
    assert term.denominator.item() == 2
    assert torch.allclose(term.loss, torch.tensor(0.75))
    term.loss.backward()
    assert logits.grad is not None
    assert logits.grad[0, c.TOKEN_END_SEMANTIC] < 0
    assert logits.grad[0, semantic] > 0
    assert logits.grad[1].abs().sum() == 0


def test_rollin_semantic_end_terms_normalize_only_selected_hard_rows() -> None:
    vocab = c.TOKEN_END_SEMANTIC + 1
    semantic = c.BICODEC_SEMANTIC_OFFSET
    logits = torch.zeros((3, vocab), requires_grad=True)
    with torch.no_grad():
        logits[0, semantic] = 6.0
        logits[0, c.TOKEN_END_SEMANTIC] = 1.0
        logits[1, semantic] = 1.0
        logits[1, c.TOKEN_END_SEMANTIC] = 6.0
        logits[2, semantic] = 7.0
        logits[2, c.TOKEN_END_SEMANTIC] = 1.0
    labels = torch.tensor(
        [c.TOKEN_END_SEMANTIC, c.TOKEN_END_SEMANTIC, c.TOKEN_END_SEMANTIC]
    )
    kinds = torch.tensor([LOSS_BOUNDARY, LOSS_BOUNDARY, LOSS_BOUNDARY])
    selected = torch.tensor([True, False, False])
    ce, margin = flattened_rollin_semantic_end_terms(
        logits, labels, kinds, selected, margin=2.0
    )
    assert ce.denominator.item() == 1
    assert margin.denominator.item() == 1
    assert margin.loss.item() == 7.0
    (ce.loss + margin.loss).backward()
    assert logits.grad is not None
    assert logits.grad[0].abs().sum() > 0
    assert logits.grad[1:].abs().sum() == 0


def test_rollin_semantic_end_terms_reject_non_end_rows() -> None:
    vocab = c.TOKEN_END_SEMANTIC + 1
    logits = torch.zeros((1, vocab))
    labels = torch.tensor([c.TOKEN_END_CONTENT])
    kinds = torch.tensor([LOSS_BOUNDARY])
    try:
        flattened_rollin_semantic_end_terms(
            logits, labels, kinds, torch.tensor([True]), margin=2.0
        )
    except ValueError as error:
        assert "non-END" in str(error)
    else:
        raise AssertionError("non-END roll-in row was accepted")


def test_rollin_semantic_continue_margin_uses_only_selected_model_history_rows() -> None:
    vocab = c.TOKEN_END_SEMANTIC + 1
    semantic = c.BICODEC_SEMANTIC_OFFSET
    logits = torch.zeros((3, vocab), requires_grad=True)
    with torch.no_grad():
        logits[0, semantic] = 2.0
        logits[0, c.TOKEN_END_SEMANTIC] = 5.0
        logits[1, semantic + 1] = 7.0
        logits[1, c.TOKEN_END_SEMANTIC] = 1.0
    labels = torch.tensor([semantic, semantic + 1, c.TOKEN_END_SEMANTIC])
    kinds = torch.tensor([LOSS_SEMANTIC, LOSS_SEMANTIC, LOSS_BOUNDARY])
    term = flattened_rollin_semantic_continue_margin_term(
        logits,
        labels,
        kinds,
        torch.tensor([True, False, False]),
        margin=1.0,
    )
    assert term.denominator.item() == 1
    assert term.loss.item() == 4.0
    term.loss.backward()
    assert logits.grad is not None
    assert logits.grad[0, semantic] < 0
    assert logits.grad[0, c.TOKEN_END_SEMANTIC] > 0
    assert logits.grad[1:].abs().sum() == 0


def test_rollin_semantic_continue_decision_margin_corrects_exact_early_end_row() -> None:
    vocab = c.TOKEN_END_SEMANTIC + 1
    semantic = c.BICODEC_SEMANTIC_OFFSET
    logits = torch.zeros((3, vocab), requires_grad=True)
    with torch.no_grad():
        logits[0, semantic + 7] = 3.0
        logits[0, c.TOKEN_END_SEMANTIC] = 5.0
        logits[1, semantic + 8] = 9.0
        logits[1, c.TOKEN_END_SEMANTIC] = 1.0
    labels = torch.tensor([semantic + 1, semantic + 2, c.TOKEN_END_SEMANTIC])
    kinds = torch.tensor([LOSS_SEMANTIC, LOSS_SEMANTIC, LOSS_BOUNDARY])
    term = flattened_rollin_semantic_continue_decision_margin_term(
        logits,
        labels,
        kinds,
        torch.tensor([True, False, False]),
        margin=1.0,
    )
    assert term.denominator.item() == 1
    assert term.loss.item() == 3.0
    term.loss.backward()
    assert logits.grad is not None
    assert logits.grad[0, semantic + 7] < 0
    assert logits.grad[0, c.TOKEN_END_SEMANTIC] > 0
    assert logits.grad[1:].abs().sum() == 0


def test_semantic_continue_margin_uses_only_same_sample_pre_end_tail() -> None:
    vocab = c.TOKEN_END_SEMANTIC + 1
    semantic = c.BICODEC_SEMANTIC_OFFSET
    logits = torch.zeros((8, vocab), requires_grad=True)
    labels = torch.tensor(
        [
            semantic,
            semantic + 1,
            c.TOKEN_END_SEMANTIC,
            c.TOKEN_END_CONTENT,
            semantic + 2,
            semantic + 3,
            c.TOKEN_END_SEMANTIC,
            c.TOKEN_END_CONTENT,
        ]
    )
    kinds = torch.tensor(
        [
            LOSS_SEMANTIC,
            LOSS_SEMANTIC,
            LOSS_BOUNDARY,
            LOSS_BOUNDARY,
            LOSS_SEMANTIC,
            LOSS_SEMANTIC,
            LOSS_BOUNDARY,
            LOSS_BOUNDARY,
        ]
    )
    with torch.no_grad():
        logits[:, c.TOKEN_END_SEMANTIC] = 2.0
        logits[0, semantic] = 4.0
        logits[1, semantic + 1] = 0.0
        logits[4, semantic + 2] = 3.0
        logits[5, semantic + 3] = 5.0
    term = flattened_semantic_continue_margin_term(
        logits,
        labels,
        kinds,
        original_seq_length=8,
        sample_boundaries=[[(0, 4), (4, 8)]],
        tail=2,
        margin=1.0,
    )
    assert term.denominator.item() == 4
    assert term.loss.item() == 0.75
    term.loss.backward()
    assert logits.grad is not None
    assert logits.grad[1].abs().sum() > 0
    assert logits.grad[[0, 2, 3, 4, 5, 6, 7]].abs().sum() == 0


def test_topk_teacher_kl_is_zero_when_student_matches_teacher_distribution() -> None:
    probabilities = torch.tensor([[0.75, 0.25], [0.60, 0.40]])
    logits = probabilities.log()
    posterior = TeacherPosterior(
        cache_kind="phase3",
        sample_id="sample",
        source_manifest_record=0,
        request_id=0,
        indices=torch.tensor([[0, 1], [0, 1]]),
        probabilities=probabilities,
        reference_labels=torch.tensor([0, 0]),
        top1=torch.tensor([0, 0]),
        confidence=torch.tensor([0.75, 0.60]),
    )
    term = topk_teacher_kl(
        [
            {
                "cache_kind": "phase3",
                "packed_start": 0,
                "packed_stop": 2,
                "posterior": posterior,
                "student_logits": logits,
            }
        ],
        cache_kind="phase3",
    )
    assert term.denominator.item() == 2
    assert abs(term.loss.item()) < 1e-6


def test_consistency_and_speaker_losses_use_stop_gradient_teacher_branch() -> None:
    previous = torch.tensor([[3.0, 1.0]], requires_grad=True)
    current = torch.tensor([[1.0, 3.0]], requires_grad=True)
    commit = commit_consistency_kl(
        [LogitConsistencyPair(previous, current)]
    )
    speaker_previous = torch.tensor([[1.0, 0.0]], requires_grad=True)
    speaker_current = torch.tensor([[0.0, 1.0]], requires_grad=True)
    speaker = speaker_continuity_loss(
        [SpeakerContinuityPair(speaker_previous, speaker_current)]
    )
    (commit.loss + speaker.loss).backward()
    assert previous.grad is None
    assert current.grad is not None
    assert speaker_previous.grad is None
    assert speaker_current.grad is not None
    assert commit.loss.item() > 0
    assert speaker.loss.item() == 1.0


def test_commit_binding_extracts_old_and_new_prefix_logits() -> None:
    logits = torch.arange(2 * 8 * 3, dtype=torch.float32).reshape(2, 8, 3)
    pairs = commit_pairs_from_full_logits(
        logits,
        [
            {
                "batch_index": 1,
                "previous_packed_start": 1,
                "previous_packed_stop": 3,
                "current_packed_start": 5,
                "current_packed_stop": 7,
            }
        ],
    )
    assert torch.equal(pairs[0].previous_logits, logits[1, 1:3])
    assert torch.equal(pairs[0].current_logits, logits[1, 5:7])


def test_full_objective_applies_documented_weights_and_balances_boundary_eos() -> None:
    logits = torch.tensor(
        [[[3.0, 1.0], [1.0, 3.0], [2.0, 0.0], [0.0, 2.0], [2.0, 0.0], [0.0, 2.0]]],
        requires_grad=True,
    )
    labels = torch.tensor([[0, 1, 0, 1, 0, 1]])
    kinds = torch.tensor(
        [[LOSS_ASR, LOSS_MT, LOSS_SEMANTIC, LOSS_BOUNDARY, LOSS_EOS, LOSS_REPLAY]]
    )
    token_nll = token_nll_from_logits(logits, labels)
    weights = E2ELossWeights(
        v1_asr_kl=0.0,
        phase3_kl=0.0,
        commit_consistency=0.0,
        speaker_continuity=0.0,
    )
    output = compute_e2e_objective(
        token_nll=token_nll,
        loss_kinds=kinds,
        weights=weights,
    )
    expected = (
        output.terms["asr_ce"].loss
        + output.terms["mt_ce"].loss
        + output.terms["semantic_ce"].loss
        + 0.5 * output.terms["replay_ce"].loss
        + 0.1
        * (
            output.terms["boundary_ce"].loss
            + output.terms["eos_ce"].loss
        )
        / 2
    )
    assert torch.allclose(output.total, expected)
    assert output.terms["boundary_eos"].denominator.item() == 2
    output.total.backward()
    assert torch.isfinite(logits.grad).all()


def test_flattened_megatron_objective_indexes_teacher_and_commit_positions() -> None:
    logits = torch.tensor(
        [
            [3.0, 1.0, 0.0],
            [1.0, 3.0, 0.0],
            [2.0, 0.0, 1.0],
            [3.0, 1.0, 0.0],
            [1.0, 3.0, 0.0],
            [2.0, 0.0, 1.0],
        ],
        requires_grad=True,
    )
    labels = torch.tensor([0, 1, 0, 0, 1, 0])
    kinds = torch.tensor(
        [LOSS_ASR, LOSS_MT, LOSS_SEMANTIC, LOSS_BOUNDARY, LOSS_EOS, LOSS_REPLAY]
    )
    batch = {
        "teacher_batch": torch.tensor([0]),
        "teacher_positions": torch.tensor([0]),
        "teacher_cache_kind": torch.tensor([0]),
        "teacher_indices": torch.tensor([[0, 1]]),
        "teacher_probabilities": torch.tensor([[0.8, 0.2]]),
        "commit_batch": torch.tensor([0]),
        "commit_previous_positions": torch.tensor([0]),
        "commit_current_positions": torch.tensor([3]),
    }
    terms = flattened_e2e_objective(
        logits=logits,
        labels=labels,
        loss_kinds=kinds,
        batch=batch,
        original_seq_length=6,
    )
    weights = E2ELossWeights(phase3_kl=0.0, speaker_continuity=0.0)
    total, metrics = distributed_e2e_objective(terms, weights=weights)
    assert terms["v1_asr_kl"].denominator.item() == 1
    assert terms["commit_consistency"].denominator.item() == 1
    assert metrics["denominator/asr_ce"].item() == 1
    assert metrics["loss/boundary_eos"].item() > 0
    total.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()

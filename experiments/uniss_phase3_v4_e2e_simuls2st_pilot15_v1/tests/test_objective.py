from __future__ import annotations

import torch

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

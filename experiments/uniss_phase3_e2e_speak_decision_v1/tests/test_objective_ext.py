"""CPU unit tests for the speak-decision and repetition terms."""

from __future__ import annotations

import pytest
import torch

from experiments.uniss_phase3_e2e_speak_decision_v1.training import objective_ext as ox
from experiments.uniss_phase3_v4_e2e_simuls2st_pilot15_v1.training.objective import (
    E2E_TERM_NAMES,
    E2E_WEIGHTED_NAMES,
)
from experiments.uniss_phase3_v4_e2e_simuls2st_pilot15_v1.training.task_samples import (
    LOSS_BOUNDARY,
    LOSS_MT,
    LOSS_NONE,
    LOSS_SEMANTIC,
)
from training import constants_uniss as c


VOCAB = max(c.TOKEN_WAIT_READ, c.TOKEN_WRITE_GENERATE) + 64


def _logits(rows: int, *, wait_bias: float = 0.0, write_bias: float = 0.0):
    value = torch.zeros(rows, VOCAB)
    value[:, c.TOKEN_WAIT_READ] = wait_bias
    value[:, c.TOKEN_WRITE_GENERATE] = write_bias
    return value


def test_masks_follow_the_packing_convention() -> None:
    """WRITE_GENERATE carries the fragment's kind; only WAIT_READ is boundary."""

    labels = torch.tensor(
        [c.TOKEN_WRITE_GENERATE, c.TOKEN_WAIT_READ, c.TOKEN_WRITE_GENERATE, 7]
    )
    kinds = torch.tensor([LOSS_MT, LOSS_BOUNDARY, LOSS_BOUNDARY, LOSS_MT])
    write, wait = ox.speak_decision_masks(labels, kinds)
    # Row 2 is WRITE_GENERATE but tagged boundary, which the packer never emits.
    assert write.tolist() == [True, False, False, False]
    assert wait.tolist() == [False, True, False, False]


def test_a_confident_correct_decision_costs_less_than_a_wrong_one() -> None:
    labels = torch.tensor([c.TOKEN_WRITE_GENERATE])
    kinds = torch.tensor([LOSS_MT])
    good = ox.speak_decision_terms(
        _logits(1, write_bias=4.0), labels, kinds, margin=1.0
    )[0]
    bad = ox.speak_decision_terms(
        _logits(1, wait_bias=4.0), labels, kinds, margin=1.0
    )[0]
    assert float(good.numerator) < float(bad.numerator)


def test_the_two_classes_are_normalized_independently() -> None:
    """Five WAIT rows against one WRITE row must not swamp the WRITE class."""

    labels = torch.tensor(
        [c.TOKEN_WRITE_GENERATE] + [c.TOKEN_WAIT_READ] * 5
    )
    kinds = torch.tensor([LOSS_MT] + [LOSS_BOUNDARY] * 5)
    write, wait = ox.speak_decision_terms(_logits(6), labels, kinds, margin=1.0)
    assert float(write.denominator) == 1.0
    assert float(wait.denominator) == 5.0
    # Same per-row loss at zero logits, so the means coincide despite 5:1.
    assert float(write.numerator) / 1.0 == pytest.approx(
        float(wait.numerator) / 5.0
    )


def test_softplus_keeps_gradient_after_the_margin_is_met() -> None:
    """A hinge would be flat here; that is why the term is softplus."""

    labels = torch.tensor([c.TOKEN_WRITE_GENERATE])
    kinds = torch.tensor([LOSS_MT])
    logits = _logits(1, write_bias=10.0).requires_grad_(True)
    term = ox.speak_decision_terms(logits, labels, kinds, margin=1.0)[0]
    term.numerator.backward()
    assert float(logits.grad.abs().sum()) > 0.0


def test_absent_class_yields_zero_with_zero_denominator() -> None:
    labels = torch.tensor([c.TOKEN_WAIT_READ])
    kinds = torch.tensor([LOSS_BOUNDARY])
    write, wait = ox.speak_decision_terms(_logits(1), labels, kinds, margin=1.0)
    assert float(write.numerator) == 0.0 and float(write.denominator) == 0.0
    assert float(wait.denominator) == 1.0


def test_negative_margin_is_rejected() -> None:
    with pytest.raises(ValueError):
        ox.speak_decision_terms(
            _logits(1), torch.tensor([7]), torch.tensor([LOSS_MT]), margin=-1.0
        )


def _repetition_case(label_row, kind_row, repeat_token, *, mass=0.9):
    width = len(label_row)
    labels = torch.tensor(label_row)
    kinds = torch.tensor(kind_row)
    logits = torch.zeros(width, VOCAB)
    logits[:, repeat_token] = mass * 20.0
    return logits, labels, kinds, width


def test_repetition_penalty_fires_on_a_repeated_earlier_token() -> None:
    logits, labels, kinds, width = _repetition_case(
        [10, 11, 12, 13], [LOSS_MT] * 4, repeat_token=10
    )
    term = ox.repetition_penalty_term(
        logits, labels, kinds, original_seq_length=width, window=4
    )
    assert float(term.denominator) > 0
    assert float(term.numerator) > 0.5


def test_repeating_the_gold_token_is_not_penalised() -> None:
    """A legitimate repeat of the next gold token is not a defect."""

    logits, labels, kinds, width = _repetition_case(
        [10, 10, 10, 10], [LOSS_MT] * 4, repeat_token=10
    )
    term = ox.repetition_penalty_term(
        logits, labels, kinds, original_seq_length=width, window=4
    )
    assert float(term.numerator) == 0.0


def test_the_window_does_not_cross_a_loss_kind_boundary() -> None:
    """Restricting to one contiguous kind keeps the window inside a fragment.

    With runs MT MT | SEM SEM and a window of four, only the within-run pairs
    (1 <- 0) and (3 <- 2) are eligible.  Row 2 is the first of the SEM run and
    would have to look back across the boundary, so it is never scored -- and
    with an unrestricted window it would have found the repeated token 10.
    """

    logits, labels, kinds, width = _repetition_case(
        [10, 11, 12, 13], [LOSS_MT, LOSS_MT, LOSS_SEMANTIC, LOSS_SEMANTIC],
        repeat_token=10,
    )
    term = ox.repetition_penalty_term(
        logits, labels, kinds, original_seq_length=width, window=4
    )
    assert float(term.denominator) == 2.0
    # All the penalty mass belongs to row 1, the only one whose look-back label
    # is the token the logits actually favour.
    assert float(term.numerator) == pytest.approx(1.0, abs=0.01)


def test_a_single_run_scores_every_position_after_the_first() -> None:
    """The contrast case: one run of four scores rows 1, 2 and 3."""

    logits, labels, kinds, width = _repetition_case(
        [10, 11, 12, 13], [LOSS_MT] * 4, repeat_token=10
    )
    term = ox.repetition_penalty_term(
        logits, labels, kinds, original_seq_length=width, window=4
    )
    assert float(term.denominator) == 3.0


def test_unsupervised_rows_are_excluded() -> None:
    logits, labels, kinds, width = _repetition_case(
        [10, 11], [LOSS_NONE, LOSS_NONE], repeat_token=10
    )
    term = ox.repetition_penalty_term(
        logits, labels, kinds, original_seq_length=width, window=2
    )
    assert float(term.numerator) == 0.0 and float(term.denominator) == 0.0


def test_penalty_is_bounded_by_one_per_row() -> None:
    """A probability, not an unbounded log-ratio, so it cannot dominate CE."""

    logits, labels, kinds, width = _repetition_case(
        [10, 11, 12, 13], [LOSS_MT] * 4, repeat_token=10, mass=1.0
    )
    term = ox.repetition_penalty_term(
        logits, labels, kinds, original_seq_length=width, window=4
    )
    assert float(term.numerator) <= float(term.denominator) + 1e-6


def test_bad_window_or_geometry_is_rejected() -> None:
    logits, labels, kinds, width = _repetition_case(
        [10, 11], [LOSS_MT] * 2, repeat_token=10
    )
    with pytest.raises(ValueError):
        ox.repetition_penalty_term(
            logits, labels, kinds, original_seq_length=width, window=0
        )
    with pytest.raises(ValueError):
        ox.repetition_penalty_term(
            logits, labels, kinds, original_seq_length=3, window=2
        )


def test_the_extended_contracts_append_rather_than_reorder() -> None:
    assert ox.EXTENDED_TERM_NAMES[: len(E2E_TERM_NAMES)] == E2E_TERM_NAMES
    assert ox.EXTENDED_WEIGHTED_NAMES[: len(E2E_WEIGHTED_NAMES)] == E2E_WEIGHTED_NAMES
    assert ox.EXTRA_TERM_NAMES == (
        "speak_decision_write",
        "speak_decision_wait",
        "repetition_penalty",
    )


def test_metric_names_are_extended_in_the_emitted_order() -> None:
    names = ox.extended_objective_metric_names(("loss/asr_ce", "denominator/asr_ce"))
    assert names[:2] == ("loss/asr_ce", "denominator/asr_ce")
    assert "loss/speak_decision_write" in names
    assert "denominator/repetition_penalty" in names
    assert "loss/speak_decision" in names
    assert "weighted/speak_decision" in names
    assert "weighted/repetition_penalty" in names


def _dummy_terms():
    """One LossTerm per established name, plus the three this module adds."""

    from experiments.uniss_phase3_v4_e2e_simuls2st_pilot15_v1.training.objective import (
        LossTerm as BaseLossTerm,
    )

    def term(value: float, denominator: float) -> BaseLossTerm:
        return BaseLossTerm(torch.tensor(value), torch.tensor(denominator))

    terms = {name: term(1.0, 2.0) for name in E2E_TERM_NAMES}
    for name in ox.EXTRA_TERM_NAMES:
        terms[name] = term(0.5, 4.0)
    return terms


def test_distributed_metric_order_matches_the_declared_contract() -> None:
    """The trainer asserts this twice; an interleaved emission broke a real run."""

    import experiments.uniss_phase3_v4_e2e_simuls2st_pilot15_v1.training.pretrain_e2e_megatron as trainer

    _, metrics = ox.distributed_with_speak_decision(
        _dummy_terms(), speak_decision=0.5, repetition_penalty=0.1
    )
    expected = ox.extended_objective_metric_names(trainer.OBJECTIVE_METRIC_NAMES)
    assert tuple(metrics) == expected


def test_distributed_rejects_a_reordered_term_dict() -> None:
    terms = _dummy_terms()
    reordered = {name: terms[name] for name in reversed(list(terms))}
    with pytest.raises(ValueError):
        ox.distributed_with_speak_decision(reordered, speak_decision=0.5)


def test_distributed_adds_both_weighted_contributions_to_the_total() -> None:
    terms = _dummy_terms()
    baseline, _ = ox.distributed_with_speak_decision(
        terms, speak_decision=0.0, repetition_penalty=0.0
    )
    with_speak, _ = ox.distributed_with_speak_decision(
        terms, speak_decision=0.5, repetition_penalty=0.0
    )
    with_both, _ = ox.distributed_with_speak_decision(
        terms, speak_decision=0.5, repetition_penalty=0.1
    )
    assert float(with_speak) > float(baseline)
    assert float(with_both) > float(with_speak)


def test_distributed_rejects_negative_extension_weights() -> None:
    with pytest.raises(ValueError):
        ox.distributed_with_speak_decision(_dummy_terms(), speak_decision=-0.1)


def test_repetition_penalty_never_materialises_a_full_softmax(monkeypatch) -> None:
    """A full-vocabulary softmax is 26 GB at this geometry and OOMed a real run."""

    def forbidden(*args, **kwargs):  # pragma: no cover - must not be reached
        raise AssertionError("full softmax is forbidden in the repetition penalty")

    monkeypatch.setattr(torch, "softmax", forbidden, raising=True)
    monkeypatch.setattr(torch.Tensor, "softmax", forbidden, raising=True)
    monkeypatch.setattr(torch.nn.functional, "softmax", forbidden, raising=True)
    logits, labels, kinds, width = _repetition_case(
        [10, 11, 12, 13], [LOSS_MT] * 4, repeat_token=10
    )
    term = ox.repetition_penalty_term(
        logits, labels, kinds, original_seq_length=width, window=4
    )
    assert float(term.numerator) > 0.0


def test_chunked_logsumexp_matches_the_dense_value() -> None:
    torch.manual_seed(0)
    logits = torch.randn(37, 129)
    scored = torch.zeros(37, dtype=torch.bool)
    scored[[0, 5, 36]] = True
    value = ox._chunked_logsumexp(logits, scored)
    expected = torch.logsumexp(logits.float(), dim=-1)
    assert torch.allclose(value[scored], expected[scored], atol=1e-5)
    # Unscored rows are never normalised, so they stay at zero.
    assert float(value[~scored].abs().max()) == 0.0


def test_chunked_logsumexp_crosses_its_chunk_boundary(monkeypatch) -> None:
    monkeypatch.setattr(ox, "LOGSUMEXP_ROW_CHUNK", 2, raising=True)
    torch.manual_seed(1)
    logits = torch.randn(9, 17)
    scored = torch.ones(9, dtype=torch.bool)
    value = ox._chunked_logsumexp(logits, scored)
    assert torch.allclose(value, torch.logsumexp(logits.float(), dim=-1), atol=1e-5)


def test_penalty_equals_the_naive_softmax_reference() -> None:
    """Numerical equivalence to the implementation the chunking replaced."""

    torch.manual_seed(2)
    width = 6
    logits = torch.randn(width, 64)
    labels = torch.tensor([10, 11, 10, 12, 11, 13])
    kinds = torch.tensor([LOSS_MT] * width)
    term = ox.repetition_penalty_term(
        logits, labels, kinds, original_seq_length=width, window=3
    )
    probabilities = logits.float().softmax(dim=-1)
    expected = 0.0
    counted = set()
    for position in range(width):
        for offset in range(1, 4):
            earlier = position - offset
            if earlier < 0:
                continue
            if int(labels[earlier]) == int(labels[position]):
                continue
            expected += float(probabilities[position, int(labels[earlier])])
            counted.add(position)
    assert float(term.numerator) == pytest.approx(expected, abs=1e-5)
    assert float(term.denominator) == len(counted)

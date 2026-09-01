"""The masks must match what the runtime actually decides, on real gold data.

The previous experiment's masks were hand-written and selected the wrong rows.
These tests derive the expectation from a packed gold record and from the
runtime's own event grammar instead.
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import pytest
import torch

from experiments.uniss_phase3_e2e_continue_end_v1.training import objective_ext as ext
from experiments.uniss_phase3_v4_e2e_simuls2st_pilot15_v1.training.task_samples import (
    LOSS_BOUNDARY,
    LOSS_MT,
)
from training import constants_uniss as c

REPO_ROOT = Path(__file__).resolve().parents[3]
POOL = (
    REPO_ROOT
    / "data/processed/uniss_phase3_v4_e2e_simuls2st_pilot15_v1"
    / "formal_gold_20260818T090515Z/task_pools"
    / "task_pool_formal_p4_20260820T154500Z_train/packed"
    / "train_interleaved_e2e_s2st.jsonl"
)
VOCAB = max(
    c.TOKEN_WRITE_GENERATE,
    c.TOKEN_WAIT_READ,
    c.TOKEN_END_CONTENT,
    c.TOKEN_TASK_TTS,
    c.TOKEN_START_GLM,
    c.TOKEN_EOS,
) + 1


@pytest.fixture(scope="module")
def gold_record() -> dict:
    if not POOL.is_file():
        pytest.skip(f"gold pool not present: {POOL}")
    with POOL.open() as handle:
        return json.loads(next(itertools.islice(handle, 1)))


def _expected_after_fragment(labels: list[int]) -> list[bool]:
    """Reference implementation of the runtime's own event bookkeeping.

    `run_event` sets ``next_family_index = 0`` per event, advances it past each
    family it writes, and breaks out on WAIT_READ, START_GLM or EOS.  A decision
    row is 'after a fragment' when at least one family token has been emitted
    since the last break.
    """
    out: list[bool] = []
    fragments = 0
    for token in labels:
        out.append(fragments >= 1)
        if token in ext.FAMILY_TOKENS:
            fragments += 1
        elif token in ext.EVENT_TERMINATORS:
            fragments = 0
    return out


def test_the_mask_matches_the_runtime_event_bookkeeping_on_gold_data(gold_record) -> None:
    labels = gold_record["labels"]
    kinds = torch.tensor(gold_record["loss_kinds"], dtype=torch.long)
    labels_t = torch.tensor(labels, dtype=torch.long)
    write, wait = ext.continue_after_fragment_mask(
        labels_t, kinds, original_seq_length=len(labels)
    )
    expected = torch.tensor(_expected_after_fragment(labels), dtype=torch.bool)
    boundary = kinds == LOSS_BOUNDARY
    want_write = (labels_t == c.TOKEN_WRITE_GENERATE) & boundary & expected
    want_wait = (labels_t == c.TOKEN_WAIT_READ) & boundary & expected
    assert torch.equal(write, want_write)
    assert torch.equal(wait, want_wait)


def test_the_mask_excludes_the_first_decision_of_every_event(gold_record) -> None:
    """The first WRITE of an event leads by +28.58 logits and must stay unsupervised."""
    labels = torch.tensor(gold_record["labels"], dtype=torch.long)
    kinds = torch.tensor(gold_record["loss_kinds"], dtype=torch.long)
    write, _ = ext.continue_after_fragment_mask(
        labels, kinds, original_seq_length=labels.numel()
    )
    all_write = (labels == c.TOKEN_WRITE_GENERATE) & (kinds == LOSS_BOUNDARY)
    assert int(write.sum()) < int(all_write.sum()), "no first-decision row was excluded"
    # Every selected row must have a family token earlier in its event.
    selected = torch.nonzero(write).flatten().tolist()
    label_list = gold_record["labels"]
    for index in selected:
        seen_family = False
        for token in reversed(label_list[:index]):
            if token in ext.EVENT_TERMINATORS:
                break
            if token in ext.FAMILY_TOKENS:
                seen_family = True
                break
        assert seen_family, f"row {index} was selected but opens its event"


def test_both_decision_classes_share_one_denominator(gold_record) -> None:
    """The previous run's per-class terms gave the minority class 2.28x weight."""
    labels = torch.tensor(gold_record["labels"], dtype=torch.long)
    kinds = torch.tensor(gold_record["loss_kinds"], dtype=torch.long)
    write, wait = ext.continue_after_fragment_mask(
        labels, kinds, original_seq_length=labels.numel()
    )
    logits = torch.zeros(labels.numel(), VOCAB)
    term = ext.continue_after_fragment_term(
        logits, labels, kinds, margin=1.0, original_seq_length=labels.numel()
    )
    assert int(term.denominator) == int(write.sum()) + int(wait.sum())
    assert int(write.sum()) > 0 and int(wait.sum()) > 0


def test_the_margin_pushes_write_and_wait_in_opposite_directions() -> None:
    labels = torch.tensor(
        [
            c.TOKEN_WRITE_GENERATE,
            c.TOKEN_TASK_ASR,
            c.TOKEN_WRITE_GENERATE,  # after a fragment -> supervised
            c.TOKEN_TASK_S2T_TRANSLATION,
            c.TOKEN_WAIT_READ,  # after fragments -> supervised
        ]
    )
    kinds = torch.full_like(labels, LOSS_BOUNDARY)
    logits = torch.zeros(labels.numel(), VOCAB, requires_grad=True)
    term = ext.continue_after_fragment_term(
        logits, labels, kinds, margin=1.0, original_seq_length=labels.numel()
    )
    (term.numerator / term.denominator).backward()
    grad = logits.grad
    # Row 2 is a WRITE row: raising WRITE_GENERATE must reduce the loss.
    assert grad[2, c.TOKEN_WRITE_GENERATE] < 0
    assert grad[2, c.TOKEN_WAIT_READ] > 0
    # Row 4 is a WAIT row: the opposite.
    assert grad[4, c.TOKEN_WAIT_READ] < 0
    assert grad[4, c.TOKEN_WRITE_GENERATE] > 0


def test_content_end_margin_selects_only_text_fragment_ends(gold_record) -> None:
    labels = torch.tensor(gold_record["labels"], dtype=torch.long)
    kinds = torch.tensor(gold_record["loss_kinds"], dtype=torch.long)
    logits = torch.zeros(labels.numel(), VOCAB)
    term = ext.content_end_margin_term(logits, labels, kinds, margin=2.0)
    expected = int(
        ((labels == c.TOKEN_END_CONTENT) & (kinds == LOSS_BOUNDARY)).sum()
    )
    assert int(term.denominator) == expected > 0
    # END_SEMANTIC is a different terminator and must not be swept in.
    assert c.TOKEN_END_SEMANTIC != c.TOKEN_END_CONTENT


def test_content_end_margin_rewards_dominance_not_mere_likelihood() -> None:
    labels = torch.tensor([c.TOKEN_END_CONTENT])
    kinds = torch.tensor([LOSS_BOUNDARY])
    logits = torch.zeros(1, VOCAB, requires_grad=True)
    term = ext.content_end_margin_term(logits, labels, kinds, margin=2.0)
    (term.numerator / term.denominator).backward()
    grad = logits.grad
    assert grad[0, c.TOKEN_END_CONTENT] < 0, "END_CONTENT must be pushed up"
    assert float(grad[0].sum()) == pytest.approx(0.0, abs=1e-5), (
        "the margin moves probability mass, it does not inflate every logit"
    )


def test_a_satisfied_margin_still_carries_gradient() -> None:
    """softplus, not a hinge: the -2.88 gap needs sustained pressure, not a dead zone."""
    labels = torch.tensor([c.TOKEN_END_CONTENT])
    kinds = torch.tensor([LOSS_BOUNDARY])
    logits = torch.zeros(1, VOCAB)
    logits[0, c.TOKEN_END_CONTENT] = 50.0
    logits.requires_grad_(True)
    term = ext.content_end_margin_term(logits, labels, kinds, margin=2.0)
    value = term.numerator / term.denominator
    assert float(value) > 0.0
    value.backward()
    assert float(logits.grad.abs().sum()) > 0.0


def test_the_speak_decision_term_is_gone() -> None:
    """It moved the decision that matters from -2.88 to -3.75."""
    assert "speak_decision" not in ext.EXTENDED_TERM_NAMES
    assert "speak_decision_write" not in ext.EXTENDED_TERM_NAMES
    assert "speak_decision_wait" not in ext.EXTENDED_TERM_NAMES


def test_the_established_terms_are_preserved_in_order() -> None:
    from experiments.uniss_phase3_v4_e2e_simuls2st_pilot15_v1.training.objective import (
        E2E_TERM_NAMES,
        E2E_WEIGHTED_NAMES,
    )

    assert ext.EXTENDED_TERM_NAMES[: len(E2E_TERM_NAMES)] == E2E_TERM_NAMES
    assert ext.EXTENDED_WEIGHTED_NAMES[: len(E2E_WEIGHTED_NAMES)] == E2E_WEIGHTED_NAMES


def test_empty_masks_return_a_connected_zero() -> None:
    """A batch with no interleaved rows must still take part in the backward pass."""
    labels = torch.tensor([c.TOKEN_START_GLM, c.TOKEN_START_GLM])
    kinds = torch.tensor([LOSS_MT, LOSS_MT])
    logits = torch.zeros(2, VOCAB, requires_grad=True)
    for term in (
        ext.continue_after_fragment_term(
        logits, labels, kinds, margin=1.0, original_seq_length=labels.numel()
    ),
        ext.content_end_margin_term(logits, labels, kinds, margin=2.0),
    ):
        assert float(term.denominator) == 0.0
        assert term.numerator.requires_grad
        assert float(term.numerator) == 0.0


def test_the_scan_does_not_leak_between_sequences_in_a_batch() -> None:
    """Two packed sequences side by side must not share event state."""
    width = 4
    labels = torch.tensor(
        [
            # sequence 0 ends mid-event, one fragment already written
            c.TOKEN_WRITE_GENERATE,
            c.TOKEN_TASK_ASR,
            c.TOKEN_WRITE_GENERATE,
            c.TOKEN_TASK_TTS,
            # sequence 1 opens a fresh event
            c.TOKEN_WRITE_GENERATE,
            c.TOKEN_TASK_ASR,
            c.TOKEN_WAIT_READ,
            c.TOKEN_START_GLM,
        ]
    )
    kinds = torch.full_like(labels, LOSS_BOUNDARY)
    write, _ = ext.continue_after_fragment_mask(
        labels, kinds, original_seq_length=width
    )
    assert bool(write[2]), "second WRITE of sequence 0 is after a fragment"
    assert not bool(write[4]), "sequence 1 opens its own event"
    flat, _ = ext.continue_after_fragment_mask(
        labels, kinds, original_seq_length=labels.numel()
    )
    assert bool(flat[4]), "the leak is real when the batch is scanned as one row"


def test_a_wait_row_is_not_erased_by_its_own_reset() -> None:
    """WAIT_READ is both a decision and an event terminator."""
    labels = torch.tensor(
        [c.TOKEN_WRITE_GENERATE, c.TOKEN_TASK_ASR, c.TOKEN_WAIT_READ, c.TOKEN_START_GLM]
    )
    kinds = torch.full_like(labels, LOSS_BOUNDARY)
    _, wait = ext.continue_after_fragment_mask(
        labels, kinds, original_seq_length=labels.numel()
    )
    assert bool(wait[2])

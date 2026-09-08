"""Roll-in candidates must land only on supervised rows.

A label inside the BiCodec semantic span is not by itself a supervised
semantic row: the p2st TTS family carries earlier fragments' codes in the
prompt, in span on both the input and label side but with LOSS_NONE.  The
binary boundary term asserts LOSS_SEMANTIC on every CONTINUE row and
LOSS_BOUNDARY on every END row, and a run aborted at iteration 15 on exactly
that assertion.
"""
import torch

import training.constants_uniss as c
from experiments.uniss_phase3_v4_e2e_simuls2st_pilot15_v1.training.pretrain_e2e_megatron import (
    semantic_boundary_rollin_candidates,
    semantic_rollin_continue_candidates,
)
from experiments.uniss_phase3_v4_e2e_simuls2st_pilot15_v1.training.task_samples import (
    LOSS_BOUNDARY,
    LOSS_NONE,
    LOSS_SEMANTIC,
)

CODE = c.BICODEC_SEMANTIC_OFFSET + 3
VOCAB = c.TOKEN_END_SEMANTIC + 16


def _row():
    """One packed row: two prompt codes, two supervised codes, then END."""
    inputs = torch.tensor([[CODE, CODE, CODE, CODE, CODE]], dtype=torch.long)
    labels = torch.tensor([[CODE, CODE, CODE, CODE, c.TOKEN_END_SEMANTIC]],
                          dtype=torch.long)
    kinds = torch.tensor(
        [[LOSS_NONE, LOSS_NONE, LOSS_SEMANTIC, LOSS_SEMANTIC, LOSS_BOUNDARY]],
        dtype=torch.long,
    )
    # CONTINUE candidates fire where the model wrongly prefers END, so make
    # END the argmax; the END side needs the opposite, handled per test.
    logits = torch.zeros((5, VOCAB))
    logits[:, CODE] = -5.0
    logits[:, c.TOKEN_END_SEMANTIC] = 5.0
    return logits, inputs, labels, kinds


def _row_preferring_codes():
    """The END side is eligible only when the model would rather continue."""
    logits, inputs, labels, kinds = _row()
    logits[:, CODE] = 5.0
    logits[:, c.TOKEN_END_SEMANTIC] = -5.0
    return logits, inputs, labels, kinds


def test_continue_candidates_skip_unsupervised_prompt_rows():
    logits, inputs, labels, kinds = _row()
    without = semantic_rollin_continue_candidates(
        logits, inputs, labels, sample_boundaries=[[(0, 5)]], tail=12
    )
    with_kinds = semantic_rollin_continue_candidates(
        logits, inputs, labels, sample_boundaries=[[(0, 5)]], tail=12,
        loss_kinds=kinds,
    )
    picked = lambda t: sorted(int(i) for i in (t.reshape(-1) >= 0).nonzero().reshape(-1))
    assert picked(without) == [1, 2, 3], "prompt rows are eligible without loss kinds"
    assert picked(with_kinds) == [2, 3], "only the supervised semantic rows survive"


def test_end_candidates_require_a_supervised_boundary():
    logits, inputs, labels, kinds = _row_preferring_codes()
    kinds[0, 4] = LOSS_NONE          # the END row is no longer supervised
    picked = semantic_boundary_rollin_candidates(logits, inputs, labels, kinds)
    assert int((picked.reshape(-1) >= 0).sum()) == 0


def test_end_candidates_still_fire_on_a_supervised_boundary():
    logits, inputs, labels, kinds = _row_preferring_codes()
    picked = semantic_boundary_rollin_candidates(logits, inputs, labels, kinds)
    assert int((picked.reshape(-1) >= 0).sum()) == 1


def test_omitting_loss_kinds_preserves_the_historical_behaviour():
    logits, inputs, labels, _ = _row_preferring_codes()
    a = semantic_boundary_rollin_candidates(logits, inputs, labels)
    b = semantic_boundary_rollin_candidates(logits, inputs, labels, None)
    assert torch.equal(a, b)

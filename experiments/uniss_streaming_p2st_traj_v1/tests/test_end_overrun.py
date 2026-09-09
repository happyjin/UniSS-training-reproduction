"""Finishing the syllable: the terminator is accepted only when it wins clearly."""
import torch

import training.constants_uniss as c
from experiments.uniss_streaming_p2st_pure_ce_v1.runtime.p2st_cascade import end_margin

END = c.TOKEN_END_SEMANTIC
CODE = c.BICODEC_SEMANTIC_OFFSET + 7
VOCAB = END + 16


def _row(end_logit, code_logit):
    v = torch.full((VOCAB,), -20.0)
    v[END] = end_logit
    v[CODE] = code_logit
    return v


def test_a_clear_win_reports_a_large_margin():
    assert end_margin(_row(5.0, 0.0), END, None) == 5.0


def test_a_narrow_win_reports_a_small_margin():
    assert abs(end_margin(_row(0.2, 0.0), END, None) - 0.2) < 1e-6


def test_a_loss_reports_a_negative_margin():
    assert end_margin(_row(0.0, 1.5), END, None) == -1.5


def test_the_allowed_mask_restricts_which_codes_compete():
    v = _row(1.0, 3.0)
    other = c.BICODEC_SEMANTIC_OFFSET + 9
    v[other] = 0.0
    allowed = torch.tensor([END, other], dtype=torch.long)
    # CODE is outside the allowed set, so it must not set the bar
    assert abs(end_margin(v, END, allowed) - 1.0) < 1e-6


def test_no_legal_code_leaves_the_terminator_unopposed():
    allowed = torch.tensor([END], dtype=torch.long)
    assert end_margin(_row(1.0, 9.0), END, allowed) == float("inf")

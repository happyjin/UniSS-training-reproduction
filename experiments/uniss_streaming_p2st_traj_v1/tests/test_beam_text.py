"""Beam search must be the greedy decoder plus lookahead, not a different one.

The point of running beams is to compare one decoding change against a
baseline.  That comparison is worthless if the beam path also changes the
repetition penalty, the terminator bias or the allowed set, so those are
pinned here against a scripted model whose next token is known by
construction.
"""

from __future__ import annotations

import pytest
import torch

from experiments.uniss_streaming_p2st_pure_ce_v1.runtime.p2st_cascade import (
    _generate,
    _generate_text,
)
from experiments.uniss_streaming_p2st_traj_v1.runtime.beam_text import beam_generate

VOCAB = 8
HIDDEN = 4
TERMINATOR = 7


class _Scripted(torch.nn.Module):
    """Emits a fixed logit row per generated position."""

    def __init__(self, rows: list[list[float]]) -> None:
        super().__init__()
        self.rows = rows
        self.step = 0
        self.embed = torch.nn.Embedding(VOCAB, HIDDEN)
        torch.nn.init.zeros_(self.embed.weight)

    def get_input_embeddings(self):
        return self.embed

    def forward(self, *, inputs_embeds, past_key_values=None, use_cache=False):
        # One row per already-generated token.  The greedy path feeds one
        # token at a time behind a KV cache, so the position cannot be read
        # off the input length there; it is tracked instead and re-seeded
        # whenever a full sequence arrives, which is what the cacheless beam
        # path always sends.
        if past_key_values is None:
            self.step = inputs_embeds.shape[1] - 1
        produced = self.step
        self.step += 1
        row = self.rows[min(produced, len(self.rows) - 1)]
        logits = torch.tensor(row, dtype=torch.float32).reshape(1, 1, VOCAB)
        logits = logits.expand(1, inputs_embeds.shape[1], VOCAB).contiguous()

        class _Out:
            pass

        out = _Out()
        out.logits = logits
        # Not None: the greedy loop feeds this back, and a None cache would
        # make every step look like a fresh sequence to the position tracker.
        out.past_key_values = ("cache", produced)
        return out


def _prompt() -> torch.Tensor:
    return torch.zeros(1, HIDDEN)


def test_one_beam_dispatches_to_the_greedy_function():
    rows = [[0.0, 5.0, 0, 0, 0, 0, 0, 1.0], [0, 0, 4.0, 0, 0, 0, 0, 1.0],
            [0, 0, 0, 0, 0, 0, 0, 9.0]]
    model = _Scripted(rows)
    greedy = _generate(model, _prompt(), terminator=TERMINATOR, max_tokens=6)
    viaone = _generate_text(
        model, _prompt(), terminator=TERMINATOR, max_tokens=6,
        num_beams=1, length_penalty=1.0,
    )
    assert viaone == greedy == ([1, 2], True)


def test_beam_generate_refuses_a_single_beam():
    model = _Scripted([[0.0] * VOCAB])
    with pytest.raises(ValueError):
        beam_generate(
            model, _prompt(), terminator=TERMINATOR, max_tokens=4, num_beams=1
        )


def test_beam_reports_an_unterminated_run_like_greedy():
    """The contract that matters: max_tokens is a failure, not a clean stop."""
    rows = [[0, 9.0, 0, 0, 0, 0, 0, -9.0]]
    model = _Scripted(rows)
    tokens, ended = beam_generate(
        model, _prompt(), terminator=TERMINATOR, max_tokens=3, num_beams=2
    )
    assert ended is False
    assert len(tokens) == 3


def test_beam_can_prefer_one_more_word_over_stopping_now():
    """The reason for doing this at all.

    Step 0 slightly favours the terminator, so greedy stops with nothing.  A
    beam that keeps the alternative alive sees that continuing scores better
    per token once the second step is near-certain, which is exactly the
    "stop now versus say one more word" comparison greedy cannot make.
    """
    rows = [
        [0, 2.0, 0, 0, 0, 0, 0, 2.1],   # terminator marginally ahead
        [0, 0, 20.0, 0, 0, 0, 0, -20.0],  # continuation then near-certain word
        [0, 0, 0, 0, 0, 0, 0, 20.0],
    ]
    model = _Scripted(rows)
    assert _generate(model, _prompt(), terminator=TERMINATOR, max_tokens=5) == ([], True)
    tokens, ended = beam_generate(
        model, _prompt(), terminator=TERMINATOR, max_tokens=5, num_beams=3
    )
    assert ended is True
    assert tokens == [1, 2]


def test_allowed_set_is_respected():
    rows = [[0, 9.0, 1.0, 0, 0, 0, 0, 0.5], [0, 0, 0, 0, 0, 0, 0, 9.0]]
    model = _Scripted(rows)
    allowed = torch.tensor([2, TERMINATOR])
    tokens, ended = beam_generate(
        model, _prompt(), terminator=TERMINATOR, max_tokens=4,
        num_beams=2, allowed=allowed,
    )
    assert 1 not in tokens


def test_terminator_bias_reaches_the_beam():
    """A large negative bias must suppress the stop the same way it does greedy."""
    rows = [[0, 1.0, 0, 0, 0, 0, 0, 5.0], [0, 0, 1.0, 0, 0, 0, 0, 5.0],
            [0, 0, 0, 0, 0, 0, 0, 5.0]]
    model = _Scripted(rows)
    plain, _ = beam_generate(
        model, _prompt(), terminator=TERMINATOR, max_tokens=4, num_beams=2
    )
    biased, _ = beam_generate(
        model, _prompt(), terminator=TERMINATOR, max_tokens=4, num_beams=2,
        terminator_bias_fn=lambda n: -50.0 if n < 2 else 0.0,
    )
    assert len(biased) > len(plain)


def test_repetition_penalty_reaches_the_beam():
    rows = [[0, 5.0, 4.9, 0, 0, 0, 0, -5.0], [0, 5.0, 4.9, 0, 0, 0, 0, -5.0],
            [0, 0, 0, 0, 0, 0, 0, 9.0]]
    model = _Scripted(rows)
    plain, _ = beam_generate(
        model, _prompt(), terminator=TERMINATOR, max_tokens=3, num_beams=2
    )
    penalised, _ = beam_generate(
        model, _prompt(), terminator=TERMINATOR, max_tokens=3, num_beams=2,
        penalty=2.0, penalty_window=4,
    )
    assert plain[:2] == [1, 1]
    assert penalised[:2] != [1, 1]

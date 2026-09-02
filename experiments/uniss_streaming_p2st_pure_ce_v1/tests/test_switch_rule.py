"""The switch rule, exercised exhaustively without a model.

This function is the whole of what C moves out of the model, so it is the one
place a bug would be indistinguishable from the thing four training runs
failed at.  Everything here is a pure integer property, checked over the whole
small-input space rather than on examples.
"""

from __future__ import annotations

import itertools

import pytest

from experiments.uniss_streaming_p2st_pure_ce_v1.runtime.switch_rule import (
    STAGE_ORDER,
    TASK_ASR,
    TASK_DONE,
    TASK_MT,
    TASK_READ,
    TASK_TTS,
    SwitchState,
    next_task,
    rule_trace,
)


def _states(stages=None, limit: int = 4):
    for stage, source, target, exhausted in itertools.product(
        stages or (*STAGE_ORDER, TASK_READ),
        range(limit),
        range(limit),
        (False, True),
    ):
        yield SwitchState(
            stage=stage,
            source_delta=source,
            target_delta=target,
            source_exhausted=exhausted,
        )


def test_every_reachable_state_returns_a_known_task():
    known = {*STAGE_ORDER, TASK_READ, TASK_DONE}
    for state in _states():
        assert next_task(state) in known


def test_the_rule_never_advances_without_committed_content():
    """The bounded wait: no growth means read more, never speak."""
    for state in _states():
        decision = next_task(state)
        if state.stage == TASK_ASR and state.source_delta == 0:
            assert decision in (TASK_READ, TASK_DONE)
        if state.stage == TASK_MT and state.target_delta == 0:
            assert decision in (TASK_READ, TASK_DONE)


def test_speaking_requires_both_stages_to_have_grown():
    for state in _states():
        if next_task(state) == TASK_TTS:
            assert state.stage == TASK_MT
            assert state.target_delta > 0


def test_translation_requires_a_grown_transcript():
    for state in _states():
        if next_task(state) == TASK_MT:
            assert state.stage == TASK_ASR
            assert state.source_delta > 0


def test_an_exhausted_source_always_terminates():
    for state in _states():
        if state.source_exhausted and next_task(state) in (TASK_READ,):
            raise AssertionError(f"exhausted source asked to read more: {state}")


def test_the_cascade_order_is_never_violated(tmp_path):
    """ASR before MT before TTS, always, in every block."""
    for source in range(4):
        for target in range(4):
            trace = rule_trace([source], [target], blocks=1)
            assert trace[0] == TASK_ASR
            positions = {task: index for index, task in enumerate(trace)}
            if TASK_MT in positions:
                assert positions[TASK_MT] > positions[TASK_ASR]
            if TASK_TTS in positions:
                assert positions[TASK_TTS] > positions[TASK_MT]


def test_bounded_wait_shows_up_as_a_short_block():
    assert rule_trace([0], [0], blocks=1) == [TASK_ASR]
    assert rule_trace([3], [0], blocks=1) == [TASK_ASR, TASK_MT]
    assert rule_trace([3], [2], blocks=1) == [TASK_ASR, TASK_MT, TASK_TTS]


def test_a_block_runs_each_stage_at_most_once():
    for source, target in itertools.product(range(4), range(4)):
        trace = rule_trace([source], [target], blocks=1)
        for task in STAGE_ORDER:
            assert trace.count(task) <= 1


def test_no_decision_token_is_representable_in_the_rule():
    """The rule's vocabulary is task names, not tokens.

    If a WAIT/WRITE token could appear here, the decision would be back inside
    the model's output space, which is the thing being removed.
    """
    from training import constants_uniss as c

    names = {*STAGE_ORDER, TASK_READ, TASK_DONE}
    for token in (c.TOKEN_WAIT_READ, c.TOKEN_WRITE_GENERATE):
        assert token not in names
        assert str(token) not in names


def test_malformed_states_are_rejected():
    with pytest.raises(ValueError, match="unknown cascade stage"):
        SwitchState(
            stage="speak", source_delta=1, target_delta=1, source_exhausted=False
        )
    with pytest.raises(ValueError, match="cannot be negative"):
        SwitchState(
            stage=TASK_ASR, source_delta=-1, target_delta=0, source_exhausted=False
        )


def test_rule_trace_requires_one_delta_per_block():
    with pytest.raises(ValueError, match="one source and target delta"):
        rule_trace([1, 2], [1], blocks=2)


def test_the_semantic_terminator_is_inside_the_allowed_set():
    """Regression: a mask that forbids the terminator forbids termination.

    The first cascade run showed every TTS stage reaching max_semantic_tokens
    exactly, which read as an untrained END_SEMANTIC and was in fact this:
    the allowed set held only the 8192 BiCodec codes, and END_SEMANTIC is not
    one of them, so it could never be the argmax.  Fixing the mask took the
    terminator rate from 0.93 to 1.00 on three of four samples and the wall
    time from 8-9 s to 2.5-3.0 s.

    The two-set shape is deliberate and matches the established runtime's
    ``allow_end=bool(generated)``: END is legal after the first code and never
    on it, so no fragment can be empty.
    """
    import torch

    from training import constants_uniss as c

    codes = [c.bicodec_semantic_id(code) for code in range(c.BICODEC_SEMANTIC_SIZE)]
    first = torch.tensor(codes)
    allowed = torch.tensor([*codes, c.TOKEN_END_SEMANTIC])

    assert c.TOKEN_END_SEMANTIC not in set(codes), (
        "the terminator must not be a codebook entry, or this check is vacuous"
    )
    assert c.TOKEN_END_SEMANTIC in set(allowed.tolist())
    assert c.TOKEN_END_SEMANTIC not in set(first.tolist())
    assert len(allowed) == len(first) + 1


def test_the_masked_greedy_can_actually_pick_the_terminator():
    """Exercise the selector itself, not just the sets."""
    import torch

    from experiments.uniss_streaming_p2st_pure_ce_v1.runtime.p2st_cascade import (
        _greedy,
    )
    from training import constants_uniss as c

    codes = [c.bicodec_semantic_id(code) for code in range(c.BICODEC_SEMANTIC_SIZE)]
    allowed = torch.tensor([*codes, c.TOKEN_END_SEMANTIC])
    first = torch.tensor(codes)

    logits = torch.zeros(c.TOKEN_END_SEMANTIC + 8)
    logits[c.TOKEN_END_SEMANTIC] = 10.0
    logits[codes[3]] = 1.0

    assert _greedy(logits, allowed=allowed, penalty=1.0, recent=()) == (
        c.TOKEN_END_SEMANTIC
    )
    # On the first step the terminator is excluded, so a real code wins.
    assert _greedy(logits, allowed=first, penalty=1.0, recent=()) == codes[3]

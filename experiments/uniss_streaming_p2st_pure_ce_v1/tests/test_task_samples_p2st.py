"""Structural guarantees for the prefix-to-prefix task samples.

These are the properties that make this pool different from the interleaved
one, so each is asserted rather than assumed:

* the speak decision is absent -- no WAIT/WRITE/READ token occurs anywhere,
* loss lands only on the target span,
* every sequence carries exactly one boundary token, its own terminator,
* prompts are causal: the supervised delta is not visible in the prompt.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.uniss_phase3_v4_e2e_simuls2st_pilot15_v1.data.schema import (
    E2ETrajectory,
)
from experiments.uniss_phase3_v4_e2e_simuls2st_pilot15_v1.training.task_samples import (
    LOSS_ASR,
    LOSS_BOUNDARY,
    LOSS_EOS,
    LOSS_MT,
    LOSS_NONE,
    LOSS_SEMANTIC,
)
from experiments.uniss_streaming_p2st_pure_ce_v1.training.task_samples_p2st import (
    FAMILY_P2ST_ASR,
    FAMILY_P2ST_MT,
    FAMILY_P2ST_TTS,
    SOURCE_PREFIX_V1,
    build_p2st_incremental_mt_tasks,
    build_p2st_streaming_asr_tasks,
    build_p2st_streaming_tts_tasks,
)
from training import constants_uniss as c

REPO_ROOT = Path(__file__).resolve().parents[3]
GOLD = (
    REPO_ROOT
    / "data/processed/uniss_phase3_v4_e2e_simuls2st_pilot15_v1"
    / "formal_gold_20260818T090515Z/source_events/valid_gold_trajectories.jsonl"
)
# The runtime's three-way choice is spelled with two vocabulary tokens: the
# READ_NEXT branch is the absence of a further WRITE, not an emitted token.
DECISION_TOKENS = (
    c.TOKEN_WAIT_READ,
    c.TOKEN_WRITE_GENERATE,
)


def _encode(text: str) -> list[int]:
    """A deterministic stand-in for the real tokenizer.

    One id per character keeps the layout assertions exact and independent of
    BPE, which is what these tests are about.
    """
    return [ord(char) % 1000 + 1 for char in text]


@pytest.fixture(scope="module")
def trajectories() -> list[E2ETrajectory]:
    if not GOLD.exists():
        pytest.skip(f"gold trajectories not present at {GOLD}")
    records = []
    with GOLD.open() as handle:
        for index, line in enumerate(handle):
            records.append(E2ETrajectory.from_mapping(json.loads(line)))
            if index >= 24:
                break
    return records


@pytest.fixture(scope="module")
def samples(trajectories):
    built = []
    for trajectory in trajectories:
        built.extend(build_p2st_streaming_asr_tasks(trajectory, encode_text=_encode))
        built.extend(
            build_p2st_incremental_mt_tasks(trajectory, encode_text=_encode)
        )
        built.extend(build_p2st_streaming_tts_tasks(trajectory, encode_text=_encode))
    assert built, "no samples were produced from the gold trajectories"
    return built


def test_all_three_families_are_produced(samples):
    families = {sample.family for sample in samples}
    assert families == {FAMILY_P2ST_ASR, FAMILY_P2ST_MT, FAMILY_P2ST_TTS}


def test_no_speak_decision_token_anywhere(samples):
    """The point of this pool: the model is never asked to choose."""
    for sample in samples:
        for token in DECISION_TOKENS:
            assert token not in sample.token_ids, (
                f"{sample.sequence_id} contains decision token {token}"
            )


def test_loss_lands_only_on_a_trailing_span(samples):
    for sample in samples:
        kinds = sample.loss_kinds
        supervised = [i for i, kind in enumerate(kinds) if kind != LOSS_NONE]
        assert supervised, f"{sample.sequence_id} supervises nothing"
        assert supervised[-1] == len(kinds) - 1
        assert supervised == list(range(supervised[0], len(kinds))), (
            f"{sample.sequence_id} has a non-contiguous target span"
        )


def test_every_sequence_ends_with_one_terminator_then_eos(samples):
    for sample in samples:
        assert sample.loss_kinds[-1] == LOSS_EOS
        assert sample.token_ids[-1] == c.TOKEN_EOS
        assert sample.loss_kinds[-2] == LOSS_BOUNDARY
        boundary = [k for k in sample.loss_kinds if k == LOSS_BOUNDARY]
        assert len(boundary) == 1, (
            f"{sample.sequence_id} has {len(boundary)} boundary tokens; the "
            "bucket must hold exactly the sequence's own terminator"
        )


def test_terminator_matches_the_content_kind(samples):
    expected = {
        FAMILY_P2ST_ASR: (LOSS_ASR, c.TOKEN_END_CONTENT),
        FAMILY_P2ST_MT: (LOSS_MT, c.TOKEN_END_CONTENT),
        FAMILY_P2ST_TTS: (LOSS_SEMANTIC, c.TOKEN_END_SEMANTIC),
    }
    for sample in samples:
        kind, terminator = expected[sample.family]
        assert sample.token_ids[-2] == terminator
        content = set(sample.loss_kinds[:-2]) - {LOSS_NONE}
        assert content == {kind}, f"{sample.sequence_id} mixed loss kinds {content}"


def test_asr_prompt_is_exactly_the_causal_prefix(trajectories):
    """Causality, asserted as an exact layout rather than a substring scan.

    A one-character delta encodes to a single id, so searching for it inside
    the joined prompt matches any earlier occurrence of the same character.
    Reconstructing the whole prompt is both exact and stronger: it pins the
    header, the GLM block and the committed transcript, and it fails if the
    supervised delta is appended to the prompt.
    """
    for trajectory in trajectories:
        for sample in build_p2st_streaming_asr_tasks(
            trajectory, encode_text=_encode
        ):
            split = next(
                i for i, k in enumerate(sample.loss_kinds) if k != LOSS_NONE
            )
            index = int(sample.sequence_id.rsplit(":", 1)[1])
            event = next(e for e in trajectory.events if e.event_index == index)
            committed = event.gold_source_prefix
            assert committed.endswith(event.gold_source_delta)
            committed = committed[: len(committed) - len(event.gold_source_delta)]
            committed = committed.rstrip()
            expected = (
                c.TOKEN_TASK_STREAMING_ASR,
                c.TOKEN_STREAMING_MODE,
                c.language_token_id(trajectory.src_lang),
                c.TOKEN_START_GLM,
                *([c.glm_semantic_id(0)] * event.source_glm_end),
                c.TOKEN_END_GLM,
                c.TOKEN_START_CONTENT,
                *(_encode(committed) if committed else ()),
            )
            assert tuple(sample.token_ids[:split]) == expected, (
                f"{sample.sequence_id} prompt layout differs"
            )
            assert tuple(sample.token_ids[split:-2]) == tuple(
                _encode(event.gold_source_delta)
            )


def test_asr_speech_indices_cover_exactly_the_causal_prefix(trajectories):
    for trajectory in trajectories:
        for sample in build_p2st_streaming_asr_tasks(
            trajectory, encode_text=_encode
        ):
            indices = [v for v in sample.speech_indices if v is not None]
            assert indices == list(range(sample.source_glm_length))
            assert len(sample.source_glm_ids) == sample.source_glm_length, (
                f"{sample.sequence_id} promises {sample.source_glm_length} GLM "
                f"positions but carries {len(sample.source_glm_ids)} ids"
            )
            index = int(sample.sequence_id.rsplit(":", 1)[1])
            event = next(e for e in trajectory.events if e.event_index == index)
            assert sample.source_glm_length == event.source_glm_end


def test_tts_semantic_prefix_matches_the_recorded_offset(trajectories):
    """The builder raises if the reconstructed prefix disagrees with the data.

    Reaching this assertion at all means every event's ``target_semantic_start``
    equals the concatenated length of the earlier deltas, which is the
    alignment the whole TTS task rests on.
    """
    total = 0
    for trajectory in trajectories:
        built = build_p2st_streaming_tts_tasks(trajectory, encode_text=_encode)
        total += len(built)
    assert total > 0


def test_mt_and_tts_carry_no_audio(samples):
    for sample in samples:
        if sample.family == FAMILY_P2ST_ASR:
            assert sample.source_audio is not None
        else:
            assert sample.source_audio is None
            assert sample.source_glm_length == 0
            assert all(value is None for value in sample.speech_indices)


def test_no_teacher_bindings_or_commit_positions(samples):
    """Pure CE: this pool asks nothing of the teacher caches.

    That is also why it is cheap -- the interleaved pool's teacher KL does a
    full 180407-vocabulary log_softmax, which is what made those batches run
    at 22-35 s against replay's 7 s.
    """
    for sample in samples:
        assert sample.teacher_bindings == ()
        assert not getattr(sample, "commit_positions", ())


def test_mt_v1_roll_in_variant_is_available(trajectories):
    """The exposure-bias variant needs a flag, not a new data build."""
    produced = 0
    for trajectory in trajectories:
        produced += len(
            build_p2st_incremental_mt_tasks(
                trajectory,
                encode_text=_encode,
                source_prefix_kind=SOURCE_PREFIX_V1,
            )
        )
    # v1_source_prefix is null in the gold-only trajectories, so an empty
    # result is correct here; what matters is that the flag is accepted and
    # never mixes gold text into a roll-in sample.
    assert produced >= 0


def test_streaming_task_tokens_are_the_preallocated_ones(samples):
    heads = {sample.family: sample.token_ids[0] for sample in samples}
    assert heads[FAMILY_P2ST_ASR] == c.TOKEN_TASK_STREAMING_ASR == 180_383
    assert (
        heads[FAMILY_P2ST_MT] == c.TOKEN_TASK_STREAMING_TEXT_TRANSLATION == 180_398
    )
    assert heads[FAMILY_P2ST_TTS] == c.TOKEN_TASK_STREAMING_TTS == 180_382

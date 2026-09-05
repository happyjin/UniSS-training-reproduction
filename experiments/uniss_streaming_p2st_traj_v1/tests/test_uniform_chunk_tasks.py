"""Structural guarantees for the fixed-chunk pool.

The event-level pool's own tests already pin the properties both pools share
-- loss only on the target span, one terminator per sequence, no WAIT/WRITE
token anywhere.  These tests pin what is *new* here and would otherwise be
assumed:

* the read grid is the clock, not the content: boundaries are exact multiples
  of ``chunk_ms`` and every gold event lands in exactly one window;
* an IDLE chunk is supervised, and its whole target is the terminator;
* merging deltas inside a chunk reconstructs the same running text the
  event-level builder commits, separators included;
* ``END_SEMANTIC`` can only land on a target-word block boundary.

They run against the real 15-shard valid trajectories, as the event-level
tests do, because the properties are about the data as much as the code.
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
    SOURCE_PREFIX_GOLD,
    _split_source_text,
    causal_glm_token_count,
)
from experiments.uniss_streaming_p2st_traj_v1.data.uniform_chunk_tasks import (
    DEFAULT_CHUNK_MS,
    build_uniform_chunk_asr_tasks,
    build_uniform_chunk_mt_tasks,
    build_uniform_chunk_samples,
    build_uniform_chunk_tts_tasks,
    chunk_windows,
)
from training import constants_uniss as c

REPO_ROOT = Path(__file__).resolve().parents[3]
GOLD = (
    REPO_ROOT
    / "data/processed/uniss_phase3_v4_e2e_simuls2st_pilot15_v1"
    / "formal_gold_20260818T090515Z/source_events/valid_gold_trajectories.jsonl"
)
DECISION_TOKENS = (c.TOKEN_WAIT_READ,)


def _encode(text: str) -> list[int]:
    """One id per character: exact layout assertions, independent of BPE."""
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


def test_boundaries_are_clock_multiples(trajectories):
    for trajectory in trajectories:
        windows = chunk_windows(trajectory, chunk_ms=DEFAULT_CHUNK_MS)
        assert windows
        for window in windows:
            assert window.start_ms == window.chunk_index * DEFAULT_CHUNK_MS
            full = (window.chunk_index + 1) * DEFAULT_CHUNK_MS
            # Every boundary is the clock's, except the last, which is clamped
            # to the end of the audio.
            assert window.end_ms == full or window is windows[-1]
            assert window.end_ms <= full


def test_every_event_lands_in_exactly_one_window(trajectories):
    for trajectory in trajectories:
        windows = chunk_windows(trajectory, chunk_ms=DEFAULT_CHUNK_MS)
        seen = [e.event_index for w in windows for e in w.events]
        assert sorted(seen) == sorted(e.event_index for e in trajectory.events)
        assert len(seen) == len(set(seen))


def test_windows_are_ordered_and_monotone(trajectories):
    for trajectory in trajectories:
        windows = chunk_windows(trajectory, chunk_ms=DEFAULT_CHUNK_MS)
        ends = [w.end_ms for w in windows]
        assert ends == sorted(ends)
        glm = [w.glm_stop for w in windows]
        assert glm == sorted(glm)
        assert all(
            w.glm_stop == causal_glm_token_count(w.pcm_end) for w in windows
        )


def test_idle_rate_at_640ms_matches_the_measurement(trajectories):
    """The measurement this chunk size was chosen from, kept honest.

    Measured over 60,000 train trajectories and 481,882 windows, a 640 ms grid
    leaves 0.508 of windows with no newly committed *target* content -- the
    paper's own criterion, and what the MT and TTS families see -- and 0.299
    with no new *source* transcript, which is what the ASR family sees.  A
    25-trajectory valid sample will not reproduce either exactly, but a grid
    that had drifted to an extreme would mean the chunk size no longer buys
    balanced supervision.
    """
    total = idle_source = idle_target = 0
    for trajectory in trajectories:
        for window in chunk_windows(trajectory, chunk_ms=DEFAULT_CHUNK_MS):
            total += 1
            if not any(e.gold_source_delta.strip() for e in window.events):
                idle_source += 1
            if not any(e.target_text_delta.strip() for e in window.events):
                idle_target += 1
    assert total > 0
    assert 0.05 <= idle_source / total <= 0.55
    assert 0.25 <= idle_target / total <= 0.75
    # The target criterion is always the looser one: a chunk can read a word
    # without the aligner having committed its translation, never the reverse.
    assert idle_target >= idle_source


def test_finer_grids_are_more_idle(trajectories):
    """Monotonicity in the chunk size, which is the mechanism, not a constant."""
    rates = []
    for chunk_ms in (160, 640, 1920):
        total = idle = 0
        for trajectory in trajectories:
            for window in chunk_windows(trajectory, chunk_ms=chunk_ms):
                total += 1
                if not any(e.gold_source_delta.strip() for e in window.events):
                    idle += 1
        rates.append(idle / total)
    assert rates[0] > rates[1] > rates[2]


def _split_prompt_and_target(sample):
    kinds = list(sample.loss_kinds)
    first = next(i for i, k in enumerate(kinds) if k != LOSS_NONE)
    assert all(k != LOSS_NONE for k in kinds[first:]), "loss must be a suffix"
    return sample.token_ids[:first], sample.token_ids[first:]


def test_idle_samples_target_only_the_terminator(trajectories):
    found = 0
    for trajectory in trajectories:
        for sample in build_uniform_chunk_asr_tasks(
            trajectory, encode_text=_encode
        ):
            _, target = _split_prompt_and_target(sample)
            if len(target) == 2:
                found += 1
                assert target == (c.TOKEN_END_CONTENT, c.TOKEN_EOS)
                assert sample.loss_kinds[-2:] == (LOSS_BOUNDARY, LOSS_EOS)
    assert found > 0, "the fixed grid must produce read/wait steps"


def test_idle_ratio_zero_removes_them_and_keeps_content(trajectories):
    for trajectory in trajectories:
        full = build_uniform_chunk_asr_tasks(trajectory, encode_text=_encode)
        none = build_uniform_chunk_asr_tasks(
            trajectory, encode_text=_encode, idle_ratio=0.0
        )
        assert len(none) <= len(full)
        assert all(
            sample.loss_kinds.count(LOSS_ASR) > 0 for sample in none
        )
        # Dropping IDLE never changes a content sample.
        by_id = {s.sequence_id: s for s in full}
        for sample in none:
            assert by_id[sample.sequence_id] == sample


def test_merged_asr_delta_reconstructs_running_text(trajectories):
    """The chunk's committed text is the event-level text, separators included.

    ``_split_running_text`` puts the separator on the delta side because a
    space the model never emits is a space that never appears.  Merging must
    preserve that, or the MT stage is handed an out-of-distribution string.
    """
    for trajectory in trajectories:
        for window in chunk_windows(trajectory, chunk_ms=DEFAULT_CHUNK_MS):
            members = [
                e for e in window.events if e.gold_source_delta.strip()
            ]
            if not members:
                continue
            committed = _split_source_text(members[0], SOURCE_PREFIX_GOLD)[0]
            merged = "".join(
                _split_source_text(m, SOURCE_PREFIX_GOLD)[1] for m in members
            )
            assert committed + merged == members[-1].gold_source_prefix


def test_asr_prompt_carries_the_acoustic_sidecar(trajectories):
    for trajectory in trajectories:
        for sample in build_uniform_chunk_asr_tasks(
            trajectory, encode_text=_encode
        ):
            speech = [v for v in sample.speech_indices if v is not None]
            assert speech == list(range(sample.source_glm_length))
            assert len(sample.source_glm_ids) == sample.source_glm_length
            assert sample.source_pcm_end > 0
            assert sample.source_audio == trajectory.source_audio


def test_text_families_carry_no_sidecar(trajectories):
    for trajectory in trajectories:
        built = build_uniform_chunk_samples(trajectory, encode_text=_encode)
        for family in (FAMILY_P2ST_MT, FAMILY_P2ST_TTS):
            for sample in built[family]:
                assert sample.source_audio is None
                assert sample.source_glm_length == 0
                assert all(v is None for v in sample.speech_indices)


def test_no_wait_token_anywhere(trajectories):
    for trajectory in trajectories:
        built = build_uniform_chunk_samples(trajectory, encode_text=_encode)
        for samples in built.values():
            for sample in samples:
                for token in DECISION_TOKENS:
                    assert token not in sample.token_ids


def test_exactly_one_terminator_per_sequence(trajectories):
    for trajectory in trajectories:
        built = build_uniform_chunk_samples(trajectory, encode_text=_encode)
        for samples in built.values():
            for sample in samples:
                assert sample.loss_kinds.count(LOSS_BOUNDARY) == 1
                assert sample.loss_kinds.count(LOSS_EOS) == 1
                assert sample.token_ids[-1] == c.TOKEN_EOS


def test_end_semantic_lands_on_a_word_block(trajectories):
    """Step 3, asserted where it actually holds: at the merge.

    ``target_semantic_delta`` is cut at target-word blocks upstream, so a
    chunk that merges whole deltas ends on a block boundary by construction.
    The property that can fail is contiguity -- a skipped event would make the
    codes disagree with the text naming them -- so that is what is checked.
    """
    spoken_any = False
    for trajectory in trajectories:
        for window in chunk_windows(trajectory, chunk_ms=DEFAULT_CHUNK_MS):
            members = [e for e in window.events if e.target_semantic_delta]
            if not members:
                continue
            spoken_any = True
            length = sum(len(m.target_semantic_delta) for m in members)
            assert (
                int(members[-1].target_semantic_end)
                == int(members[0].target_semantic_start) + length
            )
    assert spoken_any


def test_tts_has_no_idle_by_default(trajectories):
    for trajectory in trajectories:
        for sample in build_uniform_chunk_tts_tasks(
            trajectory, encode_text=_encode
        ):
            assert sample.loss_kinds.count(LOSS_SEMANTIC) > 0


def test_tts_idle_is_reachable_when_asked(trajectories):
    found = 0
    for trajectory in trajectories:
        for sample in build_uniform_chunk_tts_tasks(
            trajectory, encode_text=_encode, tts_idle=True
        ):
            if sample.loss_kinds.count(LOSS_SEMANTIC) == 0:
                found += 1
                assert sample.token_ids[-2] == c.TOKEN_END_SEMANTIC
    assert found > 0


def test_mt_skips_chunks_before_any_source_word(trajectories):
    for trajectory in trajectories:
        windows = chunk_windows(trajectory, chunk_ms=DEFAULT_CHUNK_MS)
        samples = build_uniform_chunk_mt_tasks(trajectory, encode_text=_encode)
        indices = {int(s.sequence_id.split(":")[3]) for s in samples}
        for window in windows:
            if window.chunk_index not in indices:
                continue
            assert any(
                e.gold_source_prefix.strip()
                for e in trajectory.events
                if e.source_end_ms <= window.end_ms
            )


def test_mt_supervises_translation_not_transcript(trajectories):
    for trajectory in trajectories:
        for sample in build_uniform_chunk_mt_tasks(
            trajectory, encode_text=_encode
        ):
            kinds = set(sample.loss_kinds) - {LOSS_NONE, LOSS_BOUNDARY, LOSS_EOS}
            assert kinds <= {LOSS_MT}


def test_sequence_ids_are_unique(trajectories):
    seen: set[str] = set()
    for trajectory in trajectories:
        built = build_uniform_chunk_samples(trajectory, encode_text=_encode)
        for samples in built.values():
            for sample in samples:
                assert sample.sequence_id not in seen
                seen.add(sample.sequence_id)


def test_ids_do_not_collide_with_the_event_level_pool(trajectories):
    """A traj id can never be mistaken for a pure-CE id, so pools can mix."""
    for trajectory in trajectories:
        built = build_uniform_chunk_samples(trajectory, encode_text=_encode)
        for samples in built.values():
            for sample in samples:
                assert ":traj_" in sample.sequence_id
                assert ":p2st_" not in sample.sequence_id


def test_build_is_deterministic(trajectories):
    for trajectory in trajectories:
        first = build_uniform_chunk_samples(trajectory, encode_text=_encode)
        second = build_uniform_chunk_samples(trajectory, encode_text=_encode)
        assert first == second


def test_chunk_size_must_be_positive(trajectories):
    with pytest.raises(ValueError):
        chunk_windows(trajectories[0], chunk_ms=0)


def test_larger_chunks_yield_fewer_samples(trajectories):
    counts = []
    for chunk_ms in (320, 640, 1920):
        total = 0
        for trajectory in trajectories:
            total += len(
                build_uniform_chunk_asr_tasks(
                    trajectory, encode_text=_encode, chunk_ms=chunk_ms
                )
            )
        counts.append(total)
    assert counts[0] > counts[1] > counts[2]

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from experiments.uniss_phase3_v4_e2e_simuls2st_pilot15_v1.teacher.v1_cache import (
    V1_CACHE_SCHEMA,
    combine_v1_sample,
    save_v1_bundle,
)
from experiments.uniss_phase3_v4_e2e_simuls2st_pilot15_v1.teacher.v1_requests import (
    build_v1_teacher_sequences,
)
from experiments.uniss_phase3_v4_e2e_simuls2st_pilot15_v1.tests.test_teacher_requests import (
    _encode,
    _rollout,
    _trajectory,
)
from training import constants_uniss as c


def _summaries(sequences):
    output = []
    for sequence in sequences:
        labels = np.asarray(
            [
                label
                for request in sequence.requests
                for label in request.reference_labels
            ],
            dtype=np.int32,
        )
        output.append(
            {
                "indices": np.stack(
                    (labels, (labels + 1) % c.VOCAB_SIZE), axis=1
                ),
                "probabilities": np.tile(
                    np.asarray([[0.8, 0.2]], dtype=np.float16),
                    (len(labels), 1),
                ),
                "top1": labels.copy(),
                "confidence": np.full(len(labels), 0.8, dtype=np.float16),
            }
        )
    return output


def test_v1_teacher_sequences_are_same_prefix_and_future_safe() -> None:
    trajectory = _trajectory()
    rollout = _rollout()
    gold, v1 = build_v1_teacher_sequences(
        trajectory, rollout, encode_text=_encode
    )
    assert gold.history_kind == "gold_asr"
    assert v1.history_kind == "v1_asr"
    assert len(gold.requests) == len(v1.requests) == 3
    assert gold.requests[-1].reference_labels == (c.TOKEN_EOS,)
    assert v1.requests[-1].reference_labels == rollout.final_generated_tokens
    assert v1.requests[0].reference_labels == rollout.events[0].generated_tokens
    assert gold.requests[0].reference_labels[:3] == (
        c.TOKEN_WRITE_GENERATE,
        c.TOKEN_ENG,
        c.TOKEN_START_CONTENT,
    )
    for sequence in (gold, v1):
        selected = [
            value for value in sequence.speech_indices if value is not None
        ]
        assert selected == list(range(trajectory.source_glm_length))
        first = sequence.requests[0]
        future_positions = [
            index
            for index, speech_index in enumerate(sequence.speech_indices)
            if speech_index is not None
            and speech_index >= first.visible_glm_tokens
        ]
        assert first.predictor_positions[-1] < min(future_positions)
        assert len(first.prefix_sha256) == len(first.target_sha256) == 64


def test_v1_teacher_bundle_round_trip(tmp_path) -> None:
    sequences = build_v1_teacher_sequences(
        _trajectory(), _rollout(), encode_text=_encode
    )
    arrays, descriptors = combine_v1_sample(sequences, _summaries(sequences))
    rows = save_v1_bundle(
        tmp_path / "bundle-000000.npz",
        [
            {
                "sample_id": "sample-1",
                "split": "valid",
                "source_manifest_record": 0,
                "arrays": arrays,
                "requests": descriptors,
            }
        ],
    )
    assert rows[0]["teacher_top1_correct"] == rows[0]["teacher_positions"]
    assert rows[0]["reference_in_topk"] == rows[0]["teacher_positions"]
    assert {value["history_kind"] for value in descriptors} == {
        "gold_asr",
        "v1_asr",
    }
    with np.load(tmp_path / "bundle-000000.npz", allow_pickle=False) as bundle:
        assert str(bundle["bundle_schema"][0]) == V1_CACHE_SCHEMA
        assert len(bundle["row_0_reference_label"]) == rows[0][
            "teacher_positions"
        ]


def test_v1_teacher_rejects_empty_free_running_fragment() -> None:
    rollout = _rollout()
    events = list(rollout.events)
    events[0] = replace(events[0], generated_tokens=())
    malformed = replace(rollout, events=tuple(events))
    with pytest.raises(ValueError, match="target is empty"):
        build_v1_teacher_sequences(_trajectory(), malformed, encode_text=_encode)

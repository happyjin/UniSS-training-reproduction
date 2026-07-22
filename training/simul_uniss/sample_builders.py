"""Build weighted interleaved language-model samples for Simul-UniSS."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from training import constants_uniss as c
from training.simul_uniss import SAMPLE_SCHEMA_VERSION


ACTION_WEIGHT = 4.0
TEXT_WEIGHT = 2.0
SEMANTIC_WEIGHT = 1.0
OBSERVED_WEIGHT = 0.0


@dataclass(frozen=True)
class WeightedSample:
    sample_id: str
    input_ids: list[int]
    token_weights: list[float]
    task: str = "simul_s2st"

    def __post_init__(self) -> None:
        if len(self.input_ids) != len(self.token_weights):
            raise ValueError("input_ids and token_weights must have the same length")
        if len(self.input_ids) < 2:
            raise ValueError("weighted sample must contain at least two tokens")

    def to_json(self) -> dict[str, object]:
        return {
            "schema_version": SAMPLE_SCHEMA_VERSION,
            "id": self.sample_id,
            "task": self.task,
            "input_ids": self.input_ids,
            "token_weights": self.token_weights,
            "length": len(self.input_ids),
        }


def _extend(
    input_ids: list[int],
    weights: list[float],
    tokens: Sequence[int],
    weight: float,
) -> None:
    input_ids.extend(tokens)
    weights.extend([float(weight)] * len(tokens))


def build_interleaved_sample(schedule: Mapping[str, object], speed: float = 1.0) -> WeightedSample:
    tgt_lang = str(schedule["tgt_lang"])
    input_ids: list[int] = []
    weights: list[float] = []

    header = [
        c.TOKEN_TASK_STREAMING_S2ST,
        c.TOKEN_STREAMING_MODE,
        c.TOKEN_DYNAMIC_MODE,
        c.language_token_id(tgt_lang),
        c.speed_token_id(speed),
        *c.wrap_global_tokens(schedule["speaker_tokens"]),  # type: ignore[arg-type]
    ]
    _extend(input_ids, weights, header, OBSERVED_WEIGHT)

    events = schedule["events"]
    if not isinstance(events, list) or not events:
        raise ValueError("schedule events must be a non-empty list")
    for event in events:
        if not isinstance(event, Mapping):
            raise TypeError("schedule event must be a mapping")
        source_chunk = c.encode_glm_semantic(event["source_glm"])  # type: ignore[arg-type]
        _extend(
            input_ids,
            weights,
            [c.TOKEN_START_GLM, *source_chunk, c.TOKEN_END_GLM],
            OBSERVED_WEIGHT,
        )
        action = str(event["action"])
        if action == "wait":
            if bool(event.get("source_is_final", False)):
                raise ValueError("final source event cannot be WAIT")
            _extend(input_ids, weights, [c.TOKEN_WAIT_READ], ACTION_WEIGHT)
            continue
        if action != "write":
            raise ValueError(f"unsupported action: {action}")

        _extend(input_ids, weights, [c.TOKEN_WRITE_GENERATE], ACTION_WEIGHT)
        _extend(
            input_ids,
            weights,
            [c.language_token_id(tgt_lang), c.TOKEN_START_CONTENT],
            TEXT_WEIGHT,
        )
        target_text_ids = list(event["target_text_ids"])  # type: ignore[arg-type]
        _extend(input_ids, weights, target_text_ids, TEXT_WEIGHT)
        _extend(input_ids, weights, [c.TOKEN_END_CONTENT], TEXT_WEIGHT)
        _extend(input_ids, weights, [c.TOKEN_START_SEMANTIC], TEXT_WEIGHT)
        target_semantic = c.encode_bicodec_semantic(event["target_semantic"])  # type: ignore[arg-type]
        _extend(input_ids, weights, target_semantic, SEMANTIC_WEIGHT)
        _extend(input_ids, weights, [c.TOKEN_END_SEMANTIC], TEXT_WEIGHT)

    _extend(input_ids, weights, [c.TOKEN_EOS], TEXT_WEIGHT)
    for token_id in input_ids:
        c.validate_token_id(token_id)
    return WeightedSample(str(schedule["id"]), input_ids, weights)

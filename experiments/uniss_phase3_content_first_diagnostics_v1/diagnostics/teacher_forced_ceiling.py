#!/usr/bin/env python3
"""Experiment 0-C: what is the content-first checkpoint's ceiling?

0-A established that the block-causal codes the rollout cascade feeds the model
agree with the offline codes it trained on only 14.5% of the time.  This module
answers the follow-up question the next training run depends on: is the 3%
target coverage a capability gap, or is it caused by that input mismatch?

It runs the *established* cascade (``evaluate_event_policy_session``) unchanged
and ablates exactly one thing -- where the discrete source codes come from:

* ``causal``       the shipped inference path (``objective._nearest_codes``)
* ``gold_offline`` the non-causal GLM4 codes, i.e. the exact ``source_glm`` the
  content-first SFT consumed during training

0-A verified per-component length parity between the two streams (8/8 exact),
so the gold stream can be substituted position-for-position without any
re-alignment.  Sessions are single components rather than long episodes so the
substitution stays exact and needs no gap accounting.

Nothing outside this experiment is modified: the code source is swapped by
rebinding ``objective._nearest_codes`` on the loaded objective instance.
"""

from __future__ import annotations

import argparse
import json
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
import torch
from transformers import WhisperFeatureExtractor

from experiments.uniss_phase3_content_first_diagnostics_v1.diagnostics.bridge_parity import (
    offline_codes,
    read_waveform,
    unique_components,
)
from experiments.uniss_phase3_content_first_joint_s2st_v1.runtime import (
    model_loader as content_first_loader,
)
from experiments.uniss_phase3_content_first_joint_s2st_v1.runtime.model_loader import (
    load_content_first_models,
)
from experiments.uniss_phase3_v4_quality_first_true_streaming_pilot15_v1.stage_a_causal_whisper_asr.training.frontend import (
    TrainableSharedCausalWhisperVQ,
)
from experiments.uniss_phasea_coverage_constrained_grpo_v3.runtime.event_policy_cascade import (
    evaluate_event_policy_session,
)
from experiments.uniss_phasea_coverage_constrained_grpo_v3.training.coverage import (
    audit_episode,
)
from experiments.uniss_phasea_coverage_constrained_grpo_v3.training.trace_generator import (
    EventTraceGenerator,
)
from experiments.uniss_phasea_stateful_longepisode_rl_v1.training.rollout import (
    episode_observation,
)


SCHEMA = "uniss_content_first_teacher_forced_ceiling_v1"
ARMS = ("causal", "gold_offline")


class GoldCodeServer:
    """Serve the gold offline codes position-for-position, once per block.

    The cascade consumes the same ``pre_vq_hidden`` block twice: once through
    ``objective._nearest_codes`` for the GLM semantic embedding ids, and once
    inside ``_ContentFirstBridgeNorm.forward``, which resolves the *module
    level* ``_nearest_codes`` at call time for the residual adapter.  Both must
    receive the same substituted codes, and the cursor must advance only once
    per block, so requests are memoized on the identity of the hidden tensor.
    A reference to each memoized tensor is retained so its ``id`` cannot be
    reused by a later block.
    """

    def __init__(self, codes: Sequence[int], fallback) -> None:
        self.codes = [int(value) for value in codes]
        self.fallback = fallback
        self.cursor = 0
        self.blocks = 0
        self.repeats = 0
        self.exhausted = 0
        self._memo: dict[int, tuple[torch.Tensor, list[int]]] = {}
        self._order: list[int] = []

    def _take(self, wanted: int) -> list[int]:
        start = self.cursor
        stop = min(len(self.codes), start + wanted)
        taken = self.codes[start:stop]
        self.cursor = start + wanted
        return taken

    def serve(self, hidden: torch.Tensor) -> torch.Tensor:
        key = id(hidden)
        cached = self._memo.get(key)
        if cached is not None:
            self.repeats += 1
            taken = cached[1]
        else:
            wanted = int(hidden.shape[0])
            taken = self._take(wanted)
            if len(taken) < wanted:
                # Only reachable on a length disagreement with the gold stream.
                # Fall back to the real causal codes so the session completes
                # and the shortfall is reported instead of silently padded.
                self.exhausted += wanted - len(taken)
                causal = self.fallback(hidden)
                taken = [
                    *taken,
                    *(int(value) for value in causal[len(taken) :].tolist()),
                ]
            self.blocks += 1
            self._memo[key] = (hidden, taken)
            self._order.append(key)
            while len(self._order) > 8:
                self._memo.pop(self._order.pop(0), None)
        return torch.tensor(taken, dtype=torch.long, device=hidden.device)

    def stats(self) -> dict[str, int]:
        return {
            "gold_codes": len(self.codes),
            "blocks": self.blocks,
            "consumed": min(self.cursor, len(self.codes)),
            "memoized_repeats": self.repeats,
            "exhausted": self.exhausted,
        }


@contextmanager
def gold_code_source(objective, codes: Sequence[int]) -> Iterator[GoldCodeServer]:
    """Replace both code consumers for the duration of one session."""

    had_instance_attribute = "_nearest_codes" in vars(objective)
    previous_instance = vars(objective).get("_nearest_codes")
    previous_module = content_first_loader._nearest_codes
    # Capture the unpatched causal quantizer before rebinding, so the shortfall
    # fallback cannot recurse into the substitution.
    server = GoldCodeServer(
        codes, lambda hidden: previous_module(objective, hidden)
    )
    objective._nearest_codes = server.serve
    content_first_loader._nearest_codes = lambda _objective, hidden: server.serve(hidden)
    try:
        yield server
    finally:
        content_first_loader._nearest_codes = previous_module
        if had_instance_attribute:
            objective._nearest_codes = previous_instance
        else:
            vars(objective).pop("_nearest_codes", None)


def component_row(
    component: dict[str, object],
    fixed_speaker: Sequence[int],
    speaker_global: Sequence[int],
    arm: str,
) -> dict[str, object]:
    return {
        "id": f"{component['sample_id']}_{arm}",
        "src_lang": str(component["src_lang"]),
        "tgt_lang": str(component["tgt_lang"]),
        "source_audio": str(component["source_audio"]),
        "bicodec_global": [int(value) for value in speaker_global],
        "_stage_a_fixed_speaker_global": [int(value) for value in fixed_speaker],
        "teacher_transcription": str(component["transcription"]),
        "teacher_translation": str(component["translation"]),
    }


def summarize(result: dict[str, object], row: dict[str, object]) -> dict[str, object]:
    observation = episode_observation(result, row)
    coverage = audit_episode(
        teacher_translation=str(row["teacher_translation"]),
        generated_translation=str(result["generated_streaming_translation"]),
        target_language=str(row["tgt_lang"]),
        events=result["events"],  # type: ignore[arg-type]
        eos_pending_items=len(result["tts_pending_unspoken_text"]),  # type: ignore[arg-type]
    )
    events = result["events"]  # type: ignore[assignment]
    audio_writes = sum(
        1
        for event in events
        for emission in event["tts_emissions"]
        if bool(emission.get("acknowledged", False))
    )
    return {
        "asr_teacher_similarity": float(observation.asr_teacher_similarity),
        "mt_teacher_similarity": float(observation.mt_teacher_similarity),
        "translation_length_ratio": float(observation.translation_length_ratio),
        "target_coverage": float(coverage.target_coverage),
        "spoken_target_coverage": float(coverage.spoken_target_coverage),
        "first_write_ms": float(observation.first_write_ms),
        "maximum_internal_silence_ms": float(observation.maximum_internal_silence_ms),
        "audio_writes": int(audio_writes),
        "premature_end_count": int(observation.premature_end_count),
        "tts_failure_count": int(observation.tts_failure_count),
        "generated_transcription": str(result["generated_streaming_transcription"]),
        "generated_translation": str(result["generated_streaming_translation"]),
    }


def evaluate(args: argparse.Namespace) -> dict[str, object]:
    device = torch.device(args.device)
    load_args = argparse.Namespace(
        device=str(device),
        base_hf=args.base_hf,
        adapter_checkpoint=args.adapter_checkpoint,
        whispervq_model=args.whispervq_model,
        bicodec_model=args.bicodec_model,
    )
    model, tokenizer, controller, manifest, objective, codec = load_content_first_models(
        load_args
    )
    reference_encoder = (
        TrainableSharedCausalWhisperVQ(args.whispervq_model, gradient_checkpointing=False)
        .to(device)
        .eval()
        .requires_grad_(False)
    )
    extractor = WhisperFeatureExtractor.from_pretrained(
        str(args.whispervq_model), local_files_only=True
    )
    snapshot = json.loads(args.source_snapshot.read_text(encoding="utf-8"))
    fixed_speaker = [
        int(value) for value in snapshot["fixed_system_speaker"]["global_tokens"]
    ]

    components = unique_components(args.episodes, args.components)
    directions = _component_directions(args.episodes)
    rows: list[dict[str, object]] = []
    for index, component in enumerate(components):
        sample_id = str(component["sample_id"])
        component = {**component, **directions[sample_id]}
        waveform = read_waveform(Path(str(component["source_audio"])))
        gold = offline_codes(reference_encoder.encoder, extractor, waveform)
        record: dict[str, object] = {
            "sample_id": sample_id,
            "duration_ms": int(component["duration_ms"]),
            "direction": f"{component['src_lang']}->{component['tgt_lang']}",
            "gold_code_length": len(gold),
            "arms": {},
        }
        for arm in ARMS:
            row = component_row(
                component, fixed_speaker, component["speaker_global"], arm
            )
            tracer = EventTraceGenerator(
                controller,
                policy_temperature=args.policy_temperature,
                policy_top_p=args.policy_top_p,
                action_temperature=args.action_temperature,
            )
            seed = args.seed + index * 100_000 + ARMS.index(arm) * 1_000
            output_dir = args.output_audio / arm / sample_id
            kwargs = dict(
                decision_chunk_ms=args.decision_chunk_ms,
                acoustic_rollover_ms=args.acoustic_rollover_ms,
                model=model,
                tokenizer=tokenizer,
                objective=objective,
                bicodec=codec,
                generate_fn=tracer,
                action_fn=tracer.decide,
                event_context_fn=tracer.set_event,
                output=output_dir,
                seed=seed,
            )
            if arm == "gold_offline":
                with gold_code_source(objective, gold) as server:
                    result = evaluate_event_policy_session(row, **kwargs)
                served = server.stats()
            else:
                result = evaluate_event_policy_session(row, **kwargs)
                served = {
                    "gold_codes": 0,
                    "blocks": 0,
                    "consumed": 0,
                    "memoized_repeats": 0,
                    "exhausted": 0,
                }
            metrics = summarize(result, row)
            metrics["code_substitution"] = served
            record["arms"][arm] = metrics
            print(
                json.dumps(
                    {
                        "sample_id": sample_id,
                        "arm": arm,
                        "asr": round(metrics["asr_teacher_similarity"], 4),
                        "mt": round(metrics["mt_teacher_similarity"], 4),
                        "target_coverage": round(metrics["target_coverage"], 4),
                        "length_ratio": round(metrics["translation_length_ratio"], 4),
                        "audio_writes": metrics["audio_writes"],
                        "exhausted_codes": served["exhausted"],
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
        rows.append(record)

    keys = (
        "asr_teacher_similarity",
        "mt_teacher_similarity",
        "translation_length_ratio",
        "target_coverage",
        "spoken_target_coverage",
        "first_write_ms",
        "maximum_internal_silence_ms",
        "audio_writes",
        "premature_end_count",
    )
    summary: dict[str, object] = {"components": len(rows)}
    for arm in ARMS:
        summary[arm] = {
            key: sum(float(row["arms"][arm][key]) for row in rows) / max(1, len(rows))
            for key in keys
        }
    causal = float(summary["causal"]["target_coverage"])  # type: ignore[index]
    gold = float(summary["gold_offline"]["target_coverage"])  # type: ignore[index]
    summary["coverage_gain_from_gold_codes"] = gold - causal
    summary["coverage_ratio_gold_over_causal"] = gold / max(1e-9, causal)
    summary["verdict"] = (
        "input_mismatch_dominates_retrain_on_causal_codes"
        if gold >= max(0.20, 2.0 * causal)
        else "capability_gap_dominates_data_and_budget_must_change"
        if gold < 0.10
        else "both_contribute"
    )
    return {
        "schema_version": SCHEMA,
        "experiment": "0-C_teacher_forced_ceiling",
        "question": (
            "is the 3% target coverage a capability gap or the consequence of the "
            "causal/offline code mismatch measured in 0-A"
        ),
        "runtime_manifest": manifest,
        "arms": {
            "causal": "shipped inference path, objective._nearest_codes",
            "gold_offline": "non-causal GLM4 source_glm, the exact SFT training input",
        },
        "decision_thresholds": {
            "input_mismatch_dominates": "gold target_coverage >= max(0.20, 2x causal)",
            "capability_gap_dominates": "gold target_coverage < 0.10",
        },
        "summary": summary,
        "components_detail": rows,
    }


def _component_directions(episodes_path: Path) -> dict[str, dict[str, object]]:
    """Language pair, speaker tokens and references for each component."""

    output: dict[str, dict[str, object]] = {}
    with episodes_path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            episode = json.loads(line)
            for component in episode.get("components", []):
                sample_id = str(component["sample_id"])
                output.setdefault(
                    sample_id,
                    {
                        "src_lang": str(episode["src_lang"]),
                        "tgt_lang": str(episode["tgt_lang"]),
                        "transcription": str(component["transcription"]),
                        "translation": str(component["translation"]),
                        "speaker_global": [
                            int(value) for value in component["speaker_global"]
                        ],
                    },
                )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=Path, required=True)
    parser.add_argument("--base-hf", type=Path, required=True)
    parser.add_argument("--adapter-checkpoint", type=Path, required=True)
    parser.add_argument("--whispervq-model", type=Path, required=True)
    parser.add_argument("--bicodec-model", type=Path, required=True)
    parser.add_argument("--source-snapshot", type=Path, required=True)
    parser.add_argument("--components", type=int, default=8)
    parser.add_argument("--decision-chunk-ms", type=int, default=320)
    parser.add_argument("--acoustic-rollover-ms", type=int, default=24000)
    parser.add_argument("--policy-temperature", type=float, default=0.70)
    parser.add_argument("--policy-top-p", type=float, default=0.90)
    parser.add_argument("--action-temperature", type=float, default=0.80)
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-audio", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.components <= 0:
        raise ValueError("--components must be positive")
    report = evaluate(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(report["summary"], indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

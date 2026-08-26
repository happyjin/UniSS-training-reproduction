#!/usr/bin/env python3
"""Generate group-relative free-running episode trajectories with real audio."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import sacrebleu
import torch

from experiments.uniss_phase3_v4_quality_first_true_streaming_pilot15_v1.stage_a_causal_whisper_asr import (
    evaluate_checkpoint as stage_a_eval,
)
from experiments.uniss_phasea_stateful_longepisode_rl_v1.runtime.stateful_cascade import (
    evaluate_stateful_session,
)
from experiments.uniss_phasea_stateful_longepisode_rl_v1.training.episode_reward import (
    EpisodeObservation,
    group_relative_advantages,
    score_episode,
)
from experiments.uniss_phasea_stateful_longepisode_rl_v1.training.trace_generator import (
    TraceGenerator,
)
from experiments.uniss_stagea_quality_first_joint_grpo_v1.evaluation.strict_cascade import (
    _load_models,
)


def units(text: str, language: str) -> int:
    normalized = " ".join(str(text).split())
    return len(normalized.replace(" ", "")) if language == "cmn" else len(normalized.split())


def episode_observation(result: dict[str, object], row: dict[str, object]) -> EpisodeObservation:
    _, errors, reference_units = stage_a_eval.error_counts(
        str(row["teacher_transcription"]),
        str(result["generated_streaming_transcription"]),
        str(row["src_lang"]),
    )
    asr_similarity = max(0.0, 1.0 - errors / max(1, reference_units))
    hypothesis = str(result["generated_streaming_translation"])
    reference = str(row["teacher_translation"])
    mt_similarity = float(sacrebleu.corpus_chrf([hypothesis], [[reference]]).score) / 100.0
    length_ratio = units(hypothesis, str(row["tgt_lang"])) / max(
        1, units(reference, str(row["tgt_lang"]))
    )
    emissions = [
        emission
        for event in result["events"]  # type: ignore[union-attr]
        for emission in event["tts_emissions"]
    ]
    healthy = sum(bool(value["acknowledged"]) for value in emissions)
    health_fraction = healthy / max(1, len(emissions) + int(result["tts_pending_unspoken_items"]))
    pending_units = sum(
        units(text, str(row["tgt_lang"]))
        for text in result["tts_pending_unspoken_text"]  # type: ignore[union-attr]
    )
    spoken_fraction = max(
        0.0,
        1.0 - pending_units / max(1, units(hypothesis, str(row["tgt_lang"]))),
    )
    conflicts = int(result["asr_revision_conflicts"]) + int(result["mt_revision_conflicts"])
    stability = 1.0 / (1.0 + conflicts)
    first_write = result["first_audio_source_ms"]
    if first_write is None:
        first_write = int(result["source_duration_ms"]) + 12_000
    return EpisodeObservation(
        asr_teacher_similarity=asr_similarity,
        mt_teacher_similarity=mt_similarity,
        translation_length_ratio=length_ratio,
        healthy_audio_fraction=health_fraction,
        spoken_text_fraction=spoken_fraction,
        commit_stability=stability,
        speaker_similarity=1.0,
        first_write_ms=float(first_write),
        maximum_internal_silence_ms=float(result["maximum_internal_timeline_silence_ms"]),
        premature_end_count=int(result["rejected_early_end"]),
        tts_failure_count=int(result["tts_failures"]),
        invalid_semantic_fraction=0.0,
    )


def compact_result(result: dict[str, object]) -> dict[str, object]:
    excluded = {"events", "playback_schedule"}
    return {key: value for key, value in result.items() if key not in excluded}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker-index", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--maximum-episodes", type=int)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--decision-chunk-ms", type=int, default=640)
    parser.add_argument("--acoustic-rollover-ms", type=int, default=24000)
    parser.add_argument("--base-hf", type=Path, required=True)
    parser.add_argument("--v1-checkpoint", type=Path, required=True)
    parser.add_argument("--whispervq-model", type=Path, required=True)
    parser.add_argument("--bicodec-model", type=Path, required=True)
    parser.add_argument("--source-snapshot", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--policy-temperature", type=float, default=0.7)
    parser.add_argument("--policy-top-p", type=float, default=0.9)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.group_size < 2:
        raise ValueError("group size must be at least two")
    rows = [json.loads(line) for line in args.episodes.read_text(encoding="utf-8").splitlines() if line]
    rows = [row for index, row in enumerate(rows) if index % args.num_workers == args.worker_index]
    if args.maximum_episodes is not None:
        rows = rows[: args.maximum_episodes]
    if not rows:
        raise ValueError("worker episode selection is empty")
    snapshot = json.loads(args.source_snapshot.read_text(encoding="utf-8"))
    fixed_speaker = [int(value) for value in snapshot["fixed_system_speaker"]["global_tokens"]]
    load_args = SimpleNamespace(
        device=args.device,
        adapter_checkpoint=None,
        base_hf=args.base_hf,
        v1_checkpoint=args.v1_checkpoint,
        whispervq_model=args.whispervq_model,
        bicodec_model=args.bicodec_model,
    )
    model, tokenizer, controller, model_manifest, objective, codec = _load_models(load_args)
    args.output.mkdir(parents=True)
    trajectory_path = args.output / "trajectories.jsonl"
    summaries: list[dict[str, object]] = []
    try:
        with trajectory_path.open("w", encoding="utf-8") as trajectory_file:
            for episode_index, row in enumerate(rows):
                episode_id = str(row["episode_id"])
                candidates: list[dict[str, object]] = []
                for group_index in range(args.group_size):
                    trace_generator = TraceGenerator(
                        policy_temperature=args.policy_temperature,
                        policy_top_p=args.policy_top_p,
                    )
                    runtime_row = {
                        "id": f"{episode_id}_g{group_index}",
                        "src_lang": str(row["src_lang"]),
                        "tgt_lang": str(row["tgt_lang"]),
                        "source_audio": str(row["source_audio"]),
                        "bicodec_global": [int(value) for value in row["speaker_global"]],
                        "_stage_a_fixed_speaker_global": fixed_speaker,
                    }
                    result = evaluate_stateful_session(
                        runtime_row,
                        decision_chunk_ms=args.decision_chunk_ms,
                        acoustic_rollover_ms=args.acoustic_rollover_ms,
                        model=model,
                        tokenizer=tokenizer,
                        objective=objective,
                        bicodec=codec,
                        generate_fn=trace_generator,
                        output=args.output / "audio",
                        seed=20260826 + args.worker_index * 100_000_000 + episode_index * 100_000 + group_index * 1000,
                    )
                    observation = episode_observation(result, row)
                    reward = score_episode(observation)
                    candidates.append(
                        {
                            "group_index": group_index,
                            "result": compact_result(result),
                            "observation": observation.__dict__,
                            "reward": reward.to_dict(),
                            "traces": [trace.to_dict() for trace in trace_generator.traces],
                        }
                    )
                    print(
                        json.dumps(
                            {
                                "episode_id": episode_id,
                                "group_index": group_index,
                                "reward": reward.total,
                                "traces": len(trace_generator.traces),
                                "first_write_ms": observation.first_write_ms,
                                "spoken_text_fraction": observation.spoken_text_fraction,
                            },
                            ensure_ascii=False,
                            sort_keys=True,
                        ),
                        flush=True,
                    )
                advantages = group_relative_advantages(
                    [float(candidate["reward"]["total"]) for candidate in candidates]  # type: ignore[index]
                )
                for candidate, advantage in zip(candidates, advantages):
                    candidate["advantage"] = advantage
                    for trace_index, trace in enumerate(candidate.pop("traces")):
                        trajectory_file.write(
                            json.dumps(
                                {
                                    "schema_version": "uniss_free_running_episode_grpo_trajectory_v1",
                                    "episode_id": episode_id,
                                    "group_index": candidate["group_index"],
                                    "trace_index": trace_index,
                                    "advantage": advantage,
                                    "reward": candidate["reward"]["total"],  # type: ignore[index]
                                    **trace,
                                },
                                ensure_ascii=False,
                                separators=(",", ":"),
                            )
                            + "\n"
                        )
                summaries.append(
                    {
                        "episode_id": episode_id,
                        "direction": row["direction"],
                        "source_audio": row["source_audio"],
                        "teacher_transcription": row["teacher_transcription"],
                        "teacher_translation": row["teacher_translation"],
                        "candidates": candidates,
                    }
                )
    finally:
        controller.close()
    payload = {
        "schema_version": "uniss_free_running_episode_grpo_rollout_v1",
        "status": "complete",
        "worker_index": args.worker_index,
        "num_workers": args.num_workers,
        "group_size": args.group_size,
        "episodes": len(summaries),
        "model_manifest": model_manifest,
        "trajectory_path": str(trajectory_path.resolve()),
        "summaries": summaries,
    }
    (args.output / "ROLLOUT.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"OUTPUT={args.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()


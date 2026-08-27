#!/usr/bin/env python3
"""Group-eight train-seen rollouts with ASR/MT/TTS adapter routing aligned."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from experiments.uniss_phasea_route_aligned_constrained_grpo_v1.training.constrained_reward import (
    score_constrained_episode,
)
from experiments.uniss_phasea_route_aligned_constrained_grpo_v1.training.trace_generator import (
    RouteAlignedTraceGenerator,
)
from experiments.uniss_phasea_stateful_longepisode_rl_v1.runtime.stateful_cascade import (
    evaluate_stateful_session,
)
from experiments.uniss_phasea_stateful_longepisode_rl_v1.training.episode_reward import (
    EpisodeObservation,
    group_relative_advantages,
)
from experiments.uniss_phasea_stateful_longepisode_rl_v1.training.rollout import (
    compact_result,
    episode_observation,
)
from experiments.uniss_stagea_quality_first_joint_grpo_v1.evaluation.strict_cascade import (
    _load_models,
)


def load_baselines(path: Path) -> dict[str, EpisodeObservation]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    output: dict[str, EpisodeObservation] = {}
    for summary in payload["summaries"]:
        candidates = list(summary["candidates"])
        best = max(candidates, key=lambda row: float(row["reward"]["total"]))
        output[str(summary["episode_id"])] = EpisodeObservation(
            **best["observation"]
        )
    return output


def selected_episode_ids(protocol: Path) -> set[str]:
    payload = json.loads(protocol.read_text(encoding="utf-8"))
    return {str(row["sample_id"]) for row in payload["records"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--baseline-rollout", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker-index", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--decision-chunk-ms", type=int, default=640)
    parser.add_argument("--acoustic-rollover-ms", type=int, default=24000)
    parser.add_argument("--base-hf", type=Path, required=True)
    parser.add_argument("--adapter-checkpoint", type=Path, required=True)
    parser.add_argument("--v1-checkpoint", type=Path, required=True)
    parser.add_argument("--whispervq-model", type=Path, required=True)
    parser.add_argument("--bicodec-model", type=Path, required=True)
    parser.add_argument("--source-snapshot", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--asr-temperature", type=float, default=0.30)
    parser.add_argument("--policy-temperature", type=float, default=0.70)
    parser.add_argument("--policy-top-p", type=float, default=0.90)
    parser.add_argument("--retention", type=float, default=0.98)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    requested = selected_episode_ids(args.protocol)
    rows = [
        json.loads(line)
        for line in args.episodes.read_text(encoding="utf-8").splitlines()
        if line
    ]
    rows = [row for row in rows if str(row["episode_id"]) in requested]
    observed = {str(row["episode_id"]) for row in rows}
    if observed != requested:
        raise ValueError(f"protocol episodes missing from training manifest: {requested-observed}")
    rows = [
        row
        for index, row in enumerate(rows)
        if index % int(args.num_workers) == int(args.worker_index)
    ]
    if not rows:
        raise ValueError("worker episode selection is empty")
    baselines = load_baselines(args.baseline_rollout)
    snapshot = json.loads(args.source_snapshot.read_text(encoding="utf-8"))
    fixed_speaker = [
        int(value) for value in snapshot["fixed_system_speaker"]["global_tokens"]
    ]
    load_args = SimpleNamespace(
        device=args.device,
        adapter_checkpoint=args.adapter_checkpoint,
        base_hf=args.base_hf,
        v1_checkpoint=args.v1_checkpoint,
        whispervq_model=args.whispervq_model,
        bicodec_model=args.bicodec_model,
    )
    model, tokenizer, controller, model_manifest, objective, codec = _load_models(
        load_args
    )
    args.output.mkdir(parents=True)
    trajectory_path = args.output / "trajectories.jsonl"
    summaries: list[dict[str, object]] = []
    try:
        with trajectory_path.open("w", encoding="utf-8") as trajectory_file:
            for episode_index, row in enumerate(rows):
                episode_id = str(row["episode_id"])
                baseline = baselines[episode_id]
                candidates: list[dict[str, object]] = []
                for group_index in range(int(args.group_size)):
                    trace_generator = RouteAlignedTraceGenerator(
                        asr_temperature=args.asr_temperature,
                        policy_temperature=args.policy_temperature,
                        policy_top_p=args.policy_top_p,
                    )

                    def routed_generate(*call_args, **call_kwargs):
                        with controller.route(True):
                            return trace_generator(*call_args, **call_kwargs)

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
                        generate_fn=routed_generate,
                        output=args.output / "audio",
                        seed=(
                            20260827
                            + args.worker_index * 100_000_000
                            + episode_index * 100_000
                            + group_index * 1000
                        ),
                    )
                    observation = episode_observation(result, row)
                    reward = score_constrained_episode(
                        observation, baseline, retention=args.retention
                    )
                    candidates.append(
                        {
                            "group_index": group_index,
                            "result": compact_result(result),
                            "observation": observation.__dict__,
                            "baseline_observation": baseline.__dict__,
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
                                "quality_gate": reward.quality_gate,
                                "asr": observation.asr_teacher_similarity,
                                "mt": observation.mt_teacher_similarity,
                                "first_write_ms": observation.first_write_ms,
                                "traces": len(trace_generator.traces),
                            },
                            ensure_ascii=False,
                            sort_keys=True,
                        ),
                        flush=True,
                    )
                advantages = group_relative_advantages(
                    [float(candidate["reward"]["total"]) for candidate in candidates]
                )
                for candidate, advantage in zip(candidates, advantages):
                    candidate["advantage"] = advantage
                    for trace_index, trace in enumerate(candidate.pop("traces")):
                        trajectory_file.write(
                            json.dumps(
                                {
                                    "schema_version": "uniss_route_aligned_constrained_trajectory_v1",
                                    "episode_id": episode_id,
                                    "group_index": candidate["group_index"],
                                    "trace_index": trace_index,
                                    "advantage": advantage,
                                    "reward": candidate["reward"]["total"],
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
        "schema_version": "uniss_route_aligned_constrained_rollout_v1",
        "status": "complete",
        "worker_index": args.worker_index,
        "num_workers": args.num_workers,
        "group_size": args.group_size,
        "episodes": len(summaries),
        "model_manifest": model_manifest,
        "route_semantics": "adapter_on_asr_mt_tts_control",
        "trajectory_path": str(trajectory_path.resolve()),
        "summaries": summaries,
    }
    (args.output / "ROLLOUT.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"OUTPUT={args.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()


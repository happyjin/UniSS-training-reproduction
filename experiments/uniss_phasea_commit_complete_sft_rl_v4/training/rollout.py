#!/usr/bin/env python3
"""Fresh group-four event-policy rollouts on frozen bidirectional episodes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from experiments.uniss_phasea_commit_complete_sft_rl_v4.runtime.commit_policy_cascade import (
    evaluate_event_policy_session,
)
from experiments.uniss_phasea_commit_complete_sft_rl_v4.training.event_credit import (
    assign_trace_advantages,
)
from experiments.uniss_phasea_coverage_constrained_grpo_v3.training.coverage import (
    audit_episode,
)
from experiments.uniss_phasea_commit_complete_sft_rl_v4.training.reward import (
    score_episode,
)
from experiments.uniss_phasea_coverage_constrained_grpo_v3.training.trace_generator import (
    EventTraceGenerator,
)
from experiments.uniss_phasea_route_aligned_constrained_grpo_v1.training.rollout import (
    load_baselines,
)
from experiments.uniss_phasea_stateful_longepisode_rl_v1.training.rollout import (
    compact_result,
    episode_observation,
)
from experiments.uniss_stagea_quality_first_joint_grpo_v1.evaluation.strict_cascade import (
    _load_models,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=Path, required=True)
    parser.add_argument("--baseline-rollout", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker-index", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--decision-chunk-ms", type=int, default=320)
    parser.add_argument("--acoustic-rollover-ms", type=int, default=24000)
    parser.add_argument("--base-hf", type=Path, required=True)
    parser.add_argument("--adapter-checkpoint", type=Path, required=True)
    parser.add_argument("--v1-checkpoint", type=Path, required=True)
    parser.add_argument("--whispervq-model", type=Path, required=True)
    parser.add_argument("--bicodec-model", type=Path, required=True)
    parser.add_argument("--source-snapshot", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--policy-temperature", type=float, default=0.70)
    parser.add_argument("--policy-top-p", type=float, default=0.90)
    parser.add_argument("--action-temperature", type=float, default=0.80)
    parser.add_argument("--round-index", type=int, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.group_size < 2 or args.num_workers <= 0:
        raise ValueError("invalid rollout geometry")
    rows = [
        json.loads(line)
        for line in args.episodes.read_text(encoding="utf-8").splitlines()
        if line
    ]
    rows = [
        row
        for index, row in enumerate(rows)
        if index % args.num_workers == args.worker_index
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
    model, tokenizer, controller, model_manifest, objective, codec = _load_models(load_args)
    args.output.mkdir(parents=True)
    trajectory_path = args.output / "trajectories.jsonl"
    summaries: list[dict[str, object]] = []
    try:
        with trajectory_path.open("w", encoding="utf-8") as trajectory_file:
            for episode_index, row in enumerate(rows):
                episode_id = str(row["episode_id"])
                baseline = baselines[episode_id]
                candidates: list[dict[str, object]] = []
                for group_index in range(args.group_size):
                    tracer = EventTraceGenerator(
                        controller,
                        policy_temperature=args.policy_temperature,
                        policy_top_p=args.policy_top_p,
                        action_temperature=args.action_temperature,
                    )
                    runtime_row = {
                        "id": f"{episode_id}_g{group_index}",
                        "src_lang": str(row["src_lang"]),
                        "tgt_lang": str(row["tgt_lang"]),
                        "source_audio": str(row["source_audio"]),
                        "bicodec_global": [int(value) for value in row["speaker_global"]],
                        "_stage_a_fixed_speaker_global": fixed_speaker,
                    }
                    result = evaluate_event_policy_session(
                        runtime_row,
                        decision_chunk_ms=args.decision_chunk_ms,
                        acoustic_rollover_ms=args.acoustic_rollover_ms,
                        model=model,
                        tokenizer=tokenizer,
                        objective=objective,
                        bicodec=codec,
                        generate_fn=tracer,
                        action_fn=tracer.decide,
                        event_context_fn=tracer.set_event,
                        output=args.output / "audio",
                        seed=(
                            20260827
                            + args.round_index * 1_000_000_000
                            + args.worker_index * 10_000_000
                            + episode_index * 100_000
                            + group_index * 1_000
                        ),
                    )
                    observation = episode_observation(result, row)
                    coverage = audit_episode(
                        teacher_translation=str(row["teacher_translation"]),
                        generated_translation=str(result["generated_streaming_translation"]),
                        target_language=str(row["tgt_lang"]),
                        events=result["events"],
                        eos_pending_items=int(result["tts_pending_unspoken_items"]),
                    )
                    # The control trace is sampled before the cascade knows
                    # whether stable target text is available.  Attach the
                    # runtime outcome now so the trainer can ignore control
                    # logits that could not have changed audio behaviour.
                    event_by_index = {
                        int(event["event_index"]): event for event in result["events"]
                    }
                    for trace in tracer.tagged_traces:
                        if str(trace["family"]) != "control":
                            continue
                        event = event_by_index[int(trace["event_index"])]
                        trace["actionable_commit"] = bool(
                            event["actionable_commit"]
                        )
                        trace["actual_commit"] = bool(
                            str(event["executed_action"]) == "WRITE"
                            and any(
                                bool(item.get("acknowledged", False))
                                for item in event["tts_emissions"]
                            )
                        )
                        trace["spoken_target_coverage_delta"] = float(
                            event["coverage"]["spoken_target_coverage_delta"]
                        )
                    reward = score_episode(observation, baseline, coverage)
                    candidates.append(
                        {
                            "group_index": group_index,
                            "result": result,
                            "observation": observation.__dict__,
                            "baseline_observation": baseline.__dict__,
                            "coverage_audit": coverage.to_dict(),
                            "reward": reward.to_dict(),
                            "mapped_action_events": row["mapped_action_events"],
                            "traces": tracer.tagged_traces,
                        }
                    )
                    print(
                        json.dumps(
                            {
                                "episode_id": episode_id,
                                "group_index": group_index,
                                "reward": reward.total,
                                "first_write_ms": observation.first_write_ms,
                                "silence_ms": observation.maximum_internal_silence_ms,
                                "target_coverage": coverage.target_coverage,
                                "spoken_target_coverage": coverage.spoken_target_coverage,
                                "traces": len(tracer.tagged_traces),
                            },
                            ensure_ascii=False,
                            sort_keys=True,
                        ),
                        flush=True,
                    )
                assign_trace_advantages(candidates)
                summary_candidates = []
                for candidate in candidates:
                    for trace_index, trace in enumerate(candidate.pop("traces")):
                        trajectory_file.write(
                            json.dumps(
                                {
                                    "schema_version": "uniss_content_gated_commit_trajectory_v4",
                                    "round_index": args.round_index,
                                    "episode_id": episode_id,
                                    "group_index": candidate["group_index"],
                                    "trace_index": trace_index,
                                    "episode_reward": candidate["reward"]["total"],
                                    **trace,
                                },
                                ensure_ascii=False,
                                separators=(",", ":"),
                            )
                            + "\n"
                        )
                    result = candidate.pop("result")
                    candidate.pop("mapped_action_events")
                    candidate["result"] = compact_result(result)
                    candidate["event_count"] = len(result["events"])
                    candidate["policy_write_count"] = sum(
                        str(event["policy_action"]) == "WRITE" for event in result["events"]
                    )
                    candidate["deadline_forced_write_count"] = sum(
                        bool(event["deadline_forced_write"]) for event in result["events"]
                    )
                    summary_candidates.append(candidate)
                summaries.append(
                    {
                        "episode_id": episode_id,
                        "direction": row["direction"],
                        "source_audio": row["source_audio"],
                        "teacher_transcription": row["teacher_transcription"],
                        "teacher_translation": row["teacher_translation"],
                        "candidates": summary_candidates,
                    }
                )
    finally:
        controller.close()
    payload = {
        "schema_version": "uniss_content_gated_commit_rollout_v4",
        "status": "complete",
        "round_index": args.round_index,
        "worker_index": args.worker_index,
        "num_workers": args.num_workers,
        "group_size": args.group_size,
        "episodes": len(summaries),
        "model_manifest": model_manifest,
        "route_semantics": "asr_and_incremental_mt_update_every_event; action_drains_only_nonempty_stable_commit",
        "trajectory_path": str(trajectory_path.resolve()),
        "summaries": summaries,
    }
    (args.output / "ROLLOUT.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"OUTPUT={args.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""One continuous Phase3-rooted Megatron event-rollout joint SFT."""

from __future__ import annotations

import json
import os
import types
from pathlib import Path

import torch

import experiments.uniss_phase3_dense_aligned_streaming_pilot15_v1.training.pretrain_dense_aligned_megatron as dense
import experiments.uniss_phase3_true_subsecond_deadline_full198_v1.training.pretrain_true_subsecond_megatron as base
from experiments.uniss_phase3_event_rollout_joint_full198_v1.event_rollout import (
    build_recovery_example,
    choose_recovery_event,
)
from experiments.uniss_phase3_event_rollout_joint_full198_v1.native_kv_backend import (
    NativeMegatronKVBackend,
)
from experiments.uniss_phase3_event_rollout_joint_full198_v1.rollout_policy import (
    rollout_schedule,
    rollout_session,
)
from experiments.uniss_phase3_event_rollout_joint_full198_v1.training.dataset import (
    IndexedEventRolloutTrajectoryDataset,
    collate_event_rollout,
    replace_trajectory_batch_with_recovery,
)
from experiments.uniss_phase3_event_rollout_joint_full198_v1.training.objective import (
    EventRolloutJointObjective,
    ROLLOUT_METRIC_NAMES,
    distributed_event_rollout_objective,
)
from experiments.uniss_phase3_runtime_parity_streaming_v2.generalize12_microblock.pretrain_generalize12 import (
    SynchronizedValidationDataset,
)
from experiments.uniss_phase3_runtime_parity_streaming_v2.generalize13_joint_runtime.pretrain_generalize13 import (
    is_generalize13_trainable_parameter,
)
from training.pretrain_uniss_megatron import load_megatron_runtime


METRIC_NAMES = ROLLOUT_METRIC_NAMES
PHASE3_NATIVE_PREFIXES = ("embedding.", "decoder.", "output_layer.")
NEW_MODULE_PREFIXES = ("true_subsecond_lora.", "true_subsecond_objective.")


def _metadata_base_key(value: str) -> str:
    return str(value).split("/shard_", 1)[0]


def validate_phase3_handoff_key_sets(
    checkpoint_keys, current_model_keys
) -> dict[str, object]:
    """Require every native Phase3 key and allow only isolated new modules."""

    checkpoint_native = {
        _metadata_base_key(key)
        for key in checkpoint_keys
        if _metadata_base_key(key).startswith(PHASE3_NATIVE_PREFIXES)
    }
    current = {_metadata_base_key(key) for key in current_model_keys}
    missing_native = sorted(checkpoint_native - current)
    illegal_new = sorted(
        key
        for key in current - checkpoint_native
        if not key.startswith(NEW_MODULE_PREFIXES)
    )
    if missing_native or illegal_new:
        raise RuntimeError(
            "Phase3 handoff key audit failed: "
            f"missing_native={missing_native[:20]} illegal_new={illegal_new[:20]}"
        )
    if not checkpoint_native:
        raise RuntimeError("Phase3 handoff key audit found no native model keys")
    new_keys = sorted(key for key in current if key.startswith(NEW_MODULE_PREFIXES))
    if not new_keys:
        raise RuntimeError("Phase3 handoff key audit found no isolated new modules")
    return {
        "native_checkpoint_keys": len(checkpoint_native),
        "current_model_keys": len(current),
        "allowed_new_keys": len(new_keys),
        "allowed_new_prefixes": list(NEW_MODULE_PREFIXES),
    }


def audit_phase3_handoff_structure(model, load_root: str | Path) -> dict[str, object]:
    from megatron.core import parallel_state
    from torch.distributed.checkpoint import FileSystemReader

    root = Path(load_root).resolve()
    latest = (root / "latest_checkpointed_iteration.txt").read_text().strip()
    checkpoint_dir = root / f"iter_{int(latest):07d}"
    checkpoint_keys = FileSystemReader(str(checkpoint_dir)).read_metadata().state_dict_metadata
    current = model.sharded_state_dict(
        metadata={
            "dp_cp_group": parallel_state.get_data_parallel_group(
                with_context_parallel=True
            )
        }
    )
    canonical_current = {
        getattr(value, "key", key) for key, value in current.items()
    }
    return validate_phase3_handoff_key_sets(checkpoint_keys, canonical_current)


def _event_output_processor(**kwargs) -> torch.Tensor:
    context = kwargs["context"]
    objective = context["objective"]
    hidden = kwargs["hidden_states"]
    logits, _ = kwargs["output_layer"](
        hidden,
        weight=kwargs["output_weight"],
        runtime_gather_output=kwargs["runtime_gather_output"],
    )
    logits = kwargs["scale_logits"](logits)
    if hidden.ndim != 3 or hidden.shape[1] != 1 or logits.shape[1] != 1:
        raise ValueError("event-rollout TP=PP=1 expects flattened [tokens,1,*]")
    hidden = hidden[:, 0]
    logits = logits[:, 0]
    labels = kwargs["labels"].reshape(-1)
    loss_mask = kwargs["loss_mask"].reshape(-1)
    batch = context["batch"]
    if context["sample_kind"] == "replay":
        output = objective.replay(logits, labels, loss_mask)
    elif context["sample_kind"] == "trajectory":
        output = objective.trajectory(
            hidden,
            logits,
            labels,
            loss_mask,
            batch["token_roles"].reshape(-1),
            context["word_embedding_weight"],
            batch,
            frontend_residual_rms=context["frontend_residual_rms"],
        )
    else:
        raise ValueError(f"unknown event-rollout sample kind: {context['sample_kind']}")
    total, metrics = distributed_event_rollout_objective(
        output, progress=float(context["progress"])
    )
    if tuple(metrics) != METRIC_NAMES:
        raise AssertionError("event-rollout metric order changed")
    values = (total.float(), *[metrics[name].float() for name in METRIC_NAMES])
    if not all(torch.isfinite(value).all() for value in values):
        raise FloatingPointError("non-finite event-rollout loss component")
    return torch.stack(values)


def _attach_event_rollout_forward(model, fingerprint: str | None) -> None:
    """Keep an inference-capable native forward beside the training wrapper."""

    raw_forward = model.forward
    model._event_rollout_raw_forward = raw_forward

    def forward_with_event_rollout(
        self,
        input_ids,
        position_ids,
        attention_mask,
        decoder_input=None,
        labels=None,
        inference_context=None,
        packed_seq_params=None,
        extra_block_kwargs=None,
        runtime_gather_output=None,
        *,
        inference_params=None,
        loss_mask=None,
        padding_mask=None,
        output_processor=None,
        output_processor_context=None,
        true_subsecond_batch=None,
    ):
        if output_processor is not None or output_processor_context is not None:
            raise ValueError("the event-rollout entrypoint owns the output processor")
        if true_subsecond_batch is None:
            raise ValueError("missing event-rollout sidecar batch")
        base.verify_phase3_fingerprint(self, fingerprint)
        kind = str(true_subsecond_batch["sample_kind"])
        residual_rms = base._embedding_weight(self).sum() * 0.0
        if kind == "trajectory":
            if decoder_input is not None:
                raise ValueError("trajectory frontend cannot replace decoder input")
            decoder_input = self.embedding(
                input_ids=input_ids, position_ids=position_ids
            )
            decoder_input, residual_rms = self.true_subsecond_objective.inject_frontend_residual(
                decoder_input,
                true_subsecond_batch,
                original_seq_length=int(
                    true_subsecond_batch["original_seq_length"].item()
                ),
            )
        context = {
            "objective": self.true_subsecond_objective,
            "batch": true_subsecond_batch,
            "sample_kind": kind,
            "frontend_residual_rms": residual_rms,
            "word_embedding_weight": base._embedding_weight(self),
            "progress": float(true_subsecond_batch["training_progress"].item()),
        }
        return raw_forward(
            input_ids,
            position_ids,
            attention_mask,
            decoder_input=decoder_input,
            labels=labels,
            inference_context=inference_context,
            packed_seq_params=packed_seq_params,
            extra_block_kwargs=extra_block_kwargs,
            runtime_gather_output=runtime_gather_output,
            inference_params=inference_params,
            loss_mask=loss_mask,
            padding_mask=padding_mask,
            output_processor=_event_output_processor,
            output_processor_context=context,
        )

    model.forward = types.MethodType(forward_with_event_rollout, model)


def _is_event_rollout_trainable(name: str) -> bool:
    return is_generalize13_trainable_parameter(name) or (
        "true_subsecond_objective.continuation_head." in name
    )


def install_event_rollout_model() -> None:
    original_augment = base.augment_native_gpt
    base.attach_true_subsecond_forward = _attach_event_rollout_forward

    def augment(model, args):
        summary = original_augment(model, args)
        trainable = 0
        for name, parameter in model.named_parameters():
            keep = _is_event_rollout_trainable(name)
            parameter.requires_grad_(keep)
            if keep:
                trainable += parameter.numel()
        if trainable <= 0:
            raise RuntimeError("event-rollout run found no trainable parameters")
        model._event_rollout_trainable_parameters = trainable
        if Path(args.load).resolve() == Path(
            "checkpoints/uniss_qwen0p5b_phase3_unist198_after_phase2_v4"
        ).resolve():
            audit = audit_phase3_handoff_structure(model, args.load)
            model._event_rollout_phase3_key_audit = audit
            if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
                print(
                    json.dumps(
                        {"event": "phase3_handoff_key_audit", **audit},
                        sort_keys=True,
                    ),
                    flush=True,
                )
        return summary

    base.augment_native_gpt = augment


def _trajectory_dataset(args, packed: str):
    return IndexedEventRolloutTrajectoryDataset(
        packed, seq_length=int(args.seq_length)
    )


def train_valid_test_datasets_provider(train_val_test_num_samples, vp_stage=None):
    del vp_stage
    runtime = load_megatron_runtime()
    args = runtime.megatron_gpt.get_args()
    manifest = dense._manifest(args)
    trajectory = _trajectory_dataset(args, args.true_trajectory_packed)
    replay = dense.IndexedPhase3ReplayDataset(
        args.true_replay_packed,
        args.true_replay_offsets,
        seq_length=int(args.seq_length),
        require_complete=True,
    )
    target_train = dense._target_count(train_val_test_num_samples, 0)
    if target_train is None:
        raise ValueError("Megatron did not provide an event-rollout train count")
    dp_group = int(args.data_parallel_size) * int(args.micro_batch_size)
    train = dense.ThreeEpochGlobalShuffleSchedule(
        trajectory,
        replay,
        coverage_epochs=int(args.dense_coverage_epochs),
        data_parallel_group_size=dp_group,
        global_batch_size=int(args.global_batch_size),
        shuffle_seed=int(args.seed),
        target_replay_fraction=float(args.dense_replay_fraction),
    )
    train.collate_fn = collate_event_rollout
    if len(train) != target_train or len(train) != int(manifest["total_samples"]):
        raise ValueError("event-rollout schedule differs from the frozen manifest")
    valid_sources = []
    if args.true_valid_trajectory_packed:
        valid_sources.append(
            _trajectory_dataset(args, args.true_valid_trajectory_packed)
        )
    if args.true_valid_replay_packed:
        valid_sources.append(
            dense.IndexedPhase3ReplayDataset(
                args.true_valid_replay_packed,
                args.true_valid_replay_offsets,
                seq_length=int(args.seq_length),
                require_complete=True,
            )
        )
    valid = SynchronizedValidationDataset(valid_sources) if valid_sources else None
    if valid is not None:
        valid.collate_fn = collate_event_rollout
    runtime.print_rank_0(
        "> event-rollout datasets: "
        f"trajectory={len(trajectory)} replay={len(replay)} "
        f"epochs={train.coverage_epochs} total={len(train)} "
        f"valid={0 if valid is None else len(valid)}"
    )
    return train, valid, None


train_valid_test_datasets_provider.is_distributed = True


def _unwrap_training_model(model):
    from megatron.core.utils import unwrap_model

    value = unwrap_model(model)
    if isinstance(value, list):
        if len(value) != 1:
            raise ValueError(
                "event-rollout training expects exactly one local model chunk, "
                f"got {len(value)}"
            )
        value = value[0]
    return value


def _model_training(model) -> bool:
    return bool(getattr(_unwrap_training_model(model), "training", False))


def _force_smoke_rollin() -> bool:
    raw = os.environ.get("EVENT_ROLLOUT_FORCE_ROLLIN", "0").strip().lower()
    if raw not in {"0", "1", "false", "true", "no", "yes"}:
        raise ValueError(f"invalid EVENT_ROLLOUT_FORCE_ROLLIN={raw!r}")
    enabled = raw in {"1", "true", "yes"}
    if enabled and "smoke" not in os.environ.get("RUN_NAME", "").lower():
        raise RuntimeError(
            "EVENT_ROLLOUT_FORCE_ROLLIN is a smoke-only coverage switch"
        )
    return enabled


def _rollin_examples(raw_batch, model, progress: float):
    schedule = rollout_schedule(progress)
    force = _force_smoke_rollin()
    if not force and (schedule.maximum_sessions <= 0 or schedule.fraction <= 0):
        return None
    if not force and torch.rand((), device="cpu").item() >= schedule.fraction:
        return None
    sessions_by_lane = raw_batch.get("oracle_sessions")
    if not isinstance(sessions_by_lane, list) or not sessions_by_lane:
        raise ValueError("trajectory batch is missing ordered oracle sessions")
    native = _unwrap_training_model(model)
    objective = native.true_subsecond_objective
    embedding_weight = base._embedding_weight(native)
    examples = []
    traces = []
    for lane_sessions in sessions_by_lane:
        if not lane_sessions:
            raise ValueError("trajectory lane has no oracle sessions")
        # One exact session per local MBS lane bounds the no-grad roll-in cost.
        session = lane_sessions[
            int(torch.randint(len(lane_sessions), (), device="cpu").item())
        ]
        backend = NativeMegatronKVBackend(native, objective)
        trace = rollout_session(session, backend, objective, embedding_weight)
        event = choose_recovery_event(session, trace)
        examples.append(build_recovery_example(session, trace, event))
        traces.append((session, trace, event))
    replacement = replace_trajectory_batch_with_recovery(
        raw_batch, examples, seq_length=int(raw_batch["tokens"].shape[1])
    )
    divergences = [
        -1 if trace.first_divergence(session) is None else trace.first_divergence(session)
        for session, trace, _ in traces
    ]
    replacement["event_rollout_first_divergence"] = torch.tensor(
        divergences, dtype=torch.float32
    )
    replacement["event_rollout_all_wait"] = torch.tensor(
        [all(tick.action == "WAIT" for tick in trace.generated_ticks) for _, trace, _ in traces],
        dtype=torch.float32,
    )
    replacement["event_rollout_false_write"] = torch.tensor(
        [
            any(
                generated.action == "WRITE" and oracle.events[index].action == "WAIT"
                for index, generated in enumerate(trace.generated_ticks)
            )
            for oracle, trace, _ in traces
        ],
        dtype=torch.float32,
    )
    replacement["event_rollout_grammar_valid"] = torch.ones(
        len(traces), dtype=torch.float32
    )
    replacement["event_rollout_stopped_early"] = torch.tensor(
        [trace.stopped_early for _, trace, _ in traces], dtype=torch.float32
    )
    return replacement


def forward_step(data_iterator, model):
    runtime = load_megatron_runtime()
    args = runtime.megatron_gpt.get_args()
    raw_batch = next(data_iterator)
    denominator = max(1, int(args.train_iters) * int(args.global_batch_size))
    consumed = int(getattr(args, "consumed_train_samples", 0) or 0)
    progress = min(1.0, max(0.0, consumed / denominator))
    is_trajectory = str(raw_batch.get("sample_kind")) == "trajectory"
    if _model_training(model) and is_trajectory:
        replacement = _rollin_examples(raw_batch, model, progress)
        if replacement is not None:
            raw_batch = replacement
    batch = base.prepare_packed_batch(raw_batch, int(args.seq_length))
    batch["training_progress"] = torch.tensor(
        progress, dtype=torch.float32, device=batch["tokens"].device
    )
    packed_seq_params = base.build_packed_seq_params(batch, int(args.seq_length))
    output = model(
        batch["tokens"],
        batch["position_ids"],
        None,
        labels=batch["labels"],
        loss_mask=batch["loss_mask"],
        packed_seq_params=packed_seq_params,
        true_subsecond_batch=batch,
    )
    return output, base.loss_func


def main() -> None:
    base.TrueSubsecondObjective = EventRolloutJointObjective
    base.METRIC_NAMES = METRIC_NAMES
    dense.METRIC_NAMES = METRIC_NAMES
    dense._distributed_dense_objective = distributed_event_rollout_objective
    dense.JointValidationDataset = SynchronizedValidationDataset
    install_event_rollout_model()

    runtime = load_megatron_runtime()
    args = runtime.parse_and_validate_args(
        extra_args_provider=dense.add_experiment_args,
        args_defaults={"tokenizer_type": "NullTokenizer"},
    )
    dense.validate_experiment_args(args)
    load_root = Path(args.load).resolve()
    phase3_root = Path(
        "checkpoints/uniss_qwen0p5b_phase3_unist198_after_phase2_v4"
    ).resolve()
    save_root = Path(args.save).resolve()
    fresh = load_root == phase3_root
    resume = load_root == save_root
    if not (fresh or resume):
        raise ValueError(
            "event-rollout training may load only Phase3 v4 or its own checkpoint root"
        )
    if fresh:
        latest = (load_root / "latest_checkpointed_iteration.txt").read_text().strip()
        if latest != "9075":
            raise ValueError("fresh event-rollout run must start at Phase3 iter_0009075")
        if str(args.dist_ckpt_strictness) not in {"log_all", "StrictHandling.LOG_ALL"}:
            raise ValueError("fresh Phase3 handoff requires mismatch audit with log_all")
    elif str(args.dist_ckpt_strictness) not in {"raise_all", "StrictHandling.RAISE_ALL"}:
        raise ValueError("event-rollout resume requires raise_all strictness")

    dense.install_dense_lr_overrides(args)
    dense.install_coverage_sampler()
    base.install_joint_collate()
    base.install_rerun_checkpoint_compatibility()
    model_config = runtime.gpt_config_from_args(args)
    full_config = runtime.pretrain_cfg_container_from_args(args, model_config)
    full_config.model = None
    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        print(
            json.dumps(
                {
                    "experiment": "uniss_phase3_event_rollout_joint_full198_v1",
                    "fresh_phase3": fresh,
                    "load": str(load_root),
                    "save": str(save_root),
                    "event_rollout": "exact_model_induced_variable_grammar",
                    "metric_count": len(METRIC_NAMES),
                },
                sort_keys=True,
            ),
            flush=True,
        )
    runtime.pretrain(
        full_config,
        train_valid_test_datasets_provider,
        runtime.ModelType.encoder_or_decoder,
        forward_step,
        model_provider=base.model_provider,
    )


if __name__ == "__main__":
    main()

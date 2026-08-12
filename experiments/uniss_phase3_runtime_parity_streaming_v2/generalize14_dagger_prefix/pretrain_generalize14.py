#!/usr/bin/env python3
"""DAgger-style model-prefix recovery for natural runtime parity.

Generalize13 learned oracle-prefix trajectories but failed when its own text
and semantic predictions entered the persistent KV cache.  Generalize14 runs
one or two no-gradient probes inside the Megatron step, rolls model predictions
into the length-preserving text/semantic payload positions, and optimizes the
same oracle labels from the resulting model-induced states.  It also activates
the grouped sub-second deadline objective that Generalize13 left at zero.
"""

from __future__ import annotations

from collections import OrderedDict

import torch
import torch.distributed as dist
from torch.nn import functional as F

import experiments.uniss_phase3_dense_aligned_streaming_pilot15_v1.training.pretrain_dense_aligned_megatron as dense
import experiments.uniss_phase3_runtime_parity_streaming_v2.overfit2.pretrain_overfit2 as v2
from experiments.uniss_phase3_dense_aligned_streaming_pilot15_v1.data.packing import (
    ROLE_ACTION,
    ROLE_BOUNDARY,
    ROLE_SEMANTIC,
    ROLE_TEXT,
)
from experiments.uniss_phase3_runtime_parity_streaming_v2.generalize12_microblock.microblock import (
    _balanced_example_weights,
)
from experiments.uniss_phase3_runtime_parity_streaming_v2.generalize12_microblock.pretrain_generalize12 import (
    SynchronizedValidationDataset,
)
from experiments.uniss_phase3_runtime_parity_streaming_v2.generalize13_joint_runtime.pretrain_generalize13 import (
    REPLAY_FRACTION,
    RuntimeParityGeneralize13Objective,
    V13_DIAGNOSTIC_NAMES,
    V13_TERM_NAMES,
    V13_WEIGHTS,
    is_generalize13_trainable_parameter,
)
from experiments.uniss_phase3_runtime_parity_streaming_v2.generalize14_dagger_prefix.prefix_rollout import (
    apply_prefix_predictions,
    expand_recovery_mask,
    prefix_schedule,
    probe_runtime_prefix_labels,
)
from experiments.uniss_phase3_runtime_parity_streaming_v2.overfit4.pretrain_overfit4 import (
    trajectory_token_weights,
)
from experiments.uniss_phase3_true_subsecond_deadline_full198_v1.training.joint_model import (
    DIAGNOSTIC_NAMES,
    TERM_NAMES,
    ObjectiveOutput,
    _binary_metrics,
    _parameter_anchor,
)
from experiments.uniss_phase3_true_subsecond_deadline_full198_v1.training.losses import (
    LossTerm,
    grouped_deadline_survival_term,
    token_cross_entropy_values,
    values_to_term,
    zero_term,
)
from training import constants_uniss as c


V14_EXTRA_TERMS = ("runtime_prefix_recovery",)
V14_EXTRA_DIAGNOSTICS = (
    "runtime_prefix_corruption_fraction",
    "runtime_prefix_recovery_accuracy",
    "runtime_prefix_rollout_rounds",
)
V14_TERM_NAMES = (*V13_TERM_NAMES, *V14_EXTRA_TERMS)
V14_DIAGNOSTIC_NAMES = (*V13_DIAGNOSTIC_NAMES, *V14_EXTRA_DIAGNOSTICS)
V14_METRIC_NAMES = (
    *V14_TERM_NAMES,
    *V14_DIAGNOSTIC_NAMES,
    "curriculum_deadline_weight",
    "curriculum_replay_fraction",
    "curriculum_frontend_lr_multiplier",
)

V14_WEIGHTS = OrderedDict(V13_WEIGHTS)
V14_WEIGHTS["deadline_survival"] = 1.0
V14_WEIGHTS["runtime_text_content"] = 2.0
V14_WEIGHTS["runtime_prefix_recovery"] = 2.0


def _masked_accuracy(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    anchor: torch.Tensor,
) -> torch.Tensor:
    active = mask.bool()
    if not bool(active.any()):
        return anchor.detach().new_zeros(())
    return (prediction[active] == target[active]).float().mean()


class RuntimeParityGeneralize14Objective(RuntimeParityGeneralize13Objective):
    """Generalize13 objective evaluated on scheduled model-prefix states."""

    def replay(self, logits, labels, loss_mask) -> ObjectiveOutput:
        output = super().replay(logits, labels, loss_mask)
        anchor = logits.sum() * 0.0
        terms = OrderedDict(output.terms)
        terms["runtime_prefix_recovery"] = zero_term(anchor)
        diagnostics = OrderedDict(output.diagnostics)
        zero = anchor.detach().new_zeros(())
        for name in V14_EXTRA_DIAGNOSTICS:
            diagnostics[name] = zero
        return ObjectiveOutput(terms, diagnostics)

    def trajectory(
        self,
        hidden,
        logits,
        labels,
        loss_mask,
        token_roles,
        word_embedding_weight,
        batch,
        *,
        frontend_residual_rms,
    ) -> ObjectiveOutput:
        # Exactly one full-vocabulary CE graph is shared by the original
        # runtime terms and the new corrupted-prefix recovery term.
        anchor = _parameter_anchor(self, logits)
        active = loss_mask > 0
        token_losses = token_cross_entropy_values(logits, labels)

        main_weights, legacy_boundary_weights = trajectory_token_weights(
            labels, token_roles, loss_mask
        )
        lm_trajectory = values_to_term(token_losses, main_weights)
        semantic = values_to_term(
            token_losses, (token_roles == ROLE_SEMANTIC).float() * active
        )
        legacy_boundary = values_to_term(token_losses, legacy_boundary_weights)

        text_mask = (token_roles == ROLE_TEXT) & active
        text_weights = _balanced_example_weights(
            labels,
            text_mask,
            classes=logits.shape[-1],
            minimum=0.5,
            maximum=4.0,
        )
        text_term = values_to_term(token_losses, text_weights)

        boundary_mask = (token_roles == ROLE_BOUNDARY) & active
        boundary_weights = boundary_mask.float()
        boundary_weights = torch.where(
            labels == c.TOKEN_END_CONTENT,
            boundary_weights * 4.0,
            boundary_weights,
        )
        boundary_weights = torch.where(
            labels == c.TOKEN_END_SEMANTIC,
            boundary_weights * 2.0,
            boundary_weights,
        )
        boundary_weights = torch.where(
            labels == c.TOKEN_EOS,
            boundary_weights * 4.0,
            boundary_weights,
        )
        boundary_term = values_to_term(token_losses, boundary_weights)

        recovery_mask = batch.get("predicted_prefix_recovery_mask")
        if recovery_mask is None:
            recovery_mask = torch.zeros_like(batch["loss_mask"], dtype=torch.bool)
        recovery_mask = recovery_mask.reshape(-1).bool() & active
        recovery_term = values_to_term(token_losses, recovery_mask.float())

        original_seq_length = int(batch["original_seq_length"].item())
        action_flat = (
            batch["action_batch"].long() * original_seq_length
            + batch["action_position"].long()
        )
        action_logits = self.action_head(hidden[action_flat])
        action_targets = batch["natural_action"].long()
        action_weight = action_logits.new_tensor([1.0, self.action_write_weight])
        action_losses = F.cross_entropy(
            action_logits.float(),
            action_targets,
            weight=action_weight.float(),
            reduction="none",
        )
        action_term = values_to_term(
            action_losses, torch.ones_like(action_losses, dtype=torch.float32)
        )
        interleaved = LossTerm(
            lm_trajectory.mean + action_term.mean,
            lm_trajectory.denominator.new_ones(()),
        )

        source_summary = hidden[action_flat]
        support_logits = self.support_head(source_summary)
        support_losses = F.cross_entropy(
            support_logits.float(), batch["support_bucket"].long(), reduction="none"
        )
        support = values_to_term(support_losses, torch.ones_like(support_losses))

        target_ids = batch["translation_ids"].long()
        target_hidden = F.embedding(target_ids, word_embedding_weight)
        safe_logits = self.safe_commit_head(source_summary, target_hidden)
        safe_mask = batch["translation_mask"].bool()
        safe_losses = F.binary_cross_entropy_with_logits(
            safe_logits.float(), batch["safe_commit_targets"].float(), reduction="none"
        )
        probability = torch.sigmoid(safe_logits.float())
        safe_target = batch["safe_commit_targets"].float()
        pt = torch.where(safe_target > 0.5, probability, 1.0 - probability)
        safe_alpha = torch.where(
            safe_target > 0.5,
            torch.full_like(safe_target, self.safe_positive_alpha),
            torch.full_like(safe_target, 1.0 - self.safe_positive_alpha),
        )
        safe = values_to_term(
            safe_alpha * (1.0 - pt).square() * safe_losses, safe_mask
        )
        deadline = grouped_deadline_survival_term(
            action_logits,
            batch["sample_group"],
            batch["chunk_end_ms"],
            batch["soft_deadline_ms"],
            batch["hard_deadline_ms"],
            batch.get("deadline_loss_enabled"),
        )
        kd, stability = self._teacher_terms(logits, batch, anchor)
        microblock = self.semantic_microblock_head.training_output(
            hidden, labels, token_roles, loss_mask, word_embedding_weight
        )

        prediction = logits.float().argmax(dim=-1)
        end_content_mask = (labels == c.TOKEN_END_CONTENT) & active
        terms = OrderedDict(
            (
                ("phase3_replay", zero_term(anchor)),
                ("interleaved_trajectory", interleaved),
                ("real_prefix_kd", kd),
                ("support_ordinal", support),
                ("token_safe_commit", safe),
                ("deadline_survival", deadline),
                ("prefix_stability", stability),
                ("ar_semantic_microblock", semantic),
                ("speaker_consistency", zero_term(anchor)),
                ("boundary_continuity", legacy_boundary),
                ("microblock_semantic_content", microblock.content_term),
                ("microblock_final_length", microblock.final_length_term),
                ("microblock_continue", microblock.continue_term),
                ("runtime_text_content", text_term),
                ("runtime_critical_boundary", boundary_term),
                ("runtime_action", action_term),
                ("runtime_prefix_recovery", recovery_term),
            )
        )
        support_prediction = support_logits.argmax(dim=-1)
        precision, recall, f1 = _binary_metrics(
            safe_logits, batch["safe_commit_targets"], safe_mask
        )
        diagnostics = OrderedDict(
            (
                (
                    "support_accuracy",
                    (support_prediction == batch["support_bucket"]).float().mean(),
                ),
                (
                    "support_mae",
                    (support_prediction - batch["support_bucket"])
                    .abs()
                    .float()
                    .mean(),
                ),
                ("safe_commit_precision", precision),
                ("safe_commit_recall", recall),
                ("safe_commit_f1", f1),
                (
                    "predicted_write_fraction",
                    (action_logits.argmax(dim=-1) == 1).float().mean(),
                ),
                ("natural_write_fraction", action_targets.float().mean()),
                ("deadline_forced_fraction", batch["deadline_forced"].float().mean()),
                ("frontend_residual_rms", frontend_residual_rms.float()),
                ("supervised_tokens", main_weights.sum().detach().float()),
                ("microblock_token_accuracy", microblock.token_accuracy),
                ("microblock_first_slot_accuracy", microblock.first_slot_accuracy),
                (
                    "microblock_final_length_accuracy",
                    microblock.final_length_accuracy,
                ),
                ("microblock_final_length_mae", microblock.final_length_mae),
                ("microblock_continue_accuracy", microblock.continue_accuracy),
                (
                    "microblock_predicted_continue_fraction",
                    microblock.predicted_continue_fraction,
                ),
                (
                    "microblock_target_continue_fraction",
                    microblock.target_continue_fraction,
                ),
                (
                    "microblock_predicted_unique_fraction",
                    microblock.predicted_unique_fraction,
                ),
                ("microblock_blocks", microblock.blocks),
                (
                    "runtime_text_token_accuracy",
                    _masked_accuracy(prediction, labels, text_mask, logits),
                ),
                (
                    "runtime_boundary_accuracy",
                    _masked_accuracy(prediction, labels, boundary_mask, logits),
                ),
                (
                    "runtime_end_content_accuracy",
                    _masked_accuracy(prediction, labels, end_content_mask, logits),
                ),
                (
                    "runtime_action_accuracy",
                    (action_logits.float().argmax(dim=-1) == action_targets)
                    .float()
                    .mean(),
                ),
                ("runtime_text_supervised_tokens", text_mask.sum().detach().float()),
                (
                    "runtime_prefix_corruption_fraction",
                    batch["prefix_corruption_fraction"].detach().float(),
                ),
                (
                    "runtime_prefix_recovery_accuracy",
                    _masked_accuracy(prediction, labels, recovery_mask, logits),
                ),
                (
                    "runtime_prefix_rollout_rounds",
                    batch["prefix_rollout_rounds"].detach().float(),
                ),
            )
        )
        if tuple(terms) != V14_TERM_NAMES:
            raise AssertionError("generalize14 trajectory term order changed")
        if tuple(diagnostics) != V14_DIAGNOSTIC_NAMES:
            raise AssertionError("generalize14 trajectory diagnostic order changed")
        return ObjectiveOutput(terms, diagnostics)


def distributed_generalize14_objective(output: ObjectiveOutput, *, progress: float):
    del progress
    if tuple(output.terms) != V14_TERM_NAMES:
        raise ValueError("generalize14 objective term order changed")
    if tuple(output.diagnostics) != V14_DIAGNOSTIC_NAMES:
        raise ValueError("generalize14 diagnostic order changed")
    numerators = torch.stack(
        [output.terms[name].numerator for name in V14_TERM_NAMES]
    )
    denominators = torch.stack(
        [output.terms[name].denominator.to(numerators.dtype) for name in V14_TERM_NAMES]
    )
    global_numerators = numerators.detach().clone()
    global_denominators = denominators.detach().clone()
    world_size = 1
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(global_numerators)
        dist.all_reduce(global_denominators)
        world_size = dist.get_world_size()
    active = global_denominators > 0
    local_means = torch.where(
        active,
        world_size * numerators / global_denominators.clamp_min(1),
        numerators * 0.0,
    )
    total = (local_means * numerators.new_tensor(list(V14_WEIGHTS.values()))).sum()
    global_means = torch.where(
        active,
        global_numerators / global_denominators.clamp_min(1),
        global_numerators * 0.0,
    )
    metrics = OrderedDict(
        (name, global_means[index]) for index, name in enumerate(V14_TERM_NAMES)
    )
    diagnostics = torch.stack(
        [output.diagnostics[name].detach().float() for name in V14_DIAGNOSTIC_NAMES]
    )
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(diagnostics)
        diagnostics /= dist.get_world_size()
    metrics.update(
        (name, diagnostics[index])
        for index, name in enumerate(V14_DIAGNOSTIC_NAMES)
    )
    metrics["curriculum_deadline_weight"] = total.detach().new_tensor(
        V14_WEIGHTS["deadline_survival"]
    )
    metrics["curriculum_replay_fraction"] = total.detach().new_tensor(REPLAY_FRACTION)
    metrics["curriculum_frontend_lr_multiplier"] = total.detach().new_zeros(())
    return total, metrics


def _probe_output_processor(**kwargs) -> torch.Tensor:
    context = kwargs["context"]
    hidden = kwargs["hidden_states"]
    if hidden.ndim != 3 or hidden.shape[1] != 1:
        raise ValueError("generalize14 probe expects flattened [tokens,1,hidden]")
    hidden = hidden[:, 0]
    batch = context["batch"]
    probe = probe_runtime_prefix_labels(
        hidden,
        kwargs["labels"].reshape(-1),
        batch["token_roles"].reshape(-1),
        kwargs["loss_mask"].reshape(-1),
        context["word_embedding_weight"],
        context["objective"].semantic_microblock_head,
    )
    return torch.stack((probe.labels.float(), probe.eligible.float()))


def dense_output_processor(**kwargs) -> torch.Tensor:
    context = kwargs["context"]
    batch = context["batch"]
    if bool(batch.get("runtime_prefix_probe", False)):
        return _probe_output_processor(**kwargs)

    objective = context["objective"]
    hidden = kwargs["hidden_states"]
    logits, _ = kwargs["output_layer"](
        hidden,
        weight=kwargs["output_weight"],
        runtime_gather_output=kwargs["runtime_gather_output"],
    )
    logits = kwargs["scale_logits"](logits)
    if hidden.ndim != 3 or hidden.shape[1] != 1 or logits.shape[1] != 1:
        raise ValueError("generalize14 TP=PP=1 expects flattened [tokens,1,*]")
    hidden = hidden[:, 0]
    logits = logits[:, 0]
    labels = kwargs["labels"].reshape(-1)
    loss_mask = kwargs["loss_mask"].reshape(-1)
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
        raise ValueError(
            f"unknown generalize14 sample kind: {context['sample_kind']}"
        )
    total, metrics = distributed_generalize14_objective(
        output, progress=float(context["progress"])
    )
    if tuple(metrics) != V14_METRIC_NAMES:
        raise AssertionError("generalize14 metric order changed")
    values = (total.float(), *[metrics[name].float() for name in V14_METRIC_NAMES])
    if not all(torch.isfinite(value).all() for value in values):
        raise FloatingPointError("non-finite generalize14 loss component")
    return torch.stack(values)


def _model_training(model) -> bool:
    return bool(getattr(model, "training", False))


def forward_step(data_iterator, model):
    runtime = dense.base.load_megatron_runtime()
    args = runtime.megatron_gpt.get_args()
    batch = dense.base.prepare_packed_batch(next(data_iterator), int(args.seq_length))
    denominator = max(1, int(args.train_iters) * int(args.global_batch_size))
    consumed = int(getattr(args, "consumed_train_samples", 0) or 0)
    progress = min(1.0, max(0.0, consumed / denominator))
    batch["training_progress"] = torch.tensor(
        progress, dtype=torch.float32, device=batch["tokens"].device
    )
    packed_seq_params = dense.base.build_packed_seq_params(
        batch, int(args.seq_length)
    )

    tokens = batch["tokens"]
    cumulative_corruption = torch.zeros_like(tokens, dtype=torch.bool)
    schedule = prefix_schedule(progress)
    is_trajectory = str(batch["sample_kind"]) == "trajectory"
    if _model_training(model) and is_trajectory:
        for _ in range(schedule.rounds):
            probe_batch = dict(batch)
            probe_batch["tokens"] = tokens
            probe_batch["runtime_prefix_probe"] = True
            with torch.no_grad():
                probe_output = model(
                    tokens,
                    batch["position_ids"],
                    None,
                    labels=batch["labels"],
                    loss_mask=batch["loss_mask"],
                    packed_seq_params=packed_seq_params,
                    true_subsecond_batch=probe_batch,
                )
            if probe_output.ndim != 2 or probe_output.shape[0] != 2:
                raise ValueError("generalize14 prefix probe returned invalid geometry")
            predicted = probe_output[0].round().long().reshape_as(tokens)
            eligible = probe_output[1].reshape_as(tokens) > 0.5
            tokens, corrupted = apply_prefix_predictions(
                tokens,
                predicted,
                eligible,
                probability=schedule.probability,
            )
            cumulative_corruption |= corrupted

    batch["tokens"] = tokens
    batch["predicted_prefix_recovery_mask"] = expand_recovery_mask(
        cumulative_corruption, horizon=8
    )
    if is_trajectory:
        eligible_count = (
            (
                (batch["token_roles"] == ROLE_TEXT)
                | (batch["token_roles"] == ROLE_SEMANTIC)
            )
            & (batch["loss_mask"] > 0)
        ).sum()
        corruption_fraction = (
            cumulative_corruption.sum().float()
            / eligible_count.float().clamp_min(1.0)
        )
    else:
        corruption_fraction = cumulative_corruption.sum().float()
    batch["prefix_corruption_fraction"] = corruption_fraction
    batch["prefix_rollout_rounds"] = torch.tensor(
        schedule.rounds if _model_training(model) else 0,
        dtype=torch.float32,
        device=tokens.device,
    )

    output = model(
        batch["tokens"],
        batch["position_ids"],
        None,
        labels=batch["labels"],
        loss_mask=batch["loss_mask"],
        packed_seq_params=packed_seq_params,
        true_subsecond_batch=batch,
    )
    return output, dense.base.loss_func


def freeze_base_and_train_runtime_joint() -> None:
    original = dense.base.augment_native_gpt

    def augment_and_freeze(model, args):
        summary = original(model, args)
        trainable = 0
        for name, parameter in model.named_parameters():
            keep = is_generalize13_trainable_parameter(name)
            parameter.requires_grad_(keep)
            if keep:
                trainable += parameter.numel()
        if trainable <= 0:
            raise RuntimeError("generalize14 found no joint runtime parameters")
        model._generalize14_trainable_parameters = trainable
        return summary

    dense.base.augment_native_gpt = augment_and_freeze


def main() -> None:
    v2.trajectory_token_weights = trajectory_token_weights
    dense.base.TrueSubsecondObjective = RuntimeParityGeneralize14Objective
    dense._distributed_dense_objective = distributed_generalize14_objective
    dense._dense_output_processor = dense_output_processor
    dense.METRIC_NAMES = V14_METRIC_NAMES
    dense.base.METRIC_NAMES = V14_METRIC_NAMES
    dense.JointValidationDataset = SynchronizedValidationDataset
    dense.base.forward_step = forward_step
    freeze_base_and_train_runtime_joint()
    dense.main()


if __name__ == "__main__":
    main()

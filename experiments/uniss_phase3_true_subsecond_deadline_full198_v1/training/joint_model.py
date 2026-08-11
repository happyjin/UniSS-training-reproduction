"""Native-Megatron objective modules for the true-subsecond full198 run."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Mapping

import torch
import torch.distributed as dist
from safetensors import safe_open
from torch import nn
from torch.nn import functional as F

from experiments.uniss_phase3_true_subsecond_deadline_full198_v1.data.trajectory_packing import (
    ROLE_ACTION,
    ROLE_BOUNDARY,
    ROLE_SEMANTIC,
    ROLE_TEXT,
)
from experiments.uniss_phase3_true_subsecond_deadline_full198_v1.model import (
    ActionHead,
    ChunkCausalWhisperVQAdapter,
    SafeCommitHead,
    SupportOrdinalHead,
)
from experiments.uniss_phase3_true_subsecond_deadline_full198_v1.training.curriculum import (
    point_for_progress,
)
from experiments.uniss_phase3_true_subsecond_deadline_full198_v1.training.losses import (
    LossTerm,
    LossWeights,
    grouped_deadline_survival_term,
    restricted_symmetric_topk_term,
    token_cross_entropy_term,
    token_cross_entropy_values,
    topk_teacher_kl_term,
    values_to_term,
    zero_term,
)


TERM_NAMES = (
    "phase3_replay",
    "interleaved_trajectory",
    "real_prefix_kd",
    "support_ordinal",
    "token_safe_commit",
    "deadline_survival",
    "prefix_stability",
    "ar_semantic_microblock",
    "speaker_consistency",
    "boundary_continuity",
)

DIAGNOSTIC_NAMES = (
    "support_accuracy",
    "support_mae",
    "safe_commit_precision",
    "safe_commit_recall",
    "safe_commit_f1",
    "predicted_write_fraction",
    "natural_write_fraction",
    "deadline_forced_fraction",
    "frontend_residual_rms",
    "supervised_tokens",
)


def load_whispervq_codebook(path: str | Path) -> torch.Tensor:
    path = Path(path)
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        value = handle.get_tensor("codebook.weight").float()
    if tuple(value.shape) != (16_384, 1_280):
        raise ValueError(f"unexpected WhisperVQ codebook shape: {tuple(value.shape)}")
    return value


def _parameter_anchor(module: nn.Module, reference: torch.Tensor) -> torch.Tensor:
    value = reference.sum() * 0.0
    for parameter in module.parameters():
        if parameter.requires_grad and parameter.numel():
            value = value + parameter.reshape(-1)[0] * 0.0
    return value


def _mean_term(terms: list[tuple[LossTerm, float]], anchor: torch.Tensor) -> LossTerm:
    active = [(term, weight) for term, weight in terms if float(term.denominator) > 0]
    if not active:
        return zero_term(anchor)
    values = [term.mean * weight for term, weight in active]
    denominator = sum(weight for _, weight in active)
    return LossTerm(torch.stack(values).sum(), anchor.new_tensor(denominator))


def _binary_metrics(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor):
    predicted = torch.sigmoid(logits.float()) >= 0.5
    target = targets.bool()
    active = mask.bool()
    tp = (predicted & target & active).sum().float()
    fp = (predicted & ~target & active).sum().float()
    fn = (~predicted & target & active).sum().float()
    precision = tp / (tp + fp).clamp_min(1)
    recall = tp / (tp + fn).clamp_min(1)
    f1 = 2 * precision * recall / (precision + recall).clamp_min(1e-8)
    return precision, recall, f1


@dataclass(frozen=True)
class ObjectiveOutput:
    terms: OrderedDict[str, LossTerm]
    diagnostics: OrderedDict[str, torch.Tensor]


class TrueSubsecondObjective(nn.Module):
    """New causal frontend and policy heads around the native Phase3 decoder."""

    def __init__(
        self,
        hidden_size: int,
        codebook_weight: torch.Tensor,
        *,
        adapter_layers: int = 4,
        adapter_kernel_size: int = 5,
        adapter_expansion: int = 2,
        adapter_dropout: float = 0.0,
        kd_temperature: float = 1.5,
        action_write_weight: float = 1.0,
        safe_positive_alpha: float = 0.5,
    ) -> None:
        super().__init__()
        if codebook_weight.ndim != 2 or codebook_weight.shape[1] != 1280:
            raise ValueError("WhisperVQ codebook must have shape [codes,1280]")
        self.codebook = nn.Embedding.from_pretrained(codebook_weight.float(), freeze=True)
        self.frontend_adapter = ChunkCausalWhisperVQAdapter(
            1280,
            layers=adapter_layers,
            kernel_size=adapter_kernel_size,
            expansion=adapter_expansion,
            dropout=adapter_dropout,
        )
        self.frontend_projection = nn.Linear(1280, hidden_size, bias=False)
        nn.init.zeros_(self.frontend_projection.weight)
        self.support_head = SupportOrdinalHead(hidden_size)
        self.action_head = ActionHead(hidden_size)
        self.safe_commit_head = SafeCommitHead(hidden_size)
        self.kd_temperature = float(kd_temperature)
        if action_write_weight <= 0:
            raise ValueError("action_write_weight must be positive")
        if not 0.0 < safe_positive_alpha < 1.0:
            raise ValueError("safe_positive_alpha must be in (0,1)")
        self.action_write_weight = float(action_write_weight)
        self.safe_positive_alpha = float(safe_positive_alpha)
        for parameter in self.frontend_adapter.parameters():
            parameter.uniss_lr_frontend = True
        self.frontend_projection.weight.uniss_lr_frontend = True
        for module in (self.support_head, self.action_head, self.safe_commit_head):
            for parameter in module.parameters():
                parameter.uniss_lr_new_heads = True

    def inject_frontend_residual(
        self,
        decoder_input: torch.Tensor,
        batch: Mapping[str, torch.Tensor],
        *,
        original_seq_length: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Add a zero-initialized causal correction without changing Phase3 at step 0."""

        if decoder_input.ndim != 3 or decoder_input.shape[1] != 1:
            raise ValueError("packed decoder_input must be [tokens,1,hidden]")
        ids = batch["frontend_ids"].long()
        mask = batch["frontend_mask"].bool()
        positions = batch["frontend_positions"].long()
        rows = batch["action_batch"].long()
        hidden = self.frontend_adapter(self.codebook(ids))
        residual = self.frontend_projection(hidden)
        valid = mask.nonzero(as_tuple=False)
        if not len(valid):
            return decoder_input, residual.sum() * 0.0
        annotation, time = valid[:, 0], valid[:, 1]
        flattened = rows[annotation] * int(original_seq_length) + positions[annotation, time]
        if int(flattened.max()) >= decoder_input.shape[0]:
            raise ValueError("frontend position exceeds flattened packed sequence")
        corrected = decoder_input.clone()
        corrected[:, 0].index_add_(
            0, flattened, residual[annotation, time].to(corrected.dtype)
        )
        rms = residual[mask].float().square().mean().sqrt()
        return corrected, rms

    def _teacher_terms(
        self,
        logits: torch.Tensor,
        batch: Mapping[str, torch.Tensor],
        anchor: torch.Tensor,
    ) -> tuple[LossTerm, LossTerm]:
        if not batch["kd_position"].numel():
            return zero_term(anchor), zero_term(anchor)
        flat_position = (
            batch["kd_batch"].long() * int(batch["original_seq_length"].item())
            + batch["kd_position"].long()
        )
        annotation = batch["kd_annotation"].long()
        target = batch["kd_target_index"].long()
        max_teacher = batch["teacher_indices"].shape[2]
        valid = target < max_teacher
        if not bool(valid.any()):
            return zero_term(anchor), zero_term(anchor)
        flat_position, annotation, target = (
            flat_position[valid],
            annotation[valid],
            target[valid],
        )
        student = logits[flat_position]
        indices = batch["teacher_indices"][annotation, :, target]
        probabilities = batch["teacher_probabilities"][annotation, :, target]
        teacher_mask = batch["teacher_mask"][annotation, :, target]
        kd = _mean_term(
            [
                (
                    topk_teacher_kl_term(
                        student,
                        indices[:, view],
                        probabilities[:, view],
                        teacher_mask[:, view],
                        temperature=self.kd_temperature,
                    ),
                    weight,
                )
                for view, weight in enumerate((0.50, 0.125, 0.125, 0.25))
            ],
            anchor,
        )
        stability = _mean_term(
            [
                (
                    restricted_symmetric_topk_term(
                        student,
                        indices[:, view],
                        probabilities[:, view],
                        teacher_mask[:, view],
                    ),
                    1.0,
                )
                for view in (1, 2, 3)
            ],
            anchor,
        )
        return kd, stability

    def replay(
        self, logits: torch.Tensor, labels: torch.Tensor, loss_mask: torch.Tensor
    ) -> ObjectiveOutput:
        anchor = _parameter_anchor(self, logits)
        replay = token_cross_entropy_term(logits, labels, loss_mask)
        terms = OrderedDict((name, zero_term(anchor)) for name in TERM_NAMES)
        terms["phase3_replay"] = replay
        zero = anchor.detach().new_zeros(())
        diagnostics = OrderedDict((name, zero) for name in DIAGNOSTIC_NAMES)
        diagnostics["supervised_tokens"] = replay.denominator.detach().float()
        return ObjectiveOutput(terms, diagnostics)

    def trajectory(
        self,
        hidden: torch.Tensor,
        logits: torch.Tensor,
        labels: torch.Tensor,
        loss_mask: torch.Tensor,
        token_roles: torch.Tensor,
        word_embedding_weight: torch.Tensor,
        batch: Mapping[str, torch.Tensor],
        *,
        frontend_residual_rms: torch.Tensor,
    ) -> ObjectiveOutput:
        anchor = _parameter_anchor(self, logits)
        role_weights = torch.zeros_like(loss_mask, dtype=torch.float32)
        role_weights = torch.where(token_roles == ROLE_ACTION, 4.0, role_weights)
        role_weights = torch.where(token_roles == ROLE_TEXT, 2.0, role_weights)
        role_weights = torch.where(token_roles == ROLE_SEMANTIC, 1.0, role_weights)
        role_weights = torch.where(token_roles == ROLE_BOUNDARY, 1.0, role_weights)
        role_weights = role_weights * (loss_mask > 0)
        # All three terms use the identical next-token CE.  Share the
        # unreduced values so seq=18000/MBS=2 does not retain three ~24 GiB
        # float32 CE graphs on every H200.
        token_losses = token_cross_entropy_values(logits, labels)
        lm_trajectory = values_to_term(token_losses, role_weights)
        semantic = values_to_term(
            token_losses, (token_roles == ROLE_SEMANTIC).float() * (loss_mask > 0)
        )
        boundary = values_to_term(
            token_losses, (token_roles == ROLE_BOUNDARY).float() * (loss_mask > 0)
        )

        original_seq_length = int(batch["original_seq_length"].item())
        action_flat = (
            batch["action_batch"].long() * original_seq_length
            + batch["action_position"].long()
        )
        source_summary = hidden[action_flat]
        support_logits = self.support_head(source_summary)
        support_losses = F.cross_entropy(
            support_logits.float(), batch["support_bucket"].long(), reduction="none"
        )
        support = values_to_term(support_losses, torch.ones_like(support_losses))
        action_logits = self.action_head(source_summary)
        action_weight = action_logits.new_tensor([1.0, self.action_write_weight])
        action_losses = F.cross_entropy(
            action_logits.float(),
            batch["natural_action"].long(),
            weight=action_weight.float(),
            reduction="none",
        )
        action = values_to_term(action_losses, torch.ones_like(action_losses))
        interleaved = LossTerm(
            lm_trajectory.mean + action.mean,
            lm_trajectory.denominator.new_ones(()),
        )

        target_ids = batch["translation_ids"].long()
        target_hidden = F.embedding(target_ids, word_embedding_weight)
        safe_logits = self.safe_commit_head(source_summary, target_hidden)
        safe_mask = batch["translation_mask"].bool()
        safe_losses = F.binary_cross_entropy_with_logits(
            safe_logits.float(), batch["safe_commit_targets"].float(), reduction="none"
        )
        probability = torch.sigmoid(safe_logits.float())
        target = batch["safe_commit_targets"].float()
        pt = torch.where(target > 0.5, probability, 1.0 - probability)
        safe_alpha = torch.where(
            target > 0.5,
            torch.full_like(target, self.safe_positive_alpha),
            torch.full_like(target, 1.0 - self.safe_positive_alpha),
        )
        safe = values_to_term(safe_alpha * (1.0 - pt).square() * safe_losses, safe_mask)

        deadline = grouped_deadline_survival_term(
            action_logits,
            batch["sample_group"],
            batch["chunk_end_ms"],
            batch["soft_deadline_ms"],
            batch["hard_deadline_ms"],
            batch.get("deadline_loss_enabled"),
        )
        kd, stability = self._teacher_terms(logits, batch, anchor)
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
                # Speaker consistency is intentionally disabled in v1.  Do not
                # anchor its zero placeholder to frontend_residual_rms: the
                # projection is zero-initialized, and d(sqrt(x))/dx at x=0 can
                # turn an otherwise harmless ``0 * rms`` branch into NaN
                # gradients.  The regular parameter anchor keeps every new
                # module in the graph while producing an exactly finite zero
                # gradient.
                ("speaker_consistency", zero_term(anchor)),
                ("boundary_continuity", boundary),
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
                    (support_prediction - batch["support_bucket"]).abs().float().mean(),
                ),
                ("safe_commit_precision", precision),
                ("safe_commit_recall", recall),
                ("safe_commit_f1", f1),
                (
                    "predicted_write_fraction",
                    (action_logits.argmax(dim=-1) == 1).float().mean(),
                ),
                ("natural_write_fraction", batch["natural_action"].float().mean()),
                ("deadline_forced_fraction", batch["deadline_forced"].float().mean()),
                ("frontend_residual_rms", frontend_residual_rms.float()),
                ("supervised_tokens", lm_trajectory.denominator.detach().float()),
            )
        )
        return ObjectiveOutput(terms, diagnostics)


def distributed_weighted_objective(
    output: ObjectiveOutput,
    *,
    progress: float,
    base_weights: LossWeights | None = None,
) -> tuple[torch.Tensor, OrderedDict[str, torch.Tensor]]:
    if tuple(output.terms) != TERM_NAMES:
        raise ValueError("joint objective term order changed")
    point = point_for_progress(progress)
    weights = replace(base_weights or LossWeights(), deadline=point.deadline_weight)
    weight_map = OrderedDict(
        (
            ("phase3_replay", weights.replay),
            ("interleaved_trajectory", weights.trajectory),
            ("real_prefix_kd", weights.real_prefix_kd),
            ("support_ordinal", weights.support),
            ("token_safe_commit", weights.safe_commit),
            ("deadline_survival", weights.deadline),
            ("prefix_stability", weights.stability),
            ("ar_semantic_microblock", weights.semantic),
            ("speaker_consistency", weights.speaker),
            ("boundary_continuity", weights.boundary),
        )
    )
    numerators = torch.stack([output.terms[name].numerator for name in TERM_NAMES])
    denominators = torch.stack(
        [output.terms[name].denominator.to(numerators.dtype) for name in TERM_NAMES]
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
    scales = numerators.new_tensor(list(weight_map.values()))
    total = (local_means * scales).sum()
    global_means = torch.where(
        active,
        global_numerators / global_denominators.clamp_min(1),
        global_numerators * 0.0,
    )
    metrics = OrderedDict(
        (name, global_means[index]) for index, name in enumerate(TERM_NAMES)
    )
    diagnostics = torch.stack([value.detach().float() for value in output.diagnostics.values()])
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(diagnostics)
        diagnostics /= dist.get_world_size()
    metrics.update(
        (name, diagnostics[index]) for index, name in enumerate(output.diagnostics)
    )
    metrics["curriculum_deadline_weight"] = total.detach().new_tensor(point.deadline_weight)
    metrics["curriculum_replay_fraction"] = total.detach().new_tensor(point.replay_fraction)
    metrics["curriculum_frontend_lr_multiplier"] = total.detach().new_tensor(
        point.frontend_lr_multiplier
    )
    return total, metrics


__all__ = [
    "DIAGNOSTIC_NAMES",
    "ObjectiveOutput",
    "TERM_NAMES",
    "TrueSubsecondObjective",
    "distributed_weighted_objective",
    "load_whispervq_codebook",
]

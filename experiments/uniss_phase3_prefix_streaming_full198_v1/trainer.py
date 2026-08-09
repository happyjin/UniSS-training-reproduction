#!/usr/bin/env python3
"""Megatron-orchestrated full198 Phase3 prefix-streaming joint training."""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM


ROOT = Path(__file__).resolve().parents[2]
MEGATRON_ROOT = ROOT / "third_party" / "Megatron-LM"
for value in (ROOT, MEGATRON_ROOT):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from experiments.uniss_phase3_prefix_streaming_full198_v1 import builders  # noqa: E402
from experiments.uniss_phase3_prefix_streaming_full198_v1.curriculum import (  # noqa: E402
    choose_prefix_pair,
    choose_semantic_geometry,
    choose_task,
    point_for_iteration,
    stable_uniform,
)
from experiments.uniss_phase3_prefix_streaming_full198_v1.data import (  # noqa: E402
    Full198CurriculumDataset,
    UniSTDevDataset,
)
from experiments.uniss_phase3_prefix_streaming_full198_v1.lora import (  # noqa: E402
    inject_lora,
    lora_enabled,
    lora_update_rms,
    set_lora_training,
)
from training import constants_uniss as c  # noqa: E402
from training.megatron_uniss_dataset import RepeatToLengthDataset  # noqa: E402
from training.pretrain_uniss_megatron import load_megatron_runtime  # noqa: E402


METRIC_NAMES = (
    "loss/replay_ce",
    "loss/prefix_ce",
    "loss/semantic_ce",
    "loss/commit_suffix_ce",
    "loss/teacher_kl",
    "loss/adjacent_consistency",
    "loss/action_ce",
    "loss/boundary_eos",
    "task/replay_fraction",
    "task/prefix_fraction",
    "task/semantic_fraction",
    "task/commit_fraction",
    "direction/en_zh_fraction",
    "direction/zh_en_fraction",
    "stream/prefix_ratio_mean",
    "stream/teacher_confidence",
    "stream/long_confidence",
    "stream/stable_tokens_mean",
    "stream/write_target_fraction",
    "tokens/supervised",
    "tokens/teacher",
    "lora/update_rms",
)


def patch_unused_megatron_dataset_helper_compile() -> None:
    """Skip the legacy indexed-dataset extension unused by this parquet dataset.

    Megatron compiles ``helpers_cpp`` during every startup even when a custom
    dataset provider never imports it.  The replacement is intentionally local
    to this process and leaves the shared third-party checkout untouched.
    """

    from megatron.core.datasets import utils as dataset_utils

    def no_op() -> None:
        if torch.distributed.get_rank() == 0:
            print("> skipping unused Megatron indexed-dataset helper compilation", flush=True)

    dataset_utils.compile_helpers = no_op


@dataclass
class Descriptor:
    record: dict[str, object]
    task: str
    short_ratio: float
    long_ratio: float
    primary: builders.TokenSample
    teacher: builders.TokenSample | None = None
    long: builders.TokenSample | None = None
    action_prompt: list[int] | None = None


@dataclass
class DistributionSummary:
    indices: torch.Tensor
    probabilities: torch.Tensor
    prediction: torch.Tensor
    confidence: torch.Tensor


def add_experiment_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    group = parser.add_argument_group(title="Phase3 full198 prefix-streaming experiment")
    group.add_argument("--experiment-index-json", required=True)
    group.add_argument("--experiment-valid-parquet", required=True)
    group.add_argument("--experiment-phase3-model", required=True)
    group.add_argument("--experiment-valid-limit", type=int, default=1024)
    group.add_argument("--experiment-block-size", type=int, default=4096)
    group.add_argument("--experiment-cache-shards", type=int, default=2)
    group.add_argument("--experiment-lora-rank", type=int, default=16)
    group.add_argument("--experiment-lora-alpha", type=float, default=32.0)
    group.add_argument("--experiment-lora-dropout", type=float, default=0.05)
    group.add_argument("--experiment-lora-targets", default="q_proj,v_proj")
    group.add_argument("--experiment-teacher-topk", type=int, default=32)
    group.add_argument("--experiment-teacher-temperature", type=float, default=1.5)
    group.add_argument("--experiment-confidence-threshold", type=float, default=0.70)
    group.add_argument("--experiment-min-write-tokens", type=int, default=2)
    group.add_argument("--experiment-history-tokens", type=int, default=200)
    group.add_argument("--experiment-max-sample-tokens", type=int, default=4096)
    group.add_argument("--experiment-teacher-kl-weight", type=float, default=0.25)
    group.add_argument("--experiment-semantic-kl-weight", type=float, default=0.20)
    group.add_argument("--experiment-consistency-weight", type=float, default=0.20)
    group.add_argument("--experiment-commit-consistency-weight", type=float, default=0.25)
    group.add_argument("--experiment-boundary-weight", type=float, default=0.10)
    group.add_argument("--experiment-action-weight", type=float, default=1.0)
    group.add_argument("--experiment-attention-implementation", default="flash_attention_2")
    group.add_argument("--experiment-disable-gradient-checkpointing", action="store_true")
    group.add_argument("--experiment-smoke", action="store_true")
    return parser


def validate_experiment_args(args) -> None:
    for name in ("experiment_index_json", "experiment_valid_parquet", "experiment_phase3_model"):
        if not Path(getattr(args, name)).exists():
            raise FileNotFoundError(getattr(args, name))
    if int(args.tensor_model_parallel_size) != 1 or int(args.pipeline_model_parallel_size) != 1:
        raise ValueError("this isolated HF-composite experiment requires TP=PP=1")
    if int(args.micro_batch_size) <= 0 or int(args.global_batch_size) <= 0:
        raise ValueError("batch sizes must be positive")
    if int(args.seq_length) != 18000:
        raise ValueError("the full198 experiment requires seq-length 18000")
    if int(args.train_iters) != 12000 and not args.experiment_smoke:
        raise ValueError("the formal single run requires train-iters 12000")
    if int(args.experiment_teacher_topk) <= 1:
        raise ValueError("teacher top-k must exceed one")
    if not 0.0 < float(args.experiment_confidence_threshold) < 1.0:
        raise ValueError("confidence threshold must be in (0,1)")
    if int(args.experiment_max_sample_tokens) > int(args.seq_length):
        raise ValueError("max sample tokens cannot exceed seq-length")


def train_valid_test_datasets_provider(train_val_test_num_samples, vp_stage=None):
    del vp_stage
    runtime = load_megatron_runtime()
    args = runtime.megatron_gpt.get_args()
    runtime.print_rank_0("> building independent full198 curriculum datasets ...")
    train = Full198CurriculumDataset(
        args.experiment_index_json,
        args.experiment_phase3_model,
        block_size=int(args.experiment_block_size),
        seed=int(args.seed),
        cache_shards=int(args.experiment_cache_shards),
    )
    valid = UniSTDevDataset(
        args.experiment_valid_parquet,
        args.experiment_phase3_model,
        limit=int(args.experiment_valid_limit),
    )
    valid_rows = len(valid)
    valid_target = int(train_val_test_num_samples[1])
    if valid_target > valid_rows:
        valid = RepeatToLengthDataset(valid, valid_target)
    runtime.print_rank_0(
        f"> full198 curriculum rows={len(train)} valid_rows={valid_rows} "
        f"valid_target={valid_target} valid_effective={len(valid)}"
    )
    return train, valid, None


train_valid_test_datasets_provider.is_distributed = True


def _transformer_config(args):
    from megatron.training.arguments import core_transformer_config_from_args

    return core_transformer_config_from_args(args)


def _pad_ids(sequences: Sequence[Sequence[int]], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    maximum = max(len(value) for value in sequences)
    ids = torch.full(
        (len(sequences), maximum), c.TOKEN_PAD, dtype=torch.long, device=device
    )
    attention = torch.zeros((len(sequences), maximum), dtype=torch.long, device=device)
    for row, values in enumerate(sequences):
        ids[row, : len(values)] = torch.tensor(values, dtype=torch.long, device=device)
        attention[row, : len(values)] = 1
    return ids, attention


def _safe_mean(values: Sequence[torch.Tensor], device: torch.device) -> torch.Tensor:
    if not values:
        return torch.zeros((), dtype=torch.float32, device=device)
    return torch.stack([value.float() for value in values]).mean()


def _safe_sum(values: Sequence[torch.Tensor], device: torch.device) -> torch.Tensor:
    if not values:
        return torch.zeros((), dtype=torch.float32, device=device)
    return torch.stack([value.float() for value in values]).sum()


def _topk_kl(student: torch.Tensor, summary: DistributionSummary) -> torch.Tensor:
    positions = min(student.shape[0], summary.indices.shape[0])
    if positions <= 0:
        return student.new_zeros((), dtype=torch.float32)
    student = student[:positions].float()
    indices = summary.indices[:positions]
    teacher = summary.probabilities[:positions].float()
    student_log = F.log_softmax(student, dim=-1).gather(-1, indices)
    teacher_log = teacher.clamp_min(1e-8).log()
    return (teacher * (teacher_log - student_log)).sum(dim=-1).mean()


class Phase3PrefixStreamingModel:
    @staticmethod
    def build(config, args, pg_collection=None):
        from megatron.core.transformer.module import MegatronModule

        class Composite(MegatronModule):
            def __init__(self):
                super().__init__(config)
                self.pg_collection = pg_collection
                model_path = Path(args.experiment_phase3_model)
                load_kwargs = {
                    "local_files_only": True,
                    "torch_dtype": torch.bfloat16,
                    "attn_implementation": args.experiment_attention_implementation,
                }
                self.qwen = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs).cuda()
                self.qwen.config.use_cache = False
                self.qwen.requires_grad_(False)
                targets = tuple(
                    value.strip()
                    for value in str(args.experiment_lora_targets).split(",")
                    if value.strip()
                )
                self.lora = inject_lora(
                    self.qwen,
                    target_modules=targets,
                    rank=int(args.experiment_lora_rank),
                    alpha=float(args.experiment_lora_alpha),
                    dropout=float(args.experiment_lora_dropout),
                )
                if not args.experiment_disable_gradient_checkpointing:
                    self.qwen.gradient_checkpointing_enable(
                        gradient_checkpointing_kwargs={"use_reentrant": False}
                    )
                    self.qwen.enable_input_require_grads()
                self.topk = int(args.experiment_teacher_topk)
                self.temperature = float(args.experiment_teacher_temperature)
                self.confidence_threshold = float(args.experiment_confidence_threshold)
                self.min_write_tokens = int(args.experiment_min_write_tokens)
                self.history_tokens = int(args.experiment_history_tokens)
                self.max_sample_tokens = int(args.experiment_max_sample_tokens)
                self.teacher_kl_weight = float(args.experiment_teacher_kl_weight)
                self.semantic_kl_weight = float(args.experiment_semantic_kl_weight)
                self.consistency_weight = float(args.experiment_consistency_weight)
                self.commit_consistency_weight = float(
                    args.experiment_commit_consistency_weight
                )
                self.boundary_weight = float(args.experiment_boundary_weight)
                self.action_weight = float(args.experiment_action_weight)
                self.qwen.eval()
                if torch.distributed.get_rank() == 0:
                    print(
                        json.dumps(
                            {
                                "schema_version": "uniss_phase3_prefix_streaming_full198_v1",
                                "phase3_model": str(model_path.resolve()),
                                "lora": self.lora.__dict__,
                                "teacher_topk": self.topk,
                                "teacher_temperature": self.temperature,
                                "confidence_threshold": self.confidence_threshold,
                                "max_sample_tokens": self.max_sample_tokens,
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )

            def train(self, mode: bool = True):
                super().train(mode)
                self.qwen.eval()
                set_lora_training(self.qwen, mode)
                return self

            def set_input_tensor(self, input_tensor):
                self.input_tensor = input_tensor

            def _hidden(self, samples: Sequence[builders.TokenSample]) -> tuple[torch.Tensor, list[tuple[int, int]]]:
                device = next(self.qwen.parameters()).device
                sequences = [sample.input_ids for sample in samples]
                ids, attention = _pad_ids(sequences, device)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    hidden = self.qwen.model(
                        input_ids=ids,
                        attention_mask=attention,
                        use_cache=False,
                        return_dict=True,
                    ).last_hidden_state
                spans = [
                    (len(sample.prompt_ids) - 1, len(sample.target_ids)) for sample in samples
                ]
                return hidden, spans

            def _target_logits(
                self, hidden: torch.Tensor, spans: Sequence[tuple[int, int]]
            ) -> list[torch.Tensor]:
                selected = [hidden[row, start : start + length] for row, (start, length) in enumerate(spans)]
                lengths = [value.shape[0] for value in selected]
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits = self.qwen.lm_head(torch.cat(selected, dim=0))
                return list(torch.split(logits, lengths, dim=0))

            @torch.no_grad()
            def _summaries(
                self, samples: Sequence[builders.TokenSample], *, use_lora: bool
            ) -> list[DistributionSummary]:
                if not samples:
                    return []
                set_lora_training(self.qwen, False)
                try:
                    with lora_enabled(self.qwen, use_lora):
                        hidden, spans = self._hidden(samples)
                        logits = self._target_logits(hidden, spans)
                    summaries: list[DistributionSummary] = []
                    for value in logits:
                        scaled = value.float() / self.temperature
                        top_values, indices = torch.topk(
                            scaled, k=min(self.topk, scaled.shape[-1]), dim=-1
                        )
                        probabilities = F.softmax(top_values, dim=-1)
                        raw = value.float()
                        maximum, prediction = raw.max(dim=-1)
                        confidence = (maximum - torch.logsumexp(raw, dim=-1)).exp()
                        summaries.append(
                            DistributionSummary(
                                indices.detach(),
                                probabilities.detach(),
                                prediction.detach(),
                                confidence.detach(),
                            )
                        )
                    return summaries
                finally:
                    if self.training:
                        set_lora_training(self.qwen, True)

            def _descriptor(
                self, record: dict[str, object], iteration: int, salt: int, training: bool
            ) -> Descriptor:
                point = point_for_iteration(iteration if training else 12000)
                sample_id = str(record["id"])
                task = (
                    choose_task(point, sample_id=sample_id, iteration=iteration, salt=salt)
                    if training
                    else ("replay", "prefix", "semantic", "commit")[
                        int(record["sample_index"]) % 4
                    ]
                )
                short, long = choose_prefix_pair(
                    point, sample_id=sample_id, iteration=iteration, salt=salt
                )
                if task == "replay":
                    mode = (
                        "quality"
                        if stable_uniform(sample_id, iteration, salt, "replay-mode") < 0.5
                        else "performance"
                    )
                    primary = builders.build_replay(record, mode)
                    if len(primary.input_ids) > self.max_sample_tokens and mode == "quality":
                        primary = builders.build_replay(record, "performance")
                    if len(primary.input_ids) > self.max_sample_tokens:
                        task = "prefix"
                    else:
                        return Descriptor(record, task, short, long, primary)
                if task in {"prefix", "commit"}:
                    return Descriptor(
                        record,
                        task,
                        short,
                        long,
                        builders.build_streaming_s2tt(record, short),
                        teacher=builders.build_teacher_s2tt(record),
                        long=builders.build_streaming_s2tt(record, long),
                        action_prompt=(
                            builders.build_action_prompt(record, short)
                            if task == "commit"
                            else None
                        ),
                    )
                semantic = list(record["target_bicodec"])
                text_ratio, cut, block = choose_semantic_geometry(
                    sample_id=sample_id,
                    iteration=iteration,
                    semantic_length=len(semantic),
                    salt=salt,
                )
                return Descriptor(
                    record,
                    "semantic",
                    short,
                    long,
                    builders.build_streaming_tts(
                        record,
                        text_ratio=text_ratio,
                        semantic_cut=cut,
                        block_size=block,
                        history_tokens=self.history_tokens,
                    ),
                    teacher=builders.build_teacher_tts(
                        record,
                        semantic_cut=cut,
                        block_size=block,
                        history_tokens=self.history_tokens,
                    ),
                )

            def forward(self, record_json, direction_id, sample_index):
                runtime = load_megatron_runtime()
                megatron_args = runtime.megatron_gpt.get_args()
                iteration = int(getattr(megatron_args, "iteration", 0) or 0)
                records = [json.loads(value) for value in record_json]
                descriptors = [
                    self._descriptor(record, iteration, int(sample_index[row]), self.training)
                    for row, record in enumerate(records)
                ]
                device = next(self.qwen.parameters()).device

                auxiliary = [value for value in descriptors if value.teacher is not None]
                teacher_summaries = self._summaries(
                    [value.teacher for value in auxiliary if value.teacher is not None],
                    use_lora=False,
                )
                teacher_by_id = {
                    id(value): summary for value, summary in zip(auxiliary, teacher_summaries)
                }
                adjacent = [
                    value for value in descriptors if value.long is not None
                ]
                long_summaries = self._summaries(
                    [value.long for value in adjacent if value.long is not None],
                    use_lora=True,
                )
                long_by_id = {
                    id(value): summary for value, summary in zip(adjacent, long_summaries)
                }

                hidden, spans = self._hidden([value.primary for value in descriptors])
                primary_logits = self._target_logits(hidden, spans)

                base_losses: list[torch.Tensor] = []
                replay_losses: list[torch.Tensor] = []
                prefix_losses: list[torch.Tensor] = []
                semantic_losses: list[torch.Tensor] = []
                commit_losses: list[torch.Tensor] = []
                teacher_kls: list[torch.Tensor] = []
                consistencies: list[torch.Tensor] = []
                boundary_losses: list[torch.Tensor] = []
                action_descriptors: list[Descriptor] = []
                action_targets: list[int] = []
                stable_counts: list[torch.Tensor] = []
                teacher_confidences: list[torch.Tensor] = []
                long_confidences: list[torch.Tensor] = []
                supervised_tokens = 0
                teacher_tokens = 0

                for descriptor, logits in zip(descriptors, primary_logits):
                    labels = torch.tensor(
                        descriptor.primary.target_ids, dtype=torch.long, device=device
                    )
                    per_token = F.cross_entropy(
                        logits.float(), labels, reduction="none"
                    )
                    if descriptor.task == "replay":
                        loss = per_token.mean()
                        replay_losses.append(loss)
                    elif descriptor.task == "semantic":
                        loss = per_token.mean()
                        semantic_losses.append(loss)
                        boundary = per_token[-min(2, len(per_token)) :].mean()
                        boundary_losses.append(boundary)
                        summary = teacher_by_id[id(descriptor)]
                        teacher_kls.append(
                            self.semantic_kl_weight * _topk_kl(logits, summary)
                        )
                        teacher_confidences.append(summary.confidence.mean())
                        teacher_tokens += int(summary.indices.shape[0])
                    else:
                        teacher = teacher_by_id[id(descriptor)]
                        long_summary = long_by_id[id(descriptor)]
                        count = min(
                            len(labels),
                            teacher.prediction.shape[0],
                            long_summary.prediction.shape[0],
                        )
                        reference = labels[:count]
                        stable = (
                            (teacher.prediction[:count] == reference)
                            & (long_summary.prediction[:count] == reference)
                            & (teacher.confidence[:count] >= self.confidence_threshold)
                            & (long_summary.confidence[:count] >= self.confidence_threshold)
                        )
                        stable_count = 0
                        for flag in stable.tolist():
                            if not flag:
                                break
                            stable_count += 1
                        stable_counts.append(
                            torch.tensor(float(stable_count), device=device)
                        )
                        teacher_confidences.append(teacher.confidence.mean())
                        long_confidences.append(long_summary.confidence.mean())
                        teacher_tokens += int(teacher.indices.shape[0])
                        teacher_kls.append(
                            self.teacher_kl_weight * _topk_kl(logits, teacher)
                        )
                        consistency_weight = (
                            self.commit_consistency_weight
                            if descriptor.task == "commit"
                            else self.consistency_weight
                        )
                        consistencies.append(
                            consistency_weight * _topk_kl(logits, long_summary)
                        )
                        if descriptor.task == "commit":
                            if stable_count:
                                loss = per_token[:stable_count].mean()
                                supervised_tokens += stable_count
                            else:
                                loss = per_token.sum() * 0.0
                            commit_losses.append(loss)
                            action_descriptors.append(descriptor)
                            action_targets.append(
                                1 if stable_count >= self.min_write_tokens else 0
                            )
                        else:
                            loss = per_token.mean()
                            prefix_losses.append(loss)
                    base_losses.append(loss)
                    if descriptor.task != "commit":
                        supervised_tokens += len(labels)

                action_losses: list[torch.Tensor] = []
                if action_descriptors:
                    prompts = [value.action_prompt for value in action_descriptors]
                    ids, attention = _pad_ids(prompts, device)  # type: ignore[arg-type]
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        action_hidden = self.qwen.model(
                            input_ids=ids,
                            attention_mask=attention,
                            use_cache=False,
                            return_dict=True,
                        ).last_hidden_state
                        rows = torch.arange(len(prompts), device=device)
                        positions = attention.sum(dim=-1) - 1
                        action_full = self.qwen.lm_head(action_hidden[rows, positions])
                    action_pair = action_full[:, [c.TOKEN_WAIT_READ, c.TOKEN_WRITE_GENERATE]].float()
                    targets = torch.tensor(action_targets, dtype=torch.long, device=device)
                    action_losses.append(F.cross_entropy(action_pair, targets))

                batch_size = max(1, len(descriptors))
                total = (
                    _safe_sum(base_losses, device)
                    + _safe_sum(teacher_kls, device)
                    + _safe_sum(consistencies, device)
                    + self.boundary_weight * _safe_sum(boundary_losses, device)
                    + self.action_weight * _safe_sum(action_losses, device)
                ) / batch_size

                task_counts = {
                    task: sum(value.task == task for value in descriptors)
                    for task in ("replay", "prefix", "semantic", "commit")
                }
                direction_id = direction_id.to(device=device).reshape(-1)
                prefix_ratios = [
                    torch.tensor(value.short_ratio, device=device)
                    for value in descriptors
                    if value.task in {"prefix", "commit"}
                ]
                write_fraction = (
                    torch.tensor(action_targets, dtype=torch.float32, device=device).mean()
                    if action_targets
                    else torch.zeros((), device=device)
                )
                metrics = (
                    _safe_mean(replay_losses, device),
                    _safe_mean(prefix_losses, device),
                    _safe_mean(semantic_losses, device),
                    _safe_mean(commit_losses, device),
                    _safe_mean(teacher_kls, device),
                    _safe_mean(consistencies, device),
                    _safe_mean(action_losses, device),
                    _safe_mean(boundary_losses, device),
                    *(
                        torch.tensor(task_counts[task] / batch_size, device=device)
                        for task in ("replay", "prefix", "semantic", "commit")
                    ),
                    (direction_id == 0).float().mean(),
                    (direction_id == 1).float().mean(),
                    _safe_mean(prefix_ratios, device),
                    _safe_mean(teacher_confidences, device),
                    _safe_mean(long_confidences, device),
                    _safe_mean(stable_counts, device),
                    write_fraction,
                    torch.tensor(float(supervised_tokens), device=device),
                    torch.tensor(float(teacher_tokens), device=device),
                    lora_update_rms(self.qwen).detach(),
                )
                values = (total.float(), *[value.float() for value in metrics])
                if len(metrics) != len(METRIC_NAMES):
                    raise AssertionError((len(metrics), len(METRIC_NAMES)))
                if not all(torch.isfinite(value).all() for value in values):
                    raise FloatingPointError("non-finite full198 streaming loss component")
                return torch.stack(values)

        return Composite()


def model_provider(
    pre_process=True,
    post_process=True,
    vp_stage=None,
    config=None,
    pg_collection=None,
):
    del pre_process, post_process, vp_stage
    runtime = load_megatron_runtime()
    args = runtime.megatron_gpt.get_args()
    return Phase3PrefixStreamingModel.build(
        config or _transformer_config(args), args, pg_collection=pg_collection
    )


def loss_func(output_tensor):
    from megatron.core import parallel_state
    from megatron.training.utils import average_losses_across_data_parallel_group

    loss = output_tensor[0]
    averaged = average_losses_across_data_parallel_group(
        list(output_tensor[1:]),
        group=parallel_state.get_data_parallel_group(with_context_parallel=True),
    )
    return loss, dict(zip(METRIC_NAMES, averaged))


def forward_step(data_iterator, model):
    batch = next(data_iterator)
    direction_id = batch["direction_id"].cuda(non_blocking=True).reshape(-1)
    sample_index = batch["sample_index"].reshape(-1)
    output = model(batch["record_json"], direction_id, sample_index)
    return output, loss_func


def main() -> None:
    runtime = load_megatron_runtime()
    patch_unused_megatron_dataset_helper_compile()
    args = runtime.parse_and_validate_args(
        extra_args_provider=add_experiment_args,
        args_defaults={"tokenizer_type": "NullTokenizer"},
    )
    validate_experiment_args(args)
    model_config = runtime.gpt_config_from_args(args)
    full_config = runtime.pretrain_cfg_container_from_args(args, model_config)
    full_config.model = None
    runtime.pretrain(
        full_config,
        train_valid_test_datasets_provider,
        runtime.ModelType.encoder_or_decoder,
        forward_step,
        model_provider=model_provider,
    )


if __name__ == "__main__":
    main()

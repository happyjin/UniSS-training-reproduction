"""Train corrected Stage-B with fixed-rate WhisperVQ latent distillation."""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
import time
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter

from training.simul_uniss.distributed import DistributedContext
from training.simul_uniss.policy_tokenizer import PolicyTokenizer
from training.simul_uniss.subsecond_v2.stage_b_latent_data import (
    LatentStageBAudioDataset,
    collate_stage_b_latent,
)
from training.simul_uniss.subsecond_v2.stage_b_latent_model import (
    DEFAULT_CODEBOOK_KEY,
    LatentCausalAudioStudent,
    LatentStageBModelConfig,
    load_whispervq_codebook,
    nearest_codebook_tokens,
)


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _module(model: torch.nn.Module) -> LatentCausalAudioStudent:
    value = model.module if isinstance(model, DistributedDataParallel) else model
    if not isinstance(value, LatentCausalAudioStudent):
        raise TypeError(type(value))
    return value


def _ctc_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    input_lengths: torch.Tensor,
    target_lengths: torch.Tensor,
) -> torch.Tensor:
    return F.ctc_loss(
        logits.float().log_softmax(dim=-1).transpose(0, 1),
        targets,
        input_lengths,
        target_lengths,
        blank=0,
        reduction="mean",
        zero_infinity=True,
    )


def _masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (value * mask.to(value.dtype)).sum() / mask.sum().clamp_min(1)


def _common_token_mask(
    output_lengths: torch.Tensor,
    target_lengths: torch.Tensor,
    time_steps: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    common = torch.minimum(output_lengths, target_lengths)
    positions = torch.arange(time_steps, device=common.device).reshape(1, -1)
    return positions < common.reshape(-1, 1), common


def stage_b_latent_losses(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    *,
    latent_weight: float,
    hidden_distill_weight: float,
    source_weight: float,
    capacity_weight: float,
    stability_weight: float,
    consistency_weight: float,
    compute_consistency: bool,
    chunk_samples: int,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    output = model(
        batch["waveform"],
        batch["waveform_lengths"],
        batch["utterance_sample_lengths"],
    )
    module = _module(model)
    time_steps = output["glm_latent"].shape[1]
    mask, common_lengths = _common_token_mask(
        output["token_lengths"], batch["teacher_glm_lengths"], time_steps
    )
    teacher_ids = batch["teacher_glm_ids"][:, :time_steps]
    if teacher_ids.shape[1] < time_steps:
        teacher_ids = F.pad(teacher_ids, (0, time_steps - teacher_ids.shape[1]))
    teacher_latent = F.embedding(teacher_ids, module.codebook).detach()
    difference = (output["glm_latent"].float() - teacher_latent.float()).square().mean(dim=-1)
    latent_l2 = _masked_mean(difference, mask)
    cosine = 1.0 - F.cosine_similarity(
        output["glm_latent"].float(), teacher_latent.float(), dim=-1
    )
    hidden_distill = _masked_mean(cosine, mask)

    source = _ctc_loss(
        output["source_ctc_logits"],
        batch["source_policy"],
        output["output_lengths"],
        batch["source_policy_lengths"],
    )
    final_positions = (output["token_lengths"] - 1).clamp_min(0)
    capacity_logits = output["target_capacity_logits"].gather(
        1, final_positions.reshape(-1, 1)
    ).squeeze(1)
    capacity = F.mse_loss(torch.sigmoid(capacity_logits.float()), batch["target_capacity"])
    stability_target = batch["stability_target"][:, :time_steps]
    if stability_target.shape[1] < time_steps:
        stability_target = F.pad(stability_target, (0, time_steps - stability_target.shape[1]))
    stability_raw = F.binary_cross_entropy_with_logits(
        output["stability_logits"].float(), stability_target.float(), reduction="none"
    )
    stability = _masked_mean(stability_raw, mask)

    consistency = output["glm_latent"].sum() * 0.0
    if compute_consistency and consistency_weight:
        alternate_samples = torch.div(
            batch["utterance_sample_lengths"], chunk_samples, rounding_mode="floor"
        ) * chunk_samples
        same = alternate_samples == batch["utterance_sample_lengths"]
        alternate_samples = torch.where(
            same,
            (alternate_samples - chunk_samples).clamp_min(400),
            alternate_samples.clamp_min(400),
        )
        alternate = model(
            batch["waveform"], batch["waveform_lengths"], alternate_samples
        )
        common_time = min(output["glm_latent"].shape[1], alternate["glm_latent"].shape[1])
        alternate_mask, _ = _common_token_mask(
            output["token_lengths"].clamp_max(alternate["token_lengths"]),
            alternate["token_lengths"],
            common_time,
        )
        consistency_raw = 1.0 - F.cosine_similarity(
            output["glm_latent"][:, :common_time].float(),
            alternate["glm_latent"][:, :common_time].float(),
            dim=-1,
        )
        consistency = _masked_mean(consistency_raw, alternate_mask)

    total = (
        latent_weight * latent_l2
        + hidden_distill_weight * hidden_distill
        + source_weight * source
        + capacity_weight * capacity
        + stability_weight * stability
        + consistency_weight * consistency
    )
    return {
        "total": total,
        "latent_l2": latent_l2,
        "hidden_distill": hidden_distill,
        "source": source,
        "capacity": capacity,
        "stability": stability,
        "chunk_consistency": consistency,
    }, {**output, "common_lengths": common_lengths, "teacher_ids": teacher_ids, "mask": mask}


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader[dict[str, torch.Tensor]],
    device: torch.device,
    distributed: DistributedContext,
    args: argparse.Namespace,
) -> dict[str, float]:
    model.eval()
    names = (
        "total",
        "latent_l2",
        "hidden_distill",
        "source",
        "capacity",
        "stability",
        "chunk_consistency",
    )
    sums = {name: 0.0 for name in names}
    batches = 0
    exact = 0
    tokens = 0
    for batch in loader:
        batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=args.bf16):
            losses, output = stage_b_latent_losses(
                model,
                batch,
                latent_weight=args.latent_weight,
                hidden_distill_weight=args.hidden_distill_weight,
                source_weight=args.source_weight,
                capacity_weight=args.capacity_weight,
                stability_weight=args.stability_weight,
                consistency_weight=args.consistency_weight,
                compute_consistency=True,
                chunk_samples=args.chunk_samples,
            )
        predicted = nearest_codebook_tokens(
            output["glm_latent"], _module(model).codebook, chunk_size=args.quantize_chunk_size
        )
        mask = output["mask"]
        exact += int(((predicted == output["teacher_ids"]) & mask).sum())
        tokens += int(mask.sum())
        for name in names:
            sums[name] += float(losses[name])
        batches += 1
        if batches >= args.eval_batches:
            break
    reduced = distributed.reduce_sums(
        [*(sums[name] for name in names), float(batches), float(exact), float(tokens)]
    )
    divisor = max(1.0, reduced[len(names)])
    metrics = {name: reduced[index] / divisor for index, name in enumerate(names)}
    metrics["token_agreement"] = reduced[len(names) + 1] / max(1.0, reduced[len(names) + 2])
    model.train()
    return metrics


def _save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    args: argparse.Namespace,
    model_config: LatentStageBModelConfig,
    step: int,
    epoch: int,
    consumed_audio_seconds: float,
    best_agreement: float,
    best_validation: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": "simul_uniss_stage_b_latent_checkpoint_v1",
            "model": _module(model).state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "args": vars(args),
            "model_config": model_config.__dict__,
            "codebook_model": str(Path(args.codebook_model).resolve()),
            "codebook_key": args.codebook_key,
            "step": step,
            "epoch": epoch,
            "consumed_audio_seconds": consumed_audio_seconds,
            "best_agreement": best_agreement,
            "best_validation": best_validation,
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
        path,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--valid-manifest", required=True)
    parser.add_argument("--policy-tokenizer", required=True)
    parser.add_argument("--codebook-model", required=True)
    parser.add_argument("--codebook-key", default=DEFAULT_CODEBOOK_KEY)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tensorboard-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-steps", type=int, default=50_000)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--min-learning-rate", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-fraction", type=float, default=0.05)
    parser.add_argument("--hidden-size", type=int, default=768)
    parser.add_argument("--num-layers", type=int, default=16)
    parser.add_argument("--num-heads", type=int, default=12)
    parser.add_argument("--ffn-dim", type=int, default=3_072)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max-audio-seconds", type=float, default=8.0)
    parser.add_argument("--eval-interval", type=int, default=500)
    parser.add_argument("--eval-batches", type=int, default=16)
    parser.add_argument("--save-interval", type=int, default=500)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--teacher-glm-field", default="teacher_source_glm")
    parser.add_argument("--teacher-glm-end-field", default="teacher_source_glm_end_ms")
    parser.add_argument("--latent-weight", type=float, default=1.0)
    parser.add_argument("--hidden-distill-weight", type=float, default=0.5)
    parser.add_argument("--source-weight", type=float, default=0.3)
    parser.add_argument("--capacity-weight", type=float, default=0.4)
    parser.add_argument("--stability-weight", type=float, default=0.2)
    parser.add_argument("--consistency-weight", type=float, default=0.1)
    parser.add_argument("--consistency-interval", type=int, default=4)
    parser.add_argument("--chunk-samples", type=int, default=2_560)
    parser.add_argument("--quantize-chunk-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20_260_802)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--throughput-scan",
        action="store_true",
        help="run training steps without validation/checkpoint writes",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0 <= args.warmup_fraction < 1:
        raise ValueError("warmup_fraction must be in [0, 1)")
    distributed = DistributedContext.initialize(args.device)
    torch.manual_seed(args.seed + distributed.rank)
    torch.set_float32_matmul_precision("high")
    device = distributed.device
    tokenizer = PolicyTokenizer(args.policy_tokenizer)
    codebook = load_whispervq_codebook(args.codebook_model, key=args.codebook_key)
    model_config = LatentStageBModelConfig(
        policy_vocab_size=tokenizer.ctc_vocab_size,
        codebook_size=codebook.shape[0],
        codebook_dim=codebook.shape[1],
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        ffn_dim=args.ffn_dim,
        dropout=args.dropout,
    )
    base_model = LatentCausalAudioStudent(model_config, codebook).to(device)
    output_dir = Path(args.output_dir)
    last_checkpoint = output_dir / "last.pt"
    resume_value = None
    if args.resume and last_checkpoint.is_file():
        resume_value = torch.load(last_checkpoint, map_location="cpu", weights_only=False)
        base_model.load_state_dict(resume_value["model"])
    if distributed.enabled:
        model: torch.nn.Module = DistributedDataParallel(
            base_model,
            device_ids=[distributed.local_rank] if device.type == "cuda" else None,
            output_device=distributed.local_rank if device.type == "cuda" else None,
            broadcast_buffers=False,
        )
    else:
        model = base_model

    dataset_kwargs = {
        "policy_tokenizer": tokenizer,
        "max_audio_seconds": args.max_audio_seconds,
        "teacher_glm_field": args.teacher_glm_field,
        "teacher_glm_end_field": args.teacher_glm_end_field,
    }
    train_dataset = LatentStageBAudioDataset(
        args.manifest, prefix_training=True, **dataset_kwargs
    )
    valid_dataset = LatentStageBAudioDataset(
        args.valid_manifest, prefix_training=False, **dataset_kwargs
    )
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=distributed.world_size,
        rank=distributed.rank,
        shuffle=True,
        seed=args.seed,
        drop_last=False,
    )
    valid_sampler = DistributedSampler(
        valid_dataset,
        num_replicas=distributed.world_size,
        rank=distributed.rank,
        shuffle=False,
        drop_last=False,
    )
    loader_kwargs = {
        "batch_size": args.batch_size,
        "collate_fn": collate_stage_b_latent,
        "pin_memory": device.type == "cuda",
        "num_workers": args.num_workers,
        "persistent_workers": args.num_workers > 0,
    }
    train_loader = DataLoader(train_dataset, sampler=train_sampler, **loader_kwargs)
    valid_loader = DataLoader(valid_dataset, sampler=valid_sampler, **loader_kwargs)
    optimizer = torch.optim.AdamW(
        (value for value in model.parameters() if value.requires_grad),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    warmup_steps = max(1, round(args.max_steps * args.warmup_fraction))

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return max(1e-8, (step + 1) / warmup_steps)
        progress = (step - warmup_steps) / max(1, args.max_steps - warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
        minimum_ratio = args.min_learning_rate / args.learning_rate
        return minimum_ratio + (1.0 - minimum_ratio) * cosine

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    step = 0
    epoch = 0
    consumed_audio_seconds = 0.0
    best_agreement = -1.0
    best_validation = float("inf")
    if resume_value is not None:
        optimizer.load_state_dict(resume_value["optimizer"])
        scheduler.load_state_dict(resume_value["scheduler"])
        step = int(resume_value["step"])
        epoch = int(resume_value["epoch"])
        consumed_audio_seconds = float(resume_value.get("consumed_audio_seconds", 0.0))
        best_agreement = float(resume_value.get("best_agreement", -1.0))
        best_validation = float(resume_value.get("best_validation", float("inf")))
    if distributed.is_main:
        output_dir.mkdir(parents=True, exist_ok=True)
    distributed.barrier()
    writer = (
        SummaryWriter(args.tensorboard_dir)
        if distributed.is_main and not args.throughput_scan
        else None
    )
    model.train()
    last_log_time = time.perf_counter()
    last_log_audio = consumed_audio_seconds
    while step < args.max_steps:
        train_sampler.set_epoch(epoch)
        epoch += 1
        for batch in train_loader:
            step += 1
            batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            local_audio_seconds = float(batch["utterance_sample_lengths"].sum()) / 16_000.0
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=args.bf16):
                losses, _ = stage_b_latent_losses(
                    model,
                    batch,
                    latent_weight=args.latent_weight,
                    hidden_distill_weight=args.hidden_distill_weight,
                    source_weight=args.source_weight,
                    capacity_weight=args.capacity_weight,
                    stability_weight=args.stability_weight,
                    consistency_weight=args.consistency_weight,
                    compute_consistency=(step % args.consistency_interval == 0),
                    chunk_samples=args.chunk_samples,
                )
            losses["total"].backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                (value for value in model.parameters() if value.requires_grad), 1.0
            )
            optimizer.step()
            scheduler.step()
            consumed_audio_seconds += distributed.reduce_sums([local_audio_seconds])[0]
            if distributed.is_main and (step == 1 or step % args.log_interval == 0):
                now = time.perf_counter()
                elapsed = max(now - last_log_time, 1e-6)
                metrics = {
                    "step": step,
                    **{name: float(value) for name, value in losses.items()},
                    "grad_norm": float(grad_norm),
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    "audio_seconds_per_second": (consumed_audio_seconds - last_log_audio) / elapsed,
                    "consumed_audio_hours": consumed_audio_seconds / 3_600.0,
                    "max_cuda_memory_gib": torch.cuda.max_memory_allocated(device) / 2**30
                    if device.type == "cuda"
                    else 0.0,
                }
                last_log_time = now
                last_log_audio = consumed_audio_seconds
                print(json.dumps(metrics, sort_keys=True), flush=True)
                if writer is not None:
                    for name, value in metrics.items():
                        if name != "step":
                            writer.add_scalar(f"stage_b_latent/{name}", value, step)
                    writer.flush()
            if not args.throughput_scan and (
                step % args.eval_interval == 0 or step == args.max_steps
            ):
                metrics = evaluate(model, valid_loader, device, distributed, args)
                if distributed.is_main:
                    assert writer is not None
                    for name, value in metrics.items():
                        writer.add_scalar(f"stage_b_latent/valid_{name}", value, step)
                    improved = (
                        metrics["token_agreement"] > best_agreement
                        or (
                            metrics["token_agreement"] == best_agreement
                            and metrics["total"] < best_validation
                        )
                    )
                    if improved:
                        best_agreement = metrics["token_agreement"]
                        best_validation = metrics["total"]
                        _save_checkpoint(
                            output_dir / "best.pt",
                            model,
                            optimizer,
                            scheduler,
                            args,
                            model_config,
                            step,
                            epoch,
                            consumed_audio_seconds,
                            best_agreement,
                            best_validation,
                        )
                    writer.flush()
            if (
                not args.throughput_scan
                and distributed.is_main
                and (step % args.save_interval == 0 or step == args.max_steps)
            ):
                _save_checkpoint(
                    last_checkpoint,
                    model,
                    optimizer,
                    scheduler,
                    args,
                    model_config,
                    step,
                    epoch,
                    consumed_audio_seconds,
                    best_agreement,
                    best_validation,
                )
            if step >= args.max_steps:
                break
    if writer is not None:
        writer.close()
    if distributed.is_main and args.throughput_scan:
        print(
            json.dumps(
                {
                    "status": "throughput_scan_complete",
                    "step": step,
                    "consumed_audio_hours": consumed_audio_seconds / 3_600.0,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    if distributed.is_main and not args.throughput_scan:
        _atomic_json(
            output_dir / "TRAINING_COMPLETE.json",
            {
                "schema_version": "simul_uniss_stage_b_latent_training_v1",
                "status": "complete",
                "step": step,
                "best_agreement": best_agreement,
                "best_validation": best_validation,
                "consumed_audio_hours": consumed_audio_seconds / 3_600.0,
                "checkpoint": str(last_checkpoint.resolve()),
            },
        )
    distributed.close()


if __name__ == "__main__":
    main()

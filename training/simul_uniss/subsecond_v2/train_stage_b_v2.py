"""Train quantization-aware causal Stage-B-v2 from Stage-A-v3 sidecars."""

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
from training.simul_uniss.subsecond_v2.stage_b_latent_model import (
    DEFAULT_CODEBOOK_KEY,
    LatentCausalAudioStudent,
    LatentStageBModelConfig,
    load_whispervq_codebook,
)
from training.simul_uniss.subsecond_v2.stage_b_v2_data import (
    StageBV2SidecarDataset,
    collate_stage_b_v2,
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


def _masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (value * mask.to(value.dtype)).sum() / mask.sum().clamp_min(1)


def _time_mask(lengths: torch.Tensor, time_steps: int) -> torch.Tensor:
    return torch.arange(time_steps, device=lengths.device).reshape(1, -1) < lengths.reshape(-1, 1)


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


def codebook_ce_margin(
    latent: torch.Tensor,
    target_ids: torch.Tensor,
    mask: torch.Tensor,
    codebook: torch.Tensor,
    *,
    temperature: float,
    margin: float,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Full-codebook CE and nearest-negative margin in teacher VQ geometry."""

    selected = latent.float()[mask]
    targets = target_ids[mask]
    if not len(selected):
        zero = latent.sum() * 0.0
        return zero, zero, zero.detach(), zero.detach()
    codebook_float = codebook.float()
    codebook_norm = codebook_float.square().sum(dim=1).reshape(1, -1)
    ce_sum = selected.new_zeros(())
    margin_sum = selected.new_zeros(())
    exact = selected.new_zeros(())
    top5 = selected.new_zeros(())
    dimension = selected.shape[-1]
    for start in range(0, len(selected), chunk_size):
        current = selected[start : start + chunk_size]
        current_targets = targets[start : start + chunk_size]
        distances = (
            current.square().sum(dim=1, keepdim=True)
            + codebook_norm
            - 2.0 * current @ codebook_float.T
        ) / dimension
        logits = -distances / temperature
        ce_sum = ce_sum + F.cross_entropy(logits, current_targets, reduction="sum")
        target_distance = distances.gather(1, current_targets.reshape(-1, 1)).squeeze(1)
        wrong = distances.clone()
        wrong.scatter_(1, current_targets.reshape(-1, 1), float("inf"))
        nearest_wrong = wrong.min(dim=1).values
        margin_sum = margin_sum + F.relu(margin + target_distance - nearest_wrong).sum()
        nearest = distances.argmin(dim=1)
        exact = exact + (nearest == current_targets).sum()
        top = torch.topk(distances, k=min(5, distances.shape[1]), largest=False).indices
        top5 = top5 + (top == current_targets.reshape(-1, 1)).any(dim=1).sum()
    count = mask.sum().clamp_min(1)
    return ce_sum / count, margin_sum / count, exact / count, top5 / count


def stage_b_v2_losses(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    *,
    hidden_weight: float,
    cosine_weight: float,
    codebook_ce_weight: float,
    margin_weight: float,
    full_context_weight: float,
    source_weight: float,
    capacity_weight: float,
    stability_weight: float,
    consistency_weight: float,
    auxiliary_scale: float,
    temperature: float,
    margin: float,
    quantize_chunk_size: int,
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
    common = torch.minimum(output["token_lengths"], batch["target_lengths"])
    mask = _time_mask(common, time_steps)
    targets = batch["target_ids"][:, :time_steps]
    if targets.shape[1] < time_steps:
        targets = F.pad(targets, (0, time_steps - targets.shape[1]))
    codebook_ce, codebook_margin, agreement, top5 = codebook_ce_margin(
        output["glm_latent"],
        targets,
        mask,
        module.codebook,
        temperature=temperature,
        margin=margin,
        chunk_size=quantize_chunk_size,
    )

    hidden_l1 = output["glm_latent"].sum() * 0.0
    hidden_cosine = output["glm_latent"].sum() * 0.0
    teacher_hidden = batch["teacher_hidden"][:, :time_steps]
    if teacher_hidden.shape[-1] == output["glm_latent"].shape[-1]:
        hidden_mask = mask & batch["has_teacher_hidden"].reshape(-1, 1)
        if bool(hidden_mask.any()):
            l1 = F.smooth_l1_loss(
                output["glm_latent"].float(), teacher_hidden.float(), reduction="none"
            ).mean(dim=-1)
            hidden_l1 = _masked_mean(l1, hidden_mask)
            cosine = 1.0 - F.cosine_similarity(
                output["glm_latent"].float(), teacher_hidden.float(), dim=-1
            )
            hidden_cosine = _masked_mean(cosine, hidden_mask)

    references = batch["full_reference_ids"][:, :time_steps]
    if references.shape[1] < time_steps:
        references = F.pad(references, (0, time_steps - references.shape[1]))
    reference_mask = mask & _time_mask(batch["full_reference_lengths"], time_steps)
    reference_hidden = F.embedding(references, module.codebook).detach()
    full_context = _masked_mean(
        1.0
        - F.cosine_similarity(
            output["glm_latent"].float(), reference_hidden.float(), dim=-1
        ),
        reference_mask,
    )

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
    stability = _masked_mean(
        F.binary_cross_entropy_with_logits(
            output["stability_logits"].float(), stability_target.float(), reduction="none"
        ),
        mask,
    )

    consistency = output["glm_latent"].sum() * 0.0
    if compute_consistency and consistency_weight:
        alternate_samples = (
            torch.div(batch["utterance_sample_lengths"], chunk_samples, rounding_mode="floor")
            * chunk_samples
        )
        same = alternate_samples == batch["utterance_sample_lengths"]
        alternate_samples = torch.where(
            same,
            (alternate_samples - chunk_samples).clamp_min(400),
            alternate_samples.clamp_min(400),
        )
        alternate = model(batch["waveform"], batch["waveform_lengths"], alternate_samples)
        common_time = min(time_steps, alternate["glm_latent"].shape[1])
        consistency_mask = _time_mask(
            torch.minimum(output["token_lengths"], alternate["token_lengths"]),
            common_time,
        )
        consistency = _masked_mean(
            1.0
            - F.cosine_similarity(
                output["glm_latent"][:, :common_time].float(),
                alternate["glm_latent"][:, :common_time].float(),
                dim=-1,
            ),
            consistency_mask,
        )

    total = (
        hidden_weight * hidden_l1
        + cosine_weight * hidden_cosine
        + codebook_ce_weight * codebook_ce
        + margin_weight * codebook_margin
        + full_context_weight * full_context
        + auxiliary_scale
        * (
            source_weight * source
            + capacity_weight * capacity
            + stability_weight * stability
            + consistency_weight * consistency
        )
    )
    return {
        "total": total,
        "hidden_l1": hidden_l1,
        "hidden_cosine": hidden_cosine,
        "codebook_ce": codebook_ce,
        "codebook_margin": codebook_margin,
        "full_context": full_context,
        "source": source,
        "capacity": capacity,
        "stability": stability,
        "chunk_consistency": consistency,
        "target_agreement": agreement.detach(),
        "target_top5": top5.detach(),
    }, {**output, "mask": mask, "targets": targets}


@torch.no_grad()
def evaluate(model, loader, device, distributed, args) -> dict[str, float]:
    model.eval()
    sums: dict[str, float] = {}
    batches = 0
    for batch in loader:
        batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=args.bf16):
            losses, _ = stage_b_v2_losses(
                model,
                batch,
                hidden_weight=args.hidden_weight,
                cosine_weight=args.cosine_weight,
                codebook_ce_weight=args.codebook_ce_weight,
                margin_weight=args.margin_weight,
                full_context_weight=args.full_context_weight,
                source_weight=args.source_weight,
                capacity_weight=args.capacity_weight,
                stability_weight=args.stability_weight,
                consistency_weight=args.consistency_weight,
                auxiliary_scale=1.0,
                temperature=args.codebook_temperature,
                margin=args.codebook_margin,
                quantize_chunk_size=args.quantize_chunk_size,
                compute_consistency=True,
                chunk_samples=args.chunk_samples,
            )
        for name, value in losses.items():
            sums[name] = sums.get(name, 0.0) + float(value)
        batches += 1
        if batches >= args.eval_batches:
            break
    names = sorted(sums)
    reduced = distributed.reduce_sums([*(sums[name] for name in names), float(batches)])
    divisor = max(1.0, reduced[-1])
    model.train()
    return {name: reduced[index] / divisor for index, name in enumerate(names)}


def _save(path, model, optimizer, scheduler, args, config, step, epoch, best):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": "simul_uniss_stage_b_v2_checkpoint_v1",
            "model": _module(model).state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "args": vars(args),
            "model_config": config.__dict__,
            "codebook_model": str(Path(args.codebook_model).resolve()),
            "codebook_key": args.codebook_key,
            "step": step,
            "epoch": epoch,
            "best_target_agreement": best,
        },
        path,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sidecar-manifest", required=True)
    parser.add_argument("--valid-sidecar-manifest", required=True)
    parser.add_argument("--source-manifest", required=True)
    parser.add_argument("--valid-source-manifest", required=True)
    parser.add_argument("--policy-tokenizer", required=True)
    parser.add_argument("--codebook-model", required=True)
    parser.add_argument("--codebook-key", default=DEFAULT_CODEBOOK_KEY)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tensorboard-dir", required=True)
    parser.add_argument("--initialize-from")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-steps", type=int, default=20_000)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--min-learning-rate", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-fraction", type=float, default=0.05)
    parser.add_argument("--representation-only-steps", type=int, default=5_000)
    parser.add_argument("--auxiliary-ramp-steps", type=int, default=5_000)
    parser.add_argument("--hidden-size", type=int, default=768)
    parser.add_argument("--num-layers", type=int, default=16)
    parser.add_argument("--num-heads", type=int, default=12)
    parser.add_argument("--ffn-dim", type=int, default=3_072)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max-audio-seconds", type=float, default=8.0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--eval-interval", type=int, default=500)
    parser.add_argument("--eval-batches", type=int, default=8)
    parser.add_argument("--save-interval", type=int, default=500)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--hidden-weight", type=float, default=1.0)
    parser.add_argument("--cosine-weight", type=float, default=0.5)
    parser.add_argument("--codebook-ce-weight", type=float, default=1.0)
    parser.add_argument("--margin-weight", type=float, default=0.5)
    parser.add_argument("--full-context-weight", type=float, default=0.05)
    parser.add_argument("--source-weight", type=float, default=0.1)
    parser.add_argument("--capacity-weight", type=float, default=0.1)
    parser.add_argument("--stability-weight", type=float, default=0.1)
    parser.add_argument("--consistency-weight", type=float, default=0.05)
    parser.add_argument("--consistency-interval", type=int, default=4)
    parser.add_argument("--chunk-samples", type=int, default=2_560)
    parser.add_argument("--codebook-temperature", type=float, default=0.05)
    parser.add_argument("--codebook-margin", type=float, default=0.01)
    parser.add_argument("--quantize-chunk-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20_260_802)
    parser.add_argument("--bf16", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    distributed = DistributedContext.initialize(args.device)
    torch.manual_seed(args.seed + distributed.rank)
    torch.set_float32_matmul_precision("high")
    device = distributed.device
    tokenizer = PolicyTokenizer(args.policy_tokenizer)
    codebook = load_whispervq_codebook(args.codebook_model, key=args.codebook_key)
    config = LatentStageBModelConfig(
        policy_vocab_size=tokenizer.ctc_vocab_size,
        codebook_size=codebook.shape[0],
        codebook_dim=codebook.shape[1],
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        ffn_dim=args.ffn_dim,
        dropout=args.dropout,
        mel_scale="slaney",
        mel_norm="slaney",
    )
    base_model = LatentCausalAudioStudent(config, codebook).to(device)
    if args.initialize_from:
        initial = torch.load(args.initialize_from, map_location="cpu", weights_only=False)
        missing, unexpected = base_model.load_state_dict(initial["model"], strict=False)
        if unexpected:
            raise ValueError(f"unexpected initialization keys: {unexpected}")
        if distributed.is_main:
            print(json.dumps({"initialized_from": args.initialize_from, "missing": missing}))
    model: torch.nn.Module = base_model
    if distributed.enabled:
        model = DistributedDataParallel(
            base_model,
            device_ids=[distributed.local_rank] if device.type == "cuda" else None,
            output_device=distributed.local_rank if device.type == "cuda" else None,
            broadcast_buffers=False,
        )
    dataset_args = {
        "policy_tokenizer": tokenizer,
        "max_audio_seconds": args.max_audio_seconds,
    }
    train_dataset = StageBV2SidecarDataset(
        args.sidecar_manifest,
        args.source_manifest,
        prefix_training=True,
        **dataset_args,
    )
    valid_dataset = StageBV2SidecarDataset(
        args.valid_sidecar_manifest,
        args.valid_source_manifest,
        prefix_training=False,
        **dataset_args,
    )
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=distributed.world_size,
        rank=distributed.rank,
        shuffle=True,
        seed=args.seed,
    )
    valid_sampler = DistributedSampler(
        valid_dataset,
        num_replicas=distributed.world_size,
        rank=distributed.rank,
        shuffle=False,
    )
    loader_args = {
        "batch_size": args.batch_size,
        "collate_fn": collate_stage_b_v2,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": args.num_workers > 0,
    }
    train_loader = DataLoader(train_dataset, sampler=train_sampler, **loader_args)
    valid_loader = DataLoader(valid_dataset, sampler=valid_sampler, **loader_args)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    warmup = max(1, round(args.max_steps * args.warmup_fraction))

    def schedule(step: int) -> float:
        if step < warmup:
            return max(1e-8, (step + 1) / warmup)
        progress = (step - warmup) / max(1, args.max_steps - warmup)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
        floor = args.min_learning_rate / args.learning_rate
        return floor + (1.0 - floor) * cosine

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    output_dir = Path(args.output_dir)
    if distributed.is_main:
        output_dir.mkdir(parents=True, exist_ok=True)
    distributed.barrier()
    writer = SummaryWriter(args.tensorboard_dir) if distributed.is_main else None
    step = epoch = 0
    best = -1.0
    last_log = time.perf_counter()
    model.train()
    while step < args.max_steps:
        train_sampler.set_epoch(epoch)
        epoch += 1
        for batch in train_loader:
            step += 1
            batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            if step <= args.representation_only_steps:
                auxiliary_scale = 0.0
            else:
                auxiliary_scale = min(
                    1.0,
                    (step - args.representation_only_steps) / max(1, args.auxiliary_ramp_steps),
                )
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=args.bf16):
                losses, _ = stage_b_v2_losses(
                    model,
                    batch,
                    hidden_weight=args.hidden_weight,
                    cosine_weight=args.cosine_weight,
                    codebook_ce_weight=args.codebook_ce_weight,
                    margin_weight=args.margin_weight,
                    full_context_weight=args.full_context_weight,
                    source_weight=args.source_weight,
                    capacity_weight=args.capacity_weight,
                    stability_weight=args.stability_weight,
                    consistency_weight=args.consistency_weight,
                    auxiliary_scale=auxiliary_scale,
                    temperature=args.codebook_temperature,
                    margin=args.codebook_margin,
                    quantize_chunk_size=args.quantize_chunk_size,
                    compute_consistency=step % args.consistency_interval == 0,
                    chunk_samples=args.chunk_samples,
                )
            losses["total"].backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            if distributed.is_main and (step == 1 or step % args.log_interval == 0):
                now = time.perf_counter()
                metrics = {
                    "step": step,
                    **{name: float(value) for name, value in losses.items()},
                    "grad_norm": float(grad_norm),
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    "auxiliary_scale": auxiliary_scale,
                    "steps_per_second": args.log_interval / max(1e-6, now - last_log),
                }
                last_log = now
                print(json.dumps(metrics, sort_keys=True), flush=True)
                assert writer is not None
                for name, value in metrics.items():
                    if name != "step":
                        writer.add_scalar(f"stage_b_v2/{name}", value, step)
                writer.flush()
            if step % args.eval_interval == 0 or step == args.max_steps:
                metrics = evaluate(model, valid_loader, device, distributed, args)
                if distributed.is_main:
                    assert writer is not None
                    for name, value in metrics.items():
                        writer.add_scalar(f"stage_b_v2/valid_{name}", value, step)
                    if metrics["target_agreement"] > best:
                        best = metrics["target_agreement"]
                        _save(output_dir / "best.pt", model, optimizer, scheduler, args, config, step, epoch, best)
                    writer.flush()
            if distributed.is_main and (step % args.save_interval == 0 or step == args.max_steps):
                _save(output_dir / "last.pt", model, optimizer, scheduler, args, config, step, epoch, best)
            if step >= args.max_steps:
                break
    if writer is not None:
        writer.close()
        _atomic_json(
            output_dir / "TRAINING_COMPLETE.json",
            {
                "schema_version": "simul_uniss_stage_b_v2_training_v1",
                "status": "complete",
                "step": step,
                "best_target_agreement": best,
                "checkpoint": str((output_dir / "last.pt").resolve()),
            },
        )
    distributed.close()


if __name__ == "__main__":
    main()

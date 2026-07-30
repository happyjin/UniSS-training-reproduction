"""Train the Stage-B Emformer causal audio student with eight-GPU DDP."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, DistributedSampler, Subset
from torch.utils.tensorboard import SummaryWriter

from training.simul_uniss.distributed import DistributedContext
from training.simul_uniss.policy_tokenizer import PolicyTokenizer
from training.simul_uniss.subsecond_v1.data import StageBAudioDataset, collate_stage_b
from training.simul_uniss.subsecond_v1.model import CausalAudioStudentV2, StageBModelConfig


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


def stage_b_losses(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    *,
    source_weight: float,
    capacity_weight: float,
    stability_weight: float,
    stability_holdback_frames: int,
) -> dict[str, torch.Tensor]:
    output = model(
        batch["waveform"],
        batch["waveform_lengths"],
        batch["utterance_sample_lengths"],
    )
    lengths = output["output_lengths"]
    teacher = _ctc_loss(
        output["teacher_glm_logits"],
        batch["teacher_glm"],
        lengths,
        batch["teacher_glm_lengths"],
    )
    source = _ctc_loss(
        output["source_ctc_logits"],
        batch["source_policy"],
        lengths,
        batch["source_policy_lengths"],
    )

    time_steps = output["stability_logits"].shape[1]
    positions = torch.arange(time_steps, device=lengths.device).unsqueeze(0)
    valid = positions < lengths.unsqueeze(1)
    stable_before = (lengths - stability_holdback_frames).clamp_min(0).unsqueeze(1)
    stability_target = (positions < stable_before).float()
    stability_raw = F.binary_cross_entropy_with_logits(
        output["stability_logits"].float(), stability_target, reduction="none"
    )
    stability = (stability_raw * valid).sum() / valid.sum().clamp_min(1)

    final_positions = (lengths - 1).clamp_min(0)
    capacity_logits = output["target_capacity_logits"].gather(1, final_positions.unsqueeze(1)).squeeze(1)
    capacity = F.mse_loss(torch.sigmoid(capacity_logits.float()), batch["target_capacity"])
    # Keep every DDP parameter in the graph even when a proxy head is disabled.
    connected_zero = 0.0 * (
        output["target_capacity_logits"].sum() + output["source_ctc_logits"].sum()
    )
    total = (
        teacher
        + source_weight * source
        + capacity_weight * capacity
        + stability_weight * stability
        + connected_zero
    )
    return {
        "total": total,
        "teacher": teacher,
        "source": source,
        "capacity": capacity,
        "stability": stability,
    }


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    distributed: DistributedContext,
    args: argparse.Namespace,
) -> dict[str, float]:
    model.eval()
    names = ("total", "teacher", "source", "capacity", "stability")
    sums = {name: 0.0 for name in names}
    count = 0
    for batch in loader:
        batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=args.bf16):
            losses = stage_b_losses(
                model,
                batch,
                source_weight=args.source_weight,
                capacity_weight=args.capacity_weight,
                stability_weight=args.stability_weight,
                stability_holdback_frames=args.stability_holdback_frames,
            )
        for name in names:
            sums[name] += float(losses[name])
        count += 1
        if count >= args.eval_batches:
            break
    reduced = distributed.reduce_sums([*(sums[name] for name in names), float(count)])
    divisor = max(1.0, reduced[-1])
    model.train()
    return {name: reduced[index] / divisor for index, name in enumerate(names)}


def _save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    distributed: DistributedContext,
    args: argparse.Namespace,
    model_config: StageBModelConfig,
    step: int,
    epoch: int,
    consumed_audio_seconds: float,
    best_validation: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": distributed.unwrap(model).state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "args": vars(args),
            "model_config": model_config.__dict__,
            "step": step,
            "epoch": epoch,
            "consumed_audio_seconds": consumed_audio_seconds,
            "best_validation": best_validation,
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
        path,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--policy-tokenizer", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tensorboard-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=50000)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--min-learning-rate", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-fraction", type=float, default=0.05)
    parser.add_argument("--hidden-size", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=12)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--ffn-dim", type=int, default=2048)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max-audio-seconds", type=float, default=8.0)
    parser.add_argument("--validation-records", type=int, default=256)
    parser.add_argument("--eval-interval", type=int, default=500)
    parser.add_argument("--eval-batches", type=int, default=16)
    parser.add_argument("--save-interval", type=int, default=500)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--source-weight", type=float, default=0.1)
    parser.add_argument("--capacity-weight", type=float, default=0.0)
    parser.add_argument("--stability-weight", type=float, default=0.2)
    parser.add_argument("--stability-holdback-frames", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--resume", action="store_true")
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

    train_dataset = StageBAudioDataset(
        args.manifest,
        tokenizer,
        max_audio_seconds=args.max_audio_seconds,
        prefix_training=True,
    )
    valid_dataset = StageBAudioDataset(
        args.manifest,
        tokenizer,
        max_audio_seconds=args.max_audio_seconds,
        prefix_training=False,
    )
    validation_records = min(args.validation_records, max(1, len(train_dataset) // 5))
    valid_indices = list(range(validation_records))
    train_indices = list(range(validation_records, len(train_dataset)))
    if not train_indices:
        raise ValueError("training split is empty")
    train_subset = Subset(train_dataset, train_indices)
    valid_subset = Subset(valid_dataset, valid_indices)
    train_sampler = DistributedSampler(
        train_subset,
        num_replicas=distributed.world_size,
        rank=distributed.rank,
        shuffle=True,
        seed=args.seed,
        drop_last=False,
    )
    valid_sampler = DistributedSampler(
        valid_subset,
        num_replicas=distributed.world_size,
        rank=distributed.rank,
        shuffle=False,
        drop_last=False,
    )
    loader_kwargs = {
        "batch_size": args.batch_size,
        "collate_fn": collate_stage_b,
        "pin_memory": device.type == "cuda",
        "num_workers": args.num_workers,
        "persistent_workers": args.num_workers > 0,
    }
    train_loader = DataLoader(train_subset, sampler=train_sampler, **loader_kwargs)
    valid_loader = DataLoader(valid_subset, sampler=valid_sampler, **loader_kwargs)

    model_config = StageBModelConfig(
        policy_vocab_size=tokenizer.ctc_vocab_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        ffn_dim=args.ffn_dim,
        dropout=args.dropout,
    )
    base_model = CausalAudioStudentV2(model_config).to(device)
    output_dir = Path(args.output_dir)
    last_checkpoint = output_dir / "last.pt"
    resume_value = None
    if args.resume and last_checkpoint.is_file():
        resume_value = torch.load(last_checkpoint, map_location="cpu", weights_only=False)
        base_model.load_state_dict(resume_value["model"])
    model = distributed.wrap(base_model)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
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
    best_validation = float("inf")
    if resume_value is not None:
        optimizer.load_state_dict(resume_value["optimizer"])
        scheduler.load_state_dict(resume_value["scheduler"])
        step = int(resume_value["step"])
        epoch = int(resume_value["epoch"])
        consumed_audio_seconds = float(resume_value.get("consumed_audio_seconds", 0.0))
        best_validation = float(resume_value.get("best_validation", float("inf")))

    if distributed.is_main:
        output_dir.mkdir(parents=True, exist_ok=True)
    distributed.barrier()
    writer = SummaryWriter(args.tensorboard_dir) if distributed.is_main else None
    model.train()
    last_log_time = time.perf_counter()
    last_log_audio = consumed_audio_seconds
    while step < args.max_steps:
        train_sampler.set_epoch(epoch)
        epoch += 1
        for batch in train_loader:
            step += 1
            batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            local_audio_seconds = float(batch["utterance_sample_lengths"].sum()) / 16000.0
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=args.bf16):
                losses = stage_b_losses(
                    model,
                    batch,
                    source_weight=args.source_weight,
                    capacity_weight=args.capacity_weight,
                    stability_weight=args.stability_weight,
                    stability_holdback_frames=args.stability_holdback_frames,
                )
            losses["total"].backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            global_audio = distributed.reduce_sums([local_audio_seconds])[0]
            consumed_audio_seconds += global_audio

            if distributed.is_main and (step == 1 or step % args.log_interval == 0):
                now = time.perf_counter()
                elapsed = max(now - last_log_time, 1e-6)
                audio_rate = (consumed_audio_seconds - last_log_audio) / elapsed
                last_log_time = now
                last_log_audio = consumed_audio_seconds
                metrics = {
                    "step": step,
                    **{name: float(value) for name, value in losses.items()},
                    "grad_norm": float(grad_norm),
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    "audio_seconds_per_second": audio_rate,
                    "consumed_audio_hours": consumed_audio_seconds / 3600.0,
                    "max_cuda_memory_gib": torch.cuda.max_memory_allocated(device) / 2**30
                    if device.type == "cuda"
                    else 0.0,
                }
                print(json.dumps(metrics, sort_keys=True), flush=True)
                assert writer is not None
                for name, value in metrics.items():
                    if name != "step":
                        writer.add_scalar(f"stage_b/{name}", value, step)
                writer.flush()

            if step % args.eval_interval == 0 or step == args.max_steps:
                metrics = evaluate(model, valid_loader, device, distributed, args)
                if distributed.is_main:
                    assert writer is not None
                    for name, value in metrics.items():
                        writer.add_scalar(f"stage_b/valid_{name}", value, step)
                    if metrics["total"] < best_validation:
                        best_validation = metrics["total"]
                        _save_checkpoint(
                            output_dir / "best.pt",
                            model,
                            optimizer,
                            scheduler,
                            distributed,
                            args,
                            model_config,
                            step,
                            epoch,
                            consumed_audio_seconds,
                            best_validation,
                        )
                    writer.flush()

            if distributed.is_main and (step % args.save_interval == 0 or step == args.max_steps):
                _save_checkpoint(
                    last_checkpoint,
                    model,
                    optimizer,
                    scheduler,
                    distributed,
                    args,
                    model_config,
                    step,
                    epoch,
                    consumed_audio_seconds,
                    best_validation,
                )
            if step >= args.max_steps:
                break

    if writer is not None:
        writer.close()
    if distributed.is_main:
        (output_dir / "TRAINING_COMPLETE.json").write_text(
            json.dumps(
                {
                    "status": "complete",
                    "step": step,
                    "best_validation": best_validation,
                    "consumed_audio_hours": consumed_audio_seconds / 3600.0,
                    "checkpoint": str(last_checkpoint),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    distributed.close()


if __name__ == "__main__":
    main()

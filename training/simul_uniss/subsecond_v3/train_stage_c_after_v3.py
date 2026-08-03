"""Train and calibrate an isolated Stage-C gate on the latent Stage-B-v3 Student."""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
import time
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter

from training.simul_uniss.distributed import DistributedContext
from training.simul_uniss.subsecond_v1.stage_c import (
    BayesianSourceSafeCommitGate,
    StageCGateConfig,
    stage_c_losses,
)
from training.simul_uniss.subsecond_v1.train_stage_c import calibrate
from training.simul_uniss.subsecond_v3.stage_c_after_v3 import (
    EVIDENCE_SCHEMA,
    extract_latent_gate_inputs,
    load_latent_student,
)
from training.simul_uniss.subsecond_v3.stage_c_after_v3_data import (
    StageCAfterV3PackedDataset,
    collate_stage_c_after_v3,
)


SCHEMA = "simul_uniss_stage_c_after_v3_training_v1"
SCOPE = "formal_target_microphrase_safe_commit_v2"


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _save_checkpoint(
    path: Path,
    gate: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    distributed: DistributedContext,
    gate_config: StageCGateConfig,
    args: argparse.Namespace,
    *,
    step: int,
    epoch: int,
    best_validation: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    torch.save(
        {
            "schema_version": SCHEMA,
            "gate": distributed.unwrap(gate).state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "gate_config": gate_config.to_dict(),
            "args": vars(args),
            "student_checkpoint": str(Path(args.student_checkpoint).resolve()),
            "scope": SCOPE,
            "evidence_schema": EVIDENCE_SCHEMA,
            "step": step,
            "epoch": epoch,
            "best_validation": best_validation,
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state_all()
            if torch.cuda.is_available()
            else None,
        },
        temporary,
    )
    os.replace(temporary, path)


def _build_inputs(student, batch, gate_config, args):
    with torch.no_grad(), torch.autocast(
        device_type=batch["waveform"].device.type,
        dtype=torch.bfloat16,
        enabled=args.bf16,
    ):
        output = student(
            batch["waveform"],
            batch["waveform_lengths"],
            batch["utterance_sample_lengths"],
        )
    return extract_latent_gate_inputs(
        student,
        output,
        batch,
        gate_config,
        tail_token_count=args.tail_token_count,
        codebook_temperature=args.codebook_temperature,
        codebook_chunk_size=args.codebook_chunk_size,
    )


@torch.no_grad()
def evaluate(student, gate, loader, device, distributed, gate_config, args):
    gate.eval()
    names = ("total", "posterior", "prior", "likelihood", "brier", "positive_rate")
    sums = {name: 0.0 for name in names}
    correct = count = 0.0
    batches = 0
    for batch in loader:
        batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
        inputs = _build_inputs(student, batch, gate_config, args)
        losses = stage_c_losses(gate, inputs["context"], inputs["evidence"], inputs["labels"])
        output = gate(inputs["context"], inputs["evidence"])
        for name in names:
            sums[name] += float(losses[name])
        correct += float(((output["posterior"] >= 0.5) == inputs["labels"].bool()).sum())
        count += float(inputs["labels"].numel())
        batches += 1
        if batches >= args.eval_batches:
            break
    reduced = distributed.reduce_sums(
        [*(sums[name] for name in names), correct, count, float(batches)]
    )
    divisor = max(1.0, reduced[-1])
    result = {name: reduced[index] / divisor for index, name in enumerate(names)}
    result["accuracy"] = reduced[-3] / max(1.0, reduced[-2])
    gate.train()
    return result


@torch.no_grad()
def collect_calibration(student, gate, loader, device, distributed, gate_config, args):
    gate.eval()
    local_logits: list[torch.Tensor] = []
    local_labels: list[torch.Tensor] = []
    for batch_index, batch in enumerate(loader):
        batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
        inputs = _build_inputs(student, batch, gate_config, args)
        output = gate(inputs["context"], inputs["evidence"])
        local_logits.append(output["posterior_logit"].detach().float().cpu())
        local_labels.append(inputs["labels"].detach().float().cpu())
        if batch_index + 1 >= args.calibration_batches:
            break
    payload = (
        torch.cat(local_logits).tolist() if local_logits else [],
        torch.cat(local_labels).tolist() if local_labels else [],
    )
    if distributed.enabled:
        gathered: list[Any] = [None for _ in range(distributed.world_size)]
        dist.all_gather_object(gathered, payload)
    else:
        gathered = [payload]
    if not distributed.is_main:
        return None
    logits = torch.tensor(
        [value for rank_payload in gathered for value in rank_payload[0]],
        dtype=torch.float32,
    )
    labels = torch.tensor(
        [value for rank_payload in gathered for value in rank_payload[1]],
        dtype=torch.float32,
    )
    return logits, labels


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--valid-manifest", required=True)
    parser.add_argument("--student-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tensorboard-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-steps", type=int, default=10_000)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--min-learning-rate", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-fraction", type=float, default=0.05)
    parser.add_argument("--max-audio-seconds", type=float, default=8.0)
    parser.add_argument("--minimum-commit-tokens", type=int, default=2)
    parser.add_argument("--valid-prefixes-per-record", type=int, default=4)
    parser.add_argument("--train-prefixes-per-record", type=int, default=4)
    parser.add_argument("--eval-interval", type=int, default=500)
    parser.add_argument("--eval-batches", type=int, default=4)
    parser.add_argument("--calibration-batches", type=int, default=32)
    parser.add_argument("--save-interval", type=int, default=500)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument("--tail-token-count", type=int, default=2)
    parser.add_argument("--codebook-temperature", type=float, default=0.05)
    parser.add_argument("--codebook-chunk-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20_260_803)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    distributed = DistributedContext.initialize(args.device)
    device = distributed.device
    torch.manual_seed(args.seed + distributed.rank)
    torch.set_float32_matmul_precision("high")
    student, student_metadata = load_latent_student(args.student_checkpoint, device)
    gate_config = StageCGateConfig(minimum_commit_tokens=args.minimum_commit_tokens)
    gate = distributed.wrap(BayesianSourceSafeCommitGate(gate_config).to(device))
    optimizer = torch.optim.AdamW(
        gate.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    warmup_steps = max(1, round(args.max_steps * args.warmup_fraction))

    def schedule(step: int) -> float:
        if step < warmup_steps:
            return max(1e-8, (step + 1) / warmup_steps)
        progress = (step - warmup_steps) / max(1, args.max_steps - warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
        floor = args.min_learning_rate / args.learning_rate
        return floor + (1.0 - floor) * cosine

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    train_dataset = StageCAfterV3PackedDataset(
        args.train_manifest,
        max_audio_seconds=args.max_audio_seconds,
        prefixes_per_record=args.train_prefixes_per_record,
        random_prefix=True,
    )
    valid_dataset = StageCAfterV3PackedDataset(
        args.valid_manifest,
        max_audio_seconds=args.max_audio_seconds,
        prefixes_per_record=args.valid_prefixes_per_record,
        random_prefix=False,
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
    loader_args: dict[str, object] = {
        "batch_size": args.batch_size,
        "collate_fn": collate_stage_c_after_v3,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": args.num_workers > 0,
    }
    if args.num_workers > 0:
        loader_args["prefetch_factor"] = args.prefetch_factor
    train_loader = DataLoader(train_dataset, sampler=train_sampler, **loader_args)
    valid_loader = DataLoader(valid_dataset, sampler=valid_sampler, **loader_args)

    output_dir = Path(args.output_dir)
    last_checkpoint = output_dir / "last.pt"
    best_checkpoint = output_dir / "best.pt"
    step = epoch = 0
    best_validation = float("inf")
    if args.resume and last_checkpoint.is_file():
        value = torch.load(last_checkpoint, map_location="cpu", weights_only=False)
        distributed.unwrap(gate).load_state_dict(value["gate"])
        optimizer.load_state_dict(value["optimizer"])
        scheduler.load_state_dict(value["scheduler"])
        step = int(value["step"])
        epoch = int(value["epoch"])
        best_validation = float(value["best_validation"])
    if distributed.is_main:
        output_dir.mkdir(parents=True, exist_ok=True)
    distributed.barrier()
    writer = SummaryWriter(args.tensorboard_dir) if distributed.is_main else None
    gate.train()
    last_log = time.perf_counter()
    while step < args.max_steps:
        train_sampler.set_epoch(epoch)
        epoch += 1
        for batch in train_loader:
            step += 1
            batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            inputs = _build_inputs(student, batch, gate_config, args)
            optimizer.zero_grad(set_to_none=True)
            losses = stage_c_losses(gate, inputs["context"], inputs["evidence"], inputs["labels"])
            losses["total"].backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(gate.parameters(), 5.0)
            optimizer.step()
            scheduler.step()
            if distributed.is_main and (step == 1 or step % args.log_interval == 0):
                now = time.perf_counter()
                metrics = {
                    "step": step,
                    **{name: float(value) for name, value in losses.items()},
                    "grad_norm": float(grad_norm),
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    "steps_per_second": args.log_interval / max(now - last_log, 1e-6),
                }
                last_log = now
                print(json.dumps(metrics, sort_keys=True), flush=True)
                if writer is not None:
                    for name, value in metrics.items():
                        if name != "step":
                            writer.add_scalar(f"stage_c_after_v3/train_{name}", value, step)
                    writer.flush()
            if step % args.eval_interval == 0 or step == args.max_steps:
                validation = evaluate(
                    student, gate, valid_loader, device, distributed, gate_config, args
                )
                if distributed.is_main:
                    print(json.dumps({"step": step, "validation": validation}, sort_keys=True), flush=True)
                    if writer is not None:
                        for name, value in validation.items():
                            writer.add_scalar(f"stage_c_after_v3/valid_{name}", value, step)
                    if validation["posterior"] < best_validation:
                        best_validation = validation["posterior"]
                        _save_checkpoint(
                            best_checkpoint, gate, optimizer, scheduler, distributed,
                            gate_config, args, step=step, epoch=epoch,
                            best_validation=best_validation,
                        )
                distributed.barrier()
            if step % args.save_interval == 0 or step == args.max_steps:
                if distributed.is_main:
                    _save_checkpoint(
                        last_checkpoint, gate, optimizer, scheduler, distributed,
                        gate_config, args, step=step, epoch=epoch,
                        best_validation=best_validation,
                    )
                distributed.barrier()
            if step >= args.max_steps:
                break

    distributed.barrier()
    best_value = torch.load(best_checkpoint, map_location="cpu", weights_only=False)
    distributed.unwrap(gate).load_state_dict(best_value["gate"])
    values = collect_calibration(
        student, gate, valid_loader, device, distributed, gate_config, args
    )
    if distributed.is_main and values is not None:
        calibration = calibrate(*values)
        calibration.update(
            {
                "schema_version": "simul_uniss_stage_c_after_v3_calibration_v1",
                "scope": SCOPE,
                "evidence_schema": EVIDENCE_SCHEMA,
                "student_checkpoint": str(Path(args.student_checkpoint).resolve()),
                "student_metadata": student_metadata,
                "gate_checkpoint": str(best_checkpoint.resolve()),
            }
        )
        _atomic_json(output_dir / "calibration.json", calibration)
        _atomic_json(
            output_dir / "STAGE_C_AFTER_V3_COMPLETE.json",
            {
                "schema_version": SCHEMA,
                "status": "complete",
                "scope": SCOPE,
                "evidence_schema": EVIDENCE_SCHEMA,
                "step": step,
                "best_validation": best_validation,
                "student_checkpoint": str(Path(args.student_checkpoint).resolve()),
                "checkpoint": str(best_checkpoint.resolve()),
                "calibration": str((output_dir / "calibration.json").resolve()),
            },
        )
        if writer is not None:
            writer.add_scalar("stage_c_after_v3/calibration_temperature", calibration["temperature"], step)
            writer.add_scalar("stage_c_after_v3/calibration_brier", calibration["calibrated_brier"], step)
            writer.add_scalar("stage_c_after_v3/calibration_ece", calibration["calibrated_ece"], step)
            writer.flush()
            writer.close()
    distributed.barrier()
    distributed.close()


if __name__ == "__main__":
    main()

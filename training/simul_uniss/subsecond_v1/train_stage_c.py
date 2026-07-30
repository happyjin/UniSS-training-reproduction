"""Train and calibrate the isolated Stage-C source safe-commit Bayesian pilot."""

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
from torch.nn import functional as F
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter

from training.simul_uniss.distributed import DistributedContext
from training.simul_uniss.subsecond_v1.model import CausalAudioStudentV2, StageBModelConfig
from training.simul_uniss.subsecond_v1.stage_c import (
    BayesianSourceSafeCommitGate,
    StageCGateConfig,
    extract_gate_inputs,
    stage_c_losses,
)
from training.simul_uniss.subsecond_v1.stage_c_data import (
    StageCSourceCommitDataset,
    collate_stage_c,
)


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


def _build_gate_inputs(
    student: CausalAudioStudentV2,
    batch: dict[str, torch.Tensor],
    gate_config: StageCGateConfig,
    *,
    bf16: bool,
) -> dict[str, torch.Tensor]:
    with torch.no_grad(), torch.autocast(
        device_type=batch["waveform"].device.type,
        dtype=torch.bfloat16,
        enabled=bf16,
    ):
        output = student(
            batch["waveform"],
            batch["waveform_lengths"],
            batch["utterance_sample_lengths"],
        )
    return extract_gate_inputs(
        output,
        batch,
        minimum_commit_tokens=gate_config.minimum_commit_tokens,
        segment_frames=student.config.segment_frames,
    )


@torch.no_grad()
def evaluate(
    student: CausalAudioStudentV2,
    gate: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    distributed: DistributedContext,
    gate_config: StageCGateConfig,
    args: argparse.Namespace,
) -> dict[str, float]:
    gate.eval()
    names = ("total", "posterior", "prior", "likelihood", "brier", "positive_rate")
    sums = {name: 0.0 for name in names}
    correct = 0.0
    count = 0.0
    batches = 0
    for batch in loader:
        batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
        inputs = _build_gate_inputs(student, batch, gate_config, bf16=args.bf16)
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
def collect_calibration(
    student: CausalAudioStudentV2,
    gate: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    distributed: DistributedContext,
    gate_config: StageCGateConfig,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    gate.eval()
    local_logits: list[torch.Tensor] = []
    local_labels: list[torch.Tensor] = []
    for batch_index, batch in enumerate(loader):
        batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
        inputs = _build_gate_inputs(student, batch, gate_config, bf16=args.bf16)
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
        [value for rank_payload in gathered for value in rank_payload[0]], dtype=torch.float32
    )
    labels = torch.tensor(
        [value for rank_payload in gathered for value in rank_payload[1]], dtype=torch.float32
    )
    return logits, labels


def _ece(probabilities: torch.Tensor, labels: torch.Tensor, bins: int = 15) -> float:
    result = 0.0
    for index in range(bins):
        lower = index / bins
        upper = (index + 1) / bins
        mask = (probabilities >= lower) & (
            probabilities <= upper if index == bins - 1 else probabilities < upper
        )
        if mask.any():
            result += float(mask.float().mean()) * abs(
                float(probabilities[mask].mean()) - float(labels[mask].mean())
            )
    return result


def _operating_threshold(
    probabilities: torch.Tensor, labels: torch.Tensor, target_precision: float
) -> dict[str, float]:
    order = probabilities.argsort(descending=True)
    scores = probabilities[order]
    truth = labels[order]
    true_positive = truth.cumsum(0)
    predicted_positive = torch.arange(1, len(scores) + 1, dtype=torch.float32)
    precision = true_positive / predicted_positive
    recall = true_positive / labels.sum().clamp_min(1)
    valid = torch.nonzero(precision >= target_precision).reshape(-1)
    if valid.numel() == 0:
        return {"threshold": 1.0, "precision": 1.0, "recall": 0.0}
    candidate_recall = recall[valid]
    best = int(valid[int(candidate_recall.argmax())])
    return {
        "threshold": float(scores[best]),
        "precision": float(precision[best]),
        "recall": float(recall[best]),
    }


def calibrate(logits: torch.Tensor, labels: torch.Tensor) -> dict[str, Any]:
    log_temperature = torch.zeros((), requires_grad=True)
    optimizer = torch.optim.LBFGS([log_temperature], lr=0.2, max_iter=50, line_search_fn="strong_wolfe")

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        temperature = log_temperature.exp().clamp(0.05, 20.0)
        loss = F.binary_cross_entropy_with_logits(logits / temperature, labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    temperature = float(log_temperature.exp().clamp(0.05, 20.0).detach())
    raw = torch.sigmoid(logits)
    calibrated = torch.sigmoid(logits / temperature)
    return {
        "schema_version": "simul_uniss_stage_c_source_proxy_calibration_v1",
        "scope": "source_glm_commit_proxy_not_target_microphrase_safe_commit",
        "records": labels.numel(),
        "positive_rate": float(labels.mean()),
        "temperature": temperature,
        "raw_brier": float((raw - labels).square().mean()),
        "calibrated_brier": float((calibrated - labels).square().mean()),
        "raw_ece": _ece(raw, labels),
        "calibrated_ece": _ece(calibrated, labels),
        "operating_points": {
            name: _operating_threshold(calibrated, labels, target)
            for name, target in (("fast", 0.75), ("balanced", 0.88), ("quality", 0.95))
        },
    }


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
    torch.save(
        {
            "gate": distributed.unwrap(gate).state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "gate_config": gate_config.to_dict(),
            "args": vars(args),
            "student_checkpoint": str(Path(args.student_checkpoint).resolve()),
            "scope": "source_glm_commit_proxy_not_target_microphrase_safe_commit",
            "step": step,
            "epoch": epoch,
            "best_validation": best_validation,
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
        path,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--valid-manifest", required=True)
    parser.add_argument("--student-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tensorboard-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-steps", type=int, default=10000)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--min-learning-rate", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-fraction", type=float, default=0.05)
    parser.add_argument("--max-audio-seconds", type=float, default=8.0)
    parser.add_argument("--minimum-commit-tokens", type=int, default=2)
    parser.add_argument("--safety-margin-ms", type=int, default=80)
    parser.add_argument("--valid-prefixes-per-record", type=int, default=4)
    parser.add_argument("--eval-interval", type=int, default=500)
    parser.add_argument("--eval-batches", type=int, default=4)
    parser.add_argument("--calibration-batches", type=int, default=32)
    parser.add_argument("--save-interval", type=int, default=500)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260731)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    distributed = DistributedContext.initialize(args.device)
    device = distributed.device
    torch.manual_seed(args.seed + distributed.rank)
    torch.set_float32_matmul_precision("high")

    student_value = torch.load(
        args.student_checkpoint, map_location="cpu", weights_only=False, mmap=True
    )
    student_config = StageBModelConfig.from_dict(student_value["model_config"])
    student = CausalAudioStudentV2(student_config).to(device).eval()
    student.load_state_dict(student_value["model"], strict=True)
    student.requires_grad_(False)
    del student_value

    gate_config = StageCGateConfig(minimum_commit_tokens=args.minimum_commit_tokens)
    base_gate = BayesianSourceSafeCommitGate(gate_config).to(device)
    gate = distributed.wrap(base_gate)
    optimizer = torch.optim.AdamW(
        gate.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
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
    train_dataset = StageCSourceCommitDataset(
        args.train_manifest,
        max_audio_seconds=args.max_audio_seconds,
        safety_margin_ms=args.safety_margin_ms,
        random_prefix=True,
    )
    valid_dataset = StageCSourceCommitDataset(
        args.valid_manifest,
        max_audio_seconds=args.max_audio_seconds,
        safety_margin_ms=args.safety_margin_ms,
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
    loader_kwargs = {
        "batch_size": args.batch_size,
        "collate_fn": collate_stage_c,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": args.num_workers > 0,
    }
    train_loader = DataLoader(train_dataset, sampler=train_sampler, **loader_kwargs)
    valid_loader = DataLoader(valid_dataset, sampler=valid_sampler, **loader_kwargs)

    output_dir = Path(args.output_dir)
    last_checkpoint = output_dir / "last.pt"
    best_checkpoint = output_dir / "best.pt"
    step = 0
    epoch = 0
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
            inputs = _build_gate_inputs(student, batch, gate_config, bf16=args.bf16)
            optimizer.zero_grad(set_to_none=True)
            losses = stage_c_losses(
                gate, inputs["context"], inputs["evidence"], inputs["labels"]
            )
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
                            writer.add_scalar(f"train/{name}", value, step)

            if step % args.eval_interval == 0 or step == args.max_steps:
                validation = evaluate(
                    student, gate, valid_loader, device, distributed, gate_config, args
                )
                if distributed.is_main:
                    print(json.dumps({"step": step, "validation": validation}, sort_keys=True))
                    if writer is not None:
                        for name, value in validation.items():
                            writer.add_scalar(f"validation/{name}", value, step)
                    if validation["posterior"] < best_validation:
                        best_validation = validation["posterior"]
                        _save_checkpoint(
                            best_checkpoint,
                            gate,
                            optimizer,
                            scheduler,
                            distributed,
                            gate_config,
                            args,
                            step=step,
                            epoch=epoch,
                            best_validation=best_validation,
                        )
                distributed.barrier()
            if step % args.save_interval == 0 or step == args.max_steps:
                if distributed.is_main:
                    _save_checkpoint(
                        last_checkpoint,
                        gate,
                        optimizer,
                        scheduler,
                        distributed,
                        gate_config,
                        args,
                        step=step,
                        epoch=epoch,
                        best_validation=best_validation,
                    )
                distributed.barrier()
            if step >= args.max_steps:
                break

    distributed.barrier()
    best_value = torch.load(best_checkpoint, map_location="cpu", weights_only=False)
    distributed.unwrap(gate).load_state_dict(best_value["gate"])
    calibration_values = collect_calibration(
        student, gate, valid_loader, device, distributed, gate_config, args
    )
    if distributed.is_main and calibration_values is not None:
        calibration = calibrate(*calibration_values)
        calibration["student_checkpoint"] = str(Path(args.student_checkpoint).resolve())
        calibration["gate_checkpoint"] = str(best_checkpoint.resolve())
        _atomic_json(output_dir / "calibration.json", calibration)
        _atomic_json(
            output_dir / "STAGE_C_SOURCE_PROXY_COMPLETE.json",
            {
                "schema_version": "simul_uniss_stage_c_source_proxy_complete_v1",
                "status": "complete",
                "scope": "source_glm_commit_proxy_not_target_microphrase_safe_commit",
                "warning": "Formal Stage C still requires bilingual target-support alignment.",
                "step": step,
                "best_validation": best_validation,
                "checkpoint": str(best_checkpoint.resolve()),
                "calibration": str((output_dir / "calibration.json").resolve()),
            },
        )
        if writer is not None:
            writer.add_scalar("calibration/temperature", calibration["temperature"], step)
            writer.add_scalar("calibration/brier", calibration["calibrated_brier"], step)
            writer.add_scalar("calibration/ece", calibration["calibrated_ece"], step)
            writer.flush()
            writer.close()
    distributed.barrier()
    distributed.close()


if __name__ == "__main__":
    main()

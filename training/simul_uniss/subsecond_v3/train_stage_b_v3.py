"""Fine-tune Stage-B-v3 on balanced exact-prefix-hidden and clone supervision."""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
import time
from pathlib import Path

import torch
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
from training.simul_uniss.subsecond_v2.train_stage_b_v2 import (
    _module,
    stage_b_v2_losses,
)
from training.simul_uniss.subsecond_v3.stage_b_v3_data import (
    StageBV3MixedDataset,
    collate_stage_b_v3,
)


SCHEMA = "simul_uniss_stage_b_v3_training_v1"


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


def _harmonic(left: float, right: float) -> float:
    return 2.0 * left * right / max(1e-12, left + right)


def _save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    args: argparse.Namespace,
    config: LatentStageBModelConfig,
    *,
    step: int,
    epoch: int,
    metrics: dict[str, float],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    torch.save(
        {
            "schema_version": SCHEMA,
            "model": _module(model).state_dict(),
            "args": vars(args),
            "model_config": config.__dict__,
            "codebook_model": str(Path(args.codebook_model).resolve()),
            "codebook_key": args.codebook_key,
            "step": step,
            "epoch": epoch,
            "selection_metrics": metrics,
        },
        temporary,
    )
    os.replace(temporary, path)


def _loss_arguments(args: argparse.Namespace, *, auxiliary_scale: float, consistency: bool):
    return {
        "hidden_weight": args.hidden_weight,
        "cosine_weight": args.cosine_weight,
        "codebook_ce_weight": args.codebook_ce_weight,
        "margin_weight": args.margin_weight,
        "full_context_weight": args.full_context_weight,
        "source_weight": args.source_weight,
        "capacity_weight": args.capacity_weight,
        "stability_weight": args.stability_weight,
        "consistency_weight": args.consistency_weight,
        "auxiliary_scale": auxiliary_scale,
        "temperature": args.codebook_temperature,
        "margin": args.codebook_margin,
        "quantize_chunk_size": args.quantize_chunk_size,
        "compute_consistency": consistency,
        "chunk_samples": args.chunk_samples,
    }


@torch.no_grad()
def evaluate(model, loader, device, distributed, args) -> dict[str, float]:
    model.eval()
    sums: dict[str, float] = {}
    batches = 0
    correct = torch.zeros(2, 2, dtype=torch.float64, device=device)
    tokens = torch.zeros(2, 2, dtype=torch.float64, device=device)
    for batch in loader:
        batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=args.bf16):
            losses, details = stage_b_v2_losses(
                model,
                batch,
                **_loss_arguments(args, auxiliary_scale=1.0, consistency=True),
            )
        for name, value in losses.items():
            sums[name] = sums.get(name, 0.0) + float(value)
        predicted = _module(model).quantize(
            details["glm_latent"], chunk_size=args.quantize_chunk_size
        )
        token_mask = details["mask"]
        matches = predicted.eq(details["targets"]) & token_mask
        for direction in range(2):
            for supervision in range(2):
                rows = (batch["direction_id"] == direction) & (
                    batch["supervision_id"] == supervision
                )
                if bool(rows.any()):
                    correct[direction, supervision] += matches[rows].sum()
                    tokens[direction, supervision] += token_mask[rows].sum()
        batches += 1
        if batches >= args.eval_batches:
            break
    names = sorted(sums)
    values = [*(sums[name] for name in names), float(batches)]
    values.extend(correct.reshape(-1).tolist())
    values.extend(tokens.reshape(-1).tolist())
    reduced = distributed.reduce_sums(values)
    divisor = max(1.0, reduced[len(names)])
    result = {name: reduced[index] / divisor for index, name in enumerate(names)}
    start = len(names) + 1
    group_correct = reduced[start : start + 4]
    group_tokens = reduced[start + 4 : start + 8]
    agreements = [
        group_correct[index] / max(1.0, group_tokens[index]) for index in range(4)
    ]
    result.update(
        {
            "agreement_eng_cmn_prefix": agreements[0],
            "agreement_eng_cmn_clone": agreements[1],
            "agreement_cmn_eng_prefix": agreements[2],
            "agreement_cmn_eng_clone": agreements[3],
        }
    )
    direction_eng = (
        group_correct[0] + group_correct[1]
    ) / max(1.0, group_tokens[0] + group_tokens[1])
    direction_cmn = (
        group_correct[2] + group_correct[3]
    ) / max(1.0, group_tokens[2] + group_tokens[3])
    prefix = (group_correct[0] + group_correct[2]) / max(
        1.0, group_tokens[0] + group_tokens[2]
    )
    clone = (group_correct[1] + group_correct[3]) / max(
        1.0, group_tokens[1] + group_tokens[3]
    )
    result.update(
        {
            "agreement_eng_cmn": direction_eng,
            "agreement_cmn_eng": direction_cmn,
            "agreement_prefix": prefix,
            "agreement_clone": clone,
            "direction_hmean": _harmonic(direction_eng, direction_cmn),
            "supervision_hmean": _harmonic(prefix, clone),
        }
    )
    result["selection_score"] = (
        result["target_agreement"]
        + result["direction_hmean"]
        + result["supervision_hmean"]
    ) / 3.0
    model.train()
    return result


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
    parser.add_argument("--initialize-from", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-steps", type=int, default=10_000)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--min-learning-rate", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-fraction", type=float, default=0.03)
    parser.add_argument("--representation-only-steps", type=int, default=2_000)
    parser.add_argument("--auxiliary-ramp-steps", type=int, default=2_000)
    parser.add_argument("--max-audio-seconds", type=float, default=8.0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--eval-interval", type=int, default=500)
    parser.add_argument("--eval-batches", type=int, default=8)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--keep-top-k", type=int, default=3)
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
    parser.add_argument("--seed", type=int, default=20_260_803)
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
    initial = torch.load(args.initialize_from, map_location="cpu", weights_only=False)
    config = LatentStageBModelConfig.from_dict(initial["model_config"])
    if config.policy_vocab_size != tokenizer.ctc_vocab_size:
        raise ValueError("policy tokenizer does not match initialization checkpoint")
    base_model = LatentCausalAudioStudent(config, codebook).to(device)
    missing, unexpected = base_model.load_state_dict(initial["model"], strict=False)
    if missing or unexpected:
        raise ValueError(f"initialization mismatch: missing={missing}, unexpected={unexpected}")
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
    train_dataset = StageBV3MixedDataset(
        args.sidecar_manifest,
        args.source_manifest,
        prefix_training=True,
        **dataset_args,
    )
    valid_dataset = StageBV3MixedDataset(
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
        "collate_fn": collate_stage_b_v3,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": args.num_workers > 0,
    }
    train_loader = DataLoader(train_dataset, sampler=train_sampler, **loader_args)
    valid_loader = DataLoader(valid_dataset, sampler=valid_sampler, **loader_args)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
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
    candidate_dir = output_dir / "candidates"
    if distributed.is_main:
        candidate_dir.mkdir(parents=True, exist_ok=True)
    distributed.barrier()
    writer = SummaryWriter(args.tensorboard_dir) if distributed.is_main else None
    step = epoch = 0
    top_candidates: list[tuple[float, Path, dict[str, float]]] = []
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
                    (step - args.representation_only_steps)
                    / max(1, args.auxiliary_ramp_steps),
                )
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=args.bf16):
                losses, _ = stage_b_v2_losses(
                    model,
                    batch,
                    **_loss_arguments(
                        args,
                        auxiliary_scale=auxiliary_scale,
                        consistency=step % args.consistency_interval == 0,
                    ),
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
                        writer.add_scalar(f"stage_b_v3/{name}", value, step)
                writer.flush()
            if step % args.eval_interval == 0 or step == args.max_steps:
                metrics = evaluate(model, valid_loader, device, distributed, args)
                if distributed.is_main:
                    assert writer is not None
                    print(json.dumps({"step": step, "validation": metrics}, sort_keys=True), flush=True)
                    for name, value in metrics.items():
                        writer.add_scalar(f"stage_b_v3/valid_{name}", value, step)
                    candidate = candidate_dir / f"step_{step:06d}.pt"
                    _save_checkpoint(
                        candidate,
                        model,
                        args,
                        config,
                        step=step,
                        epoch=epoch,
                        metrics=metrics,
                    )
                    top_candidates.append((metrics["selection_score"], candidate, metrics))
                    top_candidates.sort(key=lambda value: value[0], reverse=True)
                    while len(top_candidates) > args.keep_top_k:
                        _, removed, _ = top_candidates.pop()
                        removed.unlink(missing_ok=True)
                    if top_candidates[0][1] == candidate:
                        _save_checkpoint(
                            output_dir / "best_agreement.pt",
                            model,
                            args,
                            config,
                            step=step,
                            epoch=epoch,
                            metrics=metrics,
                        )
                    _atomic_json(
                        output_dir / "CANDIDATES.json",
                        {
                            "schema_version": SCHEMA,
                            "candidates": [
                                {
                                    "score": score,
                                    "checkpoint": str(path.resolve()),
                                    "metrics": candidate_metrics,
                                }
                                for score, path, candidate_metrics in top_candidates
                            ],
                        },
                    )
                    writer.flush()
            if step >= args.max_steps:
                break
    if distributed.is_main:
        _save_checkpoint(
            output_dir / "last.pt",
            model,
            args,
            config,
            step=step,
            epoch=epoch,
            metrics=top_candidates[0][2] if top_candidates else {},
        )
        _atomic_json(
            output_dir / "TRAINING_COMPLETE.json",
            {
                "schema_version": SCHEMA,
                "status": "complete",
                "step": step,
                "initialize_from": str(Path(args.initialize_from).resolve()),
                "best_agreement_checkpoint": str((output_dir / "best_agreement.pt").resolve()),
                "top_candidates": [str(path.resolve()) for _, path, _ in top_candidates],
            },
        )
    if writer is not None:
        writer.close()
    distributed.close()


if __name__ == "__main__":
    main()

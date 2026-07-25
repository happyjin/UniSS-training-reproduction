"""Train the causal audio Streaming GLM student on reconstructed/raw audio."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, DistributedSampler, Subset
from torch.utils.tensorboard import SummaryWriter

from training.simul_uniss.audio_streaming_student import (
    AudioStreamingStudent,
    AudioStudentDataset,
    audio_student_losses,
    collate_audio_student,
)
from training.simul_uniss.policy_tokenizer import PolicyTokenizer
from training.simul_uniss.distributed import DistributedContext


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_batches: int,
    distributed: DistributedContext,
) -> dict[str, float]:
    model.eval()
    names = ("total", "teacher", "source", "target")
    sums = {name: 0.0 for name in names}
    count = 0
    for batch in loader:
        batch = {key: value.to(device) for key, value in batch.items()}
        losses = audio_student_losses(model, batch)
        for name in names:
            sums[name] += float(losses[name])
        count += 1
        if count >= max_batches:
            break
    reduced = distributed.reduce_sums([*(sums[name] for name in names), float(count)])
    count = int(reduced[-1])
    return {name: reduced[index] / max(count, 1) for index, name in enumerate(names)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--policy-tokenizer", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tensorboard-dir", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--max-audio-seconds", type=float, default=12.0)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--save-interval", type=int, default=100)
    parser.add_argument("--validation-records", type=int, default=128)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--eval-batches", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260722)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    distributed = DistributedContext.initialize(args.device)
    torch.manual_seed(args.seed + distributed.rank)
    device = distributed.device
    tokenizer = PolicyTokenizer(args.policy_tokenizer)
    dataset = AudioStudentDataset(
        args.manifest,
        tokenizer,
        max_audio_seconds=args.max_audio_seconds,
        prefix_training=True,
    )
    validation_records = min(max(1, args.validation_records), max(1, len(dataset) // 5))
    valid_indices = list(range(validation_records))
    train_indices = list(range(validation_records, len(dataset))) or valid_indices
    train_subset = Subset(dataset, train_indices)
    valid_subset = Subset(dataset, valid_indices)
    train_sampler = (
        DistributedSampler(
            train_subset,
            num_replicas=distributed.world_size,
            rank=distributed.rank,
            shuffle=True,
            seed=args.seed,
        )
        if distributed.enabled
        else None
    )
    valid_sampler = (
        DistributedSampler(
            valid_subset,
            num_replicas=distributed.world_size,
            rank=distributed.rank,
            shuffle=False,
        )
        if distributed.enabled
        else None
    )
    train_loader = DataLoader(
        train_subset,
        batch_size=args.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        collate_fn=collate_audio_student,
    )
    valid_loader = DataLoader(
        valid_subset,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=valid_sampler,
        collate_fn=collate_audio_student,
    )
    base_model = AudioStreamingStudent(
        tokenizer.ctc_vocab_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
    ).to(device)
    model = distributed.wrap(base_model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    writer = SummaryWriter(args.tensorboard_dir) if distributed.is_main else None
    output_dir = Path(args.output_dir)
    if distributed.is_main:
        output_dir.mkdir(parents=True, exist_ok=True)
    distributed.barrier()
    step = 0
    epoch = 0
    best_validation = float("inf")
    while step < args.max_steps:
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        epoch += 1
        for batch in train_loader:
            step += 1
            batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            losses = audio_student_losses(model, batch)
            losses["total"].backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if distributed.is_main and (step % args.log_interval == 0 or step == 1):
                assert writer is not None
                for name, value in losses.items():
                    writer.add_scalar(f"stage1_audio/train_{name}_loss", float(value), step)
                writer.add_scalar("stage1_audio/grad_norm", float(grad_norm), step)
                writer.flush()
                print(
                    json.dumps({"step": step, **{name: float(value) for name, value in losses.items()}}),
                    flush=True,
                )
            if step % args.eval_interval == 0 or step == args.max_steps:
                metrics = evaluate(model, valid_loader, device, args.eval_batches, distributed)
                if writer is not None:
                    for name, value in metrics.items():
                        writer.add_scalar(f"stage1_audio/valid_{name}_loss", value, step)
                    writer.flush()
                if metrics["total"] < best_validation:
                    best_validation = metrics["total"]
                    if distributed.is_main:
                        torch.save(
                            {
                                "model": distributed.unwrap(model).state_dict(),
                                "optimizer": optimizer.state_dict(),
                                "args": vars(args),
                                "step": step,
                            },
                            output_dir / "best.pt",
                        )
                model.train()
            if distributed.is_main and (step % args.save_interval == 0 or step == args.max_steps):
                torch.save(
                    {
                        "model": distributed.unwrap(model).state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "args": vars(args),
                        "step": step,
                    },
                    output_dir / "last.pt",
                )
            if step >= args.max_steps:
                break
    if writer is not None:
        writer.close()
    if distributed.is_main:
        print(
            json.dumps(
                {"status": "complete", "output": str(output_dir / "last.pt"), "best_validation": best_validation}
            )
        )
    distributed.close()


if __name__ == "__main__":
    main()

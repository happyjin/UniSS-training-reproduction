"""Train the bootstrap causal student and its Source/Target CTC heads."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter

from training.simul_uniss.policy_tokenizer import PolicyTokenizer
from training.simul_uniss.streaming_student import (
    StreamingStudentDataset,
    StreamingTokenStudent,
    collate_student_batch,
    ctc_loss,
)


def resolve_checkpoint_anchor(root: Path) -> dict[str, object]:
    pointer = root / "latest_checkpointed_iteration.txt"
    if not pointer.is_file():
        raise FileNotFoundError(pointer)
    iteration = int(pointer.read_text(encoding="utf-8").strip())
    iteration_dir = root / f"iter_{iteration:07d}"
    if not iteration_dir.is_dir():
        raise FileNotFoundError(iteration_dir)
    return {"root": str(root.resolve()), "iteration": iteration, "iteration_dir": str(iteration_dir.resolve())}


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def compute_losses(model: StreamingTokenStudent, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    outputs = model(batch["source_bicodec"], batch["input_lengths"])
    lengths = outputs["output_lengths"]
    teacher = ctc_loss(
        outputs["teacher_glm_logits"], batch["teacher_glm"], lengths, batch["teacher_glm_lengths"]
    )
    source = ctc_loss(
        outputs["source_ctc_logits"], batch["source_policy"], lengths, batch["source_policy_lengths"]
    )
    target = ctc_loss(
        outputs["target_ctc_logits"], batch["target_policy"], lengths, batch["target_policy_lengths"]
    )
    total = teacher + 0.3 * source + 0.3 * target
    return {"total": total, "teacher_ctc": teacher, "source_ctc": source, "target_ctc": target}


@torch.no_grad()
def evaluate(
    model: StreamingTokenStudent,
    loader: DataLoader,
    device: torch.device,
    max_batches: int,
) -> dict[str, float]:
    model.eval()
    sums = {"total": 0.0, "teacher_ctc": 0.0, "source_ctc": 0.0, "target_ctc": 0.0}
    count = 0
    for batch in loader:
        losses = compute_losses(model, move_batch(batch, device))
        for name, value in losses.items():
            sums[name] += float(value)
        count += 1
        if count >= max_batches:
            break
    return {name: value / max(count, 1) for name, value in sums.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schedules", required=True)
    parser.add_argument("--policy-tokenizer", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tensorboard-dir", required=True)
    parser.add_argument("--qwen-checkpoint-root", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--max-source-tokens", type=int, default=1024)
    parser.add_argument("--validation-records", type=int, default=128)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--eval-batches", type=int, default=8)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--save-interval", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260722)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    tokenizer = PolicyTokenizer(args.policy_tokenizer)
    dataset = StreamingStudentDataset(
        args.schedules,
        tokenizer,
        max_source_tokens=args.max_source_tokens,
        prefix_training=True,
    )
    validation_records = min(max(1, args.validation_records), max(1, len(dataset) // 5))
    valid_indices = list(range(validation_records))
    train_indices = list(range(validation_records, len(dataset))) or valid_indices
    train_loader = DataLoader(
        Subset(dataset, train_indices),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_student_batch,
        num_workers=0,
    )
    valid_loader = DataLoader(
        Subset(dataset, valid_indices),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_student_batch,
        num_workers=0,
    )
    model = StreamingTokenStudent(
        tokenizer.ctc_vocab_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    anchor = resolve_checkpoint_anchor(Path(args.qwen_checkpoint_root))
    metadata = {
        "schema_version": "simul_uniss_streaming_student_bootstrap_v1",
        "qwen_checkpoint_anchor": anchor,
        "policy_tokenizer": str(Path(args.policy_tokenizer).resolve()),
        "args": vars(args),
        "note": "Bootstrap input is source_bicodec; formal student input will be source audio.",
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    writer = SummaryWriter(args.tensorboard_dir)
    step = 0
    best_validation = float("inf")
    model.train()
    while step < args.max_steps:
        for batch in train_loader:
            step += 1
            optimizer.zero_grad(set_to_none=True)
            losses = compute_losses(model, move_batch(batch, device))
            losses["total"].backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if step % args.log_interval == 0 or step == 1:
                for name, value in losses.items():
                    writer.add_scalar(f"stage1/train_{name}", float(value), step)
                writer.add_scalar("stage1/grad_norm", float(grad_norm), step)
                writer.add_scalar("stage1/learning_rate", optimizer.param_groups[0]["lr"], step)
                writer.flush()
                print(
                    json.dumps(
                        {"step": step, **{name: float(value) for name, value in losses.items()}},
                        sort_keys=True,
                    ),
                    flush=True,
                )
            if step % args.eval_interval == 0 or step == args.max_steps:
                metrics = evaluate(model, valid_loader, device, args.eval_batches)
                for name, value in metrics.items():
                    writer.add_scalar(f"stage1/valid_{name}", value, step)
                writer.flush()
                if metrics["total"] < best_validation:
                    best_validation = metrics["total"]
                    torch.save(
                        {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": step, "metadata": metadata},
                        output_dir / "best.pt",
                    )
                model.train()
            if step % args.save_interval == 0 or step == args.max_steps:
                torch.save(
                    {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": step, "metadata": metadata},
                    output_dir / "last.pt",
                )
            if step >= args.max_steps:
                break
    writer.close()
    print(json.dumps({"status": "complete", "step": step, "best_validation": best_validation}))


if __name__ == "__main__":
    main()

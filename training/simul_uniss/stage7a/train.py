"""Train a Stage6-initialized WAIT/WRITE head with continued SFT or GRPO."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import time
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoModelForCausalLM

from training.simul_uniss.distributed import DistributedContext
from training.simul_uniss.stage7a.data import (
    ActionBatch,
    batch_action_samples,
    iter_action_samples,
    iter_action_samples_once,
)
from training.simul_uniss.stage7a.policy import ActionHead, grpo_action_loss

CHECKPOINT_SCHEMA = "simul_uniss_stage7a_action_head_v1"


def dtype_from_name(name: str) -> torch.dtype:
    return {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[name]


def git_revision(repo_root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def file_metadata(path: Path) -> dict[str, object]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def load_base_and_head(
    model_path: Path,
    *,
    device: torch.device,
    dtype: torch.dtype,
    attention_implementation: str,
) -> tuple[nn.Module, ActionHead]:
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        attn_implementation=attention_implementation,
        local_files_only=True,
        low_cpu_mem_usage=True,
    )
    if not hasattr(model, "model") or not hasattr(model, "lm_head"):
        raise TypeError("expected a Qwen-style model with .model and .lm_head")
    action_head = ActionHead.from_lm_head(model.lm_head).to(device)
    for parameter in model.parameters():
        parameter.requires_grad = False
    model.eval().to(device)
    model.config.use_cache = False
    return model, action_head


def encode_action_hidden(model: nn.Module, batch: ActionBatch) -> torch.Tensor:
    with torch.inference_mode():
        hidden = model.model(
            input_ids=batch.input_ids,
            attention_mask=batch.attention_mask,
            use_cache=False,
            return_dict=True,
        ).last_hidden_state
        selected = hidden[batch.selected_rows, batch.selected_positions]
    return selected.detach().float()


def _f1(tp: float, fp: float, fn: float) -> float:
    precision = tp / max(1.0, tp + fp)
    recall = tp / max(1.0, tp + fn)
    return 2.0 * precision * recall / max(1e-12, precision + recall)


def evaluate_policy(
    model: nn.Module,
    head: nn.Module,
    *,
    samples_path: Path,
    distributed: DistributedContext,
    max_sequence_length: int,
    max_batch_tokens: int,
    max_batch_size: int,
    limit_records: int,
    chunk_ms: float,
) -> dict[str, float]:
    once = iter_action_samples_once(
        samples_path,
        rank=distributed.rank,
        world_size=distributed.world_size,
        max_sequence_length=max_sequence_length,
    )
    if limit_records > 0:
        rank_limit = math.ceil(limit_records / distributed.world_size)

        def limited():
            for index, sample in enumerate(once):
                if index >= rank_limit:
                    break
                yield sample

        samples = limited()
    else:
        samples = once
    batches = batch_action_samples(
        samples,
        max_batch_tokens=max_batch_tokens,
        max_batch_size=max_batch_size,
    )
    sums = {
        "loss_sum": 0.0,
        "events": 0.0,
        "correct": 0.0,
        "wait_tp": 0.0,
        "wait_fp": 0.0,
        "wait_fn": 0.0,
        "write_tp": 0.0,
        "write_fp": 0.0,
        "write_fn": 0.0,
        "premature": 0.0,
        "reference_wait": 0.0,
        "unnecessary": 0.0,
        "reference_write": 0.0,
        "final_correct": 0.0,
        "samples": 0.0,
        "first_write_abs_ms_sum": 0.0,
        "first_write_pairs": 0.0,
        "predicted_writes": 0.0,
        "reference_writes": 0.0,
    }
    head.eval()
    with torch.inference_mode():
        for batch in batches:
            batch.to(distributed.device)
            hidden = encode_action_hidden(model, batch)
            logits = head(hidden)
            predictions = logits.argmax(dim=-1)
            losses = F.cross_entropy(logits, batch.labels, reduction="none")
            labels = batch.labels
            sums["loss_sum"] += float(losses.sum())
            sums["events"] += batch.events
            sums["correct"] += float((predictions == labels).sum())
            sums["wait_tp"] += float(((predictions == 0) & (labels == 0)).sum())
            sums["wait_fp"] += float(((predictions == 0) & (labels == 1)).sum())
            sums["wait_fn"] += float(((predictions == 1) & (labels == 0)).sum())
            sums["write_tp"] += float(((predictions == 1) & (labels == 1)).sum())
            sums["write_fp"] += float(((predictions == 1) & (labels == 0)).sum())
            sums["write_fn"] += float(((predictions == 0) & (labels == 1)).sum())
            sums["premature"] += float(((predictions == 1) & (labels == 0)).sum())
            sums["reference_wait"] += float((labels == 0).sum())
            sums["unnecessary"] += float(((predictions == 0) & (labels == 1)).sum())
            sums["reference_write"] += float((labels == 1).sum())
            offset = 0
            for event_count in batch.sample_event_counts:
                sample_predictions = predictions[offset : offset + event_count]
                sample_labels = labels[offset : offset + event_count]
                final_prediction = int(sample_predictions[-1])
                sums["final_correct"] += float(final_prediction == 1)
                sums["samples"] += 1.0
                sums["predicted_writes"] += float((sample_predictions == 1).sum())
                sums["reference_writes"] += float((sample_labels == 1).sum())
                predicted_indexes = torch.nonzero(sample_predictions == 1).flatten()
                reference_indexes = torch.nonzero(sample_labels == 1).flatten()
                if predicted_indexes.numel() and reference_indexes.numel():
                    delta = abs(int(predicted_indexes[0]) - int(reference_indexes[0]))
                    sums["first_write_abs_ms_sum"] += delta * chunk_ms
                    sums["first_write_pairs"] += 1.0
                offset += event_count
            del hidden, logits, predictions, losses
    names = tuple(sums)
    reduced = distributed.reduce_sums([sums[name] for name in names])
    total = dict(zip(names, reduced))
    events = max(1.0, total["events"])
    samples_count = max(1.0, total["samples"])
    metrics = {
        "loss": total["loss_sum"] / events,
        "accuracy": total["correct"] / events,
        "wait_f1": _f1(total["wait_tp"], total["wait_fp"], total["wait_fn"]),
        "write_f1": _f1(total["write_tp"], total["write_fp"], total["write_fn"]),
        "premature_write_given_wait": total["premature"]
        / max(1.0, total["reference_wait"]),
        "unnecessary_wait_given_write": total["unnecessary"]
        / max(1.0, total["reference_write"]),
        "final_flush_success": total["final_correct"] / samples_count,
        "first_write_mae_ms": total["first_write_abs_ms_sum"]
        / max(1.0, total["first_write_pairs"]),
        "predicted_writes_per_sample": total["predicted_writes"] / samples_count,
        "reference_writes_per_sample": total["reference_writes"] / samples_count,
        "samples": total["samples"],
        "events": total["events"],
    }
    head.train()
    return metrics


def atomic_torch_save(path: Path, value: object) -> None:
    temporary = path.with_name(f".{path.name}.partial.{os.getpid()}")
    torch.save(value, temporary)
    os.replace(temporary, path)


def checkpoint_payload(
    head: nn.Module,
    reference: ActionHead,
    *,
    args: argparse.Namespace,
    step: int,
    metrics: dict[str, float],
) -> dict[str, object]:
    unwrapped = head.module if hasattr(head, "module") else head
    return {
        "schema_version": CHECKPOINT_SCHEMA,
        "action_head": unwrapped.state_dict(),
        "reference_head": reference.state_dict(),
        "base_model": str(Path(args.model).resolve()),
        "step": step,
        "mode": args.mode,
        "metrics": metrics,
        "args": vars(args),
    }


def learning_rate_factor(step: int, *, warmup_steps: int, total_steps: int) -> float:
    if step < warmup_steps:
        return max(1e-8, (step + 1) / max(1, warmup_steps))
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def train(args: argparse.Namespace) -> None:
    distributed = DistributedContext.initialize(args.device)
    torch.manual_seed(args.seed + distributed.rank)
    if distributed.device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed + distributed.rank)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    output_dir = Path(args.output_dir)
    tensorboard_dir = Path(args.tensorboard_dir)
    if distributed.is_main:
        if (
            output_dir.exists()
            and any(output_dir.iterdir())
            and not args.overwrite_output
        ):
            raise FileExistsError(f"refusing to overwrite non-empty {output_dir}")
        output_dir.mkdir(parents=True, exist_ok=True)
        tensorboard_dir.mkdir(parents=True, exist_ok=True)
    distributed.barrier()

    model_path = Path(args.model)
    model, initial_head = load_base_and_head(
        model_path,
        device=distributed.device,
        dtype=dtype_from_name(args.dtype),
        attention_implementation=args.attention_implementation,
    )
    reference = initial_head.frozen_copy().to(distributed.device)
    head = distributed.wrap(initial_head)
    optimizer = torch.optim.AdamW(
        head.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda index: learning_rate_factor(
            index, warmup_steps=args.warmup_steps, total_steps=args.train_steps
        ),
    )
    writer = SummaryWriter(tensorboard_dir) if distributed.is_main else None

    if distributed.is_main:
        manifest = {
            "schema_version": "simul_uniss_stage7a_run_v1",
            "git_revision": git_revision(Path.cwd()),
            "base_model": str(model_path.resolve()),
            "base_export_manifest": (
                json.loads((model_path / "export_manifest.json").read_text())
                if (model_path / "export_manifest.json").is_file()
                else None
            ),
            "train_samples": file_metadata(Path(args.train_samples)),
            "valid_samples": file_metadata(Path(args.valid_samples)),
            "world_size": distributed.world_size,
            "args": vars(args),
        }
        (output_dir / "run_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    sample_iterator = iter_action_samples(
        args.train_samples,
        rank=distributed.rank,
        world_size=distributed.world_size,
        max_sequence_length=args.max_sequence_length,
        shuffle_buffer_size=args.shuffle_buffer_size,
        seed=args.seed,
    )
    train_batches = iter(
        batch_action_samples(
            sample_iterator,
            max_batch_tokens=args.max_batch_tokens,
            max_batch_size=args.max_batch_size,
        )
    )
    best_score = -math.inf
    last_valid_metrics: dict[str, float] = {}
    run_started = time.perf_counter()
    for step in range(1, args.train_steps + 1):
        batch = next(train_batches).to(distributed.device)
        if distributed.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(distributed.device)
            torch.cuda.synchronize(distributed.device)
        step_started = time.perf_counter()
        hidden = encode_action_hidden(model, batch)
        optimizer.zero_grad(set_to_none=True)
        logits = head(hidden)
        use_sft = args.mode == "sft" or step <= args.sft_warmup_steps
        if use_sft:
            loss = F.cross_entropy(logits, batch.labels)
            metrics: dict[str, torch.Tensor] = {
                "loss": loss.detach(),
                "sft_loss": loss.detach(),
                "accuracy": (logits.argmax(dim=-1) == batch.labels)
                .float()
                .mean()
                .detach(),
                "write_rate": (logits.argmax(dim=-1) == 1).float().mean().detach(),
            }
        else:
            with torch.no_grad():
                reference_logits = reference(hidden)
            loss, metrics = grpo_action_loss(
                logits,
                reference_logits,
                batch.labels,
                batch.event_sample_ids,
                batch.event_fractions,
                batch.final_flags,
                sample_count=batch.samples,
                group_size=args.group_size,
                kl_beta=args.kl_beta,
                sft_weight=args.sft_replay_weight,
            )
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(head.parameters(), args.grad_clip)
        optimizer.step()
        scheduler.step()
        if distributed.device.type == "cuda":
            torch.cuda.synchronize(distributed.device)
        step_seconds = time.perf_counter() - step_started
        tokens_per_second = batch.actual_tokens / max(1e-9, step_seconds)
        padded_tokens_per_second = batch.padded_tokens / max(1e-9, step_seconds)
        peak_memory_mib = (
            torch.cuda.max_memory_allocated(distributed.device) / 1024**2
            if distributed.device.type == "cuda"
            else 0.0
        )

        if step == 1 or step % args.log_interval == 0:
            names = tuple(metrics)
            local_values = [float(metrics[name]) for name in names] + [
                float(grad_norm),
                float(step_seconds),
                float(tokens_per_second),
                float(padded_tokens_per_second),
                float(batch.samples),
                float(batch.events),
                float(batch.actual_tokens),
                float(batch.padded_tokens),
                float(peak_memory_mib),
            ]
            reduced = distributed.reduce_sums(local_values)
            averaged = [value / distributed.world_size for value in reduced]
            if writer is not None:
                for index, name in enumerate(names):
                    writer.add_scalar(f"stage7a/train_{name}", averaged[index], step)
                offset = len(names)
                extra_names = (
                    "grad_norm",
                    "step_seconds",
                    "tokens_per_second_per_rank",
                    "padded_tokens_per_second_per_rank",
                    "samples_per_rank",
                    "events_per_rank",
                    "actual_tokens_per_rank",
                    "padded_tokens_per_rank",
                    "peak_memory_mib_per_rank",
                )
                for index, name in enumerate(extra_names, start=offset):
                    writer.add_scalar(f"stage7a/{name}", averaged[index], step)
                writer.add_scalar(
                    "stage7a/learning_rate", optimizer.param_groups[0]["lr"], step
                )
                writer.flush()
                payload = {
                    "step": step,
                    "mode": "sft" if use_sft else "grpo",
                    **{name: averaged[index] for index, name in enumerate(names)},
                    **{
                        name: averaged[offset + index]
                        for index, name in enumerate(extra_names)
                    },
                    "learning_rate": optimizer.param_groups[0]["lr"],
                }
                print(json.dumps(payload, sort_keys=True), flush=True)

        should_evaluate = step % args.eval_interval == 0 or step == args.train_steps
        if should_evaluate:
            valid_metrics = evaluate_policy(
                model,
                head,
                samples_path=Path(args.valid_samples),
                distributed=distributed,
                max_sequence_length=args.max_sequence_length,
                max_batch_tokens=args.eval_max_batch_tokens,
                max_batch_size=args.eval_max_batch_size,
                limit_records=args.validation_records,
                chunk_ms=args.chunk_ms,
            )
            last_valid_metrics = valid_metrics
            score = (
                valid_metrics["write_f1"]
                - 0.25 * valid_metrics["premature_write_given_wait"]
                - 0.1 * valid_metrics["unnecessary_wait_given_write"]
            )
            if writer is not None:
                for name, value in valid_metrics.items():
                    writer.add_scalar(f"stage7a/valid_{name}", value, step)
                writer.add_scalar("stage7a/valid_selection_score", score, step)
                writer.flush()
                print(
                    json.dumps(
                        {"step": step, "validation": valid_metrics, "score": score},
                        sort_keys=True,
                    ),
                    flush=True,
                )
                payload = checkpoint_payload(
                    head, reference, args=args, step=step, metrics=valid_metrics
                )
                if score > best_score:
                    best_score = score
                    atomic_torch_save(output_dir / "best.pt", payload)
                if step % args.save_interval == 0 or step == args.train_steps:
                    atomic_torch_save(output_dir / f"step_{step:07d}.pt", payload)
            distributed.barrier()
        del batch, hidden, logits, loss

    if distributed.is_main:
        payload = checkpoint_payload(
            head,
            reference,
            args=args,
            step=args.train_steps,
            metrics=last_valid_metrics,
        )
        atomic_torch_save(output_dir / "final.pt", payload)
        (output_dir / "TRAINING_COMPLETE.json").write_text(
            json.dumps(
                {
                    "schema_version": "simul_uniss_stage7a_complete_v1",
                    "mode": args.mode,
                    "steps": args.train_steps,
                    "elapsed_seconds": time.perf_counter() - run_started,
                    "best_score": best_score,
                    "best_checkpoint": str((output_dir / "best.pt").resolve()),
                    "final_checkpoint": str((output_dir / "final.pt").resolve()),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        assert writer is not None
        writer.close()
    distributed.barrier()
    distributed.close()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("sft", "grpo"), required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--train-samples", required=True)
    parser.add_argument("--valid-samples", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tensorboard-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument(
        "--attention-implementation",
        choices=("eager", "sdpa", "flash_attention_2"),
        default="flash_attention_2",
    )
    parser.add_argument("--train-steps", type=int, default=1000)
    parser.add_argument("--sft-warmup-steps", type=int, default=100)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--kl-beta", type=float, default=0.02)
    parser.add_argument("--sft-replay-weight", type=float, default=0.2)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--max-sequence-length", type=int, default=18_000)
    parser.add_argument("--max-batch-tokens", type=int, default=131_072)
    parser.add_argument("--max-batch-size", type=int, default=256)
    parser.add_argument("--eval-max-batch-tokens", type=int, default=131_072)
    parser.add_argument("--eval-max-batch-size", type=int, default=256)
    parser.add_argument("--shuffle-buffer-size", type=int, default=8192)
    parser.add_argument("--validation-records", type=int, default=512)
    parser.add_argument("--chunk-ms", type=float, default=640.0)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--save-interval", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260727)
    parser.add_argument("--overwrite-output", action="store_true")
    args = parser.parse_args(argv)
    if args.mode == "grpo" and args.group_size < 2:
        parser.error("GRPO requires --group-size >= 2")
    if args.train_steps < 1 or args.sft_warmup_steps < 0:
        parser.error("training steps must be positive")
    if args.sft_warmup_steps > args.train_steps:
        parser.error("sft warmup cannot exceed total training steps")
    return args


def main(argv: list[str] | None = None) -> None:
    train(parse_args(argv))


if __name__ == "__main__":
    main()

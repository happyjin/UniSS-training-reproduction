#!/usr/bin/env python3
"""Distributed training for the frozen-encoder UniSS StreamSpeech CTC probe."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import Counter
from pathlib import Path

import sentencepiece as spm
import torch
import torch.distributed as dist
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset import CTCProbeDataset, DistributedContiguousBatchSampler, collate_probe
from model import CTCProbeConfig, LanguageConditionalCTCProbe


DIRECTION = {
    0: ("asr_eng", "nar_s2tt_cmn", "eng", "cmn"),
    1: ("asr_cmn", "nar_s2tt_eng", "cmn", "eng"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-index", type=Path, required=True)
    parser.add_argument("--tokenizer-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tensorboard-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=3000)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--min-learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=150)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--eval-batches", type=int, default=40)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--resume", type=Path)
    return parser.parse_args()


def ctc_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    input_lengths: torch.Tensor,
    target_lengths: torch.Tensor,
    blank: int,
) -> torch.Tensor:
    return F.ctc_loss(
        logits.float().log_softmax(dim=-1).transpose(0, 1),
        targets,
        input_lengths,
        target_lengths,
        blank=blank,
        reduction="mean",
        zero_infinity=False,
    )


def select_flat_targets(
    flat: torch.Tensor, lengths: torch.Tensor, indices: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    pieces_all = torch.split(flat, lengths.tolist())
    pieces = [pieces_all[i] for i in indices.tolist()]
    selected_lengths = lengths[indices]
    return torch.cat(pieces), selected_lengths


def batch_losses(
    model: torch.nn.Module, batch: dict[str, torch.Tensor | list[str]], vocab: dict[str, int]
) -> tuple[torch.Tensor, dict[str, float]]:
    hidden = batch["hidden"]
    hidden_lengths = batch["hidden_lengths"]
    direction_ids = batch["direction_ids"]
    assert isinstance(hidden, torch.Tensor)
    assert isinstance(hidden_lengths, torch.Tensor)
    assert isinstance(direction_ids, torch.Tensor)
    weighted = hidden.sum() * 0.0
    samples = 0
    values: dict[str, float] = {}
    for direction_id, (source_head, target_head, src_lang, tgt_lang) in DIRECTION.items():
        indices = torch.nonzero(direction_ids == direction_id, as_tuple=False).flatten()
        if not len(indices):
            continue
        source_targets, source_lengths = select_flat_targets(
            batch["source_targets"], batch["source_lengths"], indices  # type: ignore[arg-type]
        )
        target_targets, target_lengths = select_flat_targets(
            batch["target_targets"], batch["target_lengths"], indices  # type: ignore[arg-type]
        )
        local_hidden = hidden[indices]
        local_input_lengths = hidden_lengths[indices]
        source_logits = model(local_hidden, source_head)
        target_logits = model(local_hidden, target_head)
        source = ctc_loss(
            source_logits,
            source_targets,
            local_input_lengths,
            source_lengths,
            vocab[src_lang],
        )
        target = ctc_loss(
            target_logits,
            target_targets,
            local_input_lengths,
            target_lengths,
            vocab[tgt_lang],
        )
        count = len(indices)
        weighted = weighted + (4.0 * source + 4.0 * target) * count
        samples += count
        values[f"loss/{source_head}"] = float(source.detach())
        values[f"loss/{target_head}"] = float(target.detach())
    return weighted / max(1, samples), values


def collapse_ctc(path: list[int], blank: int) -> list[int]:
    output = []
    previous = None
    for token in path:
        if token != blank and token != previous:
            output.append(token)
        previous = token
    return output


def edit_distance(left: list[object], right: list[object]) -> int:
    previous = list(range(len(right) + 1))
    for i, a in enumerate(left, start=1):
        current = [i]
        for j, b in enumerate(right, start=1):
            current.append(
                min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (a != b))
            )
        previous = current
    return previous[-1]


def unigram_matches(predicted: list[int], reference: list[int]) -> int:
    pred_counts = Counter(predicted)
    ref_counts = Counter(reference)
    return sum(min(pred_counts[token], count) for token, count in ref_counts.items())


def split_flat(flat: torch.Tensor, lengths: torch.Tensor) -> list[list[int]]:
    result = []
    offset = 0
    for length in lengths.tolist():
        result.append(flat[offset : offset + length].tolist())
        offset += length
    return result


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    vocab: dict[str, int],
    processors: dict[str, spm.SentencePieceProcessor],
    max_batches: int,
) -> dict[str, float]:
    model.eval()
    sums = Counter()
    batches = 0
    for raw_batch in loader:
        batch = {
            key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
            for key, value in raw_batch.items()
        }
        hidden = batch["hidden"]
        hidden_lengths = batch["hidden_lengths"]
        direction_ids = batch["direction_ids"]
        source_sequences = split_flat(batch["source_targets"], batch["source_lengths"])
        target_sequences = split_flat(batch["target_targets"], batch["target_lengths"])
        for direction_id, (source_head, target_head, src_lang, tgt_lang) in DIRECTION.items():
            indices = torch.nonzero(direction_ids == direction_id, as_tuple=False).flatten()
            if not len(indices):
                continue
            local_hidden = hidden[indices]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                source_logits = model(local_hidden, source_head)
                target_logits = model(local_hidden, target_head)
            source_paths = source_logits.argmax(-1).cpu()
            target_paths = target_logits.argmax(-1).cpu()
            for local_row, original_index in enumerate(indices.tolist()):
                frames = int(hidden_lengths[original_index])
                source_pred = collapse_ctc(
                    source_paths[local_row, :frames].tolist(), vocab[src_lang]
                )
                target_pred = collapse_ctc(
                    target_paths[local_row, :frames].tolist(), vocab[tgt_lang]
                )
                source_ref = source_sequences[original_index]
                target_ref = target_sequences[original_index]
                pred_text = processors[src_lang].decode(source_pred)
                ref_text = processors[src_lang].decode(source_ref)
                if src_lang == "eng":
                    pred_units: list[object] = pred_text.split()
                    ref_units: list[object] = ref_text.split()
                    metric = "asr_eng_wer"
                else:
                    pred_units = list(pred_text.replace(" ", ""))
                    ref_units = list(ref_text.replace(" ", ""))
                    metric = "asr_cmn_cer"
                sums[f"{metric}_errors"] += edit_distance(pred_units, ref_units)
                sums[f"{metric}_units"] += len(ref_units)
                sums[f"nar_s2tt_{tgt_lang}_matches"] += unigram_matches(
                    target_pred, target_ref
                )
                sums[f"nar_s2tt_{tgt_lang}_units"] += len(target_ref)
                sums[f"samples_{direction_id}"] += 1
        batches += 1
        if batches >= max_batches:
            break
    keys = [
        "asr_eng_wer_errors",
        "asr_eng_wer_units",
        "asr_cmn_cer_errors",
        "asr_cmn_cer_units",
        "nar_s2tt_eng_matches",
        "nar_s2tt_eng_units",
        "nar_s2tt_cmn_matches",
        "nar_s2tt_cmn_units",
        "samples_0",
        "samples_1",
    ]
    vector = torch.tensor([float(sums[key]) for key in keys], device=device)
    dist.all_reduce(vector, op=dist.ReduceOp.SUM)
    reduced = dict(zip(keys, vector.tolist()))
    metrics = {
        "asr_eng_wer": reduced.get("asr_eng_wer_errors", 0.0)
        / max(1.0, reduced.get("asr_eng_wer_units", 0.0)),
        "asr_cmn_cer": reduced.get("asr_cmn_cer_errors", 0.0)
        / max(1.0, reduced.get("asr_cmn_cer_units", 0.0)),
        "nar_s2tt_eng_unigram_recall": reduced.get("nar_s2tt_eng_matches", 0.0)
        / max(1.0, reduced.get("nar_s2tt_eng_units", 0.0)),
        "nar_s2tt_cmn_unigram_recall": reduced.get("nar_s2tt_cmn_matches", 0.0)
        / max(1.0, reduced.get("nar_s2tt_cmn_units", 0.0)),
        "samples_eng_to_cmn": reduced.get("samples_0", 0.0),
        "samples_cmn_to_eng": reduced.get("samples_1", 0.0),
    }
    model.train()
    return metrics


def lr_scale(step: int, args: argparse.Namespace) -> float:
    if step < args.warmup_steps:
        return max(1e-4, step / max(1, args.warmup_steps))
    progress = (step - args.warmup_steps) / max(1, args.max_steps - args.warmup_steps)
    minimum = args.min_learning_rate / args.learning_rate
    return minimum + (1.0 - minimum) * 0.5 * (1.0 + math.cos(math.pi * progress))


def main() -> None:
    args = parse_args()
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    torch.manual_seed(args.seed + rank)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    processors = {
        language: spm.SentencePieceProcessor(
            model_file=str(args.tokenizer_dir / f"ctc_{language}.model")
        )
        for language in ("eng", "cmn")
    }
    vocab = {language: processor.vocab_size() for language, processor in processors.items()}
    config = CTCProbeConfig(eng_vocab_size=vocab["eng"], cmn_vocab_size=vocab["cmn"])
    model = LanguageConditionalCTCProbe(config).to(device)
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda value: lr_scale(value, args)
    )
    start_step = 0
    best_score = -float("inf")
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.module.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_step = int(checkpoint["step"])
        best_score = float(checkpoint.get("best_score", best_score))

    train_dataset = CTCProbeDataset(args.dataset_index, "train")
    valid_dataset = CTCProbeDataset(args.dataset_index, "valid")
    train_sampler = DistributedContiguousBatchSampler(
        len(train_dataset), args.batch_size, rank, world_size, shuffle=True, seed=args.seed
    )
    valid_sampler = DistributedContiguousBatchSampler(
        len(valid_dataset), args.batch_size, rank, world_size, shuffle=False, drop_last=False
    )
    loader_kwargs = {
        "num_workers": args.num_workers,
        "pin_memory": True,
        "persistent_workers": args.num_workers > 0,
        "collate_fn": collate_probe,
    }
    if args.num_workers > 0:
        loader_kwargs["prefetch_factor"] = 4
    train_loader = DataLoader(train_dataset, batch_sampler=train_sampler, **loader_kwargs)
    valid_loader = DataLoader(valid_dataset, batch_sampler=valid_sampler, **loader_kwargs)
    writer = SummaryWriter(args.tensorboard_dir) if rank == 0 else None
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if rank == 0:
        (args.output_dir / "run_config.json").write_text(
            json.dumps({**vars(args), "world_size": world_size, "vocab": vocab}, default=str, indent=2)
            + "\n",
            encoding="utf-8",
        )

    step = start_step
    epoch = 0
    last_time = time.time()
    while step < args.max_steps:
        train_sampler.set_epoch(epoch)
        for raw_batch in train_loader:
            if step >= args.max_steps:
                break
            batch = {
                key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
                for key, value in raw_batch.items()
            }
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss, components = batch_losses(model, batch, vocab)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            scheduler.step()
            step += 1
            if rank == 0 and step % args.log_every == 0:
                elapsed = max(1e-6, time.time() - last_time)
                samples_per_second = args.batch_size * world_size * args.log_every / elapsed
                last_time = time.time()
                writer.add_scalar("train/loss", float(loss.detach()), step)
                writer.add_scalar("train/learning_rate", scheduler.get_last_lr()[0], step)
                writer.add_scalar("train/samples_per_second", samples_per_second, step)
                for name, value in components.items():
                    writer.add_scalar(f"train/{name}", value, step)
                print(
                    json.dumps(
                        {
                            "step": step,
                            "loss": float(loss.detach()),
                            "lr": scheduler.get_last_lr()[0],
                            "samples_per_second": samples_per_second,
                        }
                    ),
                    flush=True,
                )
            if step % args.eval_every == 0 or step == args.max_steps:
                metrics = evaluate(
                    model, valid_loader, device, vocab, processors, args.eval_batches
                )
                score = (
                    2.0
                    - metrics["asr_eng_wer"]
                    - metrics["asr_cmn_cer"]
                    + metrics["nar_s2tt_eng_unigram_recall"]
                    + metrics["nar_s2tt_cmn_unigram_recall"]
                ) / 4.0
                if rank == 0:
                    for name, value in metrics.items():
                        writer.add_scalar(f"valid/{name}", value, step)
                    checkpoint = {
                        "schema_version": "uniss_streamspeech_ctc_probe_checkpoint_v1",
                        "step": step,
                        "epoch": epoch,
                        "model_config": model.module.metadata(),
                        "model": model.module.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "metrics": metrics,
                        "best_score": max(best_score, score),
                    }
                    path = args.output_dir / f"step-{step:06d}.pt"
                    torch.save(checkpoint, path)
                    if score > best_score:
                        best_score = score
                        torch.save(checkpoint, args.output_dir / "best.pt")
                    (args.output_dir / "latest_metrics.json").write_text(
                        json.dumps({"step": step, "score": score, **metrics}, indent=2) + "\n",
                        encoding="utf-8",
                    )
                    print(json.dumps({"validation_step": step, "score": score, **metrics}), flush=True)
                dist.barrier()
        epoch += 1
    if writer is not None:
        writer.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Joint ASR CTC, NAR-S2TT CTC and AR-S2TT CE training."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
STAGE03 = Path(__file__).resolve().parents[1]
STAGE02 = STAGE03.parent / "stage02_ctc_probe"
for path in (ROOT, STAGE03, STAGE02, Path(__file__).resolve().parent):
    sys.path.insert(0, str(path))

import sentencepiece as spm
import torch
import torch.distributed as dist
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from audio_data import DistributedLengthBucketBatchSampler, EndpointCTCAudioDataset, collate_audio
from dataset import DistributedContiguousBatchSampler
from model import EndpointCTCARStudent


DIRECTION = {
    0: ("asr_eng", "nar_s2tt_cmn", "eng", "cmn"),
    1: ("asr_cmn", "nar_s2tt_eng", "cmn", "eng"),
}


def select_flat_targets(flat, lengths, indices):
    pieces = torch.split(flat, lengths.tolist())
    chosen = [pieces[index] for index in indices.tolist()]
    return torch.cat(chosen), lengths[indices]


def split_flat(flat, lengths):
    values = []
    offset = 0
    for length in lengths.tolist():
        values.append(flat[offset : offset + length].tolist())
        offset += length
    return values


def collapse_ctc(path, blank):
    output = []
    previous = None
    for token in path:
        if token != blank and token != previous:
            output.append(token)
        previous = token
    return output


def edit_distance(left, right):
    previous = list(range(len(right) + 1))
    for row, left_value in enumerate(left, start=1):
        current = [row]
        for column, right_value in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column] + 1,
                    previous[column - 1] + (left_value != right_value),
                )
            )
        previous = current
    return previous[-1]


def unigram_matches(predicted, reference):
    predicted_counts = Counter(predicted)
    reference_counts = Counter(reference)
    return sum(
        min(predicted_counts[token], count)
        for token, count in reference_counts.items()
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-index", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--source-offsets", type=Path, required=True)
    parser.add_argument("--tokenizer-dir", type=Path, required=True)
    parser.add_argument("--initialize-from", type=Path, required=True)
    parser.add_argument("--length-index", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tensorboard-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--encoder-learning-rate", type=float, default=1e-5)
    parser.add_argument("--ctc-learning-rate", type=float, default=5e-5)
    parser.add_argument("--decoder-learning-rate", type=float, default=2e-4)
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument("--min-lr-ratio", type=float, default=0.05)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--eval-batches", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260803)
    return parser.parse_args()


def ctc_loss(logits, targets, input_lengths, target_lengths, blank):
    return F.ctc_loss(
        logits.float().log_softmax(-1).transpose(0, 1),
        targets,
        input_lengths,
        target_lengths,
        blank=blank,
        reduction="mean",
        zero_infinity=False,
    )


def losses(model, batch, vocab):
    output = model(
        batch["waveform"],
        batch["waveform_lengths"],
        batch["target_padded"],
        batch["target_lengths"],
        batch["direction_ids"],
    )
    logits = output["logits"]
    lengths = output["output_lengths"]
    total = output["ar_anchor"] + sum(value.sum() * 0.0 for value in logits.values())
    samples = 0
    metrics = {}
    for direction_id, (source_head, target_head, src_lang, tgt_lang) in DIRECTION.items():
        indices = torch.nonzero(batch["direction_ids"] == direction_id, as_tuple=False).flatten()
        if not len(indices):
            continue
        source_targets, source_lengths = select_flat_targets(
            batch["source_targets"], batch["source_lengths"], indices
        )
        target_targets, target_lengths = select_flat_targets(
            batch["target_targets"], batch["target_lengths"], indices
        )
        source = ctc_loss(
            logits[source_head][indices], source_targets, lengths[indices], source_lengths, vocab[src_lang]
        )
        target = ctc_loss(
            logits[target_head][indices], target_targets, lengths[indices], target_lengths, vocab[tgt_lang]
        )
        ar_values, ar_rows = output["ar_logits"][tgt_lang]
        references = batch["target_padded"][ar_rows]
        ar = F.cross_entropy(
            ar_values.float().reshape(-1, ar_values.shape[-1]),
            references.reshape(-1),
            ignore_index=-1,
        )
        count = len(indices)
        total = total + (4.0 * source + 4.0 * target + 8.0 * ar) * count
        samples += count
        metrics.update({source_head: float(source.detach()), target_head: float(target.detach()), f"ar_{tgt_lang}": float(ar.detach())})
    return total / max(1, samples), metrics


@torch.no_grad()
def evaluate_ctc(model, loader, device, vocab, processors, max_batches):
    model.eval()
    sums = Counter()
    batches = 0
    for value in loader:
        batch = {
            key: item.to(device, non_blocking=True)
            if isinstance(item, torch.Tensor)
            else item
            for key, item in value.items()
        }
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = model(batch["waveform"], batch["waveform_lengths"])
        logits = output["logits"]
        lengths = output["output_lengths"]
        source_sequences = split_flat(batch["source_targets"], batch["source_lengths"])
        target_sequences = split_flat(batch["target_targets"], batch["target_lengths"])
        for direction_id, (source_head, target_head, src_lang, tgt_lang) in DIRECTION.items():
            indices = torch.nonzero(
                batch["direction_ids"] == direction_id, as_tuple=False
            ).flatten()
            if not len(indices):
                continue
            source_paths = logits[source_head][indices].argmax(-1).cpu()
            target_paths = logits[target_head][indices].argmax(-1).cpu()
            for local_row, original_index in enumerate(indices.tolist()):
                frames = int(lengths[original_index])
                source_pred = collapse_ctc(
                    source_paths[local_row, :frames].tolist(), vocab[src_lang]
                )
                target_pred = collapse_ctc(
                    target_paths[local_row, :frames].tolist(), vocab[tgt_lang]
                )
                source_ref = source_sequences[original_index]
                target_ref = target_sequences[original_index]
                predicted_text = processors[src_lang].decode(source_pred)
                reference_text = processors[src_lang].decode(source_ref)
                if src_lang == "eng":
                    predicted_units = predicted_text.split()
                    reference_units = reference_text.split()
                    metric = "asr_eng_wer"
                else:
                    predicted_units = list(predicted_text.replace(" ", ""))
                    reference_units = list(reference_text.replace(" ", ""))
                    metric = "asr_cmn_cer"
                sums[f"{metric}_errors"] += edit_distance(
                    predicted_units, reference_units
                )
                sums[f"{metric}_units"] += len(reference_units)
                sums[f"nar_s2tt_{tgt_lang}_matches"] += unigram_matches(
                    target_pred, target_ref
                )
                sums[f"nar_s2tt_{tgt_lang}_units"] += len(target_ref)
                sums[f"samples_{direction_id}"] += 1
        batches += 1
        if max_batches and batches >= max_batches:
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
    dist.all_reduce(vector)
    reduced = dict(zip(keys, vector.tolist()))
    model.train()
    return {
        "asr_eng_wer": reduced["asr_eng_wer_errors"]
        / max(1.0, reduced["asr_eng_wer_units"]),
        "asr_cmn_cer": reduced["asr_cmn_cer_errors"]
        / max(1.0, reduced["asr_cmn_cer_units"]),
        "nar_s2tt_eng_unigram_recall": reduced["nar_s2tt_eng_matches"]
        / max(1.0, reduced["nar_s2tt_eng_units"]),
        "nar_s2tt_cmn_unigram_recall": reduced["nar_s2tt_cmn_matches"]
        / max(1.0, reduced["nar_s2tt_cmn_units"]),
        "samples_eng_to_cmn": reduced["samples_0"],
        "samples_cmn_to_eng": reduced["samples_1"],
    }


@torch.no_grad()
def evaluate_ar(model, loader, device, max_batches):
    model.eval()
    raw = model.module if isinstance(model, DDP) else model
    totals = torch.zeros(4, dtype=torch.float64, device=device)
    batches = 0
    for value in loader:
        batch = {key: item.to(device, non_blocking=True) if isinstance(item, torch.Tensor) else item for key, item in value.items()}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = raw(
                batch["waveform"], batch["waveform_lengths"], batch["target_padded"], batch["target_lengths"], batch["direction_ids"]
            )
        for language_index, language in enumerate(("eng", "cmn")):
            if language not in output["ar_logits"]:
                continue
            values, rows = output["ar_logits"][language]
            refs = batch["target_padded"][rows]
            mask = refs >= 0
            totals[language_index] += ((values.argmax(-1) == refs) & mask).sum()
            totals[language_index + 2] += mask.sum()
        batches += 1
        if max_batches and batches >= max_batches:
            break
    dist.all_reduce(totals)
    model.train()
    return {
        "ar_eng_token_accuracy": float(totals[0] / totals[2].clamp_min(1)),
        "ar_cmn_token_accuracy": float(totals[1] / totals[3].clamp_min(1)),
    }


def lr_scale(step, args):
    if step < args.warmup_steps:
        return max(1e-4, step / max(1, args.warmup_steps))
    progress = (step - args.warmup_steps) / max(1, args.max_steps - args.warmup_steps)
    return args.min_lr_ratio + (1 - args.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress))


def main():
    args = parse_args()
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    torch.manual_seed(args.seed + rank)
    processors = {
        language: spm.SentencePieceProcessor(model_file=str(args.tokenizer_dir / f"ctc_{language}.model"))
        for language in ("eng", "cmn")
    }
    vocab = {language: processor.vocab_size() for language, processor in processors.items()}
    base = EndpointCTCARStudent.from_stage03_checkpoint(
        args.initialize_from, eng_vocab_size=vocab["eng"], cmn_vocab_size=vocab["cmn"]
    ).to(device)
    model = DDP(base, device_ids=[local_rank], broadcast_buffers=False)
    optimizer = torch.optim.AdamW(
        [
            {"params": list(base.base.encoder_parameters()), "lr": args.encoder_learning_rate},
            {"params": base.base.heads.parameters(), "lr": args.ctc_learning_rate},
            {
                "params": [
                    *base.target_embeddings.parameters(),
                    *base.target_positions.parameters(),
                    *base.decoder.parameters(),
                    *base.target_outputs.parameters(),
                ],
                "lr": args.decoder_learning_rate,
            },
        ],
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: lr_scale(step, args))
    train_data = EndpointCTCAudioDataset(args.dataset_index, "train", args.source_manifest, args.source_offsets)
    valid_data = EndpointCTCAudioDataset(args.dataset_index, "valid", args.source_manifest, args.source_offsets)
    train_sampler = DistributedLengthBucketBatchSampler(args.length_index, args.batch_size, rank, world, seed=args.seed)
    valid_sampler = DistributedContiguousBatchSampler(len(valid_data), args.batch_size, rank, world, shuffle=False, drop_last=False, even_batches=False)
    options = {"num_workers": args.num_workers, "pin_memory": True, "persistent_workers": args.num_workers > 0, "collate_fn": collate_audio}
    if args.num_workers:
        options["prefetch_factor"] = 2
    train_loader = DataLoader(train_data, batch_sampler=train_sampler, **options)
    valid_loader = DataLoader(valid_data, batch_sampler=valid_sampler, **options)
    writer = SummaryWriter(args.tensorboard_dir) if rank == 0 else None
    args.output_dir.mkdir(parents=True, exist_ok=True)
    step = 0
    epoch = 0
    best_score = -float("inf")
    last_time = time.time()
    while step < args.max_steps:
        train_sampler.set_epoch(epoch)
        for value in train_loader:
            if step >= args.max_steps:
                break
            batch = {key: item.to(device, non_blocking=True) if isinstance(item, torch.Tensor) else item for key, item in value.items()}
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss, components = losses(model, batch, vocab)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            scheduler.step()
            step += 1
            if rank == 0 and step % args.log_every == 0:
                elapsed = max(1e-6, time.time() - last_time)
                throughput = args.batch_size * world * args.log_every / elapsed
                last_time = time.time()
                writer.add_scalar("train/loss", float(loss.detach()), step)
                writer.add_scalar("train/samples_per_second", throughput, step)
                for name, metric in components.items():
                    writer.add_scalar(f"train/{name}", metric, step)
                print(json.dumps({"step": step, "loss": float(loss.detach()), "samples_per_second": throughput}), flush=True)
            if step % args.eval_every == 0 or step == args.max_steps:
                ctc = evaluate_ctc(base.base, valid_loader, device, vocab, processors, args.eval_batches)
                ar = evaluate_ar(model, valid_loader, device, args.eval_batches)
                metrics = {**ctc, **ar}
                score = (2 - ctc["asr_eng_wer"] - ctc["asr_cmn_cer"] + ctc["nar_s2tt_eng_unigram_recall"] + ctc["nar_s2tt_cmn_unigram_recall"] + ar["ar_eng_token_accuracy"] + ar["ar_cmn_token_accuracy"]) / 6
                if rank == 0:
                    state = {"schema_version": "uniss_streamspeech_stage03b_ar_s2tt_v1", "step": step, "epoch": epoch, "model": base.state_dict(), "model_config": base.base.metadata(), "metrics": metrics, "score": score}
                    torch.save(state, args.output_dir / "latest.pt")
                    if score > best_score:
                        best_score = score
                        torch.save(state, args.output_dir / "best.pt")
                    (args.output_dir / "latest_metrics.json").write_text(json.dumps({"step": step, "score": score, **metrics}, indent=2) + "\n")
                    for name, metric in metrics.items():
                        writer.add_scalar(f"valid/{name}", metric, step)
                    print(json.dumps({"validation_step": step, "score": score, **metrics}), flush=True)
                dist.barrier()
        epoch += 1
    if writer:
        writer.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

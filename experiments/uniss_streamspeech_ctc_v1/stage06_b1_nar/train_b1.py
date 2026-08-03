#!/usr/bin/env python3
"""Train the zero-initialized B1 residual through frozen Phase3 NLL."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
TREE = Path(__file__).resolve().parents[1]
STAGE03 = TREE / "stage03_multitask_encoder"
STAGE02 = TREE / "stage02_ctc_probe"
STAGE04 = TREE / "stage04_b2_discrete_bridge"
for path in (ROOT, STAGE03, STAGE02, STAGE04, Path(__file__).resolve().parent):
    sys.path.insert(0, str(path))

import sentencepiece as spm
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoModelForCausalLM, AutoTokenizer

from audio_data import DistributedLengthBucketBatchSampler
from bridge_data import B2BridgeAudioDataset, collate_bridge
from model import FrozenB2ResidualBridge
from train_b2 import lm_batch
from training import constants_uniss as c
from training.generate_unist_eval_audio import load_hf_text_encoder


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-index", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--source-offsets", type=Path, required=True)
    parser.add_argument("--ctc-tokenizer-dir", type=Path, required=True)
    parser.add_argument("--endpoint-checkpoint", type=Path, required=True)
    parser.add_argument("--historical-stage-b-checkpoint", type=Path, required=True)
    parser.add_argument("--stage04-b2-checkpoint", type=Path, required=True)
    parser.add_argument("--codebook-model", type=Path, required=True)
    parser.add_argument("--phase3-model", type=Path, required=True)
    parser.add_argument("--length-index", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tensorboard-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--min-learning-rate", type=float, default=1e-5)
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--residual-weight", type=float, default=1e-4)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--eval-batches", type=int, default=8)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260803)
    return parser.parse_args()


def endpoint_loss(model, qwen, text_encoder, batch, args, device):
    output = model(batch["waveform"], batch["waveform_lengths"])
    inputs, attention, labels, tokens = lm_batch(
        qwen, text_encoder, batch["phase3_records"], output, device
    )
    phase3 = qwen(
        inputs_embeds=inputs,
        attention_mask=attention,
        labels=labels,
        use_cache=False,
    )
    loss = phase3.loss + args.residual_weight * output.residual_mse
    return loss, {
        "phase3_nll": float(phase3.loss.detach()),
        "residual_rms": float(output.residual_rms.detach()),
        "target_tokens": tokens,
    }


@torch.no_grad()
def evaluate(model, qwen, text_encoder, loader, args, device):
    model.eval()
    total_nll = torch.zeros(1, dtype=torch.float64, device=device)
    total_tokens = torch.zeros(1, dtype=torch.float64, device=device)
    for batches, value in enumerate(loader, start=1):
        batch = {
            key: item.to(device, non_blocking=True) if isinstance(item, torch.Tensor) else item
            for key, item in value.items()
        }
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, metrics = endpoint_loss(model, qwen, text_encoder, batch, args, device)
        tokens = float(metrics["target_tokens"])
        total_nll += float(metrics["phase3_nll"]) * tokens
        total_tokens += tokens
        if batches >= args.eval_batches:
            break
    values = torch.cat((total_nll, total_tokens))
    dist.all_reduce(values)
    model.train()
    return float(values[0] / values[1].clamp_min(1))


def lr_scale(step, args):
    if step < args.warmup_steps:
        return max(1e-4, step / max(1, args.warmup_steps))
    progress = (step - args.warmup_steps) / max(1, args.max_steps - args.warmup_steps)
    ratio = args.min_learning_rate / args.learning_rate
    return ratio + (1 - ratio) * 0.5 * (1 + math.cos(math.pi * progress))


def checkpoint_state(base, step, validation):
    return {
        "schema_version": "uniss_streamspeech_b1_continuous_residual_v1",
        "step": step,
        "model": base.state_dict(),
        "validation_phase3_nll": validation,
    }


def main():
    args = parse_args()
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    torch.manual_seed(args.seed + rank)
    tokenizer = AutoTokenizer.from_pretrained(args.phase3_model, local_files_only=True)
    text_encoder = load_hf_text_encoder(tokenizer)
    qwen = AutoModelForCausalLM.from_pretrained(
        args.phase3_model, local_files_only=True, torch_dtype=torch.bfloat16
    ).to(device).eval()
    qwen.requires_grad_(False)
    qwen_glm_embeddings = qwen.get_input_embeddings().weight[
        c.GLM_SEMANTIC_OFFSET : c.GLM_SEMANTIC_OFFSET + c.GLM_SEMANTIC_SIZE
    ].detach().float().cpu()
    eng_vocab = spm.SentencePieceProcessor(
        model_file=str(args.ctc_tokenizer_dir / "ctc_eng.model")
    ).vocab_size()
    cmn_vocab = spm.SentencePieceProcessor(
        model_file=str(args.ctc_tokenizer_dir / "ctc_cmn.model")
    ).vocab_size()
    base = FrozenB2ResidualBridge.from_checkpoints(
        endpoint_checkpoint=args.endpoint_checkpoint,
        historical_stage_b_checkpoint=args.historical_stage_b_checkpoint,
        stage04_b2_checkpoint=args.stage04_b2_checkpoint,
        codebook_model=args.codebook_model,
        qwen_glm_embeddings=qwen_glm_embeddings,
        eng_vocab_size=eng_vocab,
        cmn_vocab_size=cmn_vocab,
    ).to(device)
    model = DDP(base, device_ids=[local_rank], broadcast_buffers=False)
    optimizer = torch.optim.AdamW(
        base.residual.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: lr_scale(step, args)
    )
    train_data = B2BridgeAudioDataset(
        args.dataset_index, "train", args.source_manifest, args.source_offsets
    )
    valid_data = B2BridgeAudioDataset(
        args.dataset_index, "valid", args.source_manifest, args.source_offsets
    )
    train_sampler = DistributedLengthBucketBatchSampler(
        args.length_index, args.batch_size, rank, world, seed=args.seed
    )
    valid_sampler = DistributedSampler(
        valid_data, num_replicas=world, rank=rank, shuffle=False, drop_last=False
    )
    options = {
        "num_workers": args.num_workers,
        "pin_memory": True,
        "persistent_workers": args.num_workers > 0,
        "collate_fn": collate_bridge,
    }
    train_loader = DataLoader(train_data, batch_sampler=train_sampler, **options)
    valid_loader = DataLoader(
        valid_data, batch_size=args.batch_size, sampler=valid_sampler, **options
    )
    writer = SummaryWriter(args.tensorboard_dir) if rank == 0 else None
    args.output_dir.mkdir(parents=True, exist_ok=True)
    baseline = evaluate(model, qwen, text_encoder, valid_loader, args, device)
    best = baseline
    if rank == 0:
        state = checkpoint_state(base, 0, baseline)
        torch.save(state, args.output_dir / "initial.pt")
        torch.save(state, args.output_dir / "best.pt")
        writer.add_scalar("valid/phase3_nll", baseline, 0)
        print(json.dumps({"validation_step": 0, "phase3_nll": baseline}), flush=True)
    dist.barrier()
    step = 0
    epoch = 0
    last_time = time.time()
    while step < args.max_steps:
        train_sampler.set_epoch(epoch)
        for value in train_loader:
            if step >= args.max_steps:
                break
            batch = {
                key: item.to(device, non_blocking=True) if isinstance(item, torch.Tensor) else item
                for key, item in value.items()
            }
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss, metrics = endpoint_loss(model, qwen, text_encoder, batch, args, device)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(base.residual.parameters(), 1.0)
            if not torch.isfinite(grad_norm):
                raise FloatingPointError(
                    f"non-finite B1 residual gradient at step {step + 1}: {grad_norm}"
                )
            optimizer.step()
            scheduler.step()
            step += 1
            if rank == 0 and step % args.log_every == 0:
                elapsed = max(1e-6, time.time() - last_time)
                last_time = time.time()
                writer.add_scalar("train/loss", float(loss.detach()), step)
                for name, metric in metrics.items():
                    writer.add_scalar(f"train/{name}", metric, step)
                print(
                    json.dumps(
                        {
                            "step": step,
                            "loss": float(loss.detach()),
                            "steps_per_second": args.log_every / elapsed,
                            **metrics,
                        }
                    ),
                    flush=True,
                )
            if step % args.eval_every == 0 or step == args.max_steps:
                validation = evaluate(model, qwen, text_encoder, valid_loader, args, device)
                if rank == 0:
                    state = checkpoint_state(base, step, validation)
                    torch.save(state, args.output_dir / "latest.pt")
                    if validation < best:
                        best = validation
                        torch.save(state, args.output_dir / "best.pt")
                    (args.output_dir / "latest_metrics.json").write_text(
                        json.dumps({"step": step, "validation_phase3_nll": validation}, indent=2)
                        + "\n"
                    )
                    writer.add_scalar("valid/phase3_nll", validation, step)
                    print(
                        json.dumps({"validation_step": step, "phase3_nll": validation}),
                        flush=True,
                    )
                dist.barrier()
        epoch += 1
    if writer:
        writer.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

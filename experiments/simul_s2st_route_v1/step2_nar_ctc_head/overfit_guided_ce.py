#!/usr/bin/env python3
"""Tiny non-Megatron overfit: guided CE + CTC on 1–4 samples via teacher-forced path."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from experiments.simul_s2st_route_v1.step2_nar_ctc_head.dataset import (
    NarCtcJointDataset,
    collate_nar_ctc,
)
from experiments.simul_s2st_route_v1.step2_nar_ctc_head.duration_anchored_nar_ctc import (
    DurationAnchoredCausalNARCTC,
)
from experiments.simul_s2st_route_v1.step2_nar_ctc_head.pretrain_nar_ctc_megatron import (
    blank_probability_penalty,
    guided_duration_ce,
)
from experiments.simul_s2st_route_v1.step2_nar_ctc_head.teacher_forced import (
    batch_fields,
    batched_target_text_hidden,
)
from training import constants_uniss as c
from training.generate_unist_eval_audio import load_hf_text_encoder
from training.phase3_whisper_streamspeech_joint.losses import ctc_normalized_loss


def greedy_units(logits: torch.Tensor, frame_lengths: torch.Tensor, blank_id: int) -> list[list[int]]:
    outputs: list[list[int]] = []
    for row, length in enumerate(frame_lengths.tolist()):
        ids = logits[row, : int(length)].argmax(dim=-1).tolist()
        collapsed: list[int] = []
        previous = None
        for value in ids:
            if value != previous and value != blank_id:
                collapsed.append(int(value))
            previous = value
        outputs.append(collapsed)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT
        / "data/processed/phase3_whisper_streamspeech_joint_v5/pilot_15shard_joint/joint_train.jsonl",
    )
    parser.add_argument(
        "--phase3-model",
        type=Path,
        default=ROOT / "checkpoints/exported_hf/qwen0p5b_phase3_unist198_iter_0009075_hf",
    )
    parser.add_argument("--samples", type=int, default=2)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--guided-ce-weight", type=float, default=1.0)
    parser.add_argument("--blank-penalty", type=float, default=1.0)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dataset = NarCtcJointDataset(args.manifest, max_samples=max(args.samples, 4), max_audio_seconds=10.0)
    batch = collate_nar_ctc([dataset[index] for index in range(args.samples)])

    unit_tensor = batch["target_bicodec"]
    valid_units = unit_tensor[unit_tensor >= 0]
    unit_min = int(valid_units.min())
    unit_max = int(valid_units.max())
    blank_id = c.BICODEC_SEMANTIC_SIZE

    tokenizer = AutoTokenizer.from_pretrained(str(args.phase3_model), local_files_only=True)
    text_encoder = load_hf_text_encoder(tokenizer)
    qwen = (
        AutoModelForCausalLM.from_pretrained(
            str(args.phase3_model), local_files_only=True, torch_dtype=torch.bfloat16
        )
        .to(device)
        .eval()
    )
    qwen.requires_grad_(False)

    head = DurationAnchoredCausalNARCTC(
        qwen_hidden_size=int(qwen.config.hidden_size),
        semantic_vocab_size=c.BICODEC_SEMANTIC_SIZE,
    ).to(device)
    trainable = sum(parameter.requires_grad for parameter in head.parameters())
    total = sum(1 for _ in head.parameters())
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr)

    fields = batch_fields(batch)
    text_hidden, text_lengths, _ = batched_target_text_hidden(
        qwen,
        text_encoder,
        source_glm=fields["source_glm"],
        bicodec_global=fields["bicodec_global"],
        tgt_lang=fields["tgt_lang"],
        translation=fields["translation"],
        target_bicodec=fields["target_bicodec"],
        source_id=fields["ids"],
        device=device,
    )
    duration = fields["source_duration_ms"].to(device)
    unit_lengths = fields["target_bicodec_lengths"].to(device)
    unit_repeats = fields["unit_repeats"].to(device)
    units_padded = fields["target_bicodec_tensor"].to(device)
    units_flat = torch.cat([units_padded[row, : int(unit_lengths[row])] for row in range(args.samples)])

    history = []
    for step in range(args.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits, frame_lengths = head(
                text_hidden,
                text_lengths,
                duration,
                unit_lengths=unit_lengths,
                unit_repeats=unit_repeats,
            )
            ctc, infeasible = ctc_normalized_loss(
                logits,
                units_flat,
                frame_lengths,
                unit_lengths,
                blank_id=head.blank_id,
            )
            blank_mass = blank_probability_penalty(
                logits, frame_lengths, blank_id=head.blank_id
            )
            guided = guided_duration_ce(
                logits, units_padded, frame_lengths, unit_lengths
            )
            total_loss = (
                ctc.mean
                + args.blank_penalty * blank_mass
                + args.guided_ce_weight * guided
            )
        if step < args.steps:
            total_loss.backward()
            grad_norm = float(torch.nn.utils.clip_grad_norm_(head.parameters(), 1e9).detach())
            optimizer.step()
        else:
            grad_norm = 0.0

        with torch.no_grad():
            decoded = greedy_units(logits.float(), frame_lengths, head.blank_id)
            blank_frac = float((logits.argmax(dim=-1) == head.blank_id).float().mean())
            mean_blank_prob = float(
                torch.softmax(logits.float(), dim=-1)[..., head.blank_id].mean()
            )

        row = {
            "step": step,
            "ctc": float(ctc.mean.detach()),
            "guided_ce": float(guided.detach()),
            "blank_mass": float(blank_mass.detach()),
            "total": float(total_loss.detach()),
            "grad_norm": grad_norm,
            "blank_fraction": blank_frac,
            "mean_blank_probability": mean_blank_prob,
            "predicted_units": [len(sequence) for sequence in decoded],
            "logits_shape": list(logits.shape),
            "blank_id": head.blank_id,
        }
        history.append(row)
        print(json.dumps(row), flush=True)

    summary = {
        "stage": "done",
        "samples": args.samples,
        "steps": args.steps,
        "trainable_params": trainable,
        "total_params": total,
        "unit_min": unit_min,
        "unit_max": unit_max,
        "blank_id": blank_id,
        "log_vocab_baseline": float(torch.log(torch.tensor(float(head.blank_id + 1)))),
        "initial_guided_ce": history[0]["guided_ce"],
        "final_guided_ce": history[-1]["guided_ce"],
        "initial_predicted_units": history[0]["predicted_units"],
        "final_predicted_units": history[-1]["predicted_units"],
        "initial_blank_fraction": history[0]["blank_fraction"],
        "final_blank_fraction": history[-1]["blank_fraction"],
    }
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()

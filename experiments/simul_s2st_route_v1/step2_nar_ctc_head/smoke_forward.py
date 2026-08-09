#!/usr/bin/env python3
"""Single-process GPU smoke: one teacher-forced forward + CTC backward through the new head."""

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

from experiments.simul_s2st_route_v1.step2_nar_ctc_head.dataset import NarCtcJointDataset
from experiments.simul_s2st_route_v1.step2_nar_ctc_head.duration_anchored_nar_ctc import (
    DurationAnchoredCausalNARCTC,
)
from experiments.simul_s2st_route_v1.step2_nar_ctc_head.teacher_forced import (
    batch_fields,
    target_text_hidden,
)
from training import constants_uniss as c
from training.generate_unist_eval_audio import load_hf_text_encoder
from training.phase3_whisper_streamspeech_joint.losses import ctc_normalized_loss


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT
        / "data/processed/phase3_whisper_streamspeech_joint_v1/smoke_manifest_128/joint_train.jsonl",
    )
    parser.add_argument(
        "--phase3-model",
        type=Path,
        default=ROOT / "checkpoints/exported_hf/qwen0p5b_phase3_unist198_iter_0009075_hf",
    )
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--frames-per-second", type=float, default=75.0)
    parser.add_argument("--max-frames", type=int, default=900)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dataset = NarCtcJointDataset(args.manifest, max_samples=max(8, args.steps), max_audio_seconds=10.0)
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
        frames_per_second=args.frames_per_second,
        max_frames=args.max_frames,
    ).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=2e-4)

    history = []
    for step in range(args.steps):
        item = dataset[step % len(dataset)]
        batch = {key: value.unsqueeze(0) if isinstance(value, torch.Tensor) else [value] for key, value in item.items()}
        fields = batch_fields(batch)
        text_hidden, text_lengths, _ = target_text_hidden(
            qwen,
            text_encoder,
            source_glm=fields["source_glm"],
            bicodec_global=fields["bicodec_global"],
            tgt_lang=fields["tgt_lang"],
            translation=fields["translation"],
            target_bicodec=fields["target_bicodec"],
            source_id=fields["id"],
            device=device,
        )
        duration = fields["source_duration_ms"].to(device).reshape(1)
        units = fields["target_bicodec_tensor"].to(device)
        unit_lengths = torch.tensor([units.numel()], device=device)
        unit_repeats = fields["unit_repeats"].to(device).reshape(1)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits, frames = head(
                text_hidden,
                text_lengths,
                duration,
                unit_lengths=unit_lengths,
                unit_repeats=unit_repeats,
            )
            loss, infeasible = ctc_normalized_loss(
                logits, units, frames, unit_lengths, blank_id=head.blank_id
            )
        if int(infeasible.item()) > 0:
            raise RuntimeError(f"infeasible at step {step}: frames={int(frames)} units={int(unit_lengths)}")
        mean = loss.mean
        mean.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(head.parameters(), 1e9).detach())
        optimizer.step()
        row = {
            "step": step,
            "id": fields["id"],
            "loss": float(mean.detach()),
            "frames": int(frames),
            "units": int(unit_lengths),
            "text": int(text_lengths),
            "grad_norm": grad_norm,
        }
        history.append(row)
        print(json.dumps(row), flush=True)

    if not all(row["grad_norm"] > 0 for row in history):
        raise RuntimeError("head received zero gradients")
    print(json.dumps({"stage": "done", "steps": history}))


if __name__ == "__main__":
    main()

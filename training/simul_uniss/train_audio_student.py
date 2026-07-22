"""Train the causal audio Streaming GLM student on reconstructed/raw audio."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from training.simul_uniss.audio_streaming_student import (
    AudioStreamingStudent,
    AudioStudentDataset,
    audio_student_losses,
    collate_audio_student,
)
from training.simul_uniss.policy_tokenizer import PolicyTokenizer


def infinite_batches(loader: DataLoader):
    while True:
        yield from loader


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
    parser.add_argument("--seed", type=int, default=20260722)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    tokenizer = PolicyTokenizer(args.policy_tokenizer)
    dataset = AudioStudentDataset(
        args.manifest,
        tokenizer,
        max_audio_seconds=args.max_audio_seconds,
        prefix_training=True,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_audio_student)
    batches = infinite_batches(loader)
    model = AudioStreamingStudent(
        tokenizer.ctc_vocab_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    writer = SummaryWriter(args.tensorboard_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for step in range(1, args.max_steps + 1):
        batch = {key: value.to(device) for key, value in next(batches).items()}
        optimizer.zero_grad(set_to_none=True)
        losses = audio_student_losses(model, batch)
        losses["total"].backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % args.log_interval == 0 or step == 1:
            for name, value in losses.items():
                writer.add_scalar(f"stage1_audio/{name}_loss", float(value), step)
            writer.add_scalar("stage1_audio/grad_norm", float(grad_norm), step)
            writer.flush()
            print(json.dumps({"step": step, **{name: float(value) for name, value in losses.items()}}), flush=True)
        if step % args.save_interval == 0 or step == args.max_steps:
            torch.save({"model": model.state_dict(), "args": vars(args), "step": step}, output_dir / "last.pt")
    writer.close()
    print(json.dumps({"status": "complete", "output": str(output_dir / "last.pt")}))


if __name__ == "__main__":
    main()

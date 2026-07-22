"""NAST-S2x-style bootstrap non-autoregressive BiCodec semantic generator."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterator

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, IterableDataset
from torch.utils.tensorboard import SummaryWriter

from training import constants_uniss as c


class SemanticWriteDataset(IterableDataset):
    def __init__(self, schedule_path: str | Path, max_text_tokens: int, max_semantic_tokens: int) -> None:
        self.path = Path(schedule_path)
        self.max_text_tokens = max_text_tokens
        self.max_semantic_tokens = max_semantic_tokens

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                item = json.loads(line)
                for event in item["events"]:
                    if event["action"] != "write" or not event["target_semantic"]:
                        continue
                    text = [int(token) % 8192 for token in event["target_text_ids"][: self.max_text_tokens]]
                    semantic = [int(token) + 1 for token in event["target_semantic"][: self.max_semantic_tokens]]
                    if text and semantic:
                        yield {
                            "text": torch.tensor(text, dtype=torch.long),
                            "semantic": torch.tensor(semantic, dtype=torch.long),
                        }


def collate_nar(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    text_lengths = torch.tensor([len(item["text"]) for item in batch], dtype=torch.long)
    semantic_lengths = torch.tensor([len(item["semantic"]) for item in batch], dtype=torch.long)
    text = torch.zeros(len(batch), int(text_lengths.max()), dtype=torch.long)
    for index, item in enumerate(batch):
        text[index, : len(item["text"])] = item["text"]
    return {
        "text": text,
        "text_lengths": text_lengths,
        "semantic": torch.cat([item["semantic"] for item in batch]),
        "semantic_lengths": semantic_lengths,
    }


class NARSemanticGenerator(nn.Module):
    def __init__(
        self,
        *,
        hidden_size: int = 256,
        num_layers: int = 4,
        num_heads: int = 8,
        max_output_tokens: int = 256,
    ) -> None:
        super().__init__()
        if hidden_size % num_heads:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.max_output_tokens = max_output_tokens
        self.text_embedding = nn.Embedding(8192, hidden_size)
        self.position_embedding = nn.Embedding(max_output_tokens, hidden_size)
        layer = nn.TransformerEncoderLayer(
            hidden_size,
            num_heads,
            hidden_size * 4,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.decoder = nn.TransformerEncoder(layer, num_layers=num_layers, enable_nested_tensor=False)
        self.semantic_head = nn.Linear(hidden_size, c.BICODEC_SEMANTIC_SIZE + 1)
        self.length_head = nn.Linear(hidden_size, 1)

    def forward(self, text: torch.Tensor, text_lengths: torch.Tensor) -> dict[str, torch.Tensor]:
        embedded = self.text_embedding(text)
        positions = torch.arange(text.shape[1], device=text.device).unsqueeze(0)
        mask = positions < text_lengths.unsqueeze(1)
        context = (embedded * mask.unsqueeze(-1)).sum(dim=1) / text_lengths.clamp_min(1).unsqueeze(1)
        output_positions = torch.arange(self.max_output_tokens, device=text.device)
        hidden = context.unsqueeze(1) + self.position_embedding(output_positions).unsqueeze(0)
        hidden = self.decoder(hidden)
        predicted_length = F.softplus(self.length_head(context).squeeze(-1)) + 1.0
        return {"logits": self.semantic_head(hidden), "predicted_length": predicted_length}


def nar_losses(model: NARSemanticGenerator, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    outputs = model(batch["text"], batch["text_lengths"])
    semantic_lengths = batch["semantic_lengths"].clamp_max(model.max_output_tokens)
    input_lengths = torch.minimum(
        torch.full_like(semantic_lengths, model.max_output_tokens),
        semantic_lengths + torch.maximum(torch.ones_like(semantic_lengths) * 2, semantic_lengths // 5),
    )
    ctc = F.ctc_loss(
        outputs["logits"].log_softmax(dim=-1).transpose(0, 1),
        batch["semantic"],
        input_lengths,
        semantic_lengths,
        blank=0,
        zero_infinity=True,
    )
    length = F.smooth_l1_loss(
        torch.log1p(outputs["predicted_length"]),
        torch.log1p(semantic_lengths.float()),
    )
    return {"total": ctc + 0.1 * length, "ctc": ctc, "length": length}


def infinite_batches(loader: DataLoader):
    while True:
        yield from loader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schedules", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tensorboard-dir", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--max-text-tokens", type=int, default=64)
    parser.add_argument("--max-semantic-tokens", type=int, default=256)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260722)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    dataset = SemanticWriteDataset(args.schedules, args.max_text_tokens, args.max_semantic_tokens)
    loader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate_nar, num_workers=0)
    batches = infinite_batches(loader)
    model = NARSemanticGenerator(
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        max_output_tokens=args.max_semantic_tokens,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    writer = SummaryWriter(args.tensorboard_dir)
    for step in range(1, args.max_steps + 1):
        batch = {key: value.to(device) for key, value in next(batches).items()}
        optimizer.zero_grad(set_to_none=True)
        losses = nar_losses(model, batch)
        losses["total"].backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % args.log_interval == 0 or step == 1:
            for name, value in losses.items():
                writer.add_scalar(f"stage8/{name}_loss", float(value), step)
            writer.add_scalar("stage8/grad_norm", float(grad_norm), step)
            writer.flush()
            print(json.dumps({"step": step, **{name: float(value) for name, value in losses.items()}}), flush=True)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "args": vars(args)}, output_dir / "nar_semantic.pt")
    writer.close()
    print(json.dumps({"status": "complete", "output": str(output_dir / "nar_semantic.pt")}))


if __name__ == "__main__":
    main()

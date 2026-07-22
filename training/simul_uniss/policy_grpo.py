"""Bootstrap GRPO training for the Simul-UniSS WAIT/WRITE policy."""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
from typing import Iterator

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, IterableDataset
from torch.utils.tensorboard import SummaryWriter


class ScheduleActionDataset(IterableDataset):
    def __init__(self, schedule_path: str | Path) -> None:
        self.path = Path(schedule_path)

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                item = json.loads(line)
                source_total = max(1, int(item["source_glm_length"]))
                target_total = max(1, int(item["target_text_length"]))
                committed = 0
                previous_source = 0
                for event in item["events"]:
                    source_count = int(event["source_ctc_count_proxy"])
                    target_supported = int(event["target_ctc_count_proxy"])
                    is_final = float(bool(event["source_is_final"]))
                    features = torch.tensor(
                        [
                            source_count / source_total,
                            target_supported / target_total,
                            committed / target_total,
                            (source_count - previous_source) / source_total,
                            is_final,
                        ],
                        dtype=torch.float32,
                    )
                    label = torch.tensor(1 if event["action"] == "write" else 0, dtype=torch.long)
                    yield features, label
                    if event["action"] == "write":
                        committed += len(event["target_text_ids"])
                    previous_source = source_count


class ActionPolicy(nn.Module):
    def __init__(self, hidden_size: int = 64) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(5, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 2),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)


def rollout_rewards(
    actions: torch.Tensor,
    labels: torch.Tensor,
    features: torch.Tensor,
) -> torch.Tensor:
    """Reward correct decisions, penalize premature writes, and favor safe early writes."""

    labels = labels.unsqueeze(1).expand_as(actions)
    correct = torch.where(actions == labels, 1.0, -1.0)
    premature = ((actions == 1) & (labels == 0)).float() * -2.0
    unnecessary_wait = ((actions == 0) & (labels == 1)).float() * -0.5
    final_wait = ((actions == 0) & (features[:, 4:5] > 0.5)).float() * -5.0
    safe_write = ((actions == 1) & (labels == 1)).float()
    latency_bonus = safe_write * (1.0 - features[:, 0:1]) * 0.2
    return correct + premature + unnecessary_wait + final_wait + latency_bonus


def grpo_loss(
    policy: ActionPolicy,
    reference: ActionPolicy,
    features: torch.Tensor,
    labels: torch.Tensor,
    group_size: int,
    kl_beta: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    logits = policy(features)
    distribution = torch.distributions.Categorical(logits=logits)
    actions = distribution.sample((group_size,)).transpose(0, 1)
    rewards = rollout_rewards(actions, labels, features)
    mean = rewards.mean(dim=1, keepdim=True)
    std = rewards.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-4)
    advantages = (rewards - mean) / std
    log_probs = F.log_softmax(logits, dim=-1)
    sampled_log_probs = log_probs.unsqueeze(1).expand(-1, group_size, -1).gather(
        2, actions.unsqueeze(-1)
    ).squeeze(-1)
    with torch.no_grad():
        reference_logits = reference(features)
    reference_probs = F.softmax(reference_logits, dim=-1)
    kl = F.kl_div(F.log_softmax(logits, dim=-1), reference_probs, reduction="batchmean").clamp_min(0.0)
    loss = -(advantages.detach() * sampled_log_probs).mean() + kl_beta * kl
    metrics = {
        "reward_mean": float(rewards.mean()),
        "reward_max": float(rewards.max()),
        "kl": float(kl),
        "write_rate": float((actions == 1).float().mean()),
        "premature_write_rate": float(((actions == 1) & (labels.unsqueeze(1) == 0)).float().mean()),
    }
    return loss, metrics


def infinite_batches(loader: DataLoader):
    while True:
        yield from loader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schedules", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tensorboard-dir", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--sft-steps", type=int, default=200)
    parser.add_argument("--grpo-steps", type=int, default=500)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--kl-beta", type=float, default=0.02)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260722)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.group_size < 2:
        raise ValueError("group_size must be at least 2")
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    loader = DataLoader(ScheduleActionDataset(args.schedules), batch_size=args.batch_size, num_workers=0)
    batches = infinite_batches(loader)
    policy = ActionPolicy(args.hidden_size).to(device)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.learning_rate)
    writer = SummaryWriter(args.tensorboard_dir)

    for step in range(1, args.sft_steps + 1):
        features, labels = next(batches)
        features, labels = features.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(policy(features), labels)
        loss.backward()
        optimizer.step()
        if step % args.log_interval == 0 or step == 1:
            accuracy = (policy(features).argmax(dim=-1) == labels).float().mean()
            writer.add_scalar("stage7/sft_loss", float(loss), step)
            writer.add_scalar("stage7/sft_accuracy", float(accuracy), step)

    reference = copy.deepcopy(policy).eval()
    for parameter in reference.parameters():
        parameter.requires_grad = False
    for step in range(1, args.grpo_steps + 1):
        features, labels = next(batches)
        features, labels = features.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss, metrics = grpo_loss(
            policy, reference, features, labels, args.group_size, args.kl_beta
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimizer.step()
        global_step = args.sft_steps + step
        if step % args.log_interval == 0 or step == 1:
            writer.add_scalar("stage7/grpo_loss", float(loss), global_step)
            for name, value in metrics.items():
                writer.add_scalar(f"stage7/{name}", value, global_step)
            writer.flush()
            print(json.dumps({"step": step, "loss": float(loss), **metrics}, sort_keys=True), flush=True)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model": policy.state_dict(), "reference": reference.state_dict(), "args": vars(args)},
        output_dir / "policy_grpo.pt",
    )
    writer.close()
    print(json.dumps({"status": "complete", "output": str(output_dir / "policy_grpo.pt")}))


if __name__ == "__main__":
    main()

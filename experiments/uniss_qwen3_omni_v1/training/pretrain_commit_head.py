"""Megatron entry point for CRISP's commit head (training/commit_head_model.py).

The shape of the other trainers here -- TP=PP=1, the model a HuggingFaceModule, the batch built in
forward_step -- with rows from si_commit_head.py prep (one npz: features, both labels, the held-out
split). A sample is a chunk of --head-chunk rows; every pass over the training rows is a fresh
permutation of all of them, so chunks never repeat their rows' grouping.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import OrderedDict

import numpy as np
import torch

from training.pretrain_uniss_megatron import load_megatron_runtime

from experiments.uniss_omni_s2st_v1.training.pretrain_omni_talker_megatron import disable_dataset_helpers
from experiments.uniss_qwen3_omni_v1.training.commit_head_model import METRIC_NAMES

_ROWS = None


def rows(path: str):
    global _ROWS
    if _ROWS is None:
        _ROWS = np.load(path)
    return _ROWS


def add_head_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    g = parser.add_argument_group(title="CRISP commit head")
    g.add_argument("--head-rows", type=str, required=True, help="si_commit_head.py prep output (.npz)")
    g.add_argument("--head-label", choices=("cons", "final"), default="cons")
    g.add_argument("--head-chunk", type=int, default=1024, help="rows a sample")
    g.add_argument("--head-seed", type=int, default=20261022)
    return parser


class RowChunks(torch.utils.data.Dataset):
    """Sample i: chunk i of the training rows, permuted afresh every pass."""

    def __init__(self, x, y, train_idx, chunk: int, n_samples: int, seed: int):
        self.x, self.y, self.chunk = x, y, chunk
        self.per_pass = len(train_idx) // chunk
        rng = np.random.default_rng(seed)
        self.passes = [rng.permutation(train_idx) for _ in range(math.ceil(n_samples / self.per_pass))]
        self.n = n_samples

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        p, j = divmod(int(i), self.per_pass)
        take = np.sort(self.passes[p][j * self.chunk: (j + 1) * self.chunk])
        return {"x": torch.from_numpy(np.asarray(self.x[take], dtype=np.float32)),
                "y": torch.from_numpy(np.asarray(self.y[take], dtype=np.float32))}


def model_provider(pre_process=True, post_process=True, vp_stage=None, config=None, pg_collection=None):
    runtime = load_megatron_runtime()
    args = runtime.megatron_gpt.get_args()
    if not pre_process or not post_process or vp_stage is not None:
        raise ValueError("the commit head is restricted to TP=PP=1")
    from experiments.uniss_qwen3_omni_v1.training.commit_head_model import CommitHeadModel

    if config is None:
        config = runtime.gpt_config_from_args(args)
    data = rows(args.head_rows)
    model = CommitHeadModel(config, mean=data["mean"], std=data["std"], pg_collection=pg_collection)
    if (torch.distributed.get_rank() if torch.distributed.is_initialized() else 0) == 0:
        print(json.dumps({"model": "commit_head", "params": sum(p.numel() for p in model.parameters()),
                          "label": args.head_label}), flush=True)
    return model


def loss_func(output_tensor):
    return output_tensor[0], OrderedDict((n, output_tensor[i].detach().clone()) for i, n in enumerate(METRIC_NAMES))


def forward_step(data_iterator, model):
    batch = next(data_iterator)
    dev = torch.cuda.current_device()
    x = batch["x"].to(dev, non_blocking=True).reshape(-1, batch["x"].shape[-1])
    y = batch["y"].to(dev, non_blocking=True).reshape(-1)
    return model(x, y), loss_func


def train_valid_test_datasets_provider(train_val_test_num_samples, vp_stage=None):
    runtime = load_megatron_runtime()
    args = runtime.megatron_gpt.get_args()
    data = rows(args.head_rows)
    train_idx = np.flatnonzero(~data["val"])
    n = int(args.train_iters) * int(args.global_batch_size)
    train = RowChunks(data["x"], data[f"y_{args.head_label}"], train_idx, args.head_chunk, n, args.head_seed)
    runtime.print_rank_0(f"commit head: {len(train_idx):,} training rows, {train.per_pass:,} chunks of "
                         f"{args.head_chunk} a pass, {n / train.per_pass:.2f} passes, label {args.head_label}")
    return train, None, None


train_valid_test_datasets_provider.is_distributed = True


def main() -> None:
    runtime = load_megatron_runtime()
    args = runtime.parse_and_validate_args(extra_args_provider=add_head_args,
                                           args_defaults={"tokenizer_type": "NullTokenizer"})
    disable_dataset_helpers()
    model_config = runtime.gpt_config_from_args(args)
    full_config = runtime.pretrain_cfg_container_from_args(args, model_config)
    full_config.model = None
    runtime.pretrain(full_config, train_valid_test_datasets_provider, runtime.ModelType.encoder_or_decoder,
                     forward_step, model_provider=model_provider)
    print("COMMIT_HEAD_MEGATRON_DONE", flush=True)


if __name__ == "__main__":
    main()

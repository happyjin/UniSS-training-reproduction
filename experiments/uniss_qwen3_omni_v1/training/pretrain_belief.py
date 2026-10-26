"""Megatron entry point for the commit belief (training/belief_model.py).

TP=PP=1, the model a HuggingFaceModule, the batch built in the loader's workers from si_belief.py prep (one
npz: unit features, labels, trajectory offsets, held-out split). A sample is a chunk of --belief-chunk
trajectories; every pass is a fresh permutation of all training trajectories.
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
from experiments.uniss_qwen3_omni_v1.training.belief_model import METRIC_NAMES

_ROWS = None


def rows(path: str) -> dict:
    global _ROWS
    if _ROWS is None:
        _ROWS = dict(np.load(path))
    return _ROWS


def add_belief_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    g = parser.add_argument_group(title="commit belief")
    g.add_argument("--belief-rows", type=str, required=True, help="si_belief.py prep output (.npz)")
    g.add_argument("--belief-chunk", type=int, default=512, help="trajectories a sample")
    g.add_argument("--belief-seed", type=int, default=20261022)
    return parser


class TrajectoryChunks(torch.utils.data.Dataset):
    def __init__(self, path: str, train_hyps, chunk: int, n_samples: int, seed: int):
        self.path, self.chunk = path, chunk
        self.per_pass = len(train_hyps) // chunk
        rng = np.random.default_rng(seed)
        self.passes = [rng.permutation(train_hyps) for _ in range(math.ceil(n_samples / self.per_pass))]
        self.n = n_samples

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        from experiments.uniss_qwen3_omni_v1.evaluation.si_belief import batch_of

        p, j = divmod(int(i), self.per_pass)
        return batch_of(rows(self.path), np.sort(self.passes[p][j * self.chunk: (j + 1) * self.chunk]))

    @staticmethod
    def collate_fn(items):
        """Chunks pad to their own longest trajectory: pad again to the micro-batch's."""
        S = max(it["obs"].shape[1] for it in items)
        out = {}
        for k in ("x0", "obs", "z", "m"):
            parts = []
            for it in items:
                v = it[k]
                if k != "x0" and v.shape[1] < S:
                    pad = [0, 0, 0, S - v.shape[1]] if k == "obs" else [0, S - v.shape[1]]
                    v = torch.nn.functional.pad(v, pad)
                parts.append(v)
            out[k] = torch.cat(parts, dim=0)
        return out


def model_provider(pre_process=True, post_process=True, vp_stage=None, config=None, pg_collection=None):
    runtime = load_megatron_runtime()
    args = runtime.megatron_gpt.get_args()
    if not pre_process or not post_process or vp_stage is not None:
        raise ValueError("the belief model is restricted to TP=PP=1")
    from experiments.uniss_qwen3_omni_v1.training.belief_model import BeliefModel

    if config is None:
        config = runtime.gpt_config_from_args(args)
    d = rows(args.belief_rows)
    model = BeliefModel(config, mx=d["mx"], sx=d["sx"], mo=d["mo"], so=d["so"], pg_collection=pg_collection)
    if (torch.distributed.get_rank() if torch.distributed.is_initialized() else 0) == 0:
        print(json.dumps({"model": "belief", "params": sum(p.numel() for p in model.parameters())}), flush=True)
    return model


def loss_func(output_tensor):
    return output_tensor[0], OrderedDict((n, output_tensor[i].detach().clone()) for i, n in enumerate(METRIC_NAMES))


def forward_step(data_iterator, model):
    batch = next(data_iterator)
    dev = torch.cuda.current_device()
    return model(*(batch[k].to(dev, non_blocking=True) for k in ("x0", "obs", "z", "m"))), loss_func


def train_valid_test_datasets_provider(train_val_test_num_samples, vp_stage=None):
    runtime = load_megatron_runtime()
    args = runtime.megatron_gpt.get_args()
    d = rows(args.belief_rows)
    train_hyps = np.flatnonzero(~d["val"])
    n = int(args.train_iters) * int(args.global_batch_size)
    train = TrajectoryChunks(args.belief_rows, train_hyps, args.belief_chunk, n, args.belief_seed)
    runtime.print_rank_0(f"belief: {len(train_hyps):,} training trajectories, {train.per_pass:,} chunks of "
                         f"{args.belief_chunk} a pass, {n / train.per_pass:.2f} passes")
    return train, None, None


train_valid_test_datasets_provider.is_distributed = True


def main() -> None:
    runtime = load_megatron_runtime()
    from experiments.uniss_omni_s2st_v1.training.omni_megatron_dataset import install_omni_collate

    args = runtime.parse_and_validate_args(extra_args_provider=add_belief_args,
                                           args_defaults={"tokenizer_type": "NullTokenizer"})
    install_omni_collate()
    disable_dataset_helpers()
    model_config = runtime.gpt_config_from_args(args)
    full_config = runtime.pretrain_cfg_container_from_args(args, model_config)
    full_config.model = None
    runtime.pretrain(full_config, train_valid_test_datasets_provider, runtime.ModelType.encoder_or_decoder,
                     forward_step, model_provider=model_provider)
    print("BELIEF_MEGATRON_DONE", flush=True)


if __name__ == "__main__":
    main()

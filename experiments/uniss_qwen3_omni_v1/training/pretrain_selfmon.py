"""Megatron entry point for Plan 2's amortised self-monitoring (training/selfmon_model.py).

TP=PP=1, the model a HuggingFaceModule holding the talker, as T1. Rows: every block of si_infer --dump-seq results
(results.jsonl under --selfmon-rows), its input and the chosen continuation; blocks the guard had to force shut are
left out. A sample is a group of rows of similar length (at most --selfmon-token-budget padded tokens); every pass is
a fresh permutation of all rows, sorted by length only inside windows, and its groups are shuffled again.
"""

from __future__ import annotations

import argparse
import glob
import json
from collections import OrderedDict

import numpy as np
import torch

from training.pretrain_uniss_megatron import load_megatron_runtime

from experiments.uniss_omni_s2st_v1.training.pretrain_omni_talker_megatron import disable_dataset_helpers
from experiments.uniss_qwen3_omni_v1.training.selfmon_model import METRIC_NAMES

END_SEMANTIC, WAIT_READ = 180392, 180395


def add_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    g = parser.add_argument_group(title="Plan 2 self-monitoring distillation")
    g.add_argument("--uniss-path", required=True, help="the talker to start from (T1c@1500's export)")
    g.add_argument("--ref-path", default="", help="the frozen reference for the KL; default --uniss-path")
    g.add_argument("--selfmon-rows", required=True, help="glob of si_infer --dump-seq results.jsonl / part_*.jsonl")
    g.add_argument("--selfmon-w-ref", type=float, default=0.5)
    g.add_argument("--selfmon-close-weight", type=float, default=3.0)
    g.add_argument("--selfmon-token-budget", type=int, default=24000)
    g.add_argument("--selfmon-max-len", type=int, default=3000)
    g.add_argument("--selfmon-window", type=int, default=2000)
    g.add_argument("--selfmon-seed", type=int, default=20261022)
    g.add_argument("--omni-no-recompute", action="store_true")
    return parser


def load_rows(pattern: str, max_len: int) -> list[tuple[list[int], int]]:
    rows = []
    for f in sorted(glob.glob(pattern)):
        for line in open(f, encoding="utf-8"):
            r = json.loads(line)
            for b in r["blocks"]:
                if "train_ids" in b and not b.get("forced") and len(b["train_ids"]) <= max_len \
                        and len(b["train_ids"]) > b["label_from"]:
                    rows.append((b["train_ids"], b["label_from"]))
    return rows


class RowGroups(torch.utils.data.Dataset):
    def __init__(self, rows, budget: int, window: int, n_samples: int, seed: int):
        self.rows = rows
        lens = np.asarray([len(r[0]) for r in rows])
        rng = np.random.default_rng(seed)
        self.groups, self.passes = [], 0
        while len(self.groups) < n_samples:
            perm = rng.permutation(len(rows))
            out = []
            for s in range(0, len(perm), window):
                w = perm[s: s + window]
                w = w[np.argsort(lens[w], kind="stable")]
                cur, longest = [], 0
                for i in w:
                    nl = max(longest, int(lens[i]))
                    if cur and nl * (len(cur) + 1) > budget:
                        out.append(cur)
                        cur, nl = [], int(lens[i])
                    cur.append(int(i))
                    longest = nl
                if cur:
                    out.append(cur)
            self.groups += [out[k] for k in rng.permutation(len(out))]
            self.passes += 1
        self.n = n_samples

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        g = self.groups[int(i)]
        T = max(len(self.rows[k][0]) for k in g)
        ids = np.zeros((len(g), T), dtype=np.int64)
        mask = np.zeros((len(g), T), dtype=np.int64)
        b, p, lab, close = [], [], [], []
        for j, k in enumerate(g):
            seq, start = self.rows[k]
            ids[j, : len(seq)] = seq
            mask[j, : len(seq)] = 1
            for q in range(start, len(seq)):
                b.append(j)
                p.append(q - 1)
                lab.append(seq[q])
                close.append(seq[q] in (END_SEMANTIC, WAIT_READ))
        return {"ids": torch.from_numpy(ids), "mask": torch.from_numpy(mask), "b": torch.tensor(b), "p": torch.tensor(p),
                "labels": torch.tensor(lab), "is_close": torch.tensor(close)}


def model_provider(pre_process=True, post_process=True, vp_stage=None, config=None, pg_collection=None):
    runtime = load_megatron_runtime()
    args = runtime.megatron_gpt.get_args()
    if not pre_process or not post_process or vp_stage is not None:
        raise ValueError("restricted to TP=PP=1")
    from experiments.uniss_qwen3_omni_v1.training.selfmon_model import SelfMonModel

    if config is None:
        config = runtime.gpt_config_from_args(args)
    return SelfMonModel(config, uniss_path=args.uniss_path, ref_path=args.ref_path or args.uniss_path,
                        w_ref=args.selfmon_w_ref, close_weight=args.selfmon_close_weight,
                        recompute=not args.omni_no_recompute, pg_collection=pg_collection)


def loss_func(output_tensor):
    return output_tensor[0], OrderedDict((n, output_tensor[i].detach().clone()) for i, n in enumerate(METRIC_NAMES))


def forward_step(data_iterator, model):
    batch = next(data_iterator)
    dev = torch.cuda.current_device()
    t = {k: v[0].to(dev, non_blocking=True) for k, v in batch.items()}   # one group a micro-batch
    return model(t["ids"], t["mask"], t["b"], t["p"], t["labels"], t["is_close"]), loss_func


def train_valid_test_datasets_provider(train_val_test_num_samples, vp_stage=None):
    runtime = load_megatron_runtime()
    args = runtime.megatron_gpt.get_args()
    rows = load_rows(args.selfmon_rows, args.selfmon_max_len)
    n = int(args.train_iters) * int(args.global_batch_size)
    train = RowGroups(rows, args.selfmon_token_budget, args.selfmon_window, n, args.selfmon_seed)
    runtime.print_rank_0(f"selfmon: {len(rows):,} block rows, {len(train.groups):,} groups over {train.passes} passes "
                         f"for {n:,} samples")
    return train, None, None


train_valid_test_datasets_provider.is_distributed = True


def main() -> None:
    runtime = load_megatron_runtime()
    args = runtime.parse_and_validate_args(extra_args_provider=add_args, args_defaults={"tokenizer_type": "NullTokenizer"})
    from experiments.uniss_qwen3_omni_v1.training.pretrain_e2 import keep_lr_groups_apart

    keep_lr_groups_apart()
    disable_dataset_helpers()
    model_config = runtime.gpt_config_from_args(args)
    full_config = runtime.pretrain_cfg_container_from_args(args, model_config)
    full_config.model = None
    runtime.pretrain(full_config, train_valid_test_datasets_provider, runtime.ModelType.encoder_or_decoder,
                     forward_step, model_provider=model_provider)
    print("SELFMON_MEGATRON_DONE", flush=True)


if __name__ == "__main__":
    main()

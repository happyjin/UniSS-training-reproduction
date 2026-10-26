"""Megatron entry point for Plan 2's CTC head (training/ctc_head_model.py).

TP=PP=1, the model a HuggingFaceModule, the batch built in forward_step, as the other trainers here;
rows from si_ctc_head.py prep. A sample is a bucket of utterances of similar length holding at most
--ctc-frame-budget codes (padded); every pass is a fresh permutation of all training rows, sorted by
length only inside windows of --ctc-window rows, and its buckets are shuffled again.
"""

from __future__ import annotations

import argparse
import json
from collections import OrderedDict

import numpy as np
import torch

from training.pretrain_uniss_megatron import load_megatron_runtime

from experiments.uniss_omni_s2st_v1.training.pretrain_omni_talker_megatron import disable_dataset_helpers
from experiments.uniss_qwen3_omni_v1.training.ctc_head_model import METRIC_NAMES

_ROWS = None


def rows(path: str):
    global _ROWS
    if _ROWS is None:
        z = np.load(path)
        _ROWS = {k: z[k] for k in ("codes", "code_off", "text", "text_off", "val", "chars")}
    return _ROWS


def add_ctc_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    g = parser.add_argument_group(title="Plan 2 CTC head")
    g.add_argument("--ctc-rows", type=str, required=True, help="si_ctc_head.py prep output (.npz)")
    g.add_argument("--ctc-frame-budget", type=int, default=48000, help="padded codes a bucket")
    g.add_argument("--ctc-window", type=int, default=4000, help="rows sorted by length together")
    g.add_argument("--ctc-d", type=int, default=512)
    g.add_argument("--ctc-layers", type=int, default=8)
    g.add_argument("--ctc-heads", type=int, default=8)
    g.add_argument("--ctc-ff", type=int, default=2048)
    g.add_argument("--ctc-seed", type=int, default=20261022)
    return parser


def buckets_of(lens: np.ndarray, idx: np.ndarray, budget: int, window: int, rng) -> list[np.ndarray]:
    perm = rng.permutation(idx)
    out = []
    for s in range(0, len(perm), window):
        w = perm[s: s + window]
        w = w[np.argsort(lens[w], kind="stable")]
        cur, longest = [], 0
        for i in w:
            longest_new = max(longest, int(lens[i]))
            if cur and longest_new * (len(cur) + 1) > budget:
                out.append(np.asarray(cur))
                cur, longest_new = [], int(lens[i])
            cur.append(int(i))
            longest = longest_new
        if cur:
            out.append(np.asarray(cur))
    order = rng.permutation(len(out))
    return [out[k] for k in order]


class CodeBuckets(torch.utils.data.Dataset):
    def __init__(self, data, train_idx, budget: int, window: int, n_samples: int, seed: int):
        self.d = data
        self.lens = np.diff(data["code_off"])
        rng = np.random.default_rng(seed)
        self.buckets, self.passes = [], 0
        while len(self.buckets) < n_samples:
            self.buckets += buckets_of(self.lens, train_idx, budget, window, rng)
            self.passes += 1
        self.n = n_samples

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        from experiments.uniss_qwen3_omni_v1.evaluation.si_ctc_head import PAD_CODE

        b = self.buckets[int(i)]
        co, to = self.d["code_off"], self.d["text_off"]
        lens = self.lens[b]
        T = int(lens.max())
        codes = np.full((len(b), T), PAD_CODE, dtype=np.int64)
        for k, r in enumerate(b):
            codes[k, : lens[k]] = self.d["codes"][co[r]: co[r + 1]]
        targets = np.concatenate([self.d["text"][to[r]: to[r + 1]] for r in b]).astype(np.int64)
        text_len = np.asarray([to[r + 1] - to[r] for r in b], dtype=np.int64)
        return {"codes": torch.from_numpy(codes), "pad": torch.from_numpy(codes == PAD_CODE),
                "targets": torch.from_numpy(targets), "code_len": torch.from_numpy(lens.astype(np.int64)),
                "text_len": torch.from_numpy(text_len)}


def model_provider(pre_process=True, post_process=True, vp_stage=None, config=None, pg_collection=None):
    runtime = load_megatron_runtime()
    args = runtime.megatron_gpt.get_args()
    if not pre_process or not post_process or vp_stage is not None:
        raise ValueError("the CTC head is restricted to TP=PP=1")
    from experiments.uniss_qwen3_omni_v1.training.ctc_head_model import CTCHeadModel

    if config is None:
        config = runtime.gpt_config_from_args(args)
    n_chars = len(json.loads(str(rows(args.ctc_rows)["chars"])))
    model = CTCHeadModel(config, n_chars=n_chars, pg_collection=pg_collection, d=args.ctc_d, layers=args.ctc_layers,
                         heads=args.ctc_heads, ff=args.ctc_ff)
    if (torch.distributed.get_rank() if torch.distributed.is_initialized() else 0) == 0:
        print(json.dumps({"model": "ctc_head", "params_m": round(sum(p.numel() for p in model.parameters()) / 1e6, 1),
                          "chars": n_chars}), flush=True)
    return model


def loss_func(output_tensor):
    return output_tensor[0], OrderedDict((n, output_tensor[i].detach().clone()) for i, n in enumerate(METRIC_NAMES))


def forward_step(data_iterator, model):
    b = next(data_iterator)
    dev = torch.cuda.current_device()
    t = {k: v[0].to(dev, non_blocking=True) for k, v in b.items()}   # micro-batch of one bucket
    return model(t["codes"], t["pad"], t["targets"], t["code_len"], t["text_len"]), loss_func


def train_valid_test_datasets_provider(train_val_test_num_samples, vp_stage=None):
    runtime = load_megatron_runtime()
    args = runtime.megatron_gpt.get_args()
    data = rows(args.ctc_rows)
    train_idx = np.flatnonzero(~data["val"])
    n = int(args.train_iters) * int(args.global_batch_size)
    train = CodeBuckets(data, train_idx, args.ctc_frame_budget, args.ctc_window, n, args.ctc_seed)
    runtime.print_rank_0(f"ctc head: {len(train_idx):,} training rows, {len(train.buckets):,} buckets over "
                         f"{train.passes} passes for {n:,} samples")
    return train, None, None


train_valid_test_datasets_provider.is_distributed = True


def main() -> None:
    runtime = load_megatron_runtime()
    args = runtime.parse_and_validate_args(extra_args_provider=add_ctc_args,
                                           args_defaults={"tokenizer_type": "NullTokenizer"})
    disable_dataset_helpers()
    model_config = runtime.gpt_config_from_args(args)
    full_config = runtime.pretrain_cfg_container_from_args(args, model_config)
    full_config.model = None
    runtime.pretrain(full_config, train_valid_test_datasets_provider, runtime.ModelType.encoder_or_decoder,
                     forward_step, model_provider=model_provider)
    print("CTC_HEAD_MEGATRON_DONE", flush=True)


if __name__ == "__main__":
    main()

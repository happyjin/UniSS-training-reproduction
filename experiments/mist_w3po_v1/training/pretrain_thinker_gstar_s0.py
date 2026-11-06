"""W3PO stage-0 SFT entry (plan §4): experiments/uniss_qwen3_omni_v1/training/pretrain_thinker_gstar.py with one change --
rows may carry "ctx" (S0-4 cross-window context), placed in the user turn exactly as the inference engine does
(core/engine.Prompter: head_a + ctx + head_b + prefix, tokenised piecewise; empty ctx = the v8 prompt). Model, loss,
loader, optimiser and every other argument are the v8 ones (thinker_gstar_model.ThinkerGStarModel).
"""
from __future__ import annotations

import numpy as np
import torch

from experiments.uniss_qwen3_omni_v1.training import pretrain_thinker_gstar as pg
from experiments.uniss_qwen3_omni_v1.training.thinker_gstar_model import (
    EOS_ID, IGNORE, N_SLOT, n_speaker_tokens, post_to_tokens)

_PROMPTERS = {}


def prompter(direction: str):
    if direction not in _PROMPTERS:
        from experiments.mist_w3po_v1.core.engine import Prompter

        _PROMPTERS[direction] = Prompter(direction)
    return _PROMPTERS[direction]


def build_batch(items: list[dict], path: str) -> dict:
    from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import _get_feat_extract_output_lengths

    proc = pg.processor(path)
    tok = proc.tokenizer
    f = proc.feature_extractor([it["wav"] for it in items], sampling_rate=16000, padding=True, truncation=False,
                               return_attention_mask=True, return_tensors="pt")
    n_audio = _get_feat_extract_output_lengths(f["attention_mask"].sum(1))
    seqs, labs, posts = [], [], []
    for it, n in zip(items, n_audio.tolist()):
        if it["post"] is not None and it["inst"] == "gstar":
            p_ids = prompter(it["dir"]).ids(n, it.get("ctx") or "", tok(it["prefix"], add_special_tokens=False)["input_ids"])
        else:
            m = n_speaker_tokens(n) if it["post"] is not None else 0
            prompt = pg.head(it["inst"], it["dir"]).replace("<|audio_pad|>", "<|audio_pad|>" * (n + m)) + it["prefix"]
            p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
        r_ids = tok(it["target"], add_special_tokens=False)["input_ids"] + ([EOS_ID] if it["end"] == "eos" else [151649])
        seqs.append(p_ids + r_ids)
        labs.append([IGNORE] * len(p_ids) + r_ids)
        dur = len(it["wav"]) / 16000
        posts.append(post_to_tokens(it["post"], n, dur) if it["post"] is not None else np.zeros((n, N_SLOT), np.float32))
    L = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), L), tok.pad_token_id or 0, dtype=torch.long)
    mask = torch.zeros((len(seqs), L), dtype=torch.long)
    labels = torch.full((len(seqs), L), IGNORE, dtype=torch.long)
    for i, (s, lb) in enumerate(zip(seqs, labs)):
        ids[i, : len(s)] = torch.as_tensor(s)
        mask[i, : len(s)] = 1
        labels[i, : len(lb)] = torch.as_tensor(lb)
    nmax = max(int(n_audio.max()), 1)
    sd = torch.zeros((len(items), nmax, N_SLOT), dtype=torch.float32)
    for i, p in enumerate(posts):
        sd[i, : len(p)] = torch.as_tensor(p)
    return {"input_ids": ids, "attention_mask": mask, "labels": labels, "input_features": f["input_features"],
            "feature_attention_mask": f["attention_mask"], "sd_post": sd, "n_audio": n_audio,
            "has_sd": torch.as_tensor([it["post"] is not None for it in items])}


class GStarRowsS0(pg.GStarRows):
    def __getitem__(self, i):
        item = super().__getitem__(i)
        item["ctx"] = self.rows[int(self.order[int(i)])].get("ctx", "")
        return item

    def collate_fn(self, items):
        return build_batch(items, self.qwen3_path)


def train_valid_test_datasets_provider(train_val_test_num_samples, vp_stage=None):
    from training.pretrain_uniss_megatron import load_megatron_runtime

    runtime = load_megatron_runtime()
    args = runtime.megatron_gpt.get_args()
    n = int(args.train_iters) * int(args.global_batch_size)
    train = GStarRowsS0(args.gstar_rows, n, args.gstar_seed, args.qwen3_path, args.gstar_perturb, args.gstar_oracle)
    runtime.print_rank_0(f"thinker gstar s0: {len(train.rows):,} rows, {n / len(train.rows):.2f} passes, "
                         f"{sum(bool(r.get('ctx')) for r in train.rows):,} with context")
    return train, None, None


train_valid_test_datasets_provider.is_distributed = True


def main() -> None:
    from training.pretrain_uniss_megatron import load_megatron_runtime

    runtime = load_megatron_runtime()
    from experiments.uniss_omni_s2st_v1.training.omni_megatron_dataset import install_omni_collate
    from experiments.uniss_omni_s2st_v1.training.pretrain_omni_talker_megatron import disable_dataset_helpers

    args = runtime.parse_and_validate_args(extra_args_provider=pg.add_args, args_defaults={"tokenizer_type": "NullTokenizer"})
    install_omni_collate()
    disable_dataset_helpers()
    model_config = runtime.gpt_config_from_args(args)
    full_config = runtime.pretrain_cfg_container_from_args(args, model_config)
    full_config.model = None
    runtime.pretrain(full_config, train_valid_test_datasets_provider, runtime.ModelType.encoder_or_decoder,
                     pg.forward_step, model_provider=pg.model_provider)
    print("THINKER_GSTAR_MEGATRON_DONE", flush=True)


if __name__ == "__main__":
    main()

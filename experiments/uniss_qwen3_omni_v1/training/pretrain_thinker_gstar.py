"""Megatron entry point for G-STAR-SI (training/thinker_gstar_model.py; docs/multi_speaker_si_plan_v7.zh-CN.md).

TP=PP=1, the batch built in the loader's workers from the rows of data_clean/build_ami_gstar_si_rows.py: the
window's audio, its Nemotron-3 posteriors on the audio-token grid, the chat template with one <|audio_pad|> per
audio token plus one per speaker token (K:1), the instruction, then the reply -- the only supervised positions.
Rows are a fresh permutation of all training rows every pass (the same loader as pretrain_thinker_sd.py).
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
from experiments.uniss_qwen3_omni_v1.training.thinker_gstar_model import (
    EOS_ID, IGNORE, METRIC_NAMES, NEW0, N_SLOT, n_speaker_tokens, post_to_tokens, special_tokens)

GSTAR_SUFFIX = (" Several people are speaking. For every utterance write <|spk k|><|t a|>translation<|t b|>:"
                " k is the speaker's slot in the speaker stream, a and b are its start and end in 80 ms steps from"
                " the start of the audio, utterances in order of start.")
_PROC = None
_HEADS = {}


def processor(path: str):
    global _PROC
    if _PROC is None:   # once per process; the new tokens get the unused rows NEW0..
        from transformers import Qwen3OmniMoeProcessor

        _PROC = Qwen3OmniMoeProcessor.from_pretrained(path)
        _PROC.tokenizer.add_tokens(special_tokens(), special_tokens=True)
        assert _PROC.tokenizer.convert_tokens_to_ids("<|t0|>") == NEW0, "the new tokens must take the unused rows"
    return _PROC


def gstar_instruction(direction: str = "en2zh") -> str:
    from experiments.uniss_qwen3_omni_v1.training.e2_batch import inference_instruction

    return inference_instruction()["cmn" if direction.endswith("zh") else "eng"] + GSTAR_SUFFIX


def head(inst: str, direction: str = "en2zh") -> str:
    """The chat-template head (one <|audio_pad|> to be expanded); inst "gstar" or "sd" (plain translation)."""
    key = f"{direction}|{inst}"
    if key not in _HEADS:
        from experiments.uniss_qwen3_omni_v1.training.e2_batch import inference_instruction

        ins = gstar_instruction(direction) if inst == "gstar" else inference_instruction()[
            "cmn" if direction.endswith("zh") else "eng"]
        conv = [{"role": "user", "content": [{"type": "audio", "audio": "x.wav"}, {"type": "text", "text": ins}]}]
        _HEADS[key] = _PROC.apply_chat_template([conv], add_generation_prompt=True, tokenize=False)[0]
    return _HEADS[key]


def build_batch(items: list[dict], path: str) -> dict:
    """items: {"wav", "dir", "inst", "prefix", "target", "end", "post" ([F80, 8] or None)} -> model tensors."""
    from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import _get_feat_extract_output_lengths

    proc = processor(path)
    tok = proc.tokenizer
    f = proc.feature_extractor([it["wav"] for it in items], sampling_rate=16000, padding=True, truncation=False,
                               return_attention_mask=True, return_tensors="pt")
    n_audio = _get_feat_extract_output_lengths(f["attention_mask"].sum(1))
    seqs, labs, posts = [], [], []
    for it, n in zip(items, n_audio.tolist()):
        m = n_speaker_tokens(n) if it["post"] is not None else 0
        prompt = head(it["inst"], it["dir"]).replace("<|audio_pad|>", "<|audio_pad|>" * (n + m)) + it["prefix"]
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


def perturb(post: np.ndarray, rng: np.random.Generator, p: float) -> np.ndarray:
    """With probability p: Gaussian noise on the posteriors and a shift of up to one 80 ms frame."""
    if rng.random() >= p:
        return post
    out = np.clip(post + rng.normal(0, 0.1, post.shape), 0, 1)
    s = int(rng.integers(-1, 2))
    return np.roll(out, s, axis=0) if s else out


class GStarRows(torch.utils.data.Dataset):
    def __init__(self, path: str, n_samples: int, seed: int, qwen3_path: str, perturb_p: float, oracle: bool):
        self.qwen3_path, self.perturb_p, self.oracle = qwen3_path, perturb_p, oracle
        self.rows = [json.loads(l) for l in open(path, encoding="utf-8")]
        rng = np.random.default_rng(seed)
        self.order = np.concatenate([rng.permutation(len(self.rows)) for _ in range(math.ceil(n_samples / len(self.rows)))])
        self.n = n_samples
        self.seed = seed

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        import soundfile as sf

        r = self.rows[int(self.order[int(i)])]
        wav, sr = sf.read(r["audio"], dtype="float32", always_2d=True)
        assert sr == 16000, r["audio"]
        wav = wav.mean(axis=1)[: int(r["t_ms"]) * 16]
        if len(wav) < 1600:
            wav = np.pad(wav, (0, 1600 - len(wav)))
        post = None
        if r.get("inst") == "gstar":
            path = (r["post"].replace("/post/offline/", "/post/oracle_offline/").replace("/post/lowlat/", "/post/oracle_lowlat/")
                    if self.oracle else r["post"])
            post = perturb(np.load(path).astype(np.float32), np.random.default_rng(self.seed + int(i)), self.perturb_p)
            if r.get("post_until_ms") is not None:   # streaming rows: frames the diarizer has not finalised are zero
                post = post.copy()
                post[int(r["post_until_ms"]) // 80:] = 0.0
        return {"wav": wav, "dir": r["dir"], "inst": r.get("inst", "sd"), "prefix": r["prefix"], "target": r["target"],
                "end": r["end"], "post": post}

    def collate_fn(self, items):
        return build_batch(items, self.qwen3_path)


def add_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    g = parser.add_argument_group(title="G-STAR-SI")
    g.add_argument("--qwen3-path", default="pretrained_models/Qwen3-Omni-30B-A3B-Instruct")
    g.add_argument("--gstar-rows", required=True)
    g.add_argument("--gstar-lora-rank", type=int, default=32)
    g.add_argument("--gstar-lora-alpha", type=float, default=64.0)
    g.add_argument("--gstar-lora-from", type=int, default=0)
    g.add_argument("--gstar-seed", type=int, default=20261026)
    g.add_argument("--gstar-perturb", type=float, default=0.2)
    g.add_argument("--gstar-no-sd", action="store_true", help="speaker tokens zeroed (O-nosd)")
    g.add_argument("--gstar-oracle", action="store_true", help="reference posteriors instead of Nemotron's")
    g.add_argument("--gstar-init-merged", action="store_true",
                   help="--qwen3-path is a merged G-STAR-SI model: start from its token rows and sd_proj.pt (streaming)")
    g.add_argument("--omni-no-recompute", action="store_true")
    return parser


def model_provider(pre_process=True, post_process=True, vp_stage=None, config=None, pg_collection=None):
    runtime = load_megatron_runtime()
    args = runtime.megatron_gpt.get_args()
    if not pre_process or not post_process or vp_stage is not None:
        raise ValueError("restricted to TP=PP=1")
    from experiments.uniss_qwen3_omni_v1.training.thinker_gstar_model import ThinkerGStarModel

    if config is None:
        config = runtime.gpt_config_from_args(args)
    model = ThinkerGStarModel(config, qwen3_path=args.qwen3_path, lora_rank=args.gstar_lora_rank,
                              lora_alpha=args.gstar_lora_alpha, lora_from=args.gstar_lora_from,
                              recompute=not args.omni_no_recompute, use_sd=not args.gstar_no_sd,
                              init_merged=args.gstar_init_merged, pg_collection=pg_collection)
    processor(args.qwen3_path)
    if (torch.distributed.get_rank() if torch.distributed.is_initialized() else 0) == 0:
        print(json.dumps({"model": "thinker_gstar", **model.trainable_report()}), flush=True)
    return model


def loss_func(output_tensor):
    return output_tensor[0], OrderedDict((n, output_tensor[i].detach().clone()) for i, n in enumerate(METRIC_NAMES))


def forward_step(data_iterator, model):
    batch = next(data_iterator)
    dev = torch.cuda.current_device()
    return model(**{k: v.to(dev, non_blocking=True) for k, v in batch.items()}), loss_func


def train_valid_test_datasets_provider(train_val_test_num_samples, vp_stage=None):
    runtime = load_megatron_runtime()
    args = runtime.megatron_gpt.get_args()
    n = int(args.train_iters) * int(args.global_batch_size)
    train = GStarRows(args.gstar_rows, n, args.gstar_seed, args.qwen3_path, args.gstar_perturb, args.gstar_oracle)
    runtime.print_rank_0(f"thinker gstar: {len(train.rows):,} rows, {n / len(train.rows):.2f} passes")
    return train, None, None


train_valid_test_datasets_provider.is_distributed = True


def main() -> None:
    runtime = load_megatron_runtime()
    from experiments.uniss_omni_s2st_v1.training.omni_megatron_dataset import install_omni_collate

    args = runtime.parse_and_validate_args(extra_args_provider=add_args, args_defaults={"tokenizer_type": "NullTokenizer"})
    install_omni_collate()
    disable_dataset_helpers()
    model_config = runtime.gpt_config_from_args(args)
    full_config = runtime.pretrain_cfg_container_from_args(args, model_config)
    full_config.model = None
    runtime.pretrain(full_config, train_valid_test_datasets_provider, runtime.ModelType.encoder_or_decoder,
                     forward_step, model_provider=model_provider)
    print("THINKER_GSTAR_MEGATRON_DONE", flush=True)


if __name__ == "__main__":
    main()

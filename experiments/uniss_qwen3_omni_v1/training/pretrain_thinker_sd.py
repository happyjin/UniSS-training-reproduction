"""Megatron entry point for Plan 4A (training/thinker_sd_model.py).

TP=PP=1, the model a HuggingFaceModule, the batch built in forward_step from raw rows: the Thinker's chat template
with the audio heard so far (the E1 instruction, as si_thinker_stream prompts it), the committed translation as
the start of the reply, then the target and its end token -- the only supervised positions. Rows are a fresh
permutation of all training rows every pass.
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
from experiments.uniss_qwen3_omni_v1.training.thinker_sd_model import EOS_ID, IGNORE, METRIC_NAMES, WAIT_ID

_PROCESSOR = None
_HEADS = {}


def add_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    g = parser.add_argument_group(title="Plan 4A thinker self-distillation")
    g.add_argument("--qwen3-path", default="pretrained_models/Qwen3-Omni-30B-A3B-Instruct")
    g.add_argument("--sd-rows", required=True, help="rows_train.jsonl from build_thinker_selfdistill")
    g.add_argument("--sd-lora-rank", type=int, default=32)
    g.add_argument("--sd-lora-alpha", type=float, default=64.0)
    g.add_argument("--sd-lora-from", type=int, default=32, help="first text layer with adapters (of 48)")
    g.add_argument("--sd-seed", type=int, default=20261022)
    g.add_argument("--omni-no-recompute", action="store_true")
    return parser


class SDRows(torch.utils.data.Dataset):
    """The batch is built in the loader's workers (audio features and tokenisation on the CPU there), so the
    training process only moves tensors: built in forward_step it held eight GPUs at 30% behind one core each."""

    def __init__(self, path: str, n_samples: int, seed: int, qwen3_path: str):
        self.qwen3_path = qwen3_path
        self.rows = [json.loads(l) for l in open(path, encoding="utf-8")]
        rng = np.random.default_rng(seed)
        self.order = np.concatenate([rng.permutation(len(self.rows)) for _ in range(math.ceil(n_samples / len(self.rows)))])
        self.n = n_samples

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
        return {"wav": wav, "dir": r["dir"], "prefix": r["prefix"], "target": r["target"], "end": r["end"]}

    def collate_fn(self, items):
        global _PROCESSOR
        if _PROCESSOR is None:   # once per worker process
            from transformers import Qwen3OmniMoeProcessor

            _PROCESSOR = Qwen3OmniMoeProcessor.from_pretrained(self.qwen3_path)
        return build_batch({k: [it[k] for it in items] for k in items[0]}, "cpu")


def head(direction: str) -> str:
    if direction not in _HEADS:
        from experiments.uniss_qwen3_omni_v1.training.e2_batch import inference_instruction

        ins = inference_instruction()["cmn" if direction.endswith("zh") else "eng"]
        conv = [{"role": "user", "content": [{"type": "audio", "audio": "x.wav"}, {"type": "text", "text": ins}]}]
        _HEADS[direction] = _PROCESSOR.apply_chat_template([conv], add_generation_prompt=True, tokenize=False)[0]
    return _HEADS[direction]


def build_batch(batch: dict, device) -> dict:
    """Prompt (template + audio + committed prefix) as the processor pads it, then each reply appended after the
    widest prompt -- deconly's layout -- with labels on the reply only."""
    prompts = [head(d) + p for d, p in zip(batch["dir"], batch["prefix"])]
    enc = _PROCESSOR(text=prompts, audio=batch["wav"], sampling_rate=16000, return_tensors="pt", padding=True)
    tok = _PROCESSOR.tokenizer
    pad_id = tok.pad_token_id or 0
    replies = [list(ids) + [WAIT_ID if e == "wait" else EOS_ID]
               for ids, e in zip(tok(list(batch["target"]), add_special_tokens=False)["input_ids"], batch["end"])]
    rows, width = enc["input_ids"].shape
    total = width + max(len(r) for r in replies)
    ids = torch.full((rows, total), pad_id, dtype=torch.long)
    mask = torch.zeros((rows, total), dtype=torch.long)
    labels = torch.full((rows, total), IGNORE, dtype=torch.long)
    ids[:, :width] = enc["input_ids"]
    mask[:, :width] = enc["attention_mask"]
    for k, r in enumerate(replies):
        ids[k, width: width + len(r)] = torch.as_tensor(r)
        mask[k, width: width + len(r)] = 1
        labels[k, width: width + len(r)] = torch.as_tensor(r)
    out = {"input_ids": ids.to(device), "attention_mask": mask.to(device), "labels": labels.to(device)}
    for key in ("input_features", "feature_attention_mask"):
        if key in enc:
            out[key] = enc[key].to(device)
    return out


def model_provider(pre_process=True, post_process=True, vp_stage=None, config=None, pg_collection=None):
    global _PROCESSOR
    runtime = load_megatron_runtime()
    args = runtime.megatron_gpt.get_args()
    if not pre_process or not post_process or vp_stage is not None:
        raise ValueError("restricted to TP=PP=1")
    from transformers import Qwen3OmniMoeProcessor

    from experiments.uniss_qwen3_omni_v1.training.thinker_sd_model import ThinkerSDModel

    if config is None:
        config = runtime.gpt_config_from_args(args)
    model = ThinkerSDModel(config, qwen3_path=args.qwen3_path, lora_rank=args.sd_lora_rank,
                           lora_alpha=args.sd_lora_alpha, lora_from=args.sd_lora_from,
                           recompute=not args.omni_no_recompute, pg_collection=pg_collection)
    _PROCESSOR = Qwen3OmniMoeProcessor.from_pretrained(args.qwen3_path)
    if (torch.distributed.get_rank() if torch.distributed.is_initialized() else 0) == 0:
        print(json.dumps({"model": "thinker_sd", **model.trainable_report()}), flush=True)
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
    train = SDRows(args.sd_rows, n, args.sd_seed, args.qwen3_path)
    runtime.print_rank_0(f"thinker sd: {len(train.rows):,} rows, {n / len(train.rows):.2f} passes")
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
    print("THINKER_SD_MEGATRON_DONE", flush=True)


if __name__ == "__main__":
    main()

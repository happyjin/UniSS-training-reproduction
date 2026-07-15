"""Megatron-LM entrypoint for UniSS packed-token training.

This script is intentionally thin: UniSS samples are already converted to
next-token language-model tensors, so training reuses Megatron-LM's GPT model,
forward step, and loss function while swapping in a packed JSONL dataset.
"""

from __future__ import annotations

import argparse
import inspect
import os
import sys
import time
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MEGATRON_ROOT = REPO_ROOT / "third_party" / "Megatron-LM"
PAPER_SEQ_LENGTH = 18_000
PAPER_GLOBAL_BATCH_SEQUENCES = 128

_PROGRAM_START_TIME = time.time()


def patch_argparse_boolean_optional_action() -> None:
    signature = inspect.signature(argparse.BooleanOptionalAction)
    if "type" in signature.parameters:
        return

    base_action = argparse.BooleanOptionalAction

    class MegatronBooleanOptionalAction(base_action):
        def __init__(self, *args, **kwargs):
            kwargs.pop("type", None)
            super().__init__(*args, **kwargs)

    argparse.BooleanOptionalAction = MegatronBooleanOptionalAction


def patch_argparse_help_formatter_percent() -> None:
    original_expand_help = argparse.HelpFormatter._expand_help

    if getattr(original_expand_help, "_uniss_percent_compat", False):
        return

    def expand_help_percent_compat(self, action):
        try:
            return original_expand_help(self, action)
        except ValueError as exc:
            if "unsupported format character" not in str(exc):
                raise
            return action.help or ""

    expand_help_percent_compat._uniss_percent_compat = True
    argparse.HelpFormatter._expand_help = expand_help_percent_compat


def ensure_megatron_import_path(megatron_root: Path = DEFAULT_MEGATRON_ROOT) -> None:
    """Make this repository and Megatron-LM importable from torchrun."""

    for path in (REPO_ROOT, megatron_root):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


patch_argparse_boolean_optional_action()
patch_argparse_help_formatter_percent()
ensure_megatron_import_path()

from training import constants_uniss as c  # noqa: E402
from training.megatron_uniss_dataset import UniSSPackedJsonlDataset  # noqa: E402

from megatron.core.datasets.utils import Split  # noqa: E402
from megatron.core.enums import ModelType  # noqa: E402
from megatron.training import inprocess_restart, pretrain, print_rank_0, set_startup_timestamps  # noqa: E402
from megatron.training.argument_utils import gpt_config_from_args, pretrain_cfg_container_from_args  # noqa: E402
from megatron.training.arguments import parse_and_validate_args  # noqa: E402

import pretrain_gpt as megatron_gpt  # noqa: E402


def add_uniss_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    group = parser.add_argument_group(title="UniSS packed data")
    group.add_argument(
        "--uniss-packed-train",
        type=str,
        required=True,
        help="Packed UniSS train JSONL produced by training/pack_sequences.py.",
    )
    group.add_argument(
        "--uniss-packed-valid",
        type=str,
        default=None,
        help="Optional packed UniSS validation JSONL. Required when eval is enabled.",
    )
    group.add_argument(
        "--uniss-packed-test",
        type=str,
        default=None,
        help="Optional packed UniSS test JSONL.",
    )
    group.add_argument(
        "--uniss-strict-paper-config",
        action="store_true",
        help="Require the paper's 18k sequence packing and 128-sequence global batch.",
    )
    return parser


def add_extra_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    if megatron_gpt.has_nvidia_modelopt:
        parser = megatron_gpt.add_modelopt_args(parser)
    return add_uniss_args(parser)


def _has_eval_enabled(args: SimpleNamespace) -> bool:
    return bool(getattr(args, "full_validation", False)) or int(getattr(args, "eval_iters", 0) or 0) > 0


def validate_uniss_args(args: SimpleNamespace) -> None:
    """Validate options that must be true for UniSS packed sequence training."""

    if not getattr(args, "sft", False):
        raise ValueError("UniSS packed training requires Megatron's --sft path so cu_seqlens are honored.")
    if getattr(args, "create_attention_mask_in_dataloader", False):
        raise ValueError("UniSS packed JSONL does not emit dense attention masks; omit --create-attention-mask-in-dataloader.")
    if int(getattr(args, "context_parallel_size", 1)) != 1:
        raise ValueError("UniSS JSONL entrypoint currently supports --context-parallel-size 1 only.")
    if _has_eval_enabled(args) and not getattr(args, "uniss_packed_valid", None):
        raise ValueError("Pass --uniss-packed-valid, or disable eval with --eval-iters 0.")

    vocab_size = getattr(args, "vocab_size", None)
    if vocab_size is not None and int(vocab_size) != c.VOCAB_SIZE:
        raise ValueError(f"UniSS requires --vocab-size {c.VOCAB_SIZE}, got {vocab_size}.")

    if getattr(args, "uniss_strict_paper_config", False):
        seq_length = int(getattr(args, "seq_length", 0))
        global_batch_size = int(getattr(args, "global_batch_size", 0))
        if seq_length != PAPER_SEQ_LENGTH:
            raise ValueError(f"Paper config requires --seq-length {PAPER_SEQ_LENGTH}, got {seq_length}.")
        if global_batch_size != PAPER_GLOBAL_BATCH_SEQUENCES:
            raise ValueError(
                "Paper config requires --global-batch-size "
                f"{PAPER_GLOBAL_BATCH_SEQUENCES} packed sequences, got {global_batch_size}."
            )


def _dataset(path: str | None, seq_length: int, split: Split) -> UniSSPackedJsonlDataset | None:
    if path is None:
        return None
    dataset = UniSSPackedJsonlDataset(path, seq_length=seq_length)
    dataset.split = split
    return dataset


def build_uniss_packed_datasets(args: SimpleNamespace):
    seq_length = int(args.seq_length)
    train_ds = _dataset(args.uniss_packed_train, seq_length, Split.train)
    valid_ds = _dataset(args.uniss_packed_valid, seq_length, Split.valid)
    test_ds = _dataset(args.uniss_packed_test, seq_length, Split.test)
    return train_ds, valid_ds, test_ds


def train_valid_test_datasets_provider(train_val_test_num_samples, vp_stage=None):
    del train_val_test_num_samples, vp_stage
    args = megatron_gpt.get_args()
    print_rank_0("> building UniSS packed JSONL datasets ...")
    datasets = build_uniss_packed_datasets(args)
    print_rank_0("> finished creating UniSS packed JSONL datasets ...")
    return datasets


def maybe_wrap_pretrain_after_parse(args: SimpleNamespace):
    if not getattr(args, "inprocess_restart", False):
        return pretrain, None

    wrapped_pretrain = inprocess_restart.inprocess_restart(pretrain, args)
    import torch

    store = torch.distributed.TCPStore(
        host_name=os.environ["MASTER_ADDR"],
        port=int(os.environ["MASTER_PORT"]) + 1,
        world_size=int(os.getenv("WORLD_SIZE", "1")),
        is_master=(int(os.getenv("RANK", "0")) == 0),
        timeout=timedelta(seconds=300),
        wait_for_workers=True,
        use_libuv=True,
    )
    return wrapped_pretrain, store


def main() -> None:
    set_startup_timestamps(program_start=_PROGRAM_START_TIME, main_entry=time.time())

    args = parse_and_validate_args(
        extra_args_provider=add_extra_args,
        args_defaults={"tokenizer_type": "GPT2BPETokenizer"},
    )
    validate_uniss_args(args)
    wrapped_pretrain, store = maybe_wrap_pretrain_after_parse(args)

    if megatron_gpt.has_nvidia_modelopt:
        megatron_gpt.maybe_enable_modelopt(args)
    if megatron_gpt.has_nvidia_modelopt and getattr(args, "modelopt_enabled", False):
        model_cfg = gpt_config_from_args(args, model_config_cls=megatron_gpt.ModelOptModelConfig)
    else:
        model_cfg = gpt_config_from_args(args)

    full_config = pretrain_cfg_container_from_args(args, model_cfg)
    wrapped_pretrain(
        full_config,
        train_valid_test_datasets_provider,
        ModelType.encoder_or_decoder,
        megatron_gpt.forward_step,
        store=store,
        get_embedding_ranks=megatron_gpt.get_embedding_ranks,
    )


if __name__ == "__main__":
    main()

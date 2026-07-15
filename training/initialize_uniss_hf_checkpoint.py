"""Initialize a UniSS-compatible Hugging Face checkpoint from Qwen2 weights.

The UniSS paper keeps the Qwen2.5 decoder architecture unchanged and expands
the vocabulary to hold speech, language, speed, and task/control tokens. This
script performs that HF-side initialization step before any Megatron conversion:

* load a local Qwen2/Qwen2.5 CausalLM checkpoint,
* resize input/output embeddings to the UniSS tokenizer size,
* preserve all original Qwen rows,
* initialize added rows from the model initializer,
* save the resized model plus UniSS tokenizer files.

It deliberately does not download assets. Run scripts/download_hf_assets.sh first
inside tmux if the source checkpoints are missing.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training import constants_uniss as c


DEFAULT_BASE_MODEL = Path("pretrained_models/Qwen2.5-1.5B-Instruct")
DEFAULT_UNISS_TOKENIZER = Path("pretrained_models/UniSS")
DEFAULT_OUTPUT = Path("checkpoints/qwen2_1p5b_uniss_vocab_hf")


@dataclass(frozen=True)
class ResizeSummary:
    base_vocab_size: int
    target_vocab_size: int
    added_tokens: int
    initializer_range: float
    input_embedding_shape: tuple[int, int]
    output_embedding_shape: tuple[int, int] | None
    tied_word_embeddings: bool


def torch_dtype_from_name(name: str):
    if name == "auto":
        return "auto"
    mapping = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }
    try:
        return mapping[name]
    except KeyError as exc:
        raise ValueError(f"Unsupported dtype {name!r}") from exc


def infer_tokenizer_vocab_size(tokenizer_path: Path) -> int:
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        local_files_only=True,
        trust_remote_code=False,
    )
    return len(tokenizer)


def resolve_target_vocab_size(
    *,
    explicit_vocab_size: int | None,
    tokenizer_path: Path,
    allow_nonstandard_vocab: bool = False,
) -> int:
    if explicit_vocab_size is not None:
        target_vocab_size = explicit_vocab_size
    elif tokenizer_path.exists():
        target_vocab_size = infer_tokenizer_vocab_size(tokenizer_path)
    else:
        target_vocab_size = c.VOCAB_SIZE

    if target_vocab_size <= 0:
        raise ValueError("target vocab size must be positive")
    if target_vocab_size != c.VOCAB_SIZE and not allow_nonstandard_vocab:
        raise ValueError(
            f"UniSS paper config requires vocab size {c.VOCAB_SIZE}, got {target_vocab_size}. "
            "Pass --allow-nonstandard-vocab only for tests or ablations."
        )
    return target_vocab_size


def _embedding_shape(module: torch.nn.Module | None) -> tuple[int, int] | None:
    if module is None or not hasattr(module, "weight"):
        return None
    weight = module.weight
    return int(weight.shape[0]), int(weight.shape[1])


def _weights_are_tied(model) -> bool:
    input_embeddings = model.get_input_embeddings()
    output_embeddings = model.get_output_embeddings()
    if input_embeddings is None or output_embeddings is None:
        return False
    return input_embeddings.weight.data_ptr() == output_embeddings.weight.data_ptr()


def resize_model_to_vocab(
    model,
    *,
    target_vocab_size: int,
    seed: int = 1234,
    initializer_range: float | None = None,
) -> ResizeSummary:
    input_embeddings = model.get_input_embeddings()
    if input_embeddings is None:
        raise ValueError("model has no input embeddings")

    base_vocab_size = int(input_embeddings.num_embeddings)
    if target_vocab_size < base_vocab_size:
        raise ValueError(
            f"target vocab size {target_vocab_size} is smaller than base vocab size {base_vocab_size}"
        )

    init_range = float(
        initializer_range
        if initializer_range is not None
        else getattr(model.config, "initializer_range", 0.02)
    )
    model.config.initializer_range = init_range

    if target_vocab_size != base_vocab_size:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            model.resize_token_embeddings(target_vocab_size, mean_resizing=False)
    else:
        model.config.vocab_size = target_vocab_size
        if hasattr(model, "tie_weights"):
            model.tie_weights()

    model.config.vocab_size = target_vocab_size
    if hasattr(model, "tie_weights"):
        model.tie_weights()

    input_shape = _embedding_shape(model.get_input_embeddings())
    output_shape = _embedding_shape(model.get_output_embeddings())
    if input_shape is None:
        raise ValueError("resized model has no input embeddings")

    return ResizeSummary(
        base_vocab_size=base_vocab_size,
        target_vocab_size=target_vocab_size,
        added_tokens=target_vocab_size - base_vocab_size,
        initializer_range=init_range,
        input_embedding_shape=input_shape,
        output_embedding_shape=output_shape,
        tied_word_embeddings=_weights_are_tied(model),
    )


def ensure_output_dir(output_dir: Path, *, overwrite: bool) -> None:
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"{output_dir} already exists and is not empty; pass --overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)


def copy_tokenizer(tokenizer_path: Path, output_dir: Path) -> None:
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        local_files_only=True,
        trust_remote_code=False,
    )
    tokenizer.save_pretrained(output_dir)


def dry_run_summary(args: argparse.Namespace, target_vocab_size: int) -> dict[str, object]:
    config = AutoConfig.from_pretrained(
        args.base_model,
        local_files_only=True,
        trust_remote_code=False,
    )
    return {
        "base_model": str(args.base_model),
        "uniss_tokenizer": str(args.uniss_tokenizer),
        "output": str(args.output),
        "base_architectures": getattr(config, "architectures", None),
        "base_model_type": getattr(config, "model_type", None),
        "base_vocab_size": int(getattr(config, "vocab_size")),
        "target_vocab_size": target_vocab_size,
        "added_tokens": target_vocab_size - int(getattr(config, "vocab_size")),
        "tie_word_embeddings": bool(getattr(config, "tie_word_embeddings", False)),
        "initializer_range": float(getattr(config, "initializer_range", 0.02)),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", type=Path, default=DEFAULT_BASE_MODEL)
    parser.add_argument("--uniss-tokenizer", type=Path, default=DEFAULT_UNISS_TOKENIZER)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--target-vocab-size", type=int, default=None)
    parser.add_argument("--allow-nonstandard-vocab", action="store_true")
    parser.add_argument("--dtype", choices=["auto", "float32", "bfloat16", "float16"], default="auto")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--initializer-range", type=float, default=None)
    parser.add_argument("--max-shard-size", default="5GB")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-tokenizer-copy", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    target_vocab_size = resolve_target_vocab_size(
        explicit_vocab_size=args.target_vocab_size,
        tokenizer_path=args.uniss_tokenizer,
        allow_nonstandard_vocab=args.allow_nonstandard_vocab,
    )

    if args.dry_run:
        print(json.dumps(dry_run_summary(args, target_vocab_size), indent=2, sort_keys=True))
        return

    ensure_output_dir(args.output, overwrite=args.overwrite)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        local_files_only=True,
        trust_remote_code=False,
        torch_dtype=torch_dtype_from_name(args.dtype),
    )
    summary = resize_model_to_vocab(
        model,
        target_vocab_size=target_vocab_size,
        seed=args.seed,
        initializer_range=args.initializer_range,
    )
    model.save_pretrained(args.output, safe_serialization=True, max_shard_size=args.max_shard_size)

    if not args.skip_tokenizer_copy:
        copy_tokenizer(args.uniss_tokenizer, args.output)

    summary_path = args.output / "uniss_init_summary.json"
    summary_path.write_text(json.dumps(asdict(summary), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(asdict(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

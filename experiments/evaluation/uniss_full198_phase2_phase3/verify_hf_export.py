"""Validate a frozen UniSS HF export without loading it onto a GPU."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from transformers import AutoConfig, AutoTokenizer


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--expected-vocab-size", type=int, required=True)
    args = parser.parse_args()

    config = AutoConfig.from_pretrained(args.model, local_files_only=True, trust_remote_code=False)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, trust_remote_code=False)
    if int(config.vocab_size) != args.expected_vocab_size:
        raise ValueError(f"config vocab_size={config.vocab_size}, expected {args.expected_vocab_size}")
    if len(tokenizer) != args.expected_vocab_size:
        raise ValueError(f"tokenizer size={len(tokenizer)}, expected {args.expected_vocab_size}")
    weight_files = sorted([*args.model.glob("*.safetensors"), *args.model.glob("*.bin")])
    if not weight_files:
        raise FileNotFoundError(f"No HF weight files found under {args.model}")

    manifest = {
        "model": str(args.model.resolve()),
        "source_checkpoint": str(args.source_checkpoint.resolve()),
        "vocab_size": int(config.vocab_size),
        "tokenizer_size": len(tokenizer),
        "config_sha256": sha256(args.model / "config.json"),
        "weight_files": [{"name": path.name, "size": path.stat().st_size} for path in weight_files],
    }
    (args.model / "export_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

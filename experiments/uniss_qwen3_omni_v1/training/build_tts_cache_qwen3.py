"""Re-tokenise the TTS cache for Qwen3-Omni.

The cache stores token ids, not text: build_tts_cache writes
prompt_values and reply_values straight from a tokenizer. Those ids were
cut with Qwen2.5-Omni's tokenizer, and Qwen3-Omni's is a different one --
151,676 entries against 151,665, and a different chat template around
them. Feeding the old ids to the new embedding table looks up different
words, produces a loss that falls perfectly well, and says nothing.

Only the two text arrays need rebuilding. code_values and global_values
come from the audio and have nothing to do with which Thinker reads them,
so this writes them unchanged -- which also keeps row order identical
between the two caches, so anything that indexed one indexes the other.

The row filter has to stay identical for that to hold, so read_shard is
imported rather than reimplemented and min/max codes default to what the
original build used.
"""

from __future__ import annotations

import argparse
import json
import time
from itertools import chain
from pathlib import Path

import numpy as np

from experiments.uniss_omni_s2st_v1.training.build_tts_cache import read_shard
from experiments.uniss_omni_s2st_v1.training.tts_data import tts_prompt


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="pretrained_models/Qwen3-Omni-30B-A3B-Instruct")
    ap.add_argument("--parquet-root", default="data/raw/UniST")
    ap.add_argument("--pattern", default="train-*.parquet")
    ap.add_argument("--shards", type=int, default=0, help="0 = every shard")
    # The original build's values. Changing either changes which rows
    # survive, and the two caches would stop lining up row for row.
    ap.add_argument("--min-codes", type=int, default=4)
    ap.add_argument("--max-codes", type=int, default=600)
    ap.add_argument("--chunk", type=int, default=20000)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    from transformers import Qwen3OmniMoeProcessor

    paths = sorted(Path(args.parquet_root).glob(args.pattern))
    if args.shards:
        paths = paths[: args.shards]
    if not paths:
        raise SystemExit(f"no parquets matched {args.pattern}")

    processor = Qwen3OmniMoeProcessor.from_pretrained(args.model)
    tokenizer = processor.tokenizer
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    prompt_parts: list[np.ndarray] = []
    reply_parts: list[np.ndarray] = []
    prompt_lengths: list[np.ndarray] = []
    reply_lengths: list[np.ndarray] = []
    code_parts: list[np.ndarray] = []
    code_lengths: list[np.ndarray] = []
    global_parts: list[np.ndarray] = []
    rows = 0
    started = time.perf_counter()
    pending: list[tuple] = []

    def flush() -> None:
        nonlocal rows
        if not pending:
            return
        templated = processor.apply_chat_template(
            [tts_prompt(text, lang) for text, lang, _, _ in pending],
            add_generation_prompt=True, tokenize=False,
        )
        prompts = tokenizer(list(templated), add_special_tokens=False)["input_ids"]
        replies = tokenizer(
            [text for text, _, _, _ in pending], add_special_tokens=False
        )["input_ids"]
        prompt_parts.append(np.fromiter(chain.from_iterable(prompts), dtype=np.int32))
        reply_parts.append(np.fromiter(chain.from_iterable(replies), dtype=np.int32))
        prompt_lengths.append(np.fromiter(map(len, prompts), dtype=np.int64))
        reply_lengths.append(np.fromiter(map(len, replies), dtype=np.int64))
        code_parts.append(np.concatenate([codes for _, _, codes, _ in pending]))
        code_lengths.append(
            np.fromiter((len(codes) for _, _, codes, _ in pending), dtype=np.int64))
        global_parts.append(np.stack([g for _, _, _, g in pending]))
        rows += len(pending)
        pending.clear()

    for number, path in enumerate(paths, start=1):
        for record in read_shard(path, min_codes=args.min_codes, max_codes=args.max_codes):
            pending.append(record)
            if len(pending) >= args.chunk:
                flush()
        flush()
        if number % 10 == 0 or number == len(paths):
            rate = rows / max(1e-9, time.perf_counter() - started)
            print(f"  {number}/{len(paths)} shards  {rows:,} rows  {rate:,.0f}/s", flush=True)
    flush()
    if not rows:
        raise SystemExit("no usable rows")

    def save(name: str, parts: list[np.ndarray], lengths: list[np.ndarray]) -> None:
        np.save(out / f"{name}_values.npy", np.concatenate(parts))
        sizes = np.concatenate(lengths)
        offsets = np.zeros(sizes.size + 1, dtype=np.int64)
        np.cumsum(sizes, out=offsets[1:])
        np.save(out / f"{name}_offsets.npy", offsets)

    save("prompt", prompt_parts, prompt_lengths)
    save("reply", reply_parts, reply_lengths)
    save("code", code_parts, code_lengths)
    np.save(out / "global_values.npy", np.concatenate(global_parts))
    meta = {
        "rows": rows, "shards": len(paths), "min_codes": args.min_codes,
        "max_codes": args.max_codes, "model": args.model,
        "built_seconds": round(time.perf_counter() - started, 1),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()

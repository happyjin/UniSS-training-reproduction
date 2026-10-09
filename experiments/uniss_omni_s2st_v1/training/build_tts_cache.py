"""Build a flat, memory-mappable cache of the TTS warmup corpus.

The warmup has 19.8M utterances available -- all 198 UniST parquets are
already on this box, and the TTS task needs no audio -- but reading them
the obvious way does not scale:

  * ``Table.to_pylist()`` costs 7.2 s per file against 0.1 s for the same
    columns read through Arrow's offsets, because it materialises every
    code as a Python int;
  * tokenising 19.8M prompts costs ~33 minutes, and *every rank* pays it
    separately at every run start.

So it is done once, here, and written as flat arrays each rank can
memory-map in seconds. Token ids are int32 and codes int16; both are
concatenated with an offsets array rather than stored per row.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from itertools import chain

from experiments.uniss_omni_s2st_v1.runtime.omni_processor import (
    load_audio_text_processor,
)
from experiments.uniss_omni_s2st_v1.training.tts_data import (
    COLUMNS,
    GLOBAL_TOKEN_COUNT,
    tts_prompt,
)


def read_shard(path: Path, *, min_codes: int, max_codes: int):
    """Columns out of one parquet, via Arrow's offsets rather than Python."""
    table = pq.read_table(path, columns=COLUMNS)
    codes = table["target_bicodec"].combine_chunks()
    code_values = codes.values.to_numpy(zero_copy_only=False).astype(np.int16)
    code_offsets = codes.offsets.to_numpy().astype(np.int64)

    globals_col = table["bicodec_global"].combine_chunks()
    global_values = globals_col.values.to_numpy(zero_copy_only=False)
    if global_values.size != table.num_rows * GLOBAL_TOKEN_COUNT:
        raise ValueError(
            f"{path}: expected {GLOBAL_TOKEN_COUNT} global tokens per row,"
            f" got {global_values.size / max(1, table.num_rows):.2f}"
        )
    global_values = global_values.astype(np.int16).reshape(-1, GLOBAL_TOKEN_COUNT)

    texts = table["translation"].to_pylist()
    langs = table["tgt_lang"].to_pylist()

    lengths = np.diff(code_offsets)
    keep = (lengths >= min_codes) & (lengths <= max_codes)
    for index in np.flatnonzero(keep):
        text = (texts[index] or "").strip()
        if not text:
            continue
        start, stop = code_offsets[index], code_offsets[index + 1]
        yield text, langs[index], code_values[start:stop], global_values[index]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="pretrained_models/Qwen2.5-Omni-3B")
    ap.add_argument("--parquet-root", default="data/raw/UniST")
    ap.add_argument("--pattern", default="train-*.parquet")
    ap.add_argument("--shards", type=int, default=0, help="0 = every shard")
    ap.add_argument("--min-codes", type=int, default=4)
    ap.add_argument("--max-codes", type=int, default=600)
    ap.add_argument("--chunk", type=int, default=20000)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    paths = sorted(Path(args.parquet_root).glob(args.pattern))
    if args.shards:
        paths = paths[: args.shards]
    if not paths:
        raise SystemExit(f"no parquets matched {args.pattern}")

    processor = load_audio_text_processor(args.model)
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
            add_generation_prompt=True,
            tokenize=False,
        )
        prompts = tokenizer(list(templated), add_special_tokens=False)["input_ids"]
        replies = tokenizer(
            [text for text, _, _, _ in pending], add_special_tokens=False
        )["input_ids"]
        # Flattened in one pass per chunk rather than row by row: at 17M
        # rows the per-row np.asarray dominates the build.
        prompt_parts.append(
            np.fromiter(chain.from_iterable(prompts), dtype=np.int32)
        )
        reply_parts.append(np.fromiter(chain.from_iterable(replies), dtype=np.int32))
        prompt_lengths.append(np.fromiter(map(len, prompts), dtype=np.int64))
        reply_lengths.append(np.fromiter(map(len, replies), dtype=np.int64))
        code_parts.append(np.concatenate([codes for _, _, codes, _ in pending]))
        code_lengths.append(
            np.fromiter((len(codes) for _, _, codes, _ in pending), dtype=np.int64)
        )
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
            print(
                f"  {number}/{len(paths)} shards  {rows:,} rows  {rate:,.0f}/s",
                flush=True,
            )
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
        "rows": rows,
        "shards": len(paths),
        "min_codes": args.min_codes,
        "max_codes": args.max_codes,
        "model": args.model,
        "built_seconds": round(time.perf_counter() - started, 1),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(json.dumps(meta, indent=1))
    print(f"-> {out}")


if __name__ == "__main__":
    main()

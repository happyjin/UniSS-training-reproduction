#!/usr/bin/env python3
"""
Filter the converted t2s parquet by codec length.

The Talker self-attention builds a [T, T] matrix over codec frames, so long
audio (many frames) blows up memory and, more importantly, causes severe DDP
rank load-imbalance (fast ranks idle while a slow rank processes a long clip).

Qwen3-TTS codec runs at 12 Hz, so:
    30 s  == 360 frames
Anything longer is dropped.

IMPORTANT — row-group sizing:
    The codec column (``target_codec_codes``) and ``audios`` are nested LIST
    columns.  pyarrow encodes list offsets with int32, so if a single parquet
    row-group is huge (e.g. ~800k rows) the cumulative offset overflows int32
    and HuggingFace ``datasets`` throws ``OSError: List index overflow`` while
    scanning the file.  We therefore:
      * read the source parquet in *small batches* (never materialising a giant
        record-batch), and
      * write the output with a *small* ``row_group_size`` so downstream
        readers always get small record-batches.

Input : sft_t2a_translate_converted/qwen3_tts_codec/text_to_speech_translation.parquet
Output: sft_t2a_translate_converted/qwen3_tts_codec/text_to_speech_translation.le30s.parquet
"""
import argparse
import os
import time

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

DEFAULT_IN = "/opt/dlami/nvme/zeyangsong/data/sft_t2a_translate_converted/qwen3_tts_codec/text_to_speech_translation.parquet"
CODEC_HZ = 12.0
CODEC_COL = "target_codec_codes"


def frame_count(codes) -> int:
    """target_codec_codes is an ndarray of shape [T, C] (or [C, T])."""
    if codes is None:
        return 0
    arr = np.asarray(codes)
    if arr.ndim != 2:
        return arr.shape[0] if arr.ndim >= 1 else 0
    # C is 16 (1 main + 15 residual); T is the larger axis.
    a, b = arr.shape
    if b == 16:
        return a
    if a == 16:
        return b
    return max(a, b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=DEFAULT_IN)
    ap.add_argument("--output", default=None)
    ap.add_argument("--max-seconds", type=float, default=30.0)
    ap.add_argument("--read-batch-size", type=int, default=20000,
                    help="rows per batch when scanning the source parquet")
    ap.add_argument("--row-group-size", type=int, default=2000,
                    help="rows per row-group in the output parquet (keep small "
                         "to avoid int32 list-offset overflow on read)")
    args = ap.parse_args()

    max_frames = int(round(args.max_seconds * CODEC_HZ))
    out = args.output or args.input.replace(
        ".parquet", f".le{int(args.max_seconds)}s.parquet")

    print(f"[filter] input          : {args.input}")
    print(f"[filter] output         : {out}")
    print(f"[filter] max_seconds    : {args.max_seconds} ({max_frames} frames @ {CODEC_HZ}Hz)")
    print(f"[filter] read_batch_size: {args.read_batch_size}")
    print(f"[filter] row_group_size : {args.row_group_size}")

    t0 = time.time()
    pf = pq.ParquetFile(args.input)
    schema = pf.schema_arrow
    n_in = pf.metadata.num_rows
    print(f"[filter] source rows    : {n_in:,} | row_groups={pf.metadata.num_row_groups}")

    writer = pq.ParquetWriter(out, schema)
    n_seen = 0
    n_keep = 0
    try:
        for batch in pf.iter_batches(batch_size=args.read_batch_size):
            tbl = pa.Table.from_batches([batch], schema=schema)
            # Layout is [T, 16]: outer list length == frame count T.
            # Vectorised in C++ — no per-row python parsing.
            outer_len = pc.list_value_length(tbl.column(CODEC_COL))
            mask_arr = pc.fill_null(pc.less_equal(outer_len, max_frames), False)
            n_batch = tbl.num_rows
            n_seen += n_batch
            n_kept_batch = int(pc.sum(pc.cast(mask_arr, pa.int64())).as_py() or 0)
            if n_kept_batch:
                kept = tbl.filter(mask_arr)
                writer.write_table(kept, row_group_size=args.row_group_size)
                n_keep += n_kept_batch
            if n_seen % 100000 < args.read_batch_size:
                elapsed = time.time() - t0
                print(f"[filter] scanned {n_seen:,}/{n_in:,} "
                      f"kept {n_keep:,} ({elapsed:.1f}s)", flush=True)
    finally:
        writer.close()

    n_drop = n_seen - n_keep
    print(f"[filter] keep  : {n_keep:,} ({100*n_keep/max(n_seen,1):.2f}%)")
    print(f"[filter] drop  : {n_drop:,} ({100*n_drop/max(n_seen,1):.2f}%)  (> {max_frames} frames)")
    print(f"[filter] size  : {os.path.getsize(out)/(1024**3):.2f} GiB")
    print(f"[filter] total : {time.time()-t0:.1f}s")

    # sanity check: reopen and confirm many small row-groups
    chk = pq.ParquetFile(out)
    print(f"[filter] output rows    : {chk.metadata.num_rows:,} | "
          f"row_groups={chk.metadata.num_row_groups} "
          f"(~{chk.metadata.num_rows // max(chk.metadata.num_row_groups,1)} rows/group)")


if __name__ == "__main__":
    main()

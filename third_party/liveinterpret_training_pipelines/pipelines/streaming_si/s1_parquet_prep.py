#!/usr/bin/env python3
"""S1（Emilia-FSQ）parquet 加 `lengths` 列，供 swift `--group_by_length` 按长度分桶
（ktrain_entry.sh 的 setup_s1 调用，每节点一次、在 torchrun 之前）。

原始数据目录不动：带 lengths 列的副本写到 <out_dir>（与原文件同名），约等于原数据体积。

lengths = talker 序列长度，与 si_encode.encode_s1 + talker 拼接口径一致：
    1（sos）+ 译文 token（assistant 文本 + eos，超过 max_length 时截到 max_length-1）
    + 1（task）+ 音色 prompt（当前数据没有 ref_speech_tokens，恒为 0）+ min(speech token, SI_MAX_SPEECH_LEN)
S1 的计算量由 talker 序列主导（p50 约 219，thinker 输入 p50 约 31），且两者高度相关，按它排序即可。
tokenizer 与训练相同（AutoTokenizer.from_pretrained(MODEL_PATH)）。

用法：
    python s1_parquet_prep.py <S1 parquet 目录> <输出目录> --tokenizer <MODEL_PATH>
"""
import argparse
import glob
import json
import os
import sys
from multiprocessing import Pool

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

LENGTH_COL = 'lengths'
_TOK = None


def _tokenizer(path):
    global _TOK
    if _TOK is None:
        from transformers import AutoTokenizer
        _TOK = AutoTokenizer.from_pretrained(path)
    return _TOK


def talker_lengths(table, tokenizer_path: str, max_speech_len: int, max_length: int):
    tok = _tokenizer(tokenizer_path)
    convs = table.column('conversations').to_pylist()
    texts, n_asst = [], []
    for c in convs:
        try:
            turns = json.loads(c) if isinstance(c, str) else (c or [])
        except (TypeError, ValueError):
            turns = []
        asst = [t for t in turns if isinstance(t, dict) and t.get('role') == 'assistant']
        n_asst.append(len(asst))
        texts.append(str(asst[0].get('content') or '') if asst else '')
    tgt = np.array([len(x) for x in tok(texts, add_special_tokens=False)['input_ids']], dtype=np.int64)
    if tok.eos_token_id is not None:
        tgt += 1
    tgt = np.minimum(tgt, max_length - 1)

    # answer_speech_tokens: list<list<int>>，按 encode_s1 取第 n_asst 个 turn（不足时取最后一个），全程向量化
    outer = table.column('answer_speech_tokens').combine_chunks()
    n_turns = pc.list_value_length(outer).fill_null(0).to_numpy(zero_copy_only=False).astype(np.int64)
    offsets = outer.offsets.to_numpy().astype(np.int64)[:-1]
    inner_len = pc.list_value_length(outer.values).fill_null(0).to_numpy(zero_copy_only=False).astype(np.int64)
    turn = np.minimum(np.maximum(np.array(n_asst, dtype=np.int64), 1), n_turns) - 1
    speech = np.zeros(len(n_turns), dtype=np.int64)
    has = n_turns > 0
    speech[has] = inner_len[offsets[has] + turn[has]]
    speech = np.minimum(speech, max_speech_len)
    return (1 + tgt + 1 + speech).astype(np.int32), int((speech == 0).sum())


def _process(job):
    src, dst, tokenizer_path, max_speech_len, max_length = job
    table = pq.read_table(src)
    if LENGTH_COL in table.column_names:
        table = table.drop_columns([LENGTH_COL])
    lens, no_speech = talker_lengths(table, tokenizer_path, max_speech_len, max_length)
    table = table.append_column(LENGTH_COL, pa.array(lens, type=pa.int32()))
    tmp = f'{dst}.tmp'
    pq.write_table(table, tmp)
    os.replace(tmp, dst)
    return len(lens), no_speech, int(lens.max()) if len(lens) else 0, np.bincount(np.minimum(lens, 1023), minlength=1024)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('src_dir')
    ap.add_argument('out_dir')
    ap.add_argument('--tokenizer', required=True, help='训练用的模型目录（MODEL_PATH）')
    ap.add_argument('--workers', type=int, default=min(32, os.cpu_count() or 1))
    args = ap.parse_args(argv)

    files = sorted(glob.glob(os.path.join(args.src_dir, '*.parquet')))
    if not files:
        print(f'没有 parquet: {args.src_dir}', file=sys.stderr)
        return 1
    schemas = {str(pq.read_schema(f).remove_metadata()) for f in files}
    if len(schemas) != 1:
        print(f'S1 parquet schema 跨文件不一致（{len(schemas)} 种），datasets 无法合并读取', file=sys.stderr)
        return 1
    missing = [c for c in ('conversations', 'answer_speech_tokens') if c not in pq.read_schema(files[0]).names]
    if missing:
        print(f'缺少计算长度需要的列: {missing}', file=sys.stderr)
        return 1

    env = os.environ
    max_speech_len = int(env.get('SI_MAX_SPEECH_LEN', 750))
    max_length = int(env.get('SI_MAX_LENGTH', env.get('MAX_LENGTH', 512)))
    os.makedirs(args.out_dir, exist_ok=True)
    jobs = [(f, os.path.join(args.out_dir, os.path.basename(f)), args.tokenizer, max_speech_len, max_length)
            for f in files]
    with Pool(max(1, min(args.workers, len(files)))) as pool:
        results = pool.map(_process, jobs, chunksize=1)

    rows = sum(r[0] for r in results)
    no_speech = sum(r[1] for r in results)
    hist = sum(r[3] for r in results)
    cdf = np.cumsum(hist) / max(hist.sum(), 1)
    pct = {p: int(np.searchsorted(cdf, p / 100)) for p in (50, 90, 99)}
    print(f'[streaming_si/ktrain] S1 lengths: {len(files)} 个 parquet / {rows} 行（无 speech token {no_speech} 行），'
          f'talker 序列长度 p50/p90/p99/max = {pct[50]}/{pct[90]}/{pct[99]}/{max(r[2] for r in results)}'
          f'（SI_MAX_SPEECH_LEN={max_speech_len}）')
    return 0


if __name__ == '__main__':
    sys.exit(main())

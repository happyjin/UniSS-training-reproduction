#!/usr/bin/env python3
"""P2 合并目录的 parquet 预处理（ktrain_entry.sh 的 setup_p2 调用，每节点一次、在 torchrun 之前）。

目录里起初全是指向各机器原始 parquet 的软链。本脚本只会把软链换成实体文件，原始数据不动。

1. 统一 schema：各机器的列可能不同（如 machine_1 多一列 dataset_name），datasets 要求列名一致。
   取所有文件的公共列，带多余列的文件重写为只含公共列；公共列类型跨文件不一致直接报错。
2. `--lengths --tokenizer <模型目录>`：给每行加元数据列（register_streaming_si.P2_META_COLS + lengths）：
     src_samples   源音频 16k 采样点数。读 FLAC 头部 STREAMINFO（42 字节，不解码），与
                   soundfile.info 逐条一致；非 FLAC 退回 soundfile.info。读不到为 0
     instr_tokens  st_instruction(tgt_lang) 的 token 数
     tgt_tokens    translation 的 token 数（不含 eos、不截断）
     lengths       编码后序列长度 = audio 帧 + instr + tgt + eos，供 swift --group_by_length 分桶
   全部是**与样本口径无关的原始量**：本脚本在 train_s*.sh 设置 SI_MAX_AUDIO_SEC 等变量之前运行，
   超长过滤由训练进程里的 StreamingSIPreprocessor 按当时的口径做（与编码时同一判定）。
   tokenizer 与训练相同（AutoTokenizer.from_pretrained(MODEL_PATH)），所以 lengths 与真实编码长度一致。

用法：
    python p2_parquet_prep.py <合并目录/p2_fsq_v3> <合并目录/p2_source_wav> [--lengths --tokenizer DIR]
"""
import argparse
import glob
import io
import os
import sys
from multiprocessing import Pool

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'custom'))
from streaming_si.data.si_encode import st_instruction, st_num_frames  # noqa: E402

META_COLS = ('src_samples', 'instr_tokens', 'tgt_tokens', 'lengths')
SR = 16000


def audio_samples_16k(fp, offset: int, length: int) -> int:
    """tar 内一条音频换算到 16k 的采样点数（与 read_p2_wav 的重采样后长度一致）。"""
    fp.seek(offset)
    head = fp.read(42)
    sr = total = 0
    if head[:4] == b'fLaC' and len(head) == 42:
        si = head[8:42]
        sr = (si[10] << 12) | (si[11] << 4) | (si[12] >> 4)
        total = ((si[13] & 0x0F) << 32) | int.from_bytes(si[14:18], 'big')
    if sr <= 0 or total <= 0:                  # 非 FLAC，或 STREAMINFO 未写总采样数
        import soundfile as sf
        fp.seek(offset)
        try:
            info = sf.info(io.BytesIO(fp.read(length)))
        except Exception:
            return 0
        sr, total = info.samplerate, info.frames
    if sr <= 0:
        return 0
    return total if sr == SR else -(-total * SR // sr)     # torchaudio resample 输出长度 = ceil


_TOK = None


def _tokenizer(path):
    global _TOK
    if _TOK is None:
        from transformers import AutoTokenizer
        _TOK = AutoTokenizer.from_pretrained(path)
    return _TOK


def compute_meta(table, wav_dir: str, tokenizer_path: str):
    """返回 {列名: list}，以及读不到音频的行数。"""
    tok = _tokenizer(tokenizer_path)
    refs = table.column('source_wav').to_pylist()
    texts = [t or '' for t in table.column('translation').to_pylist()]
    langs = table.column('tgt_lang').to_pylist()
    tgt = [len(x) for x in tok(texts, add_special_tokens=False)['input_ids']]
    instr_cache = {}
    instr = []
    for lang in langs:
        if lang not in instr_cache:
            instr_cache[lang] = len(tok.encode(st_instruction(lang), add_special_tokens=False))
        instr.append(instr_cache[lang])
    fps, samples = {}, []
    try:
        for ref in refs:
            n = 0
            if ref and '::' in ref:
                name, loc = ref.split('::', 1)
                try:
                    off, ln = (int(x) for x in loc.split(':'))
                    fp = fps.get(name)
                    if fp is None:
                        fp = fps[name] = open(os.path.join(wav_dir, name), 'rb')
                    n = audio_samples_16k(fp, off, ln)
                except (OSError, ValueError):
                    n = 0
            samples.append(n)
    finally:
        for fp in fps.values():
            fp.close()
    lengths = [st_num_frames(s) + i + t + 1 for s, i, t in zip(samples, instr, tgt)]
    meta = dict(src_samples=samples, instr_tokens=instr, tgt_tokens=tgt, lengths=lengths)
    return meta, sum(1 for s in samples if s <= 0)


def _process(job):
    path, common, wav_dir, tokenizer_path = job
    schema = pq.read_schema(path)
    names = [n for n in schema.names if n not in META_COLS]
    if tokenizer_path is None and names == common and not set(META_COLS) & set(schema.names):
        return path, False, None
    table = pq.read_table(path, columns=common)
    stats = None
    if tokenizer_path is not None:
        meta, no_audio = compute_meta(table, wav_dir, tokenizer_path)
        for col in META_COLS:
            table = table.append_column(col, pa.array(meta[col], type=pa.int32()))
        L = meta['lengths']
        stats = (len(L), no_audio, max(L) if L else 0, max(meta['src_samples'] or [0]) / SR)
    tmp = f'{path}.tmp'
    pq.write_table(table, tmp)
    os.replace(tmp, path)        # 替换软链本身，不写穿到原始文件
    return path, True, stats


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('parquet_dir')
    ap.add_argument('wav_dir')
    ap.add_argument('--lengths', action='store_true', help='追加元数据列（供超长过滤与 --group_by_length）')
    ap.add_argument('--tokenizer', help='--lengths 必需：训练用的模型目录（MODEL_PATH）')
    ap.add_argument('--workers', type=int, default=min(32, os.cpu_count() or 1))
    args = ap.parse_args(argv)
    if args.lengths and not args.tokenizer:
        ap.error('--lengths 需要 --tokenizer')

    files = sorted(glob.glob(os.path.join(args.parquet_dir, '*.parquet')))
    if not files:
        print(f'没有 parquet: {args.parquet_dir}', file=sys.stderr)
        return 1
    schemas = {f: pq.read_schema(f) for f in files}
    first = [n for n in schemas[files[0]].names if n not in META_COLS]
    common = [n for n in first if all(n in s.names for s in schemas.values())]
    for n in common:
        types = {str(s.field(n).type) for s in schemas.values()}
        if len(types) > 1:
            print(f'列 {n} 类型跨文件不一致: {sorted(types)}', file=sys.stderr)
            return 1
    if args.lengths:
        missing = [c for c in ('source_wav', 'translation', 'tgt_lang') if c not in common]
        if missing:
            print(f'计算元数据需要的列不在公共列中: {missing}', file=sys.stderr)
            return 1
    dropped = sorted({n for s in schemas.values() for n in s.names} - set(common) - set(META_COLS))

    tok = args.tokenizer if args.lengths else None
    jobs = [(f, common, args.wav_dir, tok) for f in files]
    with Pool(max(1, min(args.workers, len(files)))) as pool:
        results = pool.map(_process, jobs, chunksize=1)

    rewritten = sum(1 for _, w, _ in results if w)
    msg = f'[streaming_si/ktrain] P2 schema: {len(common)} 公共列，丢弃 {dropped or "无"}，重写 {rewritten}/{len(files)} 个 parquet'
    if args.lengths:
        rows = sum(s[0] for _, _, s in results)
        no_audio = sum(s[1] for _, _, s in results)
        longest = max(s[2] for _, _, s in results)
        max_sec = max(s[3] for _, _, s in results)
        msg += f'；元数据 {rows} 行（读不到音频 {no_audio} 行，最长音频 {max_sec:.1f}s，最长序列 {longest} token）'
    print(msg)
    return 0


if __name__ == '__main__':
    sys.exit(main())

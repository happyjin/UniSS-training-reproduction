#!/usr/bin/env python3
"""TTS_omni 数据准备驱动（美西集群）。

每个 worker 进程：
  1. boto3 列出三路源的 S3 对象（emilia-zh tar / yodas-jp tar / ko parquet）
  2. 固定种子 shuffle 后按 worker_id 取模分片（静态均衡，全局模数 NWT）
  3. 逐项：跳过已完成的（done marker 断点续跑）→ 预取下载到 cache →
     复用 make_tts_dualcodec 的解码/过滤/tokenizer → 写 manifest+codes
  4. 每完成一项 flush 所有 writer 并落 marker，删本地缓存

下载预取：处理第 i 项的同时在后台下载第 i+1 项（并发分片），消除下载空等——
8.6GB 的 yodas tar 单流下载要 ~6 分钟，这是 GPU 空闲的主要来源。

tokenizer 每个 worker 只初始化一次。shard 编号从 worker_id*100000 起，
多 worker / 多 Pod / 多语言共写同一 out 根不冲突。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import boto3
from boto3.s3.transfer import TransferConfig

BUCKET = 'stevehguo-ap-southeast-3'
REGION = 'ap-southeast-3'
BASE = 'speech_model_backup/stevehguo/data_fortrain/data_fortrain_1031version/TTS'
SOURCES = [
    # (lang, reader, s3 prefix, ext)
    ('ZH', 'tar_pair', f'{BASE}/zh/emilia/', '.tar'),
    ('JA', 'yodas_seg', f'{BASE}/jp/audio/', '.tar'),
    ('KO', 'parquet', f'{BASE}/ko/patterned_utterance_with_number/data/', '.parquet'),
]
# yodas_seg 直接产出 wav 片段（已在迭代器内解码切段）
PREDECODED = {'yodas_seg'}
# 并发分片下载：单流 ~25MB/s，8 并发可达 100MB/s+
XFER = TransferConfig(max_concurrency=8, multipart_chunksize=16 * 1024 ** 2,
                      multipart_threshold=64 * 1024 ** 2)


def _log(msg: str) -> None:
    print(f'[{time.strftime("%m-%d %H:%M:%S")}] {msg}', file=sys.stderr, flush=True)


def wait_gpu_free(need_gb: float = 35.0, timeout: float = 7200.0) -> bool:
    """共用 GPU 时：等到本卡还有 need_gb 可用才处理下一项。

    宿主任务（如 s5dc 合成阶段）会临时吃满 141GB，若不等待会直接 OOM 崩掉。
    """
    import torch
    t0 = time.time()
    last = 0.0
    while True:
        free, _total = torch.cuda.mem_get_info()
        if free >= need_gb * (1024 ** 3):
            return True
        el = time.time() - t0
        if el > timeout:
            _log(f'等 GPU 显存超时（{el / 60:.0f}分钟），本轮放弃等待')
            return False
        if el - last >= 300:
            last = el
            _log(f'等待 GPU 显存：可用 {free / 1e9:.1f}GB，需要 {need_gb}GB（已等 {el / 60:.0f}分钟）')
        time.sleep(10)


def list_items(s3) -> list[dict]:
    items = []
    for lang, reader, prefix, ext in SOURCES:
        n0 = len(items)
        for page in s3.get_paginator('list_objects_v2').paginate(
                Bucket=BUCKET, Prefix=prefix):
            for o in page.get('Contents', []):
                if o['Key'].endswith(ext):
                    items.append({'lang': lang, 'reader': reader,
                                  'key': o['Key'], 'size': o['Size']})
        _log(f'列出 {lang} {reader}: {len(items) - n0} 个对象')
    return items


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--worker-id', type=int, required=True)
    ap.add_argument('--nworkers', type=int, required=True)
    ap.add_argument('--root', default='/opt/dlami/nvme/stevehguo/tts_omni')
    ap.add_argument('--code', default='', help='make_tts_dualcodec 所在目录（默认 root/code）')
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--shared-gpu', action='store_true',
                    help='与宿主任务共用 GPU：处理前等待足够显存')
    ap.add_argument('--only-keys', default='',
                    help='只处理该文件中列出的 S3 key（每行一个），用于精确补跑缺失项')
    ap.add_argument('--residue-mod', type=int, default=0,
                    help='>0 时只处理 (原始序号 %% mod) 落在 [min,max] 的项（补跑漏掉的分区）')
    ap.add_argument('--residue-min', type=int, default=0)
    ap.add_argument('--residue-max', type=int, default=-1)
    ap.add_argument('--shard-offset-base', type=int, default=-1,
                    help='shard 起始编号基准；默认 worker_id*100000。补跑用大基准避免覆盖既有 shard')
    ap.add_argument('--skip-ids', default='',
                    help='已有 id 的 uint64 哈希排序数组（.npy，blake2b-8 字节）；命中的记录在解码前跳过')
    ap.add_argument('--done-dir', default='',
                    help='完成标记目录（默认 root/done）。重跑已标记完成的 item 时用独立目录')
    ap.add_argument('--limit', type=int, default=0)
    args = ap.parse_args()

    code = args.code or os.path.join(args.root, 'code')
    sys.path.insert(0, code)
    import make_tts_dualcodec as M  # noqa: E402

    import numpy as np
    skip = None
    if args.skip_ids:
        skip = np.load(args.skip_ids, mmap_mode='r')
        _log(f'载入跳过集 {skip.size:,} 个 id 哈希')

    def seen(uid: str) -> bool:
        if skip is None:
            return False
        h = np.uint64(int.from_bytes(hashlib.blake2b(uid.encode(), digest_size=8).digest(), 'little'))
        i = int(np.searchsorted(skip, h))
        return i < skip.size and skip[i] == h

    out = os.path.join(args.root, 'TTS_omni')
    done_dir = args.done_dir or os.path.join(args.root, 'done')
    cache = os.path.join(args.root, 'cache')
    os.makedirs(done_dir, exist_ok=True)
    os.makedirs(cache, exist_ok=True)

    s3 = boto3.client('s3', region_name=REGION)
    items = list_items(s3)
    if args.only_keys:
        want = {l.strip() for l in open(args.only_keys) if l.strip()}
        items = [it for it in items if it['key'] in want]
        _log(f'按清单筛选：{len(items)}/{len(want)} 项命中（清单 {len(want)} 项）')
    elif args.residue_mod > 0:
        # 补跑：只取原始序号落在指定残差区间的项（原始顺序 = S3 列表顺序，确定性）
        rmax = args.residue_max if args.residue_max >= 0 else args.residue_mod - 1
        keep = [(i, it) for i, it in enumerate(items)
                if args.residue_min <= (i % args.residue_mod) <= rmax]
        _log(f'残差筛选 mod={args.residue_mod} [{args.residue_min},{rmax}]：'
             f'{len(keep)}/{len(items)} 项')
        items = [it for _i, it in keep]
    random.Random(42).shuffle(items)
    mine = [it for i, it in enumerate(items) if i % args.nworkers == args.worker_id]
    if args.limit:
        mine = mine[:args.limit]

    def done_path(it) -> str:
        return os.path.join(done_dir,
                            hashlib.md5(it['key'].encode()).hexdigest()[:16] + '.done')

    todo = [it for it in mine if not os.path.exists(done_path(it))]
    _log(f'worker {args.worker_id}/{args.nworkers}: 共 {len(items)} 项，'
         f'我 {len(mine)} 项，待处理 {len(todo)}')
    if not todo:
        # 没活就不载模型：容器被重启时（如 OnFailure），避免在共用 GPU 上白占显存、挤爆宿主任务
        _log(f'WORKER {args.worker_id} DONE {json.dumps({"in": 0, "ok": 0, "skip": 0, "items": 0})} '
             f'elapsed=0.0min（无待处理项，未载入模型）')
        return

    _log(f'初始化 tokenizer (device={args.device}) ...')
    tok = M.build_tokenizer(os.path.join(args.root, 'weights', 'SimulS2ST-Omni'),
                            args.device)

    writers: dict[str, M.ShardWriter] = {}

    def get_writer(lang: str) -> M.ShardWriter:
        if lang not in writers:
            base = (args.shard_offset_base if args.shard_offset_base >= 0
                    else args.worker_id * 100000)
            writers[lang] = M.ShardWriter(out, lang, 5000, shard_offset=base)
        return writers[lang]

    def fetch(it) -> str:
        local = os.path.join(cache, os.path.basename(it['key']))
        if os.path.exists(local) and os.path.getsize(local) == it['size']:
            return local
        _log(f'下载 {it["key"]} ({it["size"] / 1e9:.2f}GB)')
        s3.download_file(BUCKET, it['key'], local, Config=XFER)
        return local

    pool = ThreadPoolExecutor(max_workers=1)
    stats = {'in': 0, 'ok': 0, 'skip': 0, 'items': 0}
    t0 = time.time()
    nxt = pool.submit(fetch, todo[0]) if todo else None
    for i, it in enumerate(todo):
        fut = nxt
        nxt = pool.submit(fetch, todo[i + 1]) if i + 1 < len(todo) else None
        try:
            local = fut.result()
        except Exception as e:  # noqa: BLE001
            _log(f'[skip] 下载失败 {it["key"]}: {e}')
            continue
        if args.shared_gpu and not wait_gpu_free():
            continue  # 显存被宿主任务占满，本轮不处理（下一轮重启后会重试）
        item_ok = False
        try:
            if it['reader'] == 'tar_pair':
                itr = M.iter_tar_pair_yield([local])
            elif it['reader'] == 'yodas_seg':
                itr = M.iter_yodas_segments([local])
            else:
                itr = M.iter_parquet([local])
            predecoded = it['reader'] in PREDECODED
            w = get_writer(it['lang'])
            for uid, text, audio, dnsmos in itr:
                stats['in'] += 1
                if seen(uid):
                    stats['dup'] = stats.get('dup', 0) + 1
                    continue
                wav = audio if predecoded else M.decode_audio(audio)
                if wav is None:
                    stats['skip'] += 1
                    continue
                dur = wav.size / M.SR
                if dur < M.MIN_DURATION or dur > M.MAX_DURATION:
                    stats['skip'] += 1
                    continue
                codes = M.torch_uint16_safe(tok(wav).squeeze(0))
                n = int(codes.size)
                if n < M.MIN_CODES or n > M.MAX_CODES:
                    stats['skip'] += 1
                    continue
                w.add(uid, text, codes, dur, dnsmos)
                stats['ok'] += 1
                if stats['ok'] % 500 == 0:
                    _log(f'ok={stats["ok"]} in={stats["in"]} skip={stats["skip"]} '
                         f'items={stats["items"]} elapsed={time.time() - t0:.0f}s')
            item_ok = True
        finally:
            for w in writers.values():
                w.flush()
            # 只在整项处理成功后才落完成标记：中途异常（如共用卡 OOM）不标记，
            # 下次重启会重做；已 flush 的部分记录由最终按 id 去重消化。
            if item_ok:
                with open(done_path(it), 'w') as f:
                    f.write(it['key'])
            else:
                _log(f'[未完成] {it["key"]} 处理中断，不写完成标记')
            try:
                os.remove(local)
            except OSError:
                pass
        stats['items'] += 1
    for w in writers.values():
        w.flush()
    _log(f'WORKER {args.worker_id} DONE {json.dumps(stats)} '
         f'elapsed={(time.time() - t0) / 60:.1f}min')


if __name__ == '__main__':
    main()

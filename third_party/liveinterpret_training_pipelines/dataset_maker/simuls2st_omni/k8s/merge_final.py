#!/usr/bin/env python3
"""合并 TTS_omni + TTS_omni_recovered + quarantine → TTS_omni_final（按 id 全局去重）。

两个来源的同一 id 内容一致（tokenizer 确定性），先到先得，优先 TTS_omni。
输出干净重分片：manifest/<LANG>/shard-XXXXX.jsonl（5000 条/个）+ codes/<LANG>/shard-XXXXX.npy。
"""
import hashlib
import io
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import boto3
import numpy as np

B = 'stevehguo-ap-southeast-3'
ROOT = 'speech_model_backup/stevehguo/data_fortrain/data_fortrain_1031version'
SOURCES = [f'{ROOT}/TTS_omni', f'{ROOT}/TTS_omni_recovered',
           f'{ROOT}/TTS_omni_quarantine/sweep2_run1']
DST = f'{ROOT}/TTS_omni_final'
OUT = '/tmp/out'
SHARD = 5000
N_CODES = 16384

s3 = boto3.client('s3', region_name='ap-southeast-3')
t_start = time.time()


def log(m):
    print(f'[{time.strftime("%H:%M:%S")}] {m}', flush=True)


def h64(s):
    return int.from_bytes(hashlib.blake2b(s.encode(), digest_size=8).digest(), 'little')


# ---- 1) 列出全部 manifest ----
mans = []
for pre in SOURCES:
    ks = []
    for page in s3.get_paginator('list_objects_v2').paginate(Bucket=B, Prefix=pre + '/manifest/'):
        ks += [o['Key'] for o in page.get('Contents', []) if o['Key'].endswith('.jsonl')]
    mans.append(ks)
    log(f'{pre.rsplit("/", 1)[-1]}: {len(ks)} manifests')
if '--scan-only' in sys.argv:
    sys.exit(0)


def read_lines(key):
    return s3.get_object(Bucket=B, Key=key)['Body'].read().decode('utf-8').splitlines()


# ---- 2) 建全局 id 哈希集 ----
ids = set()


def scan(keys):
    n = 0
    for k in keys:
        for l in read_lines(k):
            if l.strip():
                ids.add(h64(json.loads(l)['id']))
                n += 1
    return n


with ThreadPoolExecutor(24) as ex:
    totals = list(ex.map(scan, mans))
log(f'总记录（含重复）{sum(totals):,} | 唯一 id {len(ids):,} | 用时 {time.time() - t_start:.0f}s')

# ---- 3) 依序写出（先到先得）----
class Writer:
    def __init__(self, lang):
        self.lang = lang
        self.rows, self.parts, self.off, self.idx = [], [], 0, 0
        os.makedirs(f'{OUT}/manifest/{lang}', exist_ok=True)
        os.makedirs(f'{OUT}/codes/{lang}', exist_ok=True)

    def add(self, rec, codes):
        n = int(codes.size)
        rec = dict(rec)
        rec['codes'] = f'{self.lang}/shard-{self.idx:05d}.npy:{self.off}:{self.off + n}'
        self.rows.append(rec)
        self.parts.append(codes)
        self.off += n
        if len(self.rows) >= SHARD:
            self.flush()

    def flush(self):
        if not self.rows:
            return
        arr = np.concatenate(self.parts)
        assert arr.min() >= 0 and arr.max() < N_CODES
        np.save(f'{OUT}/codes/{self.lang}/shard-{self.idx:05d}.npy', arr.astype(np.uint16))
        with open(f'{OUT}/manifest/{self.lang}/shard-{self.idx:05d}.jsonl', 'w', encoding='utf-8') as f:
            for r in self.rows:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')
        self.idx += 1
        self.rows, self.parts, self.off = [], [], 0


cache = {}


def get_npy(prefix, rel):
    key = f'{prefix}/codes/{rel}'
    if key not in cache:
        if len(cache) > 40:
            for k in list(cache)[:20]:
                del cache[k]
        cache[key] = np.load(io.BytesIO(s3.get_object(Bucket=B, Key=key)['Body'].read()))
    return cache[key]


writers, stats = {}, {'kept': 0, 'dup': 0, 'bad': 0}
for pi, keys in enumerate(mans):
    for k in keys:
        lang = k.split('/manifest/')[1].split('/')[0]
        w = writers.setdefault(lang, Writer(lang))
        for l in read_lines(k):
            rec = json.loads(l)
            hh = h64(rec['id'])
            if hh not in ids:
                stats['dup'] += 1
                continue
            ids.discard(hh)
            rel, s, e = rec['codes'].rsplit(':', 2)
            try:
                arr = get_npy(SOURCES[pi], rel)
                seg = np.asarray(arr[int(s):int(e)])
                assert seg.size and abs(seg.size / 12.5 - rec['duration']) < 2
            except Exception as ex:  # noqa: BLE001
                stats['bad'] += 1
                log(f'[bad] {rec["id"]}: {ex}')
                continue
            w.add(rec, seg)
            stats['kept'] += 1
        if stats['kept'] and stats['kept'] % 2_000_000 == 0:
            log(f'进度 kept={stats["kept"]:,} dup={stats["dup"]:,} bad={stats["bad"]}')
for w in writers.values():
    w.flush()
log(f'写入完成 kept={stats["kept"]:,} dup={stats["dup"]:,} bad={stats["bad"]} '
    f'剩余未匹配 id={len(ids):,} | 用时 {time.time() - t_start:.0f}s')

# ---- 4) 上传 ----
ups = []
for dp, _, fs in os.walk(OUT):
    for f in fs:
        ups.append(os.path.join(dp, f))


def up(p):
    s3.upload_file(p, B, f'{DST}/{os.path.relpath(p, OUT)}')


with ThreadPoolExecutor(16) as ex:
    list(ex.map(up, ups))
summary = {**stats, 'files': len(ups), 'shards': {l: w.idx for l, w in writers.items()},
           'elapsed_min': round((time.time() - t_start) / 60, 1)}
s3.put_object(Bucket=B, Key=f'{DST}/_merge_summary.json',
              Body=json.dumps(summary, ensure_ascii=False, indent=1).encode())
log('SUMMARY ' + json.dumps(summary, ensure_ascii=False))

#!/usr/bin/env python3
"""统计 当前 TTS_omni（最新版本）∪ TTS_omni_recovered 的 id 覆盖，并逐 item 校验 JA/KO。

输出：
  - 各语言：当前唯一 id 数、恢复唯一 id 数、恢复带来的新增 id 数
  - JA/KO：被覆盖的 146 个 item 中各 item 的 id 覆盖率，对照组为最后一轮完成的 item
  - union id 哈希集合（uint64 排序数组）上传到 TTS_omni_meta/，供补跑时跳过已有 id
"""
from __future__ import annotations

import collections
import hashlib
import io
import json
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor

import boto3
import numpy as np
import pyarrow.parquet as pq

B = 'stevehguo-ap-southeast-3'
ROOT = 'speech_model_backup/stevehguo/data_fortrain/data_fortrain_1031version'
TTS = f'{ROOT}/TTS'
NVME = '/opt/dlami/nvme/stevehguo/tts_omni'
s3 = boto3.client('s3', region_name='ap-southeast-3')


def log(m):
    print(f'[{time.strftime("%H:%M:%S")}] {m}', flush=True)


def h64(s: str) -> int:
    return int.from_bytes(hashlib.blake2b(s.encode(), digest_size=8).digest(), 'little')


def list_keys(prefix, suffix):
    out = []
    for page in s3.get_paginator('list_objects_v2').paginate(Bucket=B, Prefix=prefix):
        out += [o['Key'] for o in page.get('Contents', []) if o['Key'].endswith(suffix)]
    return out


def read_ids(key):
    body = s3.get_object(Bucket=B, Key=key)['Body'].read().decode('utf-8')
    return [json.loads(l)['id'] for l in body.splitlines() if l.strip()]


def collect(prefix):
    per = collections.defaultdict(set)
    keys = list_keys(prefix + '/manifest/', '.jsonl')
    log(f'{prefix.rsplit("/", 1)[1]}: {len(keys)} 个 manifest')
    with ThreadPoolExecutor(48) as ex:
        for key, ids in zip(keys, ex.map(read_ids, keys)):
            per[key.split('/')[-2]].update(ids)
    return per


def main():
    cur = collect(f'{ROOT}/TTS_omni')
    rec = collect(f'{ROOT}/TTS_omni_recovered')
    report = {}
    allid = {}
    for lang in sorted(set(cur) | set(rec)):
        c, r = cur.get(lang, set()), rec.get(lang, set())
        new = r - c
        report[lang] = {'current_unique': len(c), 'recovered_unique': len(r), 'recovered_new': len(new),
                        'union': len(c | r)}
        allid[lang] = c | r
        log(f'{lang}: 当前 {len(c):,} | 恢复 {len(r):,} | 恢复新增 {len(new):,} | 合计 {len(c | r):,}')

    # ---- 被覆盖 item 与对照组 ----
    import glob
    old_keys = [l.strip() for l in open(f'{NVME}/old_done_keys.txt') if l.strip()]
    new_keys = []
    for f in glob.glob(f'{NVME}/done/*.done'):
        k = open(f).read().strip()
        if k not in old_keys:
            new_keys.append(k)

    def ko_cov(key):
        local = f'/tmp/{os.path.basename(key)}'
        s3.download_file(B, key, local)
        ids = pq.read_table(local, columns=['id'])['id'].to_pylist()
        os.remove(local)
        got = sum(1 for i in ids if str(i) in allid.get('KO', ()))
        return got / max(len(ids), 1)

    def ja_cov(key):
        n = os.path.basename(key).replace('.tar', '')
        body = s3.get_object(Bucket=B, Key=f'{TTS}/jp/metadata/{n}.parquet')['Body'].read()
        files = pq.read_table(io.BytesIO(body), columns=['id'])['id'].to_pylist()
        have = {i.rsplit('_s', 1)[0] for i in allid.get('JA', ())}
        return sum(1 for f in files if f in have) / max(len(files), 1)

    res = {}
    for lang, fn, pat in (('JA', ja_cov, '/jp/audio/'), ('KO', ko_cov, '/ko/')):
        old = [k for k in old_keys if pat in k]
        ctl = random.Random(0).sample([k for k in new_keys if pat in k],
                                      min(4, len([k for k in new_keys if pat in k])))
        with ThreadPoolExecutor(8) as ex:
            o = list(ex.map(fn, old)); c = list(ex.map(fn, ctl))
        res[lang] = {'old_items': len(old), 'old_cov': [round(x, 3) for x in o],
                     'control_cov': [round(x, 3) for x in c]}
        log(f'{lang} 被覆盖 item 覆盖率: {res[lang]["old_cov"]} | 对照组: {res[lang]["control_cov"]}')
    report['item_check'] = res

    hs = np.array(sorted(h64(i) for ids in allid.values() for i in ids), dtype=np.uint64)
    buf = io.BytesIO(); np.save(buf, hs)
    s3.put_object(Bucket=B, Key=f'{ROOT}/TTS_omni_meta/union_ids.npy', Body=buf.getvalue())
    s3.put_object(Bucket=B, Key=f'{ROOT}/TTS_omni_meta/audit_report.json',
                  Body=json.dumps(report, ensure_ascii=False, indent=1).encode())
    log(f'union id 哈希 {hs.size:,} 个已上传；报告: {json.dumps(report, ensure_ascii=False)[:1500]}')


if __name__ == '__main__':
    main()

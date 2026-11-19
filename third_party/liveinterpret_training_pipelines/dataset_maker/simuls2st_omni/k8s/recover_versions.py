#!/usr/bin/env python3
"""从 S3 版本历史恢复被重启覆盖的 TTS_omni shard。

背景：驱动旧版本在重启后从 worker_id*100000 重新编号，覆盖了自己之前写好的
shard，而对应 item 已有完成标记、不会重做。桶开了版本控制，被覆盖前的版本还在。

做法：
  1. 列出 TTS_omni/{manifest,codes}/ 下所有非最新版本
  2. 同一 shard 名的 manifest/codes 版本按时间排序后一一配对
  3. 校验：manifest 每条 codes 引用都指向本 shard；最大 end == npy 长度；值域 < 16384
  4. 通过的改名为 rv-<shard>-v<k>，codes 引用同步改写，写到 TTS_omni_recovered/
不做去重（最后全局按 id 统一去重）。产出 _recover_summary.json。
"""
from __future__ import annotations

import collections
import io
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import boto3
import numpy as np

B = 'stevehguo-ap-southeast-3'
ROOT = 'speech_model_backup/stevehguo/data_fortrain/data_fortrain_1031version'
SRC = f'{ROOT}/TTS_omni'
DST = f'{ROOT}/TTS_omni_recovered'
N_CODES = 16384

s3 = boto3.client('s3', region_name='ap-southeast-3')


def log(m: str) -> None:
    print(f'[{time.strftime("%H:%M:%S")}] {m}', flush=True)


def get(key: str, vid: str) -> bytes:
    return s3.get_object(Bucket=B, Key=key, VersionId=vid)['Body'].read()


def main() -> None:
    groups = collections.defaultdict(lambda: {'manifest': [], 'codes': []})
    for page in s3.get_paginator('list_object_versions').paginate(Bucket=B, Prefix=SRC + '/'):
        for v in page.get('Versions', []):
            if v['IsLatest']:
                continue
            rel = v['Key'][len(SRC) + 1:]
            parts = rel.split('/')
            if len(parts) != 3 or parts[0] not in ('manifest', 'codes'):
                continue
            sub, lang, fn = parts
            groups[(lang, fn.rsplit('.', 1)[0])][sub].append(v)
    jobs = []
    for (lang, shard), g in groups.items():
        ms = sorted(g['manifest'], key=lambda v: v['LastModified'])
        cs = sorted(g['codes'], key=lambda v: v['LastModified'])
        for k, (m, c) in enumerate(zip(ms, cs)):
            jobs.append((lang, shard, k, m, c))
    log(f'待恢复版本对 {len(jobs)} 个（shard 名 {len(groups)}）')

    stats = collections.Counter()
    bad = []

    def work(job):
        lang, shard, k, m, c = job
        try:
            rows = [json.loads(l) for l in get(m['Key'], m['VersionId']).decode('utf-8').splitlines() if l.strip()]
            arr = np.load(io.BytesIO(get(c['Key'], c['VersionId'])))
            want = f'{lang}/{shard}.npy'
            ends = []
            for r in rows:
                rel, s, e = r['codes'].rsplit(':', 2)
                if rel != want:
                    raise ValueError(f'引用不指向本 shard: {rel}')
                ends.append(int(e))
            if not rows or max(ends) != arr.size:
                raise ValueError(f'长度不匹配: max_end={max(ends) if ends else 0} npy={arr.size}')
            if arr.dtype != np.uint16 or int(arr.max()) >= N_CODES:
                raise ValueError(f'dtype/值域异常: {arr.dtype} max={arr.max()}')
            new = f'rv-{shard}-v{k}'
            out_rows = []
            for r in rows:
                _rel, s, e = r['codes'].rsplit(':', 2)
                r['codes'] = f'{lang}/{new}.npy:{s}:{e}'
                out_rows.append(json.dumps(r, ensure_ascii=False))
            buf = io.BytesIO(); np.save(buf, arr)
            s3.put_object(Bucket=B, Key=f'{DST}/codes/{lang}/{new}.npy', Body=buf.getvalue())
            s3.put_object(Bucket=B, Key=f'{DST}/manifest/{lang}/{new}.jsonl',
                          Body=('\n'.join(out_rows) + '\n').encode('utf-8'))
            return ('ok', lang, len(rows), None)
        except Exception as ex:  # noqa: BLE001
            return ('bad', lang, 0, f'{lang}/{shard} v{k}: {ex}')

    with ThreadPoolExecutor(max_workers=32) as ex:
        for i, (st, lang, n, err) in enumerate(ex.map(work, jobs), 1):
            stats[f'{st}_pairs'] += 1
            if st == 'ok':
                stats[f'records_{lang}'] += n
            else:
                bad.append(err)
            if i % 200 == 0:
                log(f'进度 {i}/{len(jobs)} {dict(stats)}')
    summary = {'stats': dict(stats), 'bad_samples': bad[:50], 'n_bad': len(bad)}
    s3.put_object(Bucket=B, Key=f'{DST}/_recover_summary.json',
                  Body=json.dumps(summary, ensure_ascii=False, indent=1).encode('utf-8'))
    log(f'完成 {json.dumps(summary, ensure_ascii=False)[:1500]}')
    sys.exit(0 if stats['ok_pairs'] else 1)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""节点收尾：1) 导出真实完成标记 2) 本地文件与 S3 比对、补传漏传 3) 清理缓存。"""
import glob
import os
import time

import boto3

R = '/opt/dlami/nvme/stevehguo/tts_omni'
B = 'stevehguo-ap-southeast-3'
P = 'speech_model_backup/stevehguo/data_fortrain/data_fortrain_1031version/TTS_omni'

s3 = boto3.client('s3', region_name='ap-southeast-3')


def log(m):
    print(f'[{time.strftime("%H:%M:%S")}] {m}', flush=True)


# ---- 1) 真实完成标记（排除 MOVED_TO_B 占位）----
real, moved = [], 0
for f in glob.glob(R + '/done/*.done'):
    c = open(f).read().strip()
    if c.startswith('MOVED_TO_B:'):
        moved += 1
    elif c:
        real.append(c)
print('===DONE_MARKERS_START===')
print('\n'.join(sorted(real)))
print('===DONE_MARKERS_END===')
log(f'真实标记 {len(real)}，占位标记 {moved}')

# ---- 2) 本地 vs S3，补传漏传 ----
local = {}
for dp, _, fs in os.walk(R + '/TTS_omni'):
    for f in fs:
        p = os.path.join(dp, f)
        local[os.path.relpath(p, R + '/TTS_omni')] = p
remote = set()
for page in s3.get_paginator('list_objects_v2').paginate(Bucket=B, Prefix=P + '/'):
    for o in page.get('Contents', []):
        remote.add(o['Key'][len(P) + 1:])
missing = sorted(r for r in local if r not in remote)
for r in missing:
    s3.upload_file(local[r], B, f'{P}/{r}')
log(f'本地文件 {len(local)}，S3 对象 {len(remote)}，补传 {len(missing)} 个')
for r in missing[:20]:
    print('  +', r)

# ---- 3) 清理缓存（所有写入者已退出）----
n = sz = 0
for f in glob.glob(R + '/cache/*'):
    try:
        sz += os.path.getsize(f)
        os.remove(f)
        n += 1
    except OSError:
        pass
log(f'缓存清理 {n} 个文件 {sz / 1e9:.1f}GB')
print('===NODE_FINALIZE_OK===')

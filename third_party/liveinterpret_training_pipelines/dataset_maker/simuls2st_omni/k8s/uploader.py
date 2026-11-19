#!/usr/bin/env python3
"""增量上传 TTS_omni 产物到雅加达桶。

文件级幂等：marker 每行 "mtime 相对路径"。本地文件 mtime 变化（worker 重启后
shard 被重写）会触发重新上传，保证 S3 与本地最终一致。
"""
from __future__ import annotations

import argparse
import fcntl
import os
import sys
import time

import boto3

BUCKET = 'stevehguo-ap-southeast-3'
DST = 'speech_model_backup/stevehguo/data_fortrain/data_fortrain_1031version/TTS_omni'


def _parse_line(line: str):
    """'mtime rel' -> (mtime, rel)；旧格式（无 mtime 前缀）-> (None, line)。"""
    parts = line.split(' ', 1)
    try:
        return float(parts[0]), parts[1].strip()
    except (ValueError, IndexError):
        return None, line.strip()


def scan_upload(root: str, s3) -> int:
    src = os.path.join(root, 'TTS_omni')
    marker = os.path.join(root, '.uploaded.txt')
    lockf = open(os.path.join(root, '.upload.lock'), 'w')
    fcntl.flock(lockf, fcntl.LOCK_EX)
    try:
        recorded: dict[str, float | None] = {}
        if os.path.exists(marker):
            for line in open(marker):
                if not line.strip():
                    continue
                mt, rel = _parse_line(line)
                recorded[rel] = mt
        out_lines: list[str] = []
        n = 0
        now = time.time()
        for dirpath, _, files in os.walk(src):
            for fn in files:
                p = os.path.join(dirpath, fn)
                rel = os.path.relpath(p, src)
                mtime = os.path.getmtime(p)
                if now - mtime < 60:
                    # 刚写的文件可能还没落定，本轮跳过（下轮 mtime 已记录后正常判断）
                    if rel in recorded:
                        out_lines.append(f'{recorded[rel]} {rel}' if recorded[rel] is not None else rel)
                    continue
                old = recorded.get(rel, None)
                if rel in recorded and old is not None and old == mtime:
                    out_lines.append(f'{old} {rel}')
                    continue
                s3.upload_file(p, BUCKET, f'{DST}/{rel}')
                out_lines.append(f'{mtime} {rel}')
                n += 1
        with open(marker, 'w') as f:
            f.write('\n'.join(out_lines) + ('\n' if out_lines else ''))
        return n
    finally:
        lockf.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='/opt/dlami/nvme/stevehguo/tts_omni')
    ap.add_argument('--interval', type=int, default=600)
    ap.add_argument('--once', action='store_true')
    args = ap.parse_args()
    s3 = boto3.client('s3', region_name='ap-southeast-3')
    while True:
        try:
            n = scan_upload(args.root, s3)
            print(f'[{time.strftime("%m-%d %H:%M:%S")}] 上传 {n} 个文件', flush=True)
        except Exception as e:
            print(f'[上传失败] {e}', file=sys.stderr, flush=True)
        if args.once:
            break
        time.sleep(args.interval)


if __name__ == '__main__':
    main()

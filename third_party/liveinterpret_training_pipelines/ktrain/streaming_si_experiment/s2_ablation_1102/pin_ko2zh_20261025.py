#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把 ko2zh 在新 S2 训练时（10-25 生成、10-28 11:11 CST 起被重写）的原始版本固定到独立路径。

    .../Speech2Speech/ko2zh/{p2_fsq_v3,p2_source_wav}/  （S3 历史版本，< 10-28 03:00Z 的最后一个版本）
 -> .../Speech2Speech/ko2zh_20261025/{p2_fsq_v3,p2_source_wav}/

同桶服务端复制（boto3 managed copy，>5GB 自动分片），不经本机流量。幂等：目标已存在且大小一致则跳过。
完成后打印行数校验（parquet footer 合计应为 10,641,055）。
"""
import concurrent.futures as cf
import sys

import boto3
from boto3.s3.transfer import TransferConfig
from botocore.config import Config

B = "stevehguo-ap-southeast-3"
SRC = "speech_model_backup/stevehguo/data_fortrain/data_fortrain_1031version/Speech2Speech/ko2zh/"
DST = "speech_model_backup/stevehguo/data_fortrain/data_fortrain_1031version/Speech2Speech/ko2zh_20261025/"
CUT = "2026-10-28T03:00:00"
EXPECT_ROWS = 10_641_055

s3 = boto3.client("s3", region_name="ap-southeast-3", config=Config(max_pool_connections=64))
tc = TransferConfig(multipart_threshold=1 << 30, multipart_chunksize=512 << 20, max_concurrency=16)

pick = {}
for pg in s3.get_paginator("list_object_versions").paginate(Bucket=B, Prefix=SRC):
    for v in pg.get("Versions", []):
        if v["LastModified"].strftime("%Y-%m-%dT%H:%M:%S") >= CUT:
            continue
        cur = pick.get(v["Key"])
        if cur is None or v["LastModified"] > cur["LastModified"]:
            pick[v["Key"]] = v
print(f"旧版本对象 {len(pick)} 个，{sum(v['Size'] for v in pick.values()) / 1e9:.1f} GB", flush=True)


def copy(v):
    dk = DST + v["Key"][len(SRC):]
    try:
        h = s3.head_object(Bucket=B, Key=dk)
        if h["ContentLength"] == v["Size"]:
            return dk, "skip"
    except s3.exceptions.ClientError:
        pass
    s3.copy({"Bucket": B, "Key": v["Key"], "VersionId": v["VersionId"]}, B, dk, Config=tc)
    return dk, "copied"


done = 0
with cf.ThreadPoolExecutor(12) as ex:
    for dk, st in ex.map(copy, sorted(pick.values(), key=lambda v: -v["Size"])):
        done += 1
        print(f"[{done}/{len(pick)}] {st} {dk.rsplit('/', 2)[-2]}/{dk.rsplit('/', 1)[-1]}", flush=True)

import pyarrow.parquet as pq  # noqa: E402
import s3fs  # noqa: E402

fs = s3fs.S3FileSystem(client_kwargs={"region_name": "ap-southeast-3"})
files = fs.glob(f"{B}/{DST}p2_fsq_v3/*.parquet")
rows = sum(pq.ParquetFile(fs.open(f)).metadata.num_rows for f in files)
tars = fs.glob(f"{B}/{DST}p2_source_wav/*.tar")
print(f"校验：parquet {len(files)} 个 / {rows:,} 行（期望 {EXPECT_ROWS:,}），tar {len(tars)} 个", flush=True)
sys.exit(0 if rows == EXPECT_ROWS else 1)

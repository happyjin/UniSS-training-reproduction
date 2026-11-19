#!/usr/bin/env python3
"""核对节点上 G0 清单（新 S2 实际训练的 15 个目录）的行数：应为 87,895,588。"""
import glob

import pyarrow.parquet as pq

R = "/nvme/speech_model_backup/stevehguo/data_fortrain"
S = R + "/data_fortrain_1031version/Speech2Speech/"
dirs = [S + d for d in ("en2zh zh2en en2ja ja2en en2ko ko2en ja2zh zh2ja ko2zh_20261025 zh2ko "
                        "negative_noise negative_noise_dns5").split()]
dirs += [f"{R}/open_data_0803_fortrain/machine_{m}_shard" for m in range(3)]
tot = 0
for d in dirs:
    f = glob.glob(d + "/p2_fsq_v3/*.parquet")
    t = glob.glob(d + "/p2_source_wav/*.tar")
    n = sum(pq.ParquetFile(x).metadata.num_rows for x in f)
    tot += n
    print(f"  {d.rsplit('/', 1)[1]:22s} pq {len(f):3d} tar {len(t):4d} rows {n:>11,}")
print(f"G0 合计 {tot:,} 行，期望 87,895,588 ->", "OK" if tot == 87895588 else "MISMATCH")

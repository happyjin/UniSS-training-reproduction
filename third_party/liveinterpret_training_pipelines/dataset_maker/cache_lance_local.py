#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把 processed.lance 镜像到本地 / NVMe，供训练只读本地、避免 S3 随机点查。

红线（不改训练语义）
--------------------
* **默认 mode=sync**：``aws s3 sync`` 整库镜像。对象字节一致 → ``_rowid`` /
  codes / audio 与远端完全一致。已有 JSONL 里的 ``lance://<_rowid>/...``
  **无需重写**，只改 ``LANCE_TRAIN_SOURCE_URI`` 指向本地路径即可。
* **不做 plan_shuffle**：打乱清单顺序会改变 index→样本映射，同 seed 下
  loss 曲线会漂；本脚本刻意不提供该选项。
* **不做「过滤后重写 rowid」的默认物化**：会破坏已有 manifest 的 rowid
  契约。若只要子集，请继续用原 S3/本地全库 + ``TaskSpec.filter`` /
  ``make_from_lance``（仍按稳定 ``_rowid`` 引用）。

用法::

  # 1) 镜像到 NVMe（推荐，训练机上跑）
  python dataset_maker/cache_lance_local.py sync \\
      --src s3://BUCKET/data/audio/lance/test_full/processed.lance \\
      --dst /nvme/data/audio/lance/test_full/processed.lance

  # 2) 训练只读本地（语义与打 S3 相同，只是 IO 变本地）
  export LANCE_TRAIN_SOURCE_URI=/nvme/data/audio/lance/test_full/processed.lance
  export LANCE_TRAIN_SOURCE_VERSION=4   # 与远端 pin 一致
  export AWS_REGION=ap-southeast-3      # 若仍需回源偶发读，可保留
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from typing import NoReturn, Optional


def _die(msg: str) -> NoReturn:
    print(f"[FAIL] {msg}", file=sys.stderr, flush=True)
    sys.exit(1)


def _ok(msg: str) -> None:
    print(f"[ OK ] {msg}", flush=True)


def _info(msg: str) -> None:
    print(f"[INFO] {msg}", flush=True)


def _normalize_s3_prefix(uri: str) -> str:
    """Lance dataset URI -> s3 prefix for sync (no trailing slash required)."""
    u = uri.rstrip("/")
    if not u.startswith("s3://"):
        _die(f"--src 必须是 s3://.../processed.lance，得到: {uri!r}")
    return u


def cmd_sync(src: str, dst: str, *, region: Optional[str], dry_run: bool) -> int:
    src = _normalize_s3_prefix(src)
    dst = os.path.abspath(dst)
    os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)

    env = os.environ.copy()
    if region:
        env["AWS_REGION"] = region
        env["AWS_DEFAULT_REGION"] = region

    # 去掉可能干扰的本机代理（训练机上偶发）
    for k in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy",
              "ALL_PROXY", "all_proxy"):
        env.pop(k, None)

    cmd = [
        "aws", "s3", "sync", src, dst,
        "--only-show-errors",
    ]
    if region:
        cmd.extend(["--region", region])
    if dry_run:
        cmd.append("--dryrun")

    _info(f"镜像 Lance 数据集（字节级一致，保留 stable _rowid）:")
    _info(f"  src = {src}")
    _info(f"  dst = {dst}")
    _info(f"  cmd = {' '.join(cmd)}")
    _info("说明: 这是文件系统镜像，不是 rewrite；训练语义与读 S3 完全一致。")

    proc = subprocess.run(cmd, env=env)
    if proc.returncode != 0:
        _die(f"aws s3 sync 失败，exit={proc.returncode}")

    # 轻量校验：本地能否打开 + 行数
    if not dry_run:
        try:
            import lance
            ds = lance.dataset(dst)
            _ok(f"本地打开成功 version={ds.version} rows={ds.count_rows():,}")
            _ok(f"请 export LANCE_TRAIN_SOURCE_URI={dst}")
            _ok(f"请 export LANCE_TRAIN_SOURCE_VERSION={ds.version}")
        except Exception as exc:
            _die(f"sync 完成但本地 lance 打开失败: {exc}")
    else:
        _ok("dry-run 结束（未实际下载）")
    return 0


def cmd_verify(uri: str, *, region: Optional[str], expect_rows: Optional[int]) -> int:
    try:
        import lance
    except Exception as exc:
        _die(f"需要 lance: {exc}")
    opts = {"region": region} if (region and uri.startswith("s3://")) else None
    ds = lance.dataset(uri, storage_options=opts) if opts else lance.dataset(uri)
    n = ds.count_rows()
    _ok(f"uri={uri} version={ds.version} rows={n:,}")
    if expect_rows is not None and n != expect_rows:
        _die(f"行数不一致: got={n} expect={expect_rows}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description="镜像 Lance 到本地/NVMe（不改训练语义；不做 plan_shuffle）")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_sync = sub.add_parser("sync", help="aws s3 sync 整库镜像（推荐）")
    p_sync.add_argument("--src", required=True, help="s3://.../processed.lance")
    p_sync.add_argument("--dst", required=True, help="本地/NVMe 目标目录")
    p_sync.add_argument("--region", default=os.environ.get("AWS_REGION")
                        or os.environ.get("AWS_DEFAULT_REGION") or "ap-southeast-3")
    p_sync.add_argument("--dry-run", action="store_true")

    p_ver = sub.add_parser("verify", help="打开 URI 打印 version/rows")
    p_ver.add_argument("--uri", required=True)
    p_ver.add_argument("--region", default=os.environ.get("AWS_REGION")
                       or os.environ.get("AWS_DEFAULT_REGION") or "ap-southeast-3")
    p_ver.add_argument("--expect-rows", type=int, default=None)

    args = ap.parse_args()
    if args.cmd == "sync":
        return cmd_sync(args.src, args.dst, region=args.region, dry_run=args.dry_run)
    if args.cmd == "verify":
        return cmd_verify(args.uri, region=args.region, expect_rows=args.expect_rows)
    _die(f"unknown cmd {args.cmd}")


if __name__ == "__main__":
    sys.exit(main())

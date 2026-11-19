#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 0 — verify_full_lance_training.py

一票否决验证脚本：在改写任何训练侧代码之前，先确认 ``processed.lance`` 能作为
训练数据源。全部检查通过才会让你继续 Step 1~3。

设计背景（见 lance对接训练架构设计.md §5）：
  本期换源只换"音频字节 / target codes 从哪来"，训练逻辑（loss / sampler /
  长度计算）保持不变，且 ms-swift 的 shuffle / sample / group_by_length / split /
  DP 分片 / 断点续训原样保留。所有"换源后训练结果不变"的断言必须由对拍支撑，
  本脚本负责落地前最硬的几个可行性闸门。

前置（部署环境）：
  # 训练镜像需要能 import curion.storage（数据源侧仅被读，不改动）
  pip install "curion-storage[lance]" pyarrow lance
  export LANCE_TRAIN_SOURCE_URI="s3://.../processed.lance"   # 或本地路径
  export LANCE_TRAIN_SOURCE_VERSION=42                       # 可选，默认读最新

用法：
  python verify_full_lance_training.py \\
      --lance-uri "$LANCE_TRAIN_SOURCE_URI" \\
      --task-filter "direction='zh_en' AND target_audio_utmos_score>=3.0" \\
      --sample-n 8

可选字节比对（需原 wds tar，用于"逐字节一致"待验证项，缺省跳过）：
  --reference-tar-dir /path/to/wds_shards   # 含与原 lance source_audio_shard_uri 同名的 .tar
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Optional, Sequence, NoReturn

# 业务列：manifest 需要的字段 + 字节/版本比对所需的列。
# 列名以 lance_builder._DEFAULT_CARRY_COLS / _DEFAULT_CODEC_COLS 为准。
BUSINESS_COLUMNS = [
    "target_text",
    "src_lang",
    "tgt_lang",
    "duration_ms",
    "source_audio_bytes",
    "source_audio_shard_uri",
    "source_audio_byte_offset",
    "source_audio_byte_length",
    "codes",
    "codec_tokenizer_id",
    "codec_tokenizer_revision",
    "codes_layout",
]

# source 音频内联字节的列（必须存在；指针回退仅作字节比对参考）。
SOURCE_INLINE_COL = "source_audio_bytes"
CODES_COL = "codes"

# Precomputed Whisper mel (Scheme A / build_lance_input_features_ray.py).
MEL_COLUMNS = [
    "source_input_features",
    "source_feature_attention_mask",
    "source_input_features_frames",
]
MEL_FEATURE_SIZE = 128


def _die(msg: str) -> NoReturn:
    print(f"[FAIL] {msg}", file=sys.stderr, flush=True)
    sys.exit(1)


def _section(title: str) -> None:
    print("\n" + "=" * 64, flush=True)
    print(title, flush=True)
    print("=" * 64, flush=True)


def _ok(msg: str) -> None:
    print(f"[ OK ] {msg}", flush=True)


def _warn(msg: str) -> None:
    print(f"[WARN] {msg}", file=sys.stderr, flush=True)


def _load_reader(uri: str, version: Optional[int]):
    try:
        from curion.storage.lance_reader import LanceRowReader
    except Exception as exc:  # pragma: no cover - 部署前置
        _die(
            "无法 import curion.storage.lance_reader。\n"
            "  训练镜像需安装 curion-storage（含 lance extra）："
            "pip install \"curion-storage[lance]\"。\n"
            f"  原始错误：{exc}"
        )
    try:
        return LanceRowReader(uri, version=version)
    except Exception as exc:
        _die(f"LanceRowReader 打开失败 uri={uri!r} version={version!r}：{exc}")


def _resolve_task_rowids(reader, task_filter: Optional[str], task_name: str, version: int):
    from curion.storage.lance_task import TaskSpec, resolve_task

    spec = TaskSpec(
        name=task_name,
        lance_version=version,
        filter=task_filter,  # situation 1：谓词可复现
    )
    return resolve_task(reader, spec).rowids


def _check_connectivity(reader, *, require_source_audio: bool = True) -> None:
    _section("检查 1/5 — 连通性与列完整性")
    required = list(BUSINESS_COLUMNS)
    if not require_source_audio:
        required = [
            c for c in required
            if c not in {
                SOURCE_INLINE_COL,
                "source_audio_shard_uri",
                "source_audio_byte_offset",
                "source_audio_byte_length",
            }
        ]
    missing = [c for c in required if not reader.has_column(c)]
    if missing:
        _die(f"数据集缺少必需列：{missing}。请核对 lance_builder 的 carry/codec 列配置。")
    _ok(f"reader 打开成功，version={reader.version}")
    _ok(f"全部 {len(required)} 个业务列均存在：{required}")


def _check_precomputed_mel(reader, sample_rowids: Sequence[int]) -> None:
    _section("检查 mel — source_input_features 布局（预计算 Whisper mel）")
    missing = [c for c in MEL_COLUMNS if not reader.has_column(c)]
    if missing:
        _die(f"mel 表缺少列：{missing}。需要 build_lance_input_features_ray 产物。")
    tbl = reader.take_by_stable_rowid(list(sample_rowids), columns=MEL_COLUMNS)
    rows = tbl.to_pylist()
    if not rows:
        _die("采样行取回为空，无法核查 mel 布局。")
    for r in rows:
        rid = r.get("_rowid")
        flat = r.get("source_input_features")
        frames = r.get("source_input_features_frames")
        mask = r.get("source_feature_attention_mask")
        if flat is None or frames is None:
            _die(f"row {rid} mel 列为空（flat/frames）。")
        t = int(frames)
        n = len(flat) if hasattr(flat, "__len__") else -1
        if n != MEL_FEATURE_SIZE * t:
            _die(
                f"row {rid} mel flat 长度={n}，期望 feature_size*frames="
                f"{MEL_FEATURE_SIZE}*{t}={MEL_FEATURE_SIZE * t}（mel-first）。"
            )
        if mask is not None and hasattr(mask, "__len__") and len(mask) not in (t, n):
            _warn(f"row {rid} feature_attention_mask len={len(mask)} frames={t}")
    _ok(
        f"mel 列齐全；采样 {len(rows)} 行均为 mel-first "
        f"{MEL_FEATURE_SIZE}×T（frames 对齐）。"
    )


def _check_codes_layout(
    reader, sample_rowids: Sequence[int], expected_num_residual_groups: int
) -> dict:
    _section("检查 2/5 — codes 布局与 codec 版本（最大不确定项）")
    tbl = reader.take_by_stable_rowid(
        list(sample_rowids),
        columns=[CODES_COL, "codec_tokenizer_id", "codec_tokenizer_revision", "codes_layout"],
    )
    rows = tbl.to_pylist()
    if not rows:
        _die("采样行取回为空，无法核查 codes 布局。")

    report: dict[str, Any] = {"samples": []}
    for r in rows:
        rid = r.get("_rowid")
        codes = r.get(CODES_COL)
        info: dict[str, Any] = {
            "rowid": rid,
            "codec_tokenizer_id": r.get("codec_tokenizer_id"),
            "codec_tokenizer_revision": r.get("codec_tokenizer_revision"),
            "codes_layout": r.get("codes_layout"),
        }
        if codes is None:
            _die(f"row {rid} 的 codes 列为空（target 无法用预计算 codes 跳过 GPU 重编码）。")
        # 维度判定：期望 2-D [T, C]
        if isinstance(codes, list) and codes and isinstance(codes[0], list):
            shape = [len(codes), len(codes[0])]
        elif isinstance(codes, list):
            shape = [len(codes)]
        else:
            shape = [getattr(codes, "shape", "unknown")]
        info["shape"] = shape
        report["samples"].append(info)

        if len(shape) != 2:
            _die(
                f"row {rid} 的 codes 不是 2-D [T, C]（实际 shape={shape}）。"
                "template._build_precomputed_codec_fields 要求 2-D，1-D 会直接报错。"
            )
        n_c = shape[1]
        if n_c != expected_num_residual_groups + 1:
            _die(
                f"row {rid} 的 codes 第二维 C={n_c}，但训练侧 num_residual_groups+1="
                f"{expected_num_residual_groups + 1}。维度不匹配会在训练时 ValueError。"
            )
    _ok(f"codes 均为 2-D [T, C]，C={expected_num_residual_groups + 1}（期望）。样本 shape："
        f"{[s['shape'] for s in report['samples']]}")
    return report


def _check_codec_version_against_training(codes_report: dict) -> None:
    _section("检查 3/5 — lance codec 版本 vs 训练侧 get_codec_tokenizer()")
    try:
        # 训练侧 codec 工厂；能 import 则自动比对，不能则降级为人工核对。
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
        from custom.codec_tokenizer import get_codec_tokenizer  # type: ignore

        tok = get_codec_tokenizer()
        train_id = getattr(tok, "codec_tokenizer_id", None) or getattr(tok, "tokenizer_id", None)
        train_rev = getattr(tok, "codec_tokenizer_revision", None) or getattr(tok, "revision", None)
    except Exception as exc:
        _warn(
            "无法 import 训练侧 get_codec_tokenizer()，跳过自动版本比对。"
            "请人工核对下列 lance codec 版本与训练配置一致："
            f"{[ (s['codec_tokenizer_id'], s['codec_tokenizer_revision']) for s in codes_report['samples'] ]}"
            f"（原因：{exc}）"
        )
        return

    mismatched = []
    for s in codes_report["samples"]:
        if s["codec_tokenizer_id"] != train_id or s["codec_tokenizer_revision"] != train_rev:
            mismatched.append(s)
    if mismatched:
        _die(
            f"lance codec 版本与训练侧不一致（最大不确定项！）：\n"
            f"  训练侧 id={train_id!r} rev={train_rev!r}\n"
            f"  lance 样本：{[(s['codec_tokenizer_id'], s['codec_tokenizer_revision']) for s in mismatched]}\n"
            "  -> target_codec_codes 将不代表训练 codec，必须重新 join 一致版本后再接入。"
        )
    _ok(f"lance codec 版本与训练侧一致（id={train_id!r} rev={train_rev!r}）。")


def _check_source_bytes_present(reader, sample_rowids: Sequence[int]) -> list:
    _section("检查 4/5 — source 音频字节可用性（内联优先路径）")
    tbl = reader.take_by_stable_rowid(
        list(sample_rowids),
        columns=[SOURCE_INLINE_COL, "source_audio_shard_uri", "source_audio_byte_offset", "source_audio_byte_length"],
    )
    rows = tbl.to_pylist()
    nonempty = 0
    pointer_only = 0
    refs = []
    for r in rows:
        rid = r.get("_rowid")
        inline = r.get(SOURCE_INLINE_COL)
        uri = r.get("source_audio_shard_uri")
        off = r.get("source_audio_byte_offset")
        length = r.get("source_audio_byte_length")
        if inline:
            nonempty += 1
            refs.append((rid, None, None, None))  # 内联，无需 tar 对照
        elif uri and off is not None and length:
            pointer_only += 1
            refs.append((rid, uri, off, length))
        else:
            _die(f"row {rid} 既无内联字节也无有效指针，source 音频不可解析。")
    _ok(f"采样 {len(rows)} 行：内联 {nonempty} 行，指针回退 {pointer_only} 行（训练侧 resolve_lance_audio 会按此解析）。")
    return refs


def _byte_compare_segment(reader, refs: list, reference_tar_dir: Optional[str]) -> None:
    _section("检查 5/5 — source 字节逐字节一致性（可选，需原 wds tar）")
    if not reference_tar_dir:
        _warn(
            "未提供 --reference-tar-dir，跳过逐字节比对。\n"
            "  'lance source_audio_bytes == 原 wds tar 成员' 的等价性为【待验证】项，"
            "需在具备原数据集的对照环境运行本段（传入 --reference-tar-dir）。"
        )
        return
    if not os.path.isdir(reference_tar_dir):
        _warn(f"--reference-tar-dir={reference_tar_dir!r} 不是目录，跳过字节比对。")
        return

    import io

    ok = 0
    for rid, uri, off, length in refs:
        if uri is None:
            # 内联行：无法直接对照原 tar（需 join key）。标记为待人工抽样。
            continue
        tar_name = os.path.basename(str(uri))
        tar_path = os.path.join(reference_tar_dir, tar_name)
        if not os.path.isfile(tar_path):
            _warn(f"  原 tar 缺失 {tar_path}，跳过 row {rid} 字节比对。")
            continue
        with open(tar_path, "rb") as fh:
            fh.seek(int(off))
            tar_bytes = fh.read(int(length))
        inline = reader.take_by_stable_rowid(
            [rid], columns=[SOURCE_INLINE_COL]
        ).to_pylist()[0][SOURCE_INLINE_COL]
        if bytes(inline) == tar_bytes:
            ok += 1
        else:
            _die(f"row {rid} 内联字节与原 tar 成员不一致（等价性被破坏）！")
    if ok:
        _ok(f"{ok} 个指针行内联字节与原 wds tar 成员逐字节一致。")


def _check_row_count(reader, task_filter: Optional[str], rowids: Sequence[int]) -> None:
    _section("附加 — 行数核对（filter 命中集合 vs 原 parquet→JSONL）")
    total = reader.count_rows(filter=task_filter)
    _ok(
        f"filter={task_filter!r} 命中 {len(rowids):,} 行（count_rows={total:,}）。\n"
        "  'TaskSpec.filter 命中集合 == 原 parquet→JSONL 筛选' 为【待验证】项："
        "需与原 make_speech_to_speech_translation 的跳过原因计数对照行数/阈值一致。"
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="Step 0 一票否决验证：lance 作为训练数据源")
    ap.add_argument("--lance-uri", default=os.environ.get("LANCE_TRAIN_SOURCE_URI"),
                    help="processed.lance URI（默认读 env LANCE_TRAIN_SOURCE_URI）")
    ap.add_argument("--lance-version", type=int, default=int(os.environ.get("LANCE_TRAIN_SOURCE_VERSION", "0")) or None,
                    help="Lance 版本（默认最新；0 视为 None）")
    ap.add_argument("--task-filter", default="direction='zh_en' AND target_audio_utmos_score>=3.0",
                    help="TaskSpec.filter 谓词（situation 1）")
    ap.add_argument("--task-name", default="s2st_zh_en", help="TaskSpec.name")
    ap.add_argument("--sample-n", type=int, default=8, help="采样行数（默认 8）")
    ap.add_argument("--num-residual-groups", type=int, default=15,
                    help="训练侧 num_residual_groups（默认 15，codes 第二维须 == +1）")
    ap.add_argument("--reference-tar-dir", default=None,
                    help="原 wds tar 目录（用于 source 字节逐字节比对，可选）")
    ap.add_argument(
        "--check-mel",
        action="store_true",
        help="校验预计算 mel 列（source_input_features*）；mel 表可无 source_audio_bytes",
    )
    args = ap.parse_args()

    if not args.lance_uri:
        _die("必须通过 --lance-uri 或 env LANCE_TRAIN_SOURCE_URI 提供 processed.lance。")

    _section("Step 0 — 验证开始")
    reader = _load_reader(args.lance_uri, args.lance_version)
    version = reader.version

    _check_connectivity(reader, require_source_audio=not args.check_mel)

    rowids = _resolve_task_rowids(reader, args.task_filter, args.task_name, version)
    if not rowids:
        _die(f"filter={args.task_filter!r} 命中 0 行，任务为空。")
    sample_rowids = rowids[: max(1, args.sample_n)]

    codes_report = _check_codes_layout(reader, sample_rowids, args.num_residual_groups)
    _check_codec_version_against_training(codes_report)
    if args.check_mel:
        _check_precomputed_mel(reader, sample_rowids)
        _warn("mel 模式跳过 source_audio_bytes / tar 字节比对（表通常无 raw wav）。")
    else:
        refs = _check_source_bytes_present(reader, sample_rowids)
        _byte_compare_segment(reader, refs, args.reference_tar_dir)
    _check_row_count(reader, args.task_filter, rowids)

    _section("Step 0 — 结论")
    print("全部一票否决闸门通过。可以继续 Step 1~3。\n"
          "仍需在 Step 5 做对拍验证：同 seed、同模型，对比 lance 源 vs 原 wds 源前几 step loss 一致。",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 1A — make_from_lance.py（Shape A：JSONL 指针 manifest）

把训练侧 datamaker 的"源头"从 parquet 换成 Lance。JSONL 契约不变，只是把音频
引用从 ``wds://`` 改成 ``lance://``，并新增 ``target_codec_codes`` 的 lance 引用；
若表含 ``source_input_features``，则写 mel 指针（``audios=[]``）；
若表含 ``source_speaker_embedding``，再带 ``lance://…/speaker``（与 mel/raw 正交）。

设计要点（见 lance对接训练架构设计.md §6 Step 1A）：
  * 用 ``resolve_task(reader, spec)`` 取【全量 _rowid】（不分层），把 split 交回
    ms-swift 用 ``--split_dataset_ratio``（mentor 决策：split / shuffle 沿用原 ms-swift）。
  * 质量闸门前移到 ``TaskSpec.filter`` 谓词下推（BITMAP / BTREE 索引）。
  * ``source_duration_ms`` 改名输出，对齐 ``dataset.py`` 的 ``_estimate_row_lengths``。
  * lance 业务列名是 ``src_lang`` / ``tgt_lang``（非 source_lang / target_lang）。
  * **只读轻列**写指针清单：不拉 ``source_audio_bytes`` / ``codes`` / mel（训练时再按
    ``lance://`` 解引用），避免 datamaker OOM。

前置（部署环境）：
  pip install "curion-storage[lance]" pyarrow lance
  export LANCE_TRAIN_SOURCE_URI="s3://.../processed.lance"
  export LANCE_TRAIN_SOURCE_VERSION=42   # 可选

用法：
  python make_from_lance.py --lance-uri "$LANCE_TRAIN_SOURCE_URI" --output train.jsonl
  python make_from_lance.py --use-precomputed-mel --lance-uri ... --output train_mel.jsonl
  # 训练侧：swift sft ... --dataset train.jsonl --split_dataset_ratio <原值>
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, NoReturn, Optional

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from custom.lance_manifest import (  # noqa: E402
    MANIFEST_COLUMNS,
    build_s2st_record,
    resolve_s2st_task_filter,
    resolve_use_mel,
    resolve_use_speaker_embedding,
)


def _die(msg: str) -> NoReturn:
    print(f"[FAIL] {msg}", file=sys.stderr, flush=True)
    sys.exit(1)


def build_manifest_record(
    row: Dict[str, Any],
    *,
    use_mel: bool = False,
    use_speaker_embedding: bool = False,
) -> Dict[str, Any]:
    """兼容旧调用方；委托 ``build_s2st_record``。"""
    return build_s2st_record(
        row, use_mel=use_mel, use_speaker_embedding=use_speaker_embedding)


def _schema_names(reader) -> list:
    schema = getattr(reader, "schema", None)
    if schema is not None and hasattr(schema, "names"):
        return list(schema.names)
    try:
        return list(reader.ds.schema.names)  # type: ignore[attr-defined]
    except Exception:
        return []


def main() -> int:
    ap = argparse.ArgumentParser(description="Step 1A — 从 Lance 生成 JSONL 指针 manifest")
    ap.add_argument("--lance-uri", default=os.environ.get("LANCE_TRAIN_SOURCE_URI"),
                    help="processed.lance URI（默认读 env LANCE_TRAIN_SOURCE_URI）")
    ap.add_argument("--lance-version", type=int,
                    default=int(os.environ.get("LANCE_TRAIN_SOURCE_VERSION", "0")) or None,
                    help="Lance 版本（默认最新；0 视为 None）")
    ap.add_argument(
        "--task-filter",
        default=None,
        help=(
            "TaskSpec.filter 谓词；默认按表 schema 自动选择 "
            "(legacy: direction+utmos；s2st_train: src_lang+tgt_lang)"
        ),
    )
    ap.add_argument("--task-name", default="s2st_zh_en", help="TaskSpec.name")
    ap.add_argument("--output", required=True, help="输出 JSONL 路径")
    ap.add_argument("--batch-size", type=int, default=4096, help="iter_batches 批次大小")
    ap.add_argument("--latency-budget-ms", type=int, default=4000,
                    help="复刻原 make_speech_to_speech_translation 的固定字段")
    ap.add_argument(
        "--use-precomputed-mel",
        action="store_true",
        help="强制写 source_input_features 指针（audios=[]）",
    )
    ap.add_argument(
        "--no-precomputed-mel",
        action="store_true",
        help="强制写 lance://…/source（忽略表内 mel 列）",
    )
    args = ap.parse_args()

    if not args.lance_uri:
        _die("需 --lance-uri 或 env LANCE_TRAIN_SOURCE_URI。")
    if args.use_precomputed_mel and args.no_precomputed_mel:
        _die("--use-precomputed-mel 与 --no-precomputed-mel 互斥。")
    mel_explicit: Optional[bool]
    if args.use_precomputed_mel:
        mel_explicit = True
    elif args.no_precomputed_mel:
        mel_explicit = False
    else:
        mel_explicit = None

    try:
        from curion.storage.lance_reader import LanceRowReader
        from curion.storage.lance_task import TaskSpec, resolve_task, freeze_manifest, manifest_to_json
    except Exception as exc:
        _die(f"无法 import curion.storage（训练镜像需 pip install curion-storage[lance]）：{exc}")

    reader = LanceRowReader(args.lance_uri, version=args.lance_version)
    cols = _schema_names(reader)
    use_mel = resolve_use_mel(
        explicit=mel_explicit,
        column_names=cols,
    )
    use_spk_emb = resolve_use_speaker_embedding(column_names=cols)
    task_filter = resolve_s2st_task_filter(
        explicit=args.task_filter,
        column_names=cols,
    )
    version = reader.version
    spec = TaskSpec(name=args.task_name, lance_version=version, filter=task_filter)
    resolved = resolve_task(reader, spec)
    rowids = resolved.rowids
    if not rowids:
        _die(f"filter={task_filter!r} 命中 0 行，任务为空。")
    mode = "mel" if use_mel else "raw"
    if use_spk_emb:
        mode += "+speaker"
    print(f"[INFO] resolve_task 命中 {len(rowids):,} 行（全量，不分层；"
          f"由 ms-swift --split_dataset_ratio 切分）；manifest_mode={mode}。", flush=True)

    n = 0
    with open(args.output, "w", encoding="utf-8") as fout:
        for batch in reader.iter_batches(
            rowids, batch_size=args.batch_size, columns=MANIFEST_COLUMNS
        ):
            for row in batch.to_pylist():
                rec = build_s2st_record(
                    row, use_mel=use_mel, use_speaker_embedding=use_spk_emb)
                rec["latency_budget_ms"] = args.latency_budget_ms
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n += 1
    print(f"[INFO] 写出 {n:,} 行 -> {args.output}", flush=True)

    try:
        frozen = freeze_manifest(spec, dataset_uri=args.lance_uri)
        side = args.output + ".freeze.json"
        with open(side, "w", encoding="utf-8") as mf:
            mf.write(manifest_to_json(frozen) + "\n")
        print(f"[INFO] freeze_manifest 完成 -> {side}", flush=True)
    except Exception as exc:
        print(f"[WARN] freeze_manifest 调用失败（不影响 manifest 产出）：{exc}",
              file=sys.stderr, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

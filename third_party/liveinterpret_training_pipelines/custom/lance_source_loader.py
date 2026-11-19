#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 1B — lance_source_loader.py（Shape B：纯 Lance manifest）

HF datasets **GeneratorBasedBuilder** loading-script。让 ms-swift 直接::

    export LANCE_TRAIN_SOURCE_URI=.../processed.lance
    export LANCE_TRAIN_SOURCE_VERSION=4
    export LANCE_TRAIN_TASK_FILTER="src_lang='zh' AND tgt_lang='en'"
    # Legacy tables with direction/utmos still work if you set that filter explicitly.
    # Unset LANCE_TRAIN_TASK_FILTER to auto-pick from the Lance schema.
    export BRIDGE_ALIGNMENT_DATA=/abs/path/to/custom/lance_source_loader.py
    # 或：export LANCE_TRAIN_SHAPE=B  （launch_train_lance*.sh 会自动改写）
    swift sft --dataset bridge_alignment_speech_to_speech_translation ...

消费 Lance——清单由本脚本从 ``processed.lance`` 实时生成，不再落盘 JSONL。

与 Shape A 的关系（见 lance对接训练架构设计.md §2.1 / §6 Step 1B）：
  * 两者产出**完全相同**的字段（见 ``custom/lance_manifest.py``），故训练侧
    wds_audio.py / dataset.py 的改动对 A/B 通用。
  * 区别只在"清单来源形态"：A 落盘 JSONL，B 由本 loading-script 实时从 lance 生成。
  * 用 ``resolve_task`` 取【全量 _rowid，不分层】，split 交回 ms-swift 的
    ``--split_dataset_ratio``。规避 materialize_task 的 rowid 重映射与默认分层。
  * **只读轻列**生成指针（不拉 audio bytes / codes / mel），训练时再按 lance:// 解引用。

可选 env：
  LANCE_TRAIN_MAX_ROWS  — >0 时截断 rowids（smoke / 调试）
  LANCE_STORAGE_REGION / AWS_REGION / AWS_DEFAULT_REGION — S3 region
  LANCE_USE_PRECOMPUTED_MEL — 造清单开关（可选；未设则 schema 探测
    ``source_input_features`` 列）。Mel 行自描述：``audios=[]`` + features 指针。
  LANCE_USE_SPEAKER_EMBEDDING — 是否写 ``lance://…/speaker``（可选；未设则
    schema 探测 ``source_speaker_embedding``）。与 mel/raw **正交**：有列就带指针，
    训练侧 ``QWEN_OMNISPEAK_USE_SPEAKER_PROMPT=true`` 时优先离线 x-vector。
"""
from __future__ import annotations

import os
from typing import Optional

import datasets

try:
    from custom.lance_manifest import (  # type: ignore
        MANIFEST_COLUMNS,
        build_s2st_record,
        resolve_s2st_task_filter,
        resolve_use_mel,
        resolve_use_speaker_embedding,
    )
except ImportError:
    from lance_manifest import (  # type: ignore
        MANIFEST_COLUMNS,
        build_s2st_record,
        resolve_s2st_task_filter,
        resolve_use_mel,
        resolve_use_speaker_embedding,
    )


def _parse_version(raw: Optional[str]):
    if raw is None or str(raw).strip() == "":
        return None
    return int(raw)


def _storage_options():
    region = (
        os.environ.get("LANCE_STORAGE_REGION")
        or os.environ.get("AWS_REGION")
        or os.environ.get("AWS_DEFAULT_REGION")
    )
    if region:
        return {"region": region}
    return None


def _schema_names(reader) -> list:
    schema = getattr(reader, "schema", None)
    if schema is not None and hasattr(schema, "names"):
        return list(schema.names)
    try:
        return list(reader.ds.schema.names)  # type: ignore[attr-defined]
    except Exception:
        return []


class LanceSourceConfig(datasets.BuilderConfig):
    """BuilderConfig so HF fingerprinting stays stable across env changes."""

    def __init__(self, name: str = "default", **kwargs):
        super().__init__(name=name, **kwargs)


class LanceSource(datasets.GeneratorBasedBuilder):
    """Lance-backed S2ST training source (Shape B: pure-lance pointer manifest)."""

    BUILDER_CONFIGS = [LanceSourceConfig(name="default")]
    DEFAULT_CONFIG_NAME = "default"

    def _info(self):
        return datasets.DatasetInfo(
            description=(
                "Lance-backed S2ST training source (Shape B). "
                "Yields lance:// pointer rows; audio/codes/mel resolved at train time."
            ),
        )

    def _split_generators(self, dl_manager):
        uri = os.environ.get("LANCE_TRAIN_SOURCE_URI")
        version = _parse_version(os.environ.get("LANCE_TRAIN_SOURCE_VERSION"))
        # Empty / unset → schema-aware default (legacy direction/utmos vs s2st_train).
        task_filter = os.environ.get("LANCE_TRAIN_TASK_FILTER")
        task_name = os.environ.get("LANCE_TRAIN_TASK_NAME", "s2st_zh_en")
        max_rows_raw = os.environ.get("LANCE_TRAIN_MAX_ROWS", "").strip()
        max_rows = int(max_rows_raw) if max_rows_raw else None
        if not uri:
            raise ValueError(
                "LANCE_TRAIN_SOURCE_URI 未设置。训练镜像需 export "
                "LANCE_TRAIN_SOURCE_URI=s3://.../processed.lance "
                "（或本地 /nvme/.../processed.lance）"
            )
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "uri": uri,
                    "version": version,
                    "task_filter": task_filter,
                    "task_name": task_name,
                    "max_rows": max_rows,
                },
            )
        ]

    def _generate_examples(self, uri, version, task_filter, task_name, max_rows=None):
        from curion.storage.lance_reader import LanceRowReader
        from curion.storage.lance_task import TaskSpec, resolve_task

        storage_options = _storage_options()
        try:
            reader = LanceRowReader(
                uri, version=version, storage_options=storage_options)
        except TypeError:
            reader = LanceRowReader(uri, version=version)
        use_mel = resolve_use_mel(column_names=_schema_names(reader))
        cols = _schema_names(reader)
        use_spk_emb = resolve_use_speaker_embedding(column_names=cols)
        resolved_filter = resolve_s2st_task_filter(
            explicit=task_filter,
            column_names=cols,
        )
        resolved_version = reader.version
        spec = TaskSpec(
            name=task_name, lance_version=resolved_version, filter=resolved_filter
        )
        rowids = list(resolve_task(reader, spec).rowids)
        if max_rows is not None and max_rows > 0:
            rowids = rowids[:max_rows]
        if not rowids:
            raise ValueError(f"filter={resolved_filter!r} 命中 0 行，任务为空。")

        for batch in reader.iter_batches(
            rowids, batch_size=4096, columns=MANIFEST_COLUMNS
        ):
            for row in batch.to_pylist():
                rid = row["_rowid"]
                yield rid, build_s2st_record(
                    row, use_mel=use_mel, use_speaker_embedding=use_spk_emb)

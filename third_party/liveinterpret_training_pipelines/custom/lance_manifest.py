#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared S2ST Lance pointer-manifest helpers (Shape A materialize + Shape B loader).

Contract (mirrors ``target_codec_codes``):
  * Raw source:  ``audios: [lance://<_rowid>/source]``
  * Mel source:  ``audios: []`` + ``source_input_features: lance://<_rowid>/features``
                 (+ optional ``source_speaker_embedding: lance://<_rowid>/speaker``)
  * Target:      ``target_codec_codes: lance://<_rowid>/codes`` always

Mode is **self-describing** via presence of ``source_input_features`` — training
preprocess must not rewrite audios based on env.
"""
from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence

# Ray runtime-env YAML can turn split='train' into split=train, which Lance
# then parses as identifier `train`. Only repair the fully-unquoted col=val form.
_UNQUOTED_EQ = re.compile(r'^([A-Za-z_][A-Za-z0-9_]*)=([A-Za-z0-9_]+)$')


def _repair_stripped_sql_quotes(expr: str) -> str:
    m = _UNQUOTED_EQ.fullmatch(str(expr).strip())
    if m:
        return f"{m.group(1)} = '{m.group(2)}'"
    return expr


MANIFEST_COLUMNS = [
    'target_text',
    'src_lang',
    'tgt_lang',
    'duration_ms',
]

MEL_FEATURE_COLUMN = 'source_input_features'
SPEAKER_EMBEDDING_COLUMN = 'source_speaker_embedding'


def s2s_task_prompt(target_lang: Optional[str]) -> str:
    try:
        from custom.task_prompts import s2s_task_prompt as _fn  # type: ignore
        return _fn(target_lang)
    except Exception:
        try:
            from task_prompts import s2s_task_prompt as _fn  # type: ignore
            return _fn(target_lang)
        except Exception:
            return (
                'Task: Listen to the input speech, translate it, and synthesize '
                f'the translation as {target_lang} speech.'
            )


def env_wants_precomputed_mel() -> Optional[bool]:
    """Return explicit env override, or None if unset."""
    raw = os.environ.get('LANCE_USE_PRECOMPUTED_MEL')
    if raw is None or str(raw).strip() == '':
        return None
    return str(raw).strip().lower() in {'1', 'true', 'yes', 'y', 'on'}


def schema_has_precomputed_mel(column_names: Sequence[str]) -> bool:
    return MEL_FEATURE_COLUMN in set(column_names)


def schema_has_speaker_embedding(column_names: Sequence[str]) -> bool:
    return SPEAKER_EMBEDDING_COLUMN in set(column_names)


def env_wants_speaker_embedding() -> Optional[bool]:
    """Return explicit env override, or None if unset."""
    raw = os.environ.get('LANCE_USE_SPEAKER_EMBEDDING')
    if raw is None or str(raw).strip() == '':
        return None
    return str(raw).strip().lower() in {'1', 'true', 'yes', 'y', 'on'}


def resolve_use_mel(
    *,
    explicit: Optional[bool] = None,
    column_names: Optional[Sequence[str]] = None,
) -> bool:
    """Decide mel vs raw for manifest emission.

    Priority: explicit arg > ``LANCE_USE_PRECOMPUTED_MEL`` env > schema probe.
    """
    if explicit is not None:
        return bool(explicit)
    env = env_wants_precomputed_mel()
    if env is not None:
        return env
    if column_names is not None:
        return schema_has_precomputed_mel(column_names)
    return False


def resolve_use_speaker_embedding(
    *,
    explicit: Optional[bool] = None,
    column_names: Optional[Sequence[str]] = None,
) -> bool:
    """Decide whether to emit ``source_speaker_embedding`` pointers.

    Orthogonal to mel/raw: any table that materialized the ECAPA column can
    supply offline x-vectors. Priority: explicit > ``LANCE_USE_SPEAKER_EMBEDDING``
    env > schema probe.
    """
    if explicit is not None:
        return bool(explicit)
    env = env_wants_speaker_embedding()
    if env is not None:
        return env
    if column_names is not None:
        return schema_has_speaker_embedding(column_names)
    return False


# Legacy curated tables carry `direction` + utmos. The Curion s2st_train view
# (and schema v3+) does not: utmos is null and direction is absent. Auto-pick
# a filter that matches the table so default launches do not yield 0 rows.
_LEGACY_S2ST_TASK_FILTER = "direction='zh_en' AND target_audio_utmos_score>=3.0"
_TRAIN_VIEW_S2ST_TASK_FILTER = "src_lang='zh' AND tgt_lang='en'"


def default_s2st_task_filter(
    column_names: Optional[Sequence[str]] = None,
) -> str:
    """Pick a default Lance task filter for the table schema.

    Prefer the legacy ``direction``/utmos filter when those columns exist.
    Otherwise use ``src_lang``/``tgt_lang`` for the Curion s2st_train view.
    """

    cols = {str(name) for name in (column_names or ())}
    if 'direction' in cols:
        return _LEGACY_S2ST_TASK_FILTER
    if 'src_lang' in cols and 'tgt_lang' in cols:
        return _TRAIN_VIEW_S2ST_TASK_FILTER
    return _LEGACY_S2ST_TASK_FILTER


def resolve_s2st_task_filter(
    *,
    explicit: Optional[str] = None,
    column_names: Optional[Sequence[str]] = None,
) -> str:
    """Resolve the task filter: explicit > env > schema-aware default."""

    if explicit is not None and str(explicit).strip():
        return _repair_stripped_sql_quotes(str(explicit).strip())
    env = os.environ.get('LANCE_TRAIN_TASK_FILTER')
    if env is not None and str(env).strip():
        return _repair_stripped_sql_quotes(str(env).strip())
    return default_s2st_task_filter(column_names)


def build_s2st_record(
    row: Dict[str, Any],
    *,
    use_mel: bool = False,
    use_speaker_embedding: bool = False,
) -> Dict[str, Any]:
    """Map one Lance light-column row to an ms-swift S2ST JSONL record."""
    rid = row['_rowid']
    target_lang = row.get('tgt_lang')
    rec: Dict[str, Any] = {
        'schema_version': 'ms-swift-speech-simul-v1',
        'task': 'speech_to_speech_translation',
        'task_prompt': s2s_task_prompt(target_lang),
        'messages': [{'role': 'user', 'content': '<audio>'}],
        'target_codec_codes': f'lance://{rid}/codes',
        'target_text': row.get('target_text'),
        'source_lang': row.get('src_lang'),
        'target_lang': target_lang,
        'source_duration_ms': row.get('duration_ms'),
    }
    if use_mel:
        # Self-describing mel mode: empty audios; features pointer carries source.
        rec['audios'] = []
        rec['source_input_features'] = f'lance://{rid}/features'
    else:
        rec['audios'] = [f'lance://{rid}/source']
    # Speaker is orthogonal to mel/raw: offline ECAPA when the column exists.
    if use_speaker_embedding:
        rec['source_speaker_embedding'] = f'lance://{rid}/speaker'
    return rec


def _storage_options() -> Optional[Dict[str, str]]:
    region = (
        os.environ.get('LANCE_STORAGE_REGION')
        or os.environ.get('AWS_REGION')
        or os.environ.get('AWS_DEFAULT_REGION')
    )
    if region:
        return {'region': region}
    return None


def _open_reader(uri: str, version: Optional[int]):
    from curion.storage.lance_reader import LanceRowReader

    storage_options = _storage_options()
    try:
        return LanceRowReader(uri, version=version, storage_options=storage_options)
    except TypeError:
        return LanceRowReader(uri, version=version)


def iter_s2st_records(
    uri: str,
    *,
    version: Optional[int] = None,
    task_filter: Optional[str] = None,
    task_name: str = 's2st_zh_en',
    max_rows: Optional[int] = None,
    batch_size: int = 4096,
    use_mel: Optional[bool] = None,
) -> Iterable[Dict[str, Any]]:
    """Yield S2ST pointer records from a Lance table (light columns only)."""
    from curion.storage.lance_task import TaskSpec, resolve_task

    reader = _open_reader(uri, version)
    cols = list(getattr(reader, 'schema', None).names
                if getattr(reader, 'schema', None) is not None else [])
    if not cols:
        # Fallback: probe via lance dataset if reader has no .schema
        try:
            cols = list(reader.ds.schema.names)  # type: ignore[attr-defined]
        except Exception:
            cols = []
    mel = resolve_use_mel(explicit=use_mel, column_names=cols or None)
    use_spk_emb = resolve_use_speaker_embedding(column_names=cols or None)
    resolved_filter = resolve_s2st_task_filter(
        explicit=task_filter,
        column_names=cols or None,
    )

    spec = TaskSpec(
        name=task_name,
        lance_version=reader.version,
        filter=resolved_filter,
    )
    rowids = list(resolve_task(reader, spec).rowids)
    if max_rows is not None and max_rows > 0:
        rowids = rowids[:max_rows]
    if not rowids:
        raise ValueError(f'filter={resolved_filter!r} 命中 0 行，任务为空。')

    for batch in reader.iter_batches(rowids, batch_size=batch_size, columns=MANIFEST_COLUMNS):
        for row in batch.to_pylist():
            yield build_s2st_record(
                row, use_mel=mel, use_speaker_embedding=use_spk_emb)


def materialize_s2st_jsonl(
    uri: str,
    out_path: str,
    *,
    version: Optional[int] = None,
    task_filter: Optional[str] = None,
    task_name: str = 's2st_zh_en',
    max_rows: Optional[int] = None,
    batch_size: int = 4096,
    use_mel: Optional[bool] = None,
    latency_budget_ms: Optional[int] = None,
) -> int:
    """Write pointer JSONL; return row count."""
    parent = os.path.dirname(out_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    n = 0
    with open(out_path, 'w', encoding='utf-8') as fout:
        for rec in iter_s2st_records(
                uri,
                version=version,
                task_filter=task_filter,
                task_name=task_name,
                max_rows=max_rows,
                batch_size=batch_size,
                use_mel=use_mel):
            if latency_budget_ms is not None:
                rec['latency_budget_ms'] = latency_budget_ms
            fout.write(json.dumps(rec, ensure_ascii=False) + '\n')
            n += 1
    return n


def materialize_from_env(out_path: Optional[str] = None) -> str:
    """CLI/launch helper: read Lance env vars and write JSONL. Returns path."""
    uri = os.environ.get('LANCE_TRAIN_SOURCE_URI')
    if not uri:
        raise ValueError('LANCE_TRAIN_SOURCE_URI is required')
    version_raw = os.environ.get('LANCE_TRAIN_SOURCE_VERSION') or None
    version = int(version_raw) if version_raw else None
    # Empty / unset → schema-aware default inside resolve_s2st_task_filter.
    task_filter = os.environ.get('LANCE_TRAIN_TASK_FILTER')
    print(f'[lance_manifest] LANCE_TRAIN_TASK_FILTER={task_filter!r}', flush=True)
    task_name = os.environ.get('LANCE_TRAIN_TASK_NAME', 's2st_zh_en')
    max_rows_raw = os.environ.get('LANCE_TRAIN_MAX_ROWS', '').strip()
    max_rows = int(max_rows_raw) if max_rows_raw else None
    use_mel = env_wants_precomputed_mel()
    if out_path is None:
        out_path = os.environ.get('MANIFEST_OUT')
    if not out_path:
        raise ValueError('MANIFEST_OUT / out_path is required')
    n = materialize_s2st_jsonl(
        uri,
        out_path,
        version=version,
        task_filter=task_filter,
        task_name=task_name,
        max_rows=max_rows,
        use_mel=use_mel,
    )
    with open(out_path, encoding='utf-8') as f:
        first = f.readline()
    mode = 'mel' if first and 'source_input_features' in first else 'raw'
    if first and 'source_speaker_embedding' in first:
        mode += '+speaker'
    print(f'[lance_manifest] wrote {mode} manifest rows={n} → {out_path}')
    return out_path


__all__ = [
    'MANIFEST_COLUMNS',
    'MEL_FEATURE_COLUMN',
    'SPEAKER_EMBEDDING_COLUMN',
    'build_s2st_record',
    'env_wants_precomputed_mel',
    'env_wants_speaker_embedding',
    'iter_s2st_records',
    'materialize_from_env',
    'materialize_s2st_jsonl',
    'resolve_use_mel',
    'resolve_use_speaker_embedding',
    's2s_task_prompt',
    'schema_has_precomputed_mel',
    'schema_has_speaker_embedding',
]

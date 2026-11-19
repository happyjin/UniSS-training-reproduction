#!/usr/bin/env python3
"""Repair bidirectional S2S prompts without recomputing audio features.

Only ``messages`` and ``task_prompt`` are replaced. All audio paths, codec
codes, text ids, x-vectors, labels, and language columns are reused as Arrow
arrays from the input Parquet row group.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import glob
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import pyarrow as pa
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / 'custom'))

from task_prompts import normalize_language_code, s2s_task_prompt, set_system_prompt


PROMPT_COLUMNS = {'messages', 'task_prompt'}


def _system_content(messages: Any) -> str:
    for message in messages or []:
        if message and message.get('role') == 'system':
            return (message.get('content') or '').strip()
    return ''


def _rewrite_table(table: pa.Table) -> Tuple[pa.Table, Dict[str, int]]:
    target_langs = table['target_lang'].to_pylist()
    old_messages = table['messages'].to_pylist()
    old_task_prompts = table['task_prompt'].to_pylist()

    new_messages: List[List[Dict[str, Any]]] = []
    stats: Counter[str] = Counter()
    for target_lang, messages, task_prompt in zip(
        target_langs, old_messages, old_task_prompts
    ):
        normalized_lang = normalize_language_code(target_lang)
        expected = s2s_task_prompt(normalized_lang)
        old_system = _system_content(messages)
        old_task = (task_prompt or '').strip()

        stats['rows'] += 1
        stats[f'target_{normalized_lang}'] += 1
        stats['input_system_mismatch'] += int(old_system != expected)
        stats['input_task_prompt_mismatch'] += int(old_task != expected)
        stats['input_duplicate_locations'] += int(
            bool(old_task) and old_task == old_system)

        repaired = set_system_prompt(messages, expected)
        new_messages.append(repaired)
        stats['changed_messages'] += int(repaired != (messages or []))
        stats['cleared_task_prompts'] += int(bool(old_task))

    message_index = table.schema.get_field_index('messages')
    task_prompt_index = table.schema.get_field_index('task_prompt')
    message_type = table.schema.field(message_index).type
    task_prompt_type = table.schema.field(task_prompt_index).type

    repaired_table = table.set_column(
        message_index,
        'messages',
        pa.array(new_messages, type=message_type),
    )
    repaired_table = repaired_table.set_column(
        task_prompt_index,
        'task_prompt',
        pa.array([''] * table.num_rows, type=task_prompt_type),
    )

    # This is the key migration invariant: all expensive supervision columns
    # are the original Arrow arrays and remain logically identical.
    for name in table.column_names:
        if name not in PROMPT_COLUMNS:
            assert table[name].equals(repaired_table[name]), name
    return repaired_table, dict(stats)


def _add_stats(total: Counter[str], partial: Dict[str, int]) -> None:
    total.update(partial)


def repair_file(job: Tuple[str, str, str]) -> Dict[str, Any]:
    input_path, output_path, compression = job
    source = pq.ParquetFile(input_path)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + f'.tmp.{os.getpid()}')
    stats: Counter[str] = Counter()
    started = time.time()

    try:
        with pq.ParquetWriter(
            temporary,
            source.schema_arrow,
            compression=compression,
        ) as writer:
            for row_group in range(source.metadata.num_row_groups):
                table = source.read_row_group(row_group)
                repaired, row_group_stats = _rewrite_table(table)
                writer.write_table(repaired, row_group_size=repaired.num_rows)
                _add_stats(stats, row_group_stats)
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()

    result = pq.ParquetFile(output)
    if result.metadata.num_rows != source.metadata.num_rows:
        raise RuntimeError(
            f'Row count changed for {input_path}: '
            f'{source.metadata.num_rows} -> {result.metadata.num_rows}')
    if not result.schema_arrow.equals(source.schema_arrow):
        raise RuntimeError(f'Schema changed for {input_path}')

    return {
        'input': input_path,
        'output': str(output),
        'input_bytes': os.path.getsize(input_path),
        'output_bytes': os.path.getsize(output),
        'row_groups': source.metadata.num_row_groups,
        'elapsed_seconds': round(time.time() - started, 3),
        **dict(stats),
    }


def _jobs(input_glob: str, output_dir: str, compression: str) -> Iterable[Tuple[str, str, str]]:
    paths = sorted(glob.glob(input_glob))
    if not paths:
        raise FileNotFoundError(f'No Parquet files matched: {input_glob}')
    for input_path in paths:
        yield input_path, str(Path(output_dir) / Path(input_path).name), compression


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-glob', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--compression', default='zstd')
    parser.add_argument('--manifest', default=None)
    parser.add_argument('--limit-files', type=int, default=0)
    args = parser.parse_args()

    jobs = list(_jobs(args.input_glob, args.output_dir, args.compression))
    if args.limit_files:
        jobs = jobs[:args.limit_files]
    started = time.time()
    results: List[Dict[str, Any]] = []

    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        future_to_job = {pool.submit(repair_file, job): job for job in jobs}
        for future in concurrent.futures.as_completed(future_to_job):
            result = future.result()
            results.append(result)
            print(
                f"repaired {Path(result['input']).name}: {result['rows']:,} rows "
                f"in {result['elapsed_seconds']:.1f}s",
                flush=True,
            )

    results.sort(key=lambda item: item['input'])
    totals: Counter[str] = Counter()
    for result in results:
        for key, value in result.items():
            if isinstance(value, int):
                totals[key] += value

    manifest = {
        'schema_version': 1,
        'migration': 'bidirectional_s2s_prompt_v2_to_prompt_fixed_v3',
        'created_at_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'source_glob': args.input_glob,
        'output_dir': str(Path(args.output_dir).resolve()),
        'prompt_policy': {
            'canonical_location': 'messages[system]',
            'task_prompt': 'empty legacy compatibility column',
            'en': s2s_task_prompt('en'),
            'zh': s2s_task_prompt('zh'),
        },
        'preserved_columns': 'all columns except messages and task_prompt',
        'elapsed_seconds': round(time.time() - started, 3),
        'totals': dict(totals),
        'files': results,
    }
    manifest_path = Path(args.manifest or Path(args.output_dir) / 'prompt_repair_manifest.json')
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_manifest = manifest_path.with_suffix(manifest_path.suffix + '.tmp')
    temporary_manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    os.replace(temporary_manifest, manifest_path)
    print(f'manifest: {manifest_path}', flush=True)
    print(json.dumps(manifest['totals'], sort_keys=True), flush=True)


if __name__ == '__main__':
    main()

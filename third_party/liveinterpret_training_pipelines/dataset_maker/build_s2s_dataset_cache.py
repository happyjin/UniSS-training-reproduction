#!/usr/bin/env python3
"""Prebuild the exact ms-swift cache used by bidirectional S2S training."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True, help='Corrected Parquet path or glob')
    parser.add_argument('--cache-dir', required=True)
    parser.add_argument(
        '--num-proc', type=int, default=16,
        help='Mapping workers; use 0 for true in-process mapping without IPC.')
    parser.add_argument('--split-dataset-ratio', type=float, default=0.01)
    parser.add_argument('--manifest', default=None)
    args = parser.parse_args()

    data_path = os.path.abspath(args.data)
    cache_dir = os.path.abspath(args.cache_dir)
    Path(cache_dir).mkdir(parents=True, exist_ok=True)

    # These must be set before importing datasets/ms-swift, whose cache paths
    # are initialized at import time.
    os.environ['BRIDGE_ALIGNMENT_DATA'] = data_path
    os.environ['HF_DATASETS_CACHE'] = cache_dir
    # ms-swift's local-path loader explicitly uses
    # ``MODELSCOPE_CACHE/datasets`` as datasets.load_dataset(cache_dir=...).
    os.environ['MODELSCOPE_CACHE'] = cache_dir

    from custom.dataset import register_bridge_alignment_datasets
    from swift.dataset import load_dataset

    register_bridge_alignment_datasets(data_path)
    started = time.time()
    num_proc = None if args.num_proc <= 0 else args.num_proc
    train_dataset, val_dataset = load_dataset(
        'bridge_alignment_speech_to_speech_translation',
        split_dataset_ratio=args.split_dataset_ratio,
        seed=42,
        num_proc=num_proc,
        load_from_cache_file=False,
        shuffle=False,
        remove_unused_columns=False,
        strict=True,
    )

    train_rows = len(train_dataset)
    val_rows = len(val_dataset) if val_dataset is not None else 0
    for split_name, dataset in [('train', train_dataset), ('validation', val_dataset)]:
        if dataset is None or not len(dataset):
            continue
        for index in {0, len(dataset) - 1}:
            row = dataset[index]
            system_messages = [
                message.get('content', '')
                for message in row['messages']
                if message.get('role') == 'system'
            ]
            if len(system_messages) != 1:
                raise RuntimeError(
                    f'{split_name}[{index}] has {len(system_messages)} system prompts')
            if row.get('task_prompt'):
                raise RuntimeError(f'{split_name}[{index}] has non-empty task_prompt')

    manifest = {
        'schema_version': 1,
        'created_at_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'data': data_path,
        'cache_dir': cache_dir,
        'arrow_cache_dir': str(Path(cache_dir) / 'datasets'),
        'dataset_name': 'bridge_alignment_speech_to_speech_translation',
        'split_dataset_ratio': args.split_dataset_ratio,
        'seed': 42,
        'num_proc': args.num_proc,
        'effective_num_proc': num_proc,
        'load_from_cache_file_during_build': False,
        'train_rows': train_rows,
        'validation_rows': val_rows,
        'total_rows': train_rows + val_rows,
        'elapsed_seconds': round(time.time() - started, 3),
    }
    manifest_path = Path(
        args.manifest or Path(cache_dir) / 's2s_cache_manifest.json')
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Stage-level bench for map/indices (P0/P1) and DataLoader iter (D0/D1).

Import ``custom.plugin`` before ``load_dataset`` so routing-tag / flatten
patches match training. Does not load the 3B model and does not train.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / 'vendor'))
sys.path.insert(0, str(REPO_ROOT))


def _log(msg: str) -> None:
    print(f'[stage-bench] {msg}', flush=True)


def _install_flatten_counter():
    try:
        from datasets import Dataset as HfDataset
    except Exception as exc:
        _log(f'flatten_counter skipped: {exc}')
        return {'calls': 0}

    state = {'calls': 0}
    original = HfDataset.flatten_indices

    def _wrapped(self, *args, **kwargs):
        state['calls'] += 1
        _log(f'flatten_indices call #{state["calls"]} num_rows={len(self)}')
        return original(self, *args, **kwargs)

    HfDataset.flatten_indices = _wrapped
    return state


def _run_map(args) -> dict:
    t0 = time.time()
    data_path = args.data
    cache_dir = os.path.abspath(args.cache_dir)
    Path(cache_dir).mkdir(parents=True, exist_ok=True)

    os.environ['BRIDGE_ALIGNMENT_DATA'] = data_path
    os.environ['HF_DATASETS_CACHE'] = cache_dir
    os.environ['MODELSCOPE_CACHE'] = cache_dir
    os.environ.setdefault('WDS_AUDIO_MODE', 'required')
    os.environ.setdefault('WDS_AUDIO_STRICT', 'true')
    os.environ.setdefault('SCENE', 'batch4_8_adapter_cmp')
    os.environ['PLUGIN_PATH'] = str(REPO_ROOT / 'custom' / 'plugin.py')

    _log(f't0_plugin data={data_path}')
    import custom.plugin  # noqa: F401

    flatten_state = _install_flatten_counter()
    t_plugin = time.time()
    _log(f'plugin_ready elapsed_s={t_plugin - t0:.3f}')

    from custom.dataset import register_bridge_alignment_datasets
    from swift.dataset import load_dataset

    register_bridge_alignment_datasets(os.path.abspath(data_path))
    num_proc = None if args.num_proc <= 0 else args.num_proc
    t_map = time.time()
    train_dataset, val_dataset = load_dataset(
        'bridge_alignment_speech_to_speech_translation',
        split_dataset_ratio=args.split_dataset_ratio,
        seed=42,
        num_proc=num_proc,
        load_from_cache_file=args.load_from_cache,
        shuffle=False,
        remove_unused_columns=False,
        strict=True,
    )
    t_ready = time.time()
    train_rows = len(train_dataset)
    val_rows = len(val_dataset) if val_dataset is not None else 0
    flatten_calls = int(flatten_state['calls'])
    _log(
        f'routing_tag elapsed_s={t_ready - t_map:.3f} '
        f'flatten={"yes" if flatten_calls else "no"} flatten_calls={flatten_calls}'
    )
    _log(f'map_done elapsed_s={t_ready - t_map:.3f}')
    _log(
        f'dataset_ready elapsed_s={t_ready - t0:.3f} '
        f'train_rows={train_rows} val_rows={val_rows}'
    )

    manifest = {
        'schema_version': 1,
        'mode': 'map',
        'created_at_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'data': data_path,
        'cache_dir': cache_dir,
        'split_dataset_ratio': args.split_dataset_ratio,
        'num_proc': args.num_proc,
        'load_from_cache_file': args.load_from_cache,
        'train_rows': train_rows,
        'validation_rows': val_rows,
        'total_rows': train_rows + val_rows,
        'flatten_calls': flatten_calls,
        'elapsed_plugin_s': round(t_plugin - t0, 3),
        'elapsed_map_s': round(t_ready - t_map, 3),
        'elapsed_dataset_ready_s': round(t_ready - t0, 3),
        'lance_materialize_s': os.environ.get('LANCE_MATERIALIZE_ELAPSED_S'),
        'lance_manifest_reused': os.environ.get('LANCE_MANIFEST_REUSED'),
        'lance_source_uri': os.environ.get('LANCE_TRAIN_SOURCE_URI'),
    }
    return manifest


def _percentile(values, q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    idx = min(len(ordered) - 1, max(0, int(round((q / 100.0) * (len(ordered) - 1)))))
    return float(ordered[idx])


def _wrap_lazy_encode_dataset(hf_dataset):
    """Wrap HF rows like ``swift sft --lazy_tokenize`` so D0/D1 hit encode.

    Raw Arrow iteration does not run tar/Whisper or Lance ``take``; that would
    understate the DataLoader gap the mentor asked about. Requires MODEL_PATH
    (tokenizer / processor only — no 3B weights).
    """
    model_path = (os.environ.get('MODEL_PATH') or '').strip()
    if not model_path:
        raise SystemExit(
            '[stage-bench][FATAL] dataloader mode needs MODEL_PATH so '
            'LazyLLMDataset can call template.encode (same path as training).')
    from swift.dataset.utils import LazyLLMDataset
    from swift.model import get_processor
    from swift.template import get_template

    # ms-swift 4.x: get_template(processor), not get_template(type, processor).
    # get_processor(..., load_model=False) attaches model_info without 3B weights.
    model_type = (os.environ.get('MODEL_TYPE') or '').strip() or None
    template_type = (
        os.environ.get('SWIFT_TEMPLATE_TYPE') or 'qwen_omnispeak').strip()
    _log(f'lazy_wrap loading processor model_type={model_type} '
         f'template_type={template_type}')
    try:
        processor = get_processor(
            model_path, model_type=model_type, download_model=False)
        template = get_template(processor, template_type=template_type)
    except Exception as exc:
        raise SystemExit(
            '[stage-bench][FATAL] cannot build template.encode: '
            f'{type(exc).__name__}: {exc}') from exc
    if hasattr(template, 'set_mode'):
        template.set_mode('train')
    # LazyLLMDataset stores encode_func and later calls
    # encode_func(row, return_length=True). Passing the Template object
    # succeeds at init and only fails in the worker.
    lazy = LazyLLMDataset(hf_dataset, template.encode)
    _log(f'lazy_wrap dataset={type(hf_dataset).__name__} '
         f'template={type(template).__name__}')
    return lazy


def _run_dataloader(args) -> dict:
    os.environ.setdefault('DATALOADER_NUM_WORKERS', '16')
    os.environ.setdefault('DATALOADER_PREFETCH_FACTOR', '8')
    os.environ.setdefault('DATALOADER_PERSISTENT_WORKERS', 'true')
    os.environ.setdefault('DATALOADER_CPU_BIND', '1')

    args.load_from_cache = True
    map_manifest = _run_map(args)
    from torch.utils.data import DataLoader
    from swift.dataset import load_dataset
    from custom.dataset import register_bridge_alignment_datasets

    register_bridge_alignment_datasets(os.path.abspath(args.data))
    train_dataset, _ = load_dataset(
        'bridge_alignment_speech_to_speech_translation',
        split_dataset_ratio=args.split_dataset_ratio,
        seed=42,
        num_proc=None if args.num_proc <= 0 else args.num_proc,
        load_from_cache_file=True,
        shuffle=False,
        remove_unused_columns=False,
        strict=True,
    )
    train_dataset = _wrap_lazy_encode_dataset(train_dataset)

    workers = int(os.environ.get('DATALOADER_NUM_WORKERS', '16'))
    prefetch = int(os.environ.get('DATALOADER_PREFETCH_FACTOR', '8'))
    batch_size = int(os.environ.get('PER_DEVICE_TRAIN_BATCH_SIZE', '64'))
    persistent = os.environ.get('DATALOADER_PERSISTENT_WORKERS', 'true').lower() in {
        '1', 'true', 'yes',
    }

    def _collate(rows):
        # Encode still runs in the worker. Returning only the count keeps
        # prefetch=8 from parking 128 full omni batches in /dev/shm
        # (pcas died at i≈40 with "No space left on device").
        return len(rows)

    loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=workers,
        prefetch_factor=prefetch if workers > 0 else None,
        persistent_workers=persistent if workers > 0 else False,
        collate_fn=_collate,
        shuffle=False,
    )
    _log(
        f'dl_start workers={workers} prefetch={prefetch} '
        f'batch_size={batch_size} batches={args.batches}'
    )
    latencies = []
    samples = 0
    t0 = time.time()
    t_prev = t0
    for i, batch in enumerate(loader):
        now = time.time()
        if i == 0:
            t0 = now
            t_prev = now
            _log('dl_warmup_skip i=0')
            continue
        ms = (now - t_prev) * 1000.0
        t_prev = now
        latencies.append(ms)
        n_rows = batch if isinstance(batch, int) else len(batch)
        samples += n_rows
        if i <= 5 or i % 20 == 0:
            _log(f'dl_batch i={i} ms={ms:.1f} rows={n_rows}')
        if i >= args.batches:
            break
    wall = time.time() - t0
    p50 = _percentile(latencies, 50)
    p95 = _percentile(latencies, 95)
    sps = samples / wall if wall > 0 else 0.0
    _log(
        f'dl_summary batches={len(latencies)} samples={samples} '
        f'samples/s={sps:.2f} p50_ms={p50:.1f} p95_ms={p95:.1f} wall_s={wall:.3f}'
    )
    manifest = dict(map_manifest)
    manifest.update({
        'mode': 'dataloader',
        'dl_workers': workers,
        'dl_prefetch': prefetch,
        'dl_batch_size': batch_size,
        'dl_batches': len(latencies),
        'dl_samples': samples,
        'dl_samples_per_s': round(sps, 3),
        'dl_p50_ms': round(p50, 3),
        'dl_p95_ms': round(p95, 3),
        'dl_wall_s': round(wall, 3),
    })
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('map', 'dataloader'), default='map')
    parser.add_argument('--data', required=True)
    parser.add_argument('--cache-dir', required=True)
    parser.add_argument('--num-proc', type=int, default=16)
    parser.add_argument('--split-dataset-ratio', type=float, default=0.0005)
    parser.add_argument('--manifest', default=None)
    parser.add_argument('--load-from-cache', action='store_true')
    parser.add_argument('--batches', type=int, default=80)
    args = parser.parse_args()

    if args.mode == 'map':
        manifest = _run_map(args)
    else:
        manifest = _run_dataloader(args)

    manifest_path = Path(
        args.manifest or Path(args.cache_dir) / 'stage_bench_manifest.json')
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)
    _log(f'manifest_wrote {manifest_path}')


if __name__ == '__main__':
    main()

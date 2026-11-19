#!/usr/bin/env python3
"""Mirror `text_config.rope_parameters` into `text_config.rope_scaling`.

Background
----------
The exported `qwen_omnispeak_*` configs were produced under transformers
master where the rope-related fields live under the new key
`rope_parameters` (with `mrope_section`, `rope_type`, etc.).

The currently-installed transformers (5.8.1) still reads `rope_scaling`
inside `Qwen3OmniMoeThinkerTextRotaryEmbedding`. When `rope_scaling` is
absent, it crashes with::

    AttributeError: 'NoneType' object has no attribute 'get'

This script walks the config recursively. For every dict whose model_type
contains `text` or whose key ends with `text_config`, if it has
`rope_parameters` but not `rope_scaling`, we set
`rope_scaling = copy(rope_parameters)`.

It's idempotent and safe: existing `rope_scaling` blocks are left intact.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from typing import Any


def _patch_node(d: dict) -> int:
    """Return number of fields added to ``d`` (a single dict)."""
    added = 0
    if "rope_parameters" in d and "rope_scaling" not in d:
        d["rope_scaling"] = copy.deepcopy(d["rope_parameters"])
        added += 1
    return added


def _walk(obj: Any, key_path: str = "") -> int:
    added = 0
    if isinstance(obj, dict):
        # Patch this level if it looks like a text_config / talker text_config
        # / thinker text_config.
        if "rope_parameters" in obj:
            added += _patch_node(obj)
        for k, v in obj.items():
            added += _walk(v, f"{key_path}.{k}" if key_path else k)
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            added += _walk(item, f"{key_path}[{i}]")
    return added


# Token-id keys that ms-swift template reads off the *top-level* config but
# the exported Qwen3OmniMoe config keeps them nested under thinker_config /
# talker_config. We mirror them up so swift/template/templates/qwen.py works.
_TOP_LEVEL_TOKEN_KEYS = (
    'audio_token_id', 'audio_start_token_id', 'audio_end_token_id',
    'image_token_id', 'video_token_id',
    'vision_start_token_id', 'vision_end_token_id',
    'audio_token_index', 'image_token_index', 'video_token_index',
    'tts_bos_token_id', 'tts_eos_token_id', 'tts_pad_token_id',
)


def _hoist_token_ids(cfg: dict) -> int:
    """Mirror nested token_id fields up to the top level. Returns count added."""
    added = 0
    nested_sources = []
    for parent_key in ('thinker_config',):
        parent = cfg.get(parent_key)
        if isinstance(parent, dict):
            nested_sources.append(parent)
            for child_key in ('thinker_config', 'talker_config'):
                child = parent.get(child_key)
                if isinstance(child, dict):
                    nested_sources.append(child)
    for k in _TOP_LEVEL_TOKEN_KEYS:
        if k in cfg:
            continue
        for src in nested_sources:
            if k in src and src[k] is not None:
                cfg[k] = src[k]
                added += 1
                break
    return added


def _hoist_hidden_size(cfg: dict) -> int:
    """Mirror ``hidden_size`` up to the top level for DeepSpeed Zero-3 auto sizing.

    DeepSpeed reads ``model.config.hidden_size`` to fill in ``auto`` placeholders
    for ``reduce_bucket_size`` / ``stage3_prefetch_bucket_size`` /
    ``stage3_param_persistence_threshold``. The composite QwenOmniSpeak config
    has no top-level ``hidden_size``, so we copy thinker.text_config.hidden_size.
    """
    if 'hidden_size' in cfg:
        return 0
    hs = (
        cfg.get('thinker_config', {})
           .get('thinker_config', {})
           .get('text_config', {})
           .get('hidden_size')
    )
    if hs is not None:
        cfg['hidden_size'] = hs
        return 1
    return 0


def patch_config_file(path: str, dry_run: bool = False) -> int:
    if not os.path.isfile(path):
        print(f"[SKIP] {path} (not found)")
        return 0
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    added = _walk(cfg) + _hoist_token_ids(cfg) + _hoist_hidden_size(cfg)
    if added == 0:
        print(f"[OK]   {path}: nothing to patch")
        return 0
    if dry_run:
        print(f"[DRY]  {path}: would add {added} field(s)")
        return added
    backup = path + ".bak"
    if not os.path.exists(backup):
        with open(backup, "w", encoding="utf-8") as f:
            json.dump(json.load(open(path)), f, ensure_ascii=False, indent=2)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)
    print(f"[FIX]  {path}: added {added} field(s) (backup: {backup})")
    return added


def main() -> int:
    parser = argparse.ArgumentParser()
    default_paths = [
        f"../models/qwen_omnispeak_{v}/config.json"
        for v in ("a", "b", "c", "d", "e", "f")
    ]
    parser.add_argument("paths", nargs="*", default=default_paths,
                        help="config.json files to patch (default: all variants)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    total = 0
    for p in args.paths:
        total += patch_config_file(p, dry_run=args.dry_run)
    print(f"\nTOTAL: {total} fields {'would be ' if args.dry_run else ''}patched.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Public fail-fast checks for the EOS/history WDS warm-start recipe."""
from __future__ import annotations

import glob
import json
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'custom'))

from dataset import BridgeAlignmentPreprocessor, TASK_S2S_TRANSLATION  # noqa: E402
from task_prompts import s2s_task_prompt  # noqa: E402


def require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f'required environment variable is unset: {name}')
    return value


def assert_float(name: str, expected: float) -> None:
    actual = float(require_env(name))
    if abs(actual - expected) > 1e-12:
        raise AssertionError(f'{name}: expected {expected}, got {actual}')


def edge_rows(path: str):
    with open(path, 'rb') as handle:
        first = handle.readline()
        handle.seek(0, os.SEEK_END)
        end = handle.tell()
        pos = max(0, end - 2)
        while pos > 0:
            handle.seek(pos)
            if handle.read(1) == b'\n':
                break
            pos -= 1
        handle.seek(pos + 1 if pos > 0 else 0)
        last = handle.readline()
    for raw in (first, last):
        if raw.strip():
            yield json.loads(raw)


def validate_prompts() -> None:
    paths = sorted(glob.glob(require_env('BRIDGE_ALIGNMENT_DATA')))
    if not paths:
        raise FileNotFoundError('BRIDGE_ALIGNMENT_DATA did not match any manifest')
    preprocessor = BridgeAlignmentPreprocessor(TASK_S2S_TRANSLATION)
    seen = {}
    for path in paths:
        for row in edge_rows(path):
            target_lang = str(row.get('target_lang') or '').lower()
            if target_lang in seen:
                continue
            mapped = preprocessor.preprocess(row)
            expected = s2s_task_prompt(target_lang)
            actual = mapped['messages'][0]
            if actual != {'role': 'system', 'content': expected}:
                raise AssertionError(
                    f'wrong {target_lang} prompt: expected {expected!r}, got {actual!r}')
            if mapped.get('task_prompt') != '':
                raise AssertionError('legacy task_prompt survived normalization')
            seen[target_lang] = expected
        if set(seen) == {'en', 'zh'}:
            break
    if set(seen) != {'en', 'zh'}:
        raise AssertionError(f'expected en/zh targets, found {sorted(seen)}')
    for language in ('en', 'zh'):
        print(f'[wds-eos-history-preflight] target={language} prompt={seen[language]}')


def main() -> None:
    model_path = Path(require_env('MODEL_PATH'))
    if not model_path.is_dir():
        raise FileNotFoundError(f'MODEL_PATH does not exist: {model_path}')
    if os.environ.get('RESUME_FROM'):
        raise AssertionError('use MODEL_PATH warm-start; do not restore the old optimizer')
    if require_env('TALKER_TRAINING_SCOPE') != 'whole':
        raise AssertionError('TALKER_TRAINING_SCOPE must be whole')
    if require_env('UNFREEZE_THINKER').lower() not in {'1', 'true'}:
        raise AssertionError('UNFREEZE_THINKER must be enabled for this recipe')
    if require_env('BRIDGE_ALIGNMENT_NORMALIZE_S2S_PROMPT').lower() != 'true':
        raise AssertionError('prompt normalization must be enabled')
    assert_float('CODEC_EOS_LOSS_WEIGHT', 10.0)
    assert_float('CODEC_HISTORY_CORRUPTION_PROB', 0.05)
    validate_prompts()
    print('[wds-eos-history-preflight] checkpoint/loss/prompt: OK')


if __name__ == '__main__':
    main()

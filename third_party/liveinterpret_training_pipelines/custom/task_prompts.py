"""Canonical task prompts shared by S2S data construction and loading."""
from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional


_TARGET_LANGUAGE_NAMES = {
    'en': 'English',
    'eng': 'English',
    'english': 'English',
    'zh': 'Chinese',
    'zho': 'Chinese',
    'cmn': 'Chinese',
    'chinese': 'Chinese',
    'zh-cn': 'Chinese',
    'zh-tw': 'Chinese',
}


def normalize_language_code(language: Optional[str]) -> str:
    """Normalize language labels used by the bidirectional Common Voice data."""
    value = (language or '').strip().lower().replace('_', '-')
    if value.startswith('en-'):
        return 'en'
    if value.startswith('zh-'):
        return 'zh'
    if value in {'eng', 'english'}:
        return 'en'
    if value in {'zho', 'cmn', 'chinese'}:
        return 'zh'
    return value


def s2s_task_prompt(target_lang: Optional[str]) -> str:
    """Return the canonical speech-to-speech translation prompt."""
    normalized = normalize_language_code(target_lang)
    language_name = _TARGET_LANGUAGE_NAMES.get(normalized)
    if language_name is None:
        raise ValueError(
            f'Unsupported S2S target_lang={target_lang!r}; expected English or Chinese.')
    return (
        'Task: Listen to the input speech, translate it, and synthesize '
        f'the translation as {language_name} speech.')


def set_system_prompt(
    messages: Optional[List[Dict[str, Any]]], prompt: str
) -> List[Dict[str, Any]]:
    """Return a copy whose first system turn is exactly ``prompt``."""
    result = copy.deepcopy(messages or [])
    for message in result:
        if message.get('role') == 'system':
            message['content'] = prompt
            return result
    result.insert(0, {'role': 'system', 'content': prompt})
    return result


def merge_task_prompt(
    messages: Optional[List[Dict[str, Any]]], task_prompt: Optional[str]
) -> List[Dict[str, Any]]:
    """Merge a legacy ``task_prompt`` without duplicating identical content."""
    result = copy.deepcopy(messages or [])
    prompt = (task_prompt or '').strip()
    if not prompt:
        return result

    if result and result[0].get('role') == 'system':
        existing = (result[0].get('content') or '').strip()
        existing_lines = {line.strip() for line in existing.splitlines() if line.strip()}
        if prompt == existing or prompt in existing_lines:
            return result
        result[0]['content'] = f'{existing}\n{prompt}'.strip()
    else:
        result.insert(0, {'role': 'system', 'content': prompt})
    return result

"""M1a: Tokenizer compatibility check between thinker and upstream Qwen3-TTS.

The V1 plan (docs/ideas/talker_upstream_experiment_plan_v2.md) sends
``thinker_hidden + thinker.embed_tokens(input_id)`` to the talker's text channel,
then through ``talker.text_projection``. Two things matter for correctness:

1. Whether ``thinker.embed_tokens.weight`` and ``talker.text_embedding.weight`` come
   from the same word embedding matrix (same vocab id -> same semantic vector).
2. Whether the tokenization of a given string is identical, so that ``input_id``
   passed to thinker and the ``target_text_id`` passed to talker's text_projection
   during training refer to the same tokens.

This script reports:
 - vocab sizes on both sides
 - vocab intersection / diff counts
 - key special-token id agreement
 - a small sample string tokenization diff

Usage::

    python scripts/check_tokenizer_compat.py \
        --thinker-dir models/qwen2_5_3b_omnispeak_a \
        --talker-dir  project/Qwen3-TTS-12Hz-1.7B-Base
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, Set


def _load_vocab(path: str) -> Dict[str, int]:
    """Load a HF ``vocab.json`` — a plain mapping ``token -> id``."""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def _load_added(path: str) -> Dict[str, int]:
    if not os.path.exists(path):
        return {}
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def _load_special(path: str) -> Dict[str, str]:
    if not os.path.exists(path):
        return {}
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def _report_vocab_diff(name_a: str, vocab_a: Dict[str, int],
                       name_b: str, vocab_b: Dict[str, int]) -> None:
    keys_a: Set[str] = set(vocab_a.keys())
    keys_b: Set[str] = set(vocab_b.keys())
    common = keys_a & keys_b
    only_a = keys_a - keys_b
    only_b = keys_b - keys_a

    print(f'\n=== Vocab overlap: {name_a} vs {name_b} ===')
    print(f'  |{name_a}|   = {len(vocab_a):>7d}')
    print(f'  |{name_b}|   = {len(vocab_b):>7d}')
    print(f'  common       = {len(common):>7d}  ({len(common) / max(len(keys_a), 1) * 100:.2f}% of {name_a})')
    print(f'  only in {name_a:<8s} = {len(only_a):>7d}')
    print(f'  only in {name_b:<8s} = {len(only_b):>7d}')

    # Check id agreement on common tokens
    id_match = 0
    id_mismatch = 0
    mismatch_examples = []
    for tok in common:
        if vocab_a[tok] == vocab_b[tok]:
            id_match += 1
        else:
            id_mismatch += 1
            if len(mismatch_examples) < 10:
                mismatch_examples.append((tok, vocab_a[tok], vocab_b[tok]))
    print(f'  ids match    = {id_match:>7d}')
    print(f'  ids differ   = {id_mismatch:>7d}')
    if mismatch_examples:
        print(f'  first 10 id-mismatch examples:')
        for tok, ida, idb in mismatch_examples:
            print(f'    {tok!r:>20s}: {name_a}={ida}  {name_b}={idb}')

    if only_a and len(only_a) <= 30:
        print(f'  tokens only in {name_a}: {sorted(only_a)}')
    elif only_a:
        print(f'  first 15 tokens only in {name_a}: {sorted(only_a)[:15]}')

    if only_b and len(only_b) <= 30:
        print(f'  tokens only in {name_b}: {sorted(only_b)}')
    elif only_b:
        print(f'  first 15 tokens only in {name_b}: {sorted(only_b)[:15]}')


def _report_special(name_a: str, thinker_dir: str,
                    name_b: str, talker_dir: str) -> None:
    print(f'\n=== Special tokens & tts-specific ids ===')
    for side, d in [(name_a, thinker_dir), (name_b, talker_dir)]:
        cfg = os.path.join(d, 'tokenizer_config.json')
        if os.path.exists(cfg):
            with open(cfg, 'r', encoding='utf-8') as f:
                c = json.load(f)
            print(f'\n  [{side}] tokenizer_config.json highlights:')
            for k in ('bos_token', 'eos_token', 'pad_token', 'unk_token',
                      'model_max_length', 'chat_template', 'tokenizer_class'):
                v = c.get(k)
                if isinstance(v, dict):
                    v = v.get('content', v)
                if isinstance(v, str) and len(v) > 80:
                    v = v[:60] + '...(truncated)'
                if v is not None:
                    print(f'    {k}: {v}')
        added = _load_added(os.path.join(d, 'added_tokens.json'))
        if added:
            print(f'    added_tokens: {len(added)} entries')
            tts_related = {k: v for k, v in added.items()
                           if any(s in k for s in ('tts', 'audio', 'im_start', 'im_end', 'endoftext'))}
            for k, v in sorted(tts_related.items(), key=lambda x: x[1])[:20]:
                print(f'      {k!r:>40s} -> {v}')


def _tokenize_diff(name_a: str, thinker_dir: str,
                   name_b: str, talker_dir: str) -> None:
    # Try loading with AutoTokenizer; fall back to just reporting mismatch if it fails
    print(f'\n=== Sample tokenization diff ===')
    try:
        from transformers import AutoTokenizer
    except Exception as e:
        print(f'  transformers not importable: {e}')
        return

    samples = [
        '你好',
        '我叫通义千问，是阿里云的开源大模型。',
        'Hello, how are you today?',
        '<|im_start|>assistant\n<|tts|>你好<|im_end|>',
        '这是一段测试语音的文本内容。',
    ]

    try:
        tok_a = AutoTokenizer.from_pretrained(thinker_dir, trust_remote_code=False)
        tok_b = AutoTokenizer.from_pretrained(talker_dir, trust_remote_code=False)
    except Exception as e:
        print(f'  [WARN] AutoTokenizer failed: {e}')
        return

    for s in samples:
        try:
            ids_a = tok_a(s, add_special_tokens=False)['input_ids']
            ids_b = tok_b(s, add_special_tokens=False)['input_ids']
            match = '✔' if ids_a == ids_b else '✘'
            print(f'  {match}  {s!r}')
            print(f'      {name_a}: {ids_a}')
            print(f'      {name_b}: {ids_b}')
        except Exception as e:
            print(f'  [WARN] tokenize {s!r} failed: {e}')


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--thinker-dir', required=True,
                        help='Path to the omnispeak model dir whose thinker '
                             'tokenizer we want to inspect '
                             '(e.g. models/qwen2_5_3b_omnispeak_a).')
    parser.add_argument('--talker-dir', required=True,
                        help='Path to the upstream Qwen3-TTS-12Hz-1.7B-Base dir.')
    args = parser.parse_args()

    print(f'thinker dir : {args.thinker_dir}')
    print(f'talker  dir : {args.talker_dir}')

    v_thk = _load_vocab(os.path.join(args.thinker_dir, 'vocab.json'))
    v_tlk = _load_vocab(os.path.join(args.talker_dir, 'vocab.json'))

    _report_vocab_diff('thinker', v_thk, 'talker', v_tlk)
    _report_special('thinker', args.thinker_dir, 'talker', args.talker_dir)
    _tokenize_diff('thinker', args.thinker_dir, 'talker', args.talker_dir)

    return 0


if __name__ == '__main__':
    sys.exit(main())

#!/usr/bin/env python3
"""Regression tests for bidirectional S2S prompt construction."""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'custom'))

from dataset import BridgeAlignmentPreprocessor, TASK_S2S_TRANSLATION
from task_prompts import merge_task_prompt, s2s_task_prompt, set_system_prompt
from inference.s2s_infer import (
    _default_prompt,
    _set_inference_system_prompt,
    _val_messages,
)
from inference.eval_batch4_autoregressive import load_samples as load_batch4_ar_samples
from inference.eval_s2s_bridge_add import load_samples as load_bridge_samples


class S2SPromptPipelineTest(unittest.TestCase):
    def test_directional_prompts(self):
        self.assertIn('English speech', s2s_task_prompt('en'))
        self.assertIn('Chinese speech', s2s_task_prompt('zh'))
        self.assertEqual(s2s_task_prompt('en-US'), s2s_task_prompt('en'))
        self.assertEqual(s2s_task_prompt('zh_CN'), s2s_task_prompt('zh'))

    def test_legacy_merge_is_idempotent(self):
        prompt = s2s_task_prompt('zh')
        messages = [{'role': 'system', 'content': prompt}, {'role': 'user', 'content': '<audio>'}]
        merged = merge_task_prompt(messages, prompt)
        self.assertEqual(merged[0]['content'], prompt)
        self.assertEqual(merged[0]['content'].count(prompt), 1)

    def test_repair_replaces_wrong_direction_without_mutating_input(self):
        chinese = s2s_task_prompt('zh')
        english = s2s_task_prompt('en')
        messages = [{'role': 'system', 'content': chinese}, {'role': 'user', 'content': '<audio>'}]
        repaired = set_system_prompt(messages, english)
        self.assertEqual(messages[0]['content'], chinese)
        self.assertEqual(repaired[0]['content'], english)

    def test_wds_loader_can_normalize_legacy_short_prompt(self):
        row = {
            'task': TASK_S2S_TRANSLATION,
            'target_lang': 'en',
            'task_prompt': 'Task: Speech-to-speech translation; target language is English.',
            'messages': [{'role': 'user', 'content': '<audio>'}],
            'audios': ['wds://example.invalid/shard.tar?offset=0&length=1&format=wav'],
            'target_codec_codes': [[1] * 16],
            'target_text': 'Hello.',
        }
        with mock.patch.dict(os.environ, {
                'BRIDGE_ALIGNMENT_NORMALIZE_S2S_PROMPT': 'true'}):
            mapped = BridgeAlignmentPreprocessor(TASK_S2S_TRANSLATION).preprocess(row)
        self.assertEqual(mapped['messages'][0]['content'], s2s_task_prompt('en'))
        self.assertNotIn('target language is English.', mapped['messages'][0]['content'])
        self.assertEqual(mapped['task_prompt'], '')

    def test_inference_uses_exact_training_prompt_for_both_directions(self):
        for target_lang in ('en', 'zh'):
            expected = s2s_task_prompt(target_lang)
            self.assertEqual(_default_prompt(target_lang), expected)
            legacy_row = {
                'messages': [
                    {'role': 'system', 'content': 'Task: legacy mismatched prompt.'},
                    {'role': 'user', 'content': '<audio>'},
                    {'role': 'assistant', 'content': 'ground truth'},
                ],
            }
            messages = _set_inference_system_prompt(
                _val_messages(legacy_row, '/tmp/source.wav'), expected)
            self.assertEqual(
                messages[0],
                {'role': 'system', 'content': [{'type': 'text', 'text': expected}]})
            self.assertFalse(any(m.get('role') == 'assistant' for m in messages))

    def test_legacy_evaluation_loaders_use_canonical_directional_prompt(self):
        rows = [
            {
                'target_lang': target_lang,
                'task_prompt': 'Task: legacy prompt that must not be reused.',
                'messages': [{'role': 'user', 'content': '<audio>'}],
                'audios': [f'/tmp/{target_lang}.wav'],
                'target_text': 'placeholder',
            }
            for target_lang in ('en', 'zh')
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            val_path = Path(tmpdir) / 'val.jsonl'
            val_path.write_text(
                ''.join(json.dumps(row) + '\n' for row in rows),
                encoding='utf-8')
            for loader in (load_batch4_ar_samples, load_bridge_samples):
                samples = loader(val_path, n=2)
                self.assertEqual(
                    [sample['task_prompt'] for sample in samples],
                    [s2s_task_prompt('en'), s2s_task_prompt('zh')])


if __name__ == '__main__':
    unittest.main()

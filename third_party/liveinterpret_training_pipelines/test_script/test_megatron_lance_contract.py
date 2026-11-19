#!/usr/bin/env python3
"""Regression tests for the Lance-to-Megatron feature contract."""
from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'custom'))

from template import (  # noqa: E402
    QwenOmniSpeakBridgeAlignmentTemplate,
    _build_codec_control_prefix,
    _build_precomputed_source_mel_fields,
    _media_inputs_from_mel_items,
    _nonempty_vector,
    _pad_batch_input_features,
    _silence_placeholders_for_mel,
    attach_native_mel_tensors_to_rows,
)
from qwen_omnispeak.modeling_qwen_omnispeak import (  # noqa: E402
    apply_speaker_prompt_to_codec_emb,
)


def _codec_config(**overrides):
    cfg = {
        'codec_bos_id': 21,
        'codec_eos_id': 22,
        'codec_pad_id': 20,
        'codec_think_id': 23,
        'codec_nothink_id': 24,
        'codec_think_bos_id': 25,
        'codec_think_eos_id': 26,
        'codec_language_id': {'english': 27},
        'num_residual_groups': 15,
        'residual_vocab_size': 2048,
        'codec_mask_id': 19,
        'full_add_layout': False,
        'use_upstream_talker_input': False,
    }
    cfg.update(overrides)
    return cfg


class MegatronLanceContractTest(unittest.TestCase):
    def test_nonempty_vector_does_not_truth_test_numpy(self):
        self.assertFalse(_nonempty_vector(None))
        self.assertFalse(_nonempty_vector(''))
        self.assertFalse(_nonempty_vector([]))
        self.assertFalse(_nonempty_vector(np.array([])))
        self.assertTrue(_nonempty_vector('lance://11/codes'))
        self.assertTrue(_nonempty_vector([[1] * 16]))
        self.assertTrue(_nonempty_vector(np.arange(32, dtype=np.int16).reshape(2, 16)))
        # Old ``value == ''`` raised here; encode/speaker path must not.
        self.assertTrue(_nonempty_vector(np.ones(2048, dtype=np.float32)))
        self.assertTrue(_nonempty_vector(torch.ones(2, 16)))

    def test_recovers_codec_fields_from_numpy_codes(self):
        row = {
            'labels': [-100, -100, 7],
            '_extra_kwargs': {
                'target_codec_codes': np.arange(48, dtype=np.int16).reshape(3, 16),
                'target_lang': 'en',
                'task': 'speech_to_speech_translation',
            },
        }
        template = SimpleNamespace(config=SimpleNamespace(use_speaker_prompt=False))
        with mock.patch.dict(
                os.environ, {'QWEN_OMNISPEAK_USE_SPEAKER_PROMPT': 'false'}):
            QwenOmniSpeakBridgeAlignmentTemplate._ensure_codec_fields(
                template, [row], _codec_config())
        self.assertEqual(row['codec_labels'].shape, (9,))
        self.assertEqual(row['codec_residual_labels'].shape, (9, 15))

    def test_recovers_codec_fields_from_precomputed_codes(self):
        row = {
            'labels': [-100, -100, 7],
            '_extra_kwargs': {
                'target_codec_codes': [[i] * 16 for i in range(3)],
                'target_lang': 'en',
                'task': 'speech_to_speech_translation',
            },
        }
        template = SimpleNamespace(config=SimpleNamespace(use_speaker_prompt=False))

        with mock.patch.dict(
                os.environ, {'QWEN_OMNISPEAK_USE_SPEAKER_PROMPT': 'false'}):
            QwenOmniSpeakBridgeAlignmentTemplate._ensure_codec_fields(
                template, [row], _codec_config())

        self.assertEqual(row['codec_labels'].shape, (9,))
        self.assertEqual(row['codec_residual_labels'].shape, (9, 15))
        self.assertEqual(row['codec_input_ids'].shape, (9,))
        self.assertTrue(row['run_thinker'])

    def test_recovery_uses_same_full_add_prefix_as_encode(self):
        cfg = _codec_config(
            full_add_layout=True,
            use_upstream_talker_input=True,
        )
        prefix, slot = _build_codec_control_prefix(
            cfg,
            labels=[-100, -100, 7],
            target_lang='en',
            with_speaker=False,
        )
        self.assertIsNone(slot)
        self.assertEqual(prefix[:2], [cfg['codec_mask_id']] * 2)
        self.assertEqual(prefix[-2:], [cfg['codec_pad_id'], cfg['codec_bos_id']])

    def test_silence_placeholder_is_length_one_not_frame_budget(self):
        placeholders = _silence_placeholders_for_mel([
            {'frames': 696, 'flat': [0.0], 'mask': [1]},
            {'frames': 192, 'flat': [0.0], 'mask': [1]},
        ])
        self.assertEqual(len(placeholders), 2)
        for audio in placeholders:
            self.assertIsInstance(audio, np.ndarray)
            self.assertEqual(audio.shape, (1,))
            self.assertEqual(audio.dtype, np.float32)

    def test_numpy_mel_flat_matches_python_list_tensor(self):
        t = 4
        py_flat = [float(i) for i in range(128 * t)]
        np_flat = np.arange(128 * t, dtype=np.float32)
        py_pack = {
            'flat': py_flat, 'frames': t, 'mask': [1] * t, 'feature_size': 128,
        }
        np_pack = {
            'flat': np_flat, 'frames': t,
            'mask': np.ones(t, dtype=np.int8), 'feature_size': 128,
        }
        from_list = _build_precomputed_source_mel_fields([py_pack], torch.float32)
        from_np = _build_precomputed_source_mel_fields([np_pack], torch.float32)
        self.assertTrue(torch.equal(
            from_list['input_features'], from_np['input_features']))
        self.assertTrue(torch.equal(
            from_list['feature_attention_mask'],
            from_np['feature_attention_mask']))

    def test_batch_mel_attach_matches_per_row_native_tensors(self):
        short_t, long_t = 3, 7
        rows = [
            {
                'source_input_features': {
                    'flat': [float(i) for i in range(128 * short_t)],
                    'frames': short_t,
                    'mask': [1] * short_t,
                },
            },
            {
                'source_input_features': {
                    'flat': [float(i + 1) for i in range(128 * long_t)],
                    'frames': long_t,
                    'mask': [1] * long_t,
                },
            },
        ]
        per_row = [
            _build_precomputed_source_mel_fields(
                [row['source_input_features']], torch.float32)
            for row in rows
        ]
        attach_native_mel_tensors_to_rows(rows, torch.float32)
        for row, expected in zip(rows, per_row):
            packed = row['source_input_features']
            self.assertEqual(int(packed['frames']), expected['_mel_frame_lens'][0])
            self.assertTrue(torch.equal(
                packed['input_features'], expected['input_features']))
            self.assertTrue(torch.equal(
                packed['feature_attention_mask'], expected['feature_attention_mask']))
            reused = _media_inputs_from_mel_items([packed], torch.float32)
            self.assertTrue(torch.equal(
                reused['input_features'], expected['input_features']))
            self.assertEqual(reused['_mel_frame_lens'], expected['_mel_frame_lens'])

        packed_items = [row['source_input_features'] for row in rows]
        rebuilt = _build_precomputed_source_mel_fields(
            [
                {
                    'flat': [float(i) for i in range(128 * short_t)],
                    'frames': short_t,
                    'mask': [1] * short_t,
                },
                {
                    'flat': [float(i + 1) for i in range(128 * long_t)],
                    'frames': long_t,
                    'mask': [1] * long_t,
                },
            ],
            torch.float32,
        )
        reused_batch = _media_inputs_from_mel_items(packed_items, torch.float32)
        self.assertEqual(reused_batch['_mel_frame_lens'], [short_t, long_t])
        self.assertEqual(rebuilt['_mel_frame_lens'], [short_t, long_t])
        self.assertTrue(torch.equal(
            reused_batch['input_features'], rebuilt['input_features']))
        self.assertTrue(torch.equal(
            reused_batch['feature_attention_mask'],
            rebuilt['feature_attention_mask']))

    def test_builds_megatron_input_features_from_lance_mel(self):
        frames = 5
        mel = {
            'flat': [float(i) for i in range(128 * frames)],
            'frames': frames,
            'mask': [1] * frames,
        }
        fields = _build_precomputed_source_mel_fields([mel], torch.float32)
        self.assertEqual(fields['input_features'].shape, (1, 128, frames))
        self.assertEqual(fields['feature_attention_mask'].shape, (1, frames))
        self.assertEqual(fields['_mel_frame_lens'], [frames])

    def test_pads_variable_length_mels_for_swift_concat(self):
        # Reproduce B-column crash: 192 vs 696 frames cannot torch.concat.
        short_t, long_t = 192, 696
        batch = [
            {
                'input_features': torch.ones(1, 128, short_t),
                'feature_attention_mask': torch.ones(1, short_t, dtype=torch.long),
            },
            {
                'input_features': torch.ones(1, 128, long_t) * 2,
                'feature_attention_mask': torch.ones(1, long_t, dtype=torch.long),
            },
        ]
        _pad_batch_input_features(batch)
        stacked = torch.concat([row['input_features'] for row in batch])
        masks = torch.concat([row['feature_attention_mask'] for row in batch])
        self.assertEqual(stacked.shape, (2, 128, long_t))
        self.assertEqual(masks.shape, (2, long_t))
        self.assertEqual(int(masks[0].sum()), short_t)
        self.assertEqual(int(masks[1].sum()), long_t)
        self.assertTrue(torch.equal(stacked[0, :, short_t:], torch.zeros(128, long_t - short_t)))

    def test_megatron_speaker_slot_injection_matches_deepspeed_helper(self):
        cfg = SimpleNamespace(
            use_speaker_prompt=True,
            speaker_embedding_dim=4,
        )
        codec_emb = torch.zeros(2, 6, 4)
        codec_emb[:, 2] = 7.0  # placeholder pad slot
        offline = [
            torch.tensor([1.0, 2.0, 3.0, 4.0]),
            torch.tensor([5.0, 6.0, 7.0, 8.0]),
        ]
        with mock.patch.dict(
                os.environ, {'QWEN_OMNISPEAK_USE_SPEAKER_PROMPT': 'true'}):
            out = apply_speaker_prompt_to_codec_emb(
                codec_emb,
                config=cfg,
                speaker_slot_index=torch.tensor([2, 2]),
                speaker_embedding=offline,
            )
        self.assertTrue(torch.equal(out[0, 2], offline[0]))
        self.assertTrue(torch.equal(out[1, 2], offline[1]))
        self.assertTrue(torch.equal(out[:, 0], torch.zeros(2, 4)))


if __name__ == '__main__':
    unittest.main()

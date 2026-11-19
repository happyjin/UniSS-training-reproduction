"""多卡跳批：超阈值的 batch 只用 1 行做 forward（乘 0），不能用整个 batch（2026-10-09 OOM 的原因）。"""
from __future__ import annotations

import pytest
import torch

pytest.importorskip('transformers')
from simuls2st_omni.integration.register import _guard_skip, _shrink_skipped_batch  # noqa: E402


def _batch(b=96, t=768, tt=575, n_audio=None):
    x = {
        'input_ids': torch.ones(b, t, dtype=torch.long),
        'attention_mask': torch.ones(b, t, dtype=torch.long),
        'text_labels': torch.full((b, t), -100, dtype=torch.long),
        'talker_input_ids': torch.ones(b, tt, dtype=torch.long),
        'talker_attention_mask': torch.ones(b, tt, dtype=torch.long),
        'talker_labels': torch.full((b, tt), -100, dtype=torch.long),
        'reply_start_positions': torch.arange(b),
        'reply_lengths': torch.arange(b) + 1,
        'task_type': 'tts_unpaired',
    }
    if n_audio is not None:
        x['audio_features'] = torch.zeros(n_audio, 128, 30)
        x['audio_mask'] = torch.ones(n_audio, 30, dtype=torch.long)
    return x


def test_oversized_batch_is_flagged_then_shrunk_to_one_row():
    cfg = {'max_talker_tokens_per_batch': 40960}
    x = _batch()
    assert _guard_skip(cfg, x['input_ids'], x['talker_input_ids']) is not None   # 96 x 575 = 55200
    y = _shrink_skipped_batch(x)
    for k, v in y.items():
        if torch.is_tensor(v):
            assert v.shape[0] == 1, k
            assert torch.equal(v, x[k][:1]), k
    assert y['talker_input_ids'].shape == (1, 575)            # 行内长度不变，padding 语义不变
    assert y['task_type'] == 'tts_unpaired'
    assert x['input_ids'].shape[0] == 96                        # 不改原 batch


def test_audio_rows_shrink_together_when_one_per_row():
    y = _shrink_skipped_batch(_batch(b=4, n_audio=4))
    assert y['audio_features'].shape[0] == 1 and y['audio_mask'].shape[0] == 1


def test_unknown_audio_layout_falls_back_to_full_batch():
    x = _batch(b=4, n_audio=7)
    assert _shrink_skipped_batch(x) is x


def test_single_row_untouched():
    x = _batch(b=1)
    assert _shrink_skipped_batch(x) is x

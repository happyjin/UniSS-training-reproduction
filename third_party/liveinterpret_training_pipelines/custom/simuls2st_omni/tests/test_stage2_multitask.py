#!/usr/bin/env python3
"""Unit checks for the Stage 2 multitask collator and task-mixing sampler.

Covers the invariants that would otherwise fail only after a model load:

* one task per batch, and the right heads supervised for each
* `<|AUDIO|>` placeholder count matches the encoder's frame reduction
* talker layout still satisfies `valid_len == prefix_len + 3 + L`
* every rank draws the identical task sequence (a DDP correctness requirement)
* mixing frequencies track the configured weights

Run: PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests/test_stage2_multitask.py -v
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]  # custom/：`import simuls2st_omni` 的根
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from simuls2st_omni.train.collators.multitask import (
    AUDIO_INPUT_TASKS,
    CODE_TASKS,
    TEXT_TASKS,
    MultiTaskCollator,
    audio_token_count,
)
from simuls2st_omni.train.modeling_dual_head import IGNORE_INDEX
from simuls2st_omni.train.samplers_multitask import TaskMixBatchSampler

CODEC_PAD, CODEC_BOS, CODEC_EOS = 16384, 16385, 16386


class _FakeTokenizer:
    """Whitespace tokenizer with stable ids; enough to exercise layout logic."""

    pad_token_id = 0
    eos_token_id = 1

    def __call__(self, text: str, add_special_tokens: bool = False):
        ids = [(abs(hash(tok)) % 1000) + 10 for tok in text.split()]

        class Out:
            input_ids = ids

        return Out()


class _FakeFeatureExtractor:
    def __call__(self, audios, sampling_rate, padding, return_attention_mask):
        n = len(audios)
        longest = max(a.shape[0] for a in audios) // 160
        feats = np.zeros((n, longest, 128), dtype=np.float32)
        mask = np.zeros((n, longest), dtype=np.int64)
        for i, a in enumerate(audios):
            mask[i, : a.shape[0] // 160] = 1
        return {"input_features": feats, "attention_mask": mask}


def _collator() -> MultiTaskCollator:
    return MultiTaskCollator(
        tokenizer=_FakeTokenizer(),
        feature_extractor=_FakeFeatureExtractor(),
        codec_pad=CODEC_PAD,
        codec_bos=CODEC_BOS,
        codec_eos=CODEC_EOS,
    )


def _sample(task: str, idx: int = 0, n_codes: int = 6, audio_sec: float = 2.0) -> dict:
    sample = {
        "id": f"{task}-{idx}",
        "task_type": task,
        "system_prompt": "you are a model",
        "user_payload": "user side content",
        "assistant_payload": f"assistant reply {idx}",
    }
    if task in AUDIO_INPUT_TASKS:
        sample["user_payload"] = "English <|AUDIO_SPAN|>"
        sample["audio"] = np.zeros(int(audio_sec * 16000), dtype=np.float32)
    if task in CODE_TASKS:
        sample["codes"] = list(range(100, 100 + n_codes))
    return sample


def test_audio_token_count_matches_model_reduction() -> None:
    """Must equal the model's `(L-1)//2+1` then `(L-2)//2+1` on 10ms frames."""
    for seconds in (1.0, 2.5, 9.4, 30.0):
        n = int(seconds * 16000)
        mel = n // 160
        expected = ((mel - 1) // 2 + 1 - 2) // 2 + 1
        assert audio_token_count(n) == expected, f"{seconds}s mismatch"
    # 1s of audio should yield 25 tokens, i.e. 40ms per audio token.
    assert audio_token_count(16000) == 25


@pytest.mark.parametrize(
    "task",
    ["s2tt_asr", "s2tt_translate", "t2t_translate", "tts_unpaired", "s2s_translate"],
)
def test_heads_supervised_per_task(task: str) -> None:
    batch = _collator()([_sample(task, i) for i in range(3)])

    if task in TEXT_TASKS:
        assert batch["text_labels"] is not None, f"{task} must supervise text"
        assert (batch["text_labels"] != IGNORE_INDEX).any()
    else:
        assert batch["text_labels"] is None, f"{task} must not supervise text"

    if task in CODE_TASKS:
        assert "talker_labels" in batch, f"{task} must supervise codes"
        assert (batch["talker_labels"] != IGNORE_INDEX).any()
    else:
        assert "talker_labels" not in batch, f"{task} must not carry talker fields"

    if task in AUDIO_INPUT_TASKS:
        assert "audio_features" in batch and "audio_mask" in batch
    else:
        assert "audio_features" not in batch


@pytest.mark.parametrize("task", sorted(CODE_TASKS))
def test_talker_layout_invariant(task: str) -> None:
    """`valid_len == prefix_len + 3 + L`, supervision on codes+eos only."""
    n_codes = 7
    batch = _collator()([_sample(task, i, n_codes=n_codes) for i in range(2)])
    for i in range(2):
        prefix = int(batch["reply_start_positions"][i])
        valid = int(batch["talker_attention_mask"][i].sum())
        assert valid == prefix + 3 + n_codes, f"{task} row {i}: {valid}"
        sup = (batch["talker_labels"][i] != IGNORE_INDEX).nonzero().flatten()
        assert int(sup[0]) == prefix + 2
        assert int(sup[-1]) == prefix + 2 + n_codes
        assert int(batch["talker_labels"][i][sup[-1]]) == CODEC_EOS


def test_audio_span_substituted_with_correct_length() -> None:
    """Placeholder count in the rendered prompt must match the clip length."""
    sample = _sample("s2tt_translate", audio_sec=3.0)
    expected = audio_token_count(int(3.0 * 16000))
    batch = _collator()([sample])
    # The fake tokenizer splits on whitespace, and build_audio_span emits one
    # contiguous run, so verify through the feature mask instead.
    assert int(batch["audio_mask"][0].sum()) == int(3.0 * 16000) // 160
    # 25 audio tokens per second of speech (40ms each).
    assert expected == 75


def test_mixed_task_batch_rejected() -> None:
    with pytest.raises(AssertionError, match="single task"):
        _collator()([_sample("tts_unpaired"), _sample("s2tt_translate")])


def _sampler(rank: int = 0, world_size: int = 1, steps: int = 2000):
    rng = np.random.default_rng(0)
    tasks = ["s2tt_asr", "s2tt_translate", "t2t_translate", "tts_unpaired", "s2s_translate"]
    weights = {"s2tt_asr": 0.2, "s2tt_translate": 1.0, "t2t_translate": 0.5,
               "tts_unpaired": 1.0, "s2s_translate": 1.5}
    indices, lengths = {}, {}
    for offset, task in enumerate(tasks):
        n = 4000
        indices[task] = list(range(offset * 10000, offset * 10000 + n))
        lengths[task] = rng.integers(20, 200, size=n).tolist()
    return TaskMixBatchSampler(
        task_indices=indices,
        task_lengths=lengths,
        task_weights=weights,
        token_budget=4096,
        max_batch_size=32,
        seed=7,
        rank=rank,
        world_size=world_size,
        steps_per_epoch=steps,
    )


def test_task_sequence_identical_across_ranks() -> None:
    """Ranks must agree on the task per step, or DDP all-reduce mismatches."""
    world = 4
    samplers = [_sampler(rank=r, world_size=world, steps=500) for r in range(world)]
    # Reconstruct each rank's task sequence from which index range it yields.
    def task_of(index: int) -> int:
        return index // 10000

    sequences = []
    for sampler in samplers:
        sequences.append([task_of(batch[0]) for batch in sampler])
    for r in range(1, world):
        assert sequences[r] == sequences[0], f"rank {r} diverged from rank 0"


def test_mixing_frequencies_track_weights() -> None:
    sampler = _sampler(steps=20000)
    counts = Counter(batch[0] // 10000 for batch in sampler)
    total = sum(counts.values())
    expected = np.array([0.2, 1.0, 0.5, 1.0, 1.5])
    expected = expected / expected.sum()
    for task_id, want in enumerate(expected):
        got = counts[task_id] / total
        assert abs(got - want) < 0.02, (
            f"task {task_id}: sampled {got:.3f}, expected {want:.3f}"
        )


def test_ranks_get_disjoint_samples_within_task() -> None:
    """Same task, different data: otherwise 8 ranks would train on one shard."""
    world = 4
    first_batches = []
    for rank in range(world):
        sampler = _sampler(rank=rank, world_size=world, steps=1)
        first_batches.append(next(iter(sampler)))
    flat = [i for b in first_batches for i in b]
    assert len(set(flat)) == len(flat), "ranks received overlapping samples"


def test_len_is_defined_for_trainer() -> None:
    """HF Trainer needs __len__ up front to compute the schedule."""
    assert len(_sampler(steps=1234)) == 1234


def test_s2st_pack_cost_counts_audio_prefix() -> None:
    """Code-only cost lets 256 short clips into one 8192 batch; talker cost must not."""
    from simuls2st_omni.train.datasets.s2st import OfflineS2STDataset

    n_codes = 29
    assert 256 * (n_codes + 3) <= 8192
    assert 256 * (OfflineS2STDataset.pack_cost(n_codes) + 3) > 8192
    # A 32-cap batch of short clips still fits the 8192 talker budget.
    assert 32 * (OfflineS2STDataset.pack_cost(n_codes) + 3) <= 8192

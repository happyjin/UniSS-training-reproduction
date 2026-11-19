"""Multi-task collator for Stage 2 (offline, whole-utterance).

Extends the Stage 1 TTS collator to the five task families the paper mixes:

    ASR    audio -> source text          (text head only)
    S2TT   audio -> target text          (text head only)
    MT     source text -> target text    (text head only)
    TTS    text -> codes                 (talker only, thinker frozen or LoRA)
    S2ST   audio -> target text + codes  (both heads)

A batch carries exactly one task: `compute_loss` reads a scalar `task_type`, and
label validity is judged per batch (a TTS batch leaves `text_labels` fully
IGNORE_INDEX and relies on `compute_causal_lm_loss` returning a zero that still
carries grad for DDP). Task mixing therefore happens between batches, in the
sampler.

Audio input path: `input_features` from the processor's feature extractor land on
`<|AUDIO|>` placeholder positions, whose count must equal the encoder's output
frame count, or the scatter in `_scatter_audio_features` silently mismatches. The
count is derived here with the same two-stage downsample the model uses.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import torch

from simuls2st_omni.train.modeling_dual_head import IGNORE_INDEX
from simuls2st_omni.train.prompt_formats import (
    TASK_S2S_TRANSLATE,
    TASK_S2TT_ASR,
    TASK_S2TT_TRANSLATE,
    TASK_T2T_TRANSLATE,
    TASK_TTS_PAIRED_CROSS_LANG,
    TASK_TTS_PAIRED_SAME_LANG,
    TASK_TTS_UNPAIRED,
    build_audio_span,
)

SAMPLE_RATE_16K = 16_000

# Tasks whose assistant target includes semantic codes (talker is supervised).
CODE_TASKS = frozenset(
    {
        TASK_TTS_UNPAIRED,
        TASK_TTS_PAIRED_SAME_LANG,
        TASK_TTS_PAIRED_CROSS_LANG,
        TASK_S2S_TRANSLATE,
    }
)
# Tasks whose user turn contains audio.
AUDIO_INPUT_TASKS = frozenset(
    {
        TASK_S2TT_ASR,
        TASK_S2TT_TRANSLATE,
        TASK_S2S_TRANSLATE,
        TASK_TTS_PAIRED_SAME_LANG,
        TASK_TTS_PAIRED_CROSS_LANG,
    }
)
# Tasks supervising the thinker's text head.
TEXT_TASKS = frozenset(
    {
        TASK_S2TT_ASR,
        TASK_S2TT_TRANSLATE,
        TASK_T2T_TRANSLATE,
        TASK_S2S_TRANSLATE,
    }
)


def feat_extract_output_length(mel_frames: int) -> int:
    """Encoder output frames for `mel_frames` of mel input.

    Identical to the agent's `_get_feat_extract_output_lengths`, kept here so
    training and inference cannot drift apart: two conv stages of stride 2.
    """
    length = (mel_frames - 1) // 2 + 1
    length = (length - 2) // 2 + 1
    return int(length)


def audio_token_count(n_samples: int) -> int:
    """Number of `<|AUDIO|>` placeholders for an `n_samples` 16kHz clip.

    Mirrors the model's own reduction: 10ms mel hop, then two conv stages of
    `(L-1)//2+1` and `(L-2)//2+1`. Getting this wrong misaligns the scatter of
    encoder outputs onto placeholder positions.
    """
    mel_frames = n_samples // 160  # 10ms hop
    length = (mel_frames - 1) // 2 + 1
    length = (length - 2) // 2 + 1
    return max(1, int(length))


@dataclass
class MultiTaskCollator:
    """Collate one task's samples into a Stage 2 training batch."""

    tokenizer: Any
    feature_extractor: Any
    codec_pad: int
    codec_bos: int
    codec_eos: int

    def _render(self, sample: dict) -> tuple[str, str]:
        """Return (prefix_text, reply_text) for the chat-formatted sample."""
        prefix = (
            f"<|im_start|>system\n{sample['system_prompt']}<|im_end|>\n"
            f"<|im_start|>user\n{sample['user_payload']}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        return prefix, f"{sample['assistant_payload']}<|im_end|>\n"

    def __call__(self, samples: list[dict]) -> dict:
        assert samples, "empty batch"
        task = samples[0]["task_type"]
        assert all(s["task_type"] == task for s in samples), (
            "a batch must carry a single task; mixing happens in the sampler"
        )
        has_audio = task in AUDIO_INPUT_TASKS
        has_codes = task in CODE_TASKS
        supervise_text = task in TEXT_TASKS

        batch_size = len(samples)
        thinker_ids: list[list[int]] = []
        reply_starts: list[int] = []
        reply_lens: list[int] = []
        talker_ids: list[list[int]] = []
        talker_labels: list[list[int]] = []
        audios: list[np.ndarray] = []

        # Placeholder counts must come from the feature extractor's own mask, not
        # from `len(audio) // hop`: the two disagree by a frame on some lengths,
        # and the scatter of encoder outputs onto `<|AUDIO|>` positions then
        # fails with a shape mismatch. So features are computed up front and the
        # real frame counts drive the span substitution below.
        audio_feats = None
        audio_token_counts: list[int] = []
        if has_audio:
            for sample in samples:
                audio = np.asarray(sample["audio"], dtype=np.float32).reshape(-1)
                assert audio.size > 0, f"{sample['id']}: empty audio"
                audios.append(audio)
            audio_feats = self.feature_extractor(
                audios,
                sampling_rate=SAMPLE_RATE_16K,
                padding="longest",
                return_attention_mask=True,
            )
            # WhisperFeatureExtractor can return a mask one frame longer than
            # the mel it produced (reproduced at mel=200: features [B, 128, 200]
            # with mask [B, 201]). `get_audio_features` indexes the features by
            # the mask, so the extra column makes it read past the end. Trim the
            # mask to the mel's own frame count; the dropped column is padding
            # by construction.
            #
            # The time axis is located by matching it against the mask rather
            # than assumed to be last: real features are [B, mel_bins, frames]
            # but that ordering is a property of the extractor, not a contract.
            features = np.asarray(audio_feats["input_features"])
            mask = np.asarray(audio_feats["attention_mask"])
            time_axis = next(
                (
                    axis
                    for axis in range(features.ndim - 1, 0, -1)
                    if abs(features.shape[axis] - mask.shape[-1]) <= 1
                ),
                None,
            )
            assert time_axis is not None, (
                f"no feature axis matches mask length {mask.shape[-1]} "
                f"in features {features.shape}"
            )
            n_frames = min(features.shape[time_axis], mask.shape[-1])
            features = features.take(indices=range(n_frames), axis=time_axis)
            mask = mask[..., :n_frames]
            audio_feats = {"input_features": features, "attention_mask": mask}
            mel_lens = mask.sum(axis=1)
            audio_token_counts = [
                max(1, int(feat_extract_output_length(int(n)))) for n in mel_lens
            ]

        for position, sample in enumerate(samples):
            if has_audio:
                sample = dict(sample)
                sample["user_payload"] = sample["user_payload"].replace(
                    "<|AUDIO_SPAN|>", build_audio_span(audio_token_counts[position])
                )

            prefix, reply = self._render(sample)
            prefix_ids = self.tokenizer(prefix, add_special_tokens=False).input_ids
            reply_ids = self.tokenizer(reply, add_special_tokens=False).input_ids
            assert reply_ids, f"{sample['id']}: empty assistant reply"

            thinker_ids.append(list(prefix_ids) + list(reply_ids))
            reply_starts.append(len(prefix_ids))
            reply_lens.append(len(reply_ids))

            if has_codes:
                codes = np.asarray(sample["codes"], dtype=np.int64).reshape(-1)
                assert codes.size > 0, f"{sample['id']}: empty codes"
                prefix_len = len(prefix_ids)
                teacher = codes.tolist() + [self.codec_eos]
                talker_ids.append([self.codec_pad] * (prefix_len + 2) + teacher)
                talker_labels.append(
                    [IGNORE_INDEX] * (prefix_len + 2) + teacher
                )

        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id
        max_thinker = max(len(ids) for ids in thinker_ids)

        input_ids = torch.full((batch_size, max_thinker), pad_id, dtype=torch.long)
        attention_mask = torch.zeros((batch_size, max_thinker), dtype=torch.long)
        text_labels = torch.full(
            (batch_size, max_thinker), IGNORE_INDEX, dtype=torch.long
        )

        for i, ids in enumerate(thinker_ids):
            input_ids[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
            attention_mask[i, : len(ids)] = 1
            if supervise_text:
                start, length = reply_starts[i], reply_lens[i]
                text_labels[i, start : start + length] = torch.tensor(
                    ids[start : start + length], dtype=torch.long
                )

        batch: dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            # Passing None for unsupervised heads keeps the model from computing
            # a [B, T, 152k] logit tensor it would only discard.
            "text_labels": text_labels if supervise_text else None,
            "task_type": task,
            "ids": [s["id"] for s in samples],
        }

        if has_codes:
            max_talker = max(len(r) for r in talker_ids)
            t_ids = torch.full(
                (batch_size, max_talker), self.codec_pad, dtype=torch.long
            )
            t_mask = torch.zeros((batch_size, max_talker), dtype=torch.long)
            t_labels = torch.full(
                (batch_size, max_talker), IGNORE_INDEX, dtype=torch.long
            )
            for i, (row, labels) in enumerate(zip(talker_ids, talker_labels)):
                t_ids[i, : len(row)] = torch.tensor(row, dtype=torch.long)
                t_mask[i, : len(row)] = 1
                t_labels[i, : len(labels)] = torch.tensor(labels, dtype=torch.long)
            batch.update(
                talker_input_ids=t_ids,
                talker_attention_mask=t_mask,
                talker_labels=t_labels,
                reply_start_positions=torch.tensor(reply_starts, dtype=torch.long),
                reply_lengths=torch.tensor(reply_lens, dtype=torch.long),
            )

        if has_audio:
            # Reuse the features computed above; extracting twice would risk the
            # two copies disagreeing again.
            batch["audio_features"] = torch.from_numpy(
                np.asarray(audio_feats["input_features"])
            )
            batch["audio_mask"] = torch.from_numpy(
                np.asarray(audio_feats["attention_mask"])
            ).long()

        return batch

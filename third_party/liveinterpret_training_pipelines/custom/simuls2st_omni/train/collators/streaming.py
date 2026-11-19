"""Streaming (per-chunk) collator for Stage 3.

Drives `OmniTalkerQwenOmni.forward` through its
`talker_thinker_scatter_indices` path, which teacher-forces a chunk-concatenated
talker sequence:

    [mask_1][pad][bos][codes_1][eos] ... [mask_N][pad][bos][codes_N][eos]

Rather than materialising embeddings, the collator emits two index tensors that
`_build_talker_inputs_embeds_scatter` resolves on device:

* `talker_thinker_scatter_indices` -- `>= 0` gathers `thinker_cond` at that
  position and *replaces* the slot outright; `-1` falls through to the codec
  embedding path.
* `talker_text_cond_indices` -- `>= 0` gathers `thinker_cond` as additive text
  conditioning; `-2 / -3 / -4` select `text_bos / text_eos / text_pad`; `-1`
  means no conditioning. Sentinel values are fixed by
  `src/train/modeling_omni_talker.py:340-344`.

Per-position contract, for a chunk with thinker span `[t0, t1)`, reply span
`[r0, r1)`, `L` codes, and `base = r1 - r0 - 1`:

    position          scatter_idx      text_cond_idx
    mask region       t0 .. t1-1       -1 (unused; scatter wins)
    pad               -1               -2  (text_bos)
    bos               -1               r0
    codes[k], k<base  -1               r0 + 1 + k
    codes[k], k==base -1               -3  (text_eos)
    codes[k], k>base  -1               -4  (text_pad)
    eos               -1               tail rule evaluated at k = L

`L == 0` is the legitimate idle/wait shape and degrades to
`[mask][pad][bos][eos]`.

Verified elementwise against the inference construction by
`tests/test_streaming_collator_parity.py`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import torch

from simuls2st_omni.train.modeling_dual_head import IGNORE_INDEX

# Sentinels consumed by `_build_talker_inputs_embeds_scatter`.
TEXT_COND_NONE = -1
TEXT_COND_BOS = -2
TEXT_COND_EOS = -3
TEXT_COND_PAD = -4


@dataclass
class StreamingChunk:
    """One read/write step of a streaming trajectory.

    thinker_span: `[start, end)` of this chunk's tokens in the flattened thinker
        sequence (user turn + assistant reply + suffix).
    reply_span: `[start, end)` of the assistant reply within the same sequence.
    codes: semantic codes emitted during this chunk; empty for a wait step.
    """

    thinker_span: tuple[int, int]
    reply_span: tuple[int, int]
    codes: Sequence[int] = field(default_factory=list)


@dataclass
class StreamingCollator:
    """Collate streaming-trajectory samples into a talker teacher-forcing batch."""

    codec_pad: int
    codec_bos: int
    codec_eos: int
    codec_mask: int
    text_bos: int
    text_eos: int
    text_pad: int
    tokenizer: Any = None
    pad_token_id: int = 0
    feature_extractor: Any = None

    def _build_row(
        self, chunks: Sequence[StreamingChunk]
    ) -> tuple[list[int], list[int], list[int], list[int]]:
        """Return (talker_input_ids, labels, scatter_indices, text_cond_indices)."""
        ids: list[int] = []
        labels: list[int] = []
        scatter: list[int] = []
        text_cond: list[int] = []

        for chunk in chunks:
            t0, t1 = chunk.thinker_span
            r0, r1 = chunk.reply_span
            assert t1 > t0, f"empty thinker span {chunk.thinker_span}"
            assert r1 > r0, f"empty reply span {chunk.reply_span}"
            assert t0 <= r0 and r1 <= t1, (
                f"reply span {chunk.reply_span} outside thinker span {chunk.thinker_span}"
            )

            # mask region: one slot per thinker token, embedding replaced wholesale
            for pos in range(t0, t1):
                ids.append(self.codec_mask)
                labels.append(IGNORE_INDEX)
                scatter.append(pos)
                text_cond.append(TEXT_COND_NONE)

            codes = np.asarray(chunk.codes, dtype=np.int64).reshape(-1)
            n_codes = int(codes.shape[0])
            base = r1 - r0 - 1

            def tail_index(k: int) -> int:
                """Text conditioning for the k-th slot after bos."""
                if k < base:
                    return r0 + 1 + k
                if k == base:
                    return TEXT_COND_EOS
                return TEXT_COND_PAD

            # pad slot: codec_pad + text_bos
            ids.append(self.codec_pad)
            labels.append(IGNORE_INDEX)
            scatter.append(-1)
            text_cond.append(TEXT_COND_BOS)

            # bos slot: codec_bos + thinker_cond[r0]
            ids.append(self.codec_bos)
            labels.append(IGNORE_INDEX)
            scatter.append(-1)
            text_cond.append(r0)

            # code slots: supervised, conditioned on the reply tail
            for k in range(n_codes):
                ids.append(int(codes[k]))
                labels.append(int(codes[k]))
                scatter.append(-1)
                text_cond.append(tail_index(k))

            # eos slot: supervised, closes the chunk
            ids.append(self.codec_eos)
            labels.append(self.codec_eos)
            scatter.append(-1)
            text_cond.append(tail_index(n_codes))

        return ids, labels, scatter, text_cond

    def _audio_token_id(self) -> int:
        assert self.tokenizer is not None, "tokenizer is required to check audio placeholders"
        token_id = self.tokenizer.convert_tokens_to_ids("<|AUDIO|>")
        unk = getattr(self.tokenizer, "unk_token_id", None)
        assert token_id is not None and token_id != unk, "tokenizer has no <|AUDIO|> token"
        return int(token_id)

    def _extract_audio(self, audios: list[np.ndarray]) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
        """Mel features for one waveform per sample, plus encoder token counts.

        Same mask/mel trim as the Stage 2 collator: the Whisper extractor can
        return a mask one frame longer than the mel, and `get_audio_features`
        then reads past the end.
        """
        from simuls2st_omni.train.collators.multitask import feat_extract_output_length

        assert self.feature_extractor is not None, "feature_extractor is required for audio tasks"
        feats = self.feature_extractor(
            [np.asarray(audio, dtype=np.float32).reshape(-1) for audio in audios],
            sampling_rate=16_000,
            padding="longest",
            return_attention_mask=True,
        )
        features = np.asarray(feats["input_features"])
        mask = np.asarray(feats["attention_mask"])
        time_axis = next(
            (
                axis
                for axis in range(features.ndim - 1, 0, -1)
                if abs(features.shape[axis] - mask.shape[-1]) <= 1
            ),
            None,
        )
        assert time_axis is not None, (
            f"no feature axis matches mask length {mask.shape[-1]} in features {features.shape}"
        )
        n_frames = min(features.shape[time_axis], mask.shape[-1])
        features = np.take(features, indices=range(n_frames), axis=time_axis)
        mask = mask[..., :n_frames]
        counts = [
            max(1, int(feat_extract_output_length(int(n)))) for n in mask.sum(axis=1)
        ]
        return (
            torch.from_numpy(np.ascontiguousarray(features, dtype=np.float32)),
            torch.from_numpy(np.ascontiguousarray(mask, dtype=np.int64)),
            counts,
        )

    def __call__(self, samples: list[dict]) -> dict:
        batch_size = len(samples)
        assert batch_size > 0, "empty batch"
        task = samples[0]["task_type"]
        assert all(s["task_type"] == task for s in samples), (
            "a batch must carry a single task; mixing happens in the sampler"
        )
        supervise_talker = bool(samples[0].get("supervise_talker", True))
        assert all(bool(s.get("supervise_talker", True)) == supervise_talker for s in samples)

        rows = [self._build_row(s["chunks"]) for s in samples] if supervise_talker else None
        thinker_ids = [list(s["thinker_token_ids"]) for s in samples]

        max_thinker = max(len(ids) for ids in thinker_ids)
        pad_id = self.pad_token_id
        if self.tokenizer is not None and self.tokenizer.pad_token_id is not None:
            pad_id = self.tokenizer.pad_token_id

        input_ids = torch.full((batch_size, max_thinker), pad_id, dtype=torch.long)
        attention_mask = torch.zeros((batch_size, max_thinker), dtype=torch.long)
        text_labels = torch.full((batch_size, max_thinker), IGNORE_INDEX, dtype=torch.long)

        talker = None
        if supervise_talker:
            assert rows is not None
            max_talker = max(len(r[0]) for r in rows)
            t_ids = torch.full((batch_size, max_talker), self.codec_pad, dtype=torch.long)
            t_mask = torch.zeros((batch_size, max_talker), dtype=torch.long)
            t_labels = torch.full((batch_size, max_talker), IGNORE_INDEX, dtype=torch.long)
            # Padding slots keep scatter/text_cond at -1 so they resolve to a plain
            # codec embedding and never gather out of range.
            t_scatter = torch.full((batch_size, max_talker), -1, dtype=torch.long)
            t_text_cond = torch.full((batch_size, max_talker), TEXT_COND_NONE, dtype=torch.long)
            talker = (t_ids, t_mask, t_labels, t_scatter, t_text_cond)

        for i, sample in enumerate(samples):
            thinker = thinker_ids[i]
            input_ids[i, : len(thinker)] = torch.tensor(thinker, dtype=torch.long)
            attention_mask[i, : len(thinker)] = 1

            # Thinker text supervision covers each chunk's assistant reply.
            for chunk in sample["chunks"]:
                r0, r1 = chunk.reply_span
                text_labels[i, r0:r1] = torch.tensor(thinker[r0:r1], dtype=torch.long)

            if talker is not None:
                assert rows is not None
                ids, labels, scatter, text_cond = rows[i]
                t_ids, t_mask, t_labels, t_scatter, t_text_cond = talker
                n = len(ids)
                t_ids[i, :n] = torch.tensor(ids, dtype=torch.long)
                t_mask[i, :n] = 1
                t_labels[i, :n] = torch.tensor(labels, dtype=torch.long)
                t_scatter[i, :n] = torch.tensor(scatter, dtype=torch.long)
                t_text_cond[i, :n] = torch.tensor(text_cond, dtype=torch.long)
                assert int(t_scatter[i].max()) < len(thinker), (
                    "scatter index points past the thinker sequence"
                )

        batch = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "text_labels": text_labels,
            "task_type": task,
            "ids": [s["id"] for s in samples],
        }
        if talker is not None:
            t_ids, t_mask, t_labels, t_scatter, t_text_cond = talker
            batch.update(
                talker_input_ids=t_ids,
                talker_attention_mask=t_mask,
                talker_labels=t_labels,
                talker_thinker_scatter_indices=t_scatter,
                talker_text_cond_indices=t_text_cond,
            )

        chunked = [s.get("audio_chunks") for s in samples]
        if any(pieces is not None for pieces in chunked):
            assert all(pieces for pieces in chunked), f"{task}: mixed audio/text batch"
            # One feature row per chunk, sample-major, so masked_scatter fills
            # placeholders in the same order they appear in `input_ids`.
            flat = [piece for pieces in chunked for piece in pieces]
            features, feat_mask, counts = self._extract_audio(flat)
            audio_id = self._audio_token_id()
            offset = 0
            for i, ids in enumerate(thinker_ids):
                n_rows = len(chunked[i])
                expected = sum(counts[offset : offset + n_rows])
                offset += n_rows
                n_placeholders = sum(token == audio_id for token in ids)
                assert n_placeholders == expected, (
                    f"{samples[i]['id']}: {n_placeholders} audio placeholders, "
                    f"encoder produced {expected}"
                )
            batch["audio_features"] = features
            batch["audio_mask"] = feat_mask
        return batch

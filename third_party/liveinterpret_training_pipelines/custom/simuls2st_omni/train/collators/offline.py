"""Offline (whole-utterance) collator for Stage 1 / Stage 2.

Produces the field set consumed by `OmniTalkerQwenOmni.forward` via its
`reply_start_positions` / `reply_lengths` path.

Talker sequence layout, dictated by `_build_talker_inputs_embeds`
(`src/train/modeling_omni_talker.py:391-438`):

    index                       embedding source                 input_ids
    [0, prefix_len)             thinker_cond[0:prefix_len]       unused
    prefix_len                  text_bos + codec_pad             unused
    prefix_len + 1              first_reply_cond + codec_bos     unused
    [prefix_len+2, valid_len)   codec_embed + tail_cond          codes + eos

so `valid_len == prefix_len + 3 + L` for `L` codes. Only the last region is read
from `talker_input_ids`; the leading positions are filled with `codec_pad` purely
to keep the tensor well-formed.

Supervision (see the caveat below): with the causal shift applied by
`compute_causal_lm_loss`, `logits[k]` predicts `labels[k+1]`. The logit at
`prefix_len+1` (the codec_bos position) is therefore the one that must predict
`code_0`, which means `labels[prefix_len+2] = code_0`, and in general
`labels == talker_input_ids` across `[prefix_len+2, valid_len)` with
`IGNORE_INDEX` everywhere else.

CAVEAT: that supervision window is inferred, not documented. The inference path
never builds labels, so the golden-reference test for the streaming collator
cannot cover it either. Validate with the Stage 1 overfit run before trusting it
(see the spec's risk section).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from simuls2st_omni.train.modeling_dual_head import IGNORE_INDEX


@dataclass
class OfflineTTSCollator:
    """Collate unpaired-TTS samples into a talker teacher-forcing batch."""

    tokenizer: Any
    codec_pad: int
    codec_bos: int
    codec_eos: int

    def _thinker_ids(self, sample: dict) -> tuple[list[int], int, int]:
        """Return (token ids, reply_start, reply_len).

        `reply_start` indexes the first assistant *payload* token, matching the
        inference-side reply span, which begins after the assistant prefix and
        runs through the trailing `<|im_end|>\\n`.
        """
        prefix_text = (
            f"<|im_start|>system\n{sample['system_prompt']}<|im_end|>\n"
            f"<|im_start|>user\n{sample['user_payload']}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        reply_text = f"{sample['assistant_payload']}<|im_end|>\n"

        prefix_ids = self.tokenizer(prefix_text, add_special_tokens=False).input_ids
        reply_ids = self.tokenizer(reply_text, add_special_tokens=False).input_ids
        assert reply_ids, "assistant reply tokenized to nothing"
        return list(prefix_ids) + list(reply_ids), len(prefix_ids), len(reply_ids)

    def __call__(self, samples: list[dict]) -> dict:
        batch_size = len(samples)
        assert batch_size > 0, "empty batch"

        thinker_ids: list[list[int]] = []
        reply_starts: list[int] = []
        reply_lens: list[int] = []
        talker_ids: list[list[int]] = []
        talker_labels: list[list[int]] = []

        for sample in samples:
            ids, reply_start, reply_len = self._thinker_ids(sample)
            codes = np.asarray(sample["codes"], dtype=np.int64).reshape(-1)
            n_codes = int(codes.shape[0])

            # prefix_len must index into the thinker sequence, because the talker's
            # leading positions are gathered from thinker_cond[0:prefix_len].
            prefix_len = reply_start
            teacher = codes.tolist() + [self.codec_eos]
            row = [self.codec_pad] * (prefix_len + 2) + teacher
            labels = [IGNORE_INDEX] * (prefix_len + 2) + teacher

            assert len(row) == prefix_len + 3 + n_codes
            thinker_ids.append(ids)
            reply_starts.append(prefix_len)
            reply_lens.append(reply_len)
            talker_ids.append(row)
            talker_labels.append(labels)

        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id
        max_thinker = max(len(ids) for ids in thinker_ids)
        max_talker = max(len(row) for row in talker_ids)

        input_ids = torch.full((batch_size, max_thinker), pad_id, dtype=torch.long)
        attention_mask = torch.zeros((batch_size, max_thinker), dtype=torch.long)
        text_labels = torch.full((batch_size, max_thinker), IGNORE_INDEX, dtype=torch.long)
        t_input_ids = torch.full((batch_size, max_talker), self.codec_pad, dtype=torch.long)
        t_attention_mask = torch.zeros((batch_size, max_talker), dtype=torch.long)
        t_labels = torch.full((batch_size, max_talker), IGNORE_INDEX, dtype=torch.long)

        for i, (ids, row, labels) in enumerate(zip(thinker_ids, talker_ids, talker_labels)):
            input_ids[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
            attention_mask[i, : len(ids)] = 1
            # Thinker text supervision spans the assistant reply only. Stage 1
            # freezes the thinker, so this is inert there and only matters from
            # Stage 2 on.
            start, length = reply_starts[i], reply_lens[i]
            text_labels[i, start : start + length] = torch.tensor(
                ids[start : start + length], dtype=torch.long
            )
            t_input_ids[i, : len(row)] = torch.tensor(row, dtype=torch.long)
            t_attention_mask[i, : len(row)] = 1
            t_labels[i, : len(labels)] = torch.tensor(labels, dtype=torch.long)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "text_labels": text_labels,
            "talker_input_ids": t_input_ids,
            "talker_attention_mask": t_attention_mask,
            "talker_labels": t_labels,
            "reply_start_positions": torch.tensor(reply_starts, dtype=torch.long),
            "reply_lengths": torch.tensor(reply_lens, dtype=torch.long),
            "task_type": samples[0]["task_type"],
            "ids": [s["id"] for s in samples],
        }

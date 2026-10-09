"""Row-level dataset and collate for the Megatron Talker warmup.

Megatron's own dataloader uses ``default_collate``, which cannot pad. The
codebase already has an answer to that -- patch
``build_pretraining_data_loader`` to honour ``dataset.collate_fn`` -- and
this follows it, so ``--micro-batch-size`` stays the real knob for
saturating the GPUs rather than something buried in the dataset.

Rows are bucket-shuffled rather than plain-shuffled: utterances carry 4 to
600 codes, and a micro-batch drawn uniformly pads to its longest member, so
most of the batch would be padding. Sorting by code length, cutting into
micro-batch-sized chunks and shuffling the *chunks* keeps each batch
length-homogeneous while leaving the order random. This relies on the
sampler handing out contiguous, micro-batch-aligned ranges, which
``MegatronPretrainingSampler`` (``--dataloader-type single``) does.

``--dataloader-type single`` means both iterators are single-pass. The
training set runs out after one epoch -- at global batch 384 that is
iteration 2,128 of 5,000 -- and the validation set after the first
evaluation. Megatron builds each once and never restarts it.
Rather than switching to ``cyclic``, which reshuffles and so throws the
bucketing away, both are made long enough up front: the training set is
concatenated over as many epochs as the run consumes, and the validation
set is truncated to exactly one evaluation's worth of rows and repeated.
Every evaluation then scores the *same* utterances -- which is what makes
the points on a curve comparable at all.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from experiments.uniss_omni_s2st_v1.modeling.stage0_assembly import build_codec_sequence
from experiments.uniss_omni_s2st_v1.training.tts_data import build_thinker_batch


def bucket_shuffle(rows: list[dict], *, micro_batch: int, seed: int) -> list[dict]:
    """Sort by code length, then shuffle whole micro-batch-sized chunks."""
    if micro_batch < 1:
        raise ValueError("micro_batch must be positive")
    order = sorted(range(len(rows)), key=lambda i: len(rows[i]["codes"]))
    chunks = [order[start : start + micro_batch] for start in range(0, len(order), micro_batch)]
    rng = np.random.default_rng(seed)
    rng.shuffle(chunks)
    return [rows[i] for chunk in chunks for i in chunk]


class OmniTtsDataset(Dataset):
    """Target text in, BiCodec codes out; one row per utterance."""

    def __init__(
        self,
        rows: list[dict],
        *,
        bos_id: int,
        pad_id: int,
        eos_id: int,
        text_pad_id: int,
        micro_batch: int = 1,
        seed: int = 20261009,
        bucket: bool = True,
        truncate_to: int | None = None,
        repeat: int = 1,
        epochs: int = 1,
    ):
        if not rows:
            raise ValueError("no rows")
        if epochs < 1:
            raise ValueError("epochs must be positive")
        if bucket:
            # One bucket shuffle per epoch, each with its own seed: the
            # batches stay length-homogeneous while no two passes present
            # them in the same order.
            self.rows = [
                row
                for epoch in range(epochs)
                for row in bucket_shuffle(rows, micro_batch=micro_batch, seed=seed + epoch)
            ]
        else:
            self.rows = list(rows) * epochs
        if truncate_to is not None:
            if truncate_to > len(self.rows):
                raise ValueError(
                    f"asked to truncate to {truncate_to} rows but only"
                    f" {len(self.rows)} are available"
                )
            self.rows = self.rows[:truncate_to]
        if repeat < 1:
            raise ValueError("repeat must be positive")
        self.repeat = repeat
        self.bos_id = bos_id
        self.pad_id = pad_id
        self.eos_id = eos_id
        self.text_pad_id = text_pad_id
        # Megatron picks this up through the patched dataloader builder.
        self.collate_fn = self.collate

    def __len__(self) -> int:
        return len(self.rows) * self.repeat

    def __getitem__(self, index: int) -> dict:
        row = self.rows[index % len(self.rows)]
        return {
            "prompt_ids": row["prompt_ids"],
            "reply_ids": row["reply_ids"],
            "codes": row["codes"],
            "globals": row["globals"],
        }

    def collate(self, samples: list[dict]) -> dict:
        batch = build_thinker_batch(
            [s["prompt_ids"] for s in samples],
            [s["reply_ids"] for s in samples],
            pad_id=self.text_pad_id,
        )
        codec_ids, labels, codec_mask = build_codec_sequence(
            [torch.as_tensor(s["codes"], dtype=torch.long) for s in samples],
            bos_id=self.bos_id,
            pad_id=self.pad_id,
            eos_id=self.eos_id,
        )
        return {
            "thinker_input_ids": batch.input_ids,
            "thinker_attention_mask": batch.attention_mask,
            "reply_ids": batch.reply_ids,
            "reply_mask": batch.reply_mask,
            "reply_start": torch.tensor(batch.reply_start, dtype=torch.long),
            "codec_input_ids": codec_ids,
            "codec_labels": labels,
            "codec_mask": codec_mask,
            # The 32 speaker tokens. UniSS's own TTS prompt carries them
            # before the semantic codes, and the Talker is given the same.
            "bicodec_global": torch.stack(
                [torch.as_tensor(s["globals"], dtype=torch.long) for s in samples]
            ),
        }


def install_omni_collate() -> None:
    """Teach Megatron's dataloader builder to honour ``dataset.collate_fn``.

    A local copy of the pattern the phase-3 experiments use, rather than an
    import from one of them: those are finished runs and this line must not
    be able to change their behaviour.
    """
    import megatron.training.datasets.data_samplers as data_samplers
    import megatron.training.training as megatron_training

    original = data_samplers.build_pretraining_data_loader
    if getattr(original, "_uniss_omni_collate", False):
        return

    def build_with_collate(dataset, *args, **kwargs):
        collate = getattr(dataset, "collate_fn", None)
        if not callable(collate):
            return original(dataset, *args, **kwargs)
        original_loader = torch.utils.data.DataLoader

        def data_loader(*loader_args, **loader_kwargs):
            loader_kwargs.setdefault("collate_fn", collate)
            return original_loader(*loader_args, **loader_kwargs)

        torch.utils.data.DataLoader = data_loader
        try:
            return original(dataset, *args, **kwargs)
        finally:
            torch.utils.data.DataLoader = original_loader

    build_with_collate._uniss_omni_collate = True
    data_samplers.build_pretraining_data_loader = build_with_collate
    megatron_training.build_pretraining_data_loader = build_with_collate


def unwrap_to_device(batch: dict, device) -> dict:
    """Move a collated batch to the GPU."""
    return {
        key: (value.to(device, non_blocking=True) if torch.is_tensor(value) else value)
        for key, value in batch.items()
    }

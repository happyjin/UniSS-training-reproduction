"""Batches for Stage 2: one task per micro-batch, sampled at the paper's ratio.

Two reasons a batch holds a single task rather than a mixture.

The processor expands ``<|AUDIO|>`` per audio input, so a batch mixing
audio-backed rows with text-only ones has to track which row carries a
placeholder and which does not -- avoidable complexity for no gain.

And the ranks have to stay in step. The gradient all-reduce is a barrier,
an S2ST sample costs several times a MT sample, and in Stage 1 letting
ranks draw different-cost work cost 38% of the throughput. Every rank in
a step therefore draws the same task from the same seeded schedule.
"""

from __future__ import annotations

import os

import numpy as np
import torch
from torch.utils.data import Dataset

from experiments.uniss_omni_s2st_v1.training import stage2_tasks
from experiments.uniss_omni_s2st_v1.training.stage2_tasks import MIXTURE


def task_schedule(steps: int, *, seed: int, mixture: dict[str, float] = MIXTURE) -> list[str]:
    """One task per step, drawn at the mixture's ratio.

    Shared across ranks by seed rather than communicated, so a rank can
    build its own schedule and they still agree.
    """
    names = sorted(mixture)
    weights = np.asarray([mixture[n] for n in names], dtype=np.float64)
    weights = weights / weights.sum()
    rng = np.random.default_rng(seed)
    return [names[i] for i in rng.choice(len(names), size=steps, p=weights)]


class Stage2Dataset(Dataset):
    """Indexable over (step, position); the task comes from the schedule."""

    def __init__(
        self,
        corpus,
        schedule: list[str],
        *,
        global_batch: int,
        seed: int = 20261011,
    ):
        self.corpus = corpus
        self.schedule = schedule
        self.global_batch = int(global_batch)
        self.seed = int(seed)
        rng = np.random.default_rng(seed)

        # A real permutation of each pool, not a contiguous window from a
        # random offset: manifest neighbours come from the same corpus and
        # the same shard, so a window is a biased sample of one source.
        self.audio_order = rng.permutation(max(1, len(corpus.audio_rows)))
        self.text_order = rng.permutation(max(1, len(corpus.text_corpus)))

        # Where each step starts in its pool, so __getitem__ stays a pure
        # function of the index and the dataloader workers agree.
        self.offsets: list[int] = []
        consumed = {"audio": 0, "text": 0}

        # Off by default: every epoch then replays the one permutation drawn
        # above, batch for batch, which is what every run before this one
        # trained on. UNISS_RESHUFFLE_EACH_EPOCH=1 keeps the first epoch as
        # it was and draws a fresh order for each one after.
        #
        # UNISS_RESHUFFLE_FROM_STEP is for switching a run that is already
        # part-way through a later epoch. Positions served before that step
        # keep the order they were served in, and only the remainder of the
        # epoch is shuffled -- so the epoch still holds each row exactly
        # once. Shuffling the whole epoch instead would serve some rows
        # twice and skip others, since its first part has already happened.
        self.reshuffle = os.environ.get("UNISS_RESHUFFLE_EACH_EPOCH") == "1"
        from_step = int(os.environ.get("UNISS_RESHUFFLE_FROM_STEP", "0"))
        self.reshuffle_from = {"audio": 0, "text": 0}
        for step, task in enumerate(schedule):
            if step == from_step:
                self.reshuffle_from = dict(consumed)
            pool = "text" if stage2_tasks.uses_tts_cache(task) else "audio"
            self.offsets.append(consumed[pool])
            consumed[pool] += self.global_batch
        if from_step >= len(schedule):
            self.reshuffle_from = dict(consumed)
        self._orders: dict[tuple[str, int], np.ndarray] = {}
        # Only a schedule that reaches a second pass has anything to
        # reshuffle; the dev set built beside the train set never does, and
        # announcing a switch for it would read as if it applied.
        if self.reshuffle and consumed["audio"] > len(corpus.audio_rows):
            n = max(1, len(corpus.audio_rows))
            at = self.reshuffle_from["audio"]
            print(f"[stage2_dataset] reshuffle each epoch from step {from_step}:"
                  f" audio cursor {at:,} = epoch {at // n + 1}, position {at % n:,}"
                  f" of {n:,}", flush=True)
        self.epochs = {
            "audio": consumed["audio"] / max(1, len(corpus.audio_rows)),
            "text": consumed["text"] / max(1, len(corpus.text_corpus)),
        }
        self.collate_fn = self.collate

    def __len__(self) -> int:
        return len(self.schedule) * self.global_batch

    def _epoch_order(self, pool: str, epoch: int) -> np.ndarray:
        """The row order for one pass over a pool.

        Epoch 0 is always the permutation drawn at construction. A later
        epoch gets its own, seeded by (seed, epoch, pool) so every rank and
        every dataloader worker derives the same one without talking.
        """
        base = self.audio_order if pool == "audio" else self.text_order
        n = base.size
        keep = min(max(self.reshuffle_from[pool] - epoch * n, 0), n)
        if epoch == 0 or keep == n:
            return base
        key = (pool, epoch)
        if key not in self._orders:
            rng = np.random.default_rng(
                [self.seed, epoch, 0 if pool == "audio" else 1])
            if keep == 0:
                order = base[rng.permutation(n)]
            else:
                tail = base[keep:].copy()
                rng.shuffle(tail)
                order = np.concatenate([base[:keep], tail])
            # A worker straddles at most one boundary per batch; two
            # entries is enough, and each is 77 MB at int32.
            if len(self._orders) >= 2:
                self._orders.pop(next(iter(self._orders)))
            self._orders[key] = order.astype(np.int32 if n < 2**31 else np.int64)
        return self._orders[key]

    def _row(self, pool: str, cursor: int) -> int:
        base = self.audio_order if pool == "audio" else self.text_order
        if not self.reshuffle:
            return int(base[cursor % base.size])
        epoch, pos = divmod(cursor, base.size)
        return int(self._epoch_order(pool, epoch)[pos])

    def __getitem__(self, index: int) -> dict:
        step, position = divmod(index, self.global_batch)
        step = min(step, len(self.schedule) - 1)
        task = self.schedule[step]
        cursor = self.offsets[step] + position
        if stage2_tasks.uses_tts_cache(task):
            row = self.corpus.text_sample(self._row("text", cursor))
            # The cache is already tokenised; re-templating from text would
            # need raw text it deliberately does not store.
            return {
                "task": task,
                "audio": None,
                "prompt_ids": row["prompt_ids"],
                "reply_ids": row["reply_ids"],
                "user_text": None,
                "text_target": None,
                "codes": row["codes"],
                "globals": row["globals"],
            }
        row = self.corpus.audio_sample(self._row("audio", cursor))
        sample = stage2_tasks.build(task, row)
        return {
            "task": task,
            "audio": sample.audio_path if stage2_tasks.needs_audio(task) else None,
            "prompt_ids": None,
            "reply_ids": None,
            "user_text": sample.user_text,
            "text_target": sample.text_target,
            "codes": sample.codes,
            "globals": sample.globals,
        }

    def collate(self, samples: list[dict]) -> dict:
        task = samples[0]["task"]
        if any(s["task"] != task for s in samples):
            raise ValueError("a batch must hold one task; the sampler drifted")
        batch = {
            "task": task,
            "user_text": [s["user_text"] for s in samples],
            "text_target": [s["text_target"] for s in samples],
            "prompt_ids": [s["prompt_ids"] for s in samples],
            "reply_ids": [s["reply_ids"] for s in samples],
        }
        if stage2_tasks.needs_audio(task):
            batch["waveforms"] = [self.corpus.waveform(s["audio"])[0] for s in samples]
        if stage2_tasks.supervises_codes(task):
            batch["codes"] = [
                torch.as_tensor(np.asarray(s["codes"]), dtype=torch.long) for s in samples
            ]
            batch["globals"] = torch.stack(
                [torch.as_tensor(np.asarray(s["globals"]), dtype=torch.long) for s in samples]
            )
        return batch

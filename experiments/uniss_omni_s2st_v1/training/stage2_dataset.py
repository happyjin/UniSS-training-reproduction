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
        rng = np.random.default_rng(seed)
        # An independent draw per step per pool, so two steps of the same
        # task do not see the same rows.
        self.audio_cursor = rng.integers(0, max(1, len(corpus.audio_rows)), size=len(schedule))
        self.text_cursor = rng.integers(0, max(1, len(corpus.text_corpus)), size=len(schedule))
        self.collate_fn = self.collate

    def __len__(self) -> int:
        return len(self.schedule) * self.global_batch

    def __getitem__(self, index: int) -> dict:
        step, position = divmod(index, self.global_batch)
        step = min(step, len(self.schedule) - 1)
        task = self.schedule[step]
        if stage2_tasks.needs_audio(task):
            row = self.corpus.audio_sample(int(self.audio_cursor[step]) + position)
        else:
            row = self.corpus.text_sample(int(self.text_cursor[step]) + position)
        sample = stage2_tasks.build(task, row)
        return {
            "task": task,
            "audio": sample.audio_path,
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

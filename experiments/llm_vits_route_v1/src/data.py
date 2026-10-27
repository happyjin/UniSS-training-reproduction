"""LJSpeech dataset, length-bucketed DDP sampler and collate (waveforms only; spectra are made on GPU)."""
import json
import random

import numpy as np
import soundfile as sf
import torch
import torch.distributed as dist

from text import intersperse, text_to_ids


def load_manifest(path, splits):
  rows = [json.loads(l) for l in open(path)]
  return [r for r in rows if r["split"] in splits]


class TextAudioDataset(torch.utils.data.Dataset):
  def __init__(self, rows, hop=256, max_wav_value=32768.0):
    self.rows = rows
    self.hop = hop
    self.lengths = [int(r["duration_s"] * r["sample_rate"]) // hop for r in rows]

  def __len__(self):
    return len(self.rows)

  def __getitem__(self, i):
    r = self.rows[i]
    wav, sr = sf.read(r["audio_path"], dtype="float32")
    assert sr == 22050, sr
    ids = intersperse(text_to_ids(r["clean"]))
    return i, torch.LongTensor(ids), torch.from_numpy(wav)


def collate(batch):
  batch = sorted(batch, key=lambda b: -b[2].numel())
  max_x = max(b[1].numel() for b in batch)
  max_w = max(b[2].numel() for b in batch)
  idx = torch.LongTensor([b[0] for b in batch])
  x = torch.zeros(len(batch), max_x, dtype=torch.long)
  w = torch.zeros(len(batch), 1, max_w)
  xl = torch.LongTensor([b[1].numel() for b in batch])
  wl = torch.LongTensor([b[2].numel() for b in batch])
  for k, b in enumerate(batch):
    x[k, :b[1].numel()] = b[1]
    w[k, 0, :b[2].numel()] = b[2]
  return idx, x, xl, w, wl


class BucketSampler(torch.utils.data.Sampler):
  """Batches of similar length; each rank gets a disjoint slice of every epoch's batch list."""

  def __init__(self, lengths, batch_size, boundaries=(32, 300, 400, 500, 600, 700, 800, 900, 1000), seed=1234):
    self.lengths = lengths
    self.batch_size = batch_size
    self.boundaries = list(boundaries)
    self.seed = seed
    self.epoch = 0
    self.rank = dist.get_rank() if dist.is_initialized() else 0
    self.world = dist.get_world_size() if dist.is_initialized() else 1

  def set_epoch(self, epoch):
    self.epoch = epoch

  def _batches(self):
    rng = random.Random(self.seed + self.epoch)
    buckets = [[] for _ in range(len(self.boundaries) - 1)]
    for i, l in enumerate(self.lengths):
      for b in range(len(self.boundaries) - 1):
        if self.boundaries[b] < l <= self.boundaries[b + 1]:
          buckets[b].append(i)
          break
    batches = []
    for bucket in buckets:
      rng.shuffle(bucket)
      for s in range(0, len(bucket) - self.batch_size + 1, self.batch_size):
        batches.append(bucket[s:s + self.batch_size])
    rng.shuffle(batches)
    n = len(batches) // self.world * self.world
    return batches[self.rank:n:self.world]

  def __iter__(self):
    return iter(self._batches())

  def __len__(self):
    return len(self._batches())

"""Commit-step instances for streaming fine-tuning and their GPU-side assembly.

An instance is (utterance, committed unit i, visible unit j = min(i + k, U - 1)). The last unit carries the end of
stream, so j == U - 1 means the text is known to be complete and no future is imagined.
"""
import json
import random

import numpy as np
import soundfile as sf
import torch

from stream_common import prosody_y, token_char, unit_token_spans
from text import clean_text, intersperse, text_to_ids

K_PROBS = (0.3, 0.3, 0.2, 0.2)   # lookahead k = 0..3 units


def load_cands(paths):
  cands = {}
  for p in paths:
    for line in open(p):
      d = json.loads(line)
      if "meta" in d:
        continue
      cands[(d["utt_id"], d["unit"])] = d["cands"]
  return cands


def imagined_ids(prefix, cand, n_words=3):
  """Prefix text plus the first n_words of a sampled continuation, cleaned; planning only, never spoken."""
  c = clean_text(" " + cand.replace("\n", " "))
  words = c.split(" ")[:n_words]
  ext = " ".join(w for w in words if w)
  if not ext:
    text = prefix
  elif ext[0] in ",.;:!?":
    text = prefix + ext
  else:
    text = prefix + " " + ext
  return intersperse(text_to_ids(text))


class StreamDataset(torch.utils.data.Dataset):
  def __init__(self, rows, durations, prosody, qwen_dir=None, cands=None, per_utt=4, seed=0, fixed=False,
               n_cand=8, cand_words=3):
    self.rows, self.durations, self.prosody = rows, durations, prosody
    self.qwen_dir, self.cands = qwen_dir, cands
    self.per_utt, self.seed, self.fixed = per_utt, seed, fixed
    self.n_cand, self.cand_words = n_cand, cand_words

  def __len__(self):
    return len(self.rows)

  def instances(self, idx, n_units):
    rng = random.Random(self.seed * 1000003 + idx) if self.fixed else random
    out = []
    for _ in range(self.per_utt):
      i = rng.randrange(n_units)
      k = rng.choices(range(len(K_PROBS)), K_PROBS)[0]
      out.append((i, min(i + k, n_units - 1)))
    return out

  def __getitem__(self, idx):
    r = self.rows[idx]
    uid = r["utt_id"]
    ids = intersperse(text_to_ids(r["clean"]))
    dur = self.durations[uid].astype(np.int64)
    spans = unit_token_spans(r)
    y = prosody_y(self.prosody[uid])
    item = {"idx": idx, "wav": torch.from_numpy(sf.read(r["audio_path"], dtype="float32")[0]),
            "ids": torch.LongTensor(ids), "dur": torch.from_numpy(dur), "inst": []}
    if self.qwen_dir is not None:
      z = np.load(f"{self.qwen_dir}/{uid}.npz")
      item["qh"] = torch.from_numpy(z["h_mid"])
      item["qmap"] = torch.LongTensor([int(z["char_tok"][token_char(t)]) for t in range(len(ids))])
    cum = np.concatenate([[0], np.cumsum(dur)])
    for i, j in self.instances(idx, len(spans)):
      a, b = spans[i]
      tv = spans[j][1]
      inst = {"i": i, "j": j, "tv": tv, "a": a, "b": b, "f0": int(cum[a]), "f1": int(cum[b]),
              "hist": torch.from_numpy(y[:i]).float()}
      if self.cands is not None:
        prefix = r["clean"][:r["units"][j][1]]
        if j == len(spans) - 1:
          seqs = [ids[:tv]] * self.n_cand
        else:
          seqs = [imagined_ids(prefix, c, self.cand_words) for c in self.cands[(uid, j)][:self.n_cand]]
        assert all(s[:tv] == ids[:tv] for s in seqs), uid
        inst["cand"] = [torch.LongTensor(s) for s in seqs]
      item["inst"].append(inst)
    return item


def collate(items):
  pad = torch.nn.utils.rnn.pad_sequence
  out = {"wav": pad([it["wav"] for it in items], batch_first=True)[:, None],
         "wav_len": torch.LongTensor([it["wav"].numel() for it in items]),
         "ids": pad([it["ids"] for it in items], batch_first=True),
         "dur": pad([it["dur"] for it in items], batch_first=True)}
  if "qh" in items[0]:
    offs, qs, qmaps = 0, [], []
    for it in items:
      qs.append(it["qh"])
      qmaps.append(it["qmap"] + offs)
      offs += it["qh"].size(0)
    out["qh"] = torch.cat(qs)
    out["qmap"] = pad(qmaps, batch_first=True)
  keys = ["i", "j", "tv", "a", "b", "f0", "f1"]
  inst = [(u, x) for u, it in enumerate(items) for x in it["inst"]]
  out["inst_utt"] = torch.LongTensor([u for u, _ in inst])
  for k in keys:
    out[k] = torch.LongTensor([x[k] for _, x in inst])
  out["hist"] = pad([x["hist"] if len(x["hist"]) else torch.zeros(1, 2) for _, x in inst], batch_first=True)
  out["hist_len"] = torch.LongTensor([len(x["hist"]) for _, x in inst])
  if "cand" in inst[0][1]:
    seqs = [s for _, x in inst for s in x["cand"]]
    out["cand"] = pad(seqs, batch_first=True)
    out["cand_len"] = torch.LongTensor([len(s) for s in seqs])
  return out


def assemble(batch, dev):
  """Instance tensors on GPU: visible ids, block mask, block durations, Qwen features, last visible index."""
  b = {k: v.to(dev, non_blocking=True) for k, v in batch.items()}
  u, tv = b["inst_utt"], b["tv"]
  t_max = int(tv.max())
  ar = torch.arange(t_max, device=dev)[None]
  vis = ar < tv[:, None]
  ids = b["ids"][u, :t_max] * vis
  blk = ((ar >= b["a"][:, None]) & (ar < b["b"][:, None])).float()[:, None]
  w = b["dur"][u, :t_max].float()[:, None] * blk
  out = {"ids": ids, "lens": tv, "blk": blk, "w": w, "last": tv - 1}
  if "qh" in b:
    out["qfeat"] = b["qh"][b["qmap"][u, :t_max]] * vis[..., None]
  return b, out

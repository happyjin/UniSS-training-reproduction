"""Shared helpers: backbone loading, token classes and unit-level prosody targets Y."""
import json
import math

import numpy as np
import torch

from models import SynthesizerTrn
from text import char_span_to_token_span, symbols

WORD_CHARS = set("abcdefghijklmnopqrstuvwxyz'-")


def load_backbone(config, ckpt):
  hps = json.load(open(config))
  d = hps["data"]
  net = SynthesizerTrn(len(symbols), d["filter_length"] // 2 + 1, hps["train"]["segment_size"] // d["hop_length"],
                       **hps["model"])
  net.load_state_dict(torch.load(ckpt, map_location="cpu")["model"])
  return net


def token_char(t):
  """Char index a token belongs to: char tokens are odd, a blank belongs to the char before it (token 0 to char 0)."""
  return 0 if t == 0 else (t - 1) // 2


def token_is_word(clean):
  n = 2 * len(clean) + 1
  return np.array([clean[token_char(t)] in WORD_CHARS for t in range(n)], dtype=bool)


def unit_token_spans(row):
  n = len(row["clean"])
  return [char_span_to_token_span(s, e, n) for s, e in row["units"]]


def unit_word_pause(row, dur):
  """[[word_frames, pause_frames]] per unit from token durations dur (length 2*n_chars+1)."""
  is_word = token_is_word(row["clean"])
  out = []
  for a, b in unit_token_spans(row):
    seg, wm = np.asarray(dur[a:b], dtype=np.float64), is_word[a:b]
    out.append([float(seg[wm].sum()), float(seg[~wm].sum())])
  return out


def prosody_y(word_pause):
  """Handbook D1 duration coordinates: log word frames and log(1 + pause frames)."""
  wp = np.asarray(word_pause, dtype=np.float64)
  return np.stack([np.log(np.maximum(wp[:, 0], 1.0)), np.log1p(wp[:, 1])], 1)

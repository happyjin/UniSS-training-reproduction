"""Shared pieces of the acoustic-side schemes (S2, S3, S4, S5, T4): an aligned dataset with the frozen d* labels,
frame-level conditions from the frozen E0 backbone, causal convolutions, and the dual-clock interval overlap.

Frame conditions: the backbone text encoder runs on the full text (as in E0 training); token statistics are
expanded to high-rate frames (hop 256 at 22.05 kHz) with d*. At streaming time the same quantities come from the
committed plans of the S6-future planner, so every acoustic scheme is driven by identical committed durations.
"""
import json

import numpy as np
import soundfile as sf
import torch
from torch import nn
from torch.nn import functional as F

from mel_processing import spectrogram_torch
from text import intersperse, text_to_ids

HOP, SR = 256, 22050
FRAME_S = HOP / SR


class AlignedDataset(torch.utils.data.Dataset):
  """(index, token ids, waveform, d*) per utterance; extra(utt_id) may attach cached targets."""

  def __init__(self, rows, durations, extra=None):
    self.rows = [r for r in rows if r["utt_id"] in durations]
    self.durs = durations
    self.extra = extra
    self.lengths = [int(self.durs[r["utt_id"]].sum()) for r in self.rows]

  def __len__(self):
    return len(self.rows)

  def __getitem__(self, i):
    r = self.rows[i]
    wav, sr = sf.read(r["audio_path"], dtype="float32")
    assert sr == SR, sr
    ids = intersperse(text_to_ids(r["clean"]))
    d = torch.from_numpy(self.durs[r["utt_id"]].astype(np.int64))
    assert len(d) == len(ids), r["utt_id"]
    ex = self.extra(r["utt_id"]) if self.extra is not None else None
    return i, torch.LongTensor(ids), torch.from_numpy(wav), d, ex


def collate_aligned(batch):
  batch = sorted(batch, key=lambda b: -b[2].numel())
  pad = torch.nn.utils.rnn.pad_sequence
  idx = torch.LongTensor([b[0] for b in batch])
  x = pad([b[1] for b in batch], batch_first=True)
  xl = torch.LongTensor([b[1].numel() for b in batch])
  w = pad([b[2] for b in batch], batch_first=True)[:, None]
  wl = torch.LongTensor([b[2].numel() for b in batch])
  d = pad([b[3] for b in batch], batch_first=True)
  ex = [b[4] for b in batch]
  return idx, x, xl, w, wl, d, ex


def expand_matrix(d, t_max):
  """d [B, N] int frame durations -> path [B, N, T] (token n covers its frames)."""
  cum = torch.cumsum(d, 1)
  start = cum - d
  t = torch.arange(t_max, device=d.device)[None, None]
  return ((t >= start[..., None]) & (t < cum[..., None])).float()


@torch.no_grad()
def backbone_frames(net, x, xl, wav, wl, d, dd, posterior=True):
  """Frozen E0 quantities at high-rate frames: text hidden h_f, prior m_f/logs_f, posterior z/m_q/logs_q."""
  t = d.sum(1)
  t_max = int(t.max())
  h, m_p, logs_p, x_mask = net.enc_p(x, xl)
  path = expand_matrix(d, t_max)                       # [B, N, T]
  h_f = torch.bmm(h, path)
  m_f, logs_f = torch.bmm(m_p, path), torch.bmm(logs_p, path)
  y_mask = (torch.arange(t_max, device=x.device)[None] < t[:, None]).float()[:, None]
  out = {"h_f": h_f, "m_f": m_f, "logs_f": logs_f, "y_mask": y_mask, "t": t, "h": h, "m_p": m_p, "x_mask": x_mask}
  if posterior:
    spec = spectrogram_torch(wav.squeeze(1), dd["filter_length"], dd["sampling_rate"], dd["hop_length"],
                             dd["win_length"], center=False)
    spec = F.pad(spec, (0, max(0, t_max - spec.size(2))))[:, :, :t_max]
    z, m_q, logs_q, _ = net.enc_q(spec, t)
    out.update(spec=spec, z=z * y_mask, m_q=m_q, logs_q=logs_q)
  return out


class CausalConv1d(nn.Conv1d):
  """Left-padded convolution: output frame t sees inputs <= t only."""

  def __init__(self, cin, cout, k, dilation=1, **kw):
    super().__init__(cin, cout, k, dilation=dilation, **kw)
    self.left = (k - 1) * dilation

  def forward(self, x):
    return super().forward(F.pad(x, (self.left, 0)))


class CausalWN(nn.Module):
  """WaveNet-style gated residual stack (as modules.WN) with causal convolutions; optional condition channels."""

  def __init__(self, hidden, k, dilation_rate, n_layers, cond=0):
    super().__init__()
    self.n_layers, self.hidden = n_layers, hidden
    self.in_layers = nn.ModuleList()
    self.res_skip = nn.ModuleList()
    self.cond = nn.Conv1d(cond, 2 * hidden * n_layers, 1) if cond else None
    for i in range(n_layers):
      self.in_layers.append(nn.utils.parametrizations.weight_norm(CausalConv1d(hidden, 2 * hidden, k, dilation_rate ** i)))
      out = 2 * hidden if i < n_layers - 1 else hidden
      self.res_skip.append(nn.utils.parametrizations.weight_norm(nn.Conv1d(hidden, out, 1)))

  def forward(self, x, mask, g=None):
    out = torch.zeros_like(x)
    gc = self.cond(g) if (g is not None and self.cond is not None) else None
    for i in range(self.n_layers):
      a = self.in_layers[i](x)
      if gc is not None:
        a = a + gc[:, i * 2 * self.hidden:(i + 1) * 2 * self.hidden]
      acts = torch.tanh(a[:, :self.hidden]) * torch.sigmoid(a[:, self.hidden:])
      rs = self.res_skip[i](acts)
      if i < self.n_layers - 1:
        x = (x + rs[:, :self.hidden]) * mask
        out = out + rs[:, self.hidden:]
      else:
        out = out + rs
    return out * mask


class CausalCoupling(nn.Module):
  """Mean-only affine coupling (handbook T4.3 with a = 0) whose conditioner is causal in time."""

  def __init__(self, channels, hidden, k, dilation_rate, n_layers, cond=0):
    super().__init__()
    self.half = channels // 2
    self.pre = nn.Conv1d(self.half, hidden, 1)
    self.enc = CausalWN(hidden, k, dilation_rate, n_layers, cond)
    self.post = nn.Conv1d(hidden, self.half, 1)
    self.post.weight.data.zero_()
    self.post.bias.data.zero_()

  def forward(self, x, mask, g=None, reverse=False):
    x0, x1 = x[:, :self.half], x[:, self.half:]
    m = self.post(self.enc(self.pre(x0) * mask, mask, g)) * mask
    x1 = (x1 + m) * mask if not reverse else (x1 - m) * mask
    return torch.cat([x0, x1], 1)


class CausalFlow(nn.Module):
  """Stack of causal couplings with channel flips; volume preserving (logdet 0), prefix-consistent in time."""

  def __init__(self, channels=192, hidden=192, k=5, dilation_rate=1, n_layers=4, n_flows=4, cond=0):
    super().__init__()
    self.flows = nn.ModuleList([CausalCoupling(channels, hidden, k, dilation_rate, n_layers, cond)
                                for _ in range(n_flows)])

  def forward(self, x, mask, g=None, reverse=False):
    if not reverse:
      for f in self.flows:
        x = torch.flip(f(x, mask, g), [1])
    else:
      for f in reversed(self.flows):
        x = f(torch.flip(x, [1]), mask, g, reverse=True)
    return x


def interval_overlap(rows, cols):
  """Row-normalised overlap of two interval partitions given by boundary arrays (seconds): O[i, j] =
  |row_i & col_j| / |row_i & union(cols)|; rows outside every column stay zero (handbook S3.2)."""
  lo = np.maximum(rows[:-1, None], cols[None, :-1])
  hi = np.minimum(rows[1:, None], cols[None, 1:])
  o = np.clip(hi - lo, 0, None).astype(np.float32)
  s = o.sum(1, keepdims=True)
  return np.where(s > 0, o / np.maximum(s, 1e-12), 0.0).astype(np.float32)


def token_bounds(d_frames):
  return np.concatenate([[0.0], np.cumsum(d_frames)]) * FRAME_S


def clock_bounds(n, rate_hz):
  return np.arange(n + 1) / rate_hz


def overlap_matrix(d_frames, rate_hz, n_low=None):
  """Low-rate frame x token overlap (S3.2): rows are rate_hz frames, columns tokens with d_frames high-rate frames."""
  tb = token_bounds(d_frames)
  n_low = n_low if n_low is not None else int(np.ceil(tb[-1] * rate_hz - 1e-9))
  return interval_overlap(clock_bounds(n_low, rate_hz), tb)


def load_rows(manifest, splits):
  return [r for r in map(json.loads, open(manifest)) if r["split"] in splits]

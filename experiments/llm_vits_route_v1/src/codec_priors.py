"""Priors for the pretrained-codec schemes.

S3Prior (handbook S3.2-S3.4): token conditions [h; m_p] (384) are moved to the 25 Hz AudioVAE clock by the
interval-overlap matrix, a causal network gives a diagonal Gaussian base, and a 64-channel causal coupling flow
maps the AudioVAE latent u to that base: log p(u | c) = log N(f(u); mu, sigma) (mean-only couplings, logdet 0).

S4 (handbook S4.1-S4.4, S4-D: the SDP owns duration): a causal token predictor over 12.5 Hz frames factorised as
p(a_t | a_<t, c) with the 16 codebooks of a frame predicted in parallel (a simplification of the within-frame
chain in S4.1), and an adapter that maps token embeddings onto the high-rate clock and corrects the frozen VITS
prior (zero-initialised, so it starts at the E0 prior).
"""
import math

import torch
from torch import nn
from torch.nn import functional as F

from acoustic_common import CausalFlow, CausalWN

COND = 384


class S3Prior(nn.Module):
  def __init__(self, latent=64, hidden=256):
    super().__init__()
    self.pre = nn.Conv1d(COND, hidden, 1)
    self.net = CausalWN(hidden, 3, 2, 6)
    self.proj = nn.Conv1d(hidden, 2 * latent, 1)
    self.flow = CausalFlow(channels=latent, hidden=128, k=3, dilation_rate=1, n_layers=4, n_flows=4, cond=hidden)

  def base(self, c, mask):
    g = self.net(self.pre(c) * mask, mask)
    mu, logs = self.proj(g).chunk(2, 1)
    return mu * mask, torch.clamp(logs, min=-7.0) * mask, g

  def forward(self, u, c, mask):
    """Negative log density of u [B, 64, T] per valid latent frame (summed over channels)."""
    mu, logs, g = self.base(c, mask)
    v = self.flow(u, mask, g)
    nll = (logs + 0.5 * ((v - mu) * torch.exp(-logs)) ** 2 + 0.5 * math.log(2 * math.pi)) * mask
    return nll.sum() / mask.sum()

  @torch.no_grad()
  def sample(self, c, mask, eps, temp):
    mu, logs, g = self.base(c, mask)
    v = mu + eps * torch.exp(logs) * temp
    return self.flow(v, mask, g, reverse=True)


class TokenPredictor(nn.Module):
  """Causal transformer over 12.5 Hz token frames; input = projected condition + embedding of the previous frame."""

  def __init__(self, n_q=16, vocab=2048, width=512, layers=6, heads=8):
    super().__init__()
    self.n_q, self.vocab = n_q, vocab
    self.cond = nn.Linear(COND, width)
    self.emb = nn.ModuleList([nn.Embedding(vocab + 1, width) for _ in range(n_q)])   # index vocab = BOS
    self.pos = nn.Embedding(2048, width)
    layer = nn.TransformerEncoderLayer(width, heads, 4 * width, 0.1, batch_first=True, norm_first=True)
    self.tf = nn.TransformerEncoder(layer, layers)
    self.norm = nn.LayerNorm(width)
    self.heads = nn.Linear(width, n_q * vocab)

  def forward(self, cond, prev):
    """cond [B, L, 384]; prev [B, L, 16] tokens of the previous frame (BOS at t=0) -> logits [B, L, 16, V]."""
    b, l, _ = cond.shape
    x = self.cond(cond) + sum(e(prev[..., q]) for q, e in enumerate(self.emb))
    x = x + self.pos(torch.arange(l, device=cond.device))[None]
    mask = torch.triu(torch.full((l, l), float("-inf"), device=cond.device), 1)
    h = self.norm(self.tf(x, mask=mask))
    return self.heads(h).view(b, l, self.n_q, self.vocab)

  @staticmethod
  def shift(a, bos):
    return torch.cat([torch.full_like(a[:, :1], bos), a[:, :-1]], 1)


class TokenAdapter(nn.Module):
  """S4.4: e_a = sum_q E_q[a_q] (256) on the token clock, overlap to high-rate frames, fused with [h_f; m_f]."""

  def __init__(self, n_q=16, vocab=2048, width=256, hidden=192):
    super().__init__()
    self.emb = nn.ModuleList([nn.Embedding(vocab, width) for _ in range(n_q)])
    self.fuse = nn.Sequential(nn.Conv1d(192 + 192 + width, hidden, 1), nn.GELU(), nn.Conv1d(hidden, hidden, 1))
    self.out = nn.Conv1d(hidden, 2 * 192, 1)
    self.out.weight.data.zero_()
    self.out.bias.data.zero_()

  def forward(self, a, o_frm, h_f, m_f, logs_f):
    """a [B, L, 16]; o_frm [B, T, L] high-rate-frame x token-frame overlap -> corrected prior m', logs' [B,192,T]."""
    e = sum(emb(a[..., q]) for q, emb in enumerate(self.emb))            # [B, L, 256]
    e_f = torch.bmm(o_frm, e).transpose(1, 2)                               # [B, 256, T]
    dm, dl = self.out(self.fuse(torch.cat([h_f, m_f, e_f], 1))).chunk(2, 1)
    return m_f + dm, logs_f + dl

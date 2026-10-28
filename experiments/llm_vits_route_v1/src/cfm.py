"""S2-E (handbook S2.1-S2.3, engineering variant): conditional flow matching over the 192-channel VITS posterior
latents, replacing the coupling flow. A window is [committed context frames, clean] + [target block, noised]; the
DiT vector field (6 layers, width 384, 6 heads, FFN 1536, adaLN-zero on the ODE time) predicts u = z1 - z0 on the
block only. Frame conditions are the frozen text-encoder hidden state and prior mean at each frame (384).
Generation never revises context frames, so committed latents stay fixed (inter-block causal, intra-block parallel).
"""
import math

import torch
from torch import nn


def sinusoid(t, dim):
  half = dim // 2
  f = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
  e = t.float()[..., None] * f
  return torch.cat([e.sin(), e.cos()], -1)


class Block(nn.Module):
  def __init__(self, w, heads, ffn):
    super().__init__()
    self.n1 = nn.LayerNorm(w, elementwise_affine=False)
    self.att = nn.MultiheadAttention(w, heads, batch_first=True)
    self.n2 = nn.LayerNorm(w, elementwise_affine=False)
    self.ff = nn.Sequential(nn.Linear(w, ffn), nn.GELU(), nn.Linear(ffn, w))
    self.ada = nn.Linear(w, 6 * w)
    nn.init.zeros_(self.ada.weight)
    nn.init.zeros_(self.ada.bias)

  def forward(self, x, temb, pad):
    s1, b1, g1, s2, b2, g2 = self.ada(temb)[:, None].chunk(6, -1)
    h = self.n1(x) * (1 + s1) + b1
    x = x + g1 * self.att(h, h, h, key_padding_mask=pad, need_weights=False)[0]
    h = self.n2(x) * (1 + s2) + b2
    return x + g2 * self.ff(h)


class CFMField(nn.Module):
  def __init__(self, zc=192, cond=384, width=384, layers=6, heads=6, ffn=1536):
    super().__init__()
    self.inp = nn.Linear(zc + cond + 1, width)
    self.t = nn.Sequential(nn.Linear(256, width), nn.SiLU(), nn.Linear(width, width))
    self.blocks = nn.ModuleList([Block(width, heads, ffn) for _ in range(layers)])
    self.norm = nn.LayerNorm(width, elementwise_affine=False)
    self.out = nn.Linear(width, zc)
    nn.init.zeros_(self.out.weight)
    nn.init.zeros_(self.out.bias)
    self.width = width

  def forward(self, z, cond, is_ctx, tau, pad):
    """z [B, W, 192] (context clean, block noised), cond [B, W, 384], is_ctx [B, W], tau [B], pad [B, W] True=pad."""
    w = z.size(1)
    x = self.inp(torch.cat([z, cond, is_ctx[..., None].float()], -1))
    x = x + sinusoid(torch.arange(w, device=z.device), self.width)[None]
    temb = self.t(sinusoid(tau * 1000, 256))
    for b in self.blocks:
      x = b(x, temb, pad)
    return self.out(self.norm(x))

  @torch.no_grad()
  def generate(self, ctx_z, ctx_c, blk_c, z0, nfe=16):
    """Euler integration of the block given committed context; ctx_z [C,192], ctx_c [C,384], blk_c [L,384],
    z0 [L,192] -> z1 [L,192]."""
    c, l = ctx_z.size(0), blk_c.size(0)
    cond = torch.cat([ctx_c, blk_c], 0)[None]
    is_ctx = torch.cat([torch.ones(c), torch.zeros(l)]).to(z0.device)[None]
    pad = torch.zeros(1, c + l, dtype=torch.bool, device=z0.device)
    z = z0
    for k in range(nfe):
      tau = torch.full((1,), k / nfe, device=z0.device)
      v = self(torch.cat([ctx_z, z], 0)[None], cond, is_ctx, tau, pad)[0, c:]
      z = z + v / nfe
    return z

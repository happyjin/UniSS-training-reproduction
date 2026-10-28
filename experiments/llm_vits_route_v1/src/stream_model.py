"""Streaming VITS variants on a frozen E0 backbone (posterior encoder, flow and HiFi-GAN decoder stay frozen).

One commit step: the text encoder sees the visible prefix (units 1..j); the block is the unit being committed
(unit i <= j). Duration (SDP) and acoustic prior are produced for block tokens only.

Variants (all new branches are zero-initialised, so every variant starts exactly at the E1 baseline):
  E1  prefix-trained text encoder + SDP                                   (handbook E1, main baseline)
  E2  + gated Qwen3 prefix-state adapter, G2, into prior and duration      (handbook E2)
  S6  + future r: block latent with prior from LLM-sampled futures (imagined lookahead, R40 style) and
        posterior from the real block prosody (S6.2/S6.4 CVAE), and/or
      + rho: rhythm state from realised durations of committed units (deterministic variant of S6.1)
New duration conditions are added after the SDP's detach (handbook S6.2) so the duration loss trains them.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F

import modules
from models import StochasticDurationPredictor


class CondSDP(StochasticDurationPredictor):
  """Official SDP with an extra condition added after the input detach; optional fixed reverse noise."""

  def forward(self, x, x_mask, w=None, cond=None, reverse=False, noise_scale=1.0, noise=None):
    x = torch.detach(x)
    x = self.pre(x)
    if cond is not None:
      x = x + cond
    x = self.convs(x, x_mask)
    x = self.proj(x) * x_mask
    if not reverse:
      logdet_tot_q = 0
      h_w = self.post_pre(w)
      h_w = self.post_convs(h_w, x_mask)
      h_w = self.post_proj(h_w) * x_mask
      e_q = torch.randn(w.size(0), 2, w.size(2), device=x.device, dtype=x.dtype) * x_mask
      z_q = e_q
      for flow in self.post_flows:
        z_q, logdet_q = flow(z_q, x_mask, g=(x + h_w))
        logdet_tot_q += logdet_q
      z_u, z1 = torch.split(z_q, [1, 1], 1)
      u = torch.sigmoid(z_u) * x_mask
      z0 = (w - u) * x_mask
      logdet_tot_q += torch.sum((F.logsigmoid(z_u) + F.logsigmoid(-z_u)) * x_mask, [1, 2])
      logq = torch.sum(-0.5 * (math.log(2 * math.pi) + (e_q ** 2)) * x_mask, [1, 2]) - logdet_tot_q
      logdet_tot = 0
      z0, logdet = self.log_flow(z0, x_mask)
      logdet_tot += logdet
      z = torch.cat([z0, z1], 1)
      for flow in self.flows:
        z, logdet = flow(z, x_mask, g=x, reverse=False)
        logdet_tot = logdet_tot + logdet
      nll = torch.sum(0.5 * (math.log(2 * math.pi) + (z ** 2)) * x_mask, [1, 2]) - logdet_tot
      return nll + logq
    flows = list(reversed(self.flows))
    flows = flows[:-2] + [flows[-1]]
    if noise is None:
      noise = torch.randn(x.size(0), 2, x.size(2), device=x.device, dtype=x.dtype)
    z = noise * noise_scale
    for flow in flows:
      z = flow(z, x_mask, g=x, reverse=True)
    return z[:, :1]


def zero_linear(i, o):
  m = nn.Linear(i, o)
  nn.init.zeros_(m.weight)
  nn.init.zeros_(m.bias)
  return m


def masked_mean(x, mask):
  """x [N,C,T], mask [N,1,T] -> [N,C]"""
  return (x * mask).sum(-1) / mask.sum(-1).clamp_min(1.0)


class RDiffusion(nn.Module):
  """Conditional DDPM over the 64-d block latent r (handbook S1.1-S1.5): 4 residual MLP blocks of width 256,
  sinusoidal time embedding 128, condition projected to 256; epsilon prediction with linear betas over 100 steps;
  ancestral sampling with the posterior variance (S1.3) at every step."""

  def __init__(self, r_dim, c_dim, width=256, steps=100):
    super().__init__()
    self.steps, self.r_dim = steps, r_dim
    b = torch.linspace(1e-4, 0.02, steps)
    a = 1 - b
    ab = torch.cumprod(a, 0)
    ab_prev = torch.cat([torch.ones(1), ab[:-1]])
    self.register_buffer("b", b)
    self.register_buffer("a", a)
    self.register_buffer("ab", ab)
    self.register_buffer("bt", b * (1 - ab_prev) / (1 - ab))
    self.t_emb = nn.Sequential(nn.Linear(128, width), nn.SiLU(), nn.Linear(width, width))
    self.c_emb = nn.Linear(c_dim, width)
    self.inp = nn.Linear(r_dim, width)
    self.blocks = nn.ModuleList([nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width), nn.SiLU(),
                                               nn.Linear(width, width)) for _ in range(4)])
    self.out = nn.Linear(width, r_dim)

  def _temb(self, t):
    half = 64
    f = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    e = t.float()[:, None] * f[None]
    return self.t_emb(torch.cat([e.sin(), e.cos()], -1))

  def eps(self, r_t, t, c):
    h = self.inp(r_t) + self._temb(t) + self.c_emb(c)
    for blk in self.blocks:
      h = h + blk(h)
    return self.out(h)

  def loss(self, r0, c):
    t = torch.randint(0, self.steps, (r0.size(0),), device=r0.device)
    e = torch.randn_like(r0)
    ab = self.ab[t][:, None]
    return ((self.eps(ab.sqrt() * r0 + (1 - ab).sqrt() * e, t, c) - e) ** 2).sum(-1).mean()

  @torch.no_grad()
  def sample(self, c, noise=None):
    """noise [N, steps + 1, r_dim] fixes every draw (common random numbers); None draws fresh noise."""
    n = c.size(0)
    if noise is None:
      noise = torch.randn(n, self.steps + 1, self.r_dim, device=c.device)
    r = noise[:, 0]
    for k, t in enumerate(range(self.steps - 1, -1, -1)):
      tt = torch.full((n,), t, device=c.device, dtype=torch.long)
      e = self.eps(r, tt, c)
      mean = (r - self.b[t] / (1 - self.ab[t]).sqrt() * e) / self.a[t].sqrt()
      r = mean + (self.bt[t].sqrt() * noise[:, k + 1] if t > 0 else 0.0)
    return r


class StreamVITS(nn.Module):
  def __init__(self, backbone, use_qwen=False, use_future=False, use_r=False, use_rho=False, use_diff=False,
               qwen_dim=2048, r_dim=64, rho_dim=16, hidden=192):
    super().__init__()
    self.flags = dict(use_qwen=use_qwen, use_future=use_future, use_r=use_r, use_rho=use_rho, use_diff=use_diff)
    self.use_diff = use_diff
    self.use_qwen, self.use_future, self.use_r, self.use_rho = use_qwen, use_future, use_r, use_rho
    self.hidden = hidden
    self.enc_p = backbone.enc_p
    self.dp = CondSDP(hidden, 192, 3, 0.5, 4)
    self.dp.load_state_dict(backbone.dp.state_dict())
    self.frozen = nn.ModuleDict({"enc_q": backbone.enc_q, "flow": backbone.flow, "dec": backbone.dec})
    for p in self.frozen.parameters():
      p.requires_grad_(False)
    if use_qwen:
      self.q_norm = nn.LayerNorm(qwen_dim)
      self.q_gate = nn.Linear(hidden + qwen_dim, hidden)
      self.q_val = zero_linear(qwen_dim, hidden)
      self.q_dur = zero_linear(qwen_dim, hidden)
    if use_future:
      self.f_norm = nn.LayerNorm(hidden)
      self.f_dur = zero_linear(hidden, hidden)
    if use_r:
      p_in = hidden + (hidden if use_future else 0) + (hidden if use_qwen else 0)
      self.q_last = nn.Linear(qwen_dim, hidden) if use_qwen else None
      self.r_prior = nn.Sequential(nn.Linear(p_in, 256), nn.GELU(), nn.Linear(256, 2 * r_dim))
      self.r_post_conv = nn.Conv1d(hidden + 1, 64, 3, padding=1)
      self.r_post = nn.Sequential(nn.Linear(p_in + 64, 256), nn.GELU(), nn.Linear(256, 2 * r_dim))
      self.r_dur = zero_linear(r_dim, hidden)
      self.r_ac = zero_linear(r_dim, hidden)
    if use_diff:
      assert use_r
      self.diff = RDiffusion(r_dim, p_in)
      self.r_aux = nn.Sequential(nn.Conv1d(hidden + r_dim, 192, 3, padding=1), nn.GELU(), nn.Conv1d(192, 1, 1))
    if use_rho:
      self.rho_rnn = nn.GRU(2, rho_dim, batch_first=True)
      self.rho_dur = zero_linear(rho_dim, hidden)

  def trainable_parameters(self):
    return [p for n, p in self.named_parameters() if not n.startswith("frozen.")]

  def encode_base(self, ids, lens):
    enc = self.enc_p
    x = enc.emb(ids) * math.sqrt(enc.hidden_channels)
    x = x.transpose(1, -1)
    mask = torch.unsqueeze(torch.arange(ids.size(1), device=ids.device)[None] < lens[:, None], 1).to(x.dtype)
    return enc.encoder(x * mask, mask), mask

  def future_features(self, cand_ids, cand_lens, n_inst, blk_mask):
    """Mean over M imagined continuations of the base encoder state at the block tokens (no gradient)."""
    with torch.no_grad():
      xc, _ = self.encode_base(cand_ids, cand_lens)
    m = cand_ids.size(0) // n_inst
    t = blk_mask.size(-1)
    xc = xc[:, :, :t].reshape(n_inst, m, self.hidden, t).mean(1)
    return self.f_norm(xc.transpose(1, 2)).transpose(1, 2) * blk_mask

  def forward_step(self, ids, lens, blk_mask, qfeat=None, q_last_idx=None, fut=None, hist=None, hist_lens=None,
                   w=None, r_mode="prior", r_noise=None):
    """Encode the visible prefix and build block conditions.

    ids [N,T], blk_mask [N,1,T], qfeat [N,T,2048], fut [N,192,T] (future_features), hist [N,H,2], w [N,1,T] real
    block durations (only needed for the r posterior). Returns x, m_p, logs_p, x_mask, dur_cond, kl_r.
    """
    x, x_mask = self.encode_base(ids, lens)
    dur_cond = torch.zeros_like(x)
    kl_r = x.new_zeros(())
    if self.use_qwen:
      q = self.q_norm(qfeat.float())
      g = torch.sigmoid(self.q_gate(torch.cat([x.transpose(1, 2), q], -1)))
      x = x + (g * self.q_val(q)).transpose(1, 2) * x_mask
      dur_cond = dur_cond + self.q_dur(q).transpose(1, 2) * x_mask
    if self.use_future:
      dur_cond = dur_cond + self.f_dur(fut.transpose(1, 2)).transpose(1, 2) * blk_mask
    ac_off = torch.zeros_like(x)
    if self.use_r:
      parts = [masked_mean(x, blk_mask)]
      if self.use_future:
        parts.append(masked_mean(fut, blk_mask))
      if self.use_qwen:
        parts.append(self.q_last(self.q_norm(qfeat.float())[torch.arange(ids.size(0)), q_last_idx]))
      p_in = torch.cat(parts, -1)
      mu_p, ls_p = self.r_prior(p_in).chunk(2, -1)
      ls_p = ls_p.clamp(-6, 3)
      if self.use_diff:
        # S1: Gaussian r prior replaced by a conditional DDPM; the posterior is kept informative by an auxiliary
        # block-duration reconstructor and only weakly tied to N(0, I) (kl_r holds that weak KL; the denoising
        # surrogate of S1.5 is returned in self.last_diff for the trainer).
        mu_p, ls_p = torch.zeros_like(mu_p), torch.zeros_like(ls_p)
      if r_mode == "posterior":
        h = self.r_post_conv(torch.cat([x.detach(), torch.log1p(w)], 1)) * blk_mask
        mu_q, ls_q = self.r_post(torch.cat([p_in, masked_mean(h, blk_mask)], -1)).chunk(2, -1)
        ls_q = ls_q.clamp(-6, 3)
        r = mu_q + torch.randn_like(mu_q) * torch.exp(ls_q)
        kl_r = (ls_p - ls_q + 0.5 * (torch.exp(2 * ls_q) + (mu_q - mu_p) ** 2) * torch.exp(-2 * ls_p) - 0.5).sum(-1)
        if self.use_diff:
          self.last_diff = self.diff.loss(r.detach(), p_in.detach())
          aux = self.r_aux(torch.cat([x.detach(), r[:, :, None].expand(-1, -1, x.size(-1))], 1))
          self.last_aux = (((aux - torch.log1p(w)) ** 2) * blk_mask).sum() / blk_mask.sum()
      elif self.use_diff:
        r = self.diff.sample(p_in, r_noise)
      elif r_mode == "prior_sample":
        r = mu_p + torch.randn_like(mu_p) * torch.exp(ls_p)
      else:
        r = mu_p
      dur_cond = dur_cond + self.r_dur(r)[:, :, None] * blk_mask
      ac_off = ac_off + self.r_ac(r)[:, :, None] * blk_mask
    if self.use_rho:
      if hist is not None and hist.size(1) > 0:
        packed = nn.utils.rnn.pack_padded_sequence(hist, hist_lens.clamp_min(1).cpu(), batch_first=True,
                                                   enforce_sorted=False)
        _, hN = self.rho_rnn(packed)
        rho = torch.tanh(hN[-1]) * (hist_lens > 0).float()[:, None]
      else:
        rho = x.new_zeros(ids.size(0), self.rho_rnn.hidden_size)
      dur_cond = dur_cond + self.rho_dur(rho)[:, :, None] * blk_mask
    stats = self.enc_p.proj(x + ac_off) * x_mask
    m_p, logs_p = torch.split(stats, self.enc_p.out_channels, dim=1)
    return x, m_p, logs_p, x_mask, dur_cond, kl_r

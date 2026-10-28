"""Streaming commit simulator: one utterance, units arrive one at a time, committed audio never changes.

plan(i, j) returns the block plan for unit i when units 0..j are visible: token durations (from the SDP with a
noise draw fixed per (utterance, unit), so every system and policy shares common random numbers) and the block's
prior statistics. commit(i, j) freezes that plan. Waveform rendering is incremental (render_incremental): the flow
inverse only sees committed frames and the decoder keeps a left context, so no right context is used either.
"""
import zlib

import numpy as np
import torch

from stream_common import prosody_y, token_is_word, unit_token_spans
from stream_data import imagined_ids
from text import intersperse, text_to_ids


def unit_rng(seed, utt_id, unit, salt=0):
  h = zlib.crc32(f"{utt_id}|{unit}|{salt}".encode())
  g = torch.Generator()
  g.manual_seed(seed * 7919 + h)
  return g


class Simulator:
  def __init__(self, model, row, dev, qwen=None, cands=None, n_cand=8, cand_words=3, noise_w=0.0, noise_z=0.667,
               seed=1234):
    self.m, self.row, self.dev = model, row, dev
    self.uid = row["utt_id"]
    self.ids = intersperse(text_to_ids(row["clean"]))
    self.spans = unit_token_spans(row)
    self.U = len(self.spans)
    self.is_word = token_is_word(row["clean"])
    self.noise_w, self.noise_z, self.seed = noise_w, noise_z, seed
    self.cands, self.n_cand, self.cand_words = cands, n_cand, cand_words
    self.qfeat_full = None
    if qwen is not None:
      from stream_common import token_char
      h = torch.from_numpy(qwen["h_mid"]).to(dev)
      qmap = torch.LongTensor([int(qwen["char_tok"][token_char(t)]) for t in range(len(self.ids))]).to(dev)
      self.qfeat_full = h[qmap]
    self.committed_dur = []   # per unit: token durations (np int)
    self.committed_zp = []    # per unit: [192, F] prior-space latents
    self.committed_y = []     # per unit: realised Y (log word frames, log1p pause frames)
    self.decisions = []       # per unit: visible unit index j at commit

  def _inputs(self, i, j):
    tv = self.spans[j][1]
    a, b = self.spans[i]
    ids = torch.LongTensor(self.ids[:tv])[None].to(self.dev)
    lens = torch.LongTensor([tv]).to(self.dev)
    blk = torch.zeros(1, 1, tv, device=self.dev)
    blk[..., a:b] = 1.0
    qfeat = self.qfeat_full[:tv][None] if self.qfeat_full is not None else None
    if self.committed_y:
      hist = torch.from_numpy(np.stack(self.committed_y)).float()[None].to(self.dev)
      hist_len = torch.LongTensor([len(self.committed_y)]).to(self.dev)
    else:
      hist, hist_len = torch.zeros(1, 1, 2, device=self.dev), torch.LongTensor([0]).to(self.dev)
    return ids, lens, blk, qfeat, hist, hist_len, tv, a, b

  def future_seqs(self, j):
    tv = self.spans[j][1]
    if j == self.U - 1:
      return [self.ids[:tv]] * self.n_cand
    prefix = self.row["clean"][:self.row["units"][j][1]]
    return [imagined_ids(prefix, c, self.cand_words) for c in self.cands[(self.uid, j)][:self.n_cand]]

  def _encode_seqs(self, seqs):
    lens = torch.LongTensor([len(s) for s in seqs]).to(self.dev)
    ids = torch.nn.utils.rnn.pad_sequence([torch.LongTensor(s) for s in seqs], batch_first=True).to(self.dev)
    return ids, lens

  @torch.no_grad()
  def plan(self, i, j):
    """Plan unit i with units 0..j visible."""
    m = self.m
    ids, lens, blk, qfeat, hist, hist_len, tv, a, b = self._inputs(i, j)
    fut = None
    if m.use_future:
      cids, clens = self._encode_seqs(self.future_seqs(j))
      fut = m.future_features(cids, clens, 1, blk)
    r_noise = None
    if getattr(m, "use_diff", False):   # S1: the diffusion draw of r is fixed per (utterance, unit)
      r_noise = torch.randn(1, m.diff.steps + 1, m.diff.r_dim, generator=unit_rng(self.seed, self.uid, i, 4)).to(self.dev)
    x, m_p, logs_p, x_mask, dur_cond, _ = m.forward_step(ids, lens, blk, qfeat=qfeat, q_last_idx=lens - 1, fut=fut,
                                                         hist=hist, hist_lens=hist_len, r_mode="prior", r_noise=r_noise)
    logw = m.dp(x, blk, cond=dur_cond, reverse=True, noise_scale=self.noise_w, noise=self._noise(i, tv))
    w = torch.ceil(torch.exp(logw) * blk)[0, 0, a:b].long().cpu().numpy()
    return {"dur": w, "m_p": m_p[0, :, a:b], "logs_p": logs_p[0, :, a:b], "h": x[0, :, a:b], "y": self.y_of(i, w)}

  def _noise(self, i, tv):
    """SDP noise for unit i, identical at every visible length (common random numbers)."""
    full = torch.randn(1, 2, len(self.ids) + 512, generator=unit_rng(self.seed, self.uid, i, 1))
    return full[:, :, :tv].to(self.dev)

  @torch.no_grad()
  def plan_imagined(self, i, j, seqs):
    """Plan unit i as if the visible text were each of seqs (prefix plus an imagined continuation), in one batch;
    for a text-only model (the E1 probe). Measures how sampled futures would move the plan; never committed."""
    m = self.m
    assert not (m.use_qwen or m.use_future), "the impact probe must be a text-only model"
    a, b = self.spans[i]
    ids, lens = self._encode_seqs(seqs)
    n, t = ids.shape
    blk = torch.zeros(n, 1, t, device=self.dev)
    blk[..., a:b] = 1.0
    hist = torch.zeros(n, 1, 2, device=self.dev)
    x, m_p, logs_p, x_mask, dur_cond, _ = m.forward_step(ids, lens, blk, hist=hist, hist_lens=torch.zeros_like(lens))
    noise = self._noise(i, t).expand(n, -1, -1)
    logw = m.dp(x, blk, cond=dur_cond, reverse=True, noise_scale=self.noise_w, noise=noise)
    w = torch.ceil(torch.exp(logw) * blk)[:, 0, a:b].long().cpu().numpy()
    return [self.y_of(i, wk) for wk in w]

  def y_of(self, i, w):
    a, b = self.spans[i]
    wm = self.is_word[a:b]
    return prosody_y([[float(w[wm].sum()), float(w[~wm].sum())]])[0]

  def commit(self, i, j, p):
    assert i == len(self.committed_dur), "units are committed in order"
    rep = torch.from_numpy(np.repeat(np.arange(len(p["dur"])), p["dur"])).to(self.dev)
    m_f, ls_f = p["m_p"][:, rep], p["logs_p"][:, rep]
    eps = torch.randn(m_f.shape, generator=unit_rng(self.seed, self.uid, i, 2)).to(self.dev)
    self.committed_zp.append(m_f + eps * torch.exp(ls_f) * self.noise_z)
    self.committed_dur.append(p["dur"])
    self.committed_y.append(p["y"])
    self.decisions.append(j)

  @torch.no_grad()
  def render_incremental(self, left_ctx=48):
    """Committed unit by committed unit: flow inverse over committed frames only, decoder with left context."""
    flow, dec = self.m.frozen["flow"], self.m.frozen["dec"]
    hop = 256
    zp_all = torch.zeros(192, 0, device=self.dev)
    z_done = torch.zeros(192, 0, device=self.dev)
    pieces = []
    for zp in self.committed_zp:
      if zp.size(1) == 0:
        continue
      zp_all = torch.cat([zp_all, zp], 1)
      mask = torch.ones(1, 1, zp_all.size(1), device=self.dev)
      z = flow(zp_all[None], mask, reverse=True)[0]
      z_new = z[:, z_done.size(1):]
      z_done = torch.cat([z_done, z_new], 1)
      s = max(0, z_done.size(1) - z_new.size(1) - left_ctx)
      o = dec(z_done[None, :, s:])[0, 0]
      pieces.append(o[(z_done.size(1) - z_new.size(1) - s) * hop:])
    return torch.cat(pieces).float().cpu().numpy() if pieces else np.zeros(0, dtype=np.float32)

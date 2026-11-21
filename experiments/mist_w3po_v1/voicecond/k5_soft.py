"""Robust K5 soft voice condition helpers (reports/mist_w3po_v1/robust_k5_softvoice_prereg.zh-CN.md section 3).

b in R^{32x6}: the BOUNDED FSQ code of the global voice tokens (FSQ.bound(project_in(...)), in (-2.0015, 1.0015));
FSQ level index of dimension k = round(b_k) + 2, token id = sum_k level_k * 4^k.
Soft assignment (fixed a priori, no temperature): exact per-dimension barycentric (multilinear) interpolation between the
two nearest FSQ levels on the 4^6 grid -> per position at most 64 corner tokens with product weights. On level centres it
is the hard token's one-hot; for the affine decoder condition E_p[d(t)] = d(b).

    corners(b)          -> idx (32, 64) int64 token ids, w (32, 64) float64 weights (rows sum to 1)
    centres(ids)        -> b of the hard tokens (level - 2), the sanity input
    d_cont(se, b)       -> BiCodec decoder speaker condition from b (speaker_encoder.detokenize without the round)
"""
from __future__ import annotations

import numpy as np
import torch

_BITS = ((np.arange(64)[:, None] >> np.arange(6)[None, :]) & 1)      # (64, 6) corner choice per dimension
_BASIS = 4 ** np.arange(6)


def corners(b):
    b = np.clip(np.asarray(b, dtype=np.float64), -2.0, 1.0)            # (32, 6)
    lo = np.minimum(np.floor(b), 0.0)                                   # -2, -1, 0
    f = b - lo                                                          # in [0, 1]
    lev = (lo + 2)[:, None, :] + _BITS[None]                            # (32, 64, 6) level indices 0..3
    idx = (lev * _BASIS).sum(-1).astype(np.int64)
    w = np.where(_BITS[None] == 1, f[:, None, :], 1.0 - f[:, None, :]).prod(-1)
    return idx, w


def centres(ids):
    ids = np.asarray(ids, dtype=np.int64)
    return np.stack([(ids // 4 ** k) % 4 for k in range(6)], -1).astype(np.float64) - 2.0


@torch.no_grad()
def d_cont(se, b) -> torch.Tensor:
    """se = BiCodec speaker_encoder; b (32, 6) or (B, 32, 6) -> d (B, out_dim), same path as se.detokenize(ids) with the
    normalised code q = round(b) / 2 replaced by b / 2 (b clipped to the level range [-2, 1])."""
    rq = se.quantizer
    p = next(se.project.parameters())
    t = torch.as_tensor(np.clip(np.asarray(b, dtype=np.float64), -2.0, 1.0) / 2.0, dtype=p.dtype, device=p.device)
    t = t[None] if t.ndim == 2 else t
    zq = rq.project_out(t * rq.scales[0]).transpose(1, 2)               # (B, latent, 32), as detokenize
    return se.project(zq.reshape(zq.shape[0], -1))

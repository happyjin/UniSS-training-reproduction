"""TurnCRF, end-to-end version (reports/mist_w3po_v1/crf_novel_ideas.zh-CN.md §3.2, §9, §11; scheme 2): the semi-CRF
speaker-turn layer of turn_crf.py placed INSIDE the model, between the frozen diarizer's posteriors P and the speaker
projector, trained jointly with the LoRA by the translation loss plus the CRF marginal-likelihood loss.

Layer (TurnLayer): at a read, input the visible prefix P[:tau] (tau = post_until_ms // 80 frames, 80 ms, K = 8 slots)
and optionally the Thinker audio-encoder states e (aligned to the 80 ms frames); output

    P' = P + g * (pi - P),        pi_t^k = sum_{c ni k} mu_t^c   (right-censored prefix marginals, §3.2)

with a learnable scalar gate g initialised to 0, so the initial model is exactly v8 (bit-identical projector input).
mu are the frame-label marginals of the right-censored semi-CRF at tau, from an explicit log-space forward-backward
written with differentiable torch ops only (lists + stack, no in-place writes), so autograd flows from the projector
input through the marginals into every potential (structured attention, Kim et al. 2017).

Potentials: turn_crf.SemiTurnCRF's type-shared parametrisation (content u = a_type * log P(c) + b_type summed over the
segment, duration table + survival potential for the open last segment, relation transitions A, boundary weights w on
dP, start / end by type), recomputed here device-agnostically (turn_crf builds its label-space tensors on the CPU) and
checked equal to turn_crf's on the CPU (test_turn_layer.py). With e: a slot-agnostic MLP(e_t) -> 9 numbers per frame,
per-frame type biases (empty / single / pair) added to u, and boundary scores added to the transition into frame t:
left[type of the new label] + right[type of the old label]. Its last layer is zero-initialised (potentials = CRF stage).
All CRF math runs in float64 (as the CPU stage); the output is cast back to P's dtype.

Auxiliary loss (weight 1, pre-registered): (log Z~_tau - log Z~_tau,y) / tau, the right-censored prefix marginal
likelihood of the reference frame labels y at the row's read prefix; with maximal segments log Z~_tau,y is the score
of y's unique segmentation (turn_crf.semi_path_score, re-implemented here device-agnostically).

Init: the CPU module's neutral init by default (a = 1, everything else 0 = independent P); init_from(params.pt of the
CPU stage, train_turn_crf.py) loads trained semi-CRF parameters (opt-in: the CPU stage failed its pre-registered stop
rule, reports/mist_w3po_v1/turn_crf_cpu_stage.zh-CN.md). The choice is stored in the layer ("init_src").

Note: at g = 0 the translation loss gives no gradient to the CRF parameters (d P'/d phi = g d pi/d phi); they move by
the auxiliary loss first and receive translation-loss gradient as soon as g != 0 (one optimiser step).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from experiments.mist_w3po_v1.crf.turn_crf import N_TYPES, NEG, SemiTurnCRF

EPS_P = 1e-4
N_E_OUT = 9                  # 3 type biases on u, 3 left (new label type) + 3 right (old label type) boundary scores
LAYER_FILE = "turn_layer.pt"
CPU_STAGE_SEMI = "eval_outputs/mist_w3po_v1/crf/turn_v1/semi"


# ----------------------------------------------------------------------------------------------------------------
# differentiable semi-CRF forward-backward (batched, equal lengths)
# ----------------------------------------------------------------------------------------------------------------

def semi_fb(u, tr, start_c, end_c, d_c, ds_c, censored: bool = True):
    """Exact semi-CRF log-partition and frame-label marginals, autograd through both.

    u (B, T, C) content (summed over a segment); tr (B, T, C, C), tr[:, j, c', c] the potential of a boundary between
    frames j-1 and j from c' to c (index 0 unused); start_c / end_c (C,); d_c / ds_c (L, C) duration / survival
    potentials for q = 1..L. censored: the last segment is open (survival potential, no end score), the streaming
    prefix distribution of §3.2. Returns logZ (B,), mu (B, T, C) with mu[b, t, c] = p(frame t has label c).
    """
    B, T, C = u.shape
    L = d_c.shape[0]
    d_last = ds_c if censored else d_c
    e_last = torch.zeros_like(end_c) if censored else end_c
    U = torch.cat([u.new_zeros(B, 1, C), u.cumsum(1)], 1)
    Us, trs = U.unbind(1), tr.unbind(1)
    d_rev, dl_rev = d_c.flip(0), d_last.flip(0)
    # forward: beta_i = prefixes whose next segment starts at i; G_i = beta_i - U_i; alpha_j = segments ending at j
    beta = [start_c.expand(B, C)]
    G = [beta[0]]
    alpha = []
    for j in range(1, T + 1):
        qm = min(L, j)
        x = torch.stack(G[j - qm:], 1)                                   # i = j-qm .. j-1  <->  q = qm .. 1
        a_j = torch.logsumexp(x + (dl_rev if j == T else d_rev)[L - qm:], 1) + Us[j]
        alpha.append(a_j)
        if j < T:
            b_j = torch.logsumexp(a_j.unsqueeze(-1) + trs[j], 1)
            beta.append(b_j)
            G.append(b_j - Us[j])
    logZ = torch.logsumexp(alpha[-1] + e_last, -1)
    # backward: bin_i = suffixes starting with a segment at i; bout_j = suffixes after a segment ending at j
    H = [None] * (T + 1)                                                 # H_j = U_j + bout_j
    H[T] = Us[T] + e_last
    bins, bouts = [None] * T, [None] * (T + 1)
    for i in range(T - 1, -1, -1):
        qm = min(L, T - i)
        x = torch.stack(H[i + 1:i + qm + 1], 1)                          # j = i+1 .. i+qm  <->  q = 1 .. qm
        dur = torch.cat([d_c[:qm - 1], d_last[qm - 1:qm]], 0) if i + qm == T else d_c[:qm]
        bins[i] = torch.logsumexp(x + dur, 1) - Us[i]
        if i >= 1:
            bouts[i] = torch.logsumexp(trs[i] + bins[i].unsqueeze(1), 2)
            H[i] = Us[i] + bouts[i]
    lz = logZ[:, None, None]
    S = torch.exp(torch.stack(beta, 1) + torch.stack(bins, 1) - lz)    # a segment starts at i (i = 0..T-1)
    cumE = u.new_zeros(B, 1, C)
    if T > 1:
        E = torch.exp(torch.stack(alpha[:-1], 1) + torch.stack(bouts[1:T], 1) - lz)   # a segment ends at j = 1..T-1
        cumE = torch.cat([cumE, E.cumsum(1)], 1)
    mu = S.cumsum(1) - cumE
    return logZ, mu.clamp(0.0, 1.0)


def path_score(u, tr, start_c, end_c, d_c, ds_c, y, censored: bool = True):
    """Score of the unique maximal segmentation of the frame labels y (B, T) (= log Z_y; equal-length rows)."""
    B, T, C = u.shape
    L = d_c.shape[0]
    dev = u.device
    ar = torch.arange(T, device=dev).unsqueeze(0).expand(B, T)
    change = torch.zeros(B, T, dtype=torch.bool, device=dev)
    change[:, 1:] = y[:, 1:] != y[:, :-1]
    newrun = change.clone()
    newrun[:, 0] = True
    runstart = torch.cummax(torch.where(newrun, ar, torch.zeros_like(ar)), 1).values
    q = ar - runstart + 1
    ends = torch.zeros_like(change)
    ends[:, :-1] = change[:, 1:]
    ends[:, -1] = True                                                   # the last run ends at T
    cont = u.gather(2, y.unsqueeze(-1)).squeeze(-1).sum(1)
    yprev = torch.cat([y[:, :1], y[:, :-1]], 1)
    trv = tr.reshape(B, T, C * C).gather(2, (yprev * C + y).unsqueeze(-1)).squeeze(-1)
    trs = torch.where(change, trv, trv.new_zeros(())).sum(1)
    qi = (q - 1).clamp(max=L - 1)
    dsel = torch.where(ends & (ar < T - 1), d_c[qi, y], u.new_zeros(()))
    last_d = (ds_c if censored else d_c)[qi[:, -1], y[:, -1]]
    too_long = NEG * (q > L).any(1).to(u.dtype)
    e_last = 0.0 if censored else end_c[y[:, -1]]
    return start_c[y[:, 0]] + cont + trs + dsel.sum(1) + last_d + e_last + too_long


# ----------------------------------------------------------------------------------------------------------------
# the layer
# ----------------------------------------------------------------------------------------------------------------

class TurnLayer(nn.Module):
    """P[:tau] (and e) -> P + g (pi - P); see the module docstring."""

    def __init__(self, K: int = 8, L: int = 257, Q: int = 50, e_dim: int = 0, e_hidden: int = 256):
        super().__init__()
        self.cfg = {"K": K, "L": L, "Q": Q, "e_dim": e_dim, "e_hidden": e_hidden}
        # persistent, so a bare state dict (the shared trainer's extra.pt) rebuilds the layer (from_state_dict)
        self.register_buffer("cfg_vec", torch.tensor([K, L, Q, e_dim, e_hidden], dtype=torch.float64))
        self.crf = SemiTurnCRF(K=K, max_active=2, L=L, Q=Q)
        sp = self.crf.space
        self.register_buffer("M", sp.M.clone(), persistent=False)          # (C, K) float64
        self.register_buffer("types", sp.types.clone(), persistent=False)  # (C,)
        self.register_buffer("rel", sp.rel.clone(), persistent=False)      # (C, C), -1 on the diagonal
        self.g = nn.Parameter(torch.zeros((), dtype=torch.float32))
        self.e_mlp = None
        if e_dim:
            self.e_mlp = nn.Sequential(nn.Linear(e_dim, e_hidden), nn.GELU(), nn.Linear(e_hidden, N_E_OUT))
            nn.init.zeros_(self.e_mlp[-1].weight)
            nn.init.zeros_(self.e_mlp[-1].bias)
        self.init_src = "neutral (turn_crf.SemiTurnCRF init: a=1, rest 0)"

    # ---- parameters / init ----
    def init_from(self, params_pt: str) -> None:
        ck = torch.load(params_pt, map_location="cpu")
        assert ck.get("kind") == "semi", ck.get("kind")
        mk = ck.get("model_kw", {})
        assert int(mk.get("L", self.cfg["L"])) == self.cfg["L"] and int(mk.get("Q", self.cfg["Q"])) == self.cfg["Q"], mk
        self.crf.load_state_dict(ck["state_dict"])
        self.init_src = f"{params_pt} (CPU-stage semi-CRF, step {ck.get('step')})"

    def init_auto(self, cpu_stage: str = CPU_STAGE_SEMI) -> None:
        """The CPU-stage semi-CRF parameters when that training finished (DONE), else the neutral init."""
        d = Path(cpu_stage)
        if (d / "DONE").exists() and (d / "params.pt").exists():
            self.init_from(str(d / "params.pt"))

    def crf_parameters(self):
        return list(self.crf.parameters()) + ([] if self.e_mlp is None else list(self.e_mlp.parameters()))

    # ---- potentials (float64, on P's device) ----
    def potentials(self, P: torch.Tensor, e: torch.Tensor | None = None) -> dict:
        """P (B, T, K) -> semi-CRF potentials, as turn_crf.SemiTurnCRF.potentials (+ the e terms)."""
        dt = torch.float64
        P = P.to(dt)
        c = self.crf
        M, ty, rel = self.M.to(dt), self.types, self.rel
        a, b = c.a.to(dt), c.b.to(dt)
        Pc = P.clamp(EPS_P, 1.0 - EPS_P)
        lp, lq = torch.log(Pc), torch.log1p(-Pc)
        u = a[ty] * (lq.sum(-1, keepdim=True) + (lp - lq) @ M.T) + b[ty]
        dP = torch.cat([torch.zeros_like(P[..., :1, :]), P[..., 1:, :] - P[..., :-1, :]], -2)
        s = dP @ M.T
        inter = torch.einsum("pk,nk,...tk->...tpn", M, M, dP)
        enter = s.unsqueeze(-2) - inter
        leave = -(s.unsqueeze(-1) - inter)
        base = torch.where(rel >= 0, c.A.to(dt)[rel.clamp(min=0)], torch.full(rel.shape, NEG, dtype=dt, device=P.device))
        w = c.w.to(dt)
        tr = base + w[0] * enter + w[1] * leave
        L, Q = self.cfg["L"], c.d_tab.shape[1]
        d_tab, d_tail = c.d_tab.to(dt), c.d_tail.to(dt)
        if L <= Q:
            d = d_tab[:, :L]
        else:
            extra = torch.arange(1, L - Q + 1, dtype=dt, device=P.device) / (L - Q)
            d = torch.cat([d_tab, d_tab[:, -1:] + d_tail[:, None] * extra], 1)
        ds = torch.logcumsumexp(d.flip(-1), -1).flip(-1)
        if e is not None and self.e_mlp is not None:
            h = self.e_mlp(e.to(self.e_mlp[0].weight.dtype)).to(dt)       # (B, T, 9)
            u = u + h[..., 0:N_TYPES][..., ty]
            left = h[..., N_TYPES:2 * N_TYPES][..., ty]                   # by the new label's type
            right = h[..., 2 * N_TYPES:][..., ty]                         # by the old label's type
            tr = tr + left.unsqueeze(-2) + right.unsqueeze(-1)
        return {"u": u, "tr": tr, "start_c": c.start.to(dt)[ty], "end_c": c.end.to(dt)[ty],
                "d_c": d[ty].T.contiguous(), "ds_c": ds[ty].T.contiguous()}

    # ---- inference / training ----
    def marginals(self, P: torch.Tensor, e: torch.Tensor | None = None, censored: bool = True):
        """P (B, T, K) -> (logZ (B,), mu (B, T, C), pi (B, T, K), potentials)."""
        p = self.potentials(P, e)
        logZ, mu = semi_fb(p["u"], p["tr"], p["start_c"], p["end_c"], p["d_c"], p["ds_c"], censored)
        return logZ, mu, mu @ self.M.to(mu.dtype), p

    def forward(self, P: torch.Tensor, e: torch.Tensor | None = None, return_aux: bool = False):
        """P (T, K) or (B, T, K) visible prefix (equal lengths) -> P + g (pi - P) in P's dtype. return_aux: also the
        (logZ, potentials) of this call, for aux_loss without a second forward."""
        squeeze = P.dim() == 2
        Pb = P.unsqueeze(0) if squeeze else P
        eb = None if e is None else (e.unsqueeze(0) if e.dim() == 2 else e)
        if Pb.shape[1] == 0:
            return (P, None) if return_aux else P
        logZ, _, pi, pot = self.marginals(Pb, eb)
        P64 = Pb.to(torch.float64)
        out = (P64 + self.g.to(torch.float64) * (pi - P64)).to(P.dtype)
        out = out[0] if squeeze else out
        return (out, (logZ, pot)) if return_aux else out

    def aux_loss(self, P: torch.Tensor, y: torch.Tensor, e: torch.Tensor | None = None, cached=None) -> torch.Tensor:
        """(log Z~_tau - log Z~_tau,y) / tau per row (B,): right-censored at the prefix end tau = T. y (B, T) labels."""
        if P.dim() == 2:
            P, y = P.unsqueeze(0), y.unsqueeze(0)
            e = None if e is None else e.unsqueeze(0)
        if cached is None:
            p = self.potentials(P, e)
            logZ, _ = semi_fb(p["u"], p["tr"], p["start_c"], p["end_c"], p["d_c"], p["ds_c"], True)
        else:
            logZ, p = cached
        sy = path_score(p["u"], p["tr"], p["start_c"], p["end_c"], p["d_c"], p["ds_c"], y.to(p["u"].device).long(), True)
        return (logZ - sy) / P.shape[1]

    # ---- checkpoint ----
    def export(self) -> dict:
        return {"cfg": self.cfg, "init_src": self.init_src,
                "state_dict": {k: v.detach().cpu() for k, v in self.state_dict().items()}}

    @classmethod
    def from_export(cls, ck: dict) -> "TurnLayer":
        m = cls(**ck["cfg"])
        m.load_state_dict(ck["state_dict"])
        m.init_src = ck.get("init_src", "")
        return m

    @classmethod
    def from_state_dict(cls, sd: dict) -> "TurnLayer":
        K, L, Q, e_dim, e_hidden = (int(round(float(v))) for v in sd["cfg_vec"])
        m = cls(K=K, L=L, Q=Q, e_dim=e_dim, e_hidden=e_hidden)
        m.load_state_dict(sd)          # parameters are cast to the layer's dtypes (float64 CRF, float32 g / MLP)
        return m

    def summary(self) -> dict:
        c = self.crf
        return {"g": float(self.g), "a": [round(float(v), 4) for v in c.a], "b": [round(float(v), 4) for v in c.b],
                "w": [round(float(v), 4) for v in c.w], "e_mlp": self.e_mlp is not None, "init_src": self.init_src}


def load_layer(path: str) -> TurnLayer:
    """A TurnLayer from: a turn_layer.pt file (TurnLayer.export) or a dir holding one; a train_turncrf_sft.py
    checkpoint (train_crf_sft.py format: <ckpt>/extra.pt, extra_params[0] = the layer's state dict) or its extra.pt."""
    p = Path(path)
    if p.is_dir():
        p = p / LAYER_FILE if (p / LAYER_FILE).exists() else p / "extra.pt"
    ck = torch.load(p, map_location="cpu")
    if "extra_params" in ck:
        sds = [sd for sd in ck["extra_params"] if "cfg_vec" in sd]
        if len(sds) != 1:
            raise SystemExit(f"{p}: expected exactly one TurnLayer state dict in extra_params, found {len(sds)}")
        m = TurnLayer.from_state_dict(sds[0])
        m.init_src = f"{p} (step {ck.get('step')})"
        return m
    return TurnLayer.from_export(ck)


# ----------------------------------------------------------------------------------------------------------------
# helpers shared by training and inference
# ----------------------------------------------------------------------------------------------------------------

def _interp_rows(x: torch.Tensor, xp: torch.Tensor, fp: torch.Tensor) -> torch.Tensor:
    """np.interp(x, xp, fp[:, c]) for every column c in float64 (train_crf_sft.interp_rows, same formula)."""
    fp64 = fp.to(torch.float64)
    if xp.numel() == 1:
        return fp64[:1].expand(x.numel(), fp.shape[1])
    j = (torch.searchsorted(xp, x, right=True) - 1).clamp(0, xp.numel() - 2)
    slope = (fp64[j + 1] - fp64[j]) / (xp[j + 1] - xp[j])[:, None]
    y = slope * (x - xp[j])[:, None] + fp64[j]
    y = torch.where((x < xp[0])[:, None], fp64[:1].expand_as(y), y)
    return torch.where((x >= xp[-1])[:, None], fp64[-1:].expand_as(y), y)


def e_to_frames(e_tok: torch.Tensor, n_frames: int, dur_s: float) -> torch.Tensor:
    """Audio-encoder states [n, D] at the audio-token centres (i + 0.5) dur / n -> [n_frames, D] float32 at the 80 ms
    frame centres (k + 0.5) 0.08 s, linear (np.interp semantics): exactly what the shared trainer hands the hook
    (train_crf_sft.enc_to_frames; equality tested in test_turn_layer.py)."""
    n = e_tok.shape[0]
    if n_frames <= 0 or n == 0:
        return e_tok.new_zeros((0, e_tok.shape[1]), dtype=torch.float32)
    dev = e_tok.device
    tc = (torch.arange(n, device=dev, dtype=torch.float64) + 0.5) * dur_s / n
    fc = (torch.arange(n_frames, device=dev, dtype=torch.float64) + 0.5) * 0.08
    return _interp_rows(fc, tc, e_tok.detach()).to(torch.float32)


def apply_prefix_np(layer: TurnLayer, post: np.ndarray, tau: int, e_frames: torch.Tensor | None = None,
                    device=None) -> np.ndarray:
    """Inference: post (F80, K) float32 numpy, the first tau frames visible -> a copy whose first tau frames are the
    layer output (the rest unchanged)."""
    tau = min(int(tau), len(post))
    if tau <= 0:
        return post
    dev = device or next(layer.parameters()).device
    with torch.no_grad():
        x = torch.from_numpy(np.ascontiguousarray(post[:tau])).to(dev)
        y = layer(x, e_frames).cpu().numpy()
    out = post.copy()
    out[:tau] = y
    return out


def describe(layer: TurnLayer) -> str:
    return json.dumps({"cfg": layer.cfg, **layer.summary()})


__all__ = ["TurnLayer", "semi_fb", "path_score", "load_layer", "e_to_frames", "apply_prefix_np", "LAYER_FILE",
           "CPU_STAGE_SEMI", "describe"]

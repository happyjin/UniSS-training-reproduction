"""Idea 2 shared helpers: stream loading, oracle speech clocks, the window->prompt map M, bootstrap."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
I2 = Path(__import__("os").environ.get("IDEA2_ROOT", str(REPO / "data/open_voicecond_v1/idea2")))   # IDEA2_ROOT: e.g. the cvt root
OUTD = REPO / "eval_outputs/mist_w3po_v1/voicecond_idea2"
N_PILOT = {"train": 32, "dev": 4, "test": 4}
DIAR_DELAY = 1.04
T_EVAL = [1, 2, 4, 6, 8, 10, 15, 20, 30, 40, 50, 60]


def is_pilot(sid):
    split, _, i = sid.split("_")
    return int(i) < N_PILOT[split]


def sids(which="pilot", split=None):
    out = []
    for p in sorted((I2 / "streams").iterdir()):
        if not p.is_dir() or p.name.startswith(".") or not (p / "meta.json").exists():
            continue
        if which == "pilot" and not is_pilot(p.name):
            continue
        if split and not p.name.startswith(split + "_"):
            continue
        out.append(p.name)
    return out


def latest_refs():
    if not (I2 / "refs/ustar.npz").exists():
        return None
    k, p = 2, I2 / "refs/ustar.npz"
    while (I2 / f"refs/ustar_v{k}.npz").exists():
        p, k = I2 / f"refs/ustar_v{k}.npz", k + 1
    return p


def load_refs():
    z = np.load(latest_refs(), allow_pickle=True)
    return {s: (u.reshape(-1), g) for s, u, g in zip(z["spk"], z["ustar"], z["gstar"])}


def load_stream(sid, L):
    meta = json.load(open(I2 / "streams" / sid / "meta.json"))
    w = np.load(I2 / f"pre/win_L{L:g}" / f"{sid}.npz")
    f = np.load(I2 / f"pre/feats_L{L:g}" / f"{sid}.npz")
    K = min(len(w["end_s"]), len(f["end_s"]))
    d = {"meta": meta, "end": w["end_s"][:K].astype(np.float64)}
    for k in ("z_mix", "z_speech", "z_vhat"):
        d[k] = w[k][:K].reshape(K, -1).astype(np.float64)
    for k in ("v", "slot", "pi", "cov", "single", "slot_raw", "pi_raw"):
        d[k] = f[k][:K]
    for k in ("slot2spk", "ov"):
        d[k] = f[k]
    return d


def speech_clock(meta, j):
    """oracle speech intervals of true speaker j -> function t (s of speech) -> stream time when reached (or None)"""
    spk = meta["speakers"][j]
    iv = sorted((g["start"], g["end"]) for g in meta["segments"] if g["spk"] == spk)
    merged = []
    for s, e in iv:
        if merged and s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    cum = np.cumsum([e - s for s, e in merged]) if merged else np.zeros(0)
    total = float(cum[-1]) if len(cum) else 0.0

    def tau(t):
        if t > total + 1e-9:
            return None
        i = int(np.searchsorted(cum, t - 1e-9))
        s, e = merged[i]
        before = cum[i] - (e - s)
        return s + (t - before)
    return tau, total


def slot_for_speaker(d, j):
    """diarizer slot used for true speaker j (evaluation only): the slot matched to j with the largest overlap,
    else the slot with the largest overlap with j"""
    ov = d["ov"]
    cand = [s for s in range(ov.shape[0]) if d["slot2spk"][s] == j]
    if cand:
        return max(cand, key=lambda s: ov[s, j])
    return int(ov[:, j].argmax())


# ---------------------------------------------------------------- map M (window space -> prompt space)
def fit_maps(X, Y):
    """X, Y [n, 192]; returns dict name -> (W or diag, b, kind)"""
    out = {}
    out["bias"] = ("bias", None, (Y - X).mean(0))
    xm, ym = X.mean(0), Y.mean(0)
    Xc, Yc = X - xm, Y - ym
    a = (Xc * Yc).sum(0) / ((Xc * Xc).sum(0) + 1e-12)
    out["diag"] = ("diag", a, ym - a * xm)
    lam = 1e-3 * np.trace(Xc.T @ Xc) / X.shape[1]
    W = np.linalg.solve(Xc.T @ Xc + lam * np.eye(X.shape[1]), Xc.T @ Yc)
    out["full"] = ("full", W, ym - xm @ W)
    return out


def apply_map(Mk, x):
    kind, W, b = Mk
    x = np.asarray(x, np.float64)
    if kind == "bias":
        return x + b
    if kind == "diag":
        return x * W + b
    return x @ W + b


def save_map(path, Mk):
    kind, W, b = Mk
    np.savez(path, kind=kind, W=np.zeros(0) if W is None else W, b=b)


def load_map(path):
    z = np.load(path)
    kind = str(z["kind"])
    return (kind, None if kind == "bias" else z["W"], z["b"])


# ---------------------------------------------------------------- stats
def cluster_boot(values_by_cluster, fn, n=2000, seed=20261119):
    """values_by_cluster: list (one per stream) of per-unit arrays/tuples; fn(list_of_clusters) -> scalar"""
    rng = np.random.default_rng(seed)
    C = len(values_by_cluster)
    est = fn(values_by_cluster)
    bs = []
    for _ in range(n):
        idx = rng.integers(0, C, C)
        bs.append(fn([values_by_cluster[i] for i in idx]))
    bs = np.array(bs)
    return float(est), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))

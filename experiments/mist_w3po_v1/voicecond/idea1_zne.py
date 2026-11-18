"""Idea 1 -- Zero-Nuisance Extrapolation (ZNE) of the BiCodec global code (plan
reports/mist_w3po_v1/game_voice_condition_ideas_plan.zh-CN.md §2.1, §2.9, §2.12; pre-registration
reports/mist_w3po_v1/voicecond_idea1_prereg.zh-CN.md).

Notation: y = prompt (16 kHz), bh = background estimate (HDemucs: y - vocals stem, time-aligned, zero-padded to len(y)),
phi = BiCodecPhi.phi_wave (pre-bound code u in R^{32x6}, volume normalisation inside, first 6 s / tiled).
Amplitude path y_lam = y + (lam - 1) * bh, power ratio t = lam^2.  The three pre-registered families:

  R-t    Richardson in t, nodes t in {1, 2}:            u = 2 f(t=1) - f(t=2)          (2 encodes)
  Q-lam  quadratic in lam, nodes lam in {1, sqrt2, 2}:  u = sum_i g_i f(lam_i), Lagrange weights to lam = 0  (3 encodes)
  M-anc  Moebius with anchor u_b = phi(bh), t in {1,2}: A = f1 - u_b, B = f2 - u_b, rho = <A-2B, B-A>/|A-2B|^2 clipped
         at 0, u = u_b + (1 + rho) A                                                         (3 encodes)

CLI (esports streams, evaluation only): ZNE voice tokens of every stream of a talker run, keyed by the prompt stem,
with the sha1 of the prompt file (consumed by idea1_si_infer_vo.py):
    python -m experiments.mist_w3po_v1.voicecond.idea1_zne --items <talker_in/items.jsonl> --vocals <dir of <id>.wav>
        --family R-t --out <tokens.jsonl> [--device cuda:0|cpu] [--check-results <talker/results.jsonl>]
Refuses an existing --out.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

REPO = Path(__file__).resolve().parents[3]
SQ2 = math.sqrt(2.0)
FAMILIES = ("R-t", "Q-lam", "M-anc")
# amplitude multipliers lam each family encodes (lam = inf means the anchor phi(bh))
NODES = {"R-t": (1.0, SQ2), "Q-lam": (1.0, SQ2, 2.0), "M-anc": (1.0, SQ2, float("inf"))}


def lagrange_to_zero(xs):
    xs = list(xs)
    return [float(np.prod([xj / (xj - xi) for j, xj in enumerate(xs) if j != i])) for i, xi in enumerate(xs)]


Q_W = lagrange_to_zero(NODES["Q-lam"])          # weights at lam = 1, sqrt2, 2


def estimate(family: str, f: dict) -> np.ndarray:
    """f: {lam: u} with the family's nodes (inf = anchor). Returns u_hat (same shape as u)."""
    if family == "R-t":
        return 2.0 * f[1.0] - f[SQ2]
    if family == "Q-lam":
        return sum(w * f[x] for w, x in zip(Q_W, NODES["Q-lam"]))
    if family == "M-anc":
        ub = f[float("inf")]
        A, B = f[1.0] - ub, f[SQ2] - ub
        d = (A - 2 * B).ravel()
        den = float(d @ d)
        rho = max(0.0, float(d @ (B - A).ravel()) / den) if den > 0 else 0.0
        return ub + (1.0 + rho) * A
    raise ValueError(family)


def mobius_rho(f1, f2, ub) -> float:
    A, B = f1 - ub, f2 - ub
    d = (A - 2 * B).ravel()
    den = float(d @ d)
    return max(0.0, float(d @ (B - A).ravel()) / den) if den > 0 else 0.0


def path_waves(y: np.ndarray, bh: np.ndarray, lams) -> list:
    """y_lam = y + (lam-1) bh; lam = inf -> bh alone"""
    return [bh.copy() if math.isinf(l) else y + (l - 1.0) * bh for l in lams]


def read_mono64(p) -> np.ndarray:
    x, sr = sf.read(str(p), dtype="float64", always_2d=True)
    assert sr == 16000, (p, sr)
    return x[:, 0]


def aligned_bh(y: np.ndarray, vocals: np.ndarray) -> np.ndarray:
    m = min(len(y), len(vocals))
    bh = np.zeros_like(y)
    bh[:m] = y[:m] - vocals[:m]
    return bh


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--items", required=True)
    ap.add_argument("--vocals", required=True)
    ap.add_argument("--family", required=True, choices=FAMILIES)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--check-results", default="", help="baseline results.jsonl: phi(y) tokens must equal its voice")
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    sys.path.insert(0, str(REPO))
    from experiments.mist_w3po_v1.voicecond.bicodec_phi import BiCodecPhi
    P = BiCodecPhi(a.device)
    base = {}
    if a.check_results:
        base = {r["id"]: r["voice"] for r in map(json.loads, open(a.check_results, encoding="utf-8")) if r.get("voice")}
    rows, seen = [], {}
    n_same = n_chk = 0
    for it in map(json.loads, open(a.items, encoding="utf-8")):
        p = Path(it["voice_audio"])
        p = p if p.is_absolute() else REPO / p
        key = p.stem
        if key in seen:
            continue
        y = read_mono64(p)
        voc = read_mono64(Path(a.vocals) / f"{it['id']}.wav")
        bh = aligned_bh(y, voc)
        lams = NODES[a.family]
        U = P.phi_wave(path_waves(y, bh, lams))
        f = {l: U[i] for i, l in enumerate(lams)}
        u_hat = estimate(a.family, f)
        ids = P.tokens(u_hat).tolist()
        raw_ids = P.tokens(f[1.0]).tolist()
        exact = P.tokens(P.phi_path(str(p))).tolist()
        assert raw_ids == exact, f"phi_wave != tokenizer path for {p}"
        if it["id"] in base:
            n_chk += 1
            assert [int(v) for v in base[it["id"]]] == raw_ids, f"raw tokens differ from the baseline run for {it['id']}"
        n_same += int(ids == raw_ids)
        r = {"key": key, "id": it["id"], "family": a.family, "ids": [int(v) for v in ids],
             "raw_ids": [int(v) for v in raw_ids], "sha1": hashlib.sha1(p.read_bytes()).hexdigest(),
             "ham_vs_raw": int(sum(x != z for x, z in zip(ids, raw_ids))),
             "du_vs_raw": float(np.linalg.norm(u_hat - f[1.0]))}
        if a.family == "M-anc":
            r["rho"] = mobius_rho(f[1.0], f[SQ2], f[float("inf")])
        seen[key] = r
        rows.append(r)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "x", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    hams = np.array([r["ham_vs_raw"] for r in rows])
    print(json.dumps({"out": str(out), "n": len(rows), "family": a.family, "checked_vs_baseline": n_chk,
                      "unchanged": n_same, "ham_vs_raw_median": float(np.median(hams)),
                      "ham_vs_raw_mean": float(hams.mean())}))


if __name__ == "__main__":
    main()

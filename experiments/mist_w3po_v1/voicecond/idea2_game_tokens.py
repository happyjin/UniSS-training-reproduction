"""Idea 2 M1b (prereg v2 §G): voice-token tables for the esports demo (OOD-2, evaluation only) from the causal
contraction-model posterior. Per direction and excerpt (meeting): HDemucs on the whole excerpt (idea2_demucs.separate),
window codes z_k = phi(mix window) (L = 2 s, hop 0.5 s), v~_k (idea2_feats2.window_vt), slot / pi from the existing
meeting-level Nemotron lowlat posteriors (the C2 window posteriors are exact slices of them; same rule as idea2_feats).
Per item (stream) <win>__S<k>: t_commit = window start_s + first block write_ms / 1000; the slot-(k-1) posterior over all
windows with end + 1.04 s <= t_commit (ShrinkStream math, prereg v2 §D) -> u = s + b_M -> 32 FSQ tokens.
Rows {key = baseline prompt stem, id, ids, sha1 of the baseline prompt file, raw_ids, ham_vs_raw, t_commit, n_win,
pi_sum, lam_mean, ids_const, ids_V} for idea1_si_infer_vo.py. Parameters: --fit (pilot-train fit); nothing is fitted here.

    python -I experiments/mist_w3po_v1/voicecond/idea2_game_tokens.py --dir zh2en|en2zh --fit <dir> --e20 <dir>
        --out <tokens.jsonl> [--device cuda:0]
Refuses an existing --out; per-excerpt features are cached in data/open_voicecond_v1/idea2/game/<dir>/<m>.npz.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from experiments.mist_w3po_v1.voicecond.idea2_common import DIAR_DELAY, I2  # noqa: E402
from experiments.mist_w3po_v1.voicecond.idea2_feats2 import window_vt  # noqa: E402
from experiments.mist_w3po_v1.voicecond.idea2_shrink import ShrinkParams  # noqa: E402

SR, HOP, L, PI_MIN = 16000, 0.5, 2.0, 0.3
GD = REPO / "data/game_esports_si_v1"


def meeting_audio(d, m):
    return GD / d / ("session_audio/mix" if d == "zh2en" else "session_audio/ihm_mix") / (
        f"{m}.wav" if d == "zh2en" else f"{m}.Mix-Headset.wav")


def slot_pi(post, ends):
    K = len(ends)
    slot = np.full(K, -1, np.int64)
    pi = np.zeros(K)
    for k, e in enumerate(ends):
        a, b = int(round((e - L) / 0.08)), min(len(post), int(round(e / 0.08)))
        p = post[a:b]
        if len(p) == 0:
            continue
        j = int(p.sum(0).argmax())
        v = float((p[:, j] * np.prod(1 - np.delete(p, j, axis=1), axis=1)).mean())
        if v >= PI_MIN:
            slot[k], pi[k] = j, v
    return slot, pi


def features(d, m, P, model, dev):
    cache = I2 / "game" / d / f"{m}.npz"
    if cache.exists():
        z = np.load(cache)
        return {k: z[k] for k in z.files}
    from experiments.mist_w3po_v1.voicecond.idea2_demucs import separate
    y, sr = sf.read(str(meeting_audio(d, m)), dtype="float32")
    if y.ndim > 1:
        y = y.mean(1)
    assert sr == SR, sr
    vh, bh = separate(model, y, dev)
    W = int(L * SR)
    K = int((len(y) - W) // (HOP * SR)) + 1
    st = (np.arange(K) * HOP * SR).astype(int)
    ends = (st + W) / SR
    z = np.concatenate([P.phi_wave([y[s:s + W] for s in st[i:i + 256]]) for i in range(0, K, 256)]).reshape(K, -1)
    zv = np.concatenate([P.phi_wave([vh[s:s + W] for s in st[i:i + 256]]) for i in range(0, K, 256)]).reshape(K, -1)
    vt = window_vt(vh, bh, ends, L)
    post = torch.load(GD / d / "diar/nemotron_lowlat" / f"{m}.post.pt", weights_only=False)["post80"].float().numpy()
    slot, pi = slot_pi(post, ends)
    out = {"end": ends, "z": z, "zv": zv, "vt": vt, "slot": slot, "pi": pi}
    cache.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache.parent / f".{m}.partial.npz"
    np.savez(tmp, **out)
    os.rename(tmp, cache)
    return out


def posterior(p: ShrinkParams, lang, f, s, t, mode):
    avail = f["end"] + DIAR_DELAY
    sel = np.nonzero((f["slot"] == s) & (avail <= t + 1e-9))[0]
    if mode == "V":     # reference: pi-weighted running mean of HDemucs-vocals window codes (prompt space via b_M)
        if not len(sel):
            return p.mu0 + p.bM, sel
        w = f["pi"][sel]
        return (w[:, None] * f["zv"][sel]).sum(0) / w.sum() + p.bM, sel
    lam = p.lam(f["vt"][sel])
    y = f["z"][sel] - (1 - lam)[:, None] * p.cvec(lang)[None]
    w = f["pi"][sel]
    Rfi = np.linalg.inv(p.R * (p.L / HOP))
    S0i = np.linalg.inv(p.Sig0)
    m = np.linalg.solve(S0i + (w * lam * lam).sum() * Rfi, S0i @ p.mu0 + Rfi @ ((w * lam)[:, None] * y).sum(0))
    return m + p.bM, sel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, choices=["zh2en", "en2zh"])
    ap.add_argument("--fit", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--run", default="", help="baseline speech run (default game_b_<dir>__t1c)")
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        sys.exit(f"refuse to overwrite {out}")
    lang = "zh" if a.dir == "zh2en" else "en"
    pf = ShrinkParams.load(Path(a.fit) / "full.npz")
    pc = ShrinkParams.load(Path(a.fit) / "const.npz")
    S = REPO / "eval_outputs/mist_w3po_v1/speech" / (a.run or f"game_b_{a.dir}__t1c")
    items = [json.loads(l) for l in open(S / "talker_in/items.jsonl", encoding="utf-8")]
    blocks = {r["id"]: r["blocks"] for r in map(json.loads, open(S / "talker_in/blocks.jsonl", encoding="utf-8"))}
    W = {w["id"]: w for w in map(json.loads, open(REPO / f"eval_outputs/mist_w3po_v1/data/windows_game_{a.dir}_c2g06_v1/"
                                                    f"windows_test_online_{a.dir}.jsonl", encoding="utf-8"))}
    from torchaudio.models import hdemucs_high
    from experiments.mist_w3po_v1.voicecond.bicodec_phi import BiCodecPhi
    dev = torch.device(a.device)
    model = hdemucs_high(sources=["drums", "bass", "other", "vocals"])
    model.load_state_dict(torch.load(Path("/opt/dlami/nvme/jasonleeeli/cache/torch/hub/checkpoints/hdemucs_high_trained.pt"),
                                     map_location="cpu"))
    model = model.eval().to(dev)
    P = BiCodecPhi(a.device)
    feats, rows, seen = {}, [], set()
    for it in items:
        p = Path(it["voice_audio"])
        p = p if p.is_absolute() else REPO / p
        if p.stem in seen:
            continue
        seen.add(p.stem)
        wid, k = it["id"].rsplit("__S", 1)
        w = W[wid]
        m = w["meeting"]
        if m not in feats:
            feats[m] = features(a.dir, m, P, model, dev)
            print(f"[game] {a.dir} {m}: {len(feats[m]['end'])} windows", flush=True)
        f = feats[m]
        t = w["start_s"] + min(b["write_ms"] for b in blocks[it["id"]]) / 1000.0
        s = int(k) - 1
        u, sel = posterior(pf, lang, f, s, t, "full")
        uc, _ = posterior(pc, lang, f, s, t, "full")
        uv, _ = posterior(pf, lang, f, s, t, "V")
        ids, idc, idv = (P.tokens(x.reshape(32, 6).astype(np.float32)).tolist() for x in (u, uc, uv))
        raw = P.tokens(P.phi_path(str(p))).tolist()
        rows.append({"key": p.stem, "id": it["id"], "mode": "shrink_full", "fit": str(a.fit), "ids": [int(v) for v in ids],
                     "raw_ids": [int(v) for v in raw], "sha1": hashlib.sha1(p.read_bytes()).hexdigest(),
                     "ham_vs_raw": int(sum(x != y for x, y in zip(ids, raw))), "t_commit": round(t, 3),
                     "n_win": int(len(sel)), "pi_sum": float(f["pi"][sel].sum()),
                     "lam_mean": float(np.average(pf.lam(f["vt"][sel]), weights=f["pi"][sel])) if len(sel) else None,
                     "ids_const": [int(v) for v in idc], "ids_V": [int(v) for v in idv]})
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "x", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    h = np.array([r["ham_vs_raw"] for r in rows])
    nw = np.array([r["n_win"] for r in rows])
    print(json.dumps({"out": str(out), "n": len(rows), "ham_vs_raw_median": float(np.median(h)),
                      "n_win_median": float(np.median(nw)), "no_obs": int((nw == 0).sum())}))


if __name__ == "__main__":
    main()

"""Robust wave 1, K7v (prereg addendum K6-K9 / K7v; derivation reports/mist_w3po_v1/robust_novel_ideas_v1.zh-CN.md 6.3):
online soft noise-subspace (Wiener) projection on the frame features of the BiCodec speaker encoder loaded with Idea 1 C's
weights (the ECAPA output `feats`, d = 1536 channels x frames, before the Perceiver). Own implementation of 6.3; no SEEN
code, no SEEN layer rule. One setting.

Statistics (all with the C-weight encoder, second moments about 0, i.e. the energy the projection a - sum g_i (q_i'a) q_i
acts on):
  Sigma_g   global noise moment: frames of the 3 s of stream background bg.wav just before the item start (the turn-gap
            analogue; [0, 3] s if the item starts earlier), for 400 cvt TRAIN teacher items (200 per direction, sha1(id)
            order; cvt train streams use only the non-held-out train backgrounds), each scaled by the volume-normalisation
            gain of its item's noisy prompt (the level the background has inside the prompt the encoder sees)
  identity  speaker subspace U_id: the per-prompt time-averaged features of every clean prompt cvt/refs/<spk>/p*.wav of
            the cvt TRAIN speakers, centred over prompts; eigenvectors covering 95 % of that (between-prompt) energy
  Sigma_s   clean frame moment over the same clean train prompts
Per item (the P1b items): S = sum a a' over the frames of the item's own 3 s pre-item background (same gain rule),
n frames; inverse-Wishart posterior mean with nu = d + 1 + 50:  Sigma_hat = (S + 50 Sigma_g) / (n + 50).
Q = eigenvectors of Sigma_hat covering 95 % of its trace, kept iff ||U_id' q|| < 0.1; sigma2_n,i = q_i' Sigma_hat q_i,
sigma2_s,i = q_i' Sigma_s q_i, gamma_i = sigma2_n / (sigma2_n + sigma2_s); on every frame of the noisy prompt's features
a <- a - sum_i gamma_i (q_i' a) q_i, then Perceiver -> project_in -> FSQ (exactly BiCodecPhi._u_from_ref otherwise).

    <venvs/vllm_omni_030 python> -m experiments.mist_w3po_v1.voicecond.wave1_k7v_tokens --items <prep>/items.jsonl \
        --out <dir> [--device cuda:0]
-> <dir>/tokens.jsonl {id, ids, n95, n_kept, gamma_mean, ham_vs_C}, <dir>/k7v.json. Refuses an existing --out.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
CV = REPO / "data/open_voicecond_v1/idea2/cvt"
C_CKPT = REPO / "eval_outputs/mist_w3po_v1/voicecond_idea1/m2/ctrl/encoder.pt"
SR, GAP_S, PRIOR_N, ENERGY, COS_MAX, N_TRAIN_BG = 16000, 3.0, 50, 0.95, 0.1, 200


def h(s: str) -> int:
    return int(hashlib.sha1(s.encode()).hexdigest(), 16)


def read64(p) -> np.ndarray:
    x, sr = sf.read(str(p), dtype="float64", always_2d=True)
    assert sr == SR, (p, sr)
    return x[:, 0]


def gain_of(vn, y: np.ndarray) -> float:
    z = vn(np.asarray(y, np.float64))
    return float(np.dot(z, y) / max(np.dot(y, y), 1e-20))


def top_energy(evals: np.ndarray, evecs: np.ndarray, frac: float):
    o = np.argsort(evals)[::-1]
    ev, V = np.clip(evals[o], 0, None), evecs[:, o]
    k = int(np.searchsorted(np.cumsum(ev) / ev.sum(), frac) + 1)
    return ev[:k], V[:, :k]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--items", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    t0 = time.time()
    log = lambda s: print(f"{time.strftime('%F %T')} {s}", flush=True)  # noqa: E731
    from experiments.mist_w3po_v1.voicecond.bicodec_phi import BiCodecPhi
    from experiments.mist_w3po_v1.voicecond.idea1_tokens import load_encoder
    P = load_encoder(BiCodecPhi(a.device), str(C_CKPT))
    P.model.eval()
    dev = P.device

    @torch.no_grad()
    def feats(w: np.ndarray) -> torch.Tensor:
        """waveform (already at the level the encoder sees) -> ECAPA frame features (1536, T) fp64"""
        x = torch.from_numpy(np.asarray(w, np.float32))[None].to(dev)
        mel = P.model.mel_transformer(x).squeeze(1)
        _, f = P.se.speaker_encoder(mel.transpose(1, 2), True)
        return f[0].double()

    @torch.no_grad()
    def u_from_feats(f: torch.Tensor) -> np.ndarray:
        x = P.se.perceiver_sampler(f[None].float().transpose(1, 2)).transpose(1, 2).transpose(1, 2)
        return P.rq.project_in(x)[0].cpu().numpy()

    def gap_bg(bg: np.ndarray, start: float) -> np.ndarray:
        a0 = max(0.0, start - GAP_S)
        i = int(round(a0 * SR))
        return bg[i:i + int(GAP_S * SR)]

    # ---- Sigma_g from cvt train items' pre-item backgrounds
    d = 1536
    Sg, ng = torch.zeros(d, d, dtype=torch.float64, device=dev), 0
    for s in ("zh2en", "en2zh"):
        rows = sorted(map(json.loads, open(CV / f"teacher_in/items_{s}_train.jsonl", encoding="utf-8")),
                      key=lambda r: h(r["id"]))[:N_TRAIN_BG]
        bgc = {}
        for r in rows:
            sid, k = r["id"].rsplit("__", 1)
            if sid not in bgc:
                bgc = {sid: (read64(CV / "streams" / sid / "bg.wav"), json.load(open(CV / "streams" / sid / "meta.json")))}
            bg, meta = bgc[sid]
            g = gain_of(P._vn, read64(r["audio"]))
            f = feats(g * gap_bg(bg, float(meta["segments"][int(k)]["start"])))
            Sg += f @ f.T
            ng += f.shape[1]
    Sg /= ng
    log(f"[Sigma_g] {ng} frames from {2 * N_TRAIN_BG} cvt train items")

    # ---- identity subspace and clean moment from the cvt train speakers' clean prompts
    train_spk = set()
    for m in (CV / "streams").glob("cvtr_*/meta.json"):
        train_spk |= set(json.load(open(m))["speakers"])
    means, Ss, ns, npr = [], torch.zeros(d, d, dtype=torch.float64, device=dev), 0, 0
    for spk in sorted(train_spk):
        dd = CV / "refs" / spk.replace(":", "_")
        for p in sorted(dd.glob("p*.wav")):
            f = feats(P.ref_clip(read64(p)))
            means.append(f.mean(1))
            Ss += f @ f.T
            ns += f.shape[1]
            npr += 1
    Ss /= ns
    M = torch.stack(means)
    M = M - M.mean(0, keepdim=True)
    ev_id, U_id = top_energy(*[t.cpu().numpy() for t in torch.linalg.eigh(M.T @ M / len(M))], ENERGY)
    log(f"[identity] {npr} clean prompts of {len(train_spk)} train speakers; identity rank (95 %) = {U_id.shape[1]}")
    U_id_t = torch.from_numpy(U_id).to(dev)

    items = [json.loads(l) for l in open(a.items, encoding="utf-8")]
    items = [r for r in items if r["p1b"]]
    res, bgc = [], {}
    for r in items:
        if r["sid"] not in bgc:
            bgc = {r["sid"]: read64(CV / "streams" / r["sid"] / "bg.wav")}
        y = read64(r["noisy"])
        g = gain_of(P._vn, y)
        fb = feats(g * gap_bg(bgc[r["sid"]], r["start"]))
        Sh = (fb @ fb.T + PRIOR_N * Sg) / (fb.shape[1] + PRIOR_N)
        ev, V = top_energy(*[t.cpu().numpy() for t in torch.linalg.eigh(Sh)], ENERGY)
        Vt = torch.from_numpy(V).to(dev)
        cos = torch.linalg.norm(U_id_t.T @ Vt, dim=0)
        keep = cos < COS_MAX
        Q = Vt[:, keep]
        s2n = torch.einsum("di,de,ei->i", Q, Sh, Q)
        s2s = torch.einsum("di,de,ei->i", Q, Ss, Q)
        gam = s2n / (s2n + s2s)
        fa = feats(P.ref_clip(y))
        u0 = u_from_feats(fa)
        f2 = fa - Q @ (gam[:, None] * (Q.T @ fa))
        u = u_from_feats(f2)
        ids = P.tokens(u)
        c_ref = np.asarray(r["tok"]["C"])
        assert (P.tokens(u0) == c_ref).all(), f"feature path != C tokens for {r['id']}"
        res.append({"id": r["id"], "ids": [int(v) for v in ids], "n95": int(len(ev)), "n_kept": int(keep.sum()),
                    "gamma_mean": float(gam.mean()) if len(gam) else 0.0,
                    "energy_removed": float((torch.linalg.norm(fa - f2) ** 2 / torch.linalg.norm(fa) ** 2).item()),
                    "ham_vs_C": int((ids != c_ref).sum()), "gap_frames": int(fb.shape[1])})
    out.mkdir(parents=True)
    with open(out / "tokens.jsonl", "x", encoding="utf-8") as fh:
        for x in res:
            fh.write(json.dumps(x) + "\n")
    q = lambda k: {"median": float(np.median([x[k] for x in res])), "mean": float(np.mean([x[k] for x in res]))}  # noqa: E731
    info = {"n": len(res), "d": d, "nu": d + 1 + PRIOR_N, "sigma_g_frames": ng, "identity_rank": int(U_id.shape[1]),
            "clean_prompts": npr, "n95": q("n95"), "n_kept": q("n_kept"), "gamma_mean": q("gamma_mean"),
            "energy_removed": q("energy_removed"), "ham_vs_C": q("ham_vs_C"),
            "items_changed": int(sum(x["ham_vs_C"] > 0 for x in res)), "sec": round(time.time() - t0, 1)}
    (out / "k7v.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
    log(json.dumps(info))


if __name__ == "__main__":
    main()

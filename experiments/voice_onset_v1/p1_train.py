"""voice_onset_v1 P1 training (reports/voice_onset_v1/P1_P3_prereg.zh-CN.md sections 1-3): end-to-end hidden-Markov onset
with a differentiable isotropic Kalman speaker posterior over 1 s TitaNet chunk embeddings.
    m_t = normalize(a0 mu0 + sum_{k<=t} w_k y'_k),  y'_k = normalize(y_k + A y_k),  w_k = softplus(MLP_theta([y_k, rms_k]))
    onset g in {tau, tau+1, tau+2} s, stop hazard h_t = sigmoid(MLP_phi([log P_t, t-tau, w_t, cos(y'_t, m_{t-1})]))
    p(g = tau+j) = h_j prod_{i<j}(1-h_i) (forced stop at tau+K);  J = E_p[cos(m_g, c_spk)] - lambda E_p[g - tau]
c_spk = mean clean full-clip TitaNet embedding of the speaker's OTHER clips. lambda = 0.045 / s, K = 2, Adam 1e-3,
batch 256, 5 epochs, seed 20261121, last epoch kept. CPU (nemo_diar torch).
    python -m experiments.voice_onset_v1.p1_train --emb eval_outputs/voice_onset_v1/p1/emb --out eval_outputs/voice_onset_v1/p1/train
Refuses an existing --out.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SEED = 20261121
LAM = 0.045
K = 2
TAU_P = {1: 0.035, 2: 0.345, 3: 0.417, 4: 0.203}


class Onset(nn.Module):
    def __init__(self, mu0: np.ndarray):
        super().__init__()
        self.register_buffer("mu0", torch.tensor(mu0, dtype=torch.float32))
        self.A = nn.Parameter(torch.zeros(192, 192))
        self.w_net = nn.Sequential(nn.Linear(193, 64), nn.Tanh(), nn.Linear(64, 1))
        self.a0_raw = nn.Parameter(torch.tensor(0.0))
        self.h_net = nn.Sequential(nn.Linear(4, 16), nn.Tanh(), nn.Linear(16, 1))

    def posterior(self, Y, R):
        """Y (B, T, 192) chunk embeddings, R (B, T) rms dB -> m (B, T+1, 192) [m_0 = mu0], logP (B, T+1), w (B, T), y' """
        yp = F.normalize(Y + Y @ self.A.T, dim=-1)
        w = F.softplus(self.w_net(torch.cat([Y, R[..., None] / 10.0], -1)))[..., 0]
        a0 = F.softplus(self.a0_raw)
        S = a0 * self.mu0 + torch.cumsum(w[..., None] * yp, 1)                       # (B, T, 192)
        S = torch.cat([(a0 * self.mu0).expand(Y.shape[0], 1, -1), S], 1)
        P = torch.cat([a0.expand(Y.shape[0], 1), a0 + torch.cumsum(w, 1)], 1)
        return F.normalize(S, dim=-1), torch.log(P), w, yp

    def hazards(self, m, logP, w, yp, tau):
        """stop hazards at t = tau + j, j = 0..K-1 (t in seconds = chunk count heard) -> (B, K)"""
        B = m.shape[0]
        ar = torch.arange(B)
        hs = []
        for j in range(K):
            t = tau + j                                                     # chunks heard
            feats = torch.stack([logP[ar, t], torch.full((B,), float(j)), w[ar, t - 1],
                                 (yp[ar, t - 1] * m[ar, t - 1]).sum(-1)], -1)
            hs.append(torch.sigmoid(self.h_net(feats))[:, 0])
        return torch.stack(hs, 1)

    def forward(self, Y, R, tau, c):
        m, logP, w, yp = self.posterior(Y, R)
        B = Y.shape[0]
        ar = torch.arange(B)
        h = self.hazards(m, logP, w, yp, tau)
        surv = torch.ones(B)
        p, U = [], []
        for j in range(K + 1):
            pj = surv * h[:, j] if j < K else surv
            p.append(pj)
            U.append((m[ar, tau + j] * c).sum(-1))
            if j < K:
                surv = surv * (1 - h[:, j])
        p, U = torch.stack(p, 1), torch.stack(U, 1)                        # (B, K+1)
        J = (p * (U - LAM * torch.arange(K + 1).float())).sum(1)
        return J, p, U, h, w, m


def centroids(prefix_s_full, spk):
    """c_i = normalize(sum of the speaker's OTHER clips' clean full embeddings); None if the speaker has one clip"""
    E = prefix_s_full.astype(np.float64)
    tot = {}
    for i, s in enumerate(spk):
        tot[s] = tot.get(s, 0) + E[i]
    C = np.zeros_like(E)
    ok = np.zeros(len(spk), bool)
    cnt = {s: spk.count(s) for s in set(spk)}
    for i, s in enumerate(spk):
        if cnt[s] < 2:
            continue
        v = tot[s] - E[i]
        C[i] = v / (np.linalg.norm(v) + 1e-9)
        ok[i] = True
    return C.astype(np.float32), ok


def load(emb: Path, split: str):
    meta = [json.loads(l) for l in open(emb / f"{split}_meta.jsonl", encoding="utf-8")]
    a = {k: np.load(emb / f"{split}_{k}.npy").astype(np.float32) for k in
         ("chunk_y", "chunk_s", "prefix_y", "prefix_s", "rms_y", "rms_s")}
    return meta, a


def taus(n, seed):
    rng = np.random.default_rng(seed)
    return rng.choice(list(TAU_P), size=n, p=list(TAU_P.values()))


@torch.no_grad()
def evaluate(model, meta, a, C, ok, tau_seed, sel=None):
    """per condition: baselines and model at tau / at the deployed onset (first hazard >= 0.5)"""
    idx = np.nonzero(ok if sel is None else (ok & sel))[0]
    tau = taus(len(meta), tau_seed)[idx]
    out = {}
    for c in ("y", "s"):
        Y = torch.tensor(a[f"chunk_{c}"][idx]); R = torch.tensor(a[f"rms_{c}"][idx]); Cc = torch.tensor(C[idx])
        T = torch.tensor(tau)
        J, p, U, h, w, m = model(Y, R, T, Cc)
        ar = np.arange(len(idx))
        whole = (a[f"prefix_{c}"][idx][ar, tau - 1] * C[idx]).sum(-1)
        unif = np.stack([F.normalize(Y[i, : tau[i]].mean(0), dim=0).numpy() for i in ar])
        unif = (unif * C[idx]).sum(-1)
        # deployed onset: first j with h_j >= 0.5, else K
        hj = h.numpy()
        jd = np.array([next((j for j in range(K) if hj[i, j] >= 0.5), K) for i in ar])
        Ud = U.numpy()[ar, jd]
        wt = np.concatenate([w.numpy()[i, : tau[i] + K] for i in ar])
        out[{"y": "noisy", "s": "clean"}[c]] = {
            "n": int(len(idx)),
            "cos_whole_prefix_tau": round(float(whole.mean()), 4), "cos_uniform_tau": round(float(unif.mean()), 4),
            "cos_model_tau": round(float(U[:, 0].mean()), 4),
            "cos_model_tau+1": round(float(U[:, 1].mean()), 4), "cos_model_tau+2": round(float(U[:, 2].mean()), 4),
            "J_model": round(float(J.mean()), 4), "J_never_wait": round(float(U[:, 0].mean()), 4),
            "expected_wait_s": round(float((p * torch.arange(K + 1).float()).sum(1).mean()), 3),
            "deployed_wait_s": round(float(jd.mean()), 3), "deployed_wait_dist": np.bincount(jd, minlength=K + 1).tolist(),
            "cos_model_deployed": round(float(Ud.mean()), 4),
            "w_mean": round(float(wt.mean()), 4)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=5)
    a_ = ap.parse_args()
    out = Path(a_.out)
    if out.exists():
        sys.exit(f"拒绝覆盖已有输出: {out}")
    out.mkdir(parents=True)
    torch.manual_seed(SEED)
    torch.set_num_threads(32)
    emb = Path(a_.emb)
    meta, a = load(emb, "train")
    spk = [r["speaker"] for r in meta]
    C, ok = centroids(a["prefix_s"][:, 5], spk)
    mu0 = C[ok].mean(0); mu0 /= np.linalg.norm(mu0)
    hmeta, ha = load(emb, "heldout")
    hC, hok = centroids(ha["prefix_s"][:, 5], [r["speaker"] for r in hmeta])
    hdev = np.array([r.get("split") == "dev" for r in hmeta]); htest = ~hdev
    model = Onset(mu0)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    idx = np.nonzero(ok)[0]
    tau_tr = taus(len(meta), SEED)
    rng = np.random.default_rng(SEED)
    log = []
    print(json.dumps({"train_clips": int(len(idx)), "speakers": len(set(spk)), "heldout_ok": int(hok.sum())}), flush=True)
    log.append({"epoch": 0, "heldout_dev": evaluate(model, hmeta, ha, hC, hok, SEED + 1, hdev)})
    for ep in range(1, a_.epochs + 1):
        ex = np.concatenate([np.stack([idx, np.zeros_like(idx)], 1), np.stack([idx, np.ones_like(idx)], 1)])
        ex = ex[rng.permutation(len(ex))]
        tot, nb = 0.0, 0
        for b0 in range(0, len(ex), 256):
            e = ex[b0:b0 + 256]
            ii, cc = e[:, 0], e[:, 1]
            Y = np.where(cc[:, None, None] == 0, a["chunk_y"][ii], a["chunk_s"][ii])
            R = np.where(cc[:, None] == 0, a["rms_y"][ii], a["rms_s"][ii])
            J, *_ = model(torch.tensor(Y), torch.tensor(R), torch.tensor(tau_tr[ii]), torch.tensor(C[ii]))
            loss = -J.mean()
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss); nb += 1
        ev = evaluate(model, hmeta, ha, hC, hok, SEED + 1, hdev)
        log.append({"epoch": ep, "train_loss": round(tot / nb, 5), "heldout_dev": ev})
        print(json.dumps(log[-1]), flush=True)
    torch.save({"state": model.state_dict(), "mu0": mu0, "lam": LAM, "K": K}, out / "model.pt")
    # P1 checks on held-out dev (+ test report), mechanism stats
    dev, test = log[-1]["heldout_dev"], evaluate(model, hmeta, ha, hC, hok, SEED + 1, htest)
    with torch.no_grad():
        i2 = np.nonzero(hok & hdev)[0]
        _, _, wy, _ = model.posterior(torch.tensor(ha["chunk_y"][i2]), torch.tensor(ha["rms_y"][i2]))
        _, _, ws, _ = model.posterior(torch.tensor(ha["chunk_s"][i2]), torch.tensor(ha["rms_s"][i2]))
    wy, ws = wy.numpy(), ws.numpy()
    bvr = np.array([hmeta[i]["bvr"] for i in i2])
    G = {"G1_noisy_model_tau_minus_whole": round(dev["noisy"]["cos_model_tau"] - dev["noisy"]["cos_whole_prefix_tau"], 4)}
    G["G1_PASS"] = G["G1_noisy_model_tau_minus_whole"] >= -0.005
    G["G2_w_noisy_over_clean"] = round(float(wy.mean() / ws.mean()), 4)
    G["G2_corr_w_bvr_noisy"] = round(float(np.corrcoef(wy.mean(1), bvr)[0, 1]), 4)
    G["G2_corr_w_rms_noisy"] = round(float(np.corrcoef(wy.reshape(-1), ha["rms_y"][i2].reshape(-1))[0, 1]), 4)
    G["G3_wait_noisy_clean"] = [dev["noisy"]["deployed_wait_s"], dev["clean"]["deployed_wait_s"]]
    G["G3_J_gain_vs_never_wait"] = {k: round(dev[k]["J_model"] - dev[k]["J_never_wait"], 4) for k in ("noisy", "clean")}
    summ = {"checks": G, "heldout_dev": dev, "heldout_test_report": test, "log": log,
            "a0": float(F.softplus(model.a0_raw)), "A_fro": float(model.A.norm())}
    with open(out / "summary.json", "x", encoding="utf-8") as fh:
        json.dump(summ, fh, ensure_ascii=False, indent=1)
    print(json.dumps({"checks": G, "dev": dev}, ensure_ascii=False, indent=1), flush=True)


if __name__ == "__main__":
    main()

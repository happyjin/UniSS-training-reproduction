"""Experiment A step 3 (reports/mist_w3po_v1/robust_a_svanchor_prereg.zh-CN.md sections 3-4): SV-anchored voice-code
estimator, ONE setting, then the S1 dev precheck and the token tables.
  input   x = [TitaNet-large(noisy prompt) (192, L2-normalised), flatten(phi_V1(noisy prompt)) (192)], standardised with
          the training-set mean / std
  model   u_hat = phi_V1 + reshape(MLP(x), 32x6); MLP 384 -> 1024 -> 1024 -> 192, GELU, dropout 0.1, last layer
          zero-initialised (u_hat = V1's code at step 0)
  loss    mean_j ||d(u_hat_j) - d(u*_j)||^2 (squared L2 norm over the d-vector) + 0.1 MSE(u_hat, u*) (mean over the
          32x6 pre-bound code, as V1's loss); d(u) = speaker_encoder.project(
          flatten(project_out(clip(bound(u), -2, 1) / 2 * scale))) = detokenize with the round removed (k5_soft.d_cont,
          differentiable); u* = phi_0(clean crop) (training), phi_0(clean-twin prompt) (dev)
  optim   AdamW lr 1e-3, weight decay 0.01, batch 256, 5000 steps, 200 warm-up steps then cosine to 0, clip 1.0, fp32,
          seed 20261119; final weights only, monitor (200 diag pairs) logged every 500 steps, record only
Precheck (S1 dev noisy rows, per direction): median cos(d(Q(u_hat)), d(oracle ids)) must be >= the same median for the V1
ids + 0.02 in BOTH directions (go); also recorded: continuous d(bound(u_hat)), clean rows, Hamming to oracle.
Token tables: <out>/tokens/tokens_a_<d>.jsonl rows {key, id, sha1, ids = Q(u_hat), b = bound(u_hat)} (format of
k5_tokens.py; idea1_si_infer_vo reads key / sha1 / ids, k5_si_infer_soft additionally b).
    venvs/vllm_omni_030/bin/python -m experiments.mist_w3po_v1.voicecond.a_sv_train --feats <dir> --out <dir> [--device cuda:0]
Refuses an existing --out.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from experiments.mist_w3po_v1.voicecond import a_sv_common as C  # noqa: E402

LR, WD, BS, STEPS, WARM, W_CODE, HID, DROP, SEED = 1e-3, 0.01, 256, 5000, 200, 0.1, 1024, 0.1, 20261119
MARGIN = 0.02


class Net(torch.nn.Module):
    def __init__(self, mu, sd):
        super().__init__()
        self.register_buffer("mu", torch.as_tensor(mu, dtype=torch.float32))
        self.register_buffer("sd", torch.as_tensor(sd, dtype=torch.float32))
        self.f = torch.nn.Sequential(torch.nn.Linear(384, HID), torch.nn.GELU(), torch.nn.Dropout(DROP),
                                     torch.nn.Linear(HID, HID), torch.nn.GELU(), torch.nn.Dropout(DROP),
                                     torch.nn.Linear(HID, 192))
        torch.nn.init.zeros_(self.f[-1].weight)
        torch.nn.init.zeros_(self.f[-1].bias)

    def forward(self, t, v):
        x = (torch.cat([t, v.reshape(len(v), -1)], 1) - self.mu) / self.sd
        return v + self.f(x).reshape(-1, 32, 6)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--feats", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--steps", type=int, default=STEPS, help="smoke only")
    a = ap.parse_args()
    out, F = Path(a.out), Path(a.feats)
    if out.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    (out / "tokens").mkdir(parents=True)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    dev = torch.device(a.device)
    from experiments.mist_w3po_v1.voicecond.bicodec_phi import BiCodecPhi
    P = BiCodecPhi(a.device)
    P.model.eval()
    for p in P.model.parameters():
        p.requires_grad_(False)
    se, rq, fsq = P.se, P.rq, P.fsq

    def d_soft(u):   # pre-bound u (B,32,6) -> d (B, D), differentiable
        t = torch.clamp(fsq.bound(u), -2.0, 1.0) / 2.0
        zq = rq.project_out(t * rq.scales[0]).transpose(1, 2)
        return se.project(zq.reshape(zq.shape[0], -1))

    def d_ids(ids):
        t = torch.as_tensor(np.asarray(ids), dtype=torch.long, device=dev)
        with torch.no_grad():
            return se.detokenize(t[:, None, :])   # (B, 1, 32) as BiCodecTokenizer.detokenize / d1ts_tokens

    def digits(ids):
        ids = np.asarray(ids)
        return np.stack([(ids // 4 ** k) % 4 for k in range(6)], -1)

    # check: d_soft at level centres == detokenize(ids)
    D0 = np.load(F / f"codes_dev_{C.DIRS[0]}.npz")
    ids_chk = D0["ids_oracle"][:64]
    lev = torch.as_tensor(digits(ids_chk) - 2.0, dtype=torch.float32, device=dev)
    with torch.no_grad():
        t = lev / 2.0
        zq = rq.project_out(t * rq.scales[0]).transpose(1, 2)
        dchk = se.project(zq.reshape(zq.shape[0], -1))
    gap = float((dchk - d_ids(ids_chk)).abs().max())
    print(f"d-path check max |diff| = {gap:.2e}", flush=True)
    assert gap <= 1e-5, gap

    T = torch.from_numpy(np.load(F / "titanet_train.npy")).to(dev)
    Ct = np.load(F / "codes_train.npz")
    V, U = torch.from_numpy(Ct["v1"]).to(dev), torch.from_numpy(Ct["u0"]).to(dev)
    assert len(T) == len(V) == len(U), (len(T), len(V), len(U))
    Tm = torch.from_numpy(np.load(F / "titanet_mon.npy")).to(dev)
    Cm = np.load(F / "codes_mon.npz")
    Vm, Um = torch.from_numpy(Cm["v1"]).to(dev), torch.from_numpy(Cm["u0"]).to(dev)
    X = torch.cat([T, V.reshape(len(V), -1)], 1)
    net = Net(X.mean(0), X.std(0) + 1e-6).to(dev)
    with torch.no_grad():
        DU, DUm = d_soft(U), d_soft(Um)
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda st: (st + 1) / WARM if st < WARM else 0.5 * (1 + math.cos(math.pi * (st - WARM) / max(1, a.steps - WARM))))
    g = torch.Generator(device="cpu")
    g.manual_seed(SEED)
    cos = torch.nn.functional.cosine_similarity

    def loss_fn(uh, u, du):
        ld = ((d_soft(uh) - du) ** 2).sum(1).mean()    # squared L2 norm in d-vector space, mean over the batch
        lc = ((uh - u) ** 2).mean()                    # code-space MSE over the 32 x 6 pre-bound code (as V1's mse)
        return ld + W_CODE * lc, ld, lc

    def monitor(step):
        net.eval()
        with torch.no_grad():
            uh = net(Tm, Vm)
            L, ld, lc = loss_fn(uh, Um, DUm)
            rec = {"step": step, "mon_loss": float(L), "mon_ld": float(ld), "mon_lc": float(lc),
                   "mon_cos_cont_med": float(cos(d_soft(uh), DUm).median()),
                   "mon_cos_cont_v1_med": float(cos(d_soft(Vm), DUm).median())}
        net.train()
        print(json.dumps(rec), flush=True)
        return rec

    log = open(out / "train_log.jsonl", "x", encoding="utf-8")
    with torch.no_grad():   # in-sample note: V1 was trained on these pairs (6 s clips); record V1 residuals train vs mon
        r_tr = float(cos(d_soft(V[:4096]), DU[:4096]).median())
    log.write(json.dumps({"v1_cos_cont_train_med": r_tr}) + "\n")
    log.write(json.dumps(monitor(0)) + "\n")
    net.train()
    t0, n = time.time(), len(T)
    for step in range(1, a.steps + 1):
        j = torch.randint(0, n, (BS,), generator=g).to(dev)
        L, ld, lc = loss_fn(net(T[j], V[j]), U[j], DU[j])
        opt.zero_grad(set_to_none=True)
        L.backward()
        gn = torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 100 == 0:
            log.write(json.dumps({"step": step, "loss": float(L), "ld": float(ld), "lc": float(lc), "grad_norm": float(gn),
                                  "lr": sched.get_last_lr()[0], "s": time.time() - t0}) + "\n")
            log.flush()
        if step % 500 == 0:
            log.write(json.dumps(monitor(step)) + "\n")
            log.flush()
    torch.save({"state_dict": net.state_dict(),
                "meta": {"lr": LR, "wd": WD, "bs": BS, "steps": a.steps, "warm": WARM, "w_code": W_CODE, "hid": HID,
                         "drop": DROP, "seed": SEED}}, out / "mlp.pt")
    log.close()
    # dev inference, precheck, token tables
    net.eval()
    pre = {"margin": MARGIN, "dirs": {}}
    for d in C.DIRS:
        Dd = np.load(F / f"codes_dev_{d}.npz")
        Td = torch.from_numpy(np.load(F / f"titanet_dev_{d}.npy")).to(dev)
        Vd, Ud = torch.from_numpy(Dd["v1"]).to(dev), torch.from_numpy(Dd["u0"]).to(dev)
        rows = C.dev_rows(d)[: len(Vd)]
        assert len(Td) == len(Vd) == len(rows)
        with torch.no_grad():
            uh = net(Td, Vd)
            ids_a = P.tokens(uh.cpu().numpy())
            b = fsq.bound(uh).cpu().numpy()
            d_or = d_ids(Dd["ids_oracle"])
            c_a = cos(d_ids(ids_a), d_or).cpu().numpy()
            c_v = cos(d_ids(Dd["ids_v1"]), d_or).cpu().numpy()
            c_ac = cos(d_soft(uh), d_or).cpu().numpy()
            c_vc = cos(d_soft(Vd), d_or).cpu().numpy()
        noisy = np.array(["_noisy__" in r["id"] for r in rows])
        st = {}
        for nm, m in (("noisy", noisy), ("clean", ~noisy)):
            st[nm] = {"n": int(m.sum()), "cos_A_hard_med": float(np.median(c_a[m])), "cos_V1_hard_med": float(np.median(c_v[m])),
                      "cos_A_cont_med": float(np.median(c_ac[m])), "cos_V1_cont_med": float(np.median(c_vc[m])),
                      "ham_A_oracle": float((ids_a[m] != Dd["ids_oracle"][m]).sum(1).mean()),
                      "ham_V1_oracle": float((Dd["ids_v1"][m] != Dd["ids_oracle"][m]).sum(1).mean())}
        st["go"] = st["noisy"]["cos_A_hard_med"] >= st["noisy"]["cos_V1_hard_med"] + MARGIN
        pre["dirs"][d] = st
        with open(out / "tokens" / f"tokens_a_{d}.jsonl", "x", encoding="utf-8") as fh:
            for r, ii, bb in zip(rows, ids_a, b):
                fh.write(json.dumps({"key": r["key"], "id": r["id"], "sha1": r["sha1"], "ids": [int(x) for x in ii],
                                     "b": [[round(float(x), 6) for x in z] for z in bb]}) + "\n")
        with open(out / f"uhat_dev_{d}.npy", "xb") as fh:
            np.save(fh, uh.cpu().numpy())
    pre["go"] = all(pre["dirs"][d]["go"] for d in C.DIRS)
    with open(out / "precheck.json", "x", encoding="utf-8") as fh:
        json.dump(pre, fh, indent=1)
    print(json.dumps(pre, indent=1), flush=True)


if __name__ == "__main__":
    main()

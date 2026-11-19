"""Idea 2 E2E (plan §8.3, adapted to the contraction model) FEASIBILITY check only -- no training, no selection.

Checks, on a few open synthetic TRAIN speakers (no test data):
 1. the straight-through bridge from the continuous posterior code u to the Talker's 32 global-token embeddings:
    e = E[g_hard] + (q_c - sg(q_c)) W^T, q_c = FSQ-bound(u) / 2, W (1536 x 6, + bias) = least squares of the global-token
    embedding rows on their FSQ codes (reports the R^2 of that linear surrogate);
 2. teacher-forced privileged-information KL( T1c(g*) || T1c+LoRA(g(m)) ) on semantic positions, with m the differentiable
    contraction-model posterior (alpha, beta, c trainable) -> gradients reach alpha / beta / c and the LoRA;
 3. time and peak GPU memory per step for batch 4 and 16 (sequence ~ 600 tokens: 32 globals, 200 source GLM, 30 text,
    300 semantic codes; GLM / text / codes are random ids here -- only shapes and gradient flow are checked).
    python -I experiments/mist_w3po_v1/voicecond/idea2_e2e_feasibility.py --out <json> [--device cuda:0]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from experiments.mist_w3po_v1.voicecond.idea2_common import load_refs, load_stream, sids, slot_for_speaker  # noqa: E402
from experiments.mist_w3po_v1.voicecond.idea2_shrink import ShrinkParams, load_vt  # noqa: E402

T1C = REPO / "eval_outputs/uniss_qwen3_omni_v1/si_t1c_20261021T181440Z/autoext/export_1500/uniss"
GLOBAL_0, SEMANTIC_0, GLM_0 = 151665, 155761, 163953
TASK_S2S, BALANCE, START_GLOBAL, END_GLOBAL = 180379, 180400, 180387, 180391
WRITE, TASK_S2T, TASK_TTS, SPEED_0 = 180396, 180380, 180375, 180337
START_CONTENT, END_CONTENT, START_SEMANTIC = 180386, 180390, 180388
LANG = {"en": 180373, "zh": 180372}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    if Path(a.out).exists():
        sys.exit("refuse to overwrite")
    dev = torch.device(a.device)
    res = {}
    from transformers import AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model
    from experiments.mist_w3po_v1.voicecond.bicodec_phi import BiCodecPhi
    P = BiCodecPhi(a.device)
    model = AutoModelForCausalLM.from_pretrained(str(T1C), torch_dtype=torch.bfloat16).to(dev)
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    emb = model.get_input_embeddings()
    # 1. linear surrogate of the global-token embedding rows on their FSQ codes
    E = emb.weight[GLOBAL_0:GLOBAL_0 + 4096].detach().float()
    Q = torch.as_tensor(P.codes(np.arange(4096)), dtype=torch.float32, device=dev).reshape(4096, -1)
    X = torch.cat([Q, torch.ones(4096, 1, device=dev)], 1)
    Wb = torch.linalg.lstsq(X, E).solution
    r2 = 1 - ((E - X @ Wb) ** 2).sum() / ((E - E.mean(0)) ** 2).sum()
    res["surrogate_r2"] = float(r2)
    W = Wb[:6].T.contiguous()                                    # (1536, 6)
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"], lora_dropout=0.0))
    res["lora_params"] = int(sum(p.numel() for p in model.parameters() if p.requires_grad))
    # 2. differentiable contraction posterior for a few open synthetic train speakers
    pf = ShrinkParams.load(REPO / "eval_outputs/mist_w3po_v1/voicecond_idea2/shrink_fit_all/full.npz")
    t = lambda x: torch.as_tensor(np.asarray(x), dtype=torch.float32, device=dev)
    alpha = torch.tensor(pf.alpha, device=dev, requires_grad=True)
    beta = t(pf.beta).requires_grad_(True)
    cvec = {k: t(v).requires_grad_(True) for k, v in pf.c.items()}
    Rfi, S0i = t(np.linalg.inv(pf.R * 4)), t(np.linalg.inv(pf.Sig0))
    mu0, bM, vm, vs = t(pf.mu0), t(pf.bM), t(pf.vt_mean), t(pf.vt_std)
    refs = load_refs()
    units = []
    for sid in sids("pilot", "train")[:4]:
        d = load_stream(sid, 2.0)
        vt = load_vt(sid, 2.0, len(d["end"]))
        j = 0
        s = slot_for_speaker(d, j)
        sel = np.nonzero((d["slot"] == s) & (d["end"] <= 30.0))[0]
        units.append((t(d["z_mix"][sel]), t(vt[sel]), t(d["pi"][sel]), d["meta"]["lang"], refs[d["meta"]["speakers"][j]][1]))

    def posterior(z, vt, pi, lang):
        lam = torch.sigmoid(alpha + ((vt - vm) / vs) @ beta)
        y = z - (1 - lam)[:, None] * cvec[lang if pf.per_lang else "all"][None]
        prec = S0i + (pi * lam * lam).sum() * Rfi
        rhs = S0i @ mu0 + Rfi @ ((pi * lam)[:, None] * y).sum(0)
        return torch.linalg.solve(prec, rhs) + bM

    def st_embed(u):
        qc = P.fsq.bound(u.reshape(1, 32, 6))[0] / 2                       # continuous code (32, 6)
        g = torch.as_tensor(P.tokens(u.detach().reshape(32, 6).cpu().numpy()), device=dev)
        return emb.weight[GLOBAL_0 + g] + (qc - qc.detach()).to(torch.bfloat16) @ W.T.to(torch.bfloat16), g

    def batch(B, rng):
        seqs, gpos, spos = [], [], []
        for b in range(B):
            z, vt, pi, lang, gstar = units[b % len(units)]
            tgt = "en" if lang == "zh" else "zh"
            head = [TASK_S2S, BALANCE, LANG[tgt], START_GLOBAL] + [GLOBAL_0] * 32 + [END_GLOBAL]
            glm = list(GLM_0 + rng.integers(0, 16384, 200))
            txt = list(rng.integers(1000, 30000, 30))
            mid = [WRITE, TASK_S2T, LANG[tgt], SPEED_0 + 5, START_CONTENT] + txt + [END_CONTENT, WRITE, TASK_TTS,
                                                                                     LANG[tgt], SPEED_0 + 5, START_SEMANTIC]
            codes = list(SEMANTIC_0 + rng.integers(0, 8192, 300))
            s = head + glm + mid + codes
            seqs.append(s)
            gpos.append(4)
            spos.append(len(head) + len(glm) + len(mid) - 1)
        return torch.tensor(seqs, device=dev), gpos, spos

    def step(B):
        rng = np.random.default_rng(0)
        ids, gpos, spos = batch(B, rng)
        torch.cuda.reset_peak_memory_stats(dev)
        t0 = time.time()
        base_e = emb(ids)
        stu_e, tea_e = base_e.clone(), base_e.clone()
        hams = []
        for b in range(B):
            z, vt, pi, lang, gstar = units[b % len(units)]
            e_st, g = st_embed(posterior(z, vt, pi, lang))
            stu_e[b, gpos[b]:gpos[b] + 32] = e_st
            tea_e[b, gpos[b]:gpos[b] + 32] = emb.weight[GLOBAL_0 + torch.as_tensor(gstar, device=dev)]
            hams.append(int((g.cpu().numpy() != np.asarray(gstar)).sum()))
        with torch.no_grad(), model.disable_adapter():
            lt = model(inputs_embeds=tea_e.detach()).logits
        ls = model(inputs_embeds=stu_e).logits
        kl = 0.0
        for b in range(B):
            a_, b_ = spos[b], ids.shape[1] - 1
            pt = torch.log_softmax(lt[b, a_:b_, SEMANTIC_0:SEMANTIC_0 + 8192].float(), -1)
            ps = torch.log_softmax(ls[b, a_:b_, SEMANTIC_0:SEMANTIC_0 + 8192].float(), -1)
            kl = kl + (pt.exp() * (pt - ps)).sum(-1).mean()
        kl = kl / B
        kl.backward()
        torch.cuda.synchronize(dev)
        out = {"B": B, "seq_len": int(ids.shape[1]), "kl": float(kl), "sec": round(time.time() - t0, 3),
               "peak_gib": round(torch.cuda.max_memory_allocated(dev) / 2 ** 30, 2), "ham_g_vs_gstar": hams,
               "grad_alpha": float(alpha.grad.abs()), "grad_beta": float(beta.grad.norm()),
               "grad_c": float(sum(v.grad.norm() for v in cvec.values() if v.grad is not None)),
               "grad_lora": float(sum(p.grad.norm() for p in model.parameters() if p.requires_grad and p.grad is not None))}
        for v in [alpha, beta] + list(cvec.values()) + [p for p in model.parameters() if p.requires_grad]:
            v.grad = None
        return out
    res["step_B4"] = step(4)
    res["step_B16"] = step(16)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()

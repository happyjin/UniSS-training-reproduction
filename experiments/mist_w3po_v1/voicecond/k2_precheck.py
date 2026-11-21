"""K2 CPU precheck (reports/mist_w3po_v1/robust_k2_w2map_prereg.zh-CN.md, step 1). All codes are the pre-bound BiCodec
global code u (32x6 -> 192), bicodec_phi.BiCodecPhi; V1 = voicecond Idea 1 ctrl encoder (idea6_e2e_train.CKPT_C).

(a) voicecond Idea 1 dev pairs (diag_{en,zh}, split dev): v = phi_V1(y), u* = phi_0(s), c = phi_V1(s);
    spread = tr Sigma_b(v) / tr Sigma_b(u*), between-speaker covariance within language, pooled over languages.
(b) S1 dev noisy items: v = phi_V1(V1 prompt), u* = phi_0(paired clean audio cut to the prompt length, PCM16 like
    the prompt); rho = <v - mu, u* - mu> / |u* - mu|^2 (mu = mean u* of the direction); identity loss
    L = sim(V1 out, clean pair) - sim(V1 out, noisy item) (wave3_combos/V/metrics/v1_<d>_spksim.jsonl).
    Primary: Spearman(rho, L) pooled over directions; predicted < 0.
Stop rule: spread >= 0.9, or primary p >= 0.05, or primary rho >= 0.
Writes <out>/precheck.json and <out>/codes_*.npz. Refuses an existing --out.
    OMP_NUM_THREADS=48 qwen3omni/bin/python -I experiments/mist_w3po_v1/voicecond/k2_precheck.py --out <dir>
"""
from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from experiments.mist_w3po_v1.voicecond.idea6_e2e_train import CKPT_C  # noqa: E402

DIAG = REPO / "data/open_voicecond_v1/idea1"
S1 = REPO / "eval_outputs/mist_w3po_v1/wave2_S1"
V = REPO / "eval_outputs/mist_w3po_v1/wave3_combos/V"
BS = 32


def jl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8")]


def enc(P, wavs):
    out = []
    for i in range(0, len(wavs), BS):
        out.append(P.phi_wave(wavs[i:i + BS]).reshape(-1, 192))
    return np.concatenate(out)


def between_cov_trace(X, spk, lang):
    """trace of the between-speaker covariance, speakers centred on their language mean, pooled"""
    tot, n = 0.0, len(X)
    for lg in sorted(set(lang)):
        ml = X[lang == lg].mean(0)
        for s in sorted(set(spk[lang == lg])):
            m = (lang == lg) & (spk == s)
            tot += m.sum() * float(((X[m].mean(0) - ml) ** 2).sum())
    return tot / n


def total_cov_trace(X, lang):
    return float(sum(((X[lang == lg] - X[lang == lg].mean(0)) ** 2).sum() for lg in set(lang)) / len(X))


def rho_r(v, u, mu):
    du, dv = u - mu, v - mu
    return (dv * du).sum(1) / (du * du).sum(1), np.linalg.norm(dv, axis=1) / np.linalg.norm(du, axis=1)


def pcm16(x):
    b = io.BytesIO()
    sf.write(b, x, 16000, format="WAV")
    b.seek(0)
    return sf.read(b, dtype="float64")[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        sys.exit(f"拒绝覆盖已有输出: {out}")
    out.mkdir(parents=True)
    import torch
    from scipy.stats import spearmanr
    from experiments.mist_w3po_v1.voicecond.bicodec_phi import BiCodecPhi
    from experiments.mist_w3po_v1.voicecond.idea1_tokens import load_encoder
    from experiments.mist_w3po_v1.voicecond.idea1_zne import read_mono64
    torch.set_num_threads(int(__import__("os").environ.get("OMP_NUM_THREADS", "48")))
    P0 = BiCodecPhi("cpu")
    Pc = load_encoder(BiCodecPhi("cpu"), str(CKPT_C))
    Pc.model.eval()
    res = {"ckpt_v1": str(CKPT_C)}

    # ---------------- (a) voicecond dev pairs
    ys, ss, spk, lang = [], [], [], []
    for lg in ("en", "zh"):
        d = DIAG / f"diag_{lg}"
        for r in jl(d / "manifest.jsonl"):
            if r["split"] != "dev":
                continue
            ys.append(sf.read(str(d / "wav" / f"{r['id']}_y.wav"), dtype="float64")[0])
            ss.append(sf.read(str(d / "wav" / f"{r['id']}_s.wav"), dtype="float64")[0])
            spk.append(f"{lg}:{r['speaker']}")
            lang.append(lg)
    spk, lang = np.array(spk), np.array(lang)
    v, c, u = enc(Pc, ys), enc(Pc, ss), enc(P0, ss)
    np.savez(out / "codes_diag_dev.npz", v=v, c=c, u=u, spk=spk, lang=lang)
    A = {"n": len(v), "spread_between": between_cov_trace(v, spk, lang) / between_cov_trace(u, spk, lang),
         "spread_between_v1clean": between_cov_trace(c, spk, lang) / between_cov_trace(u, spk, lang),
         "spread_total": total_cov_trace(v, lang) / total_cov_trace(u, lang),
         "spread_total_v1clean": total_cov_trace(c, lang) / total_cov_trace(u, lang)}
    for lg in ("en", "zh"):
        m = lang == lg
        A[f"spread_between_{lg}"] = between_cov_trace(v[m], spk[m], lang[m]) / between_cov_trace(u[m], spk[m], lang[m])
        A[f"spread_between_v1clean_{lg}"] = between_cov_trace(c[m], spk[m], lang[m]) / between_cov_trace(u[m], spk[m], lang[m])
    rho_all, r_all, cos_all = [], [], []
    for lg in ("en", "zh"):
        m = lang == lg
        rh, rr = rho_r(v[m], u[m], u[m].mean(0))
        cs = (v[m] * u[m]).sum(1) / np.linalg.norm(v[m], axis=1) / np.linalg.norm(u[m], axis=1)
        rho_all += rh.tolist(); r_all += rr.tolist(); cos_all += cs.tolist()
    s = spearmanr(rho_all, cos_all)
    A.update({"rho_median": float(np.median(rho_all)), "r_median": float(np.median(r_all)),
              "spearman_rho_vs_cos_u*": [float(s.statistic), float(s.pvalue)]})
    res["a_diag_dev"] = A
    print(json.dumps(A), flush=True)

    # ---------------- (b) S1 dev noisy items with the V1 Talker's similarity
    B, pool_rho, pool_L, pool_r = {}, [], [], []
    for d in ("en2zh", "zh2en"):
        meta = {m["id"]: m for m in jl(S1 / "data" / f"meta_{d}.jsonl")}
        items = {it["id"]: it for it in jl(S1 / "data" / f"items_{d}.jsonl")}
        tab = {r["id"]: r for r in jl(V / "tokens" / f"tokens_v1_{d}.jsonl")}
        sim = {r["id"]: r.get("sim") for r in jl(V / "metrics" / f"v1_{d}_spksim.jsonl")}
        clean_of = {(m["set"], m["pair"]): i for i, m in meta.items() if m["cond"] == "clean"}
        ids, yw, uw = [], [], []
        for i, m in meta.items():
            if m["cond"] != "noisy" or i not in tab:
                continue
            j = clean_of.get((m["set"], m["pair"]))
            if j is None or sim.get(i) is None or sim.get(j) is None:
                continue
            p = V / "tokens" / "prompts" / d / f"{i.replace('/', '__')}_voice.wav"
            y = read_mono64(p)
            x, sr = sf.read(items[j]["audio"], dtype="float32", always_2d=True)
            assert sr == 16000
            cw = x.mean(axis=1)[0: int(items[j]["dur_ms"]) * 16][: len(y)]
            ids.append(i); yw.append(y); uw.append(pcm16(cw))
        vv, uu = enc(Pc, yw), enc(P0, uw)
        tok_ok = int(sum(list(Pc.tokens(vv[k].reshape(32, 6))) == tab[i]["ids"] for k, i in enumerate(ids)))
        rh, rr = rho_r(vv, uu, uu.mean(0))
        L = np.array([sim[clean_of[(meta[i]["set"], meta[i]["pair"])]] - sim[i] for i in ids])
        s1 = spearmanr(rh, L)
        B[d] = {"n": len(ids), "tokens_match_v1_table": tok_ok, "rho_median": float(np.median(rh)),
                "r_median": float(np.median(rr)), "L_median": float(np.median(L)),
                "spread_total": float(((vv - vv.mean(0)) ** 2).sum() / ((uu - uu.mean(0)) ** 2).sum()),
                "spearman_rho_L": [float(s1.statistic), float(s1.pvalue)],
                "spearman_r_L": [float(spearmanr(rr, L).statistic), float(spearmanr(rr, L).pvalue)]}
        np.savez(out / f"codes_s1_{d}.npz", ids=np.array(ids), v=vv, u=uu, rho=rh, r=rr, L=L)
        pool_rho += rh.tolist(); pool_L += L.tolist(); pool_r += rr.tolist()
        print(d, json.dumps(B[d]), flush=True)
    sp = spearmanr(pool_rho, pool_L)
    B["pooled"] = {"n": len(pool_L), "spearman_rho_L": [float(sp.statistic), float(sp.pvalue)],
                   "spearman_r_L": [float(spearmanr(pool_r, pool_L).statistic), float(spearmanr(pool_r, pool_L).pvalue)]}
    res["b_s1_dev"] = B
    stop = []
    if A["spread_between"] >= 0.9:
        stop.append("spread_between >= 0.9")
    if not (sp.statistic < 0 and sp.pvalue < 0.05):
        stop.append("primary Spearman(rho, L) not significant / wrong sign")
    res["verdict"] = {"pass": not stop, "stop_reasons": stop}
    with open(out / "precheck.json", "x", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    print(json.dumps(res["verdict"]), flush=True)


if __name__ == "__main__":
    main()

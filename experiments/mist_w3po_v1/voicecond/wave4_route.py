"""Robust wave 4, diagnosis #2: MoE routing flips (reports/mist_w3po_v1/robust_wave4_diag_prereg.zh-CN.md).

The base Thinker (original Qwen3-Omni-30B-A3B-Instruct, HF bf16, no adapter; loading and si_thinker_stream's base prompt
as wave3_T_thinker.BaseThinker) is teacher-forced on the REFERENCE translation (+ <|im_end|>) of each CoVoST 2 validation
sentence of the wave-1 manifest (300 per direction), once on the clean and once on the noisy twin. A forward hook on every
text MoE router (thinker.model.layers[l].mlp.gate, 48 layers, 128 experts, top-8) records, per position, the top-k expert
set and the router margin z_(k) - z_(k+1) (k = num_experts_per_tok, raw router logits).

Per sentence and layer: flip = 1 - Jaccard(clean set, noisy set), averaged over the target-text positions (the reference
tokens + <|im_end|>, identical token ids in both twins) and, separately, over the audio positions (compared frame by frame
when both twins give the same number of audio tokens, else skipped); the clean and noisy margins at the target positions
(mean, and the share below 0.1 / 0.5 / 1.0).

    worker: CUDA_VISIBLE_DEVICES=<c> python -I wave4_route.py --worker k --num-workers n [--max-gib 66] [--smoke 4]
    summary (CPU): python -I wave4_route.py --summary
Outputs eval_outputs/mist_w3po_v1/wave4_diag/route/{w<k>.jsonl, summary.json}; existing outputs are never rewritten.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
os.chdir(REPO)

OUT = REPO / "eval_outputs/mist_w3po_v1/wave4_diag/route"
S1 = REPO / "eval_outputs/mist_w3po_v1/wave2_S1"
TRUNS = REPO / "eval_outputs/mist_w3po_v1/wave3_combos/T/runs"
DIRS = ("en2zh", "zh2en")
AUDIO_TOK = 151675
IM_END = "<|im_end|>"
B = 2000


def log(m: str) -> None:
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())} {m}", flush=True)


# ------------------------------------------------------------------------------------------------------------ worker
def worker(a) -> None:
    import torch
    from experiments.mist_w3po_v1.voicecond import wave1_thinker as W1
    from experiments.mist_w3po_v1.voicecond.wave3_T_thinker import BaseThinker
    out = OUT / (f"smoke_w{a.worker}.jsonl" if a.smoke else f"w{a.worker}.jsonl")
    if out.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    OUT.mkdir(parents=True, exist_ok=True)
    eng = BaseThinker(a.max_gib)
    th = eng.thinker
    layers = th.model.layers
    k = th.config.text_config.num_experts_per_tok
    cap = {}

    def hook(li):
        def fn(mod, args, output):
            z = output[0].float()   # (N, 128) raw router logits
            top = torch.topk(z, k + 1, dim=-1)
            cap[li] = (top.indices[:, :k].sort(-1).values.to(torch.int16).cpu().numpy(),
                       (top.values[:, k - 1] - top.values[:, k]).cpu().numpy())
        return fn
    for li, L in enumerate(layers):
        L.mlp.gate.register_forward_hook(hook(li))
    nl = len(layers)
    log(f"worker {a.worker}: {nl} MoE layers, top-{k}")
    M = W1.manifest()
    jobs = [(d, x) for d in DIRS for x in M[d]]
    jobs = [j for i, j in enumerate(jobs) if i % a.num_workers == a.worker]
    if a.smoke:
        jobs = jobs[: a.smoke]
    tmp = out.with_suffix(".partial")
    with open(tmp, "w", encoding="utf-8") as fh:
        for n, (d, x) in enumerate(jobs):
            rec = {"id": x["id"], "dir": d, "bvr_db": x.get("bvr_db")}
            res = {}
            for cond in ("clean", "noisy"):
                w = W1.wav(x[cond])
                if len(w) < 1600:
                    w = np.pad(w, (0, 1600 - len(w)))
                p0 = eng.proc(text=[eng.head[d]], audio=[w], return_tensors="pt", padding=True)
                p1 = eng.proc(text=[eng.head[d] + x["ref"] + IM_END], audio=[w], return_tensors="pt", padding=True)
                ids0, ids1 = p0["input_ids"][0], p1["input_ids"][0]
                assert ids1[: len(ids0)].tolist() == ids0.tolist(), "prompt is not a prefix of the forced sequence"
                inputs = {kk: v.to(eng.dev) for kk, v in p1.items()}
                inputs["input_features"] = inputs["input_features"].to(torch.bfloat16)
                cap.clear()
                with torch.no_grad():
                    th(**inputs, use_cache=False)
                assert len(cap) == nl, (len(cap), nl)
                ids = ids1.numpy()
                tpos = np.arange(len(ids0), len(ids1))
                apos = np.nonzero(ids == AUDIO_TOK)[0]
                res[cond] = {"tgt_ids": ids[tpos], "tpos": tpos, "apos": apos,
                             "sets": [cap[li][0] for li in range(nl)], "marg": [cap[li][1] for li in range(nl)]}
            c, z = res["clean"], res["noisy"]
            assert np.array_equal(c["tgt_ids"], z["tgt_ids"])

            def jac(A, Bm):   # rowwise Jaccard of sorted top-k sets
                inter = np.array([len(np.intersect1d(r, s, assume_unique=True)) for r, s in zip(A, Bm)])
                return inter / (2 * k - inter)
            rec["n_tgt"] = int(len(c["tpos"]))
            rec["flip_tgt"] = [float(1 - jac(c["sets"][li][c["tpos"]], z["sets"][li][z["tpos"]]).mean()) for li in range(nl)]
            if len(c["apos"]) == len(z["apos"]) and len(c["apos"]):
                rec["n_audio"] = int(len(c["apos"]))
                rec["flip_audio"] = [float(1 - jac(c["sets"][li][c["apos"]], z["sets"][li][z["apos"]]).mean())
                                     for li in range(nl)]
            else:
                rec["n_audio"], rec["flip_audio"] = None, None
                rec["n_audio_pair"] = [int(len(c["apos"])), int(len(z["apos"]))]
            for cond, r in (("clean", c), ("noisy", z)):
                m = np.stack([r["marg"][li][r["tpos"]] for li in range(nl)])   # (layers, T)
                rec[f"marg_{cond}_mean"] = [float(v) for v in m.mean(1)]
                rec[f"marg_{cond}_lt"] = {str(t): float((m < t).mean()) for t in (0.1, 0.5, 1.0)}
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            if n % 25 == 0:
                log(f"worker {a.worker}: {n + 1}/{len(jobs)} {d} {x['id']} flip_tgt {np.mean(rec['flip_tgt']):.3f}")
    tmp.rename(out)
    log(f"worker {a.worker}: done, {len(jobs)} sentences -> {out}")


# ----------------------------------------------------------------------------------------------------------- summary
def partial_spearman(x, y, z):
    from scipy.stats import rankdata, t as tdist
    rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)
    r = np.corrcoef([rx, ry, rz])
    rxy, rxz, ryz = r[0, 1], r[0, 2], r[1, 2]
    pr = (rxy - rxz * ryz) / math.sqrt((1 - rxz ** 2) * (1 - ryz ** 2))
    n = len(x)
    tt = pr * math.sqrt((n - 3) / max(1e-12, 1 - pr ** 2))
    return float(pr), float(2 * tdist.sf(abs(tt), n - 3))


def chrf_drops(d: str, ids: list) -> tuple[dict, dict]:
    from sacrebleu.metrics import CHRF
    from evaluation.text_metrics import normalize_for_bleu
    from experiments.mist_w3po_v1.voicecond import wave1_thinker as W1
    NL = "cmn" if d == "en2zh" else "eng"
    nz = lambda s: normalize_for_bleu(s, NL)   # noqa: E731
    ch = CHRF()
    ref = {x["id"]: x["ref"] for x in W1.manifest()[d]}
    la2 = {r["id"]: r["translation"] for r in map(json.loads, open(S1 / f"thinker_{d}/blocks_{d}_w1.jsonl", encoding="utf-8"))}
    whole = {}
    for cond in ("clean", "noisy"):
        for p in sorted((TRUNS / f"base_{cond}_{d}").glob("s*.jsonl")):
            for r in map(json.loads, open(p, encoding="utf-8")):
                whole[(cond, r["id"])] = r["translation"]
    sc = lambda h, i: ch.sentence_score(nz(h), [nz(ref[i])]).score   # noqa: E731
    prim = {i: sc(la2[f"a_clean__{i}"], i) - sc(la2[f"a_noisy__{i}"], i) for i in ids}
    sec = {i: sc(whole[("clean", i)], i) - sc(whole[("noisy", i)], i) for i in ids if ("clean", i) in whole and ("noisy", i) in whole}
    return prim, sec


def summary(a) -> None:
    from scipy.stats import spearmanr
    out = OUT / "summary.json"
    if out.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    recs = [json.loads(l) for p in sorted(OUT.glob("w*.jsonl")) for l in open(p, encoding="utf-8")]
    S = {"n_records": len(recs)}
    for d in DIRS:
        R = {r["id"]: r for r in recs if r["dir"] == d}
        ids = sorted(R)
        prim, sec = chrf_drops(d, ids)
        flip = np.array([np.mean(R[i]["flip_tgt"]) for i in ids])
        bvr = np.array([R[i]["bvr_db"] for i in ids], float)
        D = {"n": len(ids), "flip_tgt_mean": float(flip.mean()), "flip_tgt_median": float(np.median(flip))}
        for name, drop in (("primary_la2", prim), ("secondary_whole", sec)):
            keep = [j for j, i in enumerate(ids) if i in drop]
            y = np.array([drop[ids[j]] for j in keep])
            rho, p = spearmanr(flip[keep], y)
            prho, pp = partial_spearman(flip[keep], y, bvr[keep])
            D[name] = {"n": len(keep), "spearman": float(rho), "p": float(p), "partial_spearman_bvr": prho, "partial_p": pp,
                       "mean_chrf_drop": float(y.mean())}
        # flip vs BVR itself (how much is just noise level)
        rb, pb = spearmanr(flip, bvr)
        D["spearman_flip_vs_bvr"] = {"rho": float(rb), "p": float(pb)}
        L = np.array([R[i]["flip_tgt"] for i in ids])
        D["flip_tgt_by_layer"] = [float(v) for v in L.mean(0)]
        A = [R[i]["flip_audio"] for i in ids if R[i].get("flip_audio")]
        D["n_audio_compared"] = len(A)
        D["flip_audio_by_layer"] = [float(v) for v in np.array(A).mean(0)] if A else None
        for cond in ("clean", "noisy"):
            Mg = np.array([R[i][f"marg_{cond}_mean"] for i in ids])
            D[f"margin_{cond}_by_layer"] = [float(v) for v in Mg.mean(0)]
            D[f"margin_{cond}_share_lt"] = {t: float(np.mean([R[i][f"marg_{cond}_lt"][t] for i in ids])) for t in ("0.1", "0.5", "1.0")}
        # per-sentence: does a low clean margin go with a high flip rate? (mechanism check, not a gate)
        mc = np.array([np.mean(R[i]["marg_clean_mean"]) for i in ids])
        rm, pm = spearmanr(mc, flip)
        D["spearman_clean_margin_vs_flip"] = {"rho": float(rm), "p": float(pm)}
        pr = D["primary_la2"]
        D["gate_pass"] = bool(pr["spearman"] >= 0.3 and pr["p"] < 0.01)
        S[d] = D
    S["gate_pass_both"] = bool(all(S[d]["gate_pass"] for d in DIRS))
    json.dump(S, open(out, "w", encoding="utf-8"), indent=1)
    for d in DIRS:
        D = S[d]
        print(d, {k: D[k] for k in ("n", "flip_tgt_mean", "primary_la2", "secondary_whole", "spearman_flip_vs_bvr",
                                    "spearman_clean_margin_vs_flip", "gate_pass")})
    print("gate both:", S["gate_pass_both"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", type=int)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--max-gib", type=int, default=66)
    ap.add_argument("--smoke", type=int, default=0)
    ap.add_argument("--summary", action="store_true")
    a = ap.parse_args()
    if a.summary:
        summary(a)
    else:
        if a.worker is None or not 0 <= a.worker < a.num_workers:
            raise SystemExit("--worker k (0 <= k < --num-workers) is required")
        worker(a)


if __name__ == "__main__":
    main()

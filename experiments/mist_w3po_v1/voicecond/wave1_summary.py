"""Robust wave 1 Thinker prechecks: metrics and preregistered gates from wave1_thinker.py outputs
(prereg reports/mist_w3po_v1/robust_wave1_prereg.zh-CN.md; K5 / K6 dropped by the user, K9 = lambda part only).

Text metrics of the Thinker translation against the human CoVoST 2 translation: corpus BLEU (sacrebleu; tokenize zh for
en2zh, 13a for zh2en) and sentence chrF (sacrebleu CHRF defaults). Paired bootstrap (1000 resamples of sentences, seed
20261119) 95 % intervals for every BLEU difference.
Gates:
  P2a   noise loss = BLEU(base clean) - BLEU(base noisy); en2zh < 1.0 -> K2 stops
  P2b   en2zh recovers >= 40 % of the noise loss; zh2en drops <= 0.2 BLEU (vs base noisy)
  lin   merged R^2 >= 0.8 of the per-step, per-vocabulary-entry linear fit in sigma (raw logits); quadratic share
  K3    mean encoder-output distance of the noisy sentences to their clean twins falls >= 20 % (pooled over both
        directions; per direction and the log-mel input distance reported too); en2zh BLEU does not drop
  K7    en2zh recovers >= 40 %; zh2en noisy and clean (both directions) drop <= 0.2 BLEU (the K6 comparison is void)
  K8    same as K7; reference arms k8h (pure HDemucs) and k8f (alpha 0.5) reported
  K9    Spearman(1 - lambda of the noisy sentence, chrF(base clean) - chrF(base noisy)) >= 0.3 (pooled; per direction too)

    python -I experiments/mist_w3po_v1/voicecond/wave1_summary.py [--out summary.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
OUT = REPO / "eval_outputs/mist_w3po_v1/wave1_K2K3K5"
DIRS = ("en2zh", "zh2en")
N_SHARD = 8


def load_arm(arm: str, cond: str, d: str) -> dict | None:
    D = OUT / "runs" / f"{arm}_{cond}_{d}"
    rows = {}
    for k in range(N_SHARD):
        p = D / f"s{k}.jsonl"
        if not p.exists():
            return None
        for r in map(json.loads, open(p, encoding="utf-8")):
            rows[r["id"]] = r
    return rows


def bleu_obj(d: str):
    from sacrebleu.metrics import BLEU
    return BLEU(tokenize="zh" if d == "en2zh" else "13a")


def stats(d: str, hyps: list[str], refs: list[str]) -> np.ndarray:
    b = bleu_obj(d)
    return np.array(b._extract_corpus_statistics(hyps, [refs]), dtype=np.float64)


def score(d: str, st: np.ndarray) -> float:
    return float(bleu_obj(d)._compute_score_from_stats(list(st.sum(0))).score)


def chrf(hyps: list[str], refs: list[str]) -> np.ndarray:
    from sacrebleu.metrics import CHRF
    c = CHRF()
    return np.array([c.sentence_score(h, [r]).score for h, r in zip(hyps, refs)])


def boot_diff(d: str, sa: np.ndarray, sb: np.ndarray, n: int = 1000) -> list[float]:
    rng = np.random.default_rng(20261119)
    N = len(sa)
    v = []
    for _ in range(n):
        i = rng.integers(0, N, N)
        v.append(score(d, sa[i]) - score(d, sb[i]))
    return [round(float(np.percentile(v, 2.5)), 2), round(float(np.percentile(v, 97.5)), 2)]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(OUT / "summary.json"))
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    from scipy.stats import spearmanr
    M = {d: [] for d in DIRS}
    for x in map(json.loads, open(OUT / "data/manifest.jsonl", encoding="utf-8")):
        M[x["dir"]].append(x)
    LAM = {(r["id"], r["cond"]): r for r in map(json.loads, open(OUT / "data/lambda.jsonl", encoding="utf-8"))}
    R = {"n": {d: len(M[d]) for d in DIRS}, "bleu": {}, "chrf_mean": {}, "delta": {}, "gates": {}, "missing": []}
    ST, CH = {}, {}
    arms = [("base", "clean"), ("base", "noisy"), ("p2b", "noisy"), ("k3", "noisy"), ("k7", "noisy"), ("k7", "clean"),
            ("k8", "noisy"), ("k8", "clean"), ("k8h", "noisy"), ("k8h", "clean"), ("k8f", "noisy"), ("k8f", "clean")]
    rows_all = {}
    for d in DIRS:
        ids = [x["id"] for x in M[d]]
        refs = [x["ref"] for x in M[d]]
        for arm, c in arms:
            rows = load_arm(arm, c, d)
            if rows is None:
                R["missing"].append(f"{arm}_{c}_{d}")
                continue
            assert set(rows) == set(ids), (arm, c, d, len(rows))
            rows_all[(arm, c, d)] = rows
            hyps = [rows[i]["translation"] for i in ids]
            ST[(arm, c, d)] = stats(d, hyps, refs)
            CH[(arm, c, d)] = chrf(hyps, refs)
            R["bleu"][f"{arm}_{c}_{d}"] = round(score(d, ST[(arm, c, d)]), 2)
            R["chrf_mean"][f"{arm}_{c}_{d}"] = round(float(CH[(arm, c, d)].mean()), 2)
    loss = {}
    for d in DIRS:
        if ("base", "clean", d) in ST and ("base", "noisy", d) in ST:
            loss[d] = score(d, ST[("base", "clean", d)]) - score(d, ST[("base", "noisy", d)])
            R["delta"][f"noise_loss_{d}"] = {"bleu": round(loss[d], 2), "ci95": boot_diff(d, ST[("base", "clean", d)],
                                                                                           ST[("base", "noisy", d)])}
    for (arm, c, d) in list(ST):
        if arm == "base":
            continue
        ref = ("base", c, d)
        dl = score(d, ST[(arm, c, d)]) - score(d, ST[ref])
        e = {"bleu": round(dl, 2), "ci95": boot_diff(d, ST[(arm, c, d)], ST[ref]),
             "chrf": round(float((CH[(arm, c, d)] - CH[ref]).mean()), 2)}
        if c == "noisy" and loss.get(d):
            e["recovery"] = round(dl / loss[d], 3)
        R["delta"][f"{arm}_{c}_{d}"] = e
    G = R["gates"]
    if "en2zh" in loss:
        G["P2a"] = {"en2zh_loss": round(loss["en2zh"], 2), "zh2en_loss": round(loss.get("zh2en", float("nan")), 2),
                    "pass": loss["en2zh"] >= 1.0}
    dd = R["delta"]

    def gate_tc(arm, rec_min, with_clean=True):
        need = [f"{arm}_noisy_en2zh", f"{arm}_noisy_zh2en"] + ([f"{arm}_clean_en2zh", f"{arm}_clean_zh2en"] if with_clean else [])
        if not all(k in dd for k in need):
            return None
        g = {"en2zh_recovery": dd[f"{arm}_noisy_en2zh"].get("recovery"), "zh2en_noisy_delta": dd[f"{arm}_noisy_zh2en"]["bleu"]}
        ok = g["en2zh_recovery"] is not None and g["en2zh_recovery"] >= rec_min and g["zh2en_noisy_delta"] >= -0.2
        if with_clean:
            g["clean_delta"] = {d: dd[f"{arm}_clean_{d}"]["bleu"] for d in DIRS}
            ok = ok and all(v >= -0.2 for v in g["clean_delta"].values())
        g["pass"] = bool(ok)
        return g

    if G.get("P2a", {}).get("pass"):
        G["P2b"] = gate_tc("p2b", 0.40, with_clean=False)
    elif "P2a" in G:
        G["P2b"] = {"skipped": "P2a: en2zh noise loss < 1.0 BLEU"}
    G["K7"] = gate_tc("k7", 0.40)
    if G["K7"] is not None:
        G["K7"]["vs_K6"] = "void (K6 dropped by the user)"
    G["K8"] = gate_tc("k8", 0.40)
    G["K8_ref_pure_hdemucs"] = gate_tc("k8h", 0.40)
    G["K8_ref_alpha05"] = gate_tc("k8f", 0.40)
    # K3
    kd = {}
    for d in DIRS:
        rows = load_arm("k3dist", "noisy", d)
        if rows is not None:
            kd[d] = list(rows.values())
    if kd:
        def red(rs, key):
            b = float(np.mean([r[f"{key}_before"] for r in rs]))
            a_ = float(np.mean([r[f"{key}_after"] for r in rs]))
            return {"before": round(b, 4), "after": round(a_, 4), "reduction": round(1 - a_ / b, 4)}
        allr = [r for v in kd.values() for r in v]
        K3 = {"enc": {d: red(v, "enc") for d, v in kd.items()}, "mel": {d: red(v, "mel") for d, v in kd.items()},
              "enc_pooled": red(allr, "enc"), "mel_pooled": red(allr, "mel"),
              "delta_l2_median": round(float(np.median([r["delta_l2"] for r in allr])), 3)}
        if "k3_noisy_en2zh" in dd:
            K3["en2zh_delta"] = dd["k3_noisy_en2zh"]["bleu"]
            K3["pass"] = bool(K3["enc_pooled"]["reduction"] >= 0.20 and K3["en2zh_delta"] >= 0.0)
        G["K3"] = K3
    # linearity
    L = {}
    for d in DIRS:
        rows = load_arm("lin", "noisy", d)
        if rows is not None:
            L[d] = list(rows.values())
    if L:
        def r2(rs, kind):
            t = sum(r[kind]["ss_tot"] for r in rs)
            return {"R2_lin": round(1 - sum(r[kind]["ss_res_lin"] for r in rs) / t, 4),
                    "quad_share": round((sum(r[kind]["ss_res_lin"] for r in rs) - sum(r[kind]["ss_res_quad"] for r in rs)) / t, 4),
                    "R2_sent_median": round(float(np.median([1 - r[kind]["ss_res_lin"] / r[kind]["ss_tot"] for r in rs])), 4),
                    "n": len(rs)}
        allr = [r for v in L.values() for r in v]
        G["linearity"] = {"logit": {**{d: r2(v, "logit") for d, v in L.items()}, "pooled": r2(allr, "logit")},
                          "logsoftmax": {**{d: r2(v, "logsoftmax") for d, v in L.items()}, "pooled": r2(allr, "logsoftmax")}}
        G["linearity"]["pass"] = G["linearity"]["logit"]["pooled"]["R2_lin"] >= 0.8
    # K9 (lambda part)
    if all(("base", c, d) in CH for c in ("clean", "noisy") for d in DIRS):
        xs, ys, per = [], [], {}
        for d in DIRS:
            drop = CH[("base", "clean", d)] - CH[("base", "noisy", d)]
            sig = np.array([1 - LAM[(x["id"], "noisy")]["lam"] for x in M[d]])
            rho, p = spearmanr(sig, drop)
            per[d] = {"rho": round(float(rho), 3), "p": float(p), "chrf_drop_mean": round(float(drop.mean()), 2)}
            xs += list(sig)
            ys += list(drop)
        rho, p = spearmanr(xs, ys)
        G["K9_lambda"] = {**per, "pooled": {"rho": round(float(rho), 3), "p": float(p)}, "pass": bool(rho >= 0.3)}
    # diagnostics
    diag = {}
    for d in DIRS:
        if ("p2b", "noisy", d) in rows_all:
            rs = list(rows_all[("p2b", "noisy", d)].values())
            diag[f"p2b_{d}"] = {"w_median": round(float(np.median([r["w"] for r in rs])), 3),
                                "w_capped_share": round(float(np.mean([r["w"] >= 3.0 for r in rs])), 3)}
        for c in ("noisy", "clean"):
            if ("k7", c, d) in rows_all:
                rs = list(rows_all[("k7", c, d)].values())
                diag[f"k7_{c}_{d}"] = {k: round(float(np.mean([r["k7"][k] for r in rs])), 3) for k in ("r_mean", "r95_mean", "gamma_mean")}
            if ("k8", c, d) in rows_all:
                rs = list(rows_all[("k8", c, d)].values())
                diag[f"k8_{c}_{d}"] = {"alpha_median": round(float(np.median([r["alpha"] for r in rs])), 3)}
    R["diagnostics"] = diag
    if (OUT / "k7_stats.pt").exists():
        import torch
        st = torch.load(OUT / "k7_stats.pt", map_location="cpu")
        R["k7_stats"] = {k: st[k] for k in ("n_clean_clips", "n_noise_clips", "n_clean_frames", "n_noise_frames", "k_sem")}
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "x", encoding="utf-8") as fh:
        json.dump(R, fh, ensure_ascii=False, indent=1)
    print(json.dumps({k: R[k] for k in ("bleu", "delta", "gates", "missing")}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()

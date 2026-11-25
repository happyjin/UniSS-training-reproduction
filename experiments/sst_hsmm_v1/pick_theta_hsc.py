"""prereg change 28 (HSC theta): ONE theta for both directions = the one whose dev char-weighted mean write time is closest
(mean over the two directions of |HSC - SVC-JM|) to SVC-JM on dev (dv2/svcjm_*), on all common dev items (iso-latency with
SVC-JM). Text BLEU per theta is reported (information only, not used). Writes h2/theta.json; refuses to overwrite."""
import json
import sys
from pathlib import Path

import sacrebleu

REPO = Path(__file__).resolve().parents[2]
E = REPO / "eval_outputs/sst_hsmm_v1/h2"
JM = REPO / "eval_outputs/sst_hsmm_v1/dv2"
NN = REPO / "eval_outputs/sst_hsmm_v1/dv"
DATA = REPO / "eval_outputs/mist_w3po_v1/wave2_S1/data"
THETAS = ("0.80", "0.90", "0.95", "0.98")


def rd(p):
    return {r["id"]: r for r in map(json.loads, open(p, encoding="utf-8"))}


def wt(X, ks):
    s = 0.0
    for k in ks:
        bb = [b for b in X[k]["blocks"] if b["text"].strip()]
        L = sum(len(b["text"]) for b in bb) or 1
        s += sum(b["write_ms"] * len(b["text"]) for b in bb) / L
    return s / len(ks) / 1000


def main():
    out = E / "theta.json"
    if out.exists():
        sys.exit(f"拒绝覆盖: {out}")
    res = {}
    for th in THETAS:
        g = {}
        for d in ("en2zh", "zh2en"):
            S = rd(E / f"dev_{th}_{d}" / f"blocks_{d}_w1.jsonl")
            J = {k: v for kk in range(4) for k, v in rd(JM / f"svcjm_{d}_{kk}" / f"blocks_{d}_w1.jsonl").items()}
            N = {k: v for kk in range(2) for k, v in rd(NN / f"n_{d}_{kk}" / f"blocks_{d}_w1.jsonl").items()}
            ref = {r["id"]: r["translation"] for r in map(json.loads, open(DATA / f"items_{d}.jsonl", encoding="utf-8"))}
            ks = [k for k in S if k in J and k in N]
            tok = "zh" if d == "en2zh" else "13a"
            bl = lambda X: sacrebleu.corpus_bleu([X[k]["translation"] for k in ks], [[ref[k] for k in ks]], tokenize=tok).score
            g[d] = {"n": len(ks), "wt_hsc": wt(S, ks), "wt_jm": wt(J, ks), "wt_n": wt(N, ks), "bleu_hsc": bl(S), "bleu_jm": bl(J), "bleu_n": bl(N)}
        g["gap"] = sum(abs(g[d]["wt_hsc"] - g[d]["wt_jm"]) for d in ("en2zh", "zh2en")) / 2
        res[th] = g
        print(th, json.dumps(g), flush=True)
    best = min(THETAS, key=lambda t: res[t]["gap"])
    out.write_text(json.dumps({"theta": float(best), "per_theta": res}, indent=1))
    print("theta", best, flush=True)


if __name__ == "__main__":
    main()

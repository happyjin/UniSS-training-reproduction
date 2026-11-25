"""prereg change 27 (E2): choose ONE theta for both directions = the one whose dev char-weighted mean write time is closest
(mean over the two directions of |SDC - B_dev|) to the plain-LA2 base Thinker on dev (r2/eval_base, = B's policy), on the
common items. Also reports text BLEU per theta (information only, not used). Writes e2/theta.json; refuses to overwrite."""
import json
import sys
from pathlib import Path

import sacrebleu

REPO = Path(__file__).resolve().parents[2]
E = REPO / "eval_outputs/sst_hsmm_v1/e2"
BASE = REPO / "eval_outputs/mist_w3po_v1/r2/eval_base"
DATA = REPO / "eval_outputs/mist_w3po_v1/wave2_S1/data"
THETAS = ("0.80", "0.85", "0.90", "0.95")


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
        gaps = {}
        for d in ("en2zh", "zh2en"):
            S = rd(E / f"dev_{th}_{d}" / f"blocks_{d}_w1.jsonl")
            B = rd(BASE / f"thinker_{d}" / f"blocks_{d}_w1.jsonl")
            ref = {r["id"]: r["translation"] for r in map(json.loads, open(DATA / f"items_{d}.jsonl", encoding="utf-8"))}
            ks = [k for k in S if k in B]
            tok = "zh" if d == "en2zh" else "13a"
            bl = lambda X: sacrebleu.corpus_bleu([X[k]["translation"] for k in ks], [[ref[k] for k in ks]], tokenize=tok).score
            gaps[d] = {"n": len(ks), "wt_sdc": wt(S, ks), "wt_base": wt(B, ks), "bleu_sdc": bl(S), "bleu_base": bl(B)}
        gaps["gap"] = sum(abs(gaps[d]["wt_sdc"] - gaps[d]["wt_base"]) for d in ("en2zh", "zh2en")) / 2
        res[th] = gaps
        print(th, json.dumps(gaps), flush=True)
    best = min(THETAS, key=lambda t: res[t]["gap"])
    out.write_text(json.dumps({"theta": float(best), "per_theta": res}, indent=1))
    print("theta", best, flush=True)


if __name__ == "__main__":
    main()

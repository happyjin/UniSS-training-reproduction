"""prereg change 30 selection: among dev configs (variant x theta) whose dev text BLEU >= HSC2 dev (h4/dev_0.80) - 0.05 in
BOTH directions, take the lowest mean (over directions) char-weighted write time; none -> smallest max BLEU deficit.
Also the same rule within the soft family. All on the common dev items. Writes eval_outputs/sst_hsmm_v1/s_choice.json."""
import json
import sys
from pathlib import Path

import sacrebleu

REPO = Path(__file__).resolve().parents[2]
S = REPO / "eval_outputs/sst_hsmm_v1"
DATA = REPO / "eval_outputs/mist_w3po_v1/wave2_S1/data"
FAM = {"soft": ("h6soft", "h3"), "clock": ("h6clock", "h5"), "clockpause": ("h6cp", "h5")}
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
    out = S / "s_choice.json"
    if out.exists():
        sys.exit(f"拒绝覆盖: {out}")
    rows = []
    ref = {d: {r["id"]: r["translation"] for r in map(json.loads, open(DATA / f"items_{d}.jsonl", encoding="utf-8"))} for d in ("en2zh", "zh2en")}
    base = {d: rd(S / "h4" / f"dev_0.80_{d}" / f"blocks_{d}_w1.jsonl") for d in ("en2zh", "zh2en")}
    cfgs = [(v, th) for v in FAM for th in THETAS]
    data = {(v, th, d): rd(S / FAM[v][0] / f"dev_{th}_{d}" / f"blocks_{d}_w1.jsonl") for v, th in cfgs for d in ("en2zh", "zh2en")}
    res = {}
    for d in ("en2zh", "zh2en"):
        ks = [k for k in base[d] if all(k in data[(v, th, d)] for v, th in cfgs)]
        tok = "zh" if d == "en2zh" else "13a"
        bl = lambda X: sacrebleu.corpus_bleu([X[k]["translation"] for k in ks], [[ref[d][k] for k in ks]], tokenize=tok).score
        res[("HSC2", "0.80", d)] = (bl(base[d]), wt(base[d], ks), len(ks))
        for v, th in cfgs:
            X = data[(v, th, d)]
            res[(v, th, d)] = (bl(X), wt(X, ks), len(ks))
    table = []
    for v, th in cfgs:
        deficit = max(res[("HSC2", "0.80", d)][0] - res[(v, th, d)][0] for d in ("en2zh", "zh2en"))
        mwt = sum(res[(v, th, d)][1] for d in ("en2zh", "zh2en")) / 2
        table.append({"variant": v, "theta": float(th), "params": FAM[v][1], "deficit": deficit, "mean_wt": mwt,
                      **{f"{d}_bleu": res[(v, th, d)][0] for d in ("en2zh", "zh2en")}, **{f"{d}_wt": res[(v, th, d)][1] for d in ("en2zh", "zh2en")}})
        print(json.dumps(table[-1]), flush=True)

    def choose(rows_):
        ok = [r for r in rows_ if r["deficit"] <= 0.05]
        return min(ok, key=lambda r: r["mean_wt"]) if ok else min(rows_, key=lambda r: r["deficit"])
    best = choose(table)
    best_soft = choose([r for r in table if r["variant"] == "soft"])
    hs = {d: {"bleu": res[("HSC2", "0.80", d)][0], "wt": res[("HSC2", "0.80", d)][1]} for d in ("en2zh", "zh2en")}
    out.write_text(json.dumps({"best": best, "best_soft": best_soft, "hsc2_dev": hs, "table": table}, indent=1))
    print("best", json.dumps(best), "\nbest_soft", json.dumps(best_soft), "\nhsc2", json.dumps(hs), flush=True)


if __name__ == "__main__":
    main()

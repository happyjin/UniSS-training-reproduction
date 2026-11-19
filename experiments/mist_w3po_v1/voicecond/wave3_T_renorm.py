"""Wave 3 T group re-scored with the S1 BLEU convention (evaluation.text_metrics.normalize_for_bleu before sacrebleu),
which the T summary (wave3_T_thinker.py) and wave 1 did not apply. Arm - base corpus BLEU per direction x condition, paired
bootstrap over sentences (2000, seed 0); the change-2 gates re-evaluated on these numbers. Reads T/runs only.
    <uniss-eval python> experiments/mist_w3po_v1/voicecond/wave3_T_renorm.py --out eval_outputs/mist_w3po_v1/wave3_combos/T/summary_norm.json
"""
import argparse, glob, json, sys
from pathlib import Path
import numpy as np
from sacrebleu.metrics import BLEU
REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from evaluation.text_metrics import normalize_for_bleu  # noqa: E402
RUNS = REPO / "eval_outputs/mist_w3po_v1/wave3_combos/T/runs"
MAN = REPO / "eval_outputs/mist_w3po_v1/wave1_K2K3K5/data/manifest.jsonl"

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); a = ap.parse_args()
    if Path(a.out).exists(): raise SystemExit(f"拒绝覆盖已有输出: {a.out}")
    man = {r["id"]: r for r in map(json.loads, open(MAN, encoding="utf-8"))}
    runs = {}
    for d in sorted(glob.glob(str(RUNS / "*"))):
        arm, cond, dr = Path(d).name.rsplit("_", 2)
        runs[(arm, cond, dr)] = {r["id"]: r["translation"] for p in glob.glob(d + "/s*.jsonl") for r in map(json.loads, open(p, encoding="utf-8"))}
    out = {}
    for dr in ("en2zh", "zh2en"):
        NL = "cmn" if dr == "en2zh" else "eng"; nz = lambda x: normalize_for_bleu(x, NL)
        bl = BLEU(tokenize="zh" if dr == "en2zh" else "13a")
        def stats(h, ids):
            return np.array([[s.sys_len, s.ref_len] + s.counts + s.totals for s in
                             (bl.corpus_score([nz(h[i])], [[nz(man[i]["ref"])]]) for i in ids)], float)
        def cb(S):
            sl, rl, c, t = S[0], S[1], S[2:6], S[6:10]
            if np.any(c == 0): return 0.0
            return 100 * (1.0 if sl > rl else np.exp(1 - rl / max(sl, 1))) * np.exp(np.mean(np.log(c / t)))
        for cond in ("clean", "noisy"):
            if ("base", cond, dr) not in runs: continue
            base = runs[("base", cond, dr)]
            for (arm, c2, d2), h in runs.items():
                if c2 != cond or d2 != dr or arm == "base": continue
                ids = sorted(set(base) & set(h)); rng = np.random.default_rng(0)
                Sb, Sa = stats(base, ids), stats(h, ids)
                idx = rng.integers(0, len(ids), (2000, len(ids)))
                dd = np.array([cb(Sa[j].sum(0)) - cb(Sb[j].sum(0)) for j in idx])
                out[f"{arm}_{cond}_{dr}"] = {"n": len(ids), "base": cb(Sb.sum(0)), "arm": cb(Sa.sum(0)),
                    "delta": cb(Sa.sum(0)) - cb(Sb.sum(0)), "ci": [float(np.percentile(dd, 2.5)), float(np.percentile(dd, 97.5))]}
        if ("base", "clean", dr) in runs and ("base", "noisy", dr) in runs:
            ids = sorted(set(runs[("base", "clean", dr)]) & set(runs[("base", "noisy", dr)]))
            out[f"P2a_loss_{dr}"] = cb(stats(runs[("base", "clean", dr)], ids).sum(0)) - cb(stats(runs[("base", "noisy", dr)], ids).sum(0))
    json.dump(out, open(a.out, "w", encoding="utf-8"), indent=1)
    for k, v in out.items(): print(k, v if not isinstance(v, dict) else {x: (round(y, 2) if isinstance(y, float) else [round(z, 2) for z in y] if isinstance(y, list) else y) for x, y in v.items()})
main()

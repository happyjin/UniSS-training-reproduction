"""Wave 3 LA2 commit diagnosis, D-GPU scoring (reports/mist_w3po_v1/robust_wave3_la2diag_prereg.zh-CN.md).

Text BLEU (S1 convention: normalize_for_bleu, sacrebleu zh / 13a) of the oracle-commit run (wave3_combos/D/oracle_<dir>)
next to the S1 LA2 run, per direction x set ((a) CoVoST pairs; (b) OB-SI dev single-speaker cuts, all and without
crosstalk) x condition; noise loss = clean - noisy with a paired bootstrap 95 % CI (2000, seed 0) over sentences (a) or
streams (b); the prereg reading: oracle loss <= 50 % of the LA2 loss on zh2en (a) -> the loss is mainly the commit policy.
    <uniss-eval python> experiments/mist_w3po_v1/voicecond/wave3_d_oracle_score.py --out eval_outputs/mist_w3po_v1/wave3_combos/D/oracle.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sacrebleu.metrics import BLEU

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from evaluation.text_metrics import normalize_for_bleu  # noqa: E402

S1 = REPO / "eval_outputs/mist_w3po_v1/wave2_S1"
D = REPO / "eval_outputs/mist_w3po_v1/wave3_combos/D"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if Path(a.out).exists():
        raise SystemExit(f"拒绝覆盖已有输出: {a.out}")
    res = {}
    for d in ("zh2en", "en2zh"):
        NL = "cmn" if d == "en2zh" else "eng"
        bl = BLEU(tokenize="zh" if d == "en2zh" else "13a")
        nz = lambda x: normalize_for_bleu(x, NL)  # noqa: E731
        items = {r["id"]: r for r in map(json.loads, open(S1 / f"data/items_{d}.jsonl", encoding="utf-8"))}
        meta = {r["id"]: r for r in map(json.loads, open(S1 / f"data/meta_{d}.jsonl", encoding="utf-8"))}
        sysm = {"la2": S1 / f"thinker_{d}/blocks_{d}_w1.jsonl", "oracle": D / f"oracle_{d}/blocks_{d}_w1_oraclecommit.jsonl"}
        hyp = {k: {r["id"]: "".join(b["text"] for b in r["blocks"]) for r in map(json.loads, open(p, encoding="utf-8"))}
               for k, p in sysm.items()}
        pairs = {}
        for m in meta.values():
            pairs.setdefault((m["set"], m["pair"]), {})[m["cond"] if m["cond"] == "clean" else "noisy"] = m
        st = {}

        def stat(h, r):
            s = bl.corpus_score([nz(h)], [[nz(r)]])
            return [s.sys_len, s.ref_len] + s.counts + s.totals

        def cb(S):
            sl, rl, c, t = S[0], S[1], S[2:6], S[6:10]
            if np.any(c == 0):
                return 0.0
            return 100 * (1.0 if sl > rl else np.exp(1 - rl / max(sl, 1))) * np.exp(np.mean(np.log(c / t)))
        R = {}
        for set_, sub, keep in (("a", "a", lambda v: True), ("b", "b", lambda v: True),
                                ("b", "b_no_crosstalk", lambda v: not v["noisy"]["ovl"] and not v["clean"]["ovl"])):
            ps = sorted(p for (s, p), v in pairs.items() if s == set_ and len(v) == 2 and keep(v))
            units = [pairs[(set_, p)]["noisy"]["stream"] if set_ == "b" else p for p in ps]
            uidx = {u: i for i, u in enumerate(sorted(set(units)))}
            uid = np.array([uidx[u] for u in units])
            R[sub] = {"n_sentences": len(ps), "n_units": len(uidx)}
            rng = np.random.default_rng(0)
            draws = [rng.integers(0, len(uidx), len(uidx)) for _ in range(2000)]
            for k in ("la2", "oracle"):
                S = {c: np.array([stat(hyp[k][pairs[(set_, p)][c]["id"]], items[pairs[(set_, p)][c]["id"]]["translation"])
                                  for p in ps], float) for c in ("clean", "noisy")}
                U = {c: np.zeros((len(uidx), 10)) for c in S}
                for c in S:
                    np.add.at(U[c], uid, S[c])
                full = {c: cb(U[c].sum(0)) for c in U}
                boot = np.array([cb(U["clean"][j].sum(0)) - cb(U["noisy"][j].sum(0)) for j in draws])
                R[sub][k] = {"clean": full["clean"], "noisy": full["noisy"], "loss": full["clean"] - full["noisy"],
                             "loss_ci": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]}
            R[sub]["oracle_loss_share_of_la2"] = (R[sub]["oracle"]["loss"] / R[sub]["la2"]["loss"]
                                                  if abs(R[sub]["la2"]["loss"]) > 1e-9 else None)
        res[d] = R
        for sub, v in R.items():
            print(d, sub, {k: (v[k] if not isinstance(v[k], dict) else {x: (round(y, 2) if isinstance(y, float) else [round(z, 2) for z in y]) for x, y in v[k].items()}) for k in v})
    json.dump(res, open(a.out, "w", encoding="utf-8"), indent=1)


if __name__ == "__main__":
    main()

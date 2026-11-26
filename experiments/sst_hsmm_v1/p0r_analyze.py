"""sst_hsmm_v1 prereg change 43 (P0-R): forced-commit fork analysis (regret vs survival).

Inputs: <root>/run_<d>_*/blocks_<d>_w1.jsonl (greedy base) and forks_<d>.jsonl (hsc14 clones; the first draft record of a clone
is its fork read with "forced" {k, n, cut0, cut}); items with "translation" in --data. Group = (item, fork read t); kept when
|d0| - cut0 >= 1; clones whose forced count (cut - cut0) repeats a smaller k are dropped. Deltas are clone minus the null clone
(k = 0) of the same group: dBLEU = sentence BLEU (sacrebleu, zh / 13a), dD2 = proxy speech LAAL (s; p0rl_analyze.playback).
Survival of the forced units: the null clone's final translation equals d0 through cut.
  (a) k = 1, forced unit not surviving: share with dBLEU >= -1
  (b) k = 1: harm = dBLEU < -1; 5-fold GroupKFold (by item) logistic regression AUC on the features below vs AUC of (1 - P)
  (c) half A (even item index in the items file) / half B: rules "commit 1 extra unit if predicted harm <= tau" (logistic
      regression fit on A) and "if P >= tau'"; each threshold chosen on A as the lowest per-sentence summed dD2 with the
      per-sentence summed dBLEU >= -0.3 (sum over the item's fork reads, first-order); evaluated on B.
      pass = learned dD2 <= -0.10 s, learned dBLEU >= -0.3, learned dD2 <= P-rule dD2 - 0.03 s (B half).
Also: noise floor (null clone vs the un-cloned base), oracle (per group the k with dBLEU >= 0 and the lowest dD2) and oracle
survival (the largest k whose forced units all survive). Refuses to overwrite.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from experiments.sst_hsmm_v1.p0rl_analyze import playback  # noqa: E402
from experiments.voice_onset_v1.playout import si_units  # noqa: E402
from experiments.uniss_qwen3_omni_v1.evaluation.si_thinker_stream import join, units  # noqa: E402

PUNCT = re.compile(r"^[\s，。,.!?！？；;：:、…\"“”'‘’()（）《》\-—]+$")
DIRS = ("en2zh", "zh2en")
FEATS = ["P", "lp0", "lps", "lps_na", "agree2", "cv", "sap", "punct", "rel_pos", "tfrac", "n_extra", "bleu_d0_d1", "bleu_d0_prev", "dir_zh"]


def lcp(a, b):
    k = 0
    while k < min(len(a), len(b)) and a[k] == b[k]:
        k += 1
    return k


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if Path(a.out).exists():
        raise SystemExit(f"refuse to overwrite {a.out}")
    import sacrebleu
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    G = []          # one dict per (group, k>0 clone)
    noise = []
    for d in DIRS:
        zh = d.endswith("zh")
        lang = "cmn" if zh else "eng"
        tk = "zh" if zh else "13a"
        sbleu = lambda h, r: sacrebleu.sentence_bleu(h, [r], tokenize=tk).score  # noqa: E731
        items = [json.loads(x) for x in open(Path(a.data) / f"items_{d}.jsonl", encoding="utf-8")]
        idx = {it["id"]: i for i, it in enumerate(items)}
        itm = {it["id"]: it for it in items}
        base, forks = {}, defaultdict(dict)
        for f in sorted(glob.glob(str(Path(a.root) / f"run_{d}_*" / f"blocks_{d}_w1.jsonl"))):
            for r in map(json.loads, open(f, encoding="utf-8")):
                base[r["id"]] = r
        for f in sorted(glob.glob(str(Path(a.root) / f"run_{d}_*" / f"forks_{d}.jsonl"))):
            for r in map(json.loads, open(f, encoding="utf-8")):
                forks[(r["id"], r["fork_t"])][r["fork_k"]] = r

        def stats(row, it):
            X = it["dur_ms"]
            Tn = len(si_units(it["translation"], lang))
            d2, gap, _ = playback(row["blocks"], lang, X, Tn)
            return d2, sbleu(row["translation"], it["translation"]), gap

        for (iid, t), cl in sorted(forks.items()):
            if 0 not in cl:
                continue
            it = itm[iid]
            null = cl[0]
            f0 = null["drafts"][0]
            assert f0["t"] == t and f0["forced"]["k"] == 0, (iid, t)
            dr = units(f0["draft"], zh)
            n, cut0 = f0["forced"]["n"], f0["forced"]["cut0"]
            b = base[iid]
            nd2, nbl, ngap = stats(null, it)
            bd2, bbl, _ = stats(b, it)
            noise.append({"dir": d, "changed": float(null["translation"] != b["translation"] or null["blocks"] != b["blocks"]),
                          "dbleu": nbl - bbl, "dd2": nd2 - bd2})
            if len(dr) - cut0 < 1:
                continue
            nfin = units(null["translation"], zh)
            prev = None
            for x in b["drafts"]:
                if x["t"] == t - 1000:
                    prev = units(x["draft"], zh)
            d1 = units(f0.get("draft_v1", ""), zh)
            P = f0.get("P", [])
            lp0 = f0.get("lp0", [])
            lps = f0.get("lps", [])
            i = cut0
            j = i - n
            tail = join(dr[i:], zh)
            feat = {"P": float(P[j]) if 0 <= j < len(P) else 0.0,
                    "lp0": float(lp0[j]) if 0 <= j < len(lp0) else -10.0,
                    "lps": float(lps[j]) if 0 <= j < len(lps) else -10.0,
                    "lps_na": float(not (0 <= j < len(lps))),
                    "agree2": float(i < n + int(f0.get("la2_new", 0))),
                    "cv": float(lcp(dr, d1) > i), "sap": float(int(f0.get("sv", 0)) > i),
                    "punct": float(bool(PUNCT.match(dr[i]))), "rel_pos": i / max(1, len(dr)), "tfrac": t / it["dur_ms"],
                    "n_extra": float(len(dr) - cut0),
                    "bleu_d0_d1": sbleu(tail, join(d1[i:], zh)) if d1 else 0.0,
                    "bleu_d0_prev": sbleu(tail, join(prev[i:], zh)) if prev else 0.0, "dir_zh": float(zh)}
            seen = {0}
            for k in sorted(cl):
                if k == 0:
                    continue
                r = cl[k]
                fr = r["drafts"][0]["forced"]
                m = fr["cut"] - fr["cut0"]
                if m in seen or m <= 0:
                    continue
                seen.add(m)
                d2, bl, gap = stats(r, it)
                G.append({"dir": d, "id": iid, "half": "A" if idx[iid] % 2 == 0 else "B", "t": t, "k": m,
                          "survive": float(nfin[: fr["cut"]] == dr[: fr["cut"]]), "dbleu": bl - nbl, "dd2": d2 - nd2,
                          "dgap": gap - ngap, **({f"f_{q}": v for q, v in feat.items()} if m == 1 else {})})
    res = {"groups_k_gt0": len(G), "noise": {d: {"null_vs_base_changed": float(np.mean([x["changed"] for x in noise if x["dir"] == d])),
                                                   "mean_abs_dbleu": float(np.mean([abs(x["dbleu"]) for x in noise if x["dir"] == d])),
                                                   "mean_dd2": float(np.mean([x["dd2"] for x in noise if x["dir"] == d]))} for d in DIRS}}
    k1 = [g for g in G if g["k"] == 1]
    for d in DIRS + ("all",):
        sel = [g for g in k1 if d == "all" or g["dir"] == d]
        ns = [g for g in sel if g["survive"] == 0]
        res[f"a_{d}"] = {"k1_groups": len(sel), "k1_nonsurv": len(ns), "nonsurv_share": len(ns) / max(1, len(sel)),
                         "harmless_share_nonsurv": float(np.mean([g["dbleu"] >= -1 for g in ns])) if ns else None,
                         "harmless_share_surv": float(np.mean([g["dbleu"] >= -1 for g in sel if g["survive"] == 1])) if sel else None,
                         "mean_dbleu_nonsurv": float(np.mean([g["dbleu"] for g in ns])) if ns else None,
                         "mean_dd2_k1": float(np.mean([g["dd2"] for g in sel])) if sel else None}
    Xf = np.array([[g[f"f_{q}"] for q in FEATS] for g in k1])
    y = np.array([float(g["dbleu"] < -1) for g in k1])
    grp = np.array([g["dir"] + g["id"] for g in k1])
    mk = lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))  # noqa: E731
    pr = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits=5).split(Xf, y, grp):
        pr[te] = mk().fit(Xf[tr], y[tr]).predict_proba(Xf[te])[:, 1]
    res["b"] = {"harm_rate": float(y.mean()), "auc_features": float(roc_auc_score(y, pr)),
                "auc_P": float(roc_auc_score(y, 1 - Xf[:, FEATS.index("P")])),
                "auc_features_by_dir": {d: float(roc_auc_score(y[[g["dir"] == d for g in k1]], pr[[g["dir"] == d for g in k1]])) for d in DIRS}}
    # (c) rules on halves, per-sentence sums (first-order)
    A = [g for g in k1 if g["half"] == "A"]
    B = [g for g in k1 if g["half"] == "B"]
    nsent = {h: len({(g["dir"], g["id"]) for g in G if g["half"] == h}) for h in "AB"}
    mA = mk().fit(np.array([[g[f"f_{q}"] for q in FEATS] for g in A]), np.array([float(g["dbleu"] < -1) for g in A]))
    for g in k1:
        g["ph"] = float(mA.predict_proba(np.array([[g[f"f_{q}"] for q in FEATS]]))[0, 1])

    def evalrule(sel, take, ns):
        dd2 = sum(g["dd2"] for g in sel if take(g)) / ns
        dbl = sum(g["dbleu"] for g in sel if take(g)) / ns
        return dd2, dbl, sum(1 for g in sel if take(g))

    def pick(score_key, sign):
        vals = sorted({round(sign * g[score_key], 4) for g in A})
        best = None
        for v in vals:
            take = lambda g, v=v: sign * g[score_key] <= v  # noqa: E731
            dd2, dbl, _ = evalrule(A, take, nsent["A"])
            if dbl >= -0.3 and (best is None or dd2 < best[1]):
                best = (v, dd2, dbl)
        return best
    bl_ = pick("ph", 1.0)                  # take if predicted harm <= tau
    bp_ = pick("f_P", -1.0)                # take if -P <= -tau'  (P >= tau')
    out_c = {}
    for nm, bb, key, sign in (("learned", bl_, "ph", 1.0), ("P_only", bp_, "f_P", -1.0)):
        if bb is None:
            out_c[nm] = None
            continue
        take = lambda g, v=bb[0], key=key, sign=sign: sign * g[key] <= v  # noqa: E731
        dd2, dbl, nt = evalrule(B, take, nsent["B"])
        out_c[nm] = {"threshold": bb[0] * sign, "A_dd2": bb[1], "A_dbleu": bb[2], "B_dd2_per_sent": dd2, "B_dbleu_per_sent": dbl, "B_taken": nt}
    res["c"] = out_c
    L_, P_ = out_c.get("learned"), out_c.get("P_only")
    res["P0R_pass"] = bool(L_ and L_["B_dd2_per_sent"] <= -0.10 and L_["B_dbleu_per_sent"] >= -0.3
                           and (P_ is None or L_["B_dd2_per_sent"] <= P_["B_dd2_per_sent"] - 0.03))
    # oracles (all k, per-sentence sums over all groups, both halves)
    byg = defaultdict(list)
    for g in G:
        byg[(g["dir"], g["id"], g["t"])].append(g)
    ns_all = nsent["A"] + nsent["B"]
    o_d2 = o_bl = s_d2 = s_bl = 0.0
    for gs in byg.values():
        ok = [g for g in gs if g["dbleu"] >= 0]
        if ok:
            g = min(ok, key=lambda x: x["dd2"])
            if g["dd2"] < 0:
                o_d2 += g["dd2"]; o_bl += g["dbleu"]
        sv = [g for g in gs if g["survive"] == 1]
        if sv:
            g = max(sv, key=lambda x: x["k"])
            s_d2 += g["dd2"]; s_bl += g["dbleu"]
    res["oracle_regret"] = {"dd2_per_sent": o_d2 / ns_all, "dbleu_per_sent": o_bl / ns_all}
    res["oracle_survival"] = {"dd2_per_sent": s_d2 / ns_all, "dbleu_per_sent": s_bl / ns_all}
    res["sentences"] = nsent
    with open(a.out, "x", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1, ensure_ascii=False)
    print(json.dumps(res, ensure_ascii=False))


if __name__ == "__main__":
    main()

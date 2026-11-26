"""sst_hsmm_v1 prereg change 41 (P0-ES) analysis.

Items: <data>/items_<d>.jsonl (with "translation"); half A = even line index, half B = odd. Rows: <root>/<job>/blocks_<d>_<name>.jsonl
(hsc13), every job has its own "base" member (no LoRA); members are compared with the base of their own job (paired).
Per stream: p0rl_analyze.stream_stats (proxy D2 s, gap s, zero-commit share, log score S over all reads).
Fitness of a system on a half: F = sum over directions of ( - mean D2 + 0.4 x corpus BLEU ) (sacrebleu, zh / 13a).
  phase1  --jobs <glob of job dirs>: G0-ES = mean over members and directions of the share of items whose translation or block
          timeline differs from the job base; per member F on A / B (and minus its base); ES weights w_k = (F_A(p<k>_pos) -
          F_A(p<k>_neg)) / 2 -> <out>.grad.json; spread and the best-on-A member's B result.
  phase2  --jobs <glob>: in each job: base, step_pos, step_neg. On half B: dF = F(step_pos) - F(base) with a paired item bootstrap
          (2000, seed 20261126; corpus BLEU from resampled sufficient statistics), 90% interval; F(step_pos) > F(step_neg);
          per direction gap(step_pos) - gap(base) <= 0.02 s and BLEU(step_pos) - BLEU(base) >= -0.5. Also all items and half A.
Refuses to overwrite.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from experiments.sst_hsmm_v1.p0rl_analyze import stream_stats  # noqa: E402

W_BLEU = 0.4
DIRS = ("en2zh", "zh2en")


class Bleu:
    def __init__(self, zh):
        from sacrebleu.metrics import BLEU
        self.m = BLEU(tokenize="zh" if zh else "13a")

    def stats(self, hyps, refs):
        return np.array(self.m._extract_corpus_statistics(hyps, [refs]), dtype=float)

    def score(self, st_sum):
        return float(self.m._compute_score_from_stats(list(st_sum)).score)


def load(jobs, data):
    """-> items[d] (list, file order), sys[(job, name)][d] = {id: row}"""
    items = {d: [json.loads(x) for x in open(Path(data) / f"items_{d}.jsonl", encoding="utf-8")] for d in DIRS}
    S = {}
    for j in sorted(glob.glob(jobs)):
        for f in sorted(glob.glob(f"{j}/blocks_*_*.jsonl")):
            stem = Path(f).stem[len("blocks_"):]
            d, name = stem.split("_", 1)
            S.setdefault((Path(j).name, name), {})[d] = {r["id"]: r for r in map(json.loads, open(f, encoding="utf-8"))}
    return items, S


def per_item(items, rows, d):
    zh = d.endswith("zh")
    lang = "cmn" if zh else "eng"
    import sacrebleu
    tk = "zh" if zh else "13a"
    sb = lambda h, r: sacrebleu.sentence_bleu(h, [r], tokenize=tk).score  # noqa: E731
    sc = lambda h, r: 0.0  # noqa: E731   chrF not needed here
    return [stream_stats(rows[it["id"]], it, zh, lang, 0, sb, sc) for it in items]


def fitness(stats_by_dir, idx_by_dir, bleu_st_by_dir, B):
    F, parts = 0.0, {}
    for d in DIRS:
        ix = idx_by_dir[d]
        d2 = float(np.nanmean([stats_by_dir[d][i]["d2"] for i in ix]))
        bl = B[d].score(bleu_st_by_dir[d][ix].sum(0))
        gap = float(np.mean([stats_by_dir[d][i]["gap"] for i in ix]))
        zero = float(np.nanmean([stats_by_dir[d][i]["zero"] for i in ix]))
        S_ = float(np.nanmean([stats_by_dir[d][i]["S"] for i in ix]))
        parts[d] = {"d2": d2, "bleu": bl, "gap": gap, "zero": zero, "S": S_}
        F += -d2 + W_BLEU * bl
    return F, parts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("phase1", "phase2"))
    ap.add_argument("--jobs", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"refuse to overwrite {out}")
    items, S = load(a.jobs, a.data)
    B = {d: Bleu(d.endswith("zh")) for d in DIRS}
    halves = {"A": {d: [i for i in range(len(items[d])) if i % 2 == 0] for d in DIRS},
              "B": {d: [i for i in range(len(items[d])) if i % 2 == 1] for d in DIRS},
              "all": {d: list(range(len(items[d]))) for d in DIRS}}
    cache = {}
    for key, byd in S.items():
        assert set(byd) == set(DIRS), (key, list(byd))
        st = {d: per_item(items[d], byd[d], d) for d in DIRS}
        bst = {d: B[d].stats([byd[d][it["id"]]["translation"] for it in items[d]], [it["translation"] for it in items[d]]) for d in DIRS}
        cache[key] = (st, bst)
    res = {}
    jobs = sorted({k[0] for k in S})
    for (job, name), (st, bst) in sorted(cache.items()):
        r = {}
        for h, ix in halves.items():
            F, parts = fitness(st, ix, bst, B)
            Fb, _ = fitness(cache[(job, "base")][0], ix, cache[(job, "base")][1], B)
            r[h] = {"F": F, "dF_vs_base": F - Fb, **{f"{d}_{k}": v for d, p in parts.items() for k, v in p.items()}}
        if name != "base":
            ch = []
            for d in DIRS:
                b, m = S[(job, "base")][d], S[(job, name)][d]
                ch += [float(b[i]["translation"] != m[i]["translation"] or b[i]["blocks"] != m[i]["blocks"]) for i in b]
            r["changed_share"] = float(np.mean(ch))
        res[f"{job}/{name}"] = r
    summ = {"jobs": jobs}
    if a.mode == "phase1":
        mem = {k: v for k, v in res.items() if not k.endswith("/base")}
        summ["G0_changed_share"] = float(np.mean([v["changed_share"] for v in mem.values()]))
        summ["G0_pass"] = summ["G0_changed_share"] >= 0.20
        w = {}
        for k, v in mem.items():
            nm = k.split("/")[1]
            if nm.endswith("_pos"):
                kk = nm[1:-4]
                neg = [vv for kk2, vv in mem.items() if kk2.split("/")[1] == f"p{kk}_neg"]
                assert len(neg) == 1, kk
                w[kk] = (v["A"]["F"] - neg[0]["A"]["F"]) / 2.0
        summ["weights"] = w
        dFA = {k: v["A"]["dF_vs_base"] for k, v in mem.items()}
        dFB = {k: v["B"]["dF_vs_base"] for k, v in mem.items()}
        best = max(dFA, key=dFA.get)
        summ.update(spread_dF_A_std=float(np.std(list(dFA.values()))), spread_dF_B_std=float(np.std(list(dFB.values()))),
                    corr_dF_A_B=float(np.corrcoef(list(dFA.values()), [dFB[k] for k in dFA])[0, 1]),
                    best_on_A=best, best_dF_A=dFA[best], best_dF_B=dFB[best],
                    pair_corr_pos_neg_A=float(np.corrcoef([mem[k]["A"]["dF_vs_base"] for k in mem if k.endswith("_pos")],
                                                          [mem[k.replace("_pos", "_neg")]["A"]["dF_vs_base"] for k in mem if k.endswith("_pos")])[0, 1]))
        with open(str(out) + ".grad.json", "x") as fh:
            json.dump({"weights": w}, fh, indent=1)
    else:
        rng = np.random.default_rng(20261126)
        assert len(jobs) == 1, jobs
        job = jobs[0]
        stB, bstB = cache[(job, "base")]
        stP, bstP = cache[(job, "step_pos")]
        ixB = halves["B"]
        dist_ = []
        for _ in range(2000):
            dF = 0.0
            for d in DIRS:
                ix = rng.choice(ixB[d], size=len(ixB[d]), replace=True)
                for sgn, (st, bst) in ((1.0, (stP, bstP)), (-1.0, (stB, bstB))):
                    dF += sgn * (-float(np.nanmean([st[d][i]["d2"] for i in ix])) + W_BLEU * B[d].score(bst[d][ix].sum(0)))
            dist_.append(dF)
        lo, hi = np.percentile(dist_, [5, 95])
        rP, rN, rB = res[f"{job}/step_pos"]["B"], res[f"{job}/step_neg"]["B"], res[f"{job}/base"]["B"]
        fl = {d: {"dgap": rP[f"{d}_gap"] - rB[f"{d}_gap"], "dbleu": rP[f"{d}_bleu"] - rB[f"{d}_bleu"], "dd2": rP[f"{d}_d2"] - rB[f"{d}_d2"]} for d in DIRS}
        summ.update(dF_B=rP["dF_vs_base"], dF_B_ci90=[float(lo), float(hi)], F_pos_gt_neg=bool(rP["F"] > rN["F"]),
                    per_dir=fl, guards_ok=all(v["dgap"] <= 0.02 and v["dbleu"] >= -0.5 for v in fl.values()))
        summ["P0_ES_pass"] = bool(lo > 0 and summ["F_pos_gt_neg"] and summ["guards_ok"])
    summ["systems"] = res
    with open(out, "x", encoding="utf-8") as fh:
        json.dump(summ, fh, indent=1, ensure_ascii=False)
    print(json.dumps({k: v for k, v in summ.items() if k != "systems"}, ensure_ascii=False))


if __name__ == "__main__":
    main()

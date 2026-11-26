"""sst_hsmm_v1 prereg change 40-P0: shared fork precheck analysis (G0, P0, P0-fluency, P0').

Inputs per direction: the greedy base rows (blocks_<d>_w1.jsonl of every shard dir) and the clones (forks_<d>.jsonl), from
hsc11_si_thinker --fork-reads 2 --fork-k 4, plus the items file (reference "translation", dur_ms). Group = (id, fork_t):
the K sampled clones + the greedy base. Per stream (proxies, no speech synthesis):
  bleu / chrf   sentence BLEU (sacrebleu, tokenize zh for en2zh, 13a for zh2en) and chrF of the final translation
  d2            LAAL over unit finish times of a playback queue: units of a block start at max(write_ms, queue end),
                0.25 s per zh char / 0.35 s per en word (= decomp3.d2_from with constant unit durations), LD.laal_unitwise
  gap / pauses  sum of max(0, gap - 0.3 s) and the number of gaps > 0.3 s between consecutive blocks' speech
  zero          share of non-final reads t >= fork_t that committed no unit (cv_new == 0)
  S             mean over (read t >= fork_t, uncommitted position i) of y log P + (1 - y) log(1 - P), y = 1 iff the read's
                draft through i equals the final translation through i (unit survival), P clipped to [1e-4, 1 - 1e-4]
Gates (prereg change 40): G0 >= 50% groups whose K samples are not all identical in (d2, bleu); P0 mean best dD2 <= -0.05 s
with >= 25% groups strictly better (best over greedy + samples with chrF >= greedy); P0-fluency mean best dgap <= -0.05 s
(or best dzero <= -3 points) over greedy + samples with chrF >= greedy and d2 <= greedy; P0' >= 50% groups with S varying
and median within-group Spearman(S, chrF) >= 0.

    python -m experiments.sst_hsmm_v1.p0rl_analyze --root eval_outputs/sst_hsmm_v1/p0rl --out <json>
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from experiments.voice_onset_v1.playout import LD, si_units  # noqa: E402
from experiments.uniss_qwen3_omni_v1.evaluation.si_thinker_stream import units  # noqa: E402


def playback(blocks, lang, X, Tn):
    us = 250.0 if lang == "cmn" else 350.0
    fin, end, first, gap, np_ = [], 0.0, True, 0.0, 0
    for b in blocks:
        n = len(si_units(b["text"], lang))
        if not n:
            continue
        st = max(float(b["write_ms"]), end)
        if not first:
            g = st - end
            if g > 300:
                np_ += 1
                gap += g - 300
        first = False
        for k in range(n):
            fin.append(st + (k + 1) * us)
        end = st + n * us
    d2 = LD.laal_unitwise(fin, X, Tn) if fin else float("nan")
    return d2 / 1000.0, gap / 1000.0, np_


def stream_stats(row, it, zh, lang, fork_t, sb, sc):
    X, ref = it["dur_ms"], it["translation"]
    Tn = len(si_units(ref, lang))
    d2, gap, np_ = playback(row["blocks"], lang, X, Tn)
    hyp = row["translation"]
    bleu = sb(hyp, ref)
    chrf = sc(hyp, ref)
    fu = units(hyp, zh)
    zero, nr, ls, npos = 0, 0, 0.0, 0
    for rec in row["drafts"]:
        if rec["t"] < fork_t or "P" not in rec:
            continue
        nr += 1
        zero += int(rec.get("cv_new", 0) == 0)
        du = units(rec["draft"], zh)
        P = rec["P"]
        n = len(du) - len(P)
        for j, p in enumerate(P):
            i = n + j
            y = du[: i + 1] == fu[: i + 1]
            p = min(max(float(p), 1e-4), 1 - 1e-4)
            ls += math.log(p) if y else math.log(1 - p)
            npos += 1
    return dict(d2=d2, gap=gap, pauses=np_, bleu=bleu, chrf=chrf, zero=zero / nr if nr else float("nan"),
                S=ls / npos if npos else float("nan"), hyp=hyp)


def spearman(a, b):
    from scipy.stats import spearmanr
    if len(set(a)) < 2 or len(set(b)) < 2:
        return None
    return float(spearmanr(a, b).correlation)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if Path(a.out).exists():
        raise SystemExit(f"refuse to overwrite {a.out}")
    import sacrebleu
    res = {}
    for d in ("en2zh", "zh2en"):
        zh = d.endswith("zh")
        lang = "cmn" if zh else "eng"
        tk = "zh" if zh else "13a"
        sb = lambda h, r: sacrebleu.sentence_bleu(h, [r], tokenize=tk).score  # noqa: E731
        sc = lambda h, r: sacrebleu.sentence_chrf(h, [r]).score  # noqa: E731
        items = {r["id"]: r for r in map(json.loads, open(Path(a.root) / "data" / f"items_{d}.jsonl", encoding="utf-8"))}
        base, forks = {}, defaultdict(list)
        for f in sorted(glob.glob(str(Path(a.root) / f"run_{d}_*" / f"blocks_{d}_w1.jsonl"))):
            for r in map(json.loads, open(f, encoding="utf-8")):
                base[r["id"]] = r
        for f in sorted(glob.glob(str(Path(a.root) / f"run_{d}_*" / f"forks_{d}.jsonl"))):
            for r in map(json.loads, open(f, encoding="utf-8")):
                forks[(r["id"], r["fork_t"])].append(r)
        G = []
        for (iid, ft), fr in sorted(forks.items()):
            it = items[iid]
            g = stream_stats(base[iid], it, zh, lang, ft, sb, sc)
            ss = [stream_stats(r, it, zh, lang, ft, sb, sc) for r in sorted(fr, key=lambda x: x["fork_k"])]
            G.append((g, ss))
        n = len(G)
        g0 = np.mean([len({(round(s["d2"], 4), round(s["bleu"], 3)) for s in ss}) > 1 for _, ss in G])
        dd2, better, dgap, dzero, svar, rho = [], [], [], [], [], []
        for g, ss in G:
            c1 = [s for s in ss if s["chrf"] >= g["chrf"]]
            b = min([g["d2"]] + [s["d2"] for s in c1])
            dd2.append(b - g["d2"]); better.append(b < g["d2"] - 1e-9)
            c2 = [s for s in c1 if s["d2"] <= g["d2"] + 1e-9]
            dgap.append(min([g["gap"]] + [s["gap"] for s in c2]) - g["gap"])
            zs = [s["zero"] for s in c2 if not math.isnan(s["zero"])]
            if not math.isnan(g["zero"]):
                dzero.append(100 * (min([g["zero"]] + zs) - g["zero"]))
            Ss = [x["S"] for x in [g] + ss if not math.isnan(x["S"])]
            svar.append(len({round(x, 6) for x in Ss}) > 1)
            q = spearman([x["S"] for x in [g] + ss], [x["chrf"] for x in [g] + ss])
            if q is not None and not any(math.isnan(x["S"]) for x in [g] + ss):
                rho.append(q)
        mean = lambda k, w: float(np.nanmean([x[k] for x in w]))  # noqa: E731
        allg = [g for g, _ in G]
        alls = [s for _, ss in G for s in ss]
        r = dict(groups=n,
                 greedy={k: mean(k, allg) for k in ("bleu", "chrf", "d2", "gap", "pauses", "zero", "S")},
                 samples={k: mean(k, alls) for k in ("bleu", "chrf", "d2", "gap", "pauses", "zero", "S")},
                 G0_share_varying=float(g0), G0_pass=bool(g0 >= 0.5),
                 P0_mean_best_dD2_s=float(np.mean(dd2)), P0_share_better=float(np.mean(better)),
                 P0_pass=bool(np.mean(dd2) <= -0.05 and np.mean(better) >= 0.25),
                 P0f_mean_best_dgap_s=float(np.mean(dgap)), P0f_mean_best_dzero_pts=float(np.mean(dzero)) if dzero else None,
                 P0f_pass=bool(np.mean(dgap) <= -0.05 or (dzero and np.mean(dzero) <= -3.0)),
                 P0p_share_S_varying=float(np.mean(svar)), P0p_median_spearman_S_chrf=float(np.median(rho)) if rho else None,
                 P0p_groups_with_rho=len(rho),
                 P0p_pass=bool(np.mean(svar) >= 0.5 and rho and np.median(rho) >= 0))
        res[d] = r
        print(d, json.dumps(r, ensure_ascii=False), flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "x", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()

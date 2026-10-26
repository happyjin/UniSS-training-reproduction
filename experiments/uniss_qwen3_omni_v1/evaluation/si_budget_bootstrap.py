"""Which policy family reaches the higher speech-only ASR-BLEU within a latency budget -- with a bootstrap.

A family is every operating point of one policy (all read windows, thresholds, biases); a point is one system run name
with its talker seeds (system_<name>[-seed<n>[r] | -r]_m<w>). For a budget x, a family's score is the best seed-mean
ASR-BLEU among its points whose segment LAAL (each word at the start of the speech segment carrying it, as p25_summary)
is <= x: the quality one can buy with x seconds, no interpolation. The sentences are resampled with replacement (the
same draw for every run, so the comparison is paired) and BLEU and LAAL are recomputed on each draw; reported per
budget: each family's score, the difference to the reference family, its 95% interval and the share of draws where it
is <= 0 (one-sided p).

    python -m experiments.uniss_qwen3_omni_v1.evaluation.si_budget_bootstrap --direction zh2en --ref LA2 \
        --family 'LA2=<runs dir>/system_la2*' --family 'Bayes=<runs dir>/system_vr-t0.5*' --budgets 1.5,2,2.5
"""

from __future__ import annotations

import argparse
import glob
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from evaluation.text_metrics import normalize_for_bleu, normalize_language
from experiments.uniss_qwen3_omni_v1.evaluation.si_metrics import laal, units


def main() -> None:
    from sacrebleu.metrics import BLEU

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--direction", required=True)
    ap.add_argument("--family", action="append", required=True, help="NAME=glob of system run dirs, comma-separated globs")
    ap.add_argument("--ref", required=True, help="the family the others are compared to")
    ap.add_argument("--budgets", required=True)
    ap.add_argument("--summary", action="append", default=[],
                    help="lo,hi: also the mean difference over a 0.05 s grid of budgets in [lo, hi] -- one test for the "
                         "range, no multiple comparisons")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    fams = {}
    for spec in args.family:
        name, pats = spec.split("=", 1)
        dirs = sorted({p for pat in pats.split(",") for p in glob.glob(pat)})
        pts = defaultdict(list)
        for d in dirs:
            run = Path(d) / f"realsi_{args.direction}"
            if not (run / "asr_speech.jsonl").exists():
                continue
            m = re.fullmatch(r"system_(.+?)(-seed\d+(?:r\d*)?|-r\d*)?_m([\d.]+)", Path(d).name)
            if m:
                pts[(m.group(1), m.group(3))].append(run)
        fams[name] = pts
    runs = [r for pts in fams.values() for rs in pts.values() for r in rs]
    data, lang, ids = {}, None, None
    for run in runs:
        recs = {r["id"]: r for r in map(json.loads, open(run / "results.jsonl", encoding="utf-8"))}
        heard = {r["id"]: r.get("asr_text", "") for r in map(json.loads, open(run / "asr_speech.jsonl", encoding="utf-8"))}
        lang = next(iter(recs.values()))["tgt_lang"]
        data[run] = (recs, heard)
        ids = set(recs) if ids is None else ids & set(recs)
    ids = sorted(ids)
    nl = normalize_language(lang)
    bleu = BLEU(tokenize="zh" if nl == "cmn" else "13a")
    stats, lag = {}, {}
    for run, (recs, heard) in data.items():
        h = [normalize_for_bleu(heard.get(i, ""), nl) for i in ids]
        r = [normalize_for_bleu(recs[i]["translation_ref"], nl) for i in ids]
        stats[run] = np.asarray(bleu._extract_corpus_statistics(h, [r]), dtype=np.float64)
        seg = []
        for i in ids:
            rr = recs[i]
            ds = [bl["start_ms"] for bl in rr["blocks"] for _ in units(bl["text"], lang)]
            seg.append(laal(ds, rr["source_duration_ms"], len(units(rr["translation_ref"], lang))) / 1000 if ds else np.nan)
        lag[run] = np.asarray(seg)
    budgets = [float(x) for x in args.budgets.split(",")]
    ranges = []
    for spec in args.summary:   # the range grids ride along after the listed budgets
        lo, hi = map(float, spec.split(","))
        grid = [round(v, 3) for v in np.arange(lo, hi + 1e-9, 0.05)]
        ranges.append((spec, len(budgets), len(budgets) + len(grid)))
        budgets += grid

    def scores(idx):
        out = {}
        for name, pts in fams.items():
            cur = []
            for rs in pts.values():
                b = np.mean([bleu._compute_score_from_stats(list(stats[r][idx].sum(0))).score for r in rs])
                l = np.mean([np.nanmean(lag[r][idx]) for r in rs])
                cur.append((l, b))
            out[name] = [max([b for l, b in cur if l <= x], default=np.nan) for x in budgets]
        return out

    full = scores(np.arange(len(ids)))
    rng = np.random.default_rng(args.seed)
    draws = [scores(rng.integers(0, len(ids), len(ids))) for _ in range(args.n)]
    print(json.dumps({"direction": args.direction, "sentences": len(ids),
                      "points": {n: {f"{k}_m{w}": len(rs) for (k, w), rs in pts.items()} for n, pts in fams.items()}}), flush=True)
    for spec, a, b in ranges:
        row = {"range_s": spec, "budgets": b - a}
        for n in fams:
            if n == args.ref:
                continue
            full_d = np.nanmean([full[n][j] - full[args.ref][j] for j in range(a, b)])
            d = np.asarray([np.nanmean([dr[n][j] - dr[args.ref][j] for j in range(a, b)]) for dr in draws])
            d = d[~np.isnan(d)]
            row[f"{n}-{args.ref}"] = {"mean_diff": round(float(full_d), 3),
                                      "ci95": [round(float(np.percentile(d, 2.5)), 3), round(float(np.percentile(d, 97.5)), 3)],
                                      "p": round(float((d <= 0).mean()), 4)}
        print(json.dumps(row, ensure_ascii=False), flush=True)
    for j, x in enumerate(budgets[: ranges[0][1] if ranges else len(budgets)]):
        row = {"budget_s": x, "scores": {n: (None if np.isnan(full[n][j]) else round(full[n][j], 2)) for n in fams}}
        for n in fams:
            if n == args.ref or np.isnan(full[n][j]) or np.isnan(full[args.ref][j]):
                continue
            d = np.asarray([dr[n][j] - dr[args.ref][j] for dr in draws])
            d = d[~np.isnan(d)]
            row[f"{n}-{args.ref}"] = {"diff": round(full[n][j] - full[args.ref][j], 2),
                                      "ci95": [round(float(np.percentile(d, 2.5)), 2), round(float(np.percentile(d, 97.5)), 2)],
                                      "p": round(float((d <= 0).mean()), 4)}
        print(json.dumps(row, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

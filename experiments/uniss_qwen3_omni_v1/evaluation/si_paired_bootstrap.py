"""Paired bootstrap (Koehn 2004) of speech-only ASR-BLEU between two system runs on the same sentences: resample the
sentences with replacement, score both corpora on each resample, report the mean difference, its 95% interval and the
share of resamples where B does not beat A (one-sided p). Runs may be several seeds of one system: then each resample
scores every seed and the systems are compared on their seed-mean BLEU.

    python -m experiments.uniss_qwen3_omni_v1.evaluation.si_paired_bootstrap --a <realsi_dir> [--a ...] --b <realsi_dir> [--b ...]
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from evaluation.text_metrics import corpus_bleu, normalize_for_bleu, normalize_language


def load(run: str):
    recs = {r["id"]: r for r in map(json.loads, open(f"{run}/results.jsonl", encoding="utf-8"))}
    heard = {r["id"]: r.get("asr_text", "") for r in map(json.loads, open(f"{run}/asr_speech.jsonl", encoding="utf-8"))}
    lang = next(iter(recs.values()))["tgt_lang"]
    return {i: (heard.get(i, ""), recs[i]["translation_ref"]) for i in recs}, lang


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--a", action="append", required=True)
    ap.add_argument("--b", action="append", required=True)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    A, B = [load(r) for r in args.a], [load(r) for r in args.b]
    lang = A[0][1]
    ids = sorted(set.intersection(*[set(x[0]) for x in A + B]))
    import sacrebleu
    from sacrebleu.metrics import BLEU

    nl = normalize_language(lang)
    bleu = BLEU(tokenize="zh" if nl == "cmn" else "13a")
    def stats(run):   # each sentence's n-gram statistics once, as corpus_bleu normalises them; a resample only sums
        h = [normalize_for_bleu(run[0][i][0], nl) for i in ids]
        r = [normalize_for_bleu(run[0][i][1], nl) for i in ids]
        return np.asarray(bleu._extract_corpus_statistics(h, [r]), dtype=np.float64)
    SA, SB = [stats(r) for r in A], [stats(r) for r in B]
    pos = {i: k for k, i in enumerate(ids)}
    def score(runs, pick):
        S = SA if runs is A else SB
        idx = np.asarray([pos[i] for i in pick])
        return float(np.mean([bleu._compute_score_from_stats(list(st[idx].sum(0))).score for st in S]))
    check = corpus_bleu([A[0][0][i][0] for i in ids], [A[0][0][i][1] for i in ids], language=lang)["score"]
    first = bleu._compute_score_from_stats(list(SA[0].sum(0))).score   # the cached statistics reproduce corpus_bleu
    assert abs(check - first) < 0.05, (check, first)
    base = score(B, ids) - score(A, ids)
    rng = np.random.default_rng(args.seed)
    diffs = []
    for _ in range(args.n):
        pick = [ids[k] for k in rng.integers(0, len(ids), len(ids))]
        diffs.append(score(B, pick) - score(A, pick))
    d = np.asarray(diffs)
    print(json.dumps({"sentences": len(ids), "a_bleu": round(score(A, ids), 2), "b_bleu": round(score(B, ids), 2),
                      "diff": round(base, 3), "ci95": [round(float(np.percentile(d, 2.5)), 3), round(float(np.percentile(d, 97.5)), 3)],
                      "p_b_not_better": round(float((d <= 0).mean()), 4), "seeds": [len(A), len(B)]}), flush=True)


if __name__ == "__main__":
    main()

"""Minimum Bayes Risk selection over a pool of sampled candidates.

Why MBR here
------------
This project already samples eight candidates per utterance and picks one with
a hand-built hierarchical reward copied from HPO: standardise a quality proxy
and a latency proxy, gate the latter on the former, combine as ``q - lam*l``.
That works -- 777 utterances, silence 0.198 to 0.095 -- but the quality proxy
is invented, and it needs one.

MBR needs none.  The decision rule is Bayesian: choose the candidate with the
lowest expected loss under the model's own distribution, approximated by
treating the other samples as pseudo-references,

    h* = argmax_h  (1/|R|) sum_{r in R}  u(h, r)

so the candidates score each other and no reference and no reward model is
required.  The literature's own condition for MBR paying off is exactly our
situation: several near-equivalent hypotheses sharing high model probability --
measured here, the eight candidates match to 0.08 BLEU of each other.

Self-utility is excluded from the mean.  Including it adds ``u(h,h)`` -- a
constant -- to every candidate, so the argmax is unchanged either way; leaving
it out just keeps the number interpretable as agreement with the *others*.

The flow variant
----------------
Pure MBR selects the consensus candidate, which is a statement about content
and says nothing about pauses.  Structure-Conditional MBR (EMNLP 2025) makes
the same observation for dialogue: a utility that only measures similarity
picks the broadly representative candidate and misses latent structure.  Here
the latent structure is timing, so ``--flow-weight`` subtracts a standardised
silence term from the MBR score, giving the same shape as the existing
reranker with the invented quality proxy replaced by the Bayes-risk one.
"""

from __future__ import annotations

import argparse
import json
import statistics
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import sacrebleu

from evaluation import text_metrics
from experiments.uniss_streaming_p2st_traj_v1.evaluation.rerank_candidates import (
    silence_ratio,
    standardise,
)

LANGUAGE = {"en2zh": "cmn", "zh2en": "eng"}


def utility(hypothesis: str, reference: str, *, language: str, kind: str) -> float:
    """Similarity of one candidate to one pseudo-reference.

    Both are normalised the way this project's BLEU is -- punctuation stripped,
    Chinese simplified -- so the utility agrees with the metric the result is
    finally judged by.
    """
    h = text_metrics.normalize_for_bleu(hypothesis, language)
    r = text_metrics.normalize_for_bleu(reference, language)
    if not h or not r:
        return 0.0
    if kind == "chrf":
        return float(sacrebleu.sentence_chrf(h, [r]).score)
    if kind == "bleu":
        tokenize = "zh" if language == "cmn" else "13a"
        return float(sacrebleu.sentence_bleu(h, [r], tokenize=tokenize).score)
    raise ValueError(f"unknown utility {kind!r}")


def mbr_scores(texts: list[str], *, language: str, kind: str) -> list[float]:
    """Expected utility of each candidate against the others as references."""
    n = len(texts)
    if n < 2:
        return [0.0] * n
    scores = []
    for i, h in enumerate(texts):
        total = sum(
            utility(h, r, language=language, kind=kind)
            for j, r in enumerate(texts) if j != i
        )
        scores.append(total / (n - 1))
    return scores


def select(
    texts: list[str], silences: list[float], *,
    language: str, kind: str, flow_weight: float,
) -> tuple[int, list[float]]:
    """Index of the MBR choice, and the score each candidate received."""
    quality = mbr_scores(texts, language=language, kind=kind)
    if flow_weight <= 0.0:
        return max(range(len(quality)), key=lambda i: quality[i]), quality
    combined = [
        q - flow_weight * s
        for q, s in zip(standardise(quality), standardise(silences))
    ]
    return max(range(len(combined)), key=lambda i: combined[i]), combined


# ---------------------------------------------------------------- driver


def _silence(task):
    key, path = task
    try:
        return key, silence_ratio(Path(path))
    except Exception:
        return key, None


def _choose_for_sample(task):
    """One sample's selections under every strategy being compared."""
    sample_id, direction, rows, kinds, flow_weights = task
    language = LANGUAGE.get(direction, "cmn")
    texts = [r["text"] for r in rows]
    sil = [r["silence"] for r in rows]
    out = {}
    for kind in kinds:
        quality = mbr_scores(texts, language=language, kind=kind)
        out[f"mbr_{kind}"] = max(range(len(quality)), key=lambda i: quality[i])
        for w in flow_weights:
            combined = [
                q - w * s for q, s in zip(standardise(quality), standardise(sil))
            ]
            out[f"mbr_{kind}_flow{w:g}"] = max(
                range(len(combined)), key=lambda i: combined[i]
            )
    out["first"] = 0
    out["quietest"] = min(range(len(sil)), key=lambda i: sil[i])
    # The reference-aware bounds, reported only as bounds.
    ref = rows[0]["reference"]
    tokenize = "zh" if language == "cmn" else "13a"
    bleus = [
        float(sacrebleu.sentence_bleu(
            text_metrics.normalize_for_bleu(t, language),
            [text_metrics.normalize_for_bleu(ref, language)], tokenize=tokenize).score)
        for t in texts
    ]
    out["oracle_bleu"] = max(range(len(bleus)), key=lambda i: bleus[i])
    return sample_id, out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollout-root", required=True)
    parser.add_argument("--arm", action="append", required=True)
    parser.add_argument("--utility", action="append", default=[])
    parser.add_argument("--flow-weight", action="append", type=float, default=[])
    parser.add_argument("--workers", type=int, default=64)
    parser.add_argument("--output")
    args = parser.parse_args()
    kinds = args.utility or ["chrf"]
    weights = args.flow_weight or [0.5, 1.0, 2.0]

    root = Path(args.rollout_root)
    arms = {}
    for arm in args.arm:
        manifest = root / arm / "MANIFEST.json"
        if not manifest.exists():
            raise SystemExit(f"missing {manifest}")
        arms[arm] = {
            r["sample_id"]: r
            for r in json.loads(manifest.read_text(encoding="utf-8"))["samples"]
        }
    ids = sorted(set.intersection(*(set(a) for a in arms.values())))
    print(f"{len(ids)} samples x {len(arms)} candidates", flush=True)

    tasks = [
        ((arm, i), str(arms[arm][i]["translation_placed"]))
        for arm in args.arm for i in ids
    ]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        scored = list(pool.map(_silence, tasks, chunksize=16))
    silence = {k: v for k, v in scored}
    print("silence measured", flush=True)

    per_sample = []
    for i in ids:
        rows = []
        for arm in args.arm:
            r = arms[arm][i]
            s = silence[(arm, i)]
            if s is None:
                rows = []
                break
            rows.append({
                "arm": arm, "text": r["target_hypothesis"], "silence": s,
                "reference": r["translation_reference"],
            })
        if rows:
            per_sample.append(
                (i, arms[args.arm[0]][i].get("direction", "en2zh"),
                 rows, kinds, weights)
            )
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        picks = dict(pool.map(_choose_for_sample, per_sample, chunksize=8))
    print("selections made", flush=True)

    lookup = {i: rows for i, _, rows, _, _ in per_sample}
    directions = {i: d for i, d, _, _, _ in per_sample}
    strategies = sorted(next(iter(picks.values())).keys())
    report = {}
    print(f"\n{'strategy':>22} {'en→zh':>8} {'zh→en':>8} {'合计':>8} {'静音':>8}")
    for strategy in ["first"] + [s for s in strategies if s != "first"]:
        chosen = {i: lookup[i][picks[i][strategy]] for i in picks}
        line = {}
        for direction in ("en2zh", "zh2en"):
            sel = [c for i, c in chosen.items() if directions[i] == direction]
            if not sel:
                continue
            line[direction] = text_metrics.corpus_bleu(
                [c["text"] for c in sel], [c["reference"] for c in sel],
                language=LANGUAGE[direction],
            )["score"]
        mean_sil = statistics.mean(c["silence"] for c in chosen.values())
        total = sum(line.values())
        report[strategy] = {**line, "sum": total, "silence": mean_sil}
        print(f"{strategy:>22} {line.get('en2zh', 0):>8.2f} {line.get('zh2en', 0):>8.2f} "
              f"{total:>8.2f} {mean_sil:>8.4f}")
    if args.output:
        Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\n-> {args.output}")


if __name__ == "__main__":
    main()

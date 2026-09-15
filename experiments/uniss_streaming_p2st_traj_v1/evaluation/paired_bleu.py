"""Paired bootstrap resampling for ASR-BLEU between two arms.

Corpus BLEU is a single number with no interval attached, and this project has
already been misled once by reading differences off it: the zh->en component
moved 10.30, 10.94, 11.61, 11.67, 11.92, 12.10 across six conditions that
should have responded alike, a 1.8 point spread on 431 utterances that is
mostly sampling noise.

Both arms transcribe the same utterances, so the comparison can be paired:
resample utterance *indices* and recompute corpus BLEU for both systems on the
same resample, which cancels the per-utterance difficulty that dominates the
spread above.  The reported interval is over the paired difference, and the
p-value is the share of resamples whose difference has the opposite sign to
the observed one.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import sacrebleu

from evaluation import text_metrics

# en->zh is scored with sacrebleu's Chinese tokenizer, zh->en with 13a.
TOKENIZER = {"en2zh": "zh", "zh2en": "13a"}
# The target language each direction produces, for the repo protocol.
LANGUAGE = {"en2zh": "cmn", "zh2en": "eng"}


def load(path: Path, direction: str) -> dict[str, tuple[str, str]]:
    """``sample_id -> (hypothesis, reference)`` for the placed audio."""
    rows = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("direction") != direction or "placed" not in row.get("mode", ""):
            continue
        rows[row["id"]] = (row.get("asr_text") or "", row.get("translation_ref") or "")
    return rows


def corpus_bleu(
    hypotheses: list[str], references: list[str], tokenize: str,
    *, protocol: str = "repo", language: str = "cmn",
) -> float:
    """Corpus BLEU under one of the project's two protocols.

    ``repo`` is ``evaluation.text_metrics``: strip punctuation, simplify
    Chinese, then score.  Every headline ASR-BLEU this project has published
    is that one, and it is also the better fit here because ASR output carries
    no punctuation to begin with.  ``raw`` is bare sacrebleu.

    The choice is not cosmetic: the normalisation changes the reference length
    (10509 against 11510 tokens on en->zh), which moves the brevity penalty
    onto a different system and, on one checkpoint here, flipped the sign of
    the measured difference.
    """
    if protocol not in ("repo", "raw"):
        raise ValueError(f"unknown protocol {protocol!r}")
    if protocol == "repo":
        hypotheses = normalise(hypotheses, language)
        references = normalise(references, language)
    return sacrebleu.corpus_bleu(hypotheses, [references], tokenize=tokenize).score


def normalise(texts: list[str], language: str) -> list[str]:
    """The repo protocol's normalisation: strip punctuation, simplify Chinese.

    Pulled out of the scoring call so a bootstrap normalises each string once
    rather than once per resample -- a thousand passes of OpenCC over the same
    777 utterances, which is most of the run time and changes no result.
    """
    return [text_metrics.normalize_for_bleu(t, language) for t in texts]


def paired_bootstrap(
    system: dict[str, tuple[str, str]],
    baseline: dict[str, tuple[str, str]],
    *,
    tokenize: str,
    samples: int = 1000,
    seed: int = 12345,
    protocol: str = "repo",
    language: str = "cmn",
) -> dict:
    ids = sorted(set(system) & set(baseline))
    if not ids:
        raise SystemExit("the two arms share no utterances")
    sys_h = [system[i][0] for i in ids]
    base_h = [baseline[i][0] for i in ids]
    refs = [system[i][1] for i in ids]
    if protocol == "repo":
        sys_h, base_h, refs = (normalise(x, language) for x in (sys_h, base_h, refs))

    # Already normalised above when the protocol asks for it, so the scorer is
    # bare sacrebleu either way from here on.
    score = lambda h, r: corpus_bleu(h, r, tokenize, protocol="raw", language=language)
    observed = score(sys_h, refs) - score(base_h, refs)
    rng = random.Random(seed)
    deltas = []
    n = len(ids)
    for _ in range(samples):
        pick = [rng.randrange(n) for _ in range(n)]
        h1 = [sys_h[i] for i in pick]
        h0 = [base_h[i] for i in pick]
        r = [refs[i] for i in pick]
        deltas.append(score(h1, r) - score(h0, r))
    deltas.sort()
    lo = deltas[int(0.025 * samples)]
    hi = deltas[int(0.975 * samples) - 1]
    # One-sided: how often the resampled difference contradicts the observed one.
    if observed > 0:
        against = sum(1 for d in deltas if d <= 0)
    else:
        against = sum(1 for d in deltas if d >= 0)
    return {
        "n": n, "delta": observed, "ci_low": lo, "ci_high": hi,
        "p": against / samples,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollout-root", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--arm", action="append", required=True)
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--protocol", choices=("repo", "raw"), default="repo")
    args = parser.parse_args()

    root = Path(args.rollout_root)
    print(f"protocol: {args.protocol}\n")
    print(f"{'arm':>16} {'direction':>9} {'n':>5} {'delta':>8} {'95% CI':>18} {'p':>7}")
    for arm in args.arm:
        for direction, tokenize in TOKENIZER.items():
            base = load(root / args.baseline / "asr" / "asr_results.jsonl", direction)
            system = load(root / arm / "asr" / "asr_results.jsonl", direction)
            got = paired_bootstrap(
                system, base, tokenize=tokenize, samples=args.samples,
                protocol=args.protocol, language=LANGUAGE[direction],
            )
            flag = "" if got["ci_low"] <= 0 <= got["ci_high"] else "  *"
            print(
                f"{arm:>16} {direction:>9} {got['n']:>5} {got['delta']:>+8.2f} "
                f"[{got['ci_low']:>+7.2f},{got['ci_high']:>+7.2f}] {got['p']:>7.3f}{flag}"
            )
    print("\n* marks an interval that excludes zero.")


if __name__ == "__main__":
    main()

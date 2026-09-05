"""One table over every RealSI arm, with every metric this project uses.

Why this exists
---------------
Tonight's inference comparisons were all run on eight longform utterances, and
that turned out to be too few to decide anything: samples whose semantic codes
were bit-identical still moved ASR-BLEU by 1.8, and every apparent winner was
one collapsed utterance away from being a loser.  Three conclusions were read
off that noise and later withdrawn.

So the arms are re-run on the full 777-segment RealSI selection and compared
here in one place, with the metrics kept separate by direction -- pooling
en->zh and zh->en is what produced the worst of tonight's mistakes, because
English targets carry about four times the characters of Chinese ones and the
two errors cancelled in the average.

What it reads
-------------
Each arm directory under the rollout root, as written by ``realsi_rollout``:

* ``MANIFEST.json`` -- timing, fragment structure, hypotheses;
* ``metrics/asr_bleu_raw.json`` -- ASR-BLEU by direction and variant;
* the placed audio, for the chop metrics.

Latency comes from ``simuleval_latency.score``, which loads SimulEval 1.1.0's
own scorers from disk and pins them by digest, so the headline numbers are the
field's and not a reimplementation.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

DIRECTIONS = ("en2zh", "zh2en")
# Gold-ceiling code density, measured per direction on the longform set by
# decoding the reference codes through the same BiCodec.  Pooling these two is
# meaningless: they differ by 4x because of the writing systems.
GOLD_CODES_PER_CHAR = {"en2zh": 13.62, "zh2en": 3.27}


def _direction(sample: dict) -> str:
    value = sample.get("direction")
    if value in DIRECTIONS:
        return value
    return "zh2en" if sample.get("src_lang") == "cmn" else "en2zh"


def load_arm(root: Path, arm: str) -> dict | None:
    manifest = root / arm / "MANIFEST.json"
    if not manifest.exists():
        return None
    samples = json.loads(manifest.read_text(encoding="utf-8"))["samples"]
    out: dict[str, object] = {"arm": arm, "samples": len(samples)}
    if samples:
        first = samples[0]
        out["config"] = {
            key: first.get(key)
            for key in (
                "read_stride",
                "read_step_ms",
                "source_holdback",
                "target_holdback",
                "length_prior_scale",
                "min_fragment_tokens",
                "min_final_chunk_ms",
                "text_num_beams",
                "semantic_temperature",
            )
            if first.get(key) is not None
        }

    by_direction: dict[str, list[dict]] = {}
    for sample in samples:
        by_direction.setdefault(_direction(sample), []).append(sample)

    per_direction: dict[str, dict[str, float]] = {}
    for name, rows in by_direction.items():
        codes = [int(r.get("semantic_tokens") or 0) for r in rows]
        chars = [len(r.get("target_hypothesis") or "") for r in rows]
        source = [float(r.get("source_duration_ms") or 0) / 1000 for r in rows]
        refs = [len(r.get("translation_reference") or "") for r in rows]
        density = [c / max(1, t) for c, t in zip(codes, chars)]
        gold = GOLD_CODES_PER_CHAR.get(name)
        per_direction[name] = {
            "utterances": len(rows),
            "codes_per_char": statistics.mean(density) if density else None,
            "codes_per_char_vs_gold": (
                statistics.mean(density) / gold if density and gold else None
            ),
            "speech_over_source": (
                statistics.mean(
                    c * 0.02 / s for c, s in zip(codes, source) if s > 0
                )
                if source
                else None
            ),
            "text_over_reference": (
                statistics.mean(t / max(1, r) for t, r in zip(chars, refs))
                if refs
                else None
            ),
            "fragments": statistics.mean(
                float(r.get("fragments") or 0) for r in rows
            ),
        }
    out["by_direction"] = per_direction

    # Both BLEU protocols, because the two disagree by about 2 points and the
    # headline numbers this project has published all along are the *repo*
    # one: step 1's RealSI k1 result of 14.83 / 10.85 is
    # asr_bleu_repo placed, where asr_bleu_raw placed gives 12.48 / 9.82 on
    # the same audio.  Reporting only one silently breaks comparability with
    # every earlier table.
    for protocol, filename, field in (
        ("repo", "asr_bleu_repo.json", "score"),
        ("raw", "asr_bleu_raw.json", "bleu"),
    ):
        bleu_path = root / arm / "metrics" / filename
        if not bleu_path.exists():
            continue
        groups = json.loads(bleu_path.read_text(encoding="utf-8"))["groups"]
        bleu: dict[str, dict[str, float]] = {}
        for key, value in groups.items():
            if not isinstance(value, dict):
                continue
            variant = "placed" if "_placed:" in key else "concat"
            pair = key.rsplit(":", 1)[-1]
            name = "en2zh" if pair == "eng->cmn" else "zh2en"
            bleu.setdefault(variant, {})[name] = value.get(field)
        out[f"asr_bleu_{protocol}"] = bleu

    try:
        from experiments.uniss_streaming_p2st_pure_ce_v1.evaluation import (
            simuleval_latency,
        )

        out["latency"] = {
            name: simuleval_latency.score(rows)
            for name, rows in by_direction.items()
        }
    except Exception as error:  # SimulEval tree absent, or a scorer refused
        out["latency_error"] = f"{type(error).__name__}: {error}"
    return out


def _fmt(value, width: int = 8, digits: int = 2) -> str:
    if isinstance(value, (int, float)) and value == value:
        return f"{value:>{width}.{digits}f}"
    return f"{'-':>{width}s}"


def render(arms: list[dict]) -> str:
    lines: list[str] = []
    lines.append(
        "| arm | dir | ASR-BLEU (repo) | ASR-BLEU (raw) | codes/char |"
        " vs gold | speech/src | LAAL ms | frags |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for arm in arms:
        for name in DIRECTIONS:
            stats = arm.get("by_direction", {}).get(name)
            if not stats:
                continue
            repo = arm.get("asr_bleu_repo", {}).get("placed", {}).get(name)
            raw = arm.get("asr_bleu_raw", {}).get("placed", {}).get(name)
            laal = (arm.get("latency", {}) or {}).get(name, {}).get("LAAL")
            lines.append(
                f"| `{arm['arm']}` | {name} | {_fmt(repo)} | {_fmt(raw)} |"
                f" {_fmt(stats['codes_per_char'])} |"
                f" {_fmt(stats['codes_per_char_vs_gold'], 7, 3)} |"
                f" {_fmt(stats['speech_over_source'], 7, 3)} |"
                f" {_fmt(laal, 8, 0)} | {_fmt(stats['fragments'], 6, 1)} |"
            )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollout-root", required=True)
    parser.add_argument("--arm", action="append", required=True)
    parser.add_argument("--output")
    parser.add_argument("--markdown")
    args = parser.parse_args()

    root = Path(args.rollout_root)
    arms = [a for a in (load_arm(root, name) for name in args.arm) if a]
    missing = [n for n in args.arm if n not in {a["arm"] for a in arms}]
    report = {
        "schema_version": "uniss_streaming_p2st_realsi_compare_v1",
        "rollout_root": str(root.resolve()),
        "gold_codes_per_char": GOLD_CODES_PER_CHAR,
        "missing_arms": missing,
        "arms": arms,
    }
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(
            json.dumps(report, ensure_ascii=False, indent=1, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    table = render(arms)
    print(table)
    if missing:
        print(f"\nmissing arms: {', '.join(missing)}")
    if args.markdown:
        Path(args.markdown).parent.mkdir(parents=True, exist_ok=True)
        Path(args.markdown).write_text(table + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Step 2a - choose the NAR CTC upsample ratio from this corpus, not from a paper.

``NARBiCodecCTC`` expands every target text token into ``upsample_ratio`` CTC frames with
``repeat_interleave`` and then runs a causal Transformer over the expanded sequence. The
ratio therefore sets two things at once: whether a CTC path exists at all (the head cannot
emit more BiCodec tokens than it has frames) and how expensive the head is (attention is
quadratic in ``upsample_ratio x text_length``). Joint V6 inherited 48 from StreamSpeech-style
defaults; this measures what the UniSS corpus actually needs.

Feasibility uses the same rule as ``training.phase3_whisper_streamspeech_joint.losses``:
CTC needs one extra frame between consecutive identical labels, so a row is feasible when
``upsample_ratio * text_length >= unit_length + adjacent_repeats``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

SCHEMA_VERSION = "simul_s2st_route_v1_step2a_upsample_ratio_v1"
DIRECTIONS = ("eng->cmn", "cmn->eng")
QUANTILES = (0.5, 0.9, 0.95, 0.99, 1.0)
CURRENT_RATIO = 48


@dataclass
class Row:
    direction: str
    text_length: int
    unit_length: int
    adjacent_repeats: int
    source_duration_ms: int

    @property
    def required_frames(self) -> int:
        return self.unit_length + self.adjacent_repeats

    @property
    def source_duration_s(self) -> float:
        return self.source_duration_ms / 1000.0


def adjacent_repeats(tokens: Sequence[int]) -> int:
    return sum(1 for left, right in zip(tokens, tokens[1:]) if left == right)


def iter_lines(manifest: Path, sample_rows: int | None) -> Iterator[str]:
    """Stream a manifest, or take ``sample_rows`` lines spread evenly across its bytes.

    ``joint_train.jsonl`` is ~66 GB / ~24M utterances, so reading it end to end costs far
    more than the length statistics are worth. Seeking to evenly spaced byte offsets and
    taking the next complete line samples the whole file — including whichever shards were
    written last — instead of biasing the answer toward its head.
    """

    size = manifest.stat().st_size
    if sample_rows is None or sample_rows <= 0:
        with manifest.open("r", encoding="utf-8") as handle:
            yield from handle
        return
    with manifest.open("rb") as handle:
        for index in range(sample_rows):
            offset = size * index // sample_rows
            handle.seek(offset)
            if offset:
                handle.readline()
            line = handle.readline()
            if not line:
                return
            yield line.decode("utf-8")


def read_rows(manifests: Sequence[Path], sample_rows: int | None = None) -> Iterator[Row]:
    for manifest in manifests:
        for line in iter_lines(manifest, sample_rows):
            record = json.loads(line)
            direction = f"{record['src_lang']}->{record['tgt_lang']}"
            if direction not in DIRECTIONS:
                continue
            units = record["target_bicodec"]
            text_length = len(record["target_qwen_ids"])
            duration = int(record.get("source_duration_ms", 0))
            if text_length <= 0 or not units or duration <= 0:
                continue
            yield Row(
                direction=direction,
                text_length=text_length,
                unit_length=len(units),
                adjacent_repeats=adjacent_repeats(units),
                source_duration_ms=duration,
            )


def verify_text_lengths(manifest: Path, *, limit: int) -> dict[str, object]:
    """``target_qwen_ids`` is only usable as the head's input length if it is what
    ``build_performance_sample`` would put in the ``performance_translation_text`` span."""

    from transformers import AutoTokenizer

    from training.generate_unist_eval_audio import load_hf_text_encoder

    tokenizer = AutoTokenizer.from_pretrained(
        str(ROOT / "checkpoints/exported_hf/qwen0p5b_phase3_unist198_iter_0009075_hf"),
        local_files_only=True,
    )
    encoder = load_hf_text_encoder(tokenizer)
    checked = 0
    mismatched = 0
    with manifest.open("r", encoding="utf-8") as handle:
        for line in handle:
            if checked >= limit:
                break
            record = json.loads(line)
            if len(encoder(str(record["translation"]))) != len(record["target_qwen_ids"]):
                mismatched += 1
            checked += 1
    return {"checked": checked, "mismatched": mismatched}


def describe(values: np.ndarray) -> dict[str, float]:
    return {
        "count": int(values.size),
        "mean": float(values.mean()),
        **{f"p{int(q * 100)}": float(np.quantile(values, q)) for q in QUANTILES},
    }


def feasibility(rows: Sequence[Row], ratio: int) -> dict[str, float]:
    text = np.array([row.text_length for row in rows], dtype=np.int64)
    required = np.array([row.required_frames for row in rows], dtype=np.int64)
    frames = text * ratio
    feasible = frames >= required
    occupancy = required[feasible] / frames[feasible]
    return {
        "upsample_ratio": ratio,
        "feasible_fraction": float(feasible.mean()),
        "infeasible_rows": int((~feasible).sum()),
        "mean_frames": float(frames.mean()),
        "p99_frames": float(np.quantile(frames, 0.99)),
        "max_frames": int(frames.max()),
        "mean_lattice_occupancy": float(occupancy.mean()) if occupancy.size else 0.0,
        "relative_attention_cost": float((frames.astype(np.float64) ** 2).mean())
        / float(((text * CURRENT_RATIO).astype(np.float64) ** 2).mean()),
    }


def smallest_ratio(rows: Sequence[Row], coverage: float, grid: Sequence[int]) -> int | None:
    for ratio in grid:
        if feasibility(rows, ratio)["feasible_fraction"] >= coverage:
            return ratio
    return None


def anchor_spread(rows: Sequence[Row]) -> dict[str, dict[str, float]]:
    """Compare the two quantities the head could size its frame budget from.

    A constant ``repeat_interleave`` ratio must be large enough for the worst utterance,
    so what matters is not the median frames-per-anchor but how wide the distribution is.
    ``p95_over_p50`` is that width: the factor by which the budget must exceed the typical
    case, which is exactly the fraction of the CTC lattice that ends up as padding.
    """

    required = np.array([row.required_frames for row in rows], dtype=np.float64)
    anchors = {
        "target_text_tokens": np.array([row.text_length for row in rows], dtype=np.float64),
        "source_audio_seconds": np.array([row.source_duration_s for row in rows]),
    }
    result = {}
    for name, values in anchors.items():
        per_anchor = required / values
        p50 = float(np.quantile(per_anchor, 0.5))
        result[name] = {
            **describe(per_anchor),
            "coefficient_of_variation": float(per_anchor.std() / per_anchor.mean()),
            "p95_over_p50": float(np.quantile(per_anchor, 0.95) / p50) if p50 else 0.0,
            "p99_over_p50": float(np.quantile(per_anchor, 0.99) / p50) if p50 else 0.0,
        }
    return result


def partition_degenerate(
    rows: Sequence[Row], limit: float
) -> tuple[list[Row], list[Row]]:
    """Split off rows whose unit/text ratio is too extreme to be a real alignment.

    A constant ``repeat_interleave`` ratio has to be sized for the worst row in the corpus,
    so a handful of utterances with a near-empty translation against a full-length audio
    target can dictate the ratio for everyone. Separating them turns one unanswerable
    question into two answerable ones: how big must the ratio be, and how much data is
    misaligned.
    """

    healthy = [row for row in rows if row.required_frames <= limit * row.text_length]
    degenerate = [row for row in rows if row.required_frames > limit * row.text_length]
    return healthy, degenerate


def render_markdown(payload: dict) -> str:
    overall = payload["overall"]
    recommended = payload["recommended_ratio"]
    lines = [
        "# Step 2a — sizing the NAR CTC upsample ratio",
        "",
        f"> Run `{payload['run_name']}` · {payload['generated_at']} · research only.",
        "",
        f"{overall['rows']:,} target utterances from "
        f"{', '.join(Path(p).name for p in payload['config']['manifests'])}.",
        "",
        "## 1. What the corpus needs",
        "",
        "| Direction | Rows | Text tokens p50/p95/max | BiCodec tokens p50/p95/max | "
        "Required frames per text token p50/p95/p99/max |",
        "|---|---:|---|---|---|",
    ]
    for name in ("overall", *DIRECTIONS):
        block = payload["overall"] if name == "overall" else payload["by_direction"][name]
        text = block["text_length"]
        unit = block["unit_length"]
        ratio = block["required_ratio"]
        lines.append(
            f"| {name} | {block['rows']:,} | "
            f"{text['p50']:.0f} / {text['p95']:.0f} / {text['p100']:.0f} | "
            f"{unit['p50']:.0f} / {unit['p95']:.0f} / {unit['p100']:.0f} | "
            f"{ratio['p50']:.1f} / {ratio['p95']:.1f} / {ratio['p99']:.1f} / {ratio['p100']:.1f} |"
        )
    lines += [
        "",
        "## 2. Feasibility and cost per candidate ratio",
        "",
        "`relative attention cost` is the mean of `(ratio x text_length)^2` divided by the same "
        f"quantity at the currently shipped ratio {CURRENT_RATIO}, i.e. the unit decoder's "
        "self-attention work.",
        "",
        "| Ratio | Feasible (all) | Feasible (healthy) | Infeasible | Mean frames | p99 frames | "
        "Max frames | Lattice occupancy | Relative attention cost |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for entry, healthy in zip(payload["feasibility"], payload["feasibility_healthy"]):
        marker = " **← recommended**" if entry["upsample_ratio"] == recommended else ""
        marker += " (current)" if entry["upsample_ratio"] == CURRENT_RATIO else ""
        lines.append(
            f"| {entry['upsample_ratio']}{marker} | {entry['feasible_fraction'] * 100:.3f}% | "
            f"{healthy['feasible_fraction'] * 100:.3f}% | "
            f"{entry['infeasible_rows']:,} | {entry['mean_frames']:.0f} | "
            f"{entry['p99_frames']:.0f} | {entry['max_frames']:,} | "
            f"{entry['mean_lattice_occupancy'] * 100:.1f}% | "
            f"{entry['relative_attention_cost']:.3f}x |"
        )
    degenerate = payload["degenerate"]
    lines += [
        "",
        "## 3. Smallest ratio meeting a coverage target",
        "",
        f"`healthy` excludes the {degenerate['rows']:,} rows "
        f"({degenerate['fraction'] * 100:.2f}%) whose required frames exceed "
        f"{payload['config']['degenerate_ratio_limit']} per text token — those are "
        "misaligned pairs, not evidence that the head needs a larger ratio.",
        "",
        "| Coverage | Smallest ratio (all rows) | Smallest ratio (healthy rows) |",
        "|---|---:|---:|",
    ]
    for target, value in payload["coverage_targets"].items():
        healthy_value = payload["coverage_targets_healthy"][target]
        lines.append(
            f"| {float(target) * 100:.1f}% | {value if value else 'none in grid'} | "
            f"{healthy_value if healthy_value else 'none in grid'} |"
        )
    lines += [
        "",
        "### Rows excluded as degenerate",
        "",
        "| Metric | p50 | p95 | max |",
        "|---|---:|---:|---:|",
        f"| Text tokens | {degenerate['text_length']['p50']:.0f} | "
        f"{degenerate['text_length']['p95']:.0f} | {degenerate['text_length']['p100']:.0f} |"
        if degenerate["rows"]
        else "| — | — | — | — |",
    ]
    if degenerate["rows"]:
        lines.append(
            f"| BiCodec tokens | {degenerate['unit_length']['p50']:.0f} | "
            f"{degenerate['unit_length']['p95']:.0f} | {degenerate['unit_length']['p100']:.0f} |"
        )
        lines.append(
            f"| Required frames per text token | {degenerate['required_ratio']['p50']:.0f} | "
            f"{degenerate['required_ratio']['p95']:.0f} | "
            f"{degenerate['required_ratio']['p100']:.0f} |"
        )
    lines += [
        "",
        "## 4. Is text length the right thing to size from?",
        "",
        "Measured on healthy rows only. A wide distribution means the constant ratio has to be "
        "set for the tail, and every typical utterance pays for that in padded CTC frames.",
        "",
        "| Anchor | Frames per anchor p50 | p95 | p99 | Coefficient of variation | p95/p50 | p99/p50 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, block_values in payload["anchor_spread"].items():
        lines.append(
            f"| {name.replace('_', ' ')} | {block_values['p50']:.1f} | "
            f"{block_values['p95']:.1f} | {block_values['p99']:.1f} | "
            f"{block_values['coefficient_of_variation']:.3f} | "
            f"{block_values['p95_over_p50']:.2f}x | {block_values['p99_over_p50']:.2f}x |"
        )
    lines += [
        "",
        "## 5. Configuration",
        "",
        "```json",
        json.dumps(payload["config"], indent=2),
        "```",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        action="append",
        default=[],
        help="joint manifest jsonl; repeatable, defaults to full198 train+valid",
    )
    parser.add_argument(
        "--ratio-grid",
        type=int,
        nargs="+",
        default=[8, 12, 16, 20, 24, 28, 32, 40, 48, 64, 80, 96, 128],
    )
    parser.add_argument(
        "--degenerate-ratio-limit",
        type=float,
        default=100.0,
        help="required frames per text token above which a row is treated as misaligned",
    )
    parser.add_argument("--coverage", type=float, nargs="+", default=[0.99, 0.995, 0.999, 1.0])
    parser.add_argument("--verify-text-lengths", type=int, default=2000)
    parser.add_argument(
        "--sample-rows",
        type=int,
        default=200_000,
        help="lines to draw per manifest at evenly spaced byte offsets; 0 reads everything",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    manifests = args.manifest or [
        ROOT / "data/processed/phase3_whisper_streamspeech_joint_v1/full198_joint/joint_train.jsonl",
        ROOT / "data/processed/phase3_whisper_streamspeech_joint_v1/full198_joint/joint_valid.jsonl",
    ]
    for output in (args.output_json, args.output_md):
        if output.exists() and not args.overwrite:
            raise FileExistsError(f"refusing to overwrite Step 2a report: {output}")

    verification = (
        verify_text_lengths(manifests[0], limit=args.verify_text_lengths)
        if args.verify_text_lengths
        else {"checked": 0, "mismatched": 0}
    )
    if verification["mismatched"]:
        raise RuntimeError(
            "target_qwen_ids does not match the Phase3 text encoder on "
            f"{verification['mismatched']}/{verification['checked']} rows; "
            "the measured ratio would not describe the head's real input length"
        )

    rows = list(read_rows(manifests, args.sample_rows or None))
    if not rows:
        raise RuntimeError("no usable rows in the supplied manifests")
    print(json.dumps({"stage": "loaded", "rows": len(rows)}), flush=True)

    def block(subset: Sequence[Row]) -> dict[str, object]:
        text = np.array([row.text_length for row in subset], dtype=np.int64)
        unit = np.array([row.unit_length for row in subset], dtype=np.int64)
        required = np.array([row.required_frames for row in subset], dtype=np.int64)
        return {
            "rows": len(subset),
            "text_length": describe(text),
            "unit_length": describe(unit),
            "required_ratio": describe(required / text),
        }

    grid = sorted(set(args.ratio_grid) | {CURRENT_RATIO})
    healthy, degenerate = partition_degenerate(rows, args.degenerate_ratio_limit)
    if not healthy:
        raise RuntimeError("every row was classified as degenerate; check the limit")
    payload = {
        "schema_version": SCHEMA_VERSION,
        "research_only": True,
        "run_name": args.run_name,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "config": {
            "manifests": [str(path) for path in manifests],
            "ratio_grid": grid,
            "sample_rows_per_manifest": args.sample_rows,
            "current_ratio": CURRENT_RATIO,
            "coverage_targets": args.coverage,
            "text_length_verification": verification,
        },
        "overall": block(rows),
        "by_direction": {
            direction: block([row for row in rows if row.direction == direction])
            for direction in DIRECTIONS
        },
        "direction_counts": dict(Counter(row.direction for row in rows)),
        "feasibility": [feasibility(rows, ratio) for ratio in grid],
        "coverage_targets": {
            str(target): smallest_ratio(rows, target, grid) for target in args.coverage
        },
        "coverage_targets_healthy": {
            str(target): smallest_ratio(healthy, target, grid) for target in args.coverage
        },
        "degenerate": {
            "rows": len(degenerate),
            "fraction": len(degenerate) / len(rows),
            **({} if not degenerate else block(degenerate)),
        },
        "healthy": block(healthy),
        "feasibility_healthy": [feasibility(healthy, ratio) for ratio in grid],
        "anchor_spread": anchor_spread(healthy),
    }
    payload["config"]["degenerate_ratio_limit"] = args.degenerate_ratio_limit
    payload["recommended_ratio"] = payload["coverage_targets_healthy"].get("0.999")

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(render_markdown(payload), encoding="utf-8")
    print(
        json.dumps(
            {
                "stage": "done",
                "recommended_ratio": payload["recommended_ratio"],
                "coverage_targets": payload["coverage_targets"],
                "report": str(args.output_md),
            }
        )
    )


if __name__ == "__main__":
    main()

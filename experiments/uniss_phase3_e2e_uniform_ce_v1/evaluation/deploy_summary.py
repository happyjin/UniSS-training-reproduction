"""Summarise the deployment-configuration checks across a continue-bias sweep.

The delta=0 gate and the delta=5 deployment probe bracket the target text
length band from opposite sides on this checkpoint -- 0.755 below and 1.528
above -- so the band is reachable by the scalar bias alone.  This module reads
the probe roots a sweep produced and reports the same checks the gate applies,
computed from the identical ``e_s2s_free`` fields, so the two are comparable.

WRITE_MT is deliberately absent: the gate derives it from its own event
bookkeeping, and the probe's decision log does not carry a matching event
denominator, so a number computed here would not be the gate's number.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Iterator, Mapping, Sequence

COVERAGE_FLOOR = 0.666
NATURAL_EOS_FLOOR = 0.875
LENGTH_BAND = (0.9, 1.2)
LENGTH_MAX = 2.0


def _samples(root: Path) -> Iterator[Mapping[str, object]]:
    for path in sorted((root / "workers").glob("*.json")):
        payload = json.loads(path.read_text())
        for sample in payload.get("samples") or ():
            if isinstance(sample, dict) and isinstance(sample.get("e_s2s_free"), dict):
                yield sample


def summarise(root: Path) -> dict[str, object] | None:
    rows = list(_samples(root))
    if not rows:
        return None
    free = [row["e_s2s_free"] for row in rows]
    ratios = [
        len(str(item["target_hypothesis"])) / max(1, len(str(row["translation_reference"])))
        for item, row in zip(free, rows)
    ]
    natural_eos = statistics.fmean(1.0 if item["natural_eos"] else 0.0 for item in free)
    coverage = statistics.fmean(float(item["semantic_coverage"]) for item in free)
    median_ratio = statistics.median(ratios)
    worst_ratio = max(ratios)
    checks = {
        "natural_eos": natural_eos >= NATURAL_EOS_FLOOR,
        "semantic_coverage": coverage >= COVERAGE_FLOOR,
        "text_length_ratio_median": LENGTH_BAND[0] <= median_ratio <= LENGTH_BAND[1],
        "text_length_ratio_max": worst_ratio <= LENGTH_MAX,
    }
    return {
        "root": str(root),
        "samples": len(rows),
        "natural_eos": natural_eos,
        "semantic_coverage": coverage,
        "text_length_ratio_median": median_ratio,
        "text_length_ratio_max": worst_ratio,
        "malformed_segments_mean": statistics.fmean(
            float(item["malformed_segments"]) for item in free
        ),
        "semantic_tokens_over_reference": statistics.fmean(
            float(item["semantic_tokens"]) / max(1.0, float(item["semantic_reference_tokens"]))
            for item in free
        ),
        "checks": checks,
        "checks_passed": sum(1 for value in checks.values() if value),
        "checks_total": len(checks),
    }


def _parse_runs(values: Sequence[str]) -> list[tuple[str, Path]]:
    runs: list[tuple[str, Path]] = []
    for value in values:
        if "=" not in value:
            raise ValueError(f"expected label=path, got {value!r}")
        label, _, path = value.partition("=")
        runs.append((label, Path(path)))
    return runs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", default=[], required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    report: dict[str, object] = {"schema_version": 1, "runs": {}}
    header = (
        f"{'run':<22s}{'nat_eos':>9s}{'coverage':>10s}{'len_med':>9s}"
        f"{'len_max':>9s}{'malf':>8s}{'sem/ref':>9s}{'pass':>7s}"
    )
    print(header)
    print("-" * len(header))
    for label, root in _parse_runs(args.run):
        summary = summarise(root)
        if summary is None:
            print(f"{label:<22s}  (no e_s2s_free samples)")
            continue
        report["runs"][label] = summary  # type: ignore[index]
        print(
            f"{label:<22s}{summary['natural_eos']:>9.3f}"
            f"{summary['semantic_coverage']:>10.3f}"
            f"{summary['text_length_ratio_median']:>9.3f}"
            f"{summary['text_length_ratio_max']:>9.3f}"
            f"{summary['malformed_segments_mean']:>8.2f}"
            f"{summary['semantic_tokens_over_reference']:>9.3f}"
            f"{summary['checks_passed']:>4d}/{summary['checks_total']:<2d}"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()

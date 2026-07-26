"""Aggregate UniSS generation and metric JSON files into one report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from evaluation.io_utils import write_json


METRIC_FILES = {
    "text_bleu": "metrics/text_bleu.json",
    "speech_bleu": "metrics/speech_bleu.json",
    "slc": "metrics/slc.json",
    "utmos": "metrics/utmos.json",
    "autopcp": "metrics/autopcp.json",
}


def read_optional(path: Path):
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def collect_run(path: Path) -> dict[str, object]:
    return {
        "path": str(path.resolve()),
        "run_config": read_optional(path / "run_config.json") or read_optional(path / "vllm/run_config.json"),
        "summary": read_optional(path / "summary.json"),
        "generation_summary": read_optional(path / "vllm/generation_summary.json"),
        "metrics": {name: read_optional(path / relative) for name, relative in METRIC_FILES.items()},
    }


def markdown_report(runs: dict[str, dict[str, object]]) -> str:
    lines = ["# UniSS evaluation aggregate report", ""]
    for name, run in runs.items():
        lines.extend([f"## {name}", "", f"Path: `{run['path']}`", ""])
        metrics = run["metrics"]
        for metric_name in ("text_bleu", "speech_bleu", "slc", "utmos", "autopcp"):
            report = metrics.get(metric_name)  # type: ignore[union-attr]
            if not report:
                continue
            groups = report.get("groups", {})
            lines.extend([f"### {metric_name}", "", "| Group | Samples | Primary value |", "| --- | ---: | ---: |"])
            for group_name, values in sorted(groups.items()):
                sample_count = values.get("sample_count", "")
                if "score" in values:
                    primary = values["score"]
                elif "mean" in values:
                    primary = values["mean"]
                elif "slc_0_2" in values:
                    primary = f"SLC0.2={values['slc_0_2']:.6f}; SLC0.4={values['slc_0_4']:.6f}"
                else:
                    primary = ""
                lines.append(f"| {group_name} | {sample_count} | {primary} |")
            lines.append("")
    return "\n".join(lines) + "\n"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", nargs="+", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    runs = {path.name: collect_run(path) for path in args.run}
    write_json(args.output_dir / "aggregate_report.json", {"runs": runs})
    (args.output_dir / "aggregate_report.md").write_text(markdown_report(runs), encoding="utf-8")
    print(json.dumps({"runs": list(runs)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

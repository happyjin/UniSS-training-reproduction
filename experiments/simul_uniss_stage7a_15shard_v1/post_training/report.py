"""Write the Stage7A action-policy comparison report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from evaluation.simultaneous_streaming.stage3_aggregate import aggregate_split


def fmt(value: float) -> str:
    return f"{value:.4f}"


def read_metrics(path: Path) -> dict[str, float]:
    return json.loads(path.read_text(encoding="utf-8"))["metrics"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--e0-dir", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    e0_dir = Path(args.e0_dir)
    stage6 = aggregate_split(e0_dir / "stage6" / "dev")
    fixed = json.loads((e0_dir / "fixed_wait_k_dev.json").read_text(encoding="utf-8"))
    experiments = {
        name: {
            split: read_metrics(run_dir / name / f"{split}.json")
            for split in ("dev", "test")
        }
        for name in ("e1_continued_sft", "e2_grpo_g4", "e3_grpo_g8")
    }
    result = {
        "schema_version": "simul_uniss_stage7a_action_comparison_v1",
        "stage6_dev": {"events": stage6["events"], "samples": stage6["samples"]},
        "fixed_wait_k_dev": fixed["policies"],
        "experiments": experiments,
    }
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    events = stage6["events"]
    samples = stage6["samples"]
    rows = [
        (
            "| E0 Stage6 | dev | "
            f"{fmt(events['binary_accuracy'])} | {fmt(events['write']['f1'])} | "
            f"{fmt(events['premature_write_given_wait'])} | "
            f"{fmt(events['unnecessary_wait_given_write'])} | "
            f"{samples['first_write_absolute_error_ms_mean']:.2f} | "
            f"{fmt(samples['final_flush_success_rate'])} |"
        )
    ]
    labels = {
        "e1_continued_sft": "E1 continued SFT",
        "e2_grpo_g4": "E2 GRPO G4",
        "e3_grpo_g8": "E3 GRPO G8",
    }
    for name, splits in experiments.items():
        for split, metrics in splits.items():
            rows.append(
                f"| {labels[name]} | {split} | {fmt(metrics['accuracy'])} | "
                f"{fmt(metrics['write_f1'])} | "
                f"{fmt(metrics['premature_write_given_wait'])} | "
                f"{fmt(metrics['unnecessary_wait_given_write'])} | "
                f"{metrics['first_write_mae_ms']:.2f} | "
                f"{fmt(metrics['final_flush_success'])} |"
            )
    fixed_rows = [
        f"| {name} | {fmt(values['accuracy'])} | {fmt(values['write_f1'])} | "
        f"{fmt(values['premature_write_given_wait'])} | "
        f"{fmt(values['unnecessary_wait_given_write'])} | "
        f"{values['first_write_mae_ms']:.2f} |"
        for name, values in fixed["policies"].items()
    ]
    report = "\n".join(
        [
            "# Simul-UniSS Stage7A 15-shard action-policy comparison",
            "",
            "> Action-policy proxy on pseudo-proportional schedules; not yet end-to-end S2ST quality/latency.",
            "",
            "## Learned policies",
            "",
            "| Experiment | Split | Accuracy | WRITE F1 | Premature WRITE | Unnecessary WAIT | First-WRITE MAE ms | Final flush |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            *rows,
            "",
            "## Fixed wait-k dev baselines",
            "",
            "| wait-k | Accuracy | WRITE F1 | Premature WRITE | Unnecessary WAIT | First-WRITE MAE ms |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
            *fixed_rows,
            "",
            "## Interpretation boundary",
            "",
            "E2/E3 must outperform E1, not only E0, to support a GRPO-specific claim.",
            "Lower first-WRITE is acceptable only if premature WRITE, final flush, and WRITE F1 remain stable.",
            "The untied HF exports must next run through unchanged free-running Stage4/6 generation, audio decode,",
            "offline-comparable quality metrics, and real streaming latency metrics before a full198 Stage7 decision.",
            "",
            f"Raw aggregate: `{output}`",
            "",
        ]
    )
    Path(args.report).write_text(report, encoding="utf-8")
    print(json.dumps({"output": str(output), "report": args.report}, sort_keys=True))


if __name__ == "__main__":
    main()

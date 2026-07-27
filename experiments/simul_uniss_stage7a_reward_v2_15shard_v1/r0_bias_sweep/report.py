"""Aggregate the E3-v1 dev WRITE-logit-bias action sweep."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--selected-bias", required=True)
    args = parser.parse_args()
    input_dir = Path(args.input_dir)
    rows = []
    for path in sorted(input_dir.glob("bias_*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        metrics = payload["metrics"]
        rows.append({"bias": float(payload["write_logit_bias"]), **metrics})
    if not rows:
        raise SystemExit("no R0 bias results found")
    rows.sort(key=lambda item: item["bias"])
    baseline = next(item for item in rows if item["bias"] == 0.0)
    eligible = [
        item
        for item in rows
        if item["premature_write_given_wait"]
        <= baseline["premature_write_given_wait"] + 0.01
        and item["final_flush_success"] >= baseline["final_flush_success"]
        and item["predicted_writes_per_sample"]
        >= baseline["predicted_writes_per_sample"] * 0.98
    ]
    selected = min(
        eligible or [baseline],
        key=lambda item: (
            item["first_write_delta_ms"],
            item["unnecessary_wait_given_write"],
        ),
    )
    output = {
        "schema_version": "simul_uniss_stage7a_reward_v2_r0_bias_sweep_v1",
        "baseline": baseline,
        "selected": selected,
        "results": rows,
    }
    Path(args.output_json).write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    Path(args.selected_bias).write_text(f"{selected['bias']:.2f}\n", encoding="utf-8")
    table = [
        "| Bias | First delta ms | First MAE ms | Premature | Unnecessary WAIT | Final flush | Writes/sample |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in rows:
        table.append(
            f"| {item['bias']:.2f} | {item['first_write_delta_ms']:.2f} | "
            f"{item['first_write_mae_ms']:.2f} | {item['premature_write_given_wait']:.4f} | "
            f"{item['unnecessary_wait_given_write']:.4f} | {item['final_flush_success']:.4f} | "
            f"{item['predicted_writes_per_sample']:.3f} |"
        )
    report = "\n".join(
        [
            "# Stage7A Reward-v2 R0 WRITE-bias dev sweep",
            "",
            *table,
            "",
            f"Selected dev bias: `{selected['bias']:.2f}`.",
            "Selection minimizes signed first-WRITE delay under premature, final-flush, and coverage gates.",
            "This is an action-policy diagnostic; the selected point still requires free-running dev quality evaluation.",
            "",
        ]
    )
    Path(args.report).write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()

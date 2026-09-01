#!/usr/bin/env python3
"""Answer "is every loss term actually optimizing?" from a training log.

This project has been bitten by the question.  In the content-first run
`real_prefix_kd`, `prefix_stability` and `speaker_consistency` carried non-zero
weights and were identically zero for all 717 updates, because the masks they
selected on were empty -- nobody noticed until the run was over.  A roll-in
weight with a zero roll-in rate fails the same way.

So this classifies every metric the trainer emits:

* `aggregate`   -- a weighted combination of components, not computed directly;
                   its denominator is reported as zero by construction
                   (`boundary_eos`, `semantic_boundary_binary`).
* `component`   -- one half of an aggregate, emitted for monitoring.
* `monitor`     -- weight 0.0 on purpose; costs nothing, reported anyway.
* `dead`        -- weight above zero and the denominator was never above zero.
                   This is the failure mode above and it is always a bug.
* `negligible`  -- weight above zero but supervising so few rows that its
                   contribution is noise.
* `optimizing` / `flat` / `rising` -- alive, with the direction of travel.

`rising` is not automatically wrong: an objective with terms in tension will
trade some of them.  It is reported so the trade is visible rather than implicit.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
from typing import Any

# Below this many supervised rows a term's gradient is noise at any weight.
NEGLIGIBLE_ROWS = 512
TREND_EPSILON = 1e-4

AGGREGATES = ("boundary_eos", "semantic_boundary_binary")
COMPONENTS = (
    "boundary_ce",
    "eos_ce",
    "semantic_boundary_binary_end",
    "semantic_boundary_binary_continue",
)


def parse_weights(log_path: str) -> dict[str, float]:
    """Read the weights out of Megatron's own argument dump, plus the env terms."""

    weights: dict[str, float] = {}
    # Long names get no dot padding at all in Megatron's dump, so the
    # separator is "dots or spaces", not "dots".
    argument = re.compile(r"^\s*e2e_([a-z0-9_]+?)_weight[. ]+([\d.eE+-]+)\s*$")
    extension = re.compile(r'\{"objective_extension": (\{.*?\})')
    with open(log_path, errors="replace") as handle:
        for line in handle:
            found = argument.match(line)
            if found:
                weights[found.group(1)] = float(found.group(2))
            hit = extension.search(line)
            if hit:
                for name, value in json.loads(hit.group(1)).items():
                    if isinstance(value, (int, float)) and not name.endswith(
                        ("_logit_margin", "_window")
                    ):
                        weights[name] = float(value)
    # The dump spells some terms differently from the metric names.
    alias = {
        "asr": "asr_ce",
        "mt": "mt_ce",
        "semantic": "semantic_ce",
        "replay": "replay_ce",
        "commit": "commit_consistency",
        "v1_asr_kl": "v1_asr_kl",
        "phase3_kl": "phase3_kl",
        "semantic_end": "semantic_end_ce",
        "content_end": "content_end_ce",
        "semantic_rollin_end": "semantic_rollin_end_ce",
        "speaker_continuity": "speaker_continuity",
        "boundary_eos": "boundary_eos",
        "semantic_rollin_continue_decision_margin": (
            "semantic_rollin_continue_decision_margin"
        ),
        "semantic_rollin_continue_margin": "semantic_rollin_continue_margin",
        "semantic_continue_margin": "semantic_continue_margin",
        "semantic_end_margin": "semantic_end_margin",
        "semantic_rollin_end_margin": "semantic_rollin_end_margin",
        "semantic_boundary_binary": "semantic_boundary_binary",
    }
    renamed = {}
    for name, value in weights.items():
        renamed[alias.get(name, name)] = value
    return renamed


def read_series(log_path: str) -> tuple[list[dict[str, float]], set[str]]:
    names: set[str] = set()
    rows: list[dict[str, float]] = []
    metric = re.compile(r"(loss|denominator|weighted)/([a-z0-9_]+): +([\d.eE+-]+)")
    with open(log_path, errors="replace") as handle:
        for line in handle:
            found = re.search(r"iteration +(\d+)/", line)
            if not found:
                continue
            row: dict[str, float] = {"it": float(found.group(1))}
            for kind, name, value in metric.findall(line):
                names.add(name)
                row[f"{kind}/{name}"] = float(value)
            rows.append(row)
    return rows, names


def audit(log_path: str) -> dict[str, Any]:
    weights = parse_weights(log_path)
    rows, names = read_series(log_path)
    if not rows:
        raise SystemExit(f"no iteration lines in {log_path}")
    entries = []
    for name in sorted(names):
        weight = weights.get(name)
        loss_key, denominator_key = f"loss/{name}", f"denominator/{name}"
        fired = [row for row in rows if row.get(denominator_key, 0.0) > 0.0]
        share = len(fired) / len(rows)
        median_rows = (
            statistics.median([row[denominator_key] for row in fired]) if fired else 0.0
        )
        entry: dict[str, Any] = {
            "name": name,
            "weight": weight,
            "fired_batch_share": share,
            "median_supervised_rows": median_rows,
        }
        if name in AGGREGATES:
            entry["kind"] = "aggregate"
        elif name in COMPONENTS:
            entry["kind"] = "component"
        elif not fired:
            entry["kind"] = "dead" if (weight or 0.0) > 0.0 else "monitor_dead"
        elif (weight or 0.0) == 0.0:
            entry["kind"] = "monitor"
        elif median_rows < NEGLIGIBLE_ROWS:
            entry["kind"] = "negligible"
        else:
            entry["kind"] = "active"
        if fired:
            quarter = max(3, len(fired) // 4)
            early = [row[loss_key] for row in fired[:quarter] if loss_key in row]
            late = [row[loss_key] for row in fired[-quarter:] if loss_key in row]
            if early and late:
                first, last = statistics.fmean(early), statistics.fmean(late)
                entry.update(
                    {
                        "first_quarter_mean": first,
                        "last_quarter_mean": last,
                        "delta": last - first,
                        "trend": (
                            "optimizing"
                            if last - first < -TREND_EPSILON
                            else "rising"
                            if last - first > TREND_EPSILON
                            else "flat"
                        ),
                    }
                )
        entries.append(entry)
    problems = [e for e in entries if e["kind"] == "dead"]
    negligible = [e for e in entries if e["kind"] == "negligible"]
    active = [e for e in entries if e["kind"] == "active"]
    return {
        "schema_version": "uniss_e2e_loss_audit_v1",
        "log": log_path,
        "iterations_seen": len(rows),
        "status": "failed" if problems else "passed",
        "counts": {
            "metrics_emitted": len(entries),
            "active_weighted": len(active),
            "negligible": len(negligible),
            "dead_with_weight": len(problems),
            "zero_weight_monitors": sum(
                1 for e in entries if e["kind"].startswith("monitor")
            ),
            "aggregates": sum(1 for e in entries if e["kind"] == "aggregate"),
            "components": sum(1 for e in entries if e["kind"] == "component"),
            "optimizing": sum(1 for e in active if e.get("trend") == "optimizing"),
            "rising": sum(1 for e in active if e.get("trend") == "rising"),
        },
        "dead_with_weight": [e["name"] for e in problems],
        "negligible_terms": [
            {"name": e["name"], "weight": e["weight"], "rows": e["median_supervised_rows"]}
            for e in negligible
        ],
        "entries": entries,
    }


def render(result: dict[str, Any]) -> str:
    counts = result["counts"]
    lines = [
        f"loss 审计: {result['status']}   共 {result['iterations_seen']} 个迭代",
        f"  发出的指标 {counts['metrics_emitted']}  "
        f"= 有效加权 {counts['active_weighted']}"
        f" + 权重0监控 {counts['zero_weight_monitors']}"
        f" + 合并项 {counts['aggregates']}"
        f" + 组件 {counts['components']}"
        f" + 分母过小 {counts['negligible']}"
        f" + 带权重却从未开火 {counts['dead_with_weight']}",
        f"  有效项中: 在优化 {counts['optimizing']}  上升 {counts['rising']}",
        "",
        f"{'指标':<44s}{'权重':>7s}{'开火':>7s}{'分母':>9s}{'趋势':>12s}  类别",
    ]
    order = {"dead": 0, "negligible": 1, "active": 2, "monitor": 3,
             "monitor_dead": 4, "aggregate": 5, "component": 6}
    for entry in sorted(
        result["entries"], key=lambda e: (order.get(e["kind"], 9), -(e["weight"] or 0))
    ):
        weight = f"{entry['weight']:.2f}" if entry["weight"] is not None else "?"
        lines.append(
            f"{entry['name']:<44s}{weight:>7s}"
            f"{100 * entry['fired_batch_share']:>6.0f}%"
            f"{entry['median_supervised_rows']:>9.0f}"
            f"{entry.get('trend', '-'):>12s}  {entry['kind']}"
        )
    if result["dead_with_weight"]:
        lines += ["", "带权重却从未开火(总是 bug): " + ", ".join(result["dead_with_weight"])]
    if result["negligible_terms"]:
        lines += ["", "分母过小、贡献是噪声:"]
        for item in result["negligible_terms"]:
            lines.append(f"  {item['name']}  权重 {item['weight']}  仅 {item['rows']:.0f} 行")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = audit(args.log)
    import os

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=1, sort_keys=True)
    print(render(result))


if __name__ == "__main__":
    main()

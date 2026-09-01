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

Trends are computed **per batch type**, never on the pooled mean.  Batches
alternate by task family and the families sit at very different loss levels, so
a shift in the mix moves the pooled mean on its own.  Measured on this run at
step 794: `content_end_margin` pooled +0.823 while falling in all three strata
it appears in (-0.065, -0.367, -0.052), and `boundary_ce` pooled +0.605 while
falling in all three (-0.006, -0.056, -0.011).  Both would have been reported as
regressions.  A term whose pooled and stratified directions disagree is marked
`confounded_pooled` so the artefact is named rather than silently corrected.
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
# A stratum needs this many firing batches before its trend means anything.
MIN_STRATUM_BATCHES = 8


def batch_type(row: dict[str, float]) -> str:
    """Which task family this batch came from, read off the denominators."""

    fired = lambda name: row.get(f"denominator/{name}", 0.0) > 0.0
    if fired("replay_ce"):
        return "replay"
    if fired("semantic_ce") and fired("mt_ce") and fired("asr_ce"):
        return "interleaved"
    if fired("mt_ce"):
        return "incremental_mt"
    if fired("asr_ce"):
        return "streaming_asr"
    return "other"

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
            def quarters(batch: list[dict[str, float]]) -> tuple[float, float] | None:
                quarter = max(3, len(batch) // 4)
                early = [row[loss_key] for row in batch[:quarter] if loss_key in row]
                late = [row[loss_key] for row in batch[-quarter:] if loss_key in row]
                if not early or not late:
                    return None
                return statistics.fmean(early), statistics.fmean(late)

            def label(delta: float) -> str:
                if delta < -TREND_EPSILON:
                    return "optimizing"
                return "rising" if delta > TREND_EPSILON else "flat"

            pooled = quarters(fired)
            strata: dict[str, dict[str, float]] = {}
            grouped: dict[str, list[dict[str, float]]] = {}
            for row in fired:
                grouped.setdefault(batch_type(row), []).append(row)
            for name, batch in grouped.items():
                if len(batch) < MIN_STRATUM_BATCHES:
                    continue
                pair = quarters(batch)
                if pair is None:
                    continue
                strata[name] = {
                    "batches": len(batch),
                    "first_quarter_mean": pair[0],
                    "last_quarter_mean": pair[1],
                    "delta": pair[1] - pair[0],
                }
            if pooled:
                entry["pooled_delta"] = pooled[1] - pooled[0]
                entry["pooled_trend"] = label(pooled[1] - pooled[0])
            if strata:
                entry["strata"] = strata
                # The stratified verdict is the worst stratum: a term that
                # regresses anywhere should not be hidden by an average.
                worst = max(strata.values(), key=lambda s: s["delta"])
                entry["delta"] = worst["delta"]
                entry["trend"] = label(worst["delta"])
                entry["first_quarter_mean"] = worst["first_quarter_mean"]
                entry["last_quarter_mean"] = worst["last_quarter_mean"]
                if entry.get("pooled_trend") and entry["pooled_trend"] != entry["trend"]:
                    entry["confounded_pooled"] = True
            elif pooled:
                entry["delta"] = pooled[1] - pooled[0]
                entry["trend"] = label(pooled[1] - pooled[0])
                entry["first_quarter_mean"] = pooled[0]
                entry["last_quarter_mean"] = pooled[1]
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
            "confounded_pooled": sum(1 for e in entries if e.get("confounded_pooled")),
        },
        "confounded_terms": [
            e["name"] for e in entries if e.get("confounded_pooled")
        ],
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
        f"  有效项中: 在优化 {counts['optimizing']}  上升 {counts['rising']}"
        f"   (趋势按批类型分层;{counts['confounded_pooled']} 项的合并均值方向与分层相反)",
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
            + ("  [合并均值方向相反,是构成假象]" if entry.get("confounded_pooled") else "")
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

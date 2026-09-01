#!/usr/bin/env python3
"""Score the run against the predictions the bias sweep made, and name the cause.

The bias sweep did not just say "this should work" -- it produced a specific
number for every axis, measured at the inference bias that reproduces the target
behaviour.  So a failure is diagnosable rather than merely disappointing: each
prediction that does not land maps to one named cause and one next action, and
every branch below was itself measured, not guessed.

Written so the chain can run it unattended: it reads the gate JSON and the
worker reports, prints a table, and writes VERDICT.json.
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import math
import os
import os as _os
import statistics
from typing import Any

# Measured at delta_cont = 3/4 on iter_0002264 and iter_0001132; see
# reports/uniss_phase3_e2e_speak_decision_v1/family_logit_probe/
# BIAS_SWEEP_ANALYSIS.zh-CN.md
PREDICTIONS: tuple[dict[str, Any], ...] = (
    {
        "key": "write_mt_per_event",
        "label": "WRITE_MT / event",
        "baseline": 0.168,
        "target": 0.50,
        "direction": "at_least",
        "bias_reference": 0.863,
        "cause": "the continue-after-fragment margin did not move the decision",
        "action": (
            "Do NOT raise the weight or the margin. Two runs have now moved this "
            "decision monotonically the WRONG way at inference: margin 1.0 took "
            "the median gap -2.88 -> -3.75, margin 3.0 took it -2.88 -> -4.97, "
            "while gold-row separation moved correctly both times (+0.105, "
            "+0.40). Teacher-forced supervision makes the model a sharper "
            "discriminator of gold WRITE vs gold WAIT contexts, and at inference "
            "its own error-containing ASR history classifies as WAIT, so "
            "sharpening amplifies the wrong side. This decision needs either a "
            "true roll-in form -- supervised on the model's own generated ASR/MT "
            "text, which the current task pool does not contain, since "
            "boundary_rollin_rate substitutes semantic tokens only -- or it stays "
            "an inference-side calibrated bias. Note the bias itself is damaged "
            "by such a run: delta_cont=4 gives WRITE_MT 0.947 on iter_0002264 but "
            "only 0.347 here, because +4 no longer crosses -4.97."
        ),
    },
    {
        "key": "natural_eos",
        "label": "natural_eos",
        "baseline": 0.50,
        "target": 0.875,
        "direction": "at_least",
        "bias_reference": 1.00,
        "cause": "sessions still do not terminate on their own",
        "action": (
            "natural_eos tracked WRITE_MT exactly in the sweep (0.50 at delta 0-2, "
            "1.00 at delta 3+), so if WRITE_MT passed and this did not, the cause "
            "is not the continue decision but EOS itself; look at boundary_eos."
        ),
    },
    {
        "key": "semantic_coverage",
        "label": "semantic coverage",
        "baseline": 0.666,
        "target": 0.666,
        "direction": "at_least",
        "bias_reference": 0.997,
        "cause": "speaking more did not produce more of the target speech",
        "action": (
            "The sweep reached 0.997 purely by speaking more, so a regression here "
            "with WRITE_MT up means the semantic side degraded: compare "
            "loss/semantic_ce against the parent run."
        ),
    },
    {
        "key": "text_length_ratio_median",
        "label": "text length ratio (median)",
        "baseline": 1.03,
        "target": (0.9, 1.2),
        "direction": "within",
        "bias_reference": 2.25,
        "cause": "content_end_margin did not teach the model to close a fragment",
        "action": (
            "Check the direction first. If the ratio is ABOVE the band the term "
            "did not take: content_end_margin moved END_CONTENT from 0.21 logits "
            "behind its strongest competitor to 1.81 ahead in one epoch, so a "
            "failure to move at all points at the weight. If the ratio is BELOW "
            "the band the term took and is now harmful on its own: it shortens "
            "every fragment, and with WRITE_MT still near 0.147 that means fewer "
            "fragments times shorter each -- measured at 1.033 -> 0.324 with "
            "semantic coverage 0.666 -> 0.448. The over-generation this term "
            "targets was measured at delta_cont 3-4, where the model writes on "
            "~90% of events; it is only correct jointly with a working speak "
            "mechanism. Never train it alone."
        ),
    },
    {
        "key": "text_length_ratio_max",
        "label": "text length ratio (worst case)",
        "baseline": 1.93,
        "target": 2.0,
        "direction": "at_most",
        "bias_reference": 4.07,
        "cause": "repetition loops survived",
        "action": (
            "repetition_penalty at 0.1 already cut the worst case 44.45 -> 4.07. If "
            "it is still above 2 at weight 0.3, widen repetition_window past 8: the "
            "observed loops ('in a state of mind and that they were') are longer "
            "than eight tokens."
        ),
    },
    {
        "key": "asr_error_rate",
        "label": "ASR error rate",
        "baseline": 0.2326,
        "target": 0.26,
        "direction": "at_most",
        "bias_reference": None,
        "cause": "the new terms damaged recognition",
        "action": (
            "The speak-decision run cost 0.0123 here. If this run costs more, the "
            "continue term is pulling probability mass out of the ASR fragments; "
            "lower its weight before anything else."
        ),
    },
)


def _load_events(run_root: str) -> dict[str, Any]:
    counts: collections.Counter = collections.Counter()
    events = 0
    coverage: list[float] = []
    natural: list[bool] = []
    malformed = 0
    ratios: list[float] = []
    errors: list[float] = []
    for path in sorted(glob.glob(os.path.join(run_root, "workers", "*.json"))):
        with open(path, encoding="utf-8") as handle:
            report = json.load(handle)
        for sample in report.get("samples", []):
            asr = sample.get("e_asr") or {}
            if "error_rate" in asr:
                errors.append(float(asr["error_rate"]))
            free = sample.get("e_s2s_free")
            if not free:
                continue
            coverage.append(float(free.get("semantic_coverage", 0.0)))
            natural.append(bool(free.get("natural_eos")))
            malformed += int(free.get("malformed_segments", 0))
            reference = sample.get("translation_reference") or ""
            hypothesis = free.get("target_hypothesis") or ""
            if reference:
                ratios.append(len(hypothesis) / max(1, len(reference)))
            for event in free.get("events", []):
                events += 1
                for action in event.get("chosen_continuations", []):
                    counts[action] += 1
    denominator = max(1, events)
    return {
        "events": events,
        "s2s_samples": len(coverage),
        "write_mt_per_event": counts["WRITE_MT"] / denominator,
        "write_asr_per_event": counts["WRITE_ASR"] / denominator,
        "natural_eos": (sum(natural) / len(natural)) if natural else 0.0,
        "semantic_coverage": statistics.fmean(coverage) if coverage else 0.0,
        "malformed_segments": malformed,
        "text_length_ratio_median": statistics.median(ratios) if ratios else 0.0,
        "text_length_ratio_max": max(ratios) if ratios else 0.0,
        "asr_error_rate": statistics.fmean(errors) if errors else float("nan"),
    }


def _load_gate(run_root: str) -> dict[str, Any]:
    """Translation quality and per-direction ASR live in the gate JSON.

    These come from `incremental_mt_rollout`, which loops over source prefixes
    and never calls `_choice`, so they measure incremental MT capability
    independently of the speak decision.  Omitting them made this suite call a
    checkpoint with 3.5x the eng->cmn free-source BLEU a failure.
    """

    path = _os.path.join(run_root, "E2E_FREE_RUNNING_GATE.json")
    if not _os.path.isfile(path):
        return {}
    with open(path, encoding="utf-8") as handle:
        gate = json.load(handle)
    metrics = gate.get("metrics", {})
    out: dict[str, Any] = {}
    for source in ("gold_source", "free_running_source"):
        directions = metrics.get("e_mt", {}).get(source, {}).get("directions", {})
        for direction, values in directions.items():
            key = direction.replace("->", "_to_")
            out[f"bleu.{source}.{key}"] = values.get("candidate_bleu")
            out[f"chrf.{source}.{key}"] = values.get("candidate_chrf")
            out[f"phase3_bleu.{key}"] = values.get("phase3_bleu")
    for language in ("cmn", "eng"):
        entry = metrics.get("e_asr", {}).get(language, {})
        if "error_rate" in entry:
            out[f"asr_error.{language}"] = entry["error_rate"]
    return out


def _passes(value: float, prediction: dict[str, Any]) -> bool:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return False
    target = prediction["target"]
    if prediction["direction"] == "at_least":
        return value >= target
    if prediction["direction"] == "at_most":
        return value <= target
    low, high = target
    return low <= value <= high


def evaluate(run_root: str, comparisons: dict[str, str]) -> dict[str, Any]:
    observed = _load_events(run_root)
    observed.update(_load_gate(run_root))
    others = {}
    for label, path in comparisons.items():
        entry = _load_events(path)
        entry.update(_load_gate(path))
        others[label] = entry
    checks = []
    for prediction in PREDICTIONS:
        value = observed.get(prediction["key"])
        ok = _passes(value, prediction)
        checks.append(
            {
                "key": prediction["key"],
                "label": prediction["label"],
                "observed": value,
                "baseline": prediction["baseline"],
                "target": prediction["target"],
                "direction": prediction["direction"],
                "bias_sweep_reference": prediction["bias_reference"],
                "passed": ok,
                **({} if ok else {"cause": prediction["cause"], "action": prediction["action"]}),
            }
        )
    failures = [check for check in checks if not check["passed"]]
    # Capability deltas are reported, never gated: a run can lose the speak
    # decision and still be the best model in the lineage on translation and
    # recognition, which is exactly what happened on 2026-09-01.
    capability = []
    for key in sorted(k for k in observed if k.startswith(("bleu.", "chrf.", "asr_error."))):
        row = {"metric": key, "observed": observed[key]}
        for label, other in others.items():
            if other.get(key) is not None and observed[key] is not None:
                row[label] = other[key]
                row[f"delta_vs_{label}"] = observed[key] - other[key]
        capability.append(row)
    return {
        "capability": capability,
        "schema_version": "uniss_e2e_continue_end_verdict_v1",
        "claim_scope": "frozen_fixed16_selection_train_seen",
        "run_root": run_root,
        "status": "passed" if not failures else "failed",
        "checks_passed": len(checks) - len(failures),
        "checks_total": len(checks),
        "observed": observed,
        "comparisons": others,
        "checks": checks,
        "primary_cause": failures[0]["cause"] if failures else None,
        "next_action": failures[0]["action"] if failures else (
            "Every prediction landed on 8 train-seen samples. Do not declare "
            "success: the plan's own risk note says min-style checks on 8 samples "
            "are not trustworthy. Widen the selection to 64 and re-gate before "
            "anything else."
        ),
    }


def render(verdict: dict[str, Any]) -> str:
    lines = [
        f"status: {verdict['status']}  ({verdict['checks_passed']}/{verdict['checks_total']} 项通过)",
        "  注意:通过项数只覆盖开口决策与长度;翻译质量与分方向 ASR 见下方能力对照,",
        "  它们来自不经过开口决策的 incremental_mt_rollout,不参与判定。",
        "",
        f"{'判据':<30s}{'基线':>10s}{'本次':>10s}{'门线':>16s}{'偏置参考':>10s}  ",
    ]
    for check in verdict["checks"]:
        target = check["target"]
        if check["direction"] == "within":
            shown = f"[{target[0]}, {target[1]}]"
        else:
            shown = (">= " if check["direction"] == "at_least" else "<= ") + f"{target}"
        reference = check["bias_sweep_reference"]
        lines.append(
            f"{check['label']:<30s}{check['baseline']:>10.3f}"
            f"{(check['observed'] or 0):>10.3f}{shown:>16s}"
            f"{(f'{reference:.3f}' if reference is not None else '-'):>10s}"
            f"  {'PASS' if check['passed'] else 'FAIL'}"
        )
    if verdict.get("capability"):
        lines += ["", f"{'能力指标(仅报告,不判定)':<44s}{'本次':>10s}" +
                  "".join(f"{k:>22s}" for k in
                          sorted({kk[len('delta_vs_'):] for row in verdict['capability']
                                  for kk in row if kk.startswith('delta_vs_')}))]
        labels = sorted({kk[len("delta_vs_"):] for row in verdict["capability"]
                         for kk in row if kk.startswith("delta_vs_")})
        for row in verdict["capability"]:
            line = f"{row['metric']:<44s}{(row['observed'] or 0):>10.3f}"
            for label in labels:
                delta = row.get(f"delta_vs_{label}")
                line += (f"{delta:>+22.3f}" if delta is not None else f"{'-':>22s}")
            lines.append(line)
    if verdict["primary_cause"]:
        lines += ["", f"最可能的原因: {verdict['primary_cause']}", "", f"下一步: {verdict['next_action']}"]
    else:
        lines += ["", f"下一步: {verdict['next_action']}"]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, help="the gate run root to score")
    parser.add_argument(
        "--compare", action="append", default=[], metavar="LABEL=RUN_ROOT"
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    comparisons = {}
    for item in args.compare:
        label, _, path = item.partition("=")
        if not path:
            raise SystemExit(f"--compare expects LABEL=RUN_ROOT, got {item!r}")
        comparisons[label] = path
    verdict = evaluate(args.run, comparisons)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(verdict, handle, indent=1, sort_keys=True)
    print(render(verdict))


if __name__ == "__main__":
    main()

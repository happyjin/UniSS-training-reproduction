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
            "The requirement is ~3 logits and the previous run moved 6.6 with the "
            "same weight, so first check loss/continue_after_fragment in the log: "
            "if it fell below ~0.1 the margin was satisfied on gold rows and the "
            "gap is exposure bias, which needs a roll-in form of this term. If it "
            "stayed near 0.76, raise the weight or the margin. Meanwhile the "
            "calibrated inference bias delta_cont=3 is available as a fallback."
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
            "This is the term that has never existed before, so a miss here is the "
            "most informative outcome. Check loss/content_end_margin on interleaved "
            "batches: it started at 2.311, meaning END_CONTENT trailed by 0.21 "
            "logits. If it did not fall, raise content_end_margin weight; if it "
            "fell but the ratio did not, the over-generation is re-translation of "
            "already-committed text, which is a commit-policy problem, not an "
            "END problem."
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
    others = {label: _load_events(path) for label, path in comparisons.items()}
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
    return {
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

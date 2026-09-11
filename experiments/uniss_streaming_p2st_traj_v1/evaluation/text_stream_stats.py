"""Does the preference data actually carry a flow signal?

DPO can only teach what the pairs disagree about.  The pairs here are selected
on silence measured in the *audio*, but the objective acts on the *text
stream*, so there is a gap the selection never checks: if the chosen and
rejected candidates commit text the same way, the audio difference came from
somewhere the objective cannot reach, and training would be a no-op at best.

This measures the gap directly, over the same pairs the trainer will see:

* ``steps``        -- MT read steps that ran;
* ``produced``     -- tokens the model emitted across the stream;
* ``per_step``     -- produced tokens per step, the density that fills a gap;
* ``first_commit`` -- index of the first step that committed anything, which
                      is what the opening silence is made of;
* ``empty_share``  -- share of steps that committed nothing.

A chosen side that is not denser and not earlier means the pairs are not about
flow, whatever the audio says.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from experiments.uniss_streaming_p2st_traj_v1.training.dpo_text import (
    build_examples,
    load_manifests,
)


def stream_stats(steps: list[dict]) -> dict:
    produced = sum(len(s["produced"]) for s in steps)
    committed = [i for i, s in enumerate(steps) if s.get("committed", 0) > 0]
    return {
        "steps": len(steps),
        "produced": produced,
        "per_step": produced / max(1, len(steps)),
        "first_commit": committed[0] if committed else len(steps),
        "empty_share": 1.0 - len(committed) / max(1, len(steps)),
    }


def compare(examples: list[dict]) -> dict:
    keys = ("steps", "produced", "per_step", "first_commit", "empty_share")
    rows = {"chosen": {k: [] for k in keys}, "rejected": {k: [] for k in keys}}
    wins = {k: 0 for k in keys}
    for example in examples:
        stats = {side: stream_stats(example[side]) for side in ("chosen", "rejected")}
        for side in ("chosen", "rejected"):
            for key in keys:
                rows[side][key].append(stats[side][key])
        for key in keys:
            wins[key] += int(stats["chosen"][key] > stats["rejected"][key])
    return {
        "n": len(examples),
        "chosen": {k: statistics.mean(v) for k, v in rows["chosen"].items()},
        "rejected": {k: statistics.mean(v) for k, v in rows["rejected"].items()},
        "chosen_higher_share": {k: v / max(1, len(examples)) for k, v in wins.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--arm-root", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()

    pairs = json.loads(Path(args.pairs).read_text(encoding="utf-8"))["pairs"]
    examples = build_examples(pairs, load_manifests(Path(args.arm_root)))
    if not examples:
        raise SystemExit("no pair had a recorded text stream on both sides")
    report = compare(examples)

    print(f"{report['n']} pairs with a text stream on both sides")
    print(f"{'metric':>13} {'chosen':>9} {'rejected':>9} {'chosen higher':>14}")
    for key in ("steps", "produced", "per_step", "first_commit", "empty_share"):
        print(
            f"{key:>13} {report['chosen'][key]:>9.3f} "
            f"{report['rejected'][key]:>9.3f} "
            f"{report['chosen_higher_share'][key]:>13.1%}"
        )
    if args.output:
        Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"-> {args.output}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Fit p(semantic codes | fragment text length, target language).

Why this exists
---------------
The gaps a listener hears in the rendered timeline are not a scheduling
problem.  A constant playback delay leaves them untouched, because the hole
between fragment k-1 and k is

    gap_k = source_end_ms_k - source_end_ms_{k-1} - length(fragment k-1)

and a delay D added to both arrival times cancels.  Measured over the eight
demo samples, internal silence tracks ``1 - speech_duration/source_duration``
with r = 0.664, and on the aggregate it is nearly exact: this run's long audio
speaks 0.539x the source and is 50.1% silent, while the m3 run speaks 0.838x
and is 19.6% silent.  The holes are under-generation.

So the quantity to control is how many codes a fragment gets for the text it
carries, and that has a strong empirical prior: the BiCodec semantic rate is
exactly 50 tokens per second, so codes are duration, and duration per
character of text is a well-behaved quantity.  This fits it, per target
language and per text length, as a survival function so the decoder can turn
it into a hazard rate.

The text length is measured on ``_split_target_text(event)[1]`` -- the exact
string the TTS prompt carries -- because a prior fitted on a different
tokenisation of the same fragment would not be the prior the decoder needs.
"""
from __future__ import annotations

import argparse
import bisect
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

from experiments.uniss_streaming_p2st_pure_ce_v1.training.task_samples_p2st import (
    _split_target_text,
)
from experiments.uniss_phase3_v4_e2e_simuls2st_pilot15_v1.data.schema import (
    E2ETrajectory,
)

# Lengths above this are pooled into one bucket; the tail is thin and the
# conditional spread has already stopped narrowing by then.
MAX_LENGTH = 24
# Neighbour pooling radius when a bucket has too few observations to give a
# usable survival curve.
MIN_COUNT = 200
POOL_RADIUS = 3


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", required=True)
    parser.add_argument("--limit", type=int, default=0)
    # The gold file is ordered by corpus, so its head is entirely
    # English-source (target cmn) and the Chinese-source rows that give the
    # eng prior appear much later.  A stride samples both.
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    raw: dict[tuple[str, int], list[int]] = defaultdict(list)
    trajectories = 0
    seen = 0
    with open(args.gold) as handle:
        for line in handle:
            if '"sample_id"' not in line:
                continue
            seen += 1
            if args.stride > 1 and seen % args.stride:
                continue
            trajectory = E2ETrajectory.from_mapping(json.loads(line))
            trajectories += 1
            for event in trajectory.events:
                codes = len(event.target_semantic_delta or ())
                if codes <= 0:
                    continue
                text = _split_target_text(event)[1]
                if not text.strip():
                    continue
                length = min(MAX_LENGTH, len(text))
                raw[(trajectory.tgt_lang, length)].append(codes)
            if args.limit and trajectories >= args.limit:
                break

    lengths = sorted({key[1] for key in raw})
    languages = sorted({key[0] for key in raw})
    prior: dict[str, dict] = {}
    for language in languages:
        per_length: dict[str, dict] = {}
        for length in lengths:
            pooled = list(raw.get((language, length), ()))
            radius = 0
            while len(pooled) < MIN_COUNT and radius < POOL_RADIUS:
                radius += 1
                for offset in (-radius, radius):
                    pooled.extend(raw.get((language, length + offset), ()))
            if len(pooled) < 20:
                continue
            pooled.sort()
            total = len(pooled)
            # Survival at n is the share of observations with codes >= n, which
            # is what the hazard needs; store it sparsely as the sorted sample
            # so the runtime can interpolate without carrying a full histogram.
            quantiles = {
                str(q): pooled[min(total - 1, int(q * total / 100))]
                for q in (1, 5, 10, 25, 50, 75, 90, 95, 99)
            }
            per_length[str(length)] = {
                "count": total,
                "pooled_radius": radius,
                "mean": st.fmean(pooled),
                "sd": st.pstdev(pooled) if total > 1 else 0.0,
                "quantiles": quantiles,
                "support": pooled[:: max(1, total // 256)],
            }
        prior[language] = per_length

    report = {
        "schema_version": "uniss_p2st_length_prior_v1",
        "gold": str(Path(args.gold).resolve()),
        "trajectories": trajectories,
        "max_length": MAX_LENGTH,
        "min_count": MIN_COUNT,
        "pool_radius": POOL_RADIUS,
        "languages": prior,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")
    for language, per_length in prior.items():
        print(f"=== {language} ===")
        print(
            "%6s %8s %8s %8s %8s %8s %8s"
            % ("文本长", "n", "码中位", "码/字", "p5", "p95", "sd/中位")
        )
        for key in sorted(per_length, key=int):
            row = per_length[key]
            median = float(row["quantiles"]["50"])
            print(
                "%6s %8d %8.0f %8.2f %8s %8s %8.2f"
                % (
                    key,
                    row["count"],
                    median,
                    median / int(key),
                    row["quantiles"]["5"],
                    row["quantiles"]["95"],
                    row["sd"] / max(median, 1e-9),
                )
            )
    print("wrote", args.output)


if __name__ == "__main__":
    main()

"""Where does the silence in the placed timeline come from?

The chopped-audio diagnosis established *that* 24-53% of the placed timeline
is silence.  It did not establish *why*, and the two candidate causes call for
opposite fixes:

* **the model speaks too little** -- it emits fewer semantic codes than the
  source has speech, so no placement policy can fill the timeline.  The fix is
  in the model or in the length prior.
* **the model speaks late** -- it emits enough audio but the placement pushes
  each fragment to the read boundary that justified it, spreading the same
  speech over a longer timeline.  The fix is in the pacing.

They are separable from the rollout manifest alone, with no GPU, because
``p2st_cascade`` places a fragment at ``max(source_end_ms, previous_end)``:

* ``speech_ms / source_ms`` answers the first.  Compare against the gold
  ceiling arm, which decodes the *reference* codes through the same BiCodec
  and the same placement, and so carries whatever ratio a correct system has.
* ``capped_stages`` counts TTS stages that ran to the pacing budget instead of
  stopping on ``END_SEMANTIC``.  A high count means the budget, not the model,
  decided where the fragment ended -- which is truncation caused by pacing.
* every internal gap is a wait: the fragment could not start before the audio
  that justified it.  ``wait_ms`` totals it and ``waits_per_utterance``
  counts it, so the two causes can be weighed rather than argued.

Everything here is arithmetic on the manifest, so it runs on CPU beside a
training job.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def analyse_sample(sample: dict) -> dict | None:
    """Timeline accounting for one utterance."""
    durations = [float(v) for v in sample.get("durations", [])]
    delays = [float(v) for v in sample.get("delays", [])]
    if not durations:
        return None
    intervals = [list(map(float, iv)) for iv in sample.get("intervals", [])]
    if not intervals:
        # Older arms record only the read boundary.  Rebuild the placement with
        # the runtime's own rule -- p2st_cascade starts a fragment at
        # ``max(source_end_ms, previous_end)`` -- rather than skipping the arm,
        # so the gold ceiling can serve as the reference it exists to be.
        intervals = []
        previous_end = 0.0
        for delay, duration in zip(delays, durations):
            start = max(float(delay), previous_end)
            intervals.append([start, float(duration)])
            previous_end = start + float(duration)
    if len(intervals) != len(durations):
        return None
    source_ms = float(sample.get("source_duration_ms") or 0.0)
    speech_ms = sum(durations)
    lead_ms = intervals[0][0]
    timeline_ms = intervals[-1][0] + durations[-1]
    gaps: list[float] = []
    previous_end = intervals[0][0] + durations[0]
    for (start, _), duration in zip(intervals[1:], durations[1:]):
        gap = start - previous_end
        if gap > 0:
            gaps.append(gap)
        previous_end = start + duration
    fragments = len(durations)
    capped = int(sample.get("capped_stages") or 0)
    # Codes spoken per character of the text actually committed.  This is what
    # separates "the translation is short" from "each fragment is cut off":
    # the gold ceiling, speaking the reference through the same BiCodec,
    # measures 8.44, so a lower value means the TTS stopped early regardless
    # of how much text the MT stage produced.
    spoken_text = (
        sample.get("translation_reference")
        if int(sample.get("fragments") or 0) == 1 and not sample.get("intervals")
        else sample.get("target_hypothesis")
    ) or ""
    tokens = int(sample.get("semantic_tokens") or 0)
    return {
        "sample_id": sample.get("sample_id"),
        "direction": sample.get("direction"),
        "source_ms": source_ms,
        "speech_ms": speech_ms,
        "timeline_ms": timeline_ms,
        "lead_ms": lead_ms,
        "wait_ms": sum(gaps),
        "waits": len(gaps),
        "fragments": fragments,
        # How much speech the model produced per second of source.  The gold
        # ceiling arm gives the value a correct system has on this data.
        "speech_over_source": speech_ms / source_ms if source_ms else None,
        # How much of the timeline after the first word is silence.
        "internal_silence": (
            sum(gaps) / (timeline_ms - lead_ms) if timeline_ms > lead_ms else None
        ),
        # Fraction of fragments whose end was chosen by the pacing budget
        # rather than by the model emitting END_SEMANTIC.
        "capped_fraction": capped / fragments if fragments else None,
        "terminator_rate": sample.get("terminator_rate"),
        "text_chars": len(spoken_text),
        "codes_per_char": tokens / len(spoken_text) if spoken_text else None,
        "text_over_reference": (
            len(spoken_text) / len(sample["translation_reference"])
            if sample.get("translation_reference")
            else None
        ),
        "mean_read_gap_ms": (
            statistics.mean(
                b - a for a, b in zip(delays, delays[1:])
            )
            if len(delays) > 1
            else None
        ),
    }


def summarise(rows: list[dict]) -> dict:
    def mean(key: str) -> float | None:
        values = [float(r[key]) for r in rows if r.get(key) is not None]
        return statistics.mean(values) if values else None

    return {
        "samples": len(rows),
        "speech_over_source": mean("speech_over_source"),
        "codes_per_char": mean("codes_per_char"),
        "text_over_reference": mean("text_over_reference"),
        "internal_silence": mean("internal_silence"),
        "capped_fraction": mean("capped_fraction"),
        "terminator_rate": mean("terminator_rate"),
        "lead_ms": mean("lead_ms"),
        "wait_ms": mean("wait_ms"),
        "waits_per_utterance": mean("waits"),
        "fragments": mean("fragments"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", action="append", required=True)
    parser.add_argument("--label", action="append", default=[])
    parser.add_argument("--output")
    args = parser.parse_args()

    report: dict[str, object] = {
        "schema_version": "uniss_streaming_p2st_silence_budget_v1",
        "arms": {},
    }
    print(
        f"{'arm':<24s}{'speech/src':>11s}{'codes/char':>11s}{'txt/ref':>9s}"
        f"{'int_sil':>9s}{'capped':>8s}{'term':>7s}{'lead_ms':>9s}{'frags':>7s}"
    )
    for index, path in enumerate(args.manifest):
        label = args.label[index] if index < len(args.label) else Path(path).parent.name
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        rows = [r for r in (analyse_sample(s) for s in data["samples"]) if r]
        summary = summarise(rows)
        report["arms"][label] = {"summary": summary, "rows": rows}  # type: ignore[index]
        f = lambda v, w, d: (f"{v:>{w}.{d}f}" if isinstance(v, (int, float)) else f"{'-':>{w}s}")
        print(
            f"{label:<24s}{f(summary['speech_over_source'],11,3)}"
            f"{f(summary['codes_per_char'],11,2)}{f(summary['text_over_reference'],9,2)}"
            f"{f(summary['internal_silence'],9,3)}{f(summary['capped_fraction'],8,3)}"
            f"{f(summary['terminator_rate'],7,2)}{f(summary['lead_ms'],9,0)}"
            f"{f(summary['fragments'],7,1)}"
        )
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(
            json.dumps(report, ensure_ascii=False, indent=1, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"-> {args.output}")


if __name__ == "__main__":
    main()

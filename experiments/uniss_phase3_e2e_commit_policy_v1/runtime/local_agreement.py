#!/usr/bin/env python3
"""Local-agreement commit policy for the E2E incremental MT measurement.

The established evaluator commits with
``evaluation/runtime.py::append_only_commit``, which accepts a candidate only
when it extends the already committed prefix and otherwise discards it whole.
The very first event is therefore committed with no stability evidence at all,
so one early mistake freezes the hypothesis for the rest of the utterance.

Measured on the frozen fixed-16 selection at ``endmargin_epoch23 iter_0002264``:
260 of 316 events (82.3%) conflict, and ``emilia_zh_0005985930`` commits
``"That's"`` for the whole utterance even though the model's own longest
hypothesis is ``"Such a person feels that everything is possible and then
everything in the future is full of hope"`` against the reference ``"Such a
self one who feels that anything is possible and that the future is full of
hope"``.  The translation capability is there; the commit layer discards it.

The fix reuses the audited local-agreement committer the Phase-A streaming
cascade already runs, ``uniss_phasea_stateful_longepisode_rl_v1/runtime/
commit.py::StablePrefixCommitter``: only the prefix two consecutive hypotheses
agree on is committed, and the final event flushes what remains.  That
committer operates on integer sequences, so normalized text units are interned
to integers, which keeps the established policy as the single source of truth
instead of reimplementing it for strings.  Display casing is preserved by
slicing the un-lowercased units of the same candidate.

Nothing in the E2E experiment is modified.  This module is imported by
``evaluation/run_worker_local_agreement.py``, which rebinds the symbol on the
established worker and then delegates to it.
"""

from __future__ import annotations

from typing import Sequence

from experiments.uniss_phase3_v4_e2e_simuls2st_pilot15_v1.data.schema import (
    append_text,
)
from experiments.uniss_phase3_v4_e2e_simuls2st_pilot15_v1.evaluation.gate import (
    text_units,
)
from experiments.uniss_phase3_v4_e2e_simuls2st_pilot15_v1.evaluation.runtime import (
    generate_mt_prefix,
)
from experiments.uniss_phasea_stateful_longepisode_rl_v1.runtime.commit import (
    StablePrefixCommitter,
)


# One unit of holdback is required, not optional.  With holdback 0 the policy
# commits the longest agreed prefix immediately, which on the observed
# emilia_zh_0005985930 sequence commits "Such a feeling" at event 5; event 6
# genuinely revises to "Such a person who thinks ..." and contradicts it, so
# the hypothesis freezes again -- later than the established policy, but just as
# permanently.  Holding one unit back commits only "Such a", which every later
# revision still extends, and the utterance completes.  The model really does
# revise its own early words, so an append-only policy must leave that margin.
DEFAULT_HOLDBACK = 1


def display_units(value: str, language: str) -> list[str]:
    """``text_units`` without the lowercasing, so output casing survives.

    The split is unit-for-unit identical to ``text_units``, which is what makes
    slicing by committed length valid.
    """

    normalized = " ".join(str(value).strip().split())
    if str(language) == "cmn":
        return list("".join(normalized.split()))
    return normalized.split()


class UnitInterner:
    """Map text units to stable integers so the audited committer is reusable."""

    def __init__(self) -> None:
        self._to_id: dict[str, int] = {}

    def encode(self, units: Sequence[str]) -> list[int]:
        output: list[int] = []
        for unit in units:
            value = self._to_id.get(unit)
            if value is None:
                value = len(self._to_id)
                self._to_id[unit] = value
            output.append(value)
        return output


def local_agreement_mt_rollout(
    model,
    tokenizer,
    source_prefixes: Sequence[str],
    target_lang: str,
    *,
    max_tokens: int,
    holdback: int = DEFAULT_HOLDBACK,
) -> dict[str, object]:
    """Drop-in replacement for ``incremental_mt_rollout`` with LA-2 commits.

    The return shape is identical, so ``run_worker._mt_value`` and the gate
    consume it unchanged.  ``commit_conflicts`` keeps its original meaning: a
    candidate that contradicts an already committed prefix.  Under local
    agreement that should become rare, because nothing is committed until two
    consecutive hypotheses agree on it.
    """

    interner = UnitInterner()
    committer = StablePrefixCommitter(holdback=int(holdback))
    committed_display: list[str] = []
    hypotheses: list[str] = []
    raw_hypotheses: list[str] = []
    unterminated = 0
    total = len(source_prefixes)

    def commit(raw: str, *, final: bool) -> None:
        units = display_units(raw, target_lang)
        added = committer.update(interner.encode(text_units(raw, target_lang)), final=final)
        if added:
            start = len(committed_display)
            committed_display.extend(units[start : start + len(added)])

    for index, source_prefix in enumerate(source_prefixes):
        final = index == total - 1
        if not source_prefix.strip():
            # No new source content, so nothing new may be committed.  The last
            # event must still flush whatever agreement is pending.
            if final and raw_hypotheses:
                last = next(
                    (value for value in reversed(raw_hypotheses) if value), ""
                )
                if last:
                    commit(last, final=True)
            hypotheses.append(_join(committed_display, target_lang))
            raw_hypotheses.append("")
            continue
        raw, _, reached_end = generate_mt_prefix(
            model,
            tokenizer,
            source_prefix,
            target_lang,
            max_tokens=max_tokens,
        )
        commit(raw, final=final)
        unterminated += int(not reached_end)
        hypotheses.append(_join(committed_display, target_lang))
        raw_hypotheses.append(raw)
    return {
        "hypotheses": hypotheses,
        "raw_hypotheses": raw_hypotheses,
        "commit_conflicts": int(committer.revision_conflicts),
        "unterminated_generations": int(unterminated),
    }


def _join(units: Sequence[str], language: str) -> str:
    text = ""
    for unit in units:
        text = append_text(text, unit, language)
    return text


__all__ = [
    "DEFAULT_HOLDBACK",
    "UnitInterner",
    "display_units",
    "local_agreement_mt_rollout",
]

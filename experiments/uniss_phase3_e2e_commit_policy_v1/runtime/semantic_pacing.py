#!/usr/bin/env python3
"""Bound the speech stream by the source time it has actually consumed.

A simultaneous system cannot emit target audio faster than it receives source
audio: if it does, the output can never be played back in step with the input,
no matter how fast the hardware is.  The E2E interleaved session has no such
constraint.  It caps each event's semantic fragment at a flat
``--max-s2s-semantic-tokens`` (384 in every gate run) and appends whatever comes
out, so on the fixed-16 selection the cmn->eng output audio runs 2.04x the
source and one sample reaches 2.44x.

The BiCodec semantic rate is exactly 50 tokens per second -- 20 ms per token --
verified on all seven audio samples of the local-agreement holdback-2 run, where
``semantic_tokens / audio_duration_seconds`` is 50.0 for every one of them.  The
budget is therefore directly computable:

    allowed_total = (consumed_source_ms + margin) / 20 ms
    event_budget  = allowed_total - already_emitted

Two details matter.  The budget is *cumulative*, not per event, so the model may
still burst to catch up after a quiet stretch while the total stays bounded.
And ``_restricted_semantic_choice`` forbids END until at least one token exists,
so the smallest well-formed fragment is two tokens; a budget of zero would
guarantee ``malformed_segments`` instead of silence.  The floor is therefore two
tokens, which on a seventeen-event episode can overshoot by at most 34 tokens
(0.68 s) -- an 11% bound where the current behaviour is 126% over.

This bounds the damage rather than curing it.  The real fix is the model
learning to stop: ``natural_eos`` is 0.50 at iteration 1132, 1207 and 2264
alike, so the current ``semantic_end_ce`` / ``semantic_end_margin`` weights do
nothing for termination.  Truncation also costs content coverage and can cut a
phrase mid-word, which the gate will show.
"""

from __future__ import annotations

from experiments.uniss_phase3_v4_e2e_simuls2st_pilot15_v1.evaluation.runtime import (
    PersistentInterleavedSession,
)


# 50 semantic tokens per second, measured as exactly 50.0 on every sample.
SEMANTIC_TOKEN_MS = 20.0
# Smallest fragment that can carry one content token plus END.
MINIMUM_FRAGMENT_TOKENS = 2
DEFAULT_MARGIN_MS = 0.0
# After the source ends there is no more input to keep pace with, so a tail is
# legitimate; without one the last phrase is always cut.
DEFAULT_TAIL_MS = 2000.0


def allowed_event_tokens(
    *,
    consumed_source_ms: float,
    already_emitted: int,
    source_final: bool,
    margin_ms: float = DEFAULT_MARGIN_MS,
    tail_ms: float = DEFAULT_TAIL_MS,
    minimum_fragment_tokens: int = MINIMUM_FRAGMENT_TOKENS,
    token_ms: float = SEMANTIC_TOKEN_MS,
) -> int:
    """Semantic tokens this event may emit without outpacing the input."""

    if token_ms <= 0:
        raise ValueError("semantic token duration must be positive")
    allowance = float(consumed_source_ms) + float(margin_ms)
    if source_final:
        allowance += float(tail_ms)
    total = int(allowance // float(token_ms))
    remaining = total - int(already_emitted)
    return max(int(minimum_fragment_tokens), remaining)


class PacedInterleavedSession(PersistentInterleavedSession):
    """The established session with a source-paced speech budget.

    Only ``max_semantic_tokens`` is recomputed; the event grammar, family
    ordering, EOS legality and every other behaviour is inherited unchanged, so
    the run stays comparable with the established evaluator.
    """

    def __init__(
        self,
        *args,
        pace_margin_ms: float = DEFAULT_MARGIN_MS,
        pace_tail_ms: float = DEFAULT_TAIL_MS,
        minimum_fragment_tokens: int = MINIMUM_FRAGMENT_TOKENS,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.pace_margin_ms = float(pace_margin_ms)
        self.pace_tail_ms = float(pace_tail_ms)
        self.minimum_fragment_tokens = int(minimum_fragment_tokens)
        self.pace_budgets: list[dict[str, float]] = []

    def run_event(
        self,
        event,
        *,
        max_fragments: int,
        max_text_tokens: int,
        max_semantic_tokens: int,
    ):
        budget = allowed_event_tokens(
            consumed_source_ms=float(event.source_end_ms),
            already_emitted=len(self.semantic),
            source_final=bool(event.source_final),
            margin_ms=self.pace_margin_ms,
            tail_ms=self.pace_tail_ms,
            minimum_fragment_tokens=self.minimum_fragment_tokens,
        )
        effective = min(int(max_semantic_tokens), budget)
        self.pace_budgets.append(
            {
                "event_index": float(event.event_index),
                "source_end_ms": float(event.source_end_ms),
                "already_emitted": float(len(self.semantic)),
                "budget": float(budget),
                "effective": float(effective),
            }
        )
        return super().run_event(
            event,
            max_fragments=max_fragments,
            max_text_tokens=max_text_tokens,
            max_semantic_tokens=effective,
        )


__all__ = [
    "DEFAULT_MARGIN_MS",
    "DEFAULT_TAIL_MS",
    "MINIMUM_FRAGMENT_TOKENS",
    "SEMANTIC_TOKEN_MS",
    "PacedInterleavedSession",
    "allowed_event_tokens",
]

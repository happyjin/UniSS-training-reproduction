#!/usr/bin/env python3
"""A hazard-rate bias on END_SEMANTIC from the fitted length prior.

The decoder's stopping decision is greedy: END_SEMANTIC wins when its logit
is the largest.  That gives the model no information about how long a fragment
of this much text ought to be, and the measured consequence is failure in both
directions -- fragments of 1 to 2 codes for text that needs a dozen, and
fragments that run to the cap.  Under-generation is the one that a listener
hears, because internal silence tracks ``1 - speech/source`` almost exactly.

The prior is strong and cheap.  Fitted over the training trajectories, Chinese
fragments take 12 to 13 codes per character with the conditional spread
narrowing from sd/median 0.98 at one character to 0.19 at twenty, so for any
fragment longer than a few characters the number of codes is well determined
before the model generates anything.

Combining it is exact Bayes on a one-dimensional nuisance variable:

    log p(stop at n | codes_{<n}, text) = log p_model(END | .) + log h(n | text)

where the hazard is the chance of stopping at n given that stopping has not
happened yet,

    h(n | l) = p(N = n | l) / P(N >= n | l)

estimated from the stored empirical support.  Adding ``log h`` to the
END_SEMANTIC logit is therefore not a tuned bias but the prior's own
contribution to the posterior over stopping time.

This is deliberately *not* the shape of the ``delta`` bias that four training
runs and an inference sweep failed to calibrate.  That was a single constant
added to one logit for the whole run, and its response was a step function:
delta=1 flipped none of 181 decisions, delta=2 flipped 58 of 218, and 3, 4 and
5 were bit-identical.  This bias changes at every step and depends on the text
the fragment carries, so it has no single threshold to step over.
"""
from __future__ import annotations

import bisect
import json
import math
from pathlib import Path

DEFAULT_PRIOR = Path(__file__).resolve().parent.parent / "data" / "LENGTH_PRIOR.json"
# log 0 is not a usable bias.  1e-6 caps the suppression at -13.8 logits,
# which is decisive against a premature END without being an infinity.
FLOOR = 1e-6


class LengthPrior:
    """``log h(n | text length, language)`` from the fitted support."""

    def __init__(self, payload: dict) -> None:
        self.max_length = int(payload["max_length"])
        self._support: dict[tuple[str, int], list[int]] = {}
        for language, per_length in payload["languages"].items():
            for key, row in per_length.items():
                self._support[(language, int(key))] = sorted(
                    int(value) for value in row["support"]
                )

    @classmethod
    def load(cls, path: str | Path | None = None) -> "LengthPrior":
        target = Path(path) if path else DEFAULT_PRIOR
        return cls(json.loads(target.read_text()))

    def available(self, language: str) -> bool:
        return any(key[0] == language for key in self._support)

    def log_completion(self, generated: int, *, text_length: int, language: str) -> float:
        """Additive bias for the END logit after ``generated`` codes.

        ``log P(N <= n | text length)`` -- the prior probability that a
        fragment carrying this much text is already finished by code ``n``.
        Adding it to the END logit is Bayes on a one-dimensional nuisance
        variable, and it has the right shape for the failure that matters: it
        is very negative while ``n`` is far below any plausible length, so a
        premature END loses, and it rises to zero once ``n`` reaches the range
        the prior supports, so a legitimate END is untouched.

        The bound at zero means this can only *suppress* an early stop, never
        force a late one.  That is deliberate.  Over-generation already has
        two mechanisms -- the source-time pace budget and the repetition
        penalty -- and both are measured to work; under-generation had none,
        and it is what a listener hears as silence.

        An earlier draft used the hazard ``p(N = n)/P(N >= n)``, which is the
        textbook form, but the stored support is a subsample so the density is
        full of empty bins and the hazard was noisy enough to suppress END in
        the middle of the plausible range.  The CDF has no such holes.

        Returns 0.0 when the prior has nothing to say, so an unfitted language
        or an empty fragment leaves the decoder exactly as it was.
        """
        length = max(1, min(self.max_length, int(text_length)))
        support = self._support.get((language, length))
        while support is None and length > 1:
            length -= 1
            support = self._support.get((language, length))
        if not support:
            return 0.0
        total = len(support)
        at_or_before = bisect.bisect_right(support, int(generated))
        return math.log(max(at_or_before / total, FLOOR))

def terminator_bias(
    prior: LengthPrior | None,
    *,
    text_length: int,
    language: str,
    scale: float = 1.0,
):
    """A ``_generate`` callback, or ``None`` when the prior does not apply."""
    if prior is None or scale == 0.0 or text_length <= 0:
        return None
    if not prior.available(language):
        return None

    def bias(generated: int) -> float:
        return scale * prior.log_completion(
            generated, text_length=text_length, language=language
        )

    return bias


__all__ = ["LengthPrior", "terminator_bias", "DEFAULT_PRIOR"]

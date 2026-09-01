#!/usr/bin/env python3
"""Graded logit bias on the two decisions that actually gate translation.

The decision-logit probe located them: within an event the first
WRITE_GENERATE/WAIT_READ choice leads by a median 28.58 logits and is never the
constraint, while ``logit[TASK_MT] - logit[TASK_ASR]`` has median -6.75 and the
continue-after-fragment WRITE/WAIT choice has median -2.88.

S0.1 falsified a *hard* content gate: forcing MT on every event drove the
session text length ratio 1.70 -> 15.40 with a repetition loop.  A graded bias
is a different instrument.  Sweeping delta answers the question a hard gate
cannot: is the model able to translate and merely mis-ranked, or is the ranking
correct because the incremental MT is not there yet?

* If some delta reaches the gold 1.95:1 family ratio without repetition
  degradation, the ranking is wrong and a position-split margin loss will fix
  it.
* If every delta that raises the MT share also produces repetition or garbage,
  the ranking is *right* and the wall is incremental MT capability, not the
  decision.

The bias is applied only inside ``_choice``; ``self.logits`` is untouched, so
generation elsewhere is bit-for-bit unchanged.  The recording probe stays
installed so each run yields both behaviour and logits.
"""
from __future__ import annotations

import os

from experiments.uniss_phase3_v4_e2e_simuls2st_pilot15_v1.evaluation import runtime
from experiments.uniss_phase3_e2e_commit_policy_v1.evaluation import (
    run_worker_local_agreement,
)
from experiments.uniss_phase3_e2e_commit_policy_v1.runtime.semantic_pacing import (
    PacedInterleavedSession,
)
from experiments.uniss_phase3_e2e_speak_decision_v1.diagnostics import (
    family_logit_probe as probe,
)
from training import constants_uniss as c

ENV_FAMILY_BIAS = "UNISS_E2E_FAMILY_MT_BIAS"
ENV_CONTINUE_BIAS = "UNISS_E2E_CONTINUE_WRITE_BIAS"
# Solution 2 for the audio defects measured on long form: a floor on how many
# semantic tokens a fragment must emit before END_SEMANTIC becomes legal.
# The established runtime already forbids END on the very first token
# (`allow_end=bool(generated)`), so the floor is 1 today, and the measured token
# count per event runs min 0, max 104, median 5-8 with a standard deviation of
# 17.7.  Fragments of one to four tokens are 20-80 ms, too short to carry a
# phoneme, and they double the energy-jump rate: 41.0 per thousand frames at
# delta=0 against 91.8 at delta=5.
ENV_MIN_SEMANTIC = "UNISS_E2E_MIN_SEMANTIC_FRAGMENT"
# Solution 3, the one the spectra point at.  `_restricted_semantic_choice` is a
# pure greedy argmax over the 8192-code BiCodec semantic range with no
# temperature, no top-k and no repetition penalty, run autoregressively for
# ~900 codes.  The offline phase3 evaluation this project is measured against
# used `repetition_penalty: 1.1` (see stage00 baseline_summary.json), so the
# streaming runtime is strictly greedier than the baseline it is compared to.
# Measured consequence: decoding the model's codes puts 23.4%, 45.6% and 67.9%
# of energy above 4 kHz on three of four long samples, where the same decoder
# and the same speaker tokens put gold codes at 7.1%, 7.9% and 20.7%.
ENV_SEMANTIC_REPETITION = "UNISS_E2E_SEMANTIC_REPETITION_PENALTY"
ENV_SEMANTIC_WINDOW = "UNISS_E2E_SEMANTIC_REPETITION_WINDOW"
ENV_SEMANTIC_TOPK = "UNISS_E2E_SEMANTIC_TOPK"

FAMILY_TOKENS = frozenset(
    {c.TOKEN_TASK_ASR, c.TOKEN_TASK_S2T_TRANSLATION, c.TOKEN_TASK_TTS}
)


def install_minimum_semantic_fragment(minimum_tokens: int) -> None:
    """Forbid END_SEMANTIC until a fragment has emitted `minimum_tokens` codes.

    This is a floor on fragment length, which the pacer's
    `minimum_fragment_tokens` is not -- that one is an upper bound handed to
    `max_semantic_tokens`, so raising it permits more tokens rather than
    requiring them.
    """

    if minimum_tokens <= 1:
        return
    established = runtime.PersistentInterleavedSession._generate_semantic

    def floored_generate_semantic(self, *, max_tokens: int):
        if max_tokens <= 0:
            return (), False
        generated: list[int] = []
        reached_end = False
        for _ in range(max_tokens):
            if self.logits is None:
                raise RuntimeError("missing semantic logits")
            token = runtime._restricted_semantic_choice(
                self.logits, allow_end=len(generated) >= minimum_tokens
            )
            if token == c.TOKEN_END_SEMANTIC:
                self._append((token,), (None,))
                reached_end = True
                break
            generated.append(token - c.BICODEC_SEMANTIC_OFFSET)
            self._append((token,), (None,))
        return tuple(generated), reached_end

    runtime.PersistentInterleavedSession._generate_semantic = floored_generate_semantic


def install_semantic_decoding(
    repetition_penalty: float, window: int, top_k: int, minimum_tokens: int
) -> None:
    """Give semantic decoding the repetition penalty the offline baseline had.

    The penalty divides the logit of any code already emitted inside the current
    window, matching HuggingFace's convention for positive logits, and top_k>1
    replaces the argmax with a sample over the k best codes.  END_SEMANTIC is
    excluded from the penalty so termination is never discouraged.
    """

    if repetition_penalty <= 1.0 and top_k <= 1 and minimum_tokens <= 1:
        return
    import torch

    def decoded_generate_semantic(self, *, max_tokens: int):
        if max_tokens <= 0:
            return (), False
        generated: list[int] = []
        reached_end = False
        start = c.BICODEC_SEMANTIC_OFFSET
        size = c.BICODEC_SEMANTIC_SIZE
        for _ in range(max_tokens):
            if self.logits is None:
                raise RuntimeError("missing semantic logits")
            values = self.logits.reshape(-1).float().clone()
            slice_ = values[start : start + size]
            if repetition_penalty > 1.0 and generated:
                recent = generated[-window:] if window > 0 else generated
                for code in set(recent):
                    if 0 <= code < size:
                        current = slice_[code]
                        slice_[code] = (
                            current / repetition_penalty
                            if current > 0
                            else current * repetition_penalty
                        )
            allow_end = len(generated) >= max(1, minimum_tokens)
            if top_k > 1:
                k = min(int(top_k), size)
                top_values, top_indices = slice_.topk(k)
                probabilities = torch.softmax(top_values, dim=0)
                picked = int(torch.multinomial(probabilities, 1).item())
                best_value = top_values[picked]
                best_index = int(top_indices[picked])
            else:
                best_value, best_index = slice_.max(dim=0)
                best_index = int(best_index)
            if allow_end and values[c.TOKEN_END_SEMANTIC] >= best_value:
                self._append((c.TOKEN_END_SEMANTIC,), (None,))
                reached_end = True
                break
            generated.append(best_index)
            self._append((start + best_index,), (None,))
        return tuple(generated), reached_end

    runtime.PersistentInterleavedSession._generate_semantic = decoded_generate_semantic


def install_bias(family_bias: float, continue_bias: float) -> None:
    """Bias TASK_MT, and WRITE_GENERATE only after a fragment already ran."""

    established_choice = runtime.PersistentInterleavedSession._choice
    established_run_event = PacedInterleavedSession.run_event

    def counting_run_event(self, *args, **kwargs):
        self._fragments_this_event = 0
        return established_run_event(self, *args, **kwargs)

    def biased_choice(self, candidates) -> int:
        tokens = [int(token) for token in candidates]
        if not tokens:
            return established_choice(self, candidates)
        values = self.logits.reshape(-1)
        offsets = {token: 0.0 for token in tokens}
        if set(tokens) <= FAMILY_TOKENS:
            if c.TOKEN_TASK_S2T_TRANSLATION in offsets:
                offsets[c.TOKEN_TASK_S2T_TRANSLATION] = family_bias
            chosen = max(tokens, key=lambda t: float(values[t]) + offsets[t])
            self._fragments_this_event = getattr(self, "_fragments_this_event", 0) + 1
            return chosen
        if (
            getattr(self, "_fragments_this_event", 0) >= 1
            and c.TOKEN_WRITE_GENERATE in offsets
        ):
            offsets[c.TOKEN_WRITE_GENERATE] = continue_bias
        return max(tokens, key=lambda t: float(values[t]) + offsets[t])

    # Both classes are wrapped: the worker selects the paced session when
    # pacing is on and the base session otherwise, and a missed reset would
    # leak the fragment counter across events.
    established_base_run_event = runtime.PersistentInterleavedSession.run_event

    def counting_base_run_event(self, *args, **kwargs):
        self._fragments_this_event = 0
        return established_base_run_event(self, *args, **kwargs)

    runtime.PersistentInterleavedSession.run_event = counting_base_run_event
    PacedInterleavedSession.run_event = counting_run_event
    runtime.PersistentInterleavedSession._choice = biased_choice


def main() -> None:
    family_bias = float(os.environ.get(ENV_FAMILY_BIAS, "0") or 0.0)
    continue_bias = float(os.environ.get(ENV_CONTINUE_BIAS, "0") or 0.0)
    if family_bias < 0 or continue_bias < 0:
        raise SystemExit("biases must be non-negative")
    minimum = int(float(os.environ.get(ENV_MIN_SEMANTIC, "0") or 0))
    if minimum < 0:
        raise SystemExit(f"{ENV_MIN_SEMANTIC} must be non-negative")
    penalty = float(os.environ.get(ENV_SEMANTIC_REPETITION, "1.0") or 1.0)
    window = int(float(os.environ.get(ENV_SEMANTIC_WINDOW, "64") or 64))
    top_k = int(float(os.environ.get(ENV_SEMANTIC_TOPK, "1") or 1))
    if penalty < 1.0 or window < 0 or top_k < 1:
        raise SystemExit("semantic decoding knobs are out of range")
    install_bias(family_bias, continue_bias)
    if penalty > 1.0 or top_k > 1:
        install_semantic_decoding(penalty, window, top_k, minimum)
    else:
        install_minimum_semantic_fragment(minimum)
    probe.install_probe()  # wraps the biased choice; records what it returns
    import atexit

    atexit.register(probe._flush)
    run_worker_local_agreement.main()


if __name__ == "__main__":
    main()

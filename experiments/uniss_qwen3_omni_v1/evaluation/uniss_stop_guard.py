"""A per-request logits guard that keeps UniSS from running on, in one pass.

UniSS speaking an external translation in its source-conditioned mode fails
to stop on about 0.3% of ordinary sentences and about 6% of long, dense
ones: it falls into a run of the BiCodec silence code (7645), or a babble
loop, and never emits <|end_semantic_token|> before the 1,500-token cap.
The same sentences never fail when UniSS speaks its own translation or reads
ours without the source tokens, so the trigger is the pairing of source
speech with a translation it did not write -- which is exactly the pairing
P2 needs. Measured on P2's step-zero run (CVSS-T, 992 normal outputs),
normal speech stays under 1.45x the source duration in 99.5% of sentences.

So, step by step, without a second pass:
  * before the first spoken code, at most `lead` silence codes (0.5 s);
  * once speaking, no silence run longer than `pause` codes (1 s);
  * past `soft` x the expected length, the end token is pushed up, more the
    further past; at the hard limit it is the only choice.
Expected length is 50 codes a second times the source duration, which is
what UniSS's speed token asks for.

With `ngram` set, a run of that many speech codes (no silence in it) may
not repeat: the code that would complete a repeat is banned. E2 at step
4000, asked to stretch short English over long Chinese sources, filled the
time with loops ("a-a-a-a", "back, back, back"); on P2's step-zero run
(9,794 outputs) an exact 6- to 12-code repeat occurs in 3 sentences, so
ordinary speech is left alone.

Requests opt in with SamplingParams.extra_args["stop_guard"] = {"expected": n}.
"""

from __future__ import annotations

import math

import torch
from vllm.v1.sample.logits_processor import AdapterLogitsProcessor

SEMANTIC_0, SEMANTIC_LAST = 155761, 163952  # <|bicodec_semantic_0|> .. <|bicodec_semantic_8191|>
SILENCE = SEMANTIC_0 + 7645                  # <|bicodec_semantic_7645|>
END = 180392                                 # <|end_semantic_token|>


def make_guard(expected: int, lead: int = 25, pause: int = 50, soft: float = 1.6, hard: float = 2.2,
               ngram: int = 0, min_frac: float = 0.0):
    soft_at = int(soft * expected)
    hard_at = max(int(hard * expected), expected + 100)
    # min_frac: the end token is barred until this share of the expected length is spoken.
    # The clean-data runs (R1, R2), asked for 0.85x on short English sentences, ended at
    # once on about 20 of 4,897 -- no speech at all.
    min_codes = int(min_frac * expected)
    state = {"seen": 0, "codes": 0, "run": 0, "spoken": False}
    recent: list[int] = []                 # speech codes since the last silence code
    follows: dict[tuple, set[int]] = {}    # (ngram - 1) codes -> the codes that came next

    def guard(output_ids: list[int], logits: torch.Tensor) -> torch.Tensor:
        # Incremental: only the tokens added since the last call are looked at,
        # unless the list was rebuilt (a preempted request recomputes), then all.
        if len(output_ids) < state["seen"]:
            state.update(seen=0, codes=0, run=0, spoken=False)
            recent.clear()
            follows.clear()
        for t in output_ids[state["seen"]:]:
            if SEMANTIC_0 <= t <= SEMANTIC_LAST:
                state["codes"] += 1
                if t == SILENCE:
                    state["run"] += 1
                    recent.clear()
                else:
                    state["run"], state["spoken"] = 0, True
                    if ngram and len(recent) >= ngram - 1:
                        follows.setdefault(tuple(recent[1 - ngram:]), set()).add(t)
                    recent.append(t)
        state["seen"] = len(output_ids)
        if ngram and len(recent) >= ngram - 1:
            for t in follows.get(tuple(recent[1 - ngram:]), ()):
                logits[t] = -math.inf
        if state["codes"] < min_codes:
            logits[END] = -math.inf
        if state["codes"] == 0:
            return logits  # still in the task header: <|write_generate|><|task_tts|>...<|start_semantic_token|>
        if state["codes"] >= hard_at:
            keep = logits[END].clone()
            logits.fill_(-math.inf)
            logits[END] = keep
            return logits
        if state["run"] >= (pause if state["spoken"] else lead):
            logits[SILENCE] = -math.inf
        if state["codes"] > soft_at:
            logits[END] += 4.0 * (state["codes"] - soft_at) / max(hard_at - soft_at, 1)
        return logits

    return guard


class StopGuard(AdapterLogitsProcessor):
    """vLLM wrapper: applies make_guard() to requests that ask for it."""

    def is_argmax_invariant(self) -> bool:
        return False

    def new_req_logits_processor(self, params):
        args = (params.extra_args or {}).get("stop_guard")
        if not args:
            return None
        return make_guard(**args)

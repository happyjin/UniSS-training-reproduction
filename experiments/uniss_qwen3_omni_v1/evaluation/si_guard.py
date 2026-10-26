"""uniss_stop_guard for one block of a streaming request.

A block ends with <|wait_read|> when more is to come and <|end_semantic_token|>
when it is the last. The controller knows which, so the two are one choice --
stop this block -- and their probabilities are pooled into the block's own
closing token: a model that says "stop" with the end token where it should
have said <|wait_read|> (T1 early on: offline it only ever learned the end
token) still stops. The guard's length rules (no closing before min_frac of
the expected length, pushed up past soft, forced at the hard limit) apply to
that pooled choice.

The repetition penalty is applied here, to the block's own output only. vLLM's
repetition_penalty also counts the prompt, which in a streaming request holds
the earlier blocks' codes; offline, the prompt held no codes at all, so this
is the penalty UniSS was tuned with.

With history (the codes the earlier blocks spoke), a run of hist_n speech codes
already spoken in an earlier block may not be spoken again: the code that would
complete it is banned. The base guard's n-gram ban sees only this block; T1b,
handed a fragment, said "manipulating their" in block after block.

Requests opt in with SamplingParams.extra_args["si_guard"] = {"expected": n, "final": bool, ...}.
"""

from __future__ import annotations

import math

import torch
from vllm.v1.sample.logits_processor import AdapterLogitsProcessor

from experiments.uniss_qwen3_omni_v1.evaluation.uniss_stop_guard import END, SEMANTIC_0, SEMANTIC_LAST, SILENCE, make_guard

WAIT_READ = 180395


def make_si_guard(expected: int, final: bool, penalty: float = 1.3, history: list[int] | None = None,
                  hist_n: int = 8, **guard_kw):
    base = make_guard(expected, **guard_kw)
    spoken = [c for c in (history or []) if SEMANTIC_0 + c != SILENCE]
    follows: dict[tuple, set[int]] = {}
    for k in range(len(spoken) - hist_n + 1):
        follows.setdefault(tuple(spoken[k: k + hist_n - 1]), set()).add(SEMANTIC_0 + spoken[k + hist_n - 1])

    def guard(output_ids: list[int], logits: torch.Tensor) -> torch.Tensor:
        if follows:
            mine = [t - SEMANTIC_0 for t in output_ids if SEMANTIC_0 <= t <= SEMANTIC_LAST and t != SILENCE]
            tail = (spoken + mine)[1 - hist_n:]
            for t in follows.get(tuple(tail), ()):
                logits[t] = -math.inf
        # the pooled closing choice sits on END while the base guard steers it, then moves to this block's token
        logits[END] = torch.logaddexp(logits[END], logits[WAIT_READ])
        logits[WAIT_READ] = -math.inf
        logits = base(output_ids, logits)
        if not final:
            logits[WAIT_READ], logits[END] = logits[END].clone(), -math.inf
        if penalty != 1.0 and output_ids:
            seen = torch.as_tensor(sorted(set(output_ids)), device=logits.device)
            v = logits[seen]
            logits[seen] = torch.where(v > 0, v / penalty, v * penalty)
        return logits

    return guard


class SIGuard(AdapterLogitsProcessor):
    def is_argmax_invariant(self) -> bool:
        return False

    def new_req_logits_processor(self, params):
        args = (params.extra_args or {}).get("si_guard")
        if not args:
            return None
        return make_si_guard(**args)

"""Beam search for the cascade's two text stages.

Why
---
SimulS2ST-Omni decodes its text stage with beam search -- §4.1 verbatim: "we
use **num_beams=4** for the main streaming results and greedy decoding for
comparison" -- while our ASR and MT stages are pure argmax
(``p2st_cascade._generate``).  That is the one decoding-side difference
between the two systems that bears directly on translation quality, and it
costs no training.

It also bears on the failure this experiment is chasing.  Greedy has no way to
weigh "stop now" against "say one more word and then stop": it compares single
tokens, and ``END_CONTENT`` is a single high-probability token at almost every
step.  A beam compares whole hypotheses, so an early stop has to beat the
continuation on total score rather than on one logit.

Scope
-----
Text stages only.  The semantic stage generates up to 384 codes, where the
cost of a cacheless beam would be real, and it already has two mechanisms the
text stages lack -- the length prior and the pacing budget.  The paper puts
beams on the Thinker for the same reason.

Cost
----
This runs without a KV cache: each step re-encodes prompt plus generated
tokens for every live beam.  For a text delta, which measures 2-4 tokens on
this data, that is a handful of forward passes per step against greedy's one.
Reordering a ``Cache`` across beams would be faster and is not worth the
failure modes at this length.

Compatibility
-------------
``num_beams=1`` is not merely equivalent to the greedy loop, it *is* the
greedy loop: the caller keeps using ``_generate``.  This module is reached
only when a caller asks for more than one beam, so nothing that does not ask
for it can change behaviour.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Sequence

import torch

from experiments.uniss_streaming_p2st_pure_ce_v1.runtime.p2st_cascade import _greedy


@dataclass
class _Beam:
    tokens: list[int] = field(default_factory=list)
    score: float = 0.0

    def normalised(self, length_penalty: float) -> float:
        """Score per token, so a longer hypothesis is not penalised by length.

        With ``length_penalty`` 1.0 this is the plain mean log-probability,
        which is what makes a beam able to prefer "one more word then stop"
        over "stop now": the extra token has to be worse than the running
        average to lose, rather than merely having a probability below one.
        """
        if not self.tokens:
            return self.score
        return self.score / (len(self.tokens) ** float(length_penalty))


def _step_scores(
    logits: torch.Tensor,
    *,
    allowed: torch.Tensor | None,
    penalty: float,
    recent: Sequence[int],
    terminator: int,
    terminator_bias: float,
) -> torch.Tensor:
    """Log-softmax over the same adjusted logits ``_greedy`` would argmax.

    Reusing the adjustment rather than re-deriving it is deliberate: the
    terminator bias and the repetition penalty are the two things a beam must
    see exactly as the greedy path sees them, or the two decoders stop being
    comparable and any measured difference means nothing.
    """
    values = logits.reshape(-1).float().clone()
    if terminator_bias:
        values[terminator] = values[terminator] + float(terminator_bias)
    if penalty > 1.0 and recent:
        index = torch.tensor(list(dict.fromkeys(recent)), device=values.device)
        picked = values.index_select(0, index)
        values.index_copy_(
            0, index, torch.where(picked > 0, picked / penalty, picked * penalty)
        )
    if allowed is not None:
        masked = torch.full_like(values, float("-inf"))
        masked.index_copy_(0, allowed, values.index_select(0, allowed))
        values = masked
    return torch.log_softmax(values, dim=-1)


@torch.inference_mode()
def beam_generate(
    model,
    prompt_embeds: torch.Tensor,
    *,
    terminator: int,
    max_tokens: int,
    num_beams: int,
    length_penalty: float = 1.0,
    allowed: torch.Tensor | None = None,
    first_allowed: torch.Tensor | None = None,
    penalty: float = 1.0,
    penalty_window: int = 0,
    terminator_bias_fn: Callable[[int], float] | None = None,
) -> tuple[list[int], bool]:
    """``(tokens before the terminator, whether it was reached)``.

    The contract matches ``p2st_cascade._generate`` exactly, including that a
    run which exhausts ``max_tokens`` reports ``False`` rather than pretending
    it stopped cleanly.
    """
    if int(num_beams) <= 1:
        raise ValueError("beam_generate needs at least two beams; use _generate")
    embeddings = model.get_input_embeddings()
    device = prompt_embeds.device
    live: list[_Beam] = [_Beam()]
    finished: list[_Beam] = []

    for step in range(int(max_tokens)):
        candidates: list[_Beam] = []
        for beam in live:
            if beam.tokens:
                token_embeds = embeddings(
                    torch.tensor([beam.tokens], device=device)
                )[0]
                inputs = torch.cat([prompt_embeds, token_embeds], dim=0)
            else:
                inputs = prompt_embeds
            output = model(inputs_embeds=inputs.unsqueeze(0), use_cache=False)
            recent = beam.tokens[-penalty_window:] if penalty_window > 0 else []
            step_allowed = (
                first_allowed
                if not beam.tokens and first_allowed is not None
                else allowed
            )
            scores = _step_scores(
                output.logits[0, -1],
                allowed=step_allowed,
                penalty=penalty,
                recent=recent,
                terminator=terminator,
                terminator_bias=(
                    terminator_bias_fn(len(beam.tokens))
                    if terminator_bias_fn is not None
                    else 0.0
                ),
            )
            top = torch.topk(scores, k=min(int(num_beams), scores.numel()))
            for value, index in zip(top.values.tolist(), top.indices.tolist()):
                if not math.isfinite(value):
                    continue
                if index == terminator:
                    finished.append(_Beam(list(beam.tokens), beam.score + value))
                else:
                    candidates.append(
                        _Beam(list(beam.tokens) + [int(index)], beam.score + value)
                    )
        if not candidates:
            break
        candidates.sort(key=lambda b: b.normalised(length_penalty), reverse=True)
        live = candidates[: int(num_beams)]
        # Stop only when no live beam could overtake the best finished
        # hypothesis even in the best case.  The obvious test -- compare
        # current normalised scores -- is wrong, and wrong in the direction
        # that silently defeats the whole point: a live beam's *raw* score
        # only falls, but its *normalised* score rises as it lengthens, so a
        # beam that is behind now can win later.  The optimistic bound assumes
        # every remaining token is free, which caps the divisor at
        # ``max_tokens``.
        if finished:
            best_done = max(b.normalised(length_penalty) for b in finished)
            horizon = max(1, int(max_tokens)) ** float(length_penalty)
            if all(b.score / horizon <= best_done for b in live):
                break

    if finished:
        best = max(finished, key=lambda b: b.normalised(length_penalty))
        return best.tokens, True
    if live:
        best = max(live, key=lambda b: b.normalised(length_penalty))
        return best.tokens, False
    return [], False


__all__ = ["beam_generate"]

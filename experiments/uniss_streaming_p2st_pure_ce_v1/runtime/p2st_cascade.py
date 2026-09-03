"""Run the three prefix-to-prefix tasks as a cascade driven by the switch rule.

How this differs from the interleaved runtime
---------------------------------------------
``PersistentInterleavedSession`` generates one long sequence in which
WAIT_READ, WRITE_GENERATE and the TASK_* family choice are *sampled tokens*:
``_choice`` picks them by argmax over the model's logits.  That is the decision
four training runs failed to move and a bias sweep failed to calibrate.

Here nothing of the sort is ever generated.  Each task is a separate prompted
generation that stops at its own terminator, and what runs next comes from
``switch_rule.next_task``, which sees only how many tokens the local-agreement
committer released.  A run's own trace is checked against the rule afterwards,
so "the model did not decide" is verified rather than asserted.

Timing is recorded the way the honest-rendering work established: a fragment's
audible start is where it is actually placed, ``max(source_end_ms, previous
end)``, not where generation happened to finish.  A fragment cannot be heard
before the audio that justified it has played.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import torch

from experiments.uniss_phase3_e2e_commit_policy_v1.runtime.semantic_pacing import (
    allowed_event_tokens,
)
from experiments.uniss_phasea_stateful_longepisode_rl_v1.runtime.commit import (
    StablePrefixCommitter,
)
from experiments.uniss_phase3_v4_quality_first_true_streaming_pilot15_v2.stage_a_causal_whisper_asr.checkpoint_runtime import (  # noqa: E501
    run_cached_frontend,
)
from experiments.uniss_streaming_p2st_pure_ce_v1.runtime.switch_rule import (
    TASK_ASR,
    TASK_DONE,
    TASK_MT,
    TASK_READ,
    TASK_TTS,
    SwitchState,
    next_task,
)
from experiments.uniss_streaming_p2st_pure_ce_v1.training.task_samples_p2st import (
    FAMILY_P2ST_ASR,
    FAMILY_P2ST_MT,
    FAMILY_P2ST_TTS,
    TASK_TOKENS,
)
from training import constants_uniss as c

BLOCK_MS = 160
BLOCK_SAMPLES = 16_000 * BLOCK_MS // 1000
SEMANTIC_MS_PER_TOKEN = 20.0


@dataclass(frozen=True)
class Fragment:
    """One spoken fragment and when it can actually be heard."""

    block_index: int
    source_end_ms: int
    text: str
    semantic: tuple[int, ...]
    start_ms: float
    end_ms: float


@dataclass(frozen=True)
class StageRun:
    task: str
    block_index: int
    generated: int
    committed: int
    stopped_on_terminator: bool


@dataclass
class CascadeTrace:
    stages: list[StageRun] = field(default_factory=list)
    fragments: list[Fragment] = field(default_factory=list)
    source_deltas: list[int] = field(default_factory=list)
    target_deltas: list[int] = field(default_factory=list)
    blocks: int = 0
    source_text: str = ""
    target_text: str = ""
    decision_tokens_generated: int = 0

    def task_sequence(self) -> list[str]:
        return [stage.task for stage in self.stages]


def _greedy(
    logits: torch.Tensor,
    *,
    allowed: torch.Tensor | None,
    penalty: float,
    recent: Sequence[int],
) -> int:
    values = logits.reshape(-1).float().clone()
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
    return int(torch.argmax(values))


@torch.inference_mode()
def _generate(
    model,
    prompt_embeds: torch.Tensor,
    *,
    terminator: int,
    max_tokens: int,
    allowed: torch.Tensor | None = None,
    first_allowed: torch.Tensor | None = None,
    penalty: float = 1.0,
    penalty_window: int = 0,
) -> tuple[list[int], bool]:
    """Greedy generation from prompt embeddings, stopping at ``terminator``.

    Returns the tokens before the terminator and whether the terminator was
    actually reached.  A run that hits ``max_tokens`` instead is recorded as
    unterminated rather than silently truncated, because that is the failure
    the whole isolated-sequence design is meant to remove.
    """
    embeddings = model.get_input_embeddings()
    inputs = prompt_embeds.unsqueeze(0)
    past = None
    produced: list[int] = []
    for _ in range(int(max_tokens)):
        output = model(inputs_embeds=inputs, past_key_values=past, use_cache=True)
        past = output.past_key_values
        recent = produced[-penalty_window:] if penalty_window > 0 else []
        step_allowed = (
            first_allowed if not produced and first_allowed is not None else allowed
        )
        token = _greedy(
            output.logits[0, -1],
            allowed=step_allowed,
            penalty=penalty,
            recent=recent,
        )
        if token == terminator:
            return produced, True
        produced.append(token)
        inputs = embeddings(
            torch.tensor([[token]], device=prompt_embeds.device)
        )
    return produced, False


class P2STCascadeSession:
    """One streaming session over one source utterance."""

    def __init__(
        self,
        *,
        model,
        tokenizer,
        objective,
        frontend,
        src_lang: str,
        tgt_lang: str,
        speaker_global: Sequence[int],
        speed: float = 1.0,
        holdback: int = 2,
        max_text_tokens: int = 64,
        max_semantic_tokens: int = 384,
        # 1.1 with a window of 8 was the original setting and it is measurably
        # broken.  On emilia_zh_0004122419 the TTS stage collapses into a
        # period-3 code cycle -- 212 codes with 9 distinct values, then two
        # fragments with 3 -- which BiCodec decodes to exact zero, peak 0.0000,
        # while gold codes through the same decoder and speaker tokens give
        # 0.7647.  Widening the window to 64 alone does not help because a
        # penalty of 1.1 is too weak to close the logit gap; at 1.3 the same
        # fragments come out 20/20 and 101/101 distinct at peak 0.6131, and
        # 1.5 is bit-identical to 1.3, so the response has saturated.
        semantic_penalty: float = 1.3,
        semantic_penalty_window: int = 64,
        tts_text_scope: str = "delta",
        pace: bool = True,
        pace_margin_ms: float = 1200.0,
        pace_tail_ms: float = 2000.0,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.objective = objective
        self.frontend = frontend
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.speaker_global = tuple(int(v) for v in speaker_global)
        self.speed = float(speed)
        self.max_text_tokens = int(max_text_tokens)
        self.max_semantic_tokens = int(max_semantic_tokens)
        # A simultaneous system cannot emit target audio faster than it
        # receives source audio; if it does, the output can never be played
        # back in step with the input.  Without this cap the cascade queued
        # 51.4 s behind the source on emilia_zh_0003980703 and ran the speech
        # to 2.97x the source duration, which is not simultaneous translation
        # at all.  The budget rule is imported unchanged from the commit-policy
        # experiment, which measured the BiCodec semantic rate at exactly 50
        # tokens per second on every sample.
        #
        # It is an upper bound and nothing more.  It does not reschedule audio,
        # so it does not close the gaps between fragments and it does not move
        # the first fragment earlier -- those follow from the committer.  On
        # emilia_zh_0003929091 every fragment already sits far below the
        # budget (59 codes against 156 allowed at 1920 ms) and pacing changes
        # that sample not at all.
        self.pace = bool(pace)
        self.pace_margin_ms = float(pace_margin_ms)
        self.pace_tail_ms = float(pace_tail_ms)
        self.pace_budgets: list[dict[str, float]] = []
        self.semantic_penalty = float(semantic_penalty)
        self.semantic_penalty_window = int(semantic_penalty_window)
        if tts_text_scope not in ("delta", "prefix"):
            raise ValueError(f"unknown tts text scope {tts_text_scope!r}")
        self.tts_text_scope = tts_text_scope
        self.device = next(model.parameters()).device
        self.source_committer = StablePrefixCommitter(holdback=int(holdback))
        self.target_committer = StablePrefixCommitter(holdback=int(holdback))
        self.spoken_semantic: list[int] = []
        self.trace = CascadeTrace()
        # The terminator has to be inside the allowed set or it can never be
        # the argmax.  Leaving it out made every TTS stage run to
        # max_semantic_tokens exactly, which looked like an untrained
        # END_SEMANTIC and was in fact a mask that forbade it.  The
        # established runtime avoids this with
        # ``_restricted_semantic_choice(logits, allow_end=bool(generated))``:
        # END becomes legal once at least one code exists, never on the first
        # step, so a fragment cannot be empty.
        codes = [
            c.bicodec_semantic_id(code) for code in range(c.BICODEC_SEMANTIC_SIZE)
        ]
        self._semantic_first = torch.tensor(codes, device=self.device)
        self._semantic_allowed = torch.tensor(
            [*codes, c.TOKEN_END_SEMANTIC], device=self.device
        )
        self._last_end_ms = 0.0

    # ---------- prompt construction, mirroring the training builders ----------

    def _embed(self, tokens: Sequence[int]) -> torch.Tensor:
        return self.model.get_input_embeddings()(
            torch.tensor([int(v) for v in tokens], device=self.device)
        )

    def _acoustic_embeddings(self, hidden: torch.Tensor) -> torch.Tensor:
        bridge_dtype = self.objective.bridge_norm.weight.dtype
        hidden = hidden.to(device=self.device, dtype=bridge_dtype)
        codes = self.objective._nearest_codes(hidden)
        residual = self.objective.bridge_projection(self.objective.bridge_norm(hidden))
        base = self.model.get_input_embeddings()(
            codes.long() + c.GLM_SEMANTIC_OFFSET
        )
        return base + residual.to(base.dtype)

    def _asr_prompt(self, hidden: torch.Tensor) -> torch.Tensor:
        committed = self.tokenizer.decode(self.source_committer.committed)
        # Byte for byte the builder's layout, which is Stage-A's.  A drift
        # between these two is exactly the mismatch that made the first run
        # transcribe English audio as Chinese.
        head = self._embed(
            [
                TASK_TOKENS[FAMILY_P2ST_ASR],
                c.TOKEN_STREAMING_MODE,
                c.language_token_id(self.src_lang),
                *c.wrap_global_tokens(self.speaker_global),
                c.TOKEN_START_GLM,
            ]
        )
        tail = self._embed(
            [
                c.TOKEN_END_GLM,
                c.TOKEN_WRITE_GENERATE,
                c.language_token_id(self.src_lang),
                c.TOKEN_START_CONTENT,
                *self.tokenizer.encode(committed, add_special_tokens=False),
            ]
        )
        return torch.cat((head, self._acoustic_embeddings(hidden), tail), dim=0)

    def _mt_prompt(self) -> torch.Tensor:
        source = self.tokenizer.decode(self.source_committer.committed)
        target = self.tokenizer.decode(self.target_committer.committed)
        return self._embed(
            [
                TASK_TOKENS[FAMILY_P2ST_MT],
                c.TOKEN_STREAMING_MODE,
                c.language_token_id(self.tgt_lang),
                c.TOKEN_START_CONTENT,
                *self.tokenizer.encode(source, add_special_tokens=False),
                c.TOKEN_END_CONTENT,
                c.TOKEN_WRITE_GENERATE,
                c.language_token_id(self.tgt_lang),
                c.TOKEN_START_CONTENT,
                *self.tokenizer.encode(target, add_special_tokens=False),
            ]
        )

    def _tts_prompt(self, fragment_tokens: Sequence[int]) -> torch.Tensor:
        # Only this fragment's words, matching the builder's ``delta`` scope:
        # showing the whole committed prefix leaves the stopping point to be
        # inferred, and the measured consequence was END_SEMANTIC never
        # arriving.
        target = self.tokenizer.decode(
            list(fragment_tokens)
            if self.tts_text_scope == "delta"
            else list(self.target_committer.committed)
        )
        return self._embed(
            [
                TASK_TOKENS[FAMILY_P2ST_TTS],
                c.TOKEN_STREAMING_MODE,
                c.language_token_id(self.tgt_lang),
                *c.wrap_global_tokens(self.speaker_global),
                c.TOKEN_START_CONTENT,
                *self.tokenizer.encode(target, add_special_tokens=False),
                c.TOKEN_END_CONTENT,
                c.TOKEN_WRITE_GENERATE,
                c.language_token_id(self.tgt_lang),
                c.speed_token_id(self.speed),
                c.TOKEN_START_SEMANTIC,
                *c.encode_bicodec_semantic(self.spoken_semantic),
            ]
        )

    # ------------------------------- the cascade -------------------------------

    def run(self, waveform, *, max_blocks: int | None = None) -> CascadeTrace:
        cached = run_cached_frontend(self.frontend, waveform)
        hidden_all = cached.hidden[0]
        total_blocks = max(1, (len(waveform) + BLOCK_SAMPLES - 1) // BLOCK_SAMPLES)
        if max_blocks is not None:
            total_blocks = min(total_blocks, int(max_blocks))
        self.trace.blocks = total_blocks

        for block_index in range(total_blocks):
            samples = min(len(waveform), (block_index + 1) * BLOCK_SAMPLES)
            source_end_ms = int(round(1000.0 * samples / 16_000))
            # Block causality, measured 201/201: the prefix of the full run is
            # what a session that had heard only this much would have computed.
            glm_stop = min(len(hidden_all), -(-samples // 1280))
            if glm_stop <= 0:
                continue
            hidden = hidden_all[:glm_stop]
            exhausted = block_index == total_blocks - 1

            stage = TASK_ASR
            source_delta = 0
            target_delta = 0
            target_committed_tokens: list[int] = []
            while stage not in (TASK_READ, TASK_DONE):
                if stage == TASK_ASR:
                    produced, ended = _generate(
                        self.model,
                        self._asr_prompt(hidden),
                        terminator=c.TOKEN_END_CONTENT,
                        max_tokens=self.max_text_tokens,
                    )
                    committed = self.source_committer.update(
                        list(self.source_committer.committed) + produced,
                        final=exhausted,
                    )
                    source_delta = len(committed)
                elif stage == TASK_MT:
                    produced, ended = _generate(
                        self.model,
                        self._mt_prompt(),
                        terminator=c.TOKEN_END_CONTENT,
                        max_tokens=self.max_text_tokens,
                    )
                    committed = self.target_committer.update(
                        list(self.target_committer.committed) + produced,
                        final=exhausted,
                    )
                    target_delta = len(committed)
                    target_committed_tokens = list(committed)
                else:
                    budget = self.max_semantic_tokens
                    if self.pace:
                        budget = min(
                            budget,
                            allowed_event_tokens(
                                consumed_source_ms=float(source_end_ms),
                                already_emitted=len(self.spoken_semantic),
                                source_final=exhausted,
                                margin_ms=self.pace_margin_ms,
                                tail_ms=self.pace_tail_ms,
                            ),
                        )
                        self.pace_budgets.append(
                            {
                                "block_index": float(block_index),
                                "source_end_ms": float(source_end_ms),
                                "already_emitted": float(len(self.spoken_semantic)),
                                "budget": float(budget),
                            }
                        )
                    produced, ended = _generate(
                        self.model,
                        self._tts_prompt(target_committed_tokens),
                        terminator=c.TOKEN_END_SEMANTIC,
                        max_tokens=budget,
                        allowed=self._semantic_allowed,
                        first_allowed=self._semantic_first,
                        penalty=self.semantic_penalty,
                        penalty_window=self.semantic_penalty_window,
                    )
                    codes = [
                        int(token) - c.BICODEC_SEMANTIC_OFFSET for token in produced
                    ]
                    self.spoken_semantic.extend(codes)
                    committed = codes
                    start = max(float(source_end_ms), self._last_end_ms)
                    end = start + len(codes) * SEMANTIC_MS_PER_TOKEN
                    self._last_end_ms = end
                    self.trace.fragments.append(
                        Fragment(
                            block_index=block_index,
                            source_end_ms=source_end_ms,
                            text=self.tokenizer.decode(
                                self.target_committer.committed
                            ),
                            semantic=tuple(codes),
                            start_ms=start,
                            end_ms=end,
                        )
                    )
                self.trace.stages.append(
                    StageRun(
                        task=stage,
                        block_index=block_index,
                        generated=len(produced),
                        committed=len(committed),
                        stopped_on_terminator=bool(ended),
                    )
                )
                stage = next_task(
                    SwitchState(
                        stage=stage,
                        source_delta=source_delta,
                        target_delta=target_delta,
                        source_exhausted=exhausted,
                    )
                )
            self.trace.source_deltas.append(source_delta)
            self.trace.target_deltas.append(target_delta)

        self.trace.source_text = self.tokenizer.decode(
            self.source_committer.committed
        )
        self.trace.target_text = self.tokenizer.decode(
            self.target_committer.committed
        )
        return self.trace


__all__ = [
    "BLOCK_MS",
    "BLOCK_SAMPLES",
    "CascadeTrace",
    "Fragment",
    "P2STCascadeSession",
    "StageRun",
]

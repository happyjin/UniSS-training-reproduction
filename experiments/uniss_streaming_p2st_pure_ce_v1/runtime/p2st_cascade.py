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
import time
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
from experiments.uniss_streaming_p2st_pure_ce_v1.runtime.seeded_commit import (
    SeededPrefixCommitter,
)
from experiments.uniss_streaming_p2st_pure_ce_v1.runtime.length_prior import (
    LengthPrior,
    terminator_bias,
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
    # Which read step produced it, and the wall-clock-aware emission time that
    # SimulEval's computation_aware scorers consume as "elapsed".  Reading
    # perf_counter touches no tensor, so the generated codes stay bit-identical.
    read_step: int = 0
    elapsed_ms: float = 0.0


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
    # One delta is appended per *read step*, and cascade_mechanics feeds
    # len(source_deltas) to rule_trace, which raises unless there is exactly
    # one delta per block.  So blocks counts read steps; the number of 160 ms
    # audio blocks consumed is separate.  At stride 1 they are equal, which is
    # why every existing report is unaffected.
    blocks: int = 0
    audio_blocks: int = 0
    read_stride: int = 1
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
    terminator: int | None = None,
    terminator_bias: float = 0.0,
) -> int:
    values = logits.reshape(-1).float().clone()
    if terminator is not None and terminator_bias:
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
    terminator_bias_fn: Callable[[int], float] | None = None,
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
            terminator=terminator,
            terminator_bias=(
                terminator_bias_fn(len(produced))
                if terminator_bias_fn is not None
                else 0.0
            ),
        )
        if token == terminator:
            return produced, True
        produced.append(token)
        inputs = embeddings(
            torch.tensor([[token]], device=prompt_embeds.device)
        )
    return produced, False


def _generate_text(
    model,
    prompt_embeds: torch.Tensor,
    *,
    terminator: int,
    max_tokens: int,
    num_beams: int,
    length_penalty: float,
    penalty: float = 1.0,
    penalty_window: int = 0,
) -> tuple[list[int], bool]:
    """Greedy unless more than one beam is asked for.

    Kept as a one-line dispatch rather than a flag inside ``_generate`` so the
    greedy path stays literally the function it has always been, and so the
    beam implementation can live in the experiment that introduced it.
    """
    if int(num_beams) <= 1:
        return _generate(
            model,
            prompt_embeds,
            terminator=terminator,
            max_tokens=max_tokens,
            penalty=penalty,
            penalty_window=penalty_window,
        )
    from experiments.uniss_streaming_p2st_traj_v1.runtime.beam_text import (
        beam_generate,
    )

    return beam_generate(
        model,
        prompt_embeds,
        terminator=terminator,
        max_tokens=max_tokens,
        num_beams=int(num_beams),
        length_penalty=float(length_penalty),
        penalty=float(penalty),
        penalty_window=int(penalty_window),
    )


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
        # 2 was the value inherited from the interleaved gate.  Swept on the
        # eight demo samples, 1 dominates it and 0: at 1 the short-audio
        # internal silence is 14.3% against 19.9% at 2 and 17.8% at 0, MT chrF
        # is 72.88 against 64.19 and 50.85, semantic coverage 0.833/0.892
        # against 0.810/0.740 and 0.646/0.709, and the first audible moment
        # moves from 3360/3020 ms to 2550/2190 ms.  Revision conflicts stayed
        # at zero for every sample at 1 and at 0, so the earlier release cost
        # nothing in stability; what 0 costs is content, because it commits the
        # raw longest common prefix and the unstable tail pollutes both the
        # ASR prefix MT reads and the MT prefix TTS speaks.
        holdback: int = 1,
        source_holdback: int | None = None,
        target_holdback: int | None = None,
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
        # Beam search for the two text stages, off by default so the shipped
        # path is byte-identical to the greedy one that produced every number
        # in this lineage.  SimulS2ST-Omni decodes its text stage with
        # num_beams=4 (paper section 4.1); see
        # experiments/uniss_streaming_p2st_traj_v1/runtime/beam_text.py.
        text_num_beams: int = 1,
        text_length_penalty: float = 1.0,
        # The text stages have never had a repetition penalty; the semantic
        # stage has carried 1.3 with a 64-code window since the repeating-loop
        # failure.  SimulS2ST-Omni runs 1.1 on its text stage.  1.0 is a no-op
        # and keeps the established path unchanged.
        text_penalty: float = 1.0,
        text_penalty_window: int = 0,
        # Hold a fragment shorter than this many codes back and merge it into
        # the next one instead of emitting a stub.  Measured on the eight
        # longform samples at iter 200: 13.7% of fragments are under 320 ms
        # (16 codes) and the shortest is 140 ms, and a 140 ms burst between
        # two silences is heard as a stutter rather than as speech.  0 is off
        # and leaves every number in this lineage unchanged.
        min_fragment_tokens: int = 0,
        tts_text_scope: str = "delta",
        pace: bool = True,
        # 2000 rather than 1200: with the length prior in place the binding
        # constraint is that the cascade speaks too *little* -- long audio ran
        # at 0.539x the source and was 50.1% silent -- so the pace budget
        # should not also be pressing down on it.
        pace_margin_ms: float = 2000.0,
        pace_tail_ms: float = 2000.0,
        length_prior: object | None = None,
        length_prior_scale: float = 1.0,
        # Off by default.  Seeding changed exactly one of the eight demo
        # samples -- the other seven came out bit-identical -- and it changed
        # it for the worse: emilia_zh_0005215832 went from "The past has
        # passed and he is optimistic about his future development" at chrF
        # 75.85 to "In the past it has already passed He hopes that his future
        # development" at 49.96, and its internal silence doubled from 13.0%
        # to 26.0%.  The aggregate shift the seeding appeared to cause was
        # that one sample.  The flag is kept because the mechanism is sound --
        # it removes a forced empty commit, worth 240 ms of onset -- but there
        # is no evidence for it and one sample against.
        seed_commit: bool = False,
        # How many 160 ms blocks are consumed per read step.  1 is the
        # established behaviour and must stay bit-identical.  Larger values
        # emulate the latency multiplier m that SimulS2ST-Omni sweeps over its
        # 1000 ms chunks: k=6 is a 960 ms read step, close to its m=1, k=12 is
        # 1920 ms, close to m=2.  The frontend's causal granularity is still
        # 160 ms -- only the read *step* changes -- which the comparison report
        # has to state.
        read_stride: int = 1,
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
        self.text_num_beams = max(1, int(text_num_beams))
        self.text_length_penalty = float(text_length_penalty)
        self.text_penalty = float(text_penalty)
        self.text_penalty_window = int(text_penalty_window)
        self.min_fragment_tokens = max(0, int(min_fragment_tokens))
        # Codes generated but not yet emitted as a fragment, because they were
        # too short to stand alone.  They are already in ``spoken_semantic``,
        # so the TTS prompt is unaffected; only the emission is deferred.
        self._pending_codes: list[int] = []
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
        # ``log P(N <= n | text length)`` added to the END_SEMANTIC logit.  See
        # runtime/length_prior.py for why the CDF and not the hazard, and why
        # this is not the shape of the delta bias that failed to calibrate.
        if length_prior is None and length_prior_scale:
            try:
                length_prior = LengthPrior.load()
            except (OSError, ValueError, KeyError):
                length_prior = None
        self.length_prior = length_prior
        self.length_prior_scale = float(length_prior_scale)
        self.length_prior_traces: list[dict[str, float]] = []
        if int(read_stride) < 1:
            raise ValueError("read_stride must be at least one block")
        self.read_stride = int(read_stride)
        self.semantic_penalty = float(semantic_penalty)
        self.semantic_penalty_window = int(semantic_penalty_window)
        if tts_text_scope not in ("delta", "prefix"):
            raise ValueError(f"unknown tts text scope {tts_text_scope!r}")
        self.tts_text_scope = tts_text_scope
        self.device = next(model.parameters()).device
        # The two stages do not carry the same risk.  A revision of the ASR
        # prefix is recoverable -- MT re-reads it every block -- while a
        # revision of the target prefix is not, because TTS has already spoken
        # it and audio cannot be retracted.  Keeping them separate lets the
        # irrevocable stage stay conservative while the recoverable one runs
        # ahead.
        committer = SeededPrefixCommitter if seed_commit else StablePrefixCommitter
        self.source_committer = committer(
            holdback=int(holdback if source_holdback is None else source_holdback)
        )
        self.target_committer = committer(
            holdback=int(holdback if target_holdback is None else target_holdback)
        )
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
        started = time.perf_counter()
        cached = run_cached_frontend(self.frontend, waveform)
        hidden_all = cached.hidden[0]
        total_blocks = max(1, (len(waveform) + BLOCK_SAMPLES - 1) // BLOCK_SAMPLES)
        if max_blocks is not None:
            total_blocks = min(total_blocks, int(max_blocks))

        # Block indices at which a read step ends.  The final block is always
        # included so the whole source is consumed no matter how the stride
        # divides it; with stride 1 this is exactly range(total_blocks), so the
        # established behaviour is untouched.
        steps = list(range(self.read_stride - 1, total_blocks, self.read_stride))
        if not steps or steps[-1] != total_blocks - 1:
            steps.append(total_blocks - 1)
        self.trace.blocks = len(steps)
        self.trace.audio_blocks = total_blocks
        self.trace.read_stride = self.read_stride

        for step_position, block_index in enumerate(steps):
            samples = min(len(waveform), (block_index + 1) * BLOCK_SAMPLES)
            source_end_ms = int(round(1000.0 * samples / 16_000))
            # Block causality, measured 201/201: the prefix of the full run is
            # what a session that had heard only this much would have computed.
            glm_stop = min(len(hidden_all), -(-samples // 1280))
            if glm_stop <= 0:
                continue
            hidden = hidden_all[:glm_stop]
            exhausted = step_position == len(steps) - 1

            stage = TASK_ASR
            source_delta = 0
            target_delta = 0
            target_committed_tokens: list[int] = []
            while stage not in (TASK_READ, TASK_DONE):
                if stage == TASK_ASR:
                    produced, ended = _generate_text(
                        self.model,
                        self._asr_prompt(hidden),
                        terminator=c.TOKEN_END_CONTENT,
                        max_tokens=self.max_text_tokens,
                        num_beams=self.text_num_beams,
                        length_penalty=self.text_length_penalty,
                        penalty=self.text_penalty,
                        penalty_window=self.text_penalty_window,
                    )
                    committed = self.source_committer.update(
                        list(self.source_committer.committed) + produced,
                        final=exhausted,
                    )
                    source_delta = len(committed)
                elif stage == TASK_MT:
                    produced, ended = _generate_text(
                        self.model,
                        self._mt_prompt(),
                        terminator=c.TOKEN_END_CONTENT,
                        max_tokens=self.max_text_tokens,
                        num_beams=self.text_num_beams,
                        length_penalty=self.text_length_penalty,
                        penalty=self.text_penalty,
                        penalty_window=self.text_penalty_window,
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
                    fragment_text = self.tokenizer.decode(
                        list(target_committed_tokens)
                        if self.tts_text_scope == "delta"
                        else list(self.target_committer.committed)
                    )
                    bias_fn = terminator_bias(
                        self.length_prior,
                        text_length=len(fragment_text.strip()),
                        language=self.tgt_lang,
                        scale=self.length_prior_scale,
                    )
                    produced, ended = _generate(
                        self.model,
                        self._tts_prompt(target_committed_tokens),
                        terminator=c.TOKEN_END_SEMANTIC,
                        max_tokens=budget,
                        terminator_bias_fn=bias_fn,
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
                    # Merge a stub into the next fragment rather than emitting
                    # it.  The codes stay in ``spoken_semantic`` either way, so
                    # what changes is when they are heard, not what is said.
                    if self._pending_codes:
                        codes = self._pending_codes + codes
                        self._pending_codes = []
                    if (
                        self.min_fragment_tokens
                        and len(codes) < self.min_fragment_tokens
                        and not exhausted
                    ):
                        self._pending_codes = codes
                        codes = []
                    if not codes:
                        # Nothing emitted this step; the stage still ran, so it
                        # is recorded below like any other.
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
                        continue
                    start = max(float(source_end_ms), self._last_end_ms)
                    end = start + len(codes) * SEMANTIC_MS_PER_TOKEN
                    self._last_end_ms = end
                    self.trace.fragments.append(
                        Fragment(
                            read_step=step_position,
                            # SimulEval's computation-aware delay: when the
                            # fragment could be heard by a listener who also
                            # had to wait for our compute, not just for audio.
                            elapsed_ms=max(
                                start,
                                1000.0 * (time.perf_counter() - started),
                            ),
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

        # Flush whatever the gate is still holding.  The last read step runs
        # the TTS stage only if the MT stage committed something, so without
        # this a held stub is simply lost -- audio the model generated and the
        # listener never hears.
        if self._pending_codes:
            codes = self._pending_codes
            self._pending_codes = []
            start = max(float(source_end_ms), self._last_end_ms)
            end = start + len(codes) * SEMANTIC_MS_PER_TOKEN
            self._last_end_ms = end
            self.trace.fragments.append(
                Fragment(
                    read_step=len(steps) - 1,
                    elapsed_ms=max(start, 1000.0 * (time.perf_counter() - started)),
                    block_index=steps[-1],
                    source_end_ms=source_end_ms,
                    text=self.tokenizer.decode(self.target_committer.committed),
                    semantic=tuple(codes),
                    start_ms=start,
                    end_ms=end,
                )
            )

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

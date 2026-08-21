"""Generation primitives for the fail-closed Phase-B free-running gate."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from experiments.uniss_phase3_v4_e2e_simuls2st_pilot15_v1.data.schema import (
    E2ETrajectory,
    append_text,
)
from experiments.uniss_phase3_v4_e2e_simuls2st_pilot15_v1.evaluation.gate import (
    text_units,
)
from training import constants_uniss as c


def mt_prompt_ids(tokenizer, source_text: str, target_lang: str) -> tuple[int, ...]:
    source = tuple(
        int(value)
        for value in tokenizer.encode(source_text, add_special_tokens=False)
    )
    if not source:
        raise ValueError("incremental MT source encoded to zero tokens")
    return (
        c.TOKEN_TASK_T2T_TRANSLATION,
        c.language_token_id(target_lang),
        c.TOKEN_START_CONTENT,
        *source,
        c.TOKEN_END_CONTENT,
        c.TOKEN_WRITE_GENERATE,
        c.language_token_id(target_lang),
        c.TOKEN_START_CONTENT,
    )


def _restricted_text_choice(logits: torch.Tensor) -> int:
    values = logits.reshape(-1).float()
    text_value, text_index = values[: c.QWEN_BASE_VOCAB_END + 1].max(dim=0)
    if values[c.TOKEN_END_CONTENT] >= text_value:
        return c.TOKEN_END_CONTENT
    return int(text_index)


def _restricted_semantic_choice(
    logits: torch.Tensor, *, allow_end: bool
) -> int:
    values = logits.reshape(-1).float()
    start = c.BICODEC_SEMANTIC_OFFSET
    semantic_value, semantic_index = values[
        start : start + c.BICODEC_SEMANTIC_SIZE
    ].max(dim=0)
    if allow_end and values[c.TOKEN_END_SEMANTIC] >= semantic_value:
        return c.TOKEN_END_SEMANTIC
    return int(semantic_index)


@torch.inference_mode()
def generate_mt_prefix(
    model,
    tokenizer,
    source_text: str,
    target_lang: str,
    *,
    max_tokens: int,
) -> tuple[str, tuple[int, ...], bool]:
    if max_tokens <= 0:
        raise ValueError("incremental MT generation limit must be positive")
    device = next(model.parameters()).device
    prompt = mt_prompt_ids(tokenizer, source_text, target_lang)
    ids = torch.tensor([prompt], dtype=torch.long, device=device)
    with torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=device.type == "cuda",
    ):
        output = model(input_ids=ids, use_cache=True, return_dict=True)
    cache = output.past_key_values
    logits = output.logits[:, -1]
    generated: list[int] = []
    reached_end = False
    for _ in range(max_tokens):
        token = _restricted_text_choice(logits)
        if token == c.TOKEN_END_CONTENT:
            reached_end = True
            break
        generated.append(token)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            output = model(
                input_ids=torch.tensor([[token]], dtype=torch.long, device=device),
                past_key_values=cache,
                use_cache=True,
                return_dict=True,
            )
        cache = output.past_key_values
        logits = output.logits[:, -1]
    text = " ".join(
        tokenizer.decode(generated, skip_special_tokens=True).strip().split()
    )
    return text, tuple(generated), reached_end


def append_only_commit(previous: str, candidate: str, language: str) -> tuple[str, bool]:
    old = text_units(previous, language)
    new = text_units(candidate, language)
    if new[: len(old)] != old:
        return previous, bool(new != old)
    return candidate, False


def incremental_mt_rollout(
    model,
    tokenizer,
    source_prefixes: Sequence[str],
    target_lang: str,
    *,
    max_tokens: int,
) -> dict[str, object]:
    committed = ""
    hypotheses: list[str] = []
    raw_hypotheses: list[str] = []
    conflicts = 0
    unterminated = 0
    for source_prefix in source_prefixes:
        if not source_prefix.strip():
            hypotheses.append(committed)
            raw_hypotheses.append("")
            continue
        raw, _, reached_end = generate_mt_prefix(
            model,
            tokenizer,
            source_prefix,
            target_lang,
            max_tokens=max_tokens,
        )
        committed, conflict = append_only_commit(committed, raw, target_lang)
        conflicts += int(conflict)
        unterminated += int(not reached_end)
        hypotheses.append(committed)
        raw_hypotheses.append(raw)
    return {
        "hypotheses": hypotheses,
        "raw_hypotheses": raw_hypotheses,
        "commit_conflicts": conflicts,
        "unterminated_generations": unterminated,
    }


@dataclass(frozen=True)
class InterleavedEvent:
    event_index: int
    source_end_ms: int
    source_final: bool
    source_glm_start: int
    source_glm_end: int
    chosen_continuations: tuple[str, ...]
    asr_deltas: tuple[str, ...]
    mt_deltas: tuple[str, ...]
    semantic_tokens: tuple[int, ...]
    malformed_segments: int


class PersistentInterleavedSession:
    """One shared-Qwen, append-only acoustic/ASR/MT/semantic event session."""

    def __init__(
        self,
        qwen,
        tokenizer,
        speech_embeddings: torch.Tensor,
        trajectory: E2ETrajectory,
    ) -> None:
        self.qwen = qwen
        self.tokenizer = tokenizer
        self.speech_embeddings = speech_embeddings
        self.trajectory = trajectory
        self.device = speech_embeddings.device
        self.cache = None
        self.logits: torch.Tensor | None = None
        self.closed = False
        self.source_text = ""
        self.target_text = ""
        self.semantic: list[int] = []
        self.events: list[InterleavedEvent] = []
        header = (
            c.TOKEN_TASK_STREAMING_S2ST,
            c.TOKEN_STREAMING_MODE,
            c.TOKEN_DYNAMIC_MODE,
            c.language_token_id(trajectory.tgt_lang),
            c.speed_token_id(1.0),
            *c.wrap_global_tokens(trajectory.speaker_global),
        )
        self._append(header, (None,) * len(header))

    @torch.inference_mode()
    def _append(
        self, token_ids: Sequence[int], speech_indices: Sequence[int | None]
    ) -> None:
        if self.closed or not token_ids or len(token_ids) != len(speech_indices):
            raise ValueError("invalid append to interleaved persistent session")
        ids = torch.tensor(token_ids, dtype=torch.long, device=self.device)
        embeddings = self.qwen.get_input_embeddings()(ids)
        positions = [index for index, value in enumerate(speech_indices) if value is not None]
        if positions:
            source = torch.tensor(
                [int(speech_indices[index]) for index in positions],
                dtype=torch.long,
                device=self.device,
            )
            embeddings.index_copy_(
                0,
                torch.tensor(positions, dtype=torch.long, device=self.device),
                self.speech_embeddings.index_select(0, source).to(embeddings.dtype),
            )
        with torch.autocast(
            device_type=self.device.type,
            dtype=torch.bfloat16,
            enabled=self.device.type == "cuda",
        ):
            output = self.qwen(
                inputs_embeds=embeddings.unsqueeze(0),
                past_key_values=self.cache,
                use_cache=True,
                return_dict=True,
            )
        self.cache = output.past_key_values
        self.logits = output.logits[:, -1].float()

    def _choice(self, candidates: Sequence[int]) -> int:
        if self.logits is None:
            raise RuntimeError("interleaved session has no logits")
        values = self.logits.reshape(-1)
        return int(max((int(token) for token in candidates), key=lambda token: float(values[token])))

    def _append_source(self, start: int, stop: int) -> None:
        if not 0 <= start <= stop <= len(self.speech_embeddings):
            raise ValueError("interleaved source span is outside acoustic embeddings")
        if start == stop:
            return
        self._append(
            (
                c.TOKEN_START_GLM,
                *([c.glm_semantic_id(0)] * (stop - start)),
                c.TOKEN_END_GLM,
            ),
            (None, *range(start, stop), None),
        )

    def _generate_text(self, *, max_tokens: int) -> tuple[str, bool]:
        generated: list[int] = []
        reached_end = False
        for _ in range(max_tokens):
            if self.logits is None:
                raise RuntimeError("missing text logits")
            token = _restricted_text_choice(self.logits)
            if token == c.TOKEN_END_CONTENT:
                self._append((token,), (None,))
                reached_end = True
                break
            generated.append(token)
            self._append((token,), (None,))
        text = " ".join(
            self.tokenizer.decode(generated, skip_special_tokens=True).strip().split()
        )
        return text, reached_end

    def _generate_semantic(self, *, max_tokens: int) -> tuple[tuple[int, ...], bool]:
        generated: list[int] = []
        reached_end = False
        for _ in range(max_tokens):
            if self.logits is None:
                raise RuntimeError("missing semantic logits")
            token = _restricted_semantic_choice(self.logits, allow_end=bool(generated))
            if token == c.TOKEN_END_SEMANTIC:
                self._append((token,), (None,))
                reached_end = True
                break
            generated.append(token - c.BICODEC_SEMANTIC_OFFSET)
            self._append((token,), (None,))
        return tuple(generated), reached_end

    def run_event(
        self,
        event,
        *,
        max_fragments: int,
        max_text_tokens: int,
        max_semantic_tokens: int,
    ) -> InterleavedEvent:
        if self.closed:
            raise RuntimeError("cannot run another event after E2E EOS")
        self._append_source(event.source_glm_start, event.source_glm_end)
        continuations: list[str] = []
        asr: list[str] = []
        mt: list[str] = []
        semantic: list[int] = []
        malformed = 0
        for _ in range(max_fragments):
            choice = self._choice(
                (
                    c.TOKEN_WRITE_GENERATE,
                    c.TOKEN_WAIT_READ,
                    c.TOKEN_START_GLM,
                    c.TOKEN_EOS,
                )
            )
            if choice == c.TOKEN_START_GLM:
                continuations.append("READ_NEXT")
                break
            if choice == c.TOKEN_WAIT_READ:
                self._append((choice,), (None,))
                continuations.append("WAIT")
                break
            if choice == c.TOKEN_EOS:
                self._append((choice,), (None,))
                continuations.append("EOS")
                self.closed = True
                malformed += int(not event.source_final)
                break

            self._append((c.TOKEN_WRITE_GENERATE,), (None,))
            family = self._choice(
                (c.TOKEN_TASK_ASR, c.TOKEN_TASK_S2T_TRANSLATION, c.TOKEN_TASK_TTS)
            )
            self._append((family,), (None,))
            if family == c.TOKEN_TASK_ASR:
                continuations.append("WRITE_ASR")
                self._append(
                    (c.language_token_id(self.trajectory.src_lang), c.TOKEN_START_CONTENT),
                    (None, None),
                )
                delta, ended = self._generate_text(max_tokens=max_text_tokens)
                malformed += int(not ended)
                if delta:
                    asr.append(delta)
                    self.source_text = append_text(
                        self.source_text, delta, self.trajectory.src_lang
                    )
            elif family == c.TOKEN_TASK_S2T_TRANSLATION:
                continuations.append("WRITE_MT")
                self._append(
                    (c.language_token_id(self.trajectory.tgt_lang), c.TOKEN_START_CONTENT),
                    (None, None),
                )
                delta, ended = self._generate_text(max_tokens=max_text_tokens)
                malformed += int(not ended)
                if delta:
                    mt.append(delta)
                    self.target_text = append_text(
                        self.target_text, delta, self.trajectory.tgt_lang
                    )
            else:
                continuations.append("WRITE_SEMANTIC")
                self._append(
                    (
                        c.language_token_id(self.trajectory.tgt_lang),
                        c.speed_token_id(1.0),
                        c.TOKEN_START_SEMANTIC,
                    ),
                    (None, None, None),
                )
                values, ended = self._generate_semantic(max_tokens=max_semantic_tokens)
                malformed += int(not ended or not values)
                semantic.extend(values)
                self.semantic.extend(values)
        else:
            malformed += 1
        row = InterleavedEvent(
            event_index=event.event_index,
            source_end_ms=event.source_end_ms,
            source_final=event.source_final,
            source_glm_start=event.source_glm_start,
            source_glm_end=event.source_glm_end,
            chosen_continuations=tuple(continuations),
            asr_deltas=tuple(asr),
            mt_deltas=tuple(mt),
            semantic_tokens=tuple(semantic),
            malformed_segments=malformed,
        )
        self.events.append(row)
        return row


__all__ = [
    "InterleavedEvent",
    "PersistentInterleavedSession",
    "append_only_commit",
    "generate_mt_prefix",
    "incremental_mt_rollout",
    "mt_prompt_ids",
]

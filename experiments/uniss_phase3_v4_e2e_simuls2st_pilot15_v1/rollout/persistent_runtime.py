"""Persistent-KV V1 ASR runtime using the audited causal Whisper frontend."""

from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Sequence

import torch

from experiments.uniss_phase3_v4_e2e_simuls2st_pilot15_v1.data.schema import (
    E2ETrajectory,
    append_text,
)
from experiments.uniss_phase3_v4_e2e_simuls2st_pilot15_v1.rollout.schema import (
    V1Rollout,
    V1RolloutEvent,
    validate_rollout,
)
from experiments.uniss_phase3_v4_quality_first_true_streaming_pilot15_v1.stage_a_causal_whisper_asr import (
    evaluate_checkpoint as stage_a_eval,
)
from experiments.uniss_phase3_v4_quality_first_true_streaming_pilot15_v1.stage_a_causal_whisper_asr.training.objective import (
    terminal_codec_extension_deficit_samples,
)
from experiments.uniss_phase3_v4_quality_first_true_streaming_pilot15_v2.stage_a_causal_whisper_asr.checkpoint_runtime import (
    run_cached_frontend,
)
from training import constants_uniss as c


def runtime_sha256() -> str:
    digest = hashlib.sha256()
    dependency_paths = (
        Path(__file__),
        Path(__file__).with_name("schema.py"),
        Path(stage_a_eval.__file__),
        Path(run_cached_frontend.__code__.co_filename),
        Path(c.__file__),
    )
    for path in dependency_paths:
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _event_severity(reference: str, hypothesis: str, language: str) -> str:
    if reference == hypothesis:
        return "exact"
    if not hypothesis:
        return "empty"
    _, errors, units = stage_a_eval.error_counts(reference, hypothesis, language)
    ratio = errors / max(1, units)
    return "minor_substitution" if ratio <= 0.5 else "major_error"


class PersistentV1ASRSession:
    def __init__(self, qwen, tokenizer, speech_embeddings: torch.Tensor, trajectory: E2ETrajectory) -> None:
        self.qwen = qwen
        self.tokenizer = tokenizer
        self.speech_embeddings = speech_embeddings
        self.trajectory = trajectory
        self.device = speech_embeddings.device
        self.cache = None
        self.logits: torch.Tensor | None = None
        self.visible_glm = 0
        header = [
            c.TOKEN_TASK_STREAMING_ASR,
            c.TOKEN_STREAMING_MODE,
            c.language_token_id(trajectory.src_lang),
            *c.wrap_global_tokens(trajectory.speaker_global),
        ]
        self._append(header, [None] * len(header))

    @torch.inference_mode()
    def _append(self, token_ids: Sequence[int], speech_indices: Sequence[int | None]) -> None:
        if len(token_ids) != len(speech_indices) or not token_ids:
            raise ValueError("persistent V1 append geometry is invalid")
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
            )
        self.cache = output.past_key_values
        self.logits = output.logits[:, -1].float()

    @torch.inference_mode()
    def append_source_until(self, glm_end: int) -> None:
        glm_end = int(glm_end)
        if not self.visible_glm <= glm_end <= len(self.speech_embeddings):
            raise ValueError("persistent V1 source GLM boundary is invalid")
        if glm_end == self.visible_glm:
            return
        encoded = [c.glm_semantic_id(0)] * (glm_end - self.visible_glm)
        # Token IDs at acoustic positions are placeholders; their embeddings
        # are replaced with V1 codebook+bridge embeddings below.
        ids = [c.TOKEN_START_GLM, *encoded, c.TOKEN_END_GLM]
        mapping: list[int | None] = [None, *range(self.visible_glm, glm_end), None]
        self._append(ids, mapping)
        self.visible_glm = glm_end

    @torch.inference_mode()
    def generate(self, *, stop_id: int, max_tokens: int) -> tuple[int, ...]:
        if self.logits is None:
            raise RuntimeError("persistent V1 session has no current logits")
        generated: list[int] = []
        for _ in range(max_tokens):
            logits = self.logits.clone()
            logits[:, len(self.tokenizer) :] = -torch.inf
            token = int(logits.argmax(dim=-1)[0])
            generated.append(token)
            self._append([token], [None])
            if token == stop_id or token == c.TOKEN_EOS:
                break
        return tuple(generated)


def _speech_embeddings(objective, frontend, qwen, trajectory: E2ETrajectory) -> torch.Tensor:
    waveform = stage_a_eval.load_waveform(trajectory.source_audio)
    cached = run_cached_frontend(frontend, waveform.numpy())
    device = next(objective.parameters()).device
    hidden = cached.hidden[0].to(device)
    if len(hidden) + 1 == trajectory.source_glm_length:
        deficit = terminal_codec_extension_deficit_samples(
            int(waveform.numel()), len(hidden), trajectory.source_glm_length
        )
        if deficit is None:
            raise ValueError("unaudited terminal cached-token extension in V1 rollout")
        hidden = torch.cat((hidden, hidden[-1:]), dim=0)
    if len(hidden) != trajectory.source_glm_length:
        raise ValueError("V1 cached frontend length differs from trajectory GLM length")
    codes = objective._nearest_codes(hidden)
    residual = objective.bridge_projection(objective.bridge_norm(hidden))
    base = qwen.get_input_embeddings()(codes.long() + c.GLM_SEMANTIC_OFFSET)
    return base + residual.to(base.dtype)


@torch.inference_mode()
def rollout_trajectory(
    trajectory: E2ETrajectory,
    *,
    qwen,
    tokenizer,
    objective,
    frontend,
    v1_hf_sha256: str,
    max_event_tokens: int,
    max_final_tokens: int,
) -> V1Rollout:
    started = time.perf_counter()
    embeddings = _speech_embeddings(objective, frontend, qwen, trajectory)
    session = PersistentV1ASRSession(qwen, tokenizer, embeddings, trajectory)
    prefix = ""
    rows: list[V1RolloutEvent] = []
    for event in trajectory.events:
        generated: tuple[int, ...] = ()
        content: tuple[int, ...] = ()
        delta = ""
        reached = True
        structure = True
        early_eos = False
        if event.gold_source_delta:
            session.append_source_until(event.source_glm_end)
            expected_tokens = len(tokenizer.encode(event.gold_source_delta, add_special_tokens=False)) + 4
            generated = session.generate(
                stop_id=c.TOKEN_END_CONTENT,
                max_tokens=min(max_event_tokens, max(8, expected_tokens + 16)),
            )
            early_eos = bool(generated and generated[-1] == c.TOKEN_EOS)
            reached = bool(generated and generated[-1] == c.TOKEN_END_CONTENT)
            structure = generated[:3] == (
                c.TOKEN_WRITE_GENERATE,
                c.language_token_id(trajectory.src_lang),
                c.TOKEN_START_CONTENT,
            )
            content = tuple(stage_a_eval.content_ids(generated))
            if not content and c.TOKEN_END_CONTENT in generated:
                stop = generated.index(c.TOKEN_END_CONTENT)
                content = tuple(
                    value
                    for value in generated[:stop]
                    if value
                    not in (
                        c.TOKEN_WRITE_GENERATE,
                        c.language_token_id(trajectory.src_lang),
                        c.TOKEN_START_CONTENT,
                        c.TOKEN_EOS,
                    )
                    and 0 <= value < len(tokenizer)
                )
            delta = " ".join(tokenizer.decode(content, skip_special_tokens=True).split())
        prefix = append_text(prefix, delta, trajectory.src_lang)
        rows.append(
            V1RolloutEvent(
                event_index=event.event_index,
                source_end_ms=event.source_end_ms,
                visible_glm_tokens=session.visible_glm,
                generated_tokens=generated,
                content_tokens=content,
                v1_source_delta=delta,
                v1_source_prefix=prefix,
                reached_content_stop=reached,
                write_structure_valid=structure,
                early_eos=early_eos,
                noise_severity=_event_severity(
                    event.gold_source_delta, delta, trajectory.src_lang
                ),
            )
        )
    session.append_source_until(trajectory.source_glm_length)
    final_tokens = session.generate(stop_id=c.TOKEN_EOS, max_tokens=max_final_tokens)
    metric, errors, units = stage_a_eval.error_counts(
        trajectory.normalized_transcription, prefix, trajectory.src_lang
    )
    rollout = V1Rollout(
        sample_id=trajectory.sample_id,
        split=trajectory.split,
        src_lang=trajectory.src_lang,
        source_manifest_record=trajectory.source_manifest_record,
        v1_checkpoint_sha256=trajectory.v1_checkpoint_sha256,
        v1_hf_sha256=v1_hf_sha256,
        runtime_sha256=runtime_sha256(),
        source_audio_sha256=str(trajectory.source_audio_sha256),
        events=tuple(rows),
        final_generated_tokens=final_tokens,
        final_reached_eos=bool(final_tokens and final_tokens[-1] == c.TOKEN_EOS),
        full_text=prefix,
        metric=metric,
        errors=errors,
        reference_units=units,
        error_rate=errors / max(1, units),
        empty_events=sum(not row.v1_source_delta for row in rows),
        early_eos_events=sum(row.early_eos for row in rows),
        malformed_write_events=sum(
            bool(row.generated_tokens) and not row.write_structure_valid for row in rows
        ),
        final_visible_glm_tokens=session.visible_glm,
        elapsed_seconds=time.perf_counter() - started,
    )
    validate_rollout(rollout, expected_events=len(trajectory.events))
    return rollout


__all__ = ["PersistentV1ASRSession", "rollout_trajectory", "runtime_sha256"]

"""Append-only raw-audio runtime with bounded Emformer right context."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Iterator, Sequence

import numpy as np
import torch
from torch.nn import functional as F

from experiments.uniss_streamspeech_ctc_v1.stage05_ctc_policy.policy import (
    CTCReadWritePolicy,
)

from .model_loader import Stage09Bundle


DIRECTIONS = {
    "eng->cmn": (0, "asr_eng", "nar_s2tt_cmn", "eng", "cmn"),
    "cmn->eng": (1, "asr_cmn", "nar_s2tt_eng", "cmn", "eng"),
}


@dataclass
class Stage09Event:
    index: int
    source_end_ms: float
    final: bool
    action: str
    new_target_token_ids: list[int]
    new_target_text: str
    stable_source_count: int
    stable_target_count: int
    committed_target_count: int
    source_conflicts: int
    target_conflicts: int
    encoder_seconds: float
    b1_token_count: int
    hard_code_ids: list[int]
    qwen_speech_embeddings: torch.Tensor = field(repr=False)

    def to_dict(self) -> dict[str, object]:
        return {
            "index": self.index,
            "source_end_ms": self.source_end_ms,
            "final": self.final,
            "action": self.action,
            "new_target_token_ids": self.new_target_token_ids,
            "new_target_text": self.new_target_text,
            "stable_source_count": self.stable_source_count,
            "stable_target_count": self.stable_target_count,
            "committed_target_count": self.committed_target_count,
            "source_conflicts": self.source_conflicts,
            "target_conflicts": self.target_conflicts,
            "encoder_seconds": self.encoder_seconds,
            "b1_token_count": self.b1_token_count,
            "hard_code_ids": self.hard_code_ids,
            "qwen_embedding_shape": list(self.qwen_speech_embeddings.shape),
        }


class Stage09OnlineRuntime:
    """True chunk-causal source runtime; cumulative mel recompute leaks no future."""

    def __init__(
        self,
        bundle: Stage09Bundle,
        *,
        direction: str,
        confirmations: int = 2,
        lagging_k: int = 0,
    ) -> None:
        try:
            _, self.source_head, self.target_head, source_lang, target_lang = DIRECTIONS[
                direction
            ]
        except KeyError as exc:
            raise ValueError(f"unsupported Stage09 direction: {direction}") from exc
        self.bundle = bundle
        self.direction = direction
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.base = bundle.joint.endpoint.base
        source_processor = bundle.processors[source_lang]
        target_processor = bundle.processors[target_lang]
        self.policy = CTCReadWritePolicy(
            source_blank_id=source_processor.vocab_size(),
            target_blank_id=target_processor.vocab_size(),
            target_language=target_lang,
            target_id_to_piece=target_processor.id_to_piece,
            confirmations=confirmations,
            lagging_k=lagging_k,
        )
        self.target_processor = target_processor
        self.source_processor = source_processor
        self.audio = np.zeros(0, dtype=np.float32)
        self.next_frame = 0
        self.encoder_state = None
        self.source_path: list[int] = []
        self.target_path: list[int] = []
        self.event_index = 0
        self.finalized = False

    @property
    def segment_ms(self) -> int:
        return int(self.base.config.segment_frames) * 40

    @property
    def right_context_ms(self) -> int:
        return int(self.base.config.right_context_frames) * 40

    @property
    def committed_translation(self) -> str:
        return self.target_processor.decode(self.policy.committed_target).strip()

    @property
    def source_transcription(self) -> str:
        return self.source_processor.decode(self.policy.source.stable_tokens()).strip()

    def _valid_projected_frames(
        self, sample_length: torch.Tensor, projected_frames: int, final: bool
    ) -> int:
        mel_frames = int(self.base.mel_lengths(sample_length)[0])
        stack = int(self.base.config.stack_factor)
        if final:
            valid = (mel_frames + stack - 1) // stack
        else:
            valid = mel_frames // stack
        return max(0, min(valid, projected_frames))

    @torch.no_grad()
    def push_audio(
        self, samples: Sequence[float] | np.ndarray, *, final: bool = False
    ) -> list[Stage09Event]:
        if self.finalized:
            raise RuntimeError("Stage09 session is already finalized")
        values = np.asarray(samples, dtype=np.float32).reshape(-1)
        if values.size:
            if not np.isfinite(values).all():
                raise ValueError("audio contains non-finite samples")
            self.audio = np.concatenate((self.audio, values))
        if self.audio.size < 400 and not final:
            return []
        if self.audio.size == 0:
            if final:
                self.finalized = True
            return []
        device = self.bundle.device
        waveform = torch.from_numpy(self.audio).to(device).unsqueeze(0)
        sample_length = torch.tensor([waveform.shape[1]], device=device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            projected = self.base.extract_projected(waveform)
        valid = self._valid_projected_frames(sample_length, projected.shape[1], final)
        segment = int(self.base.config.segment_frames)
        right = int(self.base.config.right_context_frames)
        expected = segment + right
        events: list[Stage09Event] = []
        while self.next_frame < valid:
            available = valid - self.next_frame
            if not final and available < expected:
                break
            real_segment = min(segment, available)
            end = min(valid, self.next_frame + expected)
            chunk = projected[:, self.next_frame:end]
            if chunk.shape[1] < expected:
                chunk = F.pad(chunk, (0, 0, 0, expected - chunk.shape[1]))
            lengths = torch.full(
                (1,), expected, dtype=torch.long, device=device
            )
            started = time.perf_counter()
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                hidden, _, self.encoder_state = self.base.encoder.infer(
                    chunk, lengths, self.encoder_state
                )
                hidden = self.base.output_norm(hidden[:, :real_segment])
                source_ids = self.base.heads[self.source_head](hidden)[0].argmax(-1)
                target_ids = self.base.heads[self.target_head](hidden)[0].argmax(-1)
                b1 = self.bundle.joint.b1_from_hidden(
                    hidden, torch.tensor([real_segment], device=device)
                )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            encoder_seconds = time.perf_counter() - started
            self.source_path.extend(source_ids.cpu().tolist())
            self.target_path.extend(target_ids.cpu().tolist())
            is_last = final and self.next_frame + real_segment >= valid
            decision = self.policy.update(
                self.source_path, self.target_path, final=is_last
            )
            new_ids = list(decision.new_target_tokens)
            consumed = min(valid, self.next_frame + real_segment + right)
            token_length = int(b1.token_lengths[0])
            event = Stage09Event(
                index=self.event_index,
                source_end_ms=min(self.audio.size / 16.0, consumed * 40.0),
                final=is_last,
                action=decision.action,
                new_target_token_ids=new_ids,
                new_target_text=self.target_processor.decode(new_ids).strip() if new_ids else "",
                stable_source_count=decision.stable_source_count,
                stable_target_count=decision.stable_target_count,
                committed_target_count=decision.committed_target_count,
                source_conflicts=decision.source_conflicts,
                target_conflicts=decision.target_conflicts,
                encoder_seconds=encoder_seconds,
                b1_token_count=token_length,
                hard_code_ids=b1.hard_code_ids[0, :token_length].detach().cpu().tolist(),
                qwen_speech_embeddings=b1.qwen_speech_embeddings[0, :token_length].detach(),
            )
            events.append(event)
            self.event_index += 1
            self.next_frame += real_segment
            if real_segment < segment:
                break
        if final:
            self.finalized = True
        return events

    def replay_waveform(
        self, waveform: Sequence[float] | np.ndarray, *, ingress_ms: int = 160
    ) -> Iterator[Stage09Event]:
        values = np.asarray(waveform, dtype=np.float32).reshape(-1)
        chunk_samples = max(1, int(round(ingress_ms * 16)))
        for start in range(0, len(values), chunk_samples):
            end = min(len(values), start + chunk_samples)
            final = end == len(values)
            yield from self.push_audio(values[start:end], final=final)

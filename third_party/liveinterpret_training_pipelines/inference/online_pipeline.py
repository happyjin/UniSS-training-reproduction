"""Inference-only token-level Thinker→Talker runtime primitives."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import time
from typing import Any, Callable, Dict, Optional, Set, Union

import torch


@dataclass(frozen=True)
class CommittedTextToken:
    """A generated token whose own causal hidden state is now available."""

    token_id: int
    hidden: torch.Tensor
    embedding: torch.Tensor


def _select_hidden(
    hidden_states: tuple[torch.Tensor, ...], selection: Union[int, str, None]
) -> torch.Tensor:
    if not hidden_states:
        raise RuntimeError("Thinker returned no hidden states")
    if selection is None or selection in {"last", "final"}:
        return hidden_states[-1]
    if isinstance(selection, str):
        selection = int(selection)
    return hidden_states[int(selection)]


class IncrementalThinkerSession:
    """Decode text with the Thinker's native cache and commit token hiddens.

    The hidden state used by the Talker belongs to the token *fed* into the
    current causal step. A newly sampled token is therefore kept pending until
    the following step instead of pairing it with the context hidden that
    predicted it.
    """

    def __init__(
        self,
        *,
        thinker: Any,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        thinker_kwargs: Dict[str, Any],
        stop_ids: Set[int],
        max_new_tokens: int,
        hidden_selection: Union[int, str, None],
        sample_fn: Callable[[torch.Tensor], torch.Tensor],
        token_callback: Optional[Callable[[tuple[int, ...]], None]] = None,
        profiling_mode: str = "sync",
    ) -> None:
        self.thinker = thinker
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.thinker_kwargs = dict(thinker_kwargs)
        self.stop_ids = {int(token_id) for token_id in stop_ids}
        self.max_new_tokens = int(max_new_tokens)
        self.hidden_selection = hidden_selection
        self.sample_fn = sample_fn
        self.token_callback = token_callback
        if profiling_mode not in {"sync", "off"}:
            raise ValueError("profiling_mode must be 'sync' or 'off'")
        self.profiling_mode = profiling_mode
        self.past_key_values = None
        self.cache_position = torch.arange(
            input_ids.size(1), dtype=torch.long, device=input_ids.device
        )
        self.generated_ids: list[int] = []
        self.pending_token_id: Optional[int] = None
        self.done = False
        self.prompt_hidden: Optional[torch.Tensor] = None
        self.prompt_embeddings: Optional[torch.Tensor] = None
        self._prefilled = False
        self.total_forward_ms = 0.0
        self.first_sample_at: Optional[float] = None
        self.first_commit_at: Optional[float] = None
        self.completed_at: Optional[float] = None

    def _prepare(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        prepare = getattr(self.thinker, "prepare_inputs_for_generation", None)
        if prepare is None:
            model_inputs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "past_key_values": self.past_key_values,
                "cache_position": self.cache_position,
                **self.thinker_kwargs,
            }
        else:
            model_inputs = prepare(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=self.past_key_values,
                cache_position=self.cache_position,
                use_cache=True,
                **self.thinker_kwargs,
            )
        model_inputs["use_cache"] = True
        model_inputs["output_hidden_states"] = True
        model_inputs["return_dict"] = True
        return model_inputs

    def _sample(self, logits: torch.Tensor) -> int:
        sampled = self.sample_fn(logits[:, -1, :])
        return int(sampled.reshape(-1)[0].item())

    def _forward(self, model_inputs: Dict[str, Any]):
        device = self.input_ids.device
        if self.profiling_mode == "sync" and device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        output = self.thinker(**model_inputs)
        if self.profiling_mode == "sync" and device.type == "cuda":
            torch.cuda.synchronize(device)
        if self.profiling_mode != "off":
            self.total_forward_ms += (time.perf_counter() - started) * 1000.0
        self.cache_position = self.cache_position[-1:] + 1
        return output

    @torch.inference_mode()
    def prefill(self) -> Optional[int]:
        if self._prefilled:
            raise RuntimeError("Thinker session has already been prefilled")
        self._prefilled = True
        if self.max_new_tokens <= 0:
            self.done = True
            self.completed_at = time.perf_counter()
            return None
        output = self._forward(
            self._prepare(self.input_ids, self.attention_mask)
        )
        self.past_key_values = output.past_key_values
        self.prompt_hidden = _select_hidden(
            tuple(output.hidden_states), self.hidden_selection
        )
        self.prompt_embeddings = self.thinker.get_input_embeddings()(
            self.input_ids
        )
        token_id = self._sample(output.logits)
        if token_id in self.stop_ids:
            self.done = True
            self.completed_at = time.perf_counter()
            return None
        self.first_sample_at = time.perf_counter()
        self.generated_ids.append(token_id)
        if self.token_callback is not None:
            self.token_callback(tuple(self.generated_ids))
        self.pending_token_id = token_id
        return token_id

    @torch.inference_mode()
    def step(self) -> Optional[CommittedTextToken]:
        if not self._prefilled:
            raise RuntimeError("Call prefill() before step()")
        if self.pending_token_id is None:
            return None

        pending_id = self.pending_token_id
        generated = torch.tensor(
            [self.generated_ids], dtype=torch.long, device=self.input_ids.device
        )
        full_ids = torch.cat([self.input_ids, generated], dim=1)
        full_mask = torch.cat(
            [self.attention_mask, torch.ones_like(generated)], dim=1
        )
        output = self._forward(self._prepare(full_ids, full_mask))
        self.past_key_values = output.past_key_values
        selected = _select_hidden(
            tuple(output.hidden_states), self.hidden_selection
        )
        token_tensor = torch.tensor(
            [[pending_id]], dtype=torch.long, device=self.input_ids.device
        )
        committed = CommittedTextToken(
            token_id=pending_id,
            hidden=selected[:, -1:, :],
            embedding=self.thinker.get_input_embeddings()(token_tensor),
        )
        if self.first_commit_at is None:
            self.first_commit_at = time.perf_counter()

        if len(self.generated_ids) >= self.max_new_tokens:
            self.pending_token_id = None
            self.done = True
            self.completed_at = time.perf_counter()
            return committed

        next_id = self._sample(output.logits)
        if next_id in self.stop_ids:
            self.pending_token_id = None
            self.done = True
            self.completed_at = time.perf_counter()
        else:
            self.generated_ids.append(next_id)
            if self.token_callback is not None:
                self.token_callback(tuple(self.generated_ids))
            self.pending_token_id = next_id
        return committed


class _TalkerForwardModule(torch.nn.Module):
    """Keep the Talker backbone registered when it is handed to Dynamo."""

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, inputs_embeds, past_key_values):
        cached_forward = self.model.forward_cached
        undecorated = getattr(cached_forward, "__wrapped__", None)
        if undecorated is not None:
            return undecorated(
                self.model,
                inputs_embeds,
                past_key_values=past_key_values,
            )
        return cached_forward(inputs_embeds, past_key_values=past_key_values)


def _module_dtype(module: torch.nn.Module, fallback: torch.dtype) -> torch.dtype:
    parameter = next(module.parameters(), None)
    return parameter.dtype if parameter is not None else fallback


class OnlineTalkerSession:
    """Append reply conditions to a DynamicCache-backed Talker session."""

    def __init__(
        self,
        *,
        model: Any,
        control_prefix_ids: torch.Tensor,
        prompt_hidden: torch.Tensor,
        prompt_embeddings: Optional[torch.Tensor],
        full_add_prompt_len: Optional[int],
        speaker_slot_index: Optional[int] = None,
        source_xvector: Optional[torch.Tensor] = None,
        compile_model: bool = False,
        compile_fn: Callable[..., Any] = torch.compile,
        profiling_mode: str = "sync",
    ) -> None:
        from transformers.cache_utils import DynamicCache

        if control_prefix_ids.ndim != 2 or control_prefix_ids.size(0) != 1:
            raise ValueError("Online Talker supports one sequence per session")
        self.model = model
        self.talker = model.talker
        self.control_prefix_ids = control_prefix_ids
        self.prompt_hidden = prompt_hidden
        self.prompt_embeddings = prompt_embeddings
        self.full_add_prompt_len = full_add_prompt_len
        self.cache = DynamicCache()
        self.compile_model = bool(compile_model)
        if profiling_mode not in {"sync", "off"}:
            raise ValueError("profiling_mode must be 'sync' or 'off'")
        self.profiling_mode = profiling_mode
        self.compile_warmup_ms = 0.0
        self.prefill_ms = 0.0
        self.step_ms = 0.0
        self._warmup_complete = not self.compile_model
        self._metadata_prefilled = False
        self._started = False

        codec_embeddings = self.talker.model.codec_embedding(
            control_prefix_ids
        )
        if speaker_slot_index is not None and source_xvector is not None:
            codec_embeddings = codec_embeddings.clone()
            codec_embeddings[:, int(speaker_slot_index)] = source_xvector.to(
                device=codec_embeddings.device, dtype=codec_embeddings.dtype
            ).reshape(1, -1)
        self.codec_embeddings = codec_embeddings
        self.codec_dtype = codec_embeddings.dtype

        forward_module = _TalkerForwardModule(self.talker.model).eval()
        self._compile_target = forward_module if self.compile_model else None
        self._forward = (
            compile_fn(forward_module, mode="default", fullgraph=False)
            if self.compile_model
            else forward_module
        )

        talker_config = model.config.talker_config
        self.pad_condition = self._project_text_id(
            int(talker_config.tts_pad_token_id)
        )
        self.text_bos_condition = self._project_text_id(
            int(getattr(talker_config, "tts_bos_token_id", 151_672))
        )

    def _project_text_id(self, token_id: int) -> torch.Tensor:
        token = torch.tensor(
            [[token_id]], dtype=torch.long, device=self.control_prefix_ids.device
        )
        text_embedding = self.talker.model.text_embedding(token)
        projection_dtype = _module_dtype(
            self.talker.text_projection, text_embedding.dtype
        )
        return self.talker.text_projection(
            text_embedding.to(projection_dtype)
        ).to(self.codec_dtype)

    def _project_condition(self, condition: Any) -> torch.Tensor:
        signal = condition.hidden + condition.embedding
        adapter_dtype = _module_dtype(self.model.thinker_adapter, signal.dtype)
        adapted = self.model.thinker_adapter(signal.to(adapter_dtype))
        projection_dtype = _module_dtype(
            self.talker.text_projection, adapted.dtype
        )
        return self.talker.text_projection(
            adapted.to(projection_dtype)
        ).to(self.codec_dtype)

    def _invoke(self, embedding: torch.Tensor, *, phase: str):
        if self.profiling_mode == "sync" and embedding.device.type == "cuda":
            torch.cuda.synchronize(embedding.device)
        phase_started = time.perf_counter()
        started = time.perf_counter() if not self._warmup_complete else None
        output = self._forward(embedding, self.cache)
        self.cache = output.past_key_values
        must_sync = (
            embedding.device.type == "cuda"
            and (self.profiling_mode == "sync" or started is not None)
        )
        if must_sync:
            torch.cuda.synchronize(embedding.device)
        elapsed_ms = (time.perf_counter() - phase_started) * 1000.0
        if started is not None:
            self.compile_warmup_ms = (time.perf_counter() - started) * 1000.0
            self._warmup_complete = True
        if self.profiling_mode != "off":
            if phase == "prefill":
                self.prefill_ms += elapsed_ms
            else:
                self.step_ms += elapsed_ms
        hidden = output.last_hidden_state[:, -1:, :]
        return self.talker.codec_head(hidden), hidden

    @torch.inference_mode()
    def prefill_metadata(self) -> None:
        if self._metadata_prefilled:
            raise RuntimeError("Talker metadata was already prefilled")
        self._metadata_prefilled = True
        metadata_codec = self.codec_embeddings[:, :-1]
        if metadata_codec.size(1) == 0:
            return
        text_channel = self.pad_condition.expand(
            1, metadata_codec.size(1), -1
        ).clone()
        if self.full_add_prompt_len is not None:
            prompt_len = min(
                int(self.full_add_prompt_len),
                metadata_codec.size(1),
                self.prompt_hidden.size(1),
            )
            if prompt_len > 0:
                prompt_embedding = (
                    self.prompt_embeddings[:, :prompt_len]
                    if self.prompt_embeddings is not None
                    else torch.zeros_like(self.prompt_hidden[:, :prompt_len])
                )
                prompt_condition = SimpleCondition(
                    hidden=self.prompt_hidden[:, :prompt_len],
                    embedding=prompt_embedding,
                )
                text_channel[:, :prompt_len] = self._project_condition(
                    prompt_condition
                )
            text_channel[:, -1:] = self.text_bos_condition
        self._invoke(metadata_codec + text_channel, phase="prefill")

    def _condition_or_pad(self, condition: Any, thinker_done: bool):
        if condition is not None:
            return self._project_condition(condition)
        if not thinker_done:
            raise RuntimeError(
                "Talker cannot advance beyond the committed text watermark"
            )
        return self.pad_condition

    @torch.inference_mode()
    def start(self, condition: Any, *, thinker_done: bool):
        if not self._metadata_prefilled:
            raise RuntimeError("Call prefill_metadata() before start()")
        if self._started:
            raise RuntimeError("Talker session has already started")
        text_condition = self._condition_or_pad(condition, thinker_done)
        self._started = True
        codec_bos = self.codec_embeddings[:, -1:]
        return self._invoke(codec_bos + text_condition, phase="prefill")

    @torch.inference_mode()
    def step(self, main_id: Any, condition: Any, *, thinker_done: bool):
        if not self._started:
            raise RuntimeError("Call start() before step()")
        if torch.is_tensor(main_id):
            token = main_id.to(
                device=self.control_prefix_ids.device, dtype=torch.long
            ).reshape(1, 1)
        else:
            token = torch.tensor(
                [[int(main_id)]],
                dtype=torch.long,
                device=self.control_prefix_ids.device,
            )
        codec_embedding = self.talker.model.codec_embedding(token)
        text_condition = self._condition_or_pad(condition, thinker_done)
        return self._invoke(codec_embedding + text_condition, phase="step")


@dataclass(frozen=True)
class SimpleCondition:
    hidden: torch.Tensor
    embedding: torch.Tensor


@dataclass(frozen=True)
class PredictedCodecFrame:
    main_code: torch.Tensor
    residual_codes: torch.Tensor


class ResidualFrameBatcher:
    """Micro-batch independent residual predictions without reordering."""

    def __init__(self, runner: Any, *, batch_size: int = 1) -> None:
        if batch_size not in {1, 2, 4}:
            raise ValueError("batch_size must be one of 1, 2, or 4")
        self.runner = runner
        self.batch_size = int(batch_size)
        self._main_codes: list[torch.Tensor] = []
        self._codec_hiddens: list[torch.Tensor] = []

    @property
    def pending(self) -> int:
        return len(self._main_codes)

    def add(
        self, main_code: torch.Tensor, codec_hidden: torch.Tensor
    ) -> list[PredictedCodecFrame]:
        self._main_codes.append(main_code.reshape(-1))
        self._codec_hiddens.append(codec_hidden[:, -1:, :])
        if self.pending < self.batch_size:
            return []
        return self.flush()

    def flush(self) -> list[PredictedCodecFrame]:
        if not self._main_codes:
            return []
        main_codes = torch.cat(self._main_codes, dim=0)
        codec_hiddens = torch.cat(self._codec_hiddens, dim=0)
        frame_count = int(main_codes.size(0))
        if frame_count < self.batch_size:
            # Keep torch.compile on its warmed, fixed batch shape. Residual
            # predictions are frame-independent, so duplicated tail rows can
            # be discarded without changing any real frame.
            pad_count = self.batch_size - frame_count
            main_codes = torch.cat(
                [main_codes, main_codes[-1:].expand(pad_count)], dim=0
            )
            codec_hiddens = torch.cat(
                [
                    codec_hiddens,
                    codec_hiddens[-1:].expand(
                        pad_count, *codec_hiddens.shape[1:]
                    ),
                ],
                dim=0,
            )
        residual_codes = self.runner.predict(codec_hiddens, main_codes)
        frames = [
            PredictedCodecFrame(
                main_code=main_codes[index],
                residual_codes=residual_codes[index],
            )
            for index in range(frame_count)
        ]
        self._main_codes.clear()
        self._codec_hiddens.clear()
        return frames


@dataclass(frozen=True)
class ScheduledCodecFrame:
    main_code: torch.Tensor
    codec_hidden: torch.Tensor


class ThinkerTalkerScheduler:
    """Interleave cached Thinker commits and Talker frame generation."""

    def __init__(
        self,
        *,
        thinker_session: Any,
        talker_session: Any,
        start_tokens: int,
        max_codec_frames: int,
        codec_eos_id: int,
        sample_main_fn: Callable[[torch.Tensor, Optional[torch.Tensor]], torch.Tensor],
        thinker_prefilled: bool = False,
        talker_metadata_prefilled: bool = False,
    ) -> None:
        if start_tokens not in {1, 2, 4}:
            raise ValueError("start_tokens must be one of 1, 2, or 4")
        self.thinker_session = thinker_session
        self.talker_session = talker_session
        self.start_tokens = int(start_tokens)
        self.max_codec_frames = int(max_codec_frames)
        self.codec_eos_id = int(codec_eos_id)
        self.sample_main_fn = sample_main_fn
        self.thinker_prefilled = bool(thinker_prefilled)
        self.talker_metadata_prefilled = bool(talker_metadata_prefilled)
        self.max_condition_backlog = 0

    def _commit_one(self, queue: deque[Any]) -> None:
        condition = self.thinker_session.step()
        if condition is not None:
            queue.append(condition)
            self.max_condition_backlog = max(
                self.max_condition_backlog, len(queue)
            )

    @torch.inference_mode()
    def run(self):
        conditions: deque[Any] = deque()
        if not self.thinker_prefilled:
            self.thinker_session.prefill()
        if not self.talker_metadata_prefilled:
            self.talker_session.prefill_metadata()

        while (
            len(conditions) < self.start_tokens
            and not self.thinker_session.done
        ):
            self._commit_one(conditions)
        # A max-token stop still leaves the final sampled token pending. Its
        # own hidden must be committed before the Talker starts.
        while (
            len(conditions) < self.start_tokens
            and getattr(self.thinker_session, "pending_token_id", None) is not None
        ):
            self._commit_one(conditions)

        first_condition = conditions.popleft() if conditions else None
        logits, codec_hidden = self.talker_session.start(
            first_condition, thinker_done=self.thinker_session.done
        )
        previous_main: Optional[torch.Tensor] = None

        for _ in range(self.max_codec_frames):
            main_code = self.sample_main_fn(logits, previous_main).reshape(-1)
            if int(main_code[0].item()) == self.codec_eos_id:
                break
            yield ScheduledCodecFrame(
                main_code=main_code, codec_hidden=codec_hidden
            )
            previous_main = (
                main_code.reshape(1, 1)
                if previous_main is None
                else torch.cat([previous_main, main_code.reshape(1, 1)], dim=1)
            )

            if not conditions and not self.thinker_session.done:
                self._commit_one(conditions)
            if (
                not conditions
                and getattr(self.thinker_session, "pending_token_id", None)
                is not None
            ):
                self._commit_one(conditions)
            condition = conditions.popleft() if conditions else None
            logits, codec_hidden = self.talker_session.step(
                main_code,
                condition,
                thinker_done=self.thinker_session.done,
            )


__all__ = [
    "CommittedTextToken",
    "IncrementalThinkerSession",
    "OnlineTalkerSession",
    "PredictedCodecFrame",
    "ResidualFrameBatcher",
    "ScheduledCodecFrame",
    "ThinkerTalkerScheduler",
]

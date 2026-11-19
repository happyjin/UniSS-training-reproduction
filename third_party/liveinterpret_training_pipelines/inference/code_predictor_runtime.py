"""Inference-only residual code predictor rollout helpers."""

from __future__ import annotations

import time
from typing import Any, Callable

import torch


class _PredictorForwardModule(torch.nn.Module):
    """Expose the selected predictor path as a real module to Inductor.

    Compiling a bound method makes buffers such as RoPE ``inv_freq`` look like
    closed-over tensors to AOTAutograd.  Keeping the predictor registered as a
    submodule lets Dynamo lift parameters and buffers through the normal module
    state path instead.
    """

    def __init__(self, model: torch.nn.Module, *, use_kv_cache: bool) -> None:
        super().__init__()
        self.model = model
        self.use_kv_cache = use_kv_cache

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        if self.use_kv_cache:
            return self.model.forward_cached(*args, **kwargs)
        return self.model(*args, **kwargs)


class CodePredictorRunner:
    """Run greedy residual-group prediction with optional KV cache/compile."""

    _COMPILE_MODES = {"default", "reduce-overhead"}

    def __init__(
        self,
        talker: Any,
        *,
        use_kv_cache: bool = False,
        compile_model: bool = False,
        compile_mode: str = "default",
        compile_fn: Callable[..., Any] = torch.compile,
    ) -> None:
        if compile_mode not in self._COMPILE_MODES:
            raise ValueError(
                "compile_mode must be 'default' or 'reduce-overhead'"
            )
        self.talker = talker
        self.predictor = talker.code_predictor
        self.use_kv_cache = bool(use_kv_cache)
        self.compile_model = bool(compile_model)
        self.compile_mode = compile_mode
        eager_target = (
            self.predictor.model.forward_cached
            if self.use_kv_cache
            else self.predictor.model.forward
        )
        if self.compile_model:
            self._compile_target = _PredictorForwardModule(
                self.predictor.model, use_kv_cache=self.use_kv_cache
            ).eval()
            self._forward = compile_fn(
                self._compile_target, mode=compile_mode, fullgraph=False
            )
        else:
            self._compile_target = None
            self._forward = eager_target
        self._warmup_complete = not self.compile_model
        self.compile_warmup_ms = 0.0

    def _call_forward(self, *args: Any, **kwargs: Any) -> Any:
        if self.compile_model and self.compile_mode == "reduce-overhead":
            torch.compiler.cudagraph_mark_step_begin()
        return self._forward(*args, **kwargs)

    @staticmethod
    def _synchronize(tensor: torch.Tensor) -> None:
        if tensor.device.type == "cuda":
            torch.cuda.synchronize(tensor.device)

    @torch.inference_mode()
    def predict(
        self, codec_hidden: torch.Tensor, main_code: torch.Tensor
    ) -> torch.LongTensor:
        """Return greedy residual codes with shape ``[batch, groups]``."""
        if not self._warmup_complete:
            self._synchronize(codec_hidden)
            started = time.perf_counter()
            self._predict_once(codec_hidden, main_code)
            self._synchronize(codec_hidden)
            self.compile_warmup_ms = (time.perf_counter() - started) * 1000.0
            self._warmup_complete = True
        return self._predict_once(codec_hidden, main_code)

    def _initial_group_sequence(
        self, codec_hidden: torch.Tensor, main_code: torch.Tensor
    ) -> list[torch.Tensor]:
        hidden = codec_hidden[:, -1, :]
        main_embedding = self.talker.model.codec_embedding
        main_code = main_code.reshape(-1).clamp(
            0, main_embedding.num_embeddings - 1
        )
        return [hidden.unsqueeze(1), main_embedding(main_code).unsqueeze(1)]

    def _predict_once(
        self, codec_hidden: torch.Tensor, main_code: torch.Tensor
    ) -> torch.LongTensor:
        if self.use_kv_cache:
            return self._predict_cached(codec_hidden, main_code)
        return self._predict_full(codec_hidden, main_code)

    def _predict_full(
        self, codec_hidden: torch.Tensor, main_code: torch.Tensor
    ) -> torch.LongTensor:
        sequence = self._initial_group_sequence(codec_hidden, main_code)
        groups = int(self.predictor.parent_config.num_code_groups)
        vocabulary_size = int(
            self.predictor.parent_config.code_predictor_vocab_size
        )
        predicted = []
        for group_index in range(groups):
            projected = self.predictor.small_to_mtp_projection(
                torch.cat(sequence, dim=1)
            )
            hidden_states = self._call_forward(
                inputs_embeds=projected, causal=True
            )
            logits = self.predictor.lm_head[group_index](
                hidden_states[:, group_index + 1, :]
            )
            next_code = torch.argmax(logits, dim=-1)
            predicted.append(next_code)
            if group_index < groups - 1:
                embedding = self.predictor.model.codec_embedding[group_index](
                    next_code.clamp(0, vocabulary_size - 1)
                )
                sequence.append(embedding.unsqueeze(1))
        return torch.stack(predicted, dim=1)

    def _predict_cached(
        self, codec_hidden: torch.Tensor, main_code: torch.Tensor
    ) -> torch.LongTensor:
        initial = torch.cat(
            self._initial_group_sequence(codec_hidden, main_code), dim=1
        )
        projected = self.predictor.small_to_mtp_projection(initial)
        output = self._call_forward(inputs_embeds=projected)
        cache = output.past_key_values
        hidden_states = output.last_hidden_state
        groups = int(self.predictor.parent_config.num_code_groups)
        vocabulary_size = int(
            self.predictor.parent_config.code_predictor_vocab_size
        )
        predicted = []
        for group_index in range(groups):
            logits = self.predictor.lm_head[group_index](
                hidden_states[:, -1, :]
            )
            next_code = torch.argmax(logits, dim=-1)
            predicted.append(next_code)
            if group_index < groups - 1:
                embedding = self.predictor.model.codec_embedding[group_index](
                    next_code.clamp(0, vocabulary_size - 1)
                )
                step = self.predictor.small_to_mtp_projection(
                    embedding.unsqueeze(1)
                )
                output = self._call_forward(
                    inputs_embeds=step, past_key_values=cache
                )
                cache = output.past_key_values
                hidden_states = output.last_hidden_state
        return torch.stack(predicted, dim=1)


__all__ = ["CodePredictorRunner"]

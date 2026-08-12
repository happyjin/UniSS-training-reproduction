"""Native Megatron persistent-KV backend for no-gradient event roll-in."""

from __future__ import annotations

from typing import Sequence

import torch

from training import constants_uniss as c
from web_demo.runtime_parity_streaming_v2.session import KVAppendResult


class NativeMegatronKVBackend:
    """Append runtime blocks through Megatron's native static inference cache.

    This backend is used only in the no-gradient roll-in pass.  The resulting
    variable transcript is rebuilt and sent through the ordinary differentiable
    packed training path for oracle recovery.  TP=PP=1 is an explicit invariant
    of the parent experiment.
    """

    def __init__(
        self,
        model,
        objective,
        *,
        maximum_cache_tokens: int = 32_768,
    ) -> None:
        from megatron.core.inference.contexts import StaticInferenceContext

        self.model = model
        self.objective = objective
        self.context = StaticInferenceContext(
            max_batch_size=1, max_sequence_length=int(maximum_cache_tokens)
        )
        self._offset = 0

    @staticmethod
    def _output_processor(**kwargs):
        hidden = kwargs["hidden_states"]
        logits, _ = kwargs["output_layer"](
            hidden,
            weight=kwargs["output_weight"],
            runtime_gather_output=True,
        )
        logits = kwargs["scale_logits"](logits)
        return logits[-1, 0].float(), hidden[-1, 0].float()

    def _forward(self, token_ids: Sequence[int], *, decoder_input=None) -> KVAppendResult:
        from megatron.core.inference.utils import InferenceMode

        values = tuple(int(value) for value in token_ids)
        if not values:
            raise ValueError("cannot append an empty native Megatron block")
        device = next(self.model.parameters()).device
        input_ids = torch.tensor([values], dtype=torch.long, device=device)
        positions = torch.arange(
            self._offset,
            self._offset + len(values),
            dtype=torch.long,
            device=device,
        ).unsqueeze(0)
        with torch.inference_mode(), InferenceMode.active():
            raw_forward = getattr(self.model, "_event_rollout_raw_forward", None)
            if not callable(raw_forward):
                raise RuntimeError(
                    "native rollout requires the isolated training wrapper to expose "
                    "_event_rollout_raw_forward"
                )
            output = raw_forward(
                input_ids,
                positions,
                None,
                decoder_input=decoder_input,
                inference_context=self.context,
                runtime_gather_output=True,
                output_processor=self._output_processor,
            )
        self.context.increment_sequence_len_offset(len(values))
        self._offset += len(values)
        if not isinstance(output, tuple) or len(output) != 2:
            raise ValueError("native Megatron rollout returned invalid logits/hidden")
        logits, hidden = output
        return KVAppendResult(
            past_key_values=self.context,
            logits=logits.unsqueeze(0),
            last_hidden=hidden.unsqueeze(0),
        )

    def append_token_ids(
        self,
        token_ids: Sequence[int],
        *,
        past_key_values,
        capture_last_hidden: bool = False,
    ) -> KVAppendResult:
        del capture_last_hidden
        if past_key_values is not None and past_key_values is not self.context:
            raise ValueError("native Megatron session supplied a foreign KV context")
        return self._forward(token_ids)

    def append_source_codes(
        self,
        source_codes: Sequence[int],
        canonical_token_ids: Sequence[int],
        *,
        past_key_values,
    ) -> KVAppendResult:
        if past_key_values is not None and past_key_values is not self.context:
            raise ValueError("native Megatron session supplied a foreign KV context")
        codes = tuple(int(value) for value in source_codes)
        canonical = tuple(int(value) for value in canonical_token_ids)
        if canonical != tuple(c.encode_glm_semantic(codes)):
            raise ValueError("source codes and canonical GLM token IDs disagree")
        device = next(self.model.parameters()).device
        ids = torch.tensor([canonical], dtype=torch.long, device=device)
        positions = torch.arange(
            self._offset,
            self._offset + len(canonical),
            dtype=torch.long,
            device=device,
        ).unsqueeze(0)
        embedded = self.model.embedding(input_ids=ids, position_ids=positions)
        code_tensor = torch.tensor([codes], dtype=torch.long, device=device)
        adapted = self.objective.frontend_adapter(
            self.objective.codebook(code_tensor)
        )
        residual = self.objective.frontend_projection(adapted)
        if embedded.ndim != 3 or embedded.shape[1] != 1:
            raise ValueError("native Megatron embedding must be [S,1,H]")
        if residual.ndim != 3 or residual.shape[0] != 1:
            raise ValueError("rollout frontend residual must be [1,S,H]")
        decoder_input = embedded + residual.transpose(0, 1).to(embedded.dtype)
        return self._forward(canonical, decoder_input=decoder_input)


__all__ = ["NativeMegatronKVBackend"]

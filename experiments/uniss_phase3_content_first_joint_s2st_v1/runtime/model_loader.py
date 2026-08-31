"""Load the content-first SFT runtime with an optional GRPO policy overlay.

The fixed SFT checkpoint owns the exact 144-tensor true-subsecond adapter and
the event objective.  A later GRPO checkpoint owns only the 64-tensor routed
policy delta.  Keeping those two states separate is required both before the
first GRPO round (where no policy delta exists) and after later rounds.
"""

from __future__ import annotations

import hashlib
import json
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Mapping

import torch
from torch import nn
from torch.nn import functional as F
from torch.distributed.checkpoint import FileSystemReader

from experiments.uniss_phase3_event_rollout_joint_pilot15_v1.evaluation.model_loader import (
    load_runtime_models,
)
from experiments.uniss_stagea_quality_first_joint_grpo_v1.evaluation.hf_routed_lora import (
    POLICY_PREFIX,
    load_policy_state,
    policy_state_to_hf,
)
from uniss.speech_tokenizer.bicodec.bicodec_tokenizer import BiCodecTokenizer
from uniss.speech_tokenizer.glm4.utils import load_quantize_encoder


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def policy_tensor_count(checkpoint: Path) -> int:
    """Return the policy-overlay tensor count without loading tensor payloads."""

    checkpoint = Path(checkpoint).resolve()
    if not (checkpoint / ".metadata").is_file():
        raise FileNotFoundError(f"missing DCP metadata: {checkpoint}")
    metadata = FileSystemReader(str(checkpoint)).read_metadata().state_dict_metadata
    return sum(str(key).startswith(POLICY_PREFIX) for key in metadata)


class IdentityRouteController:
    """Route-compatible controller used by the pre-GRPO SFT checkpoint."""

    @contextmanager
    def route(
        self, enabled: bool, mask: torch.Tensor | None = None
    ) -> Iterator[None]:
        del enabled, mask
        yield

    def close(self) -> None:
        return None


class ContentFirstPolicyOverlay:
    """Add a routed GRPO delta on top of the always-on content-first adapter.

    The underlying HF projections are already wrapped by the exact SFT LoRA.
    Hooks therefore attach to the wrapper output.  For Megatron fused
    LayerNormLinear targets, the additive policy branch uses the captured
    pre-normalization activation, matching the native checkpoint semantics.
    """

    def __init__(
        self,
        model: nn.Module,
        branches: Mapping[str, tuple[torch.Tensor, torch.Tensor]],
        *,
        scale: float,
    ) -> None:
        if scale <= 0:
            raise ValueError("LoRA scale must be positive")
        modules = dict(model.named_modules())
        self.scale = float(scale)
        self.enabled = False
        self.active_mask: torch.Tensor | None = None
        self.handles: list[torch.utils.hooks.RemovableHandle] = []
        self.branches: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        for name, (a, b) in branches.items():
            module = modules.get(name)
            base = getattr(module, "base", None)
            weight = getattr(base, "weight", None)
            if not isinstance(module, nn.Module) or not isinstance(weight, torch.Tensor):
                raise KeyError(f"content-first LoRA wrapper is missing: {name}")
            if tuple(a.shape) != (a.shape[0], weight.shape[1]) or tuple(b.shape) != (
                weight.shape[0],
                a.shape[0],
            ):
                raise ValueError(
                    f"policy shape mismatch for {name}: A={tuple(a.shape)} "
                    f"B={tuple(b.shape)} weight={tuple(weight.shape)}"
                )
            self.branches[name] = (
                a.to(device=weight.device, dtype=weight.dtype),
                b.to(device=weight.device, dtype=weight.dtype),
            )

            def hook(_module, inputs, output, *, target=name):
                if not self.enabled:
                    return output
                if not inputs or not isinstance(inputs[0], torch.Tensor):
                    raise TypeError(f"policy target {target} received malformed input")
                source = inputs[0]
                capture = getattr(_module, "_capture", None)
                if capture is not None:
                    source = capture.current(source)
                branch_a, branch_b = self.branches[target]
                update = F.linear(F.linear(source, branch_a), branch_b) * self.scale
                if self.active_mask is not None:
                    mask = self.active_mask.to(device=update.device, dtype=update.dtype)
                    if tuple(mask.shape) != tuple(update.shape[:-1]):
                        raise ValueError(
                            f"route mask {tuple(mask.shape)} cannot cover "
                            f"activation {tuple(update.shape)}"
                        )
                    update = update * mask.unsqueeze(-1)
                return output + update.to(dtype=output.dtype)

            self.handles.append(module.register_forward_hook(hook))

    @contextmanager
    def route(
        self, enabled: bool, mask: torch.Tensor | None = None
    ) -> Iterator[None]:
        if mask is not None and mask.dtype != torch.bool:
            raise TypeError("policy route mask must be boolean")
        previous = (self.enabled, self.active_mask)
        self.enabled, self.active_mask = bool(enabled), mask
        try:
            yield
        finally:
            self.enabled, self.active_mask = previous

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def load_content_first_models(args):
    """Return the six objects required by the established rollout runtime."""

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("content-first rollout requires CUDA")
    export_value = os.environ.get("UNISS_CONTENT_FIRST_RUNTIME_EXPORT", "")
    if not export_value:
        raise RuntimeError("UNISS_CONTENT_FIRST_RUNTIME_EXPORT is not set")
    export_dir = Path(export_value).resolve()
    for name in ("manifest.json", "adapter_model.safetensors", "objective_model.safetensors"):
        if not (export_dir / name).is_file():
            raise FileNotFoundError(export_dir / name)

    encoder = load_quantize_encoder(str(args.whispervq_model)).to(device).eval()
    codebook_weight = encoder.model.codebook.weight.detach().float().cpu()
    del encoder
    torch.cuda.empty_cache()
    model, tokenizer, objective, runtime_manifest, selected = load_runtime_models(
        export_dir,
        codebook_weight=codebook_weight,
        device=device,
    )
    del codebook_weight

    policy_checkpoint = Path(args.adapter_checkpoint).resolve()
    count = policy_tensor_count(policy_checkpoint)
    if count == 0:
        controller = IdentityRouteController()
        policy_manifest: dict[str, object] = {
            "enabled": False,
            "tensor_count": 0,
            "route_semantics": "pre_grpo_content_first_sft_only",
        }
    else:
        if count != 64:
            raise ValueError(f"expected zero or 64 policy tensors, found {count}")
        state = load_policy_state(policy_checkpoint)
        config = model.config
        num_heads = int(config.num_attention_heads)
        branches = policy_state_to_hf(
            state,
            num_attention_heads=num_heads,
            num_query_groups=int(config.num_key_value_heads),
            head_dim=int(config.hidden_size) // num_heads,
        )
        controller = ContentFirstPolicyOverlay(model, branches, scale=2.0)
        policy_manifest = {
            "enabled": True,
            "tensor_count": len(state),
            "hf_route_targets": len(branches),
            "checkpoint": str(policy_checkpoint),
            "checkpoint_metadata_sha256": _sha256(policy_checkpoint / ".metadata"),
            "route_semantics": "disabled_for_asr_enabled_for_control_mt_tts",
        }

    codec = BiCodecTokenizer(model_dir=args.bicodec_model, device=device)
    codec.model.eval().requires_grad_(False)
    manifest = {
        "schema_version": "uniss_content_first_runtime_with_optional_grpo_v1",
        "runtime_export": str(export_dir),
        "runtime_export_manifest": runtime_manifest,
        "exact_sft_lora_targets": len(selected),
        "policy_overlay": policy_manifest,
    }
    return model, tokenizer, controller, manifest, objective, codec


__all__ = [
    "ContentFirstPolicyOverlay",
    "IdentityRouteController",
    "load_content_first_models",
    "policy_tensor_count",
]


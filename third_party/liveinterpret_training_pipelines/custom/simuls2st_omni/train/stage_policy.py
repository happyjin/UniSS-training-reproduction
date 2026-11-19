"""Per-stage freeze / LoRA / adapter-merge policy.

The three stages differ only in which parameters train and how adapters are
carried forward; the model wrapper and Trainer are shared. That difference is
isolated here so `train.py` stays stage-agnostic.

LoRA settings for Stage 3 are read off the released adapters rather than the
paper (verified against
`models/SimulS2ST-Omni/simuls2st_adapter/{thinker,talker}_lora/adapter_config.json`):

    thinker: r=8, alpha=16, dropout=0.0, task_type=CAUSAL_LM
             targets = q,k,v,o,gate,up,down_proj                     (7)
    talker : r=8, alpha=16, dropout=0.0, task_type=FEATURE_EXTRACTION
             targets = the same 7 plus thinker_to_talker_proj         (8)

`thinker_to_talker_proj` is the bridge layer. Omitting it from the talker's
target list would leave the thinker->talker streaming alignment untrained, which
is precisely what Stage 3 exists to learn.

Both configs carry `modules_to_save: null` and `trainable_token_indices: null`,
so embeddings and heads are outside the trainable set by construction. The
paper's "freeze embeddings and prediction heads" is therefore a consequence of
the LoRA placement, not a separate operation.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

THINKER_LORA_DIR_NAME = "thinker_lora"
TALKER_LORA_DIR_NAME = "talker_lora"

ATTN_MLP_TARGETS = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)
BRIDGE_TARGET = "thinker_to_talker_proj"


@dataclass
class LoraSpec:
    r: int
    alpha: int
    dropout: float = 0.0
    targets: tuple[str, ...] = ATTN_MLP_TARGETS
    task_type: str = "CAUSAL_LM"


@dataclass
class StagePolicy:
    """What trains, what is frozen, and which adapters to load or merge."""

    name: str
    freeze_thinker: bool = False
    freeze_talker: bool = False
    thinker_lora: Optional[LoraSpec] = None
    talker_lora: Optional[LoraSpec] = None
    # Checkpoint whose talker weights seed this stage (Stage 1 -> 2 handoff).
    init_talker_from: Optional[str] = None
    # Checkpoint whose adapters are merged into the base before new LoRA is
    # attached (Stage 2 -> 3 handoff).
    merge_adapters_from: Optional[str] = None
    extra: dict = field(default_factory=dict)


def stage1_policy() -> StagePolicy:
    """Talker warmup: thinker frozen, talker trained in full."""
    return StagePolicy(name="stage1_talker_warmup", freeze_thinker=True)


def stage2_policy(init_talker_from: Optional[str] = None) -> StagePolicy:
    """Multitask joint: thinker LoRA r32, talker still full-parameter."""
    return StagePolicy(
        name="stage2_multitask",
        thinker_lora=LoraSpec(r=32, alpha=64, task_type="CAUSAL_LM"),
        init_talker_from=init_talker_from,
    )


def stage3_policy(merge_adapters_from: str) -> StagePolicy:
    """Streaming trajectory finetune: fresh r8 LoRA on both sides."""
    return StagePolicy(
        name="stage3_streaming",
        thinker_lora=LoraSpec(r=8, alpha=16, task_type="CAUSAL_LM"),
        talker_lora=LoraSpec(
            r=8,
            alpha=16,
            targets=ATTN_MLP_TARGETS + (BRIDGE_TARGET,),
            task_type="FEATURE_EXTRACTION",
        ),
        merge_adapters_from=merge_adapters_from,
    )


STAGE_BUILDERS = {
    "stage1": stage1_policy,
    "stage2": stage2_policy,
    "stage3": stage3_policy,
}


def _lora_config(spec: LoraSpec):
    from peft import LoraConfig, TaskType

    return LoraConfig(
        task_type=getattr(TaskType, spec.task_type),
        r=spec.r,
        lora_alpha=spec.alpha,
        lora_dropout=spec.dropout,
        target_modules=list(spec.targets),
        modules_to_save=None,
        bias="none",
    )


def _assert_targets_exist(module: nn.Module, targets: tuple[str, ...], label: str) -> None:
    """Fail loudly if a target module name matches nothing.

    PEFT raises on a fully empty match, but a partial miss (e.g. a renamed bridge
    layer) would silently leave that projection untrained.
    """
    present = {name.rsplit(".", 1)[-1] for name, _ in module.named_modules()}
    missing = [t for t in targets if t not in present]
    assert not missing, f"{label}: LoRA targets not found in module tree: {missing}"


def _clear_stale_peft_state(module: nn.Module, label: str) -> None:
    """Drop `peft_config` left behind by `merge_and_unload()`.

    PEFT leaves the attribute on the returned base model. Attaching fresh LoRA
    afterwards then warns "Already found a `peft_config` attribute ... multiple
    adapters", and the stale entry makes later `save_pretrained` /
    `from_pretrained` ambiguous about which adapter is which. The merge has
    already folded the old weights in, so the config is pure residue.
    """
    if hasattr(module, "peft_config"):
        delattr(module, "peft_config")
        logger.info("cleared stale peft_config on %s after merge", label)


def merge_adapters_into_base(model, checkpoint_dir: str) -> None:
    """Fold a previous stage's adapters into the base weights.

    Stage 3 starts from a merged Stage 2 model so its fresh r8 adapters learn on
    top of the multitask model rather than stacking on r32 adapters.
    """
    from peft import PeftModel

    from simuls2st_omni.train.modeling_omni_talker import load_talker_components

    ckpt = Path(checkpoint_dir)
    assert ckpt.is_dir(), f"merge source not found: {ckpt}"

    thinker_lora = ckpt / THINKER_LORA_DIR_NAME
    if thinker_lora.is_dir():
        logger.info("merging thinker LoRA from %s", thinker_lora)
        model.thinker = PeftModel.from_pretrained(model.thinker, str(thinker_lora))
        model.thinker = model.thinker.merge_and_unload()
        _clear_stale_peft_state(model.thinker, "thinker")
    else:
        logger.warning("no %s under %s", THINKER_LORA_DIR_NAME, ckpt)

    # Handles both the talker_lora/ directory and the talker_components.pt blob,
    # merging LoRA when present.
    if load_talker_components(model, str(ckpt), merge_lora=True):
        _clear_stale_peft_state(model.talker, "talker")
        logger.info("merged talker weights from %s", ckpt)
    else:
        logger.warning("no talker artifacts under %s", ckpt)


def apply_policy(model, policy: StagePolicy):
    """Apply `policy` to a freshly loaded base model.

    Order matters: merge previous adapters, then seed talker weights, then attach
    new adapters, then freeze. Attaching LoRA before merging would stack adapters;
    freezing before attaching would mark the new LoRA params non-trainable.
    """
    if policy.merge_adapters_from:
        merge_adapters_into_base(model, policy.merge_adapters_from)

    if policy.init_talker_from:
        from simuls2st_omni.train.modeling_omni_talker import load_talker_components

        assert load_talker_components(model, policy.init_talker_from, merge_lora=True), (
            f"failed to seed talker weights from {policy.init_talker_from}"
        )
        logger.info("seeded talker from %s", policy.init_talker_from)

    if policy.thinker_lora:
        from peft import get_peft_model

        _assert_targets_exist(model.thinker, policy.thinker_lora.targets, "thinker")
        model.thinker = get_peft_model(model.thinker, _lora_config(policy.thinker_lora))
        logger.info(
            "thinker LoRA r=%d alpha=%d on %d target module names",
            policy.thinker_lora.r,
            policy.thinker_lora.alpha,
            len(policy.thinker_lora.targets),
        )

    if policy.talker_lora:
        from peft import get_peft_model

        _assert_targets_exist(model.talker, policy.talker_lora.targets, "talker")
        model.talker = get_peft_model(model.talker, _lora_config(policy.talker_lora))
        logger.info(
            "talker LoRA r=%d alpha=%d on %d target module names (bridge included: %s)",
            policy.talker_lora.r,
            policy.talker_lora.alpha,
            len(policy.talker_lora.targets),
            BRIDGE_TARGET in policy.talker_lora.targets,
        )

    return model


def build_wrapped_model(model, policy: StagePolicy):
    """Wrap the policy-adjusted base model for training."""
    from simuls2st_omni.train.modeling_omni_talker import OmniTalkerQwenOmni

    model = apply_policy(model, policy)
    wrapper = OmniTalkerQwenOmni(
        model,
        freeze_thinker=policy.freeze_thinker,
        freeze_talker=policy.freeze_talker,
    )

    # When LoRA is attached the base weights must not train; PEFT already marks
    # them frozen, so only assert here rather than re-freezing.
    if policy.thinker_lora:
        trainable = [n for n, p in wrapper.thinker.named_parameters() if p.requires_grad]
        assert trainable, "thinker LoRA attached but nothing is trainable"
        assert all("lora_" in n for n in trainable), (
            f"non-LoRA thinker params are trainable, e.g. {trainable[:3]}"
        )
    if policy.talker_lora:
        trainable = [n for n, p in wrapper.talker.named_parameters() if p.requires_grad]
        assert trainable, "talker LoRA attached but nothing is trainable"
        assert all("lora_" in n for n in trainable), (
            f"non-LoRA talker params are trainable, e.g. {trainable[:3]}"
        )

    return wrapper


def describe_trainable(model) -> dict:
    """Trainable/total parameter counts for thinker, talker and overall."""
    def count(module: nn.Module) -> tuple[int, int]:
        total = sum(p.numel() for p in module.parameters())
        trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
        return trainable, total

    thinker_trainable, thinker_total = count(model.thinker)
    talker_trainable, talker_total = count(model.talker)
    return {
        "thinker_trainable": thinker_trainable,
        "thinker_total": thinker_total,
        "talker_trainable": talker_trainable,
        "talker_total": talker_total,
        "trainable": thinker_trainable + talker_trainable,
        "total": thinker_total + talker_total,
    }

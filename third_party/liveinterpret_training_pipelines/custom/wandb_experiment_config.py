# Copyright (c) ModelScope Contributors. All rights reserved.
"""Inject extra experiment metadata into the active wandb run's ``config``.

Motivation
----------
HF's ``WandbCallback`` auto-logs ``TrainingArguments`` (learning_rate,
per_device_train_batch_size, gradient_accumulation_steps, ...) and
``model.config`` into ``wandb.config``. However, several of our ablation knobs
are driven by *environment variables* (e.g. ``QWEN_OMNISPEAK_FULL_ADD_LAYOUT``,
``TALKER_METHOD``, ``UNFREEZE_TALKER_LAYERS``) and therefore never reach
``wandb.config`` on their own — which makes it impossible to group / filter /
build parallel-coordinate views over the talker-input ablation in the wandb UI.

Mechanism
---------
We wrap ``wandb.init`` once (idempotent). After the real run object is created,
we merge the JSON object found in the ``WANDB_EXTRA_CONFIG`` environment
variable into ``run.config`` (``allow_val_change=True`` so it is safe even if a
key already exists). The launch script populates ``WANDB_EXTRA_CONFIG`` with the
experiment-specific fields, keeping this module fully generic and reusable.

No-ops safely when wandb is not installed or ``WANDB_EXTRA_CONFIG`` is
unset / not valid JSON.
"""
from __future__ import annotations

import json
import os

_PATCHED_ATTR = '_qwen_omnispeak_wandb_extra_config_patched'
_EXTRA_CONFIG_ENV = 'WANDB_EXTRA_CONFIG'


def _load_extra_config() -> dict:
    raw = os.environ.get(_EXTRA_CONFIG_ENV)
    if not raw:
        return {}
    try:
        data = json.loads(raw)
    except (ValueError, TypeError):
        return {}
    return data if isinstance(data, dict) else {}


def install_wandb_extra_config_patch() -> None:
    """Idempotently wrap ``wandb.init`` to merge ``WANDB_EXTRA_CONFIG``."""
    try:
        import wandb
    except Exception:  # pragma: no cover - wandb optional / not installed.
        return

    if getattr(wandb, _PATCHED_ATTR, False):
        return

    original_init = wandb.init

    def _patched_init(*args, **kwargs):
        run = original_init(*args, **kwargs)
        extra = _load_extra_config()
        if extra and run is not None:
            try:
                run.config.update(extra, allow_val_change=True)
            except Exception:  # pragma: no cover - never break training on logging.
                pass
        return run

    wandb.init = _patched_init
    setattr(wandb, _PATCHED_ATTR, True)


__all__ = ['install_wandb_extra_config_patch']

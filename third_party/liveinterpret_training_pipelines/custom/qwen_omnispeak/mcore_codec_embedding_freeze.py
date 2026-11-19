# Copyright (c) junyaoyang. All rights reserved.
"""Re-freeze structurally-unused ``code_predictor`` embedding params (Megatron).

Background
----------
``Qwen3CodePredictorTransformer.__init__`` (in ``modeling_qwen_omnispeak.py``)
already calls ``self.codec_embedding[-1].weight.requires_grad_(False)`` at
construction time — the last group's embedding is provably never read as an
*input* by ``forward_teacher_forced`` (only indices ``0..num_code_groups-2``
are looked up there; the last group is only ever a *prediction target* via
the separate ``lm_head[-1]`` module). See the long comment at that call site
for the full rationale and the exact crash this fixes.

Why that alone isn't enough (Megatron path only)
--------------------------------------------------
``swift.megatron.utils.utils.prepare_mcore_model()`` runs, in order:
  1. ``freeze_parameters(model, freeze_parameters_ratio=1.0, ...)`` — freezes
     every param.
  2. ``activate_parameters(model, args.trainable_parameters, ...)`` — for
     ``--trainable_parameters visual.thinker_adapter visual.talker``, this
     does a **blanket** ``p.requires_grad = True`` for every param whose name
     starts with ``visual.talker`` — including our already-frozen
     ``codec_embedding[-1]``, silently undoing the ``__init__``-time freeze.

This module re-applies the freeze *after* ``activate_parameters()`` has run
but *before* ``wrap_model()`` constructs Megatron's
``DistributedDataParallel`` (which is what actually groups params into
buckets based on ``requires_grad`` at that point in time — freezing too late,
after DDP-wrap, would not remove it from the tracked bucket). The exact
window between those two steps is
``BaseMegatronTrainer.prepare_model()``'s
``self.peft_models = self._prepare_peft_model(self.unwrapped_models)``
(includes step 1+2 above) followed by
``self.wrapped_models = wrap_model(args, self.unwrapped_models)``
(``swift/megatron/trainers/base.py``) — so we patch
``BaseMegatronTrainer._prepare_peft_model`` to call the original, then
re-freeze, then return.

Repro'd 2026-09-17: without this, enabling ``OVERLAP_GRAD_REDUCE=true``
crashes on literally every run, all ranks, always the exact same missing
param (``megatron.core.distributed.param_and_grad_buffer.
_ParamAndGradBucketGroup.reset()``'s
``assert len(per_param_grad_ready_counts) == len(params)`` —
see ``custom/megatron_grad_ready_diag.py`` for how this was pinpointed):
``visual.talker.code_predictor.model.codec_embedding.14.weight`` — because it
can never register grad-ready in any micro-batch (it structurally never
enters the backward graph), yet ``activate_parameters()`` had it marked
trainable, so Megatron's bucket group expects it every micro-batch.

No effect on DeepSpeed (that backend never calls ``prepare_mcore_model``);
no effect when this specific frozen embedding is renamed/removed/the model
architecture changes num_code_groups (best-effort name match, logs a warning
instead of failing if it can't find the expected submodule — this is a
correctness/perf refinement, must never be fatal to training).
"""
from __future__ import annotations

_INSTALLED = False


def _refreeze_dead_codec_embedding(model) -> None:
    """Best-effort: find ``*.talker.code_predictor.model.codec_embedding``
    on ``model`` and re-freeze its last entry. Silent no-op if the model
    doesn't have this submodule (e.g. talker not present in this variant).
    """
    for name, module in model.named_modules():
        if name.endswith('talker.code_predictor.model'):
            codec_embedding = getattr(module, 'codec_embedding', None)
            if codec_embedding is not None and len(codec_embedding) > 0:
                codec_embedding[-1].weight.requires_grad_(False)


def install_code_predictor_freeze_patch() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    try:
        from swift.megatron.trainers.base import BaseMegatronTrainer
    except ImportError:
        return

    _orig = BaseMegatronTrainer._prepare_peft_model

    def _prepare_peft_model_with_refreeze(self, models):
        peft_models = _orig(self, models)
        for m in models:
            try:
                _refreeze_dead_codec_embedding(m)
            except Exception as e:  # pragma: no cover - must never break training
                try:
                    from swift.utils import get_logger
                    get_logger().warning(
                        f'[qwen_omnispeak] failed to re-freeze dead codec_embedding: {e}')
                except Exception:
                    pass
        return peft_models

    _prepare_peft_model_with_refreeze.__wrapped__ = _orig
    BaseMegatronTrainer._prepare_peft_model = _prepare_peft_model_with_refreeze
    _INSTALLED = True
    print('[qwen_omnispeak] installed code_predictor dead-embedding re-freeze patch '
          '(post activate_parameters, pre wrap_model)', flush=True)

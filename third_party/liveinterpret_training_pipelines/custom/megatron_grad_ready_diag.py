# Copyright (c) junyaoyang. All rights reserved.
"""Diagnostic-only patch (opt-in via ``DIAG_GRAD_READY=1``).

When Megatron's ``overlap_grad_reduce`` grad-ready invariant assert fires
(``megatron.core.distributed.param_and_grad_buffer._ParamAndGradBucketGroup.
reset()``'s ``assert len(per_param_grad_ready_counts) == len(params)``),
this prints the actual missing/extra parameter *names* (not just the bare
counts the stock traceback gives you) before re-raising — so "which
conditional forward branch skips which trainable submodule this
micro-batch" can be answered directly from one crash's stdout instead of
more rounds of speculative code reading + submit-and-wait.

Zero overhead / zero behaviour change when ``DIAG_GRAD_READY`` is unset:
this is purely additive (wraps, doesn't alter control flow) and only
touches Megatron internals when explicitly armed.
"""
from __future__ import annotations

import os

_ID_TO_NAME: dict = {}
_INSTALLED = False


def _unwrap(model):
    """Same recursive-unwrap pattern as ``mfu_logging._unwrap_model`` —
    Megatron hands back ``Float16Module``/mcore wrappers where the real
    ``nn.Module`` (with meaningful ``named_parameters()``) lives one or more
    ``.module``/``.model`` hops deeper.
    """
    seen = set()
    while model is not None and id(model) not in seen:
        seen.add(id(model))
        for attr in ('module', 'model', '_module'):
            inner = getattr(model, attr, None)
            if inner is not None and inner is not model:
                model = inner
                break
        else:
            break
    return model


def _refresh_id_to_name(trainer) -> None:
    """Rebuild the ``id(param) -> qualified name`` map from the trainer's
    live model list. Called at the start of every ``train_step`` (cheap:
    just a dict rebuild from ``named_parameters()``, no GPU sync) so the map
    is always current even if models/params are ever re-wrapped.
    """
    models = getattr(trainer, 'unwrapped_models', None) or getattr(trainer, 'wrapped_models', None)
    if not models:
        return
    for m in models:
        try:
            mm = _unwrap(m)
            for name, p in mm.named_parameters():
                _ID_TO_NAME[id(p)] = name
        except Exception:
            continue


def install_grad_ready_diag() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    if os.environ.get('DIAG_GRAD_READY') != '1':
        return
    try:
        from swift.megatron.trainers.base import BaseMegatronTrainer
    except ImportError:
        return
    try:
        from megatron.core.distributed.param_and_grad_buffer import _ParamAndGradBucketGroup
    except ImportError:
        return

    _orig_step = BaseMegatronTrainer.train_step

    def _step_with_diag(self, *args, **kwargs):
        _refresh_id_to_name(self)
        return _orig_step(self, *args, **kwargs)

    _step_with_diag.__wrapped__ = _orig_step
    BaseMegatronTrainer.train_step = _step_with_diag

    _orig_reset = _ParamAndGradBucketGroup.reset

    def _reset_with_diag(self):
        try:
            return _orig_reset(self)
        except AssertionError:
            try:
                ready_ids = {id(p) for p in self.per_param_grad_ready_counts.keys()}
                all_ids = {id(p) for p in self.params}
                missing = all_ids - ready_ids
                extra = ready_ids - all_ids
                names_missing = sorted(_ID_TO_NAME.get(i, f'<unknown id={i} shape=?>') for i in missing)
                names_extra = sorted(_ID_TO_NAME.get(i, f'<unknown id={i}>') for i in extra)
                # Fall back to shape/dtype fingerprints for anything we
                # couldn't name (e.g. the id->name map is stale/wrong scope) —
                # still gives enough to eyeball which submodule it is.
                shape_hints = []
                for p in self.params:
                    if id(p) in missing:
                        shape_hints.append(f'{_ID_TO_NAME.get(id(p), "?")}:{tuple(p.shape)}:{p.dtype}')
                print(f'[grad-ready-diag] MISMATCH: {len(all_ids)} total trainable params in this '
                      f'bucket group, only {len(ready_ids)} registered grad-ready this batch.',
                      flush=True)
                print(f'[grad-ready-diag] MISSING ({len(missing)}): {names_missing}', flush=True)
                print(f'[grad-ready-diag] MISSING shape/dtype detail: {shape_hints}', flush=True)
                if names_extra:
                    print(f'[grad-ready-diag] EXTRA ({len(extra)}, unexpected — should not happen): '
                          f'{names_extra}', flush=True)
            except Exception as e:  # pragma: no cover - diagnostics must never mask the real error
                print(f'[grad-ready-diag] diag itself failed: {e!r}', flush=True)
            raise

    _ParamAndGradBucketGroup.reset = _reset_with_diag

    _INSTALLED = True
    print('[grad-ready-diag] installed (DIAG_GRAD_READY=1): will name missing params on '
          'overlap_grad_reduce grad-ready AssertionError', flush=True)

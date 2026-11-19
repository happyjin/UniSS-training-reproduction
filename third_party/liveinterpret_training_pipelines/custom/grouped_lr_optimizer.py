"""Custom optimizer callback: group-wise learning rate for thinker vs rest.

When ``THINKER_LR`` env var is set to a positive float, all trainable
parameters whose name starts with ``model.thinker.`` (the thinker LLM body,
audio tower, etc. — but **not** ``model.thinker_adapter`` because the prefix
uses a dot, not an underscore) use ``THINKER_LR``, while every other
trainable parameter (``model.thinker_adapter``, ``model.talker`` …) uses
``args.learning_rate``.

If ``THINKER_LR`` is unset / <= 0 the callback delegates to the default
HuggingFace optimizer, making it a safe no-op when the feature is off.

Registered in plugin.py as ``optimizers_map['grouped_lr']`` and activated
via ``--optimizer grouped_lr`` on the swift CLI.
"""
import os

from transformers.trainer import Trainer as HfTrainer

from swift.optimizers.base import OptimizerCallback


class GroupedLrOptimizerCallback(OptimizerCallback):
    """Thinker vs. rest group-wise LR."""

    def create_optimizer(self, model=None):
        args = self.args
        if model is None:
            model = self.trainer.model

        thinker_lr_str = os.environ.get('THINKER_LR', '')
        try:
            thinker_lr = float(thinker_lr_str) if thinker_lr_str else 0.0
        except ValueError:
            thinker_lr = 0.0

        if thinker_lr <= 0:
            # No override — identical to the default OptimizerCallback.
            return HfTrainer.create_optimizer(self.trainer)

        decay_parameters = set(HfTrainer.get_decay_parameter_names(None, model))

        thinker_params = []
        other_params = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            # 'model.thinker.' (trailing dot) matches the thinker body but
            # NOT 'model.thinker_adapter' (underscore after 'thinker').
            if name.startswith('model.thinker.'):
                thinker_params.append((name, param))
            else:
                other_params.append((name, param))

        optimizer_grouped_parameters = []
        for lr, parameters in [(thinker_lr, thinker_params),
                               (args.learning_rate, other_params)]:
            for use_wd, wd in [(False, 0.0), (True, args.weight_decay)]:
                if use_wd:
                    params = [p for n, p in parameters if n in decay_parameters]
                else:
                    params = [p for n, p in parameters if n not in decay_parameters]
                if not params:
                    continue
                optimizer_grouped_parameters.append({
                    'params': params,
                    'weight_decay': wd,
                    'lr': lr,
                })

        optimizer_cls, optimizer_kwargs = HfTrainer.get_optimizer_cls_and_kwargs(args, model)
        optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

        n_thinker = sum(p.numel() for _, p in thinker_params)
        n_other = sum(p.numel() for _, p in other_params)
        try:
            from swift.utils import get_logger
            get_logger().info(
                f'[grouped_lr] thinker_lr={thinker_lr} '
                f'({n_thinker / 1e6:.1f}M params, {len(thinker_params)} tensors), '
                f'default_lr={args.learning_rate} '
                f'({n_other / 1e6:.1f}M params, {len(other_params)} tensors)'
            )
        except Exception:
            pass
        return optimizer

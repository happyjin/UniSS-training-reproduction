"""对拍用的额外 swift 插件：记录与 native_trace.py 相同格式的轨迹（写到 $TRACE_OUT，非 0 rank 加 .rank<r> 后缀）。

    swift sft --external_plugins custom/plugin.py custom/simuls2st_omni/tests/parity/swift_trace_plugin.py ...
"""
import json
import os

from simuls2st_omni.integration import register as reg

_TRACE = {'steps': [], 'log': [], 'checksum': {}, 'rank': int(os.environ.get('RANK', 0))}
_orig_compute = reg._compute_loss


def _compute_loss(template, model, inputs, trainer):
    if model.training:
        _TRACE['steps'].append({'task': inputs.get('task_type'), 'ids': list(inputs.get('ids') or []),
                 'fp': {k: float(v.detach().double().sum()) for k, v in inputs.items()
                        if hasattr(v, 'detach') and k != 'labels'}})
        if os.environ.get('TRACE_DUMP_STEP') and len(_TRACE['steps']) == int(os.environ['TRACE_DUMP_STEP']):
            import torch as _t; _t.save({k: v.detach().cpu() for k, v in inputs.items() if hasattr(v, 'detach')}, os.environ['TRACE_OUT_DUMP'] if 'TRACE_OUT_DUMP' in os.environ else (os.environ.get('TRACE_OUT', 'trace') + '.step.pt'))
    return _orig_compute(template, model, inputs, trainer)


reg._compute_loss = _compute_loss

# 记录每步梯度裁剪前的逐参数梯度指纹（定位分叉）
import torch as _torch
_orig_clip = _torch.nn.utils.clip_grad_norm_
def _clip(parameters, *a, **k):
    params = list(parameters)
    _TRACE.setdefault('grad_fp', []).append(
        [None if p.grad is None else float(p.grad.detach().double().sum()) for p in params])
    return _orig_clip(params, *a, **k)
_torch.nn.utils.clip_grad_norm_ = _clip

# 每个训练步开始前记录可训参数指纹（定位分叉）
import transformers as _tf
_orig_ts = _tf.Trainer.training_step
def _training_step(self, model, inputs, *args, **kwargs):
    m = self.accelerator.unwrap_model(model)
    _TRACE.setdefault('lr_applied', []).append([g['lr'] for g in self.optimizer.param_groups] if self.optimizer else None)
    _TRACE.setdefault('param_fp', []).append(
        [float(p.detach().double().sum()) for p in m.parameters() if p.requires_grad])
    return _orig_ts(self, model, inputs, *args, **kwargs)
_tf.Trainer.training_step = _training_step

from swift.trainers.mixin import SwiftMixin  # noqa: E402

_orig_train = SwiftMixin.train


def _train(self, *args, **kwargs):
    out = _orig_train(self, *args, **kwargs)
    m = self.accelerator.unwrap_model(self.model)
    for name in ('thinker', 'talker'):
        _TRACE['checksum'][name] = {n: float(p.detach().cpu().double().sum())
                                    for n, p in getattr(m, name).named_parameters() if p.requires_grad}
    _TRACE['log'] = self.state.log_history
    path = os.environ['TRACE_OUT']
    if _TRACE['rank']:
        path = f'{path}.rank{_TRACE["rank"]}'
    with open(path, 'w') as f:
        json.dump(_TRACE, f, ensure_ascii=False)
    return out


SwiftMixin.train = _train

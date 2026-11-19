"""运行原生（HF Trainer）训练入口并记录逐步轨迹，作为 swift 版本的对拍参照。

    # 源项目（源环境：Python 3.10 / transformers 4.52.3 / peft 0.18.1 / torch 2.6）
    python native_trace.py --impl source --src_root <SimulS2ST-Omni-stage123> --out trace.json -- --config s1.yaml
    # 迁入本仓库的同一份代码（swift 环境，库版本与 swift 相同 -> 应与 swift 逐位一致）
    python native_trace.py --impl ported --out trace.json -- --config s1.yaml

记录：每个训练步的 (rank, task, 样本 id)、log_history、训练结束时各可训参数的和（float64）。
为了可比，统一做三件事（都不改变训练语义）：固定 CPU（关 MPS）；挂 LoRA 前 `torch.manual_seed(TRACE_SEED)`
（源没有播种，LoRA 初始化随进程而变）；优化器固定 `adamw_torch`（torch>=2.8 时 HF 默认选 fused 实现）。
`--ddp_forward`：把源 `compute_loss` 里的 `model.module(...)` 换成经 DDP 的 `model(...)`，即 swift 版本的多卡语义。
"""
import argparse
import inspect
import json
import os
import runpy
import sys
import textwrap

import torch

torch.backends.mps.is_available = lambda: False

HERE = os.path.dirname(os.path.abspath(__file__))
CUSTOM_DIR = os.path.abspath(os.path.join(HERE, '..', '..', '..'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--impl', choices=('source', 'ported'), required=True)
    ap.add_argument('--src_root', default=None)
    ap.add_argument('--out', required=True)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--ddp_forward', action='store_true')
    ap.add_argument('--find_unused', action='store_true', help='ddp_find_unused_parameters=True（配合 --ddp_forward）')
    ap.add_argument('train_args', nargs=argparse.REMAINDER)
    a = ap.parse_args()
    train_args = [x for x in a.train_args if x != '--']

    if a.impl == 'source':
        sys.path.insert(0, a.src_root)
        pkg, entry = 'src', os.path.join(a.src_root, 'scripts', 'train.py')
    else:
        sys.path.insert(0, CUSTOM_DIR)
        pkg, entry = 'simuls2st_omni', os.path.join(CUSTOM_DIR, 'simuls2st_omni', 'train', 'native_train.py')
    mot = __import__(f'{pkg}.train.modeling_omni_talker', fromlist=['x'])
    sp = __import__(f'{pkg}.train.stage_policy', fromlist=['x'])
    # 与 swift 版本相同的 Stage 3 延迟倍数抽取（源的 hash() 随进程 / Python 版本而变，见 integration/latency.py）
    if CUSTOM_DIR not in sys.path:
        sys.path.append(CUSTOM_DIR)
    from simuls2st_omni.integration.latency import install as install_stable_latency
    install_stable_latency(__import__(f'{pkg}.train.datasets.streaming', fromlist=['x']).StreamingS2STDataset)

    orig_apply = sp.apply_policy

    def apply_policy(model, policy):
        torch.manual_seed(a.seed)
        return orig_apply(model, policy)
    sp.apply_policy = apply_policy

    if a.ddp_forward:
        src = textwrap.dedent(inspect.getsource(mot.QwenOmniTalkerTrainer.compute_loss))
        old = 'base_model = model.module if hasattr(model, "module") else model'
        assert old in src
        ns = {}
        exec(src.replace(old, 'base_model = model'), mot.__dict__, ns)
        mot.QwenOmniTalkerTrainer.compute_loss = ns['compute_loss']

    from transformers import TrainingArguments
    orig_init = TrainingArguments.__init__

    def ta_init(self, *args, **kwargs):
        kwargs['optim'] = 'adamw_torch'
        kwargs['use_cpu'] = True
        if a.find_unused:
            kwargs['ddp_find_unused_parameters'] = True
        orig_init(self, *args, **kwargs)
    TrainingArguments.__init__ = ta_init

    trace = {'steps': [], 'log': [], 'checksum': {}, 'rank': int(os.environ.get('RANK', 0))}
    orig_cl = mot.QwenOmniTalkerTrainer.compute_loss

    def compute_loss(self, model, inputs, *args, **kwargs):
        if model.training:
            trace['steps'].append({'task': inputs.get('task_type'), 'ids': list(inputs.get('ids') or []),
                 'fp': {k: float(v.detach().double().sum()) for k, v in inputs.items()
                        if hasattr(v, 'detach') and k != 'labels'}})
            if os.environ.get('TRACE_DUMP_STEP') and len(trace['steps']) == int(os.environ['TRACE_DUMP_STEP']):
                import torch as _t; _t.save({k: v.detach().cpu() for k, v in inputs.items() if hasattr(v, 'detach')}, os.environ['TRACE_OUT_DUMP'] if 'TRACE_OUT_DUMP' in os.environ else (os.environ.get('TRACE_OUT', 'trace') + '.step.pt'))
        return orig_cl(self, model, inputs, *args, **kwargs)
    mot.QwenOmniTalkerTrainer.compute_loss = compute_loss


    # 每个训练步开始前记录可训参数指纹（定位分叉）
    import transformers as _tf
    _orig_ts = _tf.Trainer.training_step
    def _training_step(self, model, inputs, *args, **kwargs):
        m = self.accelerator.unwrap_model(model)
        trace.setdefault('lr_applied', []).append([g['lr'] for g in self.optimizer.param_groups] if self.optimizer else None)
        trace.setdefault('param_fp', []).append(
            [float(p.detach().double().sum()) for p in m.parameters() if p.requires_grad])
        return _orig_ts(self, model, inputs, *args, **kwargs)
    _tf.Trainer.training_step = _training_step


    # 记录每步梯度裁剪前的逐参数梯度指纹（定位分叉）
    import torch as _torch
    _orig_clip = _torch.nn.utils.clip_grad_norm_
    def _clip(parameters, *a, **k):
        params = list(parameters)
        trace.setdefault('grad_fp', []).append(
            [None if p.grad is None else float(p.grad.detach().double().sum()) for p in params])
        return _orig_clip(params, *a, **k)
    _torch.nn.utils.clip_grad_norm_ = _clip

    orig_train = mot.QwenOmniTalkerTrainer.train

    def train(self, *args, **kwargs):
        out = orig_train(self, *args, **kwargs)
        m = self.model.module if hasattr(self.model, 'module') else self.model
        for name in ('thinker', 'talker'):
            trace['checksum'][name] = {n: float(p.detach().cpu().double().sum())
                                       for n, p in getattr(m, name).named_parameters() if p.requires_grad}
        trace['log'] = self.state.log_history
        path = a.out if trace['rank'] == 0 else f'{a.out}.rank{trace["rank"]}'
        with open(path, 'w') as f:
            json.dump(trace, f, ensure_ascii=False)
        return out
    mot.QwenOmniTalkerTrainer.train = train

    sys.argv = ['train.py'] + train_args
    runpy.run_path(entry, run_name='__main__')


if __name__ == '__main__':
    main()

"""阶段配置 yaml 的 `training:` -> `swift sft` 命令行参数，规则与源 `build_training_args` 一一对应。

    python -m simuls2st_omni.integration.swift_args <stage.yaml>      # 每行一个参数，供 bash mapfile 读取

源 TrainingArguments 的取值：

* 只取 `supported` 集合里的键（与源同一份列表）；`min_lr_ratio` -> `lr_scheduler_kwargs.min_lr_rate`；
* `report_to=[]`、`dataloader_pin_memory=True`、`per_device_*_batch_size=1`（batch 由 sampler 决定）；
* 有 `eval_steps` 且存在验证集时 `eval_strategy=steps`，多任务（Stage 2）关闭评估；
* `ddp_find_unused_parameters` 源默认 False，这里 Stage 2/3 改为 True（见 register.py 模块说明）。

另外固定 swift 侧的接线参数（指针行读取方式、关掉 swift 自身的打乱 / 截断 / 梯度检查点等）。
`MAX_STEPS`、`NUM_TRAIN_EPOCHS`、`LEARNING_RATE`、`SAVE_STEPS`、`LOGGING_STEPS`、`EVAL_STEPS`、
`DATALOADER_NUM_WORKERS` 环境变量非空时覆盖 yaml（冒烟 / 试跑用）。
"""
from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, List

from .stage_config import is_multitask, load_stage_config, train_seed

# 源 build_training_args 的 supported 集合（逐项照抄）
SUPPORTED = (
    'num_train_epochs', 'max_steps', 'learning_rate', 'lr_scheduler_type', 'warmup_ratio', 'warmup_steps',
    'weight_decay', 'max_grad_norm', 'gradient_accumulation_steps', 'bf16', 'fp16', 'gradient_checkpointing',
    'logging_steps', 'save_steps', 'save_total_limit', 'eval_steps', 'dataloader_num_workers',
    'dataloader_pin_memory', 'seed', 'report_to', 'ddp_find_unused_parameters', 'ddp_timeout',
    'lr_scheduler_kwargs', 'metric_for_best_model', 'greater_is_better',
)
# 由 swift 侧另行处理、不能原样透传的键
_HANDLED_ELSEWHERE = {'gradient_checkpointing', 'bf16', 'fp16', 'report_to', 'seed'}
_ENV_OVERRIDES = {
    'MAX_STEPS': 'max_steps', 'NUM_TRAIN_EPOCHS': 'num_train_epochs', 'LEARNING_RATE': 'learning_rate',
    'SAVE_STEPS': 'save_steps', 'LOGGING_STEPS': 'logging_steps', 'EVAL_STEPS': 'eval_steps',
    'DATALOADER_NUM_WORKERS': 'dataloader_num_workers',
}
_DTYPES = {'bf16': 'bfloat16', 'fp16': 'float16', 'fp32': 'float32'}
# swift 改过默认值、而 yaml 未写时源会取 HF 默认值的字段（逐项对照 HF 4.57 TrainingArguments 默认值实测）。
# 其中 adam_beta2 最要紧：swift 默认 0.95，源（HF）为 0.999，不钉住时第 2 步起参数更新就不同。
_HF_DEFAULTS = {
    'learning_rate': 5e-05, 'weight_decay': 0.0, 'lr_scheduler_type': 'linear', 'logging_steps': 500,
    'dataloader_num_workers': 0, 'gradient_accumulation_steps': 1,
}
_HF_FIXED = ['--adam_beta1', '0.9', '--adam_beta2', '0.999', '--adam_epsilon', '1e-08',
             '--logging_first_step', 'false']


def _fmt(value: Any) -> str:
    if isinstance(value, bool):
        return 'true' if value else 'false'
    if isinstance(value, (dict, list)):
        return json.dumps(value)
    return str(value)


def training_kwargs(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """返回源 TrainingArguments 会拿到的 kwargs（再叠加环境变量覆盖）。"""
    train_cfg = dict(cfg['training'])
    for env, key in _ENV_OVERRIDES.items():
        if os.environ.get(env, '') != '':
            train_cfg[key] = os.environ[env]
            if key == 'max_steps' and int(os.environ[env]) > 0:
                train_cfg.pop('num_train_epochs', None)    # 源 --max-steps：去掉 epoch 数
    kwargs = {k: v for k, v in train_cfg.items() if k in SUPPORTED}
    for key, value in _HF_DEFAULTS.items():
        kwargs.setdefault(key, value)
    if 'min_lr_ratio' in train_cfg:
        kwargs.setdefault('lr_scheduler_kwargs', {})
        kwargs['lr_scheduler_kwargs']['min_lr_rate'] = float(train_cfg['min_lr_ratio'])
    kwargs.setdefault('dataloader_pin_memory', True)
    kwargs.setdefault('ddp_find_unused_parameters', cfg['stage'] != 'stage1')
    return kwargs


def build_args(cfg: Dict[str, Any], has_val: bool) -> List[str]:
    kw = training_kwargs(cfg)
    dtype = cfg['model'].get('dtype', 'bf16')
    args: List[str] = [
        # ---- 接线：指针行 = 完整 batch，按计划顺序读，不让 swift 改动 ----
        '--model_type', 'simuls2st_omni', '--template', 'simuls2st_omni', '--tuner_type', 'simuls2st_omni',
        '--dataset', 'simuls2st_omni',
        '--per_device_train_batch_size', '1', '--per_device_eval_batch_size', '1',
        '--dataset_shuffle', 'false', '--train_dataloader_shuffle', 'false',
        '--split_dataset_ratio', '0', '--lazy_tokenize', 'true', '--strict', 'true',
        '--remove_unused_columns', 'false', '--load_from_cache_file', 'false', '--dataset_num_proc', '1',
        '--packing', 'false', '--padding_free', 'false',
        # 梯度检查点由 tuner 按源做法开启（thinker / talker 分别、use_reentrant=False）；
        # swift 自己的开关会改写 decoder 层 forward，必须关
        '--gradient_checkpointing', 'false', '--gradient_checkpointing_kwargs', '{"use_reentrant": false}',
        # 源环境（torch 2.6）默认 adamw_torch；torch>=2.8 时 HF 会改选 fused 实现，固定回去
        '--optim', 'adamw_torch',
        '--torch_dtype', _DTYPES[dtype], '--attn_impl', 'sdpa',
        '--seed', str(train_seed(cfg)), '--report_to', 'none', '--add_version', 'false',
        '--save_only_model', 'false', '--check_model', 'false',
        *_HF_FIXED,
    ]
    if dtype == 'bf16':
        args += ['--bf16', 'true']
    elif dtype == 'fp16':
        args += ['--fp16', 'true']
    for key in SUPPORTED:
        if key in kw and key not in _HANDLED_ELSEWHERE:
            if key == 'eval_steps' and not has_val:
                continue
            args += [f'--{key}', _fmt(kw[key])]
    if has_val:
        args += ['--val_dataset', 'simuls2st_omni_val', '--eval_strategy', 'steps']
    else:
        args += ['--eval_strategy', 'no']
    return args


def has_validation(cfg: Dict[str, Any]) -> bool:
    return (not is_multitask(cfg)) and int(cfg['data'].get('eval_samples', 0)) > 0 and 'eval_steps' in cfg['training']


def resolved_paths(cfg: Dict[str, Any]) -> Dict[str, str]:
    """配置里的模型 / 输出 / 交接路径按数据根解析后的绝对路径（供 shell 脚本使用）。"""
    from ..train.native_train import resolve_path
    model = cfg['model']
    out = {
        'stage': cfg['stage'],
        'model_path': resolve_path(model.get('path', 'models/SimulS2ST-Omni/offline')),
        'output_dir': resolve_path(cfg['training'].get('output_dir', f'outputs/{cfg["stage"]}')),
        'init_talker_from': resolve_path(model['init_talker_from']) if model.get('init_talker_from') else '',
        'merge_adapters_from': resolve_path(model['merge_adapters_from']) if model.get('merge_adapters_from') else '',
    }
    return out


def main(argv: List[str]) -> None:
    """`swift_args <yaml>` 打印 swift 参数；`swift_args --paths <yaml>` 打印 key=value 形式的解析后路径。"""
    if argv and argv[0] == '--paths':
        for k, v in resolved_paths(load_stage_config(argv[1] if len(argv) > 1 else None)).items():
            print(f'{k}={v}')
        return
    cfg = load_stage_config(argv[0] if argv else None)
    for item in build_args(cfg, has_validation(cfg)):
        print(item)


if __name__ == '__main__':
    main(sys.argv[1:])

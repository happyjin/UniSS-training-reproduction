"""阶段配置：源项目 `configs/stage{1,2,3}.yaml` 的同一份格式，由 `SIMULS2ST_CONFIG` 指定。

配置里的相对路径（`models/...`、`data/...`、`outputs/...`）与源项目一样相对「数据根」解析，数据根取
`SIMULS2ST_ROOT`（缺省为本仓库根，见 `train/native_train.py:resolve_path`）。

swift 侧只读 `stage` / `model` / `data` / `sampler` 与 `training` 里影响样本与 loss 的几项
（seed、跳批阈值、gradient_checkpointing）；优化器、学习率、步数等由
`integration/swift_args.py` 按源 `build_training_args` 的同一套规则翻译成 swift 命令行参数。
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
from typing import Any, Dict

import yaml

CONFIG_ENV = 'SIMULS2ST_CONFIG'
# 覆盖配置里的阶段交接路径（集群上路径与源项目的 outputs/checkpoint/ 不同时用）
INIT_TALKER_ENV = 'SIMULS2ST_INIT_TALKER_FROM'
MERGE_ADAPTERS_ENV = 'SIMULS2ST_MERGE_ADAPTERS_FROM'

STAGES = ('stage1', 'stage2', 'stage3')


def config_path() -> str:
    return os.environ.get(CONFIG_ENV, '').strip()


def load_stage_config(path: str | None = None) -> Dict[str, Any]:
    """读取并校验阶段配置；返回深拷贝（下游的源代码会往 data_cfg 里写 `_effective_weights`）。"""
    path = path or config_path()
    if not path:
        raise RuntimeError(f'需要 {CONFIG_ENV}（阶段配置 yaml，见 pipelines/simuls2st_omni/configs/）')
    with open(path, encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict) or cfg.get('stage') not in STAGES:
        raise ValueError(f'{path}: `stage` 必须是 {STAGES} 之一，收到 {cfg.get("stage") if isinstance(cfg, dict) else cfg!r}')
    for key in ('model', 'data', 'sampler', 'training'):
        cfg.setdefault(key, {})
    model = cfg['model']
    if os.environ.get(INIT_TALKER_ENV):
        model['init_talker_from'] = os.environ[INIT_TALKER_ENV]
    if os.environ.get(MERGE_ADAPTERS_ENV):
        model['merge_adapters_from'] = os.environ[MERGE_ADAPTERS_ENV]
    if cfg['stage'] == 'stage3' and not model.get('merge_adapters_from'):
        raise ValueError('stage3 需要 model.merge_adapters_from（或环境变量 SIMULS2ST_MERGE_ADAPTERS_FROM）')
    cfg['_path'] = os.path.abspath(path)
    return copy.deepcopy(cfg)


def train_seed(cfg: Dict[str, Any]) -> int:
    """源 train.py：`seed = int(train_cfg.get("seed", 42))`，同一个值喂给 batch sampler 与 TrainingArguments。"""
    return int(cfg['training'].get('seed', 42))


def is_multitask(cfg: Dict[str, Any]) -> bool:
    return cfg['data'].get('kind', 'tts') == 'multitask'


def config_digest(cfg: Dict[str, Any], world_size: int) -> str:
    """采样计划的身份：数据 / 采样相关配置 + 数据根 + world_size。写进每个指针行，template 编码时核对。"""
    from ..train.native_train import REPO_ROOT
    payload = {
        'stage': cfg['stage'], 'data': cfg['data'], 'sampler': cfg['sampler'],
        'seed': train_seed(cfg), 'root': str(REPO_ROOT), 'world_size': int(world_size),
    }
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def world_size() -> int:
    """数据集加载时的 DDP 进程数（torchrun 注入 WORLD_SIZE；单进程为 1）。"""
    return max(1, int(os.environ.get('WORLD_SIZE', '1')))

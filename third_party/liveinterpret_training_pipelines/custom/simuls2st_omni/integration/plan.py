"""把源项目的 batch sampler 物化成「一行 = 一个 batch」的指针行。

源 `QwenOmniTalkerTrainer` 用 `DataLoader(batch_sampler=...)`，batch 的组成（token-budget 打包、Stage 2 按步抽任务）
与 DDP 分片都在 sampler 里完成。swift 的 dataloader 只接受 `per_device_train_batch_size` + 自己的
`BatchSamplerShard`，所以这里在加载阶段把源 sampler 为每个 rank 产出的 batch 序列算出来，排成

    row[s * W + r] = rank r 在第 s 步的 batch（W = world_size）

再让 swift 以 `per_device_train_batch_size=1`、`train_dataloader_shuffle=false` 读取：它的
`BatchSamplerShard` 按 `range(rank, N, W)` 分片，rank r 恰好按步序拿到自己的 batch，逐 batch 与源一致。

* token-budget（Stage 1 / Stage 3）：源每个 rank 先构造同一个全局 batch 列表（排序 + 以 seed 打乱，
  与 rank 无关），再取 `batches[r:usable:W]`。交错排列后正好是 `batches[:usable]`，算一次即可。
* 任务混合（Stage 2）：任务序列由共享种子决定、任务内样本按 rank 播种，各 rank 的 batch 池不同，
  逐 rank 用源 `TaskMixBatchSampler(rank=r)` 迭代一遍再交错。
* epoch：源 DataLoader 不经 accelerate 包装，sampler 从未收到 `set_epoch`，每个 epoch 的 batch 顺序相同；
  swift 关掉打乱后同样逐 epoch 重复同一序列。

源采样器的一个行为照原样保留：`build_sampler(dataset, indices=...)` 按子集算长度，但产出的是子集内位置，
DataLoader 直接拿位置当数据集下标。Stage 1 / Stage 3 的验证集（尾部 `eval_samples` 条）因此实际评的是
前 `eval_samples` 条（训练样本）；多源 TTS（`sources:`）时训练集同样错位。设
`SIMULS2ST_MAP_SUBSET_INDICES=1` 改为按真实下标取样（与源不再逐样本一致）。
"""
from __future__ import annotations

import hashlib
import logging
import os
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import numpy as np

from .stage_config import config_digest, is_multitask, train_seed, world_size

logger = logging.getLogger(__name__)

MAP_SUBSET_ENV = 'SIMULS2ST_MAP_SUBSET_INDICES'

# 进程内缓存：swift 的数据集 loader 与 template 在同一进程，template 编码时复用这里已构造的
# 源 dataset 对象（Stage 1 全量 3810 万条，只构造一次；dataloader worker fork 后共享）。
_STAGES: Dict[str, 'StagePlan'] = {}


class StagePlan(SimpleNamespace):
    """dataset：源 dataset（多任务时为 MultiTaskConcatDataset）；train_rows / eval_rows：指针行（下标列表）。"""


def _map_subset() -> bool:
    return os.environ.get(MAP_SUBSET_ENV, '0') == '1'


def _budget_batches(dataset, indices, sampler_cfg: dict, seed: int, shuffle: bool, world: int) -> List[List[int]]:
    """源 `build_sampler` + `TokenBudgetBatchSampler`，返回交错后的全局 batch 序列（长度为 world 的整数倍）。"""
    from ..train.samplers import TokenBudgetBatchSampler
    indices = list(indices)
    lengths = [dataset.code_length(i) for i in indices]
    sampler = TokenBudgetBatchSampler(
        lengths=lengths,
        token_budget=int(sampler_cfg.get('token_budget', 8192)),
        max_batch_size=int(sampler_cfg.get('max_batch_size', 32)),
        shuffle=shuffle,
        seed=seed,
        rank=0,
        world_size=1,
    )
    batches = sampler._build_batches()
    usable = (len(batches) // world) * world
    batches = batches[:usable]
    if _map_subset():
        batches = [[indices[p] for p in b] for b in batches]
    return batches


def _task_mix_batches(dataset, data_cfg: dict, sampler_cfg: dict, seed: int, world: int) -> List[List[int]]:
    """源 `build_multitask_sampler`，逐 rank 迭代后按步交错。"""
    from ..train.native_train import build_multitask_sampler
    per_rank = []
    saved = {k: os.environ.get(k) for k in ('RANK', 'WORLD_SIZE')}
    try:
        for rank in range(world):
            # 源函数从环境变量读 rank / world_size；这里临时设定以复用它（构造时只读这两项）
            os.environ['RANK'], os.environ['WORLD_SIZE'] = str(rank), str(world)
            sampler = build_multitask_sampler(dataset, data_cfg, sampler_cfg, seed)
            per_rank.append(list(sampler))
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    steps = {len(b) for b in per_rank}
    assert len(steps) == 1, f'各 rank 步数不同: {steps}'
    return [per_rank[r][s] for s in range(steps.pop()) for r in range(world)]


def build_stage(cfg: Dict[str, Any], tokenizer=None) -> StagePlan:
    """按源 `train.py:main` 的同一顺序构造 dataset 与 train / eval batch 序列（结果按配置缓存）。"""
    world = world_size()
    digest = config_digest(cfg, world)
    cached = _STAGES.get(digest)
    if cached is not None:
        if tokenizer is not None:
            _attach_tokenizer(cached, tokenizer)
        return cached

    from ..train.native_train import MultiTaskConcatDataset, build_dataset
    data_cfg, sampler_cfg = cfg['data'], cfg['sampler']
    seed = train_seed(cfg)
    built = build_dataset(data_cfg, processor=SimpleNamespace(tokenizer=tokenizer))
    if isinstance(built, dict):
        dataset = MultiTaskConcatDataset(built)
        logger.info('multitask dataset: %d samples across %s', len(dataset), {t: len(built[t]) for t in dataset.tasks})
        train_rows = _task_mix_batches(dataset, data_cfg, sampler_cfg, seed, world)
        eval_rows: List[List[int]] = []   # 源：多任务不留验证集（_held_out 从未设置）
    else:
        dataset = built
        eval_n = int(data_cfg.get('eval_samples', 0))
        if eval_n and hasattr(dataset, 'train_eval_split'):
            train_indices, eval_indices = dataset.train_eval_split(eval_n)
        elif eval_n and eval_n < len(dataset):
            eval_indices = list(range(len(dataset) - eval_n, len(dataset)))
            train_indices = list(range(len(dataset) - eval_n))
        else:
            eval_indices, train_indices = [], list(range(len(dataset)))
        train_rows = _budget_batches(dataset, train_indices, sampler_cfg, seed, True, world)
        eval_rows = _budget_batches(dataset, eval_indices, sampler_cfg, seed, False, world) if eval_indices else []
    h = hashlib.sha256(digest.encode())
    for rows in (train_rows, eval_rows):
        h.update(np.asarray([len(rows)] + [x for b in rows[:2048] for x in b], dtype=np.int64).tobytes())
    plan = StagePlan(cfg=cfg, dataset=dataset, train_rows=train_rows, eval_rows=eval_rows,
                     world_size=world, digest=h.hexdigest()[:16], multitask=is_multitask(cfg))
    logger.info('[simuls2st_omni] %s plan: world=%d train batches=%d (%d/rank) eval batches=%d digest=%s',
                cfg['stage'], world, len(train_rows), len(train_rows) // world, len(eval_rows), plan.digest)
    _STAGES[digest] = plan
    if tokenizer is not None:
        _attach_tokenizer(plan, tokenizer)
    return plan


def _attach_tokenizer(plan: StagePlan, tokenizer) -> None:
    """Stage 3 的 StreamingS2STDataset 在 __getitem__ 里分词；加载阶段拿不到 tokenizer 时先空着，编码前补上。"""
    ds = plan.dataset
    if getattr(ds, 'tokenizer', None) is None and hasattr(ds, 'tokenizer'):
        ds.tokenizer = tokenizer


def cached_plan() -> Optional[StagePlan]:
    if len(_STAGES) == 1:
        return next(iter(_STAGES.values()))
    return None

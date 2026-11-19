# Copyright (c) ModelScope Contributors. All rights reserved.
"""消除 ms-swift 数据加载链路上不必要的 ``Flattening the indices`` 全量重写。

背景（unist 1899 万条数据集实测，8×H200 单节点）
--------------------------------------------------
``swift/dataset/loader.py`` 在 ``load_dataset()`` 里给每个数据集打 routing tag::

    def _inject_dataset_routing_tag(dataset, ds_name):
        return dataset.add_column('dataset', [ds_name] * len(dataset))

而 ``datasets.Dataset.add_column``（datasets 3.6.0, arrow_dataset.py:5803）是::

    dataset = self.flatten_indices() if self._indices is not None else self
    table = concat_tables([dataset._data, column_table], axis=1)
    return Dataset(table, ..., indices_table=None, fingerprint=new_fingerprint)

三个事实叠加导致这一步成为整条链路最慢的环节：

1. ``--split_dataset_ratio`` 非 0 时，``DatasetLoader.post_process`` 会走
   ``train_test_split(shuffle=True)``，返回的 train/val **必然带 ``_indices``**
   （乱序索引映射，底层 Arrow 表并未被重排）。于是 ``add_column`` 必定触发
   ``flatten_indices()``。
2. 该 ``flatten_indices()`` **没有传 ``num_proc``** —— 它内部就是一次
   ``map(batched=True)``，单进程把整张表按 indices 顺序重写一遍。unist 数据集
   1899 万行、带音频 token 列，实测约 2600 examples/s，**单 rank 耗时约 2 小时**。
3. 这段代码位于 ``loader.py`` 中 ``shuffle_dataset()`` 的 ``safe_ddp_context``
   **之前**，没有任何 DDP 串行保护 —— 8 个 rank 各自完整跑一遍，写 8 份内容
   相同的临时文件再互相覆盖同一个 cache 文件（实测缓存目录膨胀到 784G）。

最关键的一点是：**这次重写的产出在下一步就被作废**。``add_column`` 之后
``loader.py`` 紧接着调用 ``shuffle_dataset()`` → ``dataset.shuffle(seed=...)``，
又给数据集套上一层新的 indices mapping。也就是说不管这里有没有 flatten，
交给训练的最终形态都是"底层 Arrow 表 + 一层 indices mapping"，随机访问性能
完全一致。这 2 小时（×8 rank）纯属白烧。

本 patch 的做法
----------------
注入的是**常量列**（整列取同一个值 ``ds_name``），它与行序无关，因此完全不需要
先把底层表按 indices 重排：

* 原实现：``flatten(D)`` 得到按 indices 物化的表 ``T'``，再给 ``T'`` 横向拼常量列。
* 本实现：直接给底层表 ``T`` 横向拼常量列得 ``T2``，并**原样保留 indices**。

读第 i 行时，原实现拿到 ``(T[indices[i]], ds_name)``；本实现拿到
``(T2[indices[i]])`` 即 ``(T[indices[i]], ds_name)`` —— 逐行严格相同。
差别只是省掉了整表重写，复杂度从 O(全表字节数) 降到 O(行数) 的一次常量数组构造
（这个数组原实现里的 ``[ds_name] * len(dataset)`` 本来也要建，开销持平）。

兜底：并行 flatten
-------------------
主 patch 让这条链路不再触发 flatten。但若上游版本变动、或别处仍触发
``flatten_indices()``，``install_parallel_flatten_indices_patch()`` 会给它补上
默认的 ``num_proc``，用容器的多核并行重写，避免再次退化成单进程 2 小时。
默认并行度按"整机核数 / 本机 rank 数"分配，避免 8 个 rank 各开满导致
进程数远超核数而互相抢占（可用 ``SWIFT_FLATTEN_NUM_PROC`` 覆盖，设 0 关闭）。

关于"按 DP 分片只处理自己那份"
--------------------------------
在这一层做不到，也不该做：``add_column`` / ``shuffle`` / ``train_test_split``
以及 megatron 的 ``global_batch_size`` 换算都依赖**完整数据集**的长度与顺序，
提前按 rank 切分会让 split 比例、shuffle 种子语义和 step 数全部算错。DP 分片
是训练时由 sampler 按 rank 取模完成的。这里真正要消灭的是"8 个 rank 重复做同
一件全量重活"，而本 patch 直接把这件重活本身消掉了，重复与否已无关紧要。
"""
from __future__ import annotations

import os
from typing import Any, Optional

try:
    from swift.utils import get_logger

    logger = get_logger()
except Exception:  # pragma: no cover - swift 不可用时退化成标准 logging
    import logging

    logger = logging.getLogger(__name__)

__all__ = [
    'install_dataset_routing_tag_fast_patch',
    'install_parallel_flatten_indices_patch',
]

_ROUTING_TAG_PATCHED = False
_FLATTEN_PATCHED = False


# ---------------------------------------------------------------------------
# 主 patch：常量列直接挂到底层表，跳过 flatten
# ---------------------------------------------------------------------------
def _add_constant_column_keep_indices(dataset, name: str, value: Any):
    """给 ``dataset`` 追加一个常量列，**不触发** ``flatten_indices()``。

    实现照抄 ``datasets.Dataset.add_column``（3.6.0）的后半段，唯二的区别：

    * 不做 ``self.flatten_indices()``，常量列直接与 ``self._data`` 横向拼接；
    * ``indices_table`` 传原 ``self._indices`` 而不是 ``None``。

    注意常量数组的长度必须取**底层表行数** ``_data.num_rows``，而不是
    ``len(dataset)``（后者是 indices 的行数，两者在 split 之后并不相等）。
    """
    import pyarrow as pa
    from datasets import Dataset as HfDataset
    from datasets.arrow_dataset import _check_column_names, update_metadata_with_features
    from datasets.features import Features
    from datasets.fingerprint import update_fingerprint
    from datasets.table import InMemoryTable, concat_tables

    base_table = dataset._data
    num_base_rows = base_table.num_rows

    column_table = InMemoryTable.from_pydict({name: pa.array([value] * num_base_rows)})
    _check_column_names(base_table.column_names + column_table.column_names)

    table = concat_tables([base_table, column_table], axis=1)
    info = dataset.info.copy()
    info.features.update(Features.from_arrow_schema(column_table.schema))
    table = update_metadata_with_features(table, info.features)

    new_fingerprint = update_fingerprint(
        dataset._fingerprint,
        'custom.dataset_routing_tag_patch._add_constant_column_keep_indices',
        {
            'name': name,
            'value': value
        },
    )
    return HfDataset(
        table,
        info=info,
        split=dataset.split,
        indices_table=dataset._indices,  # ← 关键：保留 indices，省掉整表重写
        fingerprint=new_fingerprint,
    )


def install_dataset_routing_tag_fast_patch() -> None:
    """把 ``swift.dataset.loader._inject_dataset_routing_tag`` 换成零重写版本。

    ``load_dataset`` 是在同一模块内按名字调用该函数的，Python 的模块内调用走
    module ``globals()`` 查找，因此替换模块属性即可对既有调用生效。
    """
    global _ROUTING_TAG_PATCHED
    if _ROUTING_TAG_PATCHED:
        return

    try:
        from datasets import Dataset as HfDataset
        from datasets import IterableDataset as HfIterableDataset
        from swift.dataset import loader as swift_dataset_loader
    except Exception as e:  # pragma: no cover
        logger.warning(f'Skip dataset routing-tag fast patch, import failed: {e}')
        return

    original = getattr(swift_dataset_loader, '_inject_dataset_routing_tag', None)
    if original is None:  # pragma: no cover - 上游重构后函数改名
        logger.warning('Skip dataset routing-tag fast patch: '
                       'swift.dataset.loader._inject_dataset_routing_tag not found.')
        return

    def _fast_inject_dataset_routing_tag(dataset, ds_name: str):
        # IterableDataset 走 lazy map，本来就没有重写开销，保持原逻辑。
        if isinstance(dataset, HfIterableDataset):
            return original(dataset, ds_name)
        if not isinstance(dataset, HfDataset):  # pragma: no cover - 防御未知类型
            return original(dataset, ds_name)
        # 没有 indices mapping 时，原生 add_column 本身就是 O(1) 的横向拼接。
        if dataset._indices is None:
            return original(dataset, ds_name)
        if 'dataset' in dataset.column_names:  # pragma: no cover - 上游已注入过
            return dataset

        num_rows = len(dataset)
        logger.info(f'[routing-tag] inject constant column dataset={ds_name!r} '
                    f'on {num_rows} rows without flatten_indices '
                    f'(saves a full-table rewrite).')
        return _add_constant_column_keep_indices(dataset, 'dataset', ds_name)

    swift_dataset_loader._inject_dataset_routing_tag = _fast_inject_dataset_routing_tag
    _ROUTING_TAG_PATCHED = True
    logger.info('Installed dataset routing-tag fast patch '
                '(skip Flattening the indices in add_column).')


# ---------------------------------------------------------------------------
# 兜底 patch：真要 flatten 时用多核并行
# ---------------------------------------------------------------------------
def _default_flatten_num_proc() -> int:
    """按"整机可用核数 / 本机 rank 数"分配并行度。

    同一节点上 8 个 rank 会同时进入数据准备阶段，若每个 rank 都开满核数，
    进程数会数倍于物理核并互相抢占，反而更慢。
    """
    env = os.environ.get('SWIFT_FLATTEN_NUM_PROC')
    if env is not None:
        try:
            return max(0, int(env))
        except ValueError:
            logger.warning(f'Invalid SWIFT_FLATTEN_NUM_PROC={env!r}, fallback to auto.')

    try:
        cpu_count = len(os.sched_getaffinity(0))
    except AttributeError:  # pragma: no cover - 非 Linux
        cpu_count = os.cpu_count() or 1

    local_world_size = 1
    for key in ('LOCAL_WORLD_SIZE', 'NPROC_PER_NODE', 'GPUS_PER_NODE'):
        val = os.environ.get(key)
        if val and val.isdigit() and int(val) > 0:
            local_world_size = int(val)
            break

    return max(1, min(32, cpu_count // local_world_size))


def install_parallel_flatten_indices_patch() -> None:
    """给 ``Dataset.flatten_indices()`` 补上默认 ``num_proc``。

    仅在调用方**没有显式指定** ``num_proc`` 时生效；``num_proc<=1`` 时不做任何事。
    """
    global _FLATTEN_PATCHED
    if _FLATTEN_PATCHED:
        return

    num_proc = _default_flatten_num_proc()
    if num_proc <= 1:
        logger.info(f'Skip parallel flatten_indices patch (num_proc={num_proc}).')
        return

    try:
        from datasets import Dataset as HfDataset
    except Exception as e:  # pragma: no cover
        logger.warning(f'Skip parallel flatten_indices patch, import failed: {e}')
        return

    original_flatten_indices = HfDataset.flatten_indices

    def _flatten_indices(self, *args, **kwargs):
        # 位置参数最多到 disable_nullable（第 5 个），num_proc 只可能以关键字传入。
        if kwargs.get('num_proc') is None and len(args) < 6:
            kwargs['num_proc'] = num_proc
            logger.info(f'[flatten_indices] no num_proc given, use num_proc={num_proc}.')
        return original_flatten_indices(self, *args, **kwargs)

    HfDataset.flatten_indices = _flatten_indices
    _FLATTEN_PATCHED = True
    logger.info(f'Installed parallel flatten_indices patch (default num_proc={num_proc}).')


def install_dataset_loading_speedup_patches() -> None:
    """一次性安装本模块的全部 patch。"""
    install_dataset_routing_tag_fast_patch()
    install_parallel_flatten_indices_patch()

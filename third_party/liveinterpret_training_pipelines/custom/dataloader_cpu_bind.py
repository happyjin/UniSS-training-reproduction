# Copyright (c) ModelScope Contributors. All rights reserved.
"""给 fork 出的 dataloader worker 进程绑定 CPU 亲和。

与 pipelines/bridge_alignment/common_args.sh 的 DATALOADER_NUM_WORKERS 动态取值配套：

* 每节点可用 CPU 核按 LOCAL_WORLD_SIZE 均分，每 rank 占一段
  [rank_base, rank_base + cores_per_rank)。
* 一个 rank 内有 train(NW) + val(NW) 个常驻 worker（persistent_workers=true 下
  train 与 val worker 在训练前就同时 fork、常驻到训练结束），共 2*NW 个。
  动态取值 NW = floor(CPU / (2*LWS)) 保证 2*NW <= 每 rank 核数，正好落在段内。
* train worker i    绑 allowed[rank_base + i]
  val   worker j    绑 allowed[rank_base + NW + j]
  即 train 占段内前 NW 个核，val 占后 NW 个核，互不重叠。
* 每个 worker 绑独立 CPU（仅 1 个核的掩码，非继承父进程全量核），避免跨节点越界
  （base 偏移用节点内 rank 序号，不用 process_index）。
* 必须链式包裹 ms-swift 自带的 worker_init_fn（seed_worker），保留其随机种子逻辑，
  不能跳过。

ray 模式适配（2026-08-28，stall 根因修复）：
  swift RayHelper 模式下每个 actor 的 LOCAL_RANK=0、LOCAL_WORLD_SIZE 缺省=1
  （见 anyscale/deadlock_analysis.md），若直接读 LOCAL_* 会让同节点全部 actor 的
  worker 绑到同一段 cpu0..NW-1（8 actor × 16 worker 挤 16 核、fbank 排队 → 周期性
  数据断供 stall，周期=num_workers）。ray 下改为从 ray_entrypoint 注入的
  RAY_NPROC_PER_NODE 派生节点内序号：local_rank = RANK % RAY_NPROC_PER_NODE、
  local_world = RAY_NPROC_PER_NODE，与 torchrun 的核段分摊对齐。

生效方式：通过 --external_plugins 在 swift 训练进程内、DataLoader 构建前 import 本模块
（plugin.py 已挂载）。实际绑定动作受 env DATALOADER_CPU_BIND 控制，默认开启（设 "0" 关闭）。

Lance / pylance 不是 fork-safe（官方警告：用 spawn 或 forkserver）。
F 列 2-node torchrun 在默认 fork 下 DataLoader worker SIGSEGV。
env DATALOADER_MP_CONTEXT=fork|spawn|forkserver 在 num_workers>0 且调用方
未显式传 multiprocessing_context 时注入。默认不设（PyTorch fork）。
DeepSpeed 下 forkserver/spawn 会因 hook 无法 pickle 失败。
forkserver/spawn 要求 worker_init_fn 可 pickle，因此用顶层 _WorkerInit。
"""
import os

import torch.utils.data as tud

_ALLOWED_MP_CONTEXT = frozenset({"fork", "spawn", "forkserver"})
_MP_CONTEXT_LOGGED = False

# pid -> rank 内第几个 DataLoader（train=0, val=1）。
# _prepare_dataloader 先建 train 后建 val，故 dl_idx 顺序固定。
_DL_INDEX = {}


def _node_local_placement() -> tuple:
    """返回 (local_rank, local_world)：本进程在**节点内**的序号与节点内进程数。

    ray 模式（swift RayHelper）：每个 actor 的 LOCAL_RANK 恒 0、无 LOCAL_WORLD_SIZE，
    节点内序号必须由 ray_entrypoint.py 注入的 RAY_NPROC_PER_NODE + 全局 RANK 派生。
    torchrun/单机：直接用 LOCAL_RANK / LOCAL_WORLD_SIZE（缺省回退 NPROC_PER_NODE）。
    """
    ray_nproc = os.environ.get("RAY_NPROC_PER_NODE")
    if ray_nproc:
        try:
            local_world = max(1, int(ray_nproc))
        except ValueError:
            local_world = 1
        try:
            rank = int(os.environ.get("RANK", "0"))
        except ValueError:
            rank = 0
        return rank % local_world, local_world
    try:
        local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))
    except ValueError:
        local_rank = 0
    try:
        local_world = int(os.environ.get("LOCAL_WORLD_SIZE", os.environ.get("NPROC_PER_NODE", "1")))
    except ValueError:
        local_world = 1
    return local_rank, max(1, local_world)


def _bind_cpu(worker_id: int, dataloader_index: int, nw: int) -> None:
    allowed = sorted(os.sched_getaffinity(0))
    if not allowed or nw < 1:
        return
    local_rank, local_world = _node_local_placement()
    cores_per_rank = max(1, len(allowed) // local_world)
    rank_base = local_rank * cores_per_rank
    rank_cpus = allowed[rank_base: rank_base + cores_per_rank]
    if not rank_cpus:
        rank_cpus = allowed
    # dataloader_index: 0=train, 1=val；train 占前 NW 个核，val 占后 NW 个核。
    # nw 为当前 DataLoader 实例的真实 num_workers（非 env 重读），不强制下限：
    # nw=0 时根本无 worker、_bind_cpu 不会被调用。
    idx = dataloader_index * nw + worker_id
    cpu = rank_cpus[idx % len(rank_cpus)]
    os.sched_setaffinity(0, {cpu})


class _WorkerInit:
    """Picklable worker_init_fn（forkserver/spawn 必须能 pickle；闭包不行）。"""

    def __init__(self, orig_wfn, dataloader_index, nw):
        self.orig_wfn = orig_wfn
        self.dataloader_index = dataloader_index
        self.nw = nw

    def __call__(self, worker_id):
        # Always drop inherited pylance/Arrow state. DeepSpeed cannot use
        # spawn/forkserver (unpickleable engine hooks), so workers stay on
        # fork and must reopen Lance in this process.
        try:
            try:
                from custom.wds_audio import reset_native_io_after_fork
            except ImportError:
                from wds_audio import reset_native_io_after_fork
            reset_native_io_after_fork()
        except Exception:
            pass
        if self.orig_wfn is not None:
            self.orig_wfn(worker_id)
        if os.environ.get("DATALOADER_CPU_BIND", "1") != "0":
            _bind_cpu(worker_id, self.dataloader_index, self.nw)


def _resolve_mp_context():
    raw = os.environ.get("DATALOADER_MP_CONTEXT", "").strip().lower()
    if raw in _ALLOWED_MP_CONTEXT:
        return raw
    return None


_orig_init = tud.DataLoader.__init__


def _patched_init(self, *args, **kwargs):
    orig_wfn = kwargs.get("worker_init_fn")
    # num_workers=0（合入训练进程）时不 fork worker：worker_init_fn 不会被调用、
    # 绑定自然不触发，也不参与 dl_idx 计数（避免浪费/错位 CPU 段）。
    nw = kwargs.get("num_workers", 0)
    if not isinstance(nw, int):
        try:
            nw = int(nw)
        except (TypeError, ValueError):
            nw = 0
    if nw and nw > 0:
        pid = os.getpid()
        dl_idx = _DL_INDEX.get(pid, 0)
        _DL_INDEX[pid] = dl_idx + 1
        # 用当前 DataLoader 实例真实的 nw 计算核偏移，不再重读 env、不再强制下限。
        kwargs["worker_init_fn"] = _WorkerInit(orig_wfn, dl_idx, nw)
        ctx = _resolve_mp_context()
        if ctx is not None and kwargs.get("multiprocessing_context") is None:
            kwargs["multiprocessing_context"] = ctx
            global _MP_CONTEXT_LOGGED
            if not _MP_CONTEXT_LOGGED:
                print(
                    f"[dataloader_cpu_bind] multiprocessing_context={ctx} "
                    f"num_workers={nw}",
                    flush=True,
                )
                _MP_CONTEXT_LOGGED = True
    _orig_init(self, *args, **kwargs)


tud.DataLoader.__init__ = _patched_init

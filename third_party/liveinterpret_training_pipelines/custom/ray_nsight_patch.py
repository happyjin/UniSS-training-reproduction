# Copyright (c) junyaoyang. All rights reserved.
"""Attach nsys (Nsight Systems) to ONE ray GPU *actor* worker process (by
default rank 0), for profiling ray (``USE_RAY=true``) multi-node training.

Background / root cause (verified, not guesswork — see history below)
-----------------------------------------------------------------------
The actual training step (forward/backward/optimizer, i.e. everything
``nsys_step_profile.py``'s ``cudaProfilerStart/Stop`` brackets) runs inside
a **ray actor** process, created by ``swift.ray_utils.base.RayHelper.
_create_workers()`` via ``worker_cls.options(runtime_env=...).remote(...)``.
The driver itself never calls ``cudaProfilerStart`` — it just waits for the
ray cluster and dispatches, so wrapping the *driver* with nsys is a no-op
for capturing training kernels (and can be actively harmful — see
``ktrain/launch_train.sh``'s ``_exec_train`` comment for the
``ActorUnschedulableError`` ptrace interaction).

**Critical timing constraint** (confirmed via ``anyscale/deadlock_analysis.
md`` §2.1 call-order trace + a live crash traceback): ``SwiftSft`` is
decorated with ``@RayHelper.worker(group=['default'])``, which replaces
``__init__`` with a ``new_init`` that does::

    def new_init(self, *args, **kwargs):
        if not RayHelper.is_worker():
            RayHelper._create_workers(group, *args, **kwargs)  # (1) actors submitted to Ray HERE
        init_method(self, *args, **kwargs)                      # (2) original __init__ runs AFTER —
                                                                  #     THIS is where --external_plugins
                                                                  #     (custom/plugin.py) gets loaded,
                                                                  #     via _parse_args() ->
                                                                  #     _import_external_plugins().

So by the time ``custom/plugin.py`` loads and could patch anything, step
(1) has **already fully executed** — all 16 actors have already been
submitted to Ray via the pristine, unpatched ``ray.actor.ActorClass.
options``. This was empirically re-confirmed three times (three different
monkeypatch strategies — rebinding the ``ray.runtime_env.RuntimeEnv``
module attribute, rewriting ``RuntimeEnv.__init__`` in place, wrapping
``RayHelper._create_workers`` itself — all installed from
``custom/plugin.py``): every one of them was verified, via
``<ray_session>/logs/runtime_env_agent.log`` inspection and ``/proc/<pid>/
status`` parent-chain tracing, to have **zero effect** on the real training
actors (father process was always ``raylet``, never ``nsys-launcher``).
The exact same "patch ``ray.actor.ActorClass.options`` at the class level"
logic, verified via an isolated ``python3 -c`` experiment in the same
container, works **immediately** once applied *before*
``ray.remote(SwiftSft)`` is ever called (Python attribute lookup walks the
MRO at call time, so a dynamically-generated ``ActorClass`` subclass
correctly inherits a patch applied to its base class *before* the subclass
was created).

Fix
---
Patch ``ray.actor.ActorClass.options`` — but from ``ktrain/nsys_early_hook/
sitecustomize.py`` (auto-imported by CPython's ``site`` module at
**interpreter startup**, before ``swift.cli.sft`` or anything else is
imported), not from ``custom/plugin.py``. This guarantees the patch is in
place before ``import swift.pipelines.train.sft`` ever runs
``ray.remote(SwiftSft)`` to build the ``ActorClass`` subclass, which in
turn is long before ``RayHelper._create_workers()`` calls ``.options()`` on
it. See ``ktrain/launch_train.sh`` / ``ktrain/launch_train_lance.sh``'s
``PYTHONPATH`` export for how that directory gets onto ``sys.path``.

Only ONE actor (default: the one whose ``RANK`` env var equals
``NSYS_TARGET_RANK``, default ``0``) is nsys-wrapped, not all 16 — a
single rank's kernel/NCCL/idle timeline is sufficient for
``nsys-analyze``-style diagnosis, and wrapping fewer processes means less
overhead and a single, much smaller ``.nsys-rep`` to download.

**Auto-stop caveat (root-caused, not guesswork — two separate bugs found
and fixed on 2026-09-11)**: we do NOT rely on nsys's
``capture-range-end=stop-shutdown`` to kill the wrapped process once the
capture window closes. Two independent issues were involved:

1. ``nsys_step_profile.py``'s ``_start()``/``_stop()`` originally only
   tried ``from cuda import cudart; cudart.cudaProfilerStart/Stop()``
   (the ``cuda-python`` pip package), which is **not installed** in the
   training image — the resulting ``ModuleNotFoundError`` was silently
   swallowed by a bare ``except Exception: pass``, so nsys never actually
   received a Start/Stop signal at all (capture range never opened —
   explains an empty ``logs/nsight/`` dir with zero files, not even a
   partial trace). Fixed by switching to ``torch.cuda.profiler.start/
   stop()`` (same underlying ``libcudart`` symbol, available in this
   image's torch build without extra deps).
2. Once (1) was fixed and ``cudaProfilerStop()`` actually reached nsys,
   nsys's **default** ``capture-range-end`` (confirmed via ``nsys profile
   --help``: default really is ``stop-shutdown``, *not* ``stop`` as
   originally assumed here) kicked in and tried to kill the wrapped
   process — which made the ``cudaProfilerStop()`` call itself return a
   non-zero CUDA error code, and PyTorch's ``torch.cuda.profiler.stop()``
   `check_error()` turns any non-zero code into a bare ``SystemExit``
   (not caught by ``except Exception`` — had to widen those catches to
   ``BaseException`` too as defense-in-depth), crashing the whole ray
   actor on the spot (reproduced: job died at step 30 with
   ``SystemExit: 1`` propagating out of ``_stop()``). Fixed by explicitly
   setting ``'capture-range-end': 'stop'`` below — finalizes/flushes the
   ``.nsys-rep`` to disk right when ``cudaProfilerStop()`` fires, without
   touching the process, so the call returns cleanly and training
   continues. ``nsys_step_profile.py``'s ``_copy_ray_nsight_report()``
   callback then immediately copies that file out of ray's ephemeral
   session dir onto ``NSYS_OUTPUT_DIR`` — no need to wait for (or force)
   the process to exit. To actually end the job once the report is
   confirmed copied, just ``ktrain job stop`` manually.

The actual start/stop of the capture window is still driven by
``nsys_step_profile.py``'s HF ``TrainerCallback`` (installed on every
actor; harmless no-op CUDA API calls on the 15 non-wrapped ranks, since no
nsys process is attached to intercept them there).

Env:
  NSYS_PROFILE_STEP=<n>   same switch as ``nsys_step_profile.py``; 0/unset
                          -> this patch is a no-op (zero overhead).
  USE_RAY=true|1          only patch when the ray backend is actually in use.
  NSYS_TARGET_RANK=<r>    which single global RANK to nsys-wrap (default 0).
  NSYS_OUTPUT_DIR         cosmetic only (log message); actual nsys output
                          for the wrapped actor lands under Ray's own
                          session dir (``<ray_session>/logs/nsight/``), not
                          under ``NSYS_OUTPUT_DIR`` -- see the printed path
                          hint in the container log to find it.
"""
from __future__ import annotations

import os

_INSTALLED = False


def _target_step() -> int:
    try:
        return max(0, int(os.environ.get('NSYS_PROFILE_STEP', '0')))
    except ValueError:
        return 0


def _target_rank() -> int:
    try:
        return int(os.environ.get('NSYS_TARGET_RANK', '0'))
    except ValueError:
        return 0


def _use_ray() -> bool:
    return os.environ.get('USE_RAY', 'false').strip().lower() in ('1', 'true')


def install_ray_nsight_patch() -> None:
    """Patch ``ray.actor.ActorClass.options`` in place.

    MUST be called before ``ray.remote(SwiftSft)`` runs (i.e. from
    ``ktrain/nsys_early_hook/sitecustomize.py``, not from
    ``custom/plugin.py`` — see module docstring for why the timing
    matters). Idempotent / safe to call multiple times.
    """
    global _INSTALLED
    if _INSTALLED:
        return
    step = _target_step()
    if step <= 0 or not _use_ray():
        # No-op path: not a ray run, or NSYS_PROFILE_STEP unset -> nothing
        # patched, zero overhead, no ray import even attempted.
        return

    try:
        from ray.actor import ActorClass as _ActorClassCls
    except ImportError:
        return

    if getattr(_ActorClassCls.options, '_ray_nsight_patched', False):
        _INSTALLED = True
        return

    target_rank = _target_rank()

    # nsys option dict, translated by ray's own nsight plugin:
    #   single-char key  -> "-k value" (space separated)
    #   multi-char key   -> "--key=value"
    # (mirrors ray._private.runtime_env.nsight.parse_nsight_config)
    _nsight_cfg = {
        't': 'cuda,nvtx',
        'o': f"'step{step}_worker_%p'",
        'capture-range': 'cudaProfilerApi',
        # 2026-09-11 更正（之前的注释猜错了 nsys 的默认值）：
        # `nsys profile --help` 明确写着 --capture-range-end 的 Default
        # 是 'stop-shutdown'，不是 'stop'！所以早前"干脆不设置这个 key"
        # 的做法其实等价于显式写 stop-shutdown，根本没改变行为——那一轮
        # "报告拿不到"的真正原因另有其人：`nsys_step_profile.py` 的
        # `_start()/_stop()` 当时只用 `from cuda import cudart`
        # （cuda-python 包）触发 cudaProfilerStart/Stop，但训练镜像根本
        # 没装这个包，ImportError 被 bare except 吞掉，nsys 全程没收到
        # 任何 Start/Stop 信号，capture range 从未打开，自然无所谓
        # "结束时杀不杀进程"。已改用 `torch.cuda.profiler.start/stop`
        # 顶替（同一个 libcudart 符号，镜像自带的 torch 里可用）。
        # 换了这个真正生效的触发方式后，stop-shutdown 的问题才终于暴露：
        # cudaProfilerStop() 一旦真正到达 nsys，stop-shutdown 会让 nsys
        # 立刻着手关停会话/杀进程，这个过程会让 cudaProfilerStop() 本身
        # 返回一个非 0 的 CUDA 错误码——PyTorch 的 check_error() 见到非 0
        # 就直接 raise SystemExit，把整个训练 actor 当场带崩（实测复现：
        # step 30 时整个 job 因为这个 SystemExit 直接 rc=1 退出，DataLoader
        # worker 被连带杀死）。
        # 显式改成 'stop'：只在 capture window 结束时 finalize 报告文件、
        # 不碰目标进程，cudaProfilerStop() 能正常返回成功，训练不受影响；
        # 报告落盘后由 nsys_step_profile.py 的 `_copy_ray_nsight_report()`
        # 主动从 ray 临时 session 目录拷到 NSYS_OUTPUT_DIR。目标 step 跑完
        # 想结束整个任务，手动 `ktrain job stop` 即可（不依赖自动关闭）。
        'capture-range-end': 'stop',
        'stop-on-exit': 'true',
        'cuda-memory-usage': 'true',
        # 2026-09-15 追加：显式关掉 CPU IP/backtrace 采样。nsys 在"直接
        # launch 目标程序"这种用法下 --sample 的默认值是 'process-tree'
        # （不是 'none'，见 `nsys profile --help`），所以即使我们从没
        # 主动要过这份数据，nsys 也会默默尝试用 perf_event 去采集它——
        # 但 k8s pod 容器默认没有 CAP_SYS_ADMIN/放开的 perf_event_paranoid，
        # 这个采集必然失败，在 Nsight GUI 的 Diagnostics 面板里报两条
        # Error（"Sampling and/or context switch data cannot be
        # collected..." / "Unable to configure the collection of CPU
        # IP/backtrace samples..."）。这个失败只影响 CPU 侧采样数据（我们
        # 从没用过——所有分析都是基于 CUPTI 的 GPU kernel/NVTX/CUDA API
        # trace，跟 perf 采样是两套独立机制），不影响任何已有分析结论，
        # 纯粹是报告里多一份"注定失败、我们也用不到"的诊断噪音。显式设
        # 'none' 后 nsys 不会再尝试这个采集，Diagnostics 面板干净。
        's': 'none',
    }

    _orig_options = _ActorClassCls.options

    def _patched_options(self, **actor_options):
        runtime_env = actor_options.get('runtime_env')
        rank = None
        if isinstance(runtime_env, dict):
            env_vars = runtime_env.get('env_vars')
            if isinstance(env_vars, dict):
                rank = env_vars.get('RANK')
        if (rank is not None and rank == str(target_rank)
                and isinstance(runtime_env, dict)
                and '_nsight' not in runtime_env and 'nsight' not in runtime_env):
            runtime_env['_nsight'] = _nsight_cfg
            print(f'[ray-nsight] (pid={os.getpid()}) injected nsys wrap into '
                  f'actor RANK={rank} (target_rank={target_rank}, step={step})',
                  flush=True)
        return _orig_options(self, **actor_options)

    _patched_options._ray_nsight_patched = True
    _ActorClassCls.options = _patched_options
    print(f'[ray-nsight] (pid={os.getpid()}) patched ray.actor.ActorClass.options '
          f'(via sitecustomize, before ray.remote(SwiftSft) runs) to nsys-wrap '
          f'only RANK={target_rank} (nsight={_nsight_cfg}, step {step}); '
          f'report lands under <ray_session_dir>/logs/nsight/', flush=True)
    _INSTALLED = True

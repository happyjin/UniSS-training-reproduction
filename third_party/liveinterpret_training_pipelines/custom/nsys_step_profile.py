# Copyright (c) junyaoyang. All rights reserved.
"""Capture one target optimizer step with nsys (cudaProfilerApi capture-range).

Used together with the launch wrapper in ``ktrain/launch_train.sh`` /
``ktrain/launch_train_lance.sh``: when ``NSYS_PROFILE_STEP=<n>`` is set,
the launcher wraps the train process with
``nsys profile --capture-range=cudaProfilerApi``. nsys then samples from
start-up but only *writes* the window between ``cudaProfilerStart()`` and
``cudaProfilerStop()``. This module drives that window from inside the
trainer's per-step boundary so exactly the n-th optimizer step (forward +
backward + optimizer) is captured — with zero nsys overhead outside the
window (no trace file growth during the other ~200 steps, no MFU pollution).

Backend-agnostic (mirrors ``mfu_logging.py``'s dual-backend pattern):
DeepSpeed's swift trainer *is* an HF ``transformers.Trainer`` subclass, so
the window is driven by an ``on_step_end`` ``TrainerCallback``. Megatron's
``swift.megatron.trainers.base.BaseMegatronTrainer`` is a plain ``ABC`` with
its own hand-rolled ``train()`` loop that never calls
``transformers.Trainer.train`` — for that backend the window is instead
driven by patching ``BaseMegatronTrainer.train_step`` directly. Both hooks
are installed independently (whichever backend's classes are importable).

Env:
  NSYS_PROFILE_STEP=<n>   capture the n-th optimizer step (1-based). 0/unset =
                          disabled (no-op, no overhead).
  NSYS_STEP_RANK=<r>      rank that decides start/stop (default 0).
  NSYS_OUTPUT_DIR         only used by the launch wrapper for -o; this module
                          just calls Start/Stop on all ranks so every rank's
                          GPU activity lands in the per-rank report.
"""
from __future__ import annotations

import os

_IMPLEMENTED = False


def _target_step() -> int:
    try:
        return max(0, int(os.environ.get("NSYS_PROFILE_STEP", "0")))
    except ValueError:
        return 0


def _step_range_rank() -> int:
    try:
        return int(os.environ.get("NSYS_STEP_RANK", "0"))
    except ValueError:
        return 0


def _rank() -> int:
    try:
        return int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
    except ValueError:
        return 0


def _start() -> None:
    # 2026-09-11：原来只用 `from cuda import cudart`（cuda-python 包）调用
    # cudaProfilerStart，但当前训练镜像里根本没装 cuda-python
    # （`ModuleNotFoundError: No module named 'cuda'`），这行 import 每次
    # 都被下面的 bare except 吞掉——nsys 因此从没收到过真正的
    # cudaProfilerStart 信号，capture range 从未打开，nsight 目录里连一个
    # 临时文件都不会有。之前排查 `capture-range-end=stop-shutdown`
    # 不生效走错了方向：不是那个参数的问题，是根本没有 kernel 被录进去，
    # nsys 自然也没有 capture range 可以结束。
    # 改用 torch 自带的 `torch.cuda.profiler.start/stop`（底层调用同一个
    # libcudart 符号，nsys 的 cudaProfilerApi 拦截对它一样生效），且已确认
    # 当前镜像的 torch 版本里可用，不依赖额外包。cuda-python 仍保留作为
    # 优先尝试（如果以后镜像装了这个包，用它更直接），失败了再退回 torch。
    try:
        from cuda import cudart
        cudart.cudaProfilerStart()
        return
    except Exception:
        pass
    try:
        import torch
        torch.cuda.profiler.start()
    except BaseException as e:
        # 2026-09-11 追加：torch.cuda.profiler.start/stop 内部的
        # check_error() 对非 0 的 cudaError 会直接 raise SystemExit（不是
        # 普通 Exception！），上面的 except Exception 接不住，会直接把
        # 整个 actor 进程带崩（见 _stop() 同款坑，实测复现：训练在 step
        # 30 直接因为这里的 SystemExit 挂掉，DataLoader worker 被连带
        # kill）。这里必须用 BaseException 兜底，profiler 状态不对不该
        # 拖垮整次训练。
        print(f'[nsys-step][WARN] cudaProfilerStart failed via both cuda-python '
              f'and torch.cuda.profiler: {e!r}', flush=True)


def _stop() -> None:
    try:
        from cuda import cudart
        cudart.cudaProfilerStop()
        return
    except Exception:
        pass
    try:
        import torch
        torch.cuda.profiler.stop()
    except BaseException as e:
        # 同 _start() 的 BaseException 注释：torch 的 check_error 在这里
        # 真实抛过 SystemExit(1)，必须兜住，不能让 nsys 会话状态异常
        # 传染成训练崩溃。
        print(f'[nsys-step][WARN] cudaProfilerStop failed via both cuda-python '
              f'and torch.cuda.profiler: {e!r}', flush=True)


def _use_ray() -> bool:
    return os.environ.get('USE_RAY', 'false').strip().lower() in ('1', 'true')


def _copy_ray_nsight_report() -> None:
    """Copy this actor's nsys report out of ray's ephemeral session dir.

    Only relevant for the ray path: ``custom/ray_nsight_patch.py`` makes each
    actor's own worker process nsys-wrapped via ``runtime_env={"nsight":
    ...}}``, but Ray writes the report under
    ``<ray_session_dir>/logs/nsight/`` on the actor pod's *local* (ephemeral)
    filesystem — it disappears once the pod is torn down. Copy it to
    ``NSYS_OUTPUT_DIR`` (expected to be a persistent mount, e.g. FSX) so it
    survives job teardown and can actually be downloaded/analyzed.

    Best-effort / idempotent: safe to call more than once (e.g. once right
    after ``cudaProfilerStop()`` and once again at process exit via
    ``atexit``, in case nsys hadn't finished flushing the file the first
    time).
    """
    if not _use_ray():
        return
    try:
        import glob
        import shutil

        import ray
        node = ray._private.worker.global_worker.node
        session_dir = node.get_session_dir_path()
        nsight_dir = os.path.join(session_dir, 'logs', 'nsight')
        out_dir = os.environ.get('NSYS_OUTPUT_DIR', '/nvme/output/nsys')
        os.makedirs(out_dir, exist_ok=True)
        pid = os.getpid()
        matched = glob.glob(os.path.join(nsight_dir, f'*{pid}*'))
        if not matched:
            matched = glob.glob(os.path.join(nsight_dir, '*.nsys-rep'))
        for f in matched:
            dst = os.path.join(out_dir, os.path.basename(f))
            shutil.copy2(f, dst)
            print(f'[nsys-step] copied ray nsight report {f} -> {dst}', flush=True)
        if not matched:
            print(f'[nsys-step][WARN] no ray nsight report found yet under {nsight_dir}', flush=True)
    except Exception as e:  # pragma: no cover - best-effort diagnostics only
        print(f'[nsys-step][WARN] failed to copy ray nsight report: {e!r}', flush=True)


def install_nsys_step_profile() -> None:
    """Install backend-agnostic hooks, mirroring ``mfu_logging.py``'s dual-
    backend pattern (deepspeed via HF TrainerCallback, megatron via
    ``BaseMegatronTrainer.train_step`` patch).

    2026-09-16 fix: this used to *only* patch ``transformers.Trainer.train``.
    That's correct for the deepspeed path (swift's deepspeed trainer *is* an
    HF ``Trainer`` subclass), but ``swift.megatron.trainers.base.
    BaseMegatronTrainer`` is a plain ``ABC`` with its own hand-rolled
    ``train()`` loop (``while ...: self.run_train_step(...)``) — it never
    calls ``transformers.Trainer.train``, so the HF-only callback never
    fired for megatron jobs. Symptom looked identical to the earlier
    cuda-python-not-installed bug (no report ever produced), but the root
    cause here is different: ``cudaProfilerStart/Stop`` were never even
    *attempted*, because the callback that calls them was never wired into
    megatron's training loop in the first place. Fixed by adding a second,
    independent hook on ``BaseMegatronTrainer.train_step`` (same idea as
    ``install_mfu_logging_patch()``'s existing megatron branch).
    """
    global _IMPLEMENTED
    if _IMPLEMENTED:
        return
    step = _target_step()
    if step <= 0:
        # No-op path: NSYS_PROFILE_STEP unset → nothing armed, zero overhead.
        return

    if _use_ray():
        import atexit
        atexit.register(_copy_ray_nsight_report)

    _installed_any = False

    # ---- DeepSpeed / HF Trainer path ----
    try:
        from transformers import Trainer as _HFTrainer, TrainerCallback as _HFCallback
    except ImportError:
        _HFTrainer, _HFCallback = None, None

    if _HFCallback is not None:
        _state = {"started": False}

        class _NsysStepCallback(_HFCallback):
            def on_step_end(self, args, state, control, **kwargs):
                n = state.global_step
                # We want step == `step` to be captured. HF fires on_step_end
                # after step n completes, with global_step already n. So
                # capture the window covering step n itself:
                #   - after step (step-1) ends -> cudaProfilerStart (covers step `step`)
                #   - after step  step   ends -> cudaProfilerStop
                if n == step - 1:
                    _start()
                    _state["started"] = True
                    print(f"[nsys-step] started capture before step {step}", flush=True)
                elif n == step and _state["started"]:
                    _stop()
                    _state["started"] = False
                    print(f"[nsys-step] stopped capture after step {step}", flush=True)
                    # ray path only: pull the report out of ray's ephemeral
                    # session dir onto persistent storage as soon as the
                    # capture window closes (best-effort; atexit above
                    # retries in case nsys hadn't finished flushing yet).
                    _copy_ray_nsight_report()

        _orig_train = _HFTrainer.train

        def _train_with_nsys(self, *args, **kwargs):
            self.add_callback(_NsysStepCallback())
            return _orig_train(self, *args, **kwargs)

        _train_with_nsys.__wrapped__ = _orig_train
        _HFTrainer.train = _train_with_nsys
        print(f"[nsys-step] armed HF TrainerCallback capture at step {step} (deepspeed path)",
              flush=True)
        _installed_any = True

    # ---- Megatron / BaseMegatronTrainer path ----
    # Maintain our own step counter (megatron's train_step doesn't expose a
    # `global_step`-like arg the way HF's callback does) rather than relying
    # on a trainer-internal attribute name.
    try:
        from swift.megatron.trainers.base import BaseMegatronTrainer
    except ImportError:
        BaseMegatronTrainer = None

    if BaseMegatronTrainer is not None:
        _mg_state = {"n": 0, "started": False}
        _orig_mg_step = BaseMegatronTrainer.train_step

        def _mg_step_with_nsys(self, *args, **kwargs):
            # Same before/after semantics as the HF callback above: start
            # right before the target step runs (n == step-1, 0-based count
            # of steps completed so far), stop right after it completes
            # (n == step).
            if _mg_state["n"] == step - 1 and not _mg_state["started"]:
                _start()
                _mg_state["started"] = True
                print(f"[nsys-step] started capture before step {step} (megatron)", flush=True)
            result = _orig_mg_step(self, *args, **kwargs)
            _mg_state["n"] += 1
            if _mg_state["n"] == step and _mg_state["started"]:
                _stop()
                _mg_state["started"] = False
                print(f"[nsys-step] stopped capture after step {step} (megatron)", flush=True)
                _copy_ray_nsight_report()
            return result

        _mg_step_with_nsys.__wrapped__ = _orig_mg_step
        BaseMegatronTrainer.train_step = _mg_step_with_nsys
        print(f"[nsys-step] armed BaseMegatronTrainer.train_step capture at step {step} (megatron path)",
              flush=True)
        _installed_any = True

    if not _installed_any:
        return

    _IMPLEMENTED = True

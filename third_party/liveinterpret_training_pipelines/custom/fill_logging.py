# Copyright (c) ModelScope Contributors. All rights reserved.
"""Per-step DataLoader wait / FillRate — local JSONL (no GPU, no wandb).

FillRate is defined in ``docs/FILL_RATE_GPU_MFU.md``:

    t_wait  = wall time blocked on ``next(train_dataloader)`` this optimizer step
              (sum of every micro-batch fetch when grad-accum > 1)
    t_step  = wall time between consecutive ``on_step_end`` calls
    FillRate = 1 - t_wait / t_step

This module does **not** change batch contents. It wraps
``Trainer.get_train_dataloader()`` so each ``next()`` is timed with
``time.perf_counter()``, then rank 0 writes one JSONL line per optimizer step.

Eval loaders are not wrapped. Megatron's data loop is not ``next(DataLoader)``;
this patch is the DeepSpeed / HF Trainer path (SCENE B on 0284).

Env::

    FILL_LOG_ENABLED=1       master switch (default **0**; others' jobs stay quiet)
    FILL_LOG_PATH=<path>     JSONL output (default ``${OUTPUT_DIR}/fill_log.jsonl``)
    FILL_LOG_RANK=<n>        rank that logs (default 0)
    FILL_LOG_STEP_INTERVAL=k log every k optimizer steps (default 1)
    FILL_LOG_STDOUT=1        also print ``[fill-log]`` to stdout (default 0)

When enabled, records go to JSONL only. Stdout is opt-in so shared training
logs are not flooded with FillRate lines.
"""
from __future__ import annotations

import json
import os
import time

_IMPLEMENTED = False

# Accumulated since the previous optimizer-step flush.
_state = {
    "wait_s": 0.0,
    "n_next": 0,
    "last_step_t": None,
}


def _rank() -> int:
    try:
        return int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
    except ValueError:
        return 0


def _enabled() -> bool:
    return os.environ.get("FILL_LOG_ENABLED", "0") == "1"


def _stdout() -> bool:
    return os.environ.get("FILL_LOG_STDOUT", "0") == "1"


def _target_rank() -> int:
    try:
        return int(os.environ.get("FILL_LOG_RANK", "0"))
    except ValueError:
        return 0


def _log_path() -> str:
    p = os.environ.get("FILL_LOG_PATH")
    if p:
        return p
    out = os.environ.get("OUTPUT_DIR", os.environ.get("OUTPUT_ROOT", "/tmp"))
    return os.path.join(out, "fill_log.jsonl")


def note_wait(seconds: float) -> None:
    """Accumulate one ``next()`` duration. Safe to call from tests."""
    if not _enabled() or _rank() != _target_rank():
        return
    try:
        dt = float(seconds)
    except (TypeError, ValueError):
        return
    if dt < 0:
        return
    _state["wait_s"] += dt
    _state["n_next"] += 1


def reset_wait() -> None:
    _state["wait_s"] = 0.0
    _state["n_next"] = 0


def _write_jsonl(record: dict) -> None:
    try:
        path = _log_path()
        d = os.path.dirname(path)
        if d:
            os.makedirs(d, exist_ok=True)
        with open(path, "a") as f:
            f.write(json.dumps(record) + "\n")
    except Exception:
        pass


def flush_step(n_step: int, step_interval: int = 1) -> dict | None:
    """Pair accumulated ``t_wait`` with wall ``t_step`` and write one record."""
    if not _enabled() or _rank() != _target_rank():
        reset_wait()
        return None
    if n_step % step_interval != 0:
        return None
    wait_s = float(_state["wait_s"])
    n_next = int(_state["n_next"])
    reset_wait()
    now = time.monotonic()
    step_s = 0.0
    if _state["last_step_t"] is not None:
        step_s = now - _state["last_step_t"]
    _state["last_step_t"] = now
    if step_s <= 0:
        return None
    fill = max(0.0, min(1.0, 1.0 - (wait_s / step_s)))
    record = {
        "step": int(n_step),
        "t_wait_ms": round(wait_s * 1000.0, 3),
        "t_step_ms": round(step_s * 1000.0, 3),
        "fill_rate": round(fill, 6),
        "n_next": n_next,
    }
    _write_jsonl(record)
    if _stdout():
        print(
            f"[fill-log] step {n_step}: wait={record['t_wait_ms']:.1f}ms "
            f"step={record['t_step_ms']:.1f}ms n_next={n_next} "
            f"FillRate={fill:.4f}",
            flush=True,
        )
    return record


class _TimedIterator:
    """Wraps a DataLoader iterator; times each ``next()`` with perf_counter."""

    def __init__(self, inner):
        self._inner = inner

    def __iter__(self):
        return self

    def __next__(self):
        t0 = time.perf_counter()
        try:
            batch = next(self._inner)
        except StopIteration:
            raise
        note_wait(time.perf_counter() - t0)
        try:
            from mfu_logging import accumulate_pad_waste
            mask = batch.get("codec_attention_mask") if isinstance(batch, dict) else None
            if mask is not None and hasattr(mask, "sum"):
                accumulate_pad_waste(mask.sum(dim=-1).tolist())
        except Exception:
            pass
        return batch

    def __getattr__(self, name):
        return getattr(self._inner, name)


class _FillTimedLoader:
    """Delegates to the real DataLoader; only ``__iter__`` is intercepted.

    HuggingFace looks up ``__iter__`` / ``__len__`` on the type, so a thin
    wrapper is required — assigning ``loader.__iter__`` on the instance is
    ignored by ``iter(loader)``.
    """

    def __init__(self, loader):
        object.__setattr__(self, "_loader", loader)

    def __iter__(self):
        return _TimedIterator(iter(object.__getattribute__(self, "_loader")))

    def __len__(self):
        return len(object.__getattribute__(self, "_loader"))

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "_loader"), name)


def attach_next_timer(loader):
    """Return a loader whose ``next()`` contributes to ``t_wait``. Idempotent."""
    if loader is None:
        return loader
    if isinstance(loader, _FillTimedLoader):
        return loader
    return _FillTimedLoader(loader)


def _wrap_trainer_loader_instance(trainer) -> None:
    """Wrap the concrete trainer's loader method, including Swift overrides."""
    marker = "_fill_train_loader_wrapped"
    if getattr(trainer, marker, False):
        return
    original_get = trainer.get_train_dataloader

    def _timed_get(*args, **kwargs):
        loader = original_get(*args, **kwargs)
        timed = attach_next_timer(loader)
        if _stdout():
            print(
                f"[fill-log] wrapped trainer={type(trainer).__name__} "
                f"loader={type(loader).__name__}",
                flush=True,
            )
        return timed

    trainer.get_train_dataloader = _timed_get
    setattr(trainer, marker, True)


def install_fill_logging_patch() -> None:
    """Wrap HF ``get_train_dataloader`` + ``on_step_end`` flush. Rank 0 only."""
    global _IMPLEMENTED
    if _IMPLEMENTED:
        return
    if not _enabled():
        return
    if _rank() != _target_rank():
        return

    step_interval = max(1, int(os.environ.get("FILL_LOG_STEP_INTERVAL", "1")))
    _state["last_step_t"] = None
    reset_wait()

    try:
        from transformers import Trainer as _HFTrainer, TrainerCallback as _HFCallback
    except ImportError:
        _IMPLEMENTED = True
        return

    class _FillHFCallback(_HFCallback):
        def on_train_begin(self, args, state, control, **kwargs):
            reset_wait()
            _state["last_step_t"] = time.monotonic()

        def on_step_end(self, args, state, control, **kwargs):
            flush_step(state.global_step, step_interval=step_interval)

        def on_epoch_begin(self, args, state, control, **kwargs):
            # Drop the inter-epoch gap from t_step; keep wait accumulator
            # (first next of the new epoch is a real fetch).
            _state["last_step_t"] = time.monotonic()

    _orig_get = _HFTrainer.get_train_dataloader

    def _get_train_dataloader_timed(self, *args, **kwargs):
        loader = _orig_get(self, *args, **kwargs)
        return attach_next_timer(loader)

    _get_train_dataloader_timed.__wrapped__ = _orig_get
    _HFTrainer.get_train_dataloader = _get_train_dataloader_timed

    _orig_train = _HFTrainer.train

    def _train_with_fill(self, *args, **kwargs):
        # ms-swift trainer subclasses override get_train_dataloader, so patching
        # only transformers.Trainer misses the concrete method selected by Python
        # dispatch. Wrap the bound method on the actual trainer instance before
        # Trainer.train asks it for the loader.
        _wrap_trainer_loader_instance(self)
        self.add_callback(_FillHFCallback())
        return _orig_train(self, *args, **kwargs)

    _train_with_fill.__wrapped__ = _orig_train
    _HFTrainer.train = _train_with_fill
    _IMPLEMENTED = True


__all__ = [
    "attach_next_timer",
    "flush_step",
    "install_fill_logging_patch",
    "note_wait",
    "reset_wait",
]

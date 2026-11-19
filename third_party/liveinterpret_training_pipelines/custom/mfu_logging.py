# Copyright (c) junyaoyang. All rights reserved.
"""Per-step MFU / TFLOP-s accounting — local file + log (no nsys, no wandb).

Why no nsys / no online wandb
-----------------------------
Formal training runs must not attach nsys (overhead) and the self-hosted wandb
server may be unavailable. So MFU/TFLOP-s is computed *inside the training
process* from data already available at each step:
  * per-step supervised-token total  — accumulated by model/loss code
  * per-step wall time              — measured at the step boundary
  * parameter count / DP / peak     — read live (model, env, GPU name)
and written to a local JSONL file + printed to stdout. Nothing here blocks the
GPU (no .item()/.cpu()/synchronize on hot paths).

Shared state
------------
A module-level ``_state`` accumulates the per-step token count. Both backends
feed it and flush it at the step boundary:
  * DeepSpeed  — model forward calls ``accumulate_tokens()``; HF
                 ``TrainerCallback.on_step_end`` calls ``flush_step()``.
  * Megatron   — ``mcore_talker_loss._combined_loss_func`` calls
                 ``accumulate_tokens()``; patched step boundary calls
                 ``flush_step()``.
The token-count *definition* is unified so both backends are comparable: the
total supervised token count (text + codec) for the step.

Formula (dense 6N upper bound, same as docs/mfu.md):
    per_gpu_tflops = FLOP_FACTOR * N * tokens_per_step / s_per_step / DP / 1e12
    MFU            = per_gpu_tflops / PEAK_TFLOPS

All scalars are read dynamically, never hard-coded:
  * ``N``           — trainable params (``requires_grad``) from the live model;
                      adapter training naturally uses its small trainable set,
                      full-param training its full set.
  * ``tokens``      — accumulated per step (text + codec) by the model/loss.
  * ``step_sec``    — wall-clock step interval (or trainer speed log).
  * ``DP``          — world_size / (TP·PP·CP), read from the runtime / env.
  * ``PEAK``        — per-GPU bf16 peak, by GPU name, overridable via env.
  * ``FLOP_FACTOR`` — 6 by default (dense fwd+bwd), overridable via env for
                      multimodal / non-dense models.

Output
------
A JSONL file at ``MFU_LOG_PATH`` (default ``${OUTPUT_DIR}/mfu_log.jsonl``), one
line per logged step:
    {"step":N,"tokens":T,"step_sec":S,"dp":D,"params":P,"flop_factor":F,
     "tflops":TF,"mfu":M,"tokens_per_sec":X}

Env gates / knobs:
    MFU_LOG_ENABLED=1        (default 1) master switch
    MFU_LOG_PATH=<path>      JSONL output path
    MFU_PEAK_TFLOPS=<n>      override per-GPU bf16 peak (TFLOP/s)
    MFU_FLOP_FACTOR=<n>      override FLOPs coefficient (default 6; set <6 for
                             adapter/multimodal where 6N over-estimates)
    MFU_STEP_INTERVAL=<k>    log every k optimizer steps (default 1)
    MFU_LOGGING_RANK=<n>     rank that computes+logs (default 0)
    MFU_USE_SPEED_LOG=<0/1>  prefer trainer's ``train_speed(s/it)`` log over the
                             wall-clock step interval when both available
"""
from __future__ import annotations

import json
import os
import time

_IMPLEMENTED = False

# Module-level shared state. ``tokens`` is accumulated per micro-batch by the
# model / loss code; ``last_t`` is set at each flush to measure step time.
_state = {"tokens": 0, "last_t": None, "pad_sum": 0.0, "pad_n": 0}


def _rank() -> int:
    try:
        return int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
    except ValueError:
        return 0


def _enabled() -> bool:
    return os.environ.get("MFU_LOG_ENABLED", "1") == "1"


def _target_rank() -> int:
    try:
        return int(os.environ.get("MFU_LOGGING_RANK", "0"))
    except ValueError:
        return 0


def _log_path() -> str:
    p = os.environ.get("MFU_LOG_PATH")
    if p:
        return p
    out = os.environ.get("OUTPUT_DIR", os.environ.get("OUTPUT_ROOT", "/tmp"))
    return os.path.join(out, "mfu_log.jsonl")


def _peak_tflops() -> float:
    """Resolve per-GPU bf16 peak FLOP/s (TFLOP/s) from env or GPU name."""
    env = os.environ.get("MFU_PEAK_TFLOPS")
    if env:
        try:
            return float(env)
        except ValueError:
            pass
    try:
        import torch
        name = torch.cuda.get_device_name(_rank() % torch.cuda.device_count()).lower()
    except Exception:
        name = ""
    if any(s in name for s in ("h200", "h100", "gh100", "h800")):
        return 989.0
    if "a100" in name:
        return 312.0
    return 989.0  # safe default (H200)


def _data_parallel_degree() -> int:
    """DP = world_size / (TP·PP·CP).

    world_size from the distributed runtime if initialised, else from the
    ``WORLD_SIZE`` env var that torchrun sets (covers both real training and
    standalone runs where dist may not be initialised at flush time).
    """
    world = 1
    try:
        import torch.distributed as dist
        if dist.is_initialized():
            world = dist.get_world_size()
    except Exception:
        pass
    if world == 1:
        try:
            world = int(os.environ.get("WORLD_SIZE", "1"))
        except ValueError:
            world = 1
    try:
        tp = int(os.environ.get("TP", "1"))
        pp = int(os.environ.get("PP", "1"))
        cp = int(os.environ.get("CP", "1"))
    except ValueError:
        tp = pp = cp = 1
    denom = max(1, tp * pp * cp)
    return max(1, world // denom)


def _unwrap_model(model):
    """Recursively unwrap common distributed / mcore wrappers to reach the
    underlying ``nn.Module`` whose ``parameters()`` is meaningful.

    Megatron's trainer hands us ``Float16Module``/``DDP``-style wrappers where
    ``model.parameters()`` may be empty (params live in ``.module``). DeepSpeed /
    HF may hand us ``DistributedDataParallel`` (params under ``.module``).
    """
    seen = set()
    while model is not None and id(model) not in seen:
        seen.add(id(model))
        # megatron Float16Module / mcore wrappers expose the real module here
        for attr in ("module", "model", "_module"):
            m = getattr(model, attr, None)
            if m is not None and m is not model:
                model = m
                break
        else:
            break
    return model


def _param_count(model) -> float:
    """Trainable parameter count (``requires_grad=True`` only).

    Must NOT use ``model.parameters()`` unfiltered: that includes frozen params
    (e.g. the frozen thinker in adapter training), which would inflate N and
    over-estimate MFU. Only params that actually receive gradient participate in
    backward FLOPs, so filter on ``requires_grad``.
    """
    try:
        m = _unwrap_model(model)
        params = [p for p in m.parameters() if p.requires_grad]
        if params:
            return float(sum(p.numel() for p in params))
        # Some megatron wrappers expose params only via .parameters() on the
        # unwrapped module but with requires_grad unset on frozen ones; fall
        # back to counting all params if the trainable filter is empty.
        all_params = [p for p in m.parameters()]
        if all_params:
            return float(sum(p.numel() for p in all_params))
    except Exception:
        pass
    return 0.0


def _param_count_from_trainer(trainer) -> float:
    """Megatron trainer holds its real models as a list
    (``unwrapped_models`` / ``wrapped_models``); ``self.model`` is not the
    module holder. Sum trainable params across those models.
    """
    if trainer is None:
        return 0.0
    models = getattr(trainer, "unwrapped_models", None)
    if not models:
        models = getattr(trainer, "wrapped_models", None)
    if not models:
        return 0.0
    total = 0.0
    for m in models:
        try:
            m = _unwrap_model(m)
            ps = [p for p in m.parameters() if p.requires_grad]
            if not ps:
                ps = [p for p in m.parameters()]
            total += float(sum(p.numel() for p in ps))
        except Exception:
            continue
    return total


def _flop_factor() -> float:
    """FLOPs coefficient for the P×tokens formula.

    Dense fwd+bwd = 6P (fwd 2P + bwd 4P). With Gradient Checkpointing
    (recompute) enabled the backward re-runs the forward, so true FLOPs = 8P
    (fwd 2P + recompute 2P + bwd 4P). Auto-derive from the trainer's
    ``GRADIENT_CHECKPOINTING`` / ``gradient_checkpointing`` setting; override via
    env ``MFU_FLOP_FACTOR``.
    """
    env = os.environ.get("MFU_FLOP_FACTOR")
    if env:
        try:
            return float(env)
        except ValueError:
            pass
    # DeepSpeed uses GRADIENT_CHECKPOINTING; Megatron uses RECOMPUTE_GRANULARITY.
    gc_vals = [
        os.environ.get("GRADIENT_CHECKPOINTING", ""),
        os.environ.get("RECOMPUTE_GRANULARITY", ""),
        os.environ.get("recompute_granularity", ""),
    ]
    for v in gc_vals:
        if v.lower() in ("true", "1", "full", "selective", "yes", "on"):
            return 8.0
    return 6.0


def accumulate_tokens(n) -> None:
    """Accumulate supervised-token count for the current step (micro-batch)."""
    global _state
    if not _enabled() or _rank() != _target_rank():
        return
    try:
        _state["tokens"] += int(n)
    except (TypeError, ValueError):
        pass


def accumulate_pad_waste(lengths) -> None:
    """Accumulate PadWaste = 1 - mean(len)/max(len) for one micro-batch.

    ``lengths`` is the per-row true length (codec attention-mask sum). Averaged
    across micro-batches at ``flush_step``. Rank-0 only; never touches GPU.
    """
    global _state
    if not _enabled() or _rank() != _target_rank():
        return
    try:
        ls = [int(x) for x in lengths if int(x) > 0]
    except (TypeError, ValueError):
        return
    if not ls:
        return
    mx = max(ls)
    if mx <= 0:
        return
    mean_l = sum(ls) / float(len(ls))
    _state["pad_sum"] += 1.0 - (mean_l / mx)
    _state["pad_n"] += 1


def _compute_mfu(n_param: float, tokens: int, step_sec: float, dp: int) -> dict:
    """Return {per_gpu_tflops, mfu_pct, tokens_per_sec, peak, flop_factor}."""
    peak = _peak_tflops()
    factor = _flop_factor()
    if n_param <= 0 or tokens <= 0 or step_sec <= 0:
        return {"tflops": 0.0, "mfu": 0.0, "tokens_per_sec": 0.0,
                "peak": peak, "flop_factor": factor}
    flops = factor * n_param * tokens          # fwd+bwd upper bound
    per_gpu = flops / step_sec / dp / 1e12     # TFLOP/s
    mfu = per_gpu / peak * 100.0
    return {
        "tflops": per_gpu,
        "mfu": mfu,
        "tokens_per_sec": tokens / step_sec,
        "peak": peak,
        "flop_factor": factor,
    }


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


def flush_step(n_step: int, model=None, trainer=None,
               step_sec_override: float | None = None,
               use_speed_log: bool = True, step_interval: int = 1) -> None:
    """Called at each optimizer-step boundary.

    Reads the accumulated token count, computes MFU/TFLOP-s, writes JSONL and
    prints to stdout, then resets the token counter. Only rank 0 (or the
    configured rank) computes/logs.

    ``model`` is the HF/deepspeed module; ``trainer`` is the megatron trainer
    whose real models live in ``unwrapped_models``/``wrapped_models`` (not
    ``self.model``). Pass whichever is available for the backend.
    """
    if not _enabled() or _rank() != _target_rank():
        return
    if n_step % step_interval != 0:
        return
    # Token source: megatron feeds module-level _state via accumulate_tokens();
    # deepspeed feeds model._mfu_tokens in the model forward. Sum both.
    tokens = _state["tokens"]
    _state["tokens"] = 0
    pad_n = int(_state.get("pad_n") or 0)
    pad_waste = (_state.get("pad_sum") or 0.0) / pad_n if pad_n else None
    _state["pad_sum"] = 0.0
    _state["pad_n"] = 0
    if model is not None:
        try:
            tokens += int(getattr(model, "_mfu_tokens", 0) or 0)
            model._mfu_tokens = 0
        except Exception:
            pass
    now = time.monotonic()
    step_sec = 0.0
    if _state["last_t"] is not None:
        step_sec = now - _state["last_t"]
    _state["last_t"] = now
    if step_sec_override is not None and step_sec_override > 0:
        step_sec = step_sec_override
    if tokens <= 0 or step_sec <= 0:
        return
    n_param = _param_count_from_trainer(trainer)
    if n_param <= 0 and model is not None:
        n_param = _param_count(model)
    dp = _data_parallel_degree()
    res = _compute_mfu(n_param, tokens, step_sec, dp)
    record = {
        "step": int(n_step),
        "tokens": int(tokens),
        "step_sec": round(step_sec, 6),
        "dp": int(dp),
        "params": int(n_param),
        "flop_factor": res["flop_factor"],
        "tflops": round(res["tflops"], 3),
        "mfu": round(res["mfu"], 3),
        "tokens_per_sec": round(res["tokens_per_sec"], 3),
    }
    if pad_waste is not None:
        record["pad_waste"] = round(pad_waste, 6)
    _write_jsonl(record)
    pad_s = f" PadWaste={pad_waste:.3f}" if pad_waste is not None else ""
    print(
        f"[mfu-log] step {n_step}: N={n_param/1e9:.2f}B tokens={tokens} "
        f"step={step_sec:.3f}s DP={dp} → {res['tflops']:.0f} TFLOP/s "
        f"MFU={res['mfu']:.1f}%{pad_s}",
        flush=True,
    )


def install_mfu_logging_patch() -> None:
    """Install backend-agnostic hooks. DeepSpeed via HF TrainerCallback,
    Megatron via BaseMegatronTrainer.train_step patch."""
    global _IMPLEMENTED
    if _IMPLEMENTED:
        return
    if not _enabled():
        print("[mfu-log] MFU_LOG_ENABLED != 1; mfu logging not armed", flush=True)
        return
    if _rank() != _target_rank():
        return

    step_interval = max(1, int(os.environ.get("MFU_STEP_INTERVAL", "1")))
    use_speed_log = os.environ.get("MFU_USE_SPEED_LOG", "1") == "1"
    _state["last_t"] = time.monotonic()

    # ---- DeepSpeed / HF Trainer path ----
    try:
        from transformers import Trainer as _HFTrainer, TrainerCallback as _HFCallback
    except ImportError:
        _HFTrainer, _HFCallback = None, None

    if _HFCallback is not None:
        class _MfuHFCallback(_HFCallback):
            def on_step_end(self, args, state, control, **kwargs):
                n = state.global_step
                logs = kwargs.get("logs") or {}
                speed = logs.get("train_speed(s/it)")
                so = float(speed) if (use_speed_log and speed) else None
                flush_step(n, model=kwargs.get("model"),
                           step_sec_override=so,
                           step_interval=step_interval)
        _orig_train = _HFTrainer.train
        def _train_with_mfu(self, *args, **kwargs):
            self.add_callback(_MfuHFCallback())
            return _orig_train(self, *args, **kwargs)
        _train_with_mfu.__wrapped__ = _orig_train
        _HFTrainer.train = _train_with_mfu
        print("[mfu-log] armed HF TrainerCallback (deepspeed path)", flush=True)

    # ---- Megatron / BaseMegatronTrainer path ----
    # Maintain our own step counter (like nsys_step_range) rather than relying
    # on a trainer-internal attribute name; flush at the end of each train_step.
    try:
        from swift.megatron.trainers.base import BaseMegatronTrainer
    except ImportError:
        BaseMegatronTrainer = None

    if BaseMegatronTrainer is not None:
        _mg_state = {"n": 0}
        _orig_step = BaseMegatronTrainer.train_step
        def _step_with_mfu(self, *args, **kwargs):
            _mg_state["n"] += 1
            result = _orig_step(self, *args, **kwargs)
            # Pass the trainer itself so _param_count can read the real
            # megatron models (self.unwrapped_models / wrapped_models) —
            # self.model is NOT the megatron module holder.
            flush_step(_mg_state["n"], model=getattr(self, "model", None),
                       trainer=self, step_interval=step_interval)
            return result
        _step_with_mfu.__wrapped__ = _orig_step
        BaseMegatronTrainer.train_step = _step_with_mfu
        print("[mfu-log] armed BaseMegatronTrainer.train_step (megatron path)", flush=True)

    _IMPLEMENTED = True

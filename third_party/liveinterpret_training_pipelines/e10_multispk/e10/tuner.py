"""E10 PEFT boundary: the S5 adapter of checkpoint-29773 plus eight speaker-tag rows and `e10_sd_proj`.

`--tuner_type e10_ms` (swift tuners_map). Fresh start = warm start from the S5 adapter in $E10_INIT_ADAPTER:
  1. the eight speaker-tag rows (unused `<|dur:N|>` rows, e10.common) get S5's own distinct norm-matched random
     initialisation (`reinit_stream_token_rows`, seed = --seed);
  2. `e10_sd_proj` = Linear(8 -> hidden, no bias), zero-initialised, is added to the model;
  3. a LoRA config identical to the S5 adapter's (same target regex, r, alpha) but with token rows
     [4 S5 stream rows] + [8 speaker rows] and modules_to_save + e10_sd_proj is attached;
  4. every S5 tensor is copied in (LoRA A/B, thinker_proj/fusion copies, the 4 stream rows); peft 0.15.2 stores
     the full row values (`index_copy`), so with a zero `e10_sd_proj` the model computes exactly checkpoint-29773
     (checked by tests/smoke_tuner_equivalence.py);
  5. the scope (SI_S4_TRAIN_SCOPE, B form = thinker): thinker LoRA + the 12 token rows + e10_sd_proj trainable,
     Talker LoRA / thinker_proj / fusion frozen. The trainable set is checked exactly.
`--resume_from_checkpoint` / inference: `from_pretrained` re-adds `e10_sd_proj` and loads an E10 adapter.
"""
from __future__ import annotations

import json
import os

import torch
from peft import LoraConfig, PeftModel, get_peft_model

from .common import N_SLOT, SD_DELAY_FRAMES

E10_METADATA = "e10_metadata.json"
E10_FORMAT = "e10_ms_peft_v1"


def _s4():
    from streaming_si.integration import s4_peft
    return s4_peft


def spk_ids_from_stream(stream_ids):
    """S5 adds the stream tokens in `stream_special_tokens()` order: wait, W, chunk, dur:1, dur:2, ... so the
    eight speaker rows are chunk+1 .. chunk+8 (verified against the tokenizer in plugin_e10 / tests)."""
    chunk = int(stream_ids[2])
    return [chunk + 1 + i for i in range(N_SLOT)]


def add_sd_module(raw):
    if getattr(raw, "e10_sd_proj", None) is not None:
        return raw.e10_sd_proj
    w = raw.thinker.embed_tokens.weight
    sd = torch.nn.Linear(N_SLOT, w.size(1), bias=False)
    torch.nn.init.zeros_(sd.weight)
    raw.e10_sd_proj = sd.to(device=w.device, dtype=w.dtype)
    return raw.e10_sd_proj


def _assert_rows_tied(peft_model, ids):
    raw = _s4().unwrap_s4_model(peft_model)
    embed, head = raw.thinker_model.model.embed_tokens, raw.thinker_model.lm_head
    ea, ha = getattr(embed, "token_adapter", None), getattr(head, "token_adapter", None)
    if ea is None or ha is None or ha.tied_adapter is not ea:
        raise RuntimeError("E10 token rows are not tied to lm_head (see s4_peft docstring)")
    if [int(i) for i in ea.token_indices["default"]] != [int(i) for i in ids]:
        raise RuntimeError(f"token rows {ea.token_indices['default']} != expected {list(ids)}")
    if raw.lm_head is not head:
        raise RuntimeError("SI.lm_head alias not refreshed after PEFT")


def _groups(model):
    s4 = _s4()
    raw, groups = s4._s4_trainable_groups(model)
    groups["e10_sd"] = s4._adapter_parameters(raw.e10_sd_proj, "modules_to_save")
    if not groups["e10_sd"]:
        raise RuntimeError("E10: e10_sd_proj has no modules_to_save copy")
    return raw, groups


def apply_scope(model, scope):
    raw, g = _groups(model)
    joint = set().union(*g.values())
    names = {id(p): n for n, p in raw.named_parameters()}
    current = {p for p in raw.parameters() if p.requires_grad}
    if current - joint:
        raise RuntimeError(f"unexpected trainable: {sorted(names.get(id(p), '?') for p in current - joint)}")
    selected = {"joint": joint,
                "thinker": g["thinker_lora"] | g["thinker_token"] | g["e10_sd"],
                "talker": g["talker_lora"] | g["thinker_proj"] | g["fusion"]}[scope]
    for p in raw.parameters():
        p.requires_grad_(p in selected)
    return sorted(names[id(p)] for p in selected)


def _scope():
    s = os.environ.get("SI_S4_TRAIN_SCOPE", "thinker")
    if s not in ("joint", "thinker", "talker"):
        raise ValueError(f"SI_S4_TRAIN_SCOPE={s!r}")
    return s


def _log(msg):
    try:
        from swift.utils import get_logger
        get_logger().info(msg)
    except Exception:  # noqa: BLE001
        print(msg, flush=True)


def warm_start(model, init_dir, seed, lora_dropout=0.0):
    """Steps 1-4 of the module docstring; returns the PeftModel (scope not applied)."""
    from peft.utils import set_peft_model_state_dict
    from safetensors.torch import load_file
    s4 = _s4()
    raw = s4.unwrap_s4_model(model)
    stream_ids = s4._check_token_ids(raw.config.s4_stream_token_ids)
    cfg0 = json.load(open(os.path.join(init_dir, "adapter_config.json")))
    meta0 = json.load(open(os.path.join(init_dir, s4.S4_METADATA)))
    if [int(i) for i in meta0["stream_token_ids"]] != stream_ids or \
            [int(i) for i in cfg0["trainable_token_indices"]] != stream_ids:
        raise RuntimeError(f"init adapter stream ids {meta0['stream_token_ids']} != model {stream_ids}")
    spk_ids = spk_ids_from_stream(stream_ids)
    with torch.no_grad():
        info = raw.reinit_stream_token_rows(spk_ids, seed=int(seed))
    add_sd_module(raw)
    config = LoraConfig(r=int(cfg0["r"]), lora_alpha=int(cfg0["lora_alpha"]), lora_dropout=float(lora_dropout),
                        bias="none", target_modules=cfg0["target_modules"],
                        modules_to_save=list(cfg0["modules_to_save"]) + ["e10_sd_proj"],
                        trainable_token_indices=stream_ids + spk_ids)
    pm = s4._refresh_thinker_aliases(get_peft_model(raw, config))
    _assert_rows_tied(pm, stream_ids + spk_ids)

    state = load_file(os.path.join(init_dir, "adapter_model.safetensors"))
    tok = {k: v for k, v in state.items() if "trainable_tokens_delta" in k}
    rest = {k: v for k, v in state.items() if k not in tok}
    delta = s4.unwrap_s4_model(pm).thinker_model.model.embed_tokens.token_adapter.trainable_tokens_delta["default"]
    if not tok:
        raise RuntimeError("init adapter has no token rows")
    for k, v in tok.items():   # 4 S5 rows (full values) + the 8 freshly initialised speaker rows
        if tuple(v.shape) != (len(stream_ids), delta.size(1)):
            raise RuntimeError(f"{k}: shape {tuple(v.shape)}")
        full = delta.detach().clone()
        full[: len(stream_ids)] = v.to(full)
        rest[k] = full
    from peft.utils import get_peft_model_state_dict
    own = get_peft_model_state_dict(pm)
    added = sorted(set(own) - set(rest))      # only e10_sd_proj may be new
    if any("e10_sd_proj" not in k for k in added):
        raise RuntimeError(f"E10 adapter tensors missing from the init adapter: {added[:5]}")
    for k in added:
        rest[k] = own[k]
    res = set_peft_model_state_dict(pm, rest)
    if res.unexpected_keys:
        raise RuntimeError(f"init adapter keys not in the E10 model: {res.unexpected_keys[:5]}")
    loaded = {k for k in pm.state_dict() if ("lora_" in k or "modules_to_save" in k)} - \
        {k for k in res.missing_keys}
    need = {k for k in pm.state_dict() if ("lora_" in k or "modules_to_save" in k) and "e10_sd_proj" not in k}
    if need - loaded:
        raise RuntimeError(f"E10 tensors not filled from the init adapter: {sorted(need - loaded)[:5]}")
    with torch.no_grad():
        if not torch.equal(delta[: len(stream_ids)], next(iter(tok.values())).to(delta)):
            raise RuntimeError("S5 stream rows were not copied")
    _log(f"[E10] warm start from {init_dir}: {len(rest)} tensors + 4 stream rows; speaker rows {spk_ids} "
         f"(norm {info['old_norm_mean']:.3f} -> {info['target_norm']:.3f}); e10_sd_proj zero")
    return pm


def save_e10(model, path, state_dict=None, safe_serialization=True, extra=None):
    s4 = _s4()
    pm = getattr(model, "module", model)
    if not isinstance(pm, PeftModel):
        raise TypeError("save_e10 needs a PeftModel")
    raw = s4.unwrap_s4_model(pm)
    stream_ids = s4._check_token_ids(raw.config.s4_stream_token_ids)
    os.makedirs(path, exist_ok=True)
    pm.save_pretrained(path, state_dict=state_dict, safe_serialization=safe_serialization,
                       save_embedding_layers=False, selected_adapters=["default"])
    with open(os.path.join(path, s4.S4_METADATA), "w", encoding="utf-8") as f:
        json.dump({"format": E10_FORMAT, "stream_token_ids": stream_ids, "train_scope": _scope(),
                   "base_model_dir": getattr(raw.config, "_name_or_path", None)}, f, indent=2)
    with open(os.path.join(path, E10_METADATA), "w", encoding="utf-8") as f:
        json.dump({"format": E10_FORMAT, "stream_token_ids": stream_ids,
                   "spk_token_ids": spk_ids_from_stream(stream_ids), "n_slot": N_SLOT,
                   "sd_delay_frames": SD_DELAY_FRAMES, "init_adapter": os.environ.get("E10_INIT_ADAPTER"),
                   **(extra or {})}, f, indent=2)


def load_e10(model, path, is_trainable=False):
    """Base model -> PeftModel with an E10 adapter (inference: merge_and_unload() afterwards)."""
    s4 = _s4()
    raw = s4.unwrap_s4_model(model)
    meta = json.load(open(os.path.join(path, E10_METADATA)))
    stream_ids = s4._check_token_ids(raw.config.s4_stream_token_ids)
    if [int(i) for i in meta["stream_token_ids"]] != stream_ids or \
            [int(i) for i in meta["spk_token_ids"]] != spk_ids_from_stream(stream_ids):
        raise RuntimeError(f"adapter token ids {meta} do not match the model {stream_ids}")
    add_sd_module(raw)
    pm = s4._refresh_thinker_aliases(PeftModel.from_pretrained(raw, path, is_trainable=is_trainable))
    _assert_rows_tied(pm, stream_ids + spk_ids_from_stream(stream_ids))
    return pm


def _finalize(model, origin):
    s4 = _s4()
    scope = _scope()
    names = apply_scope(model, scope)
    raw = s4.unwrap_s4_model(model)
    n_lora = sum(1 for _, m in raw.named_modules() if hasattr(m, "lora_A"))
    if n_lora != s4.expected_s4_lora_modules(raw):
        raise RuntimeError(f"LoRA on {n_lora} layers, expected {s4.expected_s4_lora_modules(raw)}")
    n = sum(p.numel() for p in raw.parameters() if p.requires_grad)
    _log(f"[E10 adapter] {origin} | scope={scope} tensors={len(names)} trainable={n / 1e6:.2f}M LoRA={n_lora}")
    return model


try:
    from swift.tuner_plugin import Tuner as _SwiftTuner
except Exception:  # noqa: BLE001
    _SwiftTuner = object


class E10Tuner(_SwiftTuner):
    @staticmethod
    def prepare_model(args, model):
        init = os.environ.get("E10_INIT_ADAPTER")
        if not init:
            raise RuntimeError("E10_INIT_ADAPTER (S5 adapter dir, e.g. checkpoint-29773) is required")
        pm = warm_start(model, init, seed=int(args.seed), lora_dropout=float(args.lora_dropout))
        return _finalize(pm, f"warm start from {init}")

    @staticmethod
    def save_pretrained(model, save_directory, state_dict=None, safe_serialization=True, **kwargs):
        save_e10(model, save_directory, state_dict=state_dict, safe_serialization=safe_serialization)

    @staticmethod
    def from_pretrained(model, model_id, **kwargs):
        pm = load_e10(model, model_id, is_trainable=kwargs.get("is_trainable", True))
        return _finalize(pm, f"loaded E10 adapter {model_id}")


def _sd_lr_optimizer():
    from transformers.trainer import Trainer as HfTrainer
    from swift.optimizers.base import OptimizerCallback

    class E10OptimizerCallback(OptimizerCallback):
        """`--optimizer e10_grouped`: e10_sd_proj at $E10_SD_LR (B's new-module LR), the rest at --learning_rate.
        The eight speaker rows share PEFT's single token-row tensor with the S5 rows and so use --learning_rate."""

        def create_optimizer(self, model=None):
            args = self.args
            model = model or self.trainer.model
            sd_lr = float(os.environ.get("E10_SD_LR", "0") or 0)
            if sd_lr <= 0:
                return HfTrainer.create_optimizer(self.trainer)
            decay = set(HfTrainer.get_decay_parameter_names(None, model))
            sd, other = [], []
            for n, p in model.named_parameters():
                if p.requires_grad:
                    (sd if "e10_sd_proj" in n else other).append((n, p))
            groups = []
            for lr, ps in ((sd_lr, sd), (args.learning_rate, other)):
                for wd in (0.0, args.weight_decay):
                    sel = [p for n, p in ps if (n in decay) == (wd > 0)]
                    if sel:
                        groups.append({"params": sel, "weight_decay": wd, "lr": lr})
            cls, kw = HfTrainer.get_optimizer_cls_and_kwargs(args, model)
            _log(f"[E10 optimizer] e10_sd_proj lr={sd_lr} ({sum(p.numel() for _, p in sd)} params), "
                 f"rest lr={args.learning_rate} ({sum(p.numel() for _, p in other) / 1e6:.2f}M)")
            return cls(groups, **kw)

    return E10OptimizerCallback


def register():
    from swift.tuner_plugin import tuners_map
    tuners_map["e10_ms"] = E10Tuner
    try:
        from swift.optimizers.mapping import optimizers_map
        optimizers_map["e10_grouped"] = _sd_lr_optimizer()
    except Exception as e:  # noqa: BLE001
        _log(f"[E10] e10_grouped optimizer not registered: {e}")

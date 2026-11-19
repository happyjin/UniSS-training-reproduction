"""S4 的 PEFT 边界：两套 Transformer LoRA + 四行 Thinker token + full bridge/fusion。

迁移自源 `trainer/s4_peft.py`，训练参数分组、scope 语义、checkpoint 格式（含 s4_metadata.json）
保持不变，因此源项目训出的 S4 adapter 目录可直接用 `--adapters` 加载。

与源实现唯一的差异是 peft 版本：源用 0.19 的 `LoraConfig(ensure_weight_tying=True)` 把
lm_head 绑到 embedding 的 token adapter；本仓库 peft 0.15.2 没有该参数，改由它的自动判定完成
（`config.tie_word_embeddings=True` 且 `model._tied_weights_keys` 为列表时，把 lm_head 包成共享
`tied_adapter` 的 TrainableTokensWrapper）。这一判定条件一旦不满足（例如改了 config），lm_head
会停留在旧行而 embedding 学新行，决策 token 的 logit 不随训练变化且不报错 —— 故 attach / load
后都用 `_assert_token_rows_tied` 硬校验。

另外提供 `StreamingSIS4Tuner`（swift `tuners_map` 插件，`--tuner_type streaming_si_s4`）。
"""

import json
import os

import torch
from peft import LoraConfig, PeftModel, get_peft_model

S4_LORA_TARGETS = (
    r"^(?:thinker_model\.model|talker\.decoder)\.layers\.\d+\."
    r"(?:self_attn\.(?:q_proj|k_proj|v_proj|o_proj)|"
    r"linear_attn\.(?:in_proj_qkv|in_proj_z|out_proj)|"
    r"mlp\.(?:gate_proj|up_proj|down_proj))$"
)
# Qwen3.5 thinker（streaming_si_4b）的 Gated DeltaNet 层没有 q/k/v/o：对应位置是 in_proj_qkv（q/k/v）、
# in_proj_z（输出门，等价于全注意力 q_proj 里的 gate 半边）、out_proj（o）。in_proj_b / in_proj_a
# 只输出每头 1 个标量（beta / 衰减），不挂 LoRA。Qwen3 变体没有 linear_attn，匹配集合与源项目完全相同。
_LORA_PER_LAYER = {'self_attn': 7, 'linear_attn': 6}
S4_FULL_MODULES = ["talker.thinker_proj", "talker.fusion"]
S4_METADATA = "s4_metadata.json"
S4_ADAPTER_FORMAT = "streaming_si_s4_peft_v2"
TRAIN_SCOPES = ("joint", "thinker", "talker")


def unwrap_s4_model(model):
    """去掉 DDP/PEFT 外壳，返回 SI 本体。"""
    raw = getattr(model, "module", model)
    if isinstance(raw, PeftModel):
        raw = raw.get_base_model()
    return raw


def expected_s4_lora_modules(model):
    """按已经实例化的 Thinker/Talker 层（及层类型）计算 LoRA 目标数：全注意力层 7，线性注意力层 6。"""
    raw = unwrap_s4_model(model)
    total = 0
    for layer in list(raw.thinker_model.model.layers) + list(raw.talker.decoder.layers):
        kinds = [k for k in _LORA_PER_LAYER if hasattr(layer, k)]
        if len(kinds) != 1:
            raise RuntimeError(f'无法识别 decoder 层的注意力类型: {type(layer).__name__}')
        total += _LORA_PER_LAYER[kinds[0]]
    return total


def _adapter_parameters(owner, attribute):
    """Collect PEFT parameters by adapter-module identity under a semantic owner."""
    parameters = set()
    for module in owner.modules():
        container = getattr(module, attribute, None)
        if container is not None and hasattr(container, "parameters"):
            parameters.update(container.parameters())
    return parameters


def _s4_trainable_groups(model):
    raw = unwrap_s4_model(model)
    thinker = raw.thinker_model.model
    talker = raw.talker
    groups = {
        "thinker_lora": _adapter_parameters(thinker, "lora_A")
        | _adapter_parameters(thinker, "lora_B"),
        "thinker_token": _adapter_parameters(thinker, "trainable_tokens_delta"),
        "talker_lora": _adapter_parameters(talker.decoder, "lora_A")
        | _adapter_parameters(talker.decoder, "lora_B"),
        "thinker_proj": _adapter_parameters(talker.thinker_proj, "modules_to_save"),
        "fusion": _adapter_parameters(talker.fusion, "modules_to_save"),
    }
    empty = sorted(name for name, parameters in groups.items() if not parameters)
    if empty:
        raise RuntimeError(f"empty required S4 trainable groups: {empty}")
    return raw, groups


def apply_s4_train_scope(model, scope):
    """Select the exact PEFT parameter groups for joint/Thinker/Talker training."""
    if scope not in TRAIN_SCOPES:
        raise ValueError(f"unknown S4 train scope: {scope!r}")
    raw, groups = _s4_trainable_groups(model)
    joint = set().union(*groups.values())
    current = {parameter for parameter in raw.parameters() if parameter.requires_grad}
    unexpected = current - joint
    missing = joint - current
    names_by_id = {id(parameter): name for name, parameter in raw.named_parameters()}
    if unexpected:
        names = sorted(names_by_id.get(id(parameter), "<unnamed>") for parameter in unexpected)
        raise RuntimeError(f"unexpected trainable S4 parameters: {names}")
    if missing:
        names = sorted(names_by_id.get(id(parameter), "<unnamed>") for parameter in missing)
        raise RuntimeError(f"required S4 parameters are not trainable: {names}")

    selected = {
        "joint": joint,
        "thinker": groups["thinker_lora"] | groups["thinker_token"],
        "talker": groups["talker_lora"] | groups["thinker_proj"] | groups["fusion"],
    }[scope]
    if scope != "joint":
        for parameter in raw.parameters():
            parameter.requires_grad_(parameter in selected)
    return sorted(names_by_id[id(parameter)] for parameter in selected)


def _refresh_thinker_aliases(model):
    """PEFT 会替换 embedding/lm_head 子模块，刷新构造期建立的非注册 alias。"""
    raw = unwrap_s4_model(model)
    object.__setattr__(raw, "thinker", raw.thinker_model.model)
    object.__setattr__(raw, "lm_head", raw.thinker_model.lm_head)
    return model


def _assert_token_rows_tied(model, token_ids):
    """embedding 与 lm_head 必须共享同一个 token adapter（见模块 docstring）。"""
    raw = unwrap_s4_model(model)
    embed = raw.thinker_model.model.embed_tokens
    head = raw.thinker_model.lm_head
    embed_adapter = getattr(embed, "token_adapter", None)
    head_adapter = getattr(head, "token_adapter", None)
    if embed_adapter is None or head_adapter is None or head_adapter.tied_adapter is not embed_adapter:
        raise RuntimeError(
            "S4 token 行未与 lm_head 绑定：peft 0.15.2 需要 config.tie_word_embeddings=True 且 "
            "_tied_weights_keys 含 thinker_model.lm_head.weight")
    if list(embed_adapter.token_indices["default"]) != [int(t) for t in token_ids]:
        raise RuntimeError(
            f"token adapter 行 {embed_adapter.token_indices['default']} 与预期 {list(token_ids)} 不一致")
    if raw.lm_head is not head:
        raise RuntimeError("SI.lm_head alias 未刷新到 PEFT 包装后的 lm_head")


def _check_token_ids(token_ids):
    ids = [int(token_id) for token_id in token_ids]
    if len(ids) != 4 or len(set(ids)) != 4:
        raise ValueError(f"S4 必须提供四个不同的流式 token id，实际为 {ids}")
    return ids


def attach_s4_peft(model, token_ids, r=8, alpha=16, dropout=0.0):
    """给未挂 adapter 的 S3 基座接上 S4 fresh adapter。"""
    ids = _check_token_ids(token_ids)
    config = LoraConfig(
        r=int(r),
        lora_alpha=int(alpha),
        lora_dropout=float(dropout),
        bias="none",
        target_modules=S4_LORA_TARGETS,
        modules_to_save=S4_FULL_MODULES,
        trainable_token_indices=ids,
    )
    peft_model = _refresh_thinker_aliases(get_peft_model(model, config))
    _assert_token_rows_tied(peft_model, ids)
    return peft_model


def save_s4_adapter(model, path, metadata, state_dict=None, safe_serialization=True):
    """保存 LoRA、四行 token 与 modules_to_save；不复制 S3 基座。"""
    peft_model = getattr(model, "module", model)
    if not isinstance(peft_model, PeftModel):
        raise TypeError("save_s4_adapter 需要已经 attach 的 PeftModel")
    os.makedirs(path, exist_ok=True)
    peft_model.save_pretrained(
        path, state_dict=state_dict, safe_serialization=safe_serialization,
        save_embedding_layers=False, selected_adapters=["default"])
    with open(os.path.join(path, S4_METADATA), "w", encoding="utf-8") as handle:
        json.dump(dict(metadata), handle, ensure_ascii=False, indent=2)


def load_s4_adapter(model, path, token_ids, is_trainable=True):
    """在重新构造的 S3 基座上恢复 S4 adapter，并核对四个 token 的 ID。"""
    with open(os.path.join(path, S4_METADATA), encoding="utf-8") as handle:
        metadata = json.load(handle)
    expected = _check_token_ids(token_ids)
    saved = [int(token_id) for token_id in metadata["stream_token_ids"]]
    if saved != expected:
        raise ValueError(f"adapter token ids {saved} 与当前 tokenizer {expected} 不一致")
    peft_model = PeftModel.from_pretrained(model, path, is_trainable=is_trainable)
    peft_model = _refresh_thinker_aliases(peft_model)
    _assert_token_rows_tied(peft_model, expected)
    return peft_model


# ---------------------------------------------------------------------------
# swift tuner 插件
# ---------------------------------------------------------------------------
def _stream_token_ids(model):
    ids = getattr(unwrap_s4_model(model).config, "s4_stream_token_ids", None)
    if not ids:
        raise RuntimeError(
            "config.s4_stream_token_ids 为空：S4 需要由 StreamingSILoader 从 tokenizer 解析四个流式 token")
    return _check_token_ids(ids)


def _train_scope():
    scope = os.environ.get("SI_S4_TRAIN_SCOPE", "joint")
    if scope not in TRAIN_SCOPES:
        raise ValueError(f"SI_S4_TRAIN_SCOPE={scope!r} 应为 {TRAIN_SCOPES}")
    return scope


def _finalize(model, token_ids, origin):
    from swift.utils import get_logger
    logger = get_logger()
    scope = _train_scope()
    names = apply_s4_train_scope(model, scope)
    raw = unwrap_s4_model(model)
    n_lora = sum(1 for _, module in raw.named_modules() if hasattr(module, "lora_A"))
    expected = expected_s4_lora_modules(raw)
    if n_lora != expected:
        raise RuntimeError(f"LoRA 命中 {n_lora} 个线性层，预期 {expected}（全注意力层 7 个 / 线性注意力层 6 个）")
    n_train = sum(p.numel() for p in raw.parameters() if p.requires_grad)
    logger.info(f"[S4 adapter] {origin} | scope={scope} tensors={len(names)} trainable={n_train / 1e6:.2f}M "
                f"| LoRA modules={n_lora} | token rows={token_ids}")
    return model


try:
    from swift.tuner_plugin import Tuner as _SwiftTuner
except Exception:  # 纯 PEFT 单测环境不装 swift
    _SwiftTuner = object


class StreamingSIS4Tuner(_SwiftTuner):
    """`--tuner_type streaming_si_s4`：fresh / `--adapters` 初始化 / `--resume_from_checkpoint`。

    LoRA 超参复用 swift 的 `--lora_rank/--lora_alpha/--lora_dropout`；训练范围由
    `SI_S4_TRAIN_SCOPE`（joint | thinker | talker）给出；token 行重置种子取 `--seed`。
    """

    @staticmethod
    def prepare_model(args, model):
        token_ids = _stream_token_ids(model)
        with torch.no_grad():
            info = unwrap_s4_model(model).reinit_stream_token_rows(token_ids, seed=int(args.seed))
        from swift.utils import get_logger
        get_logger().info(
            f"[S4 embed] 重置四个 active token 行：范数 {info['old_norm_mean']:.4f} -> "
            f"{info['target_norm']:.4f}，两两余弦 max={info['max_pair_cos']:.4f}")
        model = attach_s4_peft(model, token_ids, r=args.lora_rank, alpha=args.lora_alpha,
                               dropout=args.lora_dropout)
        return _finalize(model, token_ids, "fresh Thinker/Talker LoRA；thinker_proj/fusion 全量训练")

    @staticmethod
    def save_pretrained(model, save_directory, state_dict=None, safe_serialization=True, **kwargs):
        token_ids = _stream_token_ids(model)
        raw = unwrap_s4_model(model)
        save_s4_adapter(model, save_directory, {
            "format": S4_ADAPTER_FORMAT,
            "stream_token_ids": token_ids,
            "train_scope": _train_scope(),
            "base_model_dir": getattr(raw.config, "_name_or_path", None),
        }, state_dict=state_dict, safe_serialization=safe_serialization)

    @staticmethod
    def from_pretrained(model, model_id, **kwargs):
        token_ids = _stream_token_ids(model)
        model = load_s4_adapter(model, model_id, token_ids, is_trainable=kwargs.get("is_trainable", True))
        return _finalize(model, token_ids, f"从 {model_id} 加载 S4 LoRA/token/bridge/fusion")


def register_s4_tuner():
    from swift.tuner_plugin import tuners_map
    tuners_map["streaming_si_s4"] = StreamingSIS4Tuner

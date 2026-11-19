"""SI 的 HF 标准包装：可被 ms-swift 通过 `from_pretrained` 加载、在 forward 内算 loss。

与原生 `SI` 的差异只有三处，其余（talker、音频注入、流式推理方法）完全复用：

1. **构造不读权重**：原 `SI.__init__` 会从 `thinker_path` / Voxtral 目录现场加载预训练
   权重；HF 约定是「按 config 建空壳 -> from_pretrained 统一灌 state_dict」。这里按
   `config.to_qwen_config()` 建空 thinker，Voxtral 按 `config.voxtral_config_dir` 只建结构。
2. **audio_encoder 是真子模块**：原实现用 `object.__setattr__` 挂载，不在 state_dict、
   不随 `.to()` / DeepSpeed 搬运。这里注册为冻结子模块，权重键为 `audio_encoder.*`，
   保证多卡下设备与 dtype 一致。
3. **forward 返回 loss**：swift trainer 只读 `outputs.loss`。shift 在模型内部完成（见
   `_shifted_views`），template 产出未 shift 的完整序列，与 swift 约定一致。
"""
from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn.functional as F
from torch import nn
from transformers.modeling_outputs import ModelOutput
from transformers.models.qwen3 import Qwen3ForCausalLM

from .base_model import MMAudioProjector, OmniSIConfig, VoxtralAudioBackend
from .model_si import SI

IGNORE_INDEX = -100


class StreamingSIConfig(OmniSIConfig):
    """`OmniSIConfig` 的 HF 可注册版本。

    原 model_type `streaming-si` 含连字符，不能作为 AutoConfig 映射键；超参定义完全继承。
    新增字段：
      * voxtral_config_dir: Voxtral 目录（只读其 config.json 建 encoder 结构）；导出后为空，
        改由 `voxtral_audio_config` 内嵌的结构描述构建。
      * voxtral_audio_config / voxtral_projector_config: 导出时内嵌的 Voxtral 结构，使导出目录
        自包含、加载时不再依赖外部 Voxtral 目录。
      * text_loss_weight / speech_loss_weight: 各阶段 loss 权重（S1 text=0；S2 speech=0）。
      * s4_decision_weight: 非 None 即 S4 目标（译文/决策/步边界分项 CE，见 s4_objective）。
      * s4_stream_token_ids: [wait, W, chunk, src_end]，由 loader 从 tokenizer 解析写入。
      * thinker_arch: 'qwen3'（Qwen3-0.6B，默认）| 'qwen3_5'（Qwen3.5-4B，见 qwen3_5_compat）。
      * thinker_text_config: qwen3_5 时的完整文本塔配置（含 layer_types / linear_*）；扁平的
        hidden_size / vocab_size 等同时保留，供 audio_proj / talker 桥接等 SI 公共代码使用。
      * talker_preset: '0.1b' | '0.6b'（仅记录，结构以 talker_* 字段为准，见 variants.py）。
    """

    model_type = "streaming_si"

    def __init__(self, voxtral_config_dir=None, voxtral_audio_config=None,
                 voxtral_projector_config=None, text_loss_weight=1.0, speech_loss_weight=1.0,
                 s4_decision_weight=None, s4_stream_token_ids=None, thinker_arch='qwen3',
                 thinker_text_config=None, talker_preset='0.1b', **kwargs):
        super().__init__(**kwargs)
        self.thinker_arch = thinker_arch or 'qwen3'
        self.thinker_text_config = dict(thinker_text_config) if thinker_text_config else None
        self.talker_preset = talker_preset
        if self.thinker_arch == 'qwen3_5' and not self.thinker_text_config:
            raise ValueError("thinker_arch='qwen3_5' 需要 thinker_text_config")
        self.voxtral_config_dir = voxtral_config_dir
        self.voxtral_audio_config = voxtral_audio_config
        self.voxtral_projector_config = voxtral_projector_config
        self.text_loss_weight = float(text_loss_weight)
        self.speech_loss_weight = float(speech_loss_weight)
        self.s4_decision_weight = None if s4_decision_weight is None else float(s4_decision_weight)
        self.s4_stream_token_ids = list(s4_stream_token_ids) if s4_stream_token_ids else None


@dataclass
class StreamingSIOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    speech_logits: Optional[torch.FloatTensor] = None
    speech_target: Optional[torch.LongTensor] = None
    past_key_values: Optional[tuple] = None
    # 各分项 loss / 准确率（detach 后的 0 维张量），由 token_accuracy 补丁汇入 swift 日志
    metrics: Optional[Dict[str, torch.Tensor]] = None


_FLAT_THINKER_FIELDS = ('hidden_size', 'num_hidden_layers', 'vocab_size', 'num_attention_heads',
                        'num_key_value_heads', 'head_dim', 'tie_word_embeddings')


def build_thinker(config):
    """按 thinker_arch 构建空壳 thinker（权重随后由 from_pretrained / 导出脚本灌入）。"""
    arch = getattr(config, 'thinker_arch', 'qwen3') or 'qwen3'
    if arch == 'qwen3':
        return Qwen3ForCausalLM(config.to_qwen_config())
    if arch == 'qwen3_5':
        from .qwen3_5_compat import Qwen3_5CompatForCausalLM, Qwen3_5TextCompatConfig
        tcfg = Qwen3_5TextCompatConfig(**config.thinker_text_config)
        bad = {k: (getattr(config, k), getattr(tcfg, k)) for k in _FLAT_THINKER_FIELDS
               if getattr(config, k) != getattr(tcfg, k)}
        if bad:
            raise ValueError(f'扁平 thinker 字段与 thinker_text_config 不一致: {bad}')
        return Qwen3_5CompatForCausalLM(tcfg)
    raise ValueError(f'未知 thinker_arch={arch!r}')


def _build_audio_backend(config):
    """按 config 构建冻结的 Voxtral 结构（随机权重，等待 state_dict 灌入）。"""
    from .voxtral_compat import (VoxtralEncoder, VoxtralEncoderConfig, VoxtralMultiModalProjector,
                                 VoxtralProjectorConfig)
    if config.voxtral_audio_config:
        encoder = VoxtralEncoder(VoxtralEncoderConfig(**config.voxtral_audio_config))
        proj_cfg = VoxtralProjectorConfig(**config.voxtral_projector_config)
        return VoxtralAudioBackend(encoder, VoxtralMultiModalProjector(proj_cfg),
                                   proj_cfg.downsample_factor)
    if config.voxtral_config_dir:
        return VoxtralAudioBackend.from_config_dir(config.voxtral_config_dir)
    return None


class StreamingSIForConditionalGeneration(SI):
    config_class = StreamingSIConfig
    base_model_prefix = "streaming_si"
    main_input_name = "input_ids"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Qwen3DecoderLayer", "Qwen3_5CompatDecoderLayer"]
    # forward 返回的是 batch 内已平均的 loss（与源 trainer 的 loss/accumulation_steps 语义一致）。
    # 必须显式声明：否则 HF 见到 forward(**kwargs) 会判定「模型按 num_items_in_batch 自行归一」，
    # 从而跳过 GA 除法，并在 average_tokens_across_devices 下由 swift 再乘 world_size。
    accepts_loss_kwargs = False
    # audio_encoder 全程冻结且不参与反传：加载阶段缺失（未导出音频塔）时由调用方检查
    _keys_to_ignore_on_load_unexpected = [r"^speech_decoder\."]

    def __init__(self, config: StreamingSIConfig):
        # 刻意不调用 SI.__init__（它会从磁盘现场加载 thinker / Voxtral 权重）；
        # 下面逐项复现其结构，模块名与 SI checkpoint 键一一对应。
        audio_encoder = _build_audio_backend(config)
        if audio_encoder is not None:
            audio_encoder.attach_to_config(config)   # audio_hidden_size 512 -> 3072，先于 audio_proj
        super(SI, self).__init__(config)             # PreTrainedModel.__init__
        self.audio_proj = MMAudioProjector(config.audio_hidden_size, config.hidden_size)
        self.thinker_model = build_thinker(config)
        object.__setattr__(self, 'thinker', self.thinker_model.model)
        object.__setattr__(self, 'lm_head', self.thinker_model.lm_head)
        self.talker = self.talker_cls(config)
        w = self.thinker.embed_tokens.weight
        self.talker_text_embed = nn.Embedding(w.size(0), w.size(1))
        self.talker_text_embed.weight.requires_grad = False
        self._text_embed_hook = None
        if audio_encoder is not None:
            self.audio_encoder = audio_encoder      # 真子模块：随 .to()/DeepSpeed 搬运
            self.audio_encoder.requires_grad_(False)
        else:
            object.__setattr__(self, 'audio_encoder', None)
        object.__setattr__(self, 'audio_processor', None)
        object.__setattr__(self, 'speech_decoder', None)
        self.post_init()

    # ---- 初始化 / 训练态 ----

    def _init_weights(self, module):
        # 与 SI 一致：只做 Linear/Embedding 的标准初始化；预训练部分随后由 state_dict 覆盖
        std = getattr(self.config, 'initializer_range', 0.02)
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(0.0, std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(0.0, std)
        if getattr(module, 'scale_channel', False) and hasattr(module, 'scale_gain'):
            module.scale_gain.data.zero_()     # TalkerFusion 幅值通道：缺失键必须零初始化（起步即原模型）

    def train(self, mode=True):
        super().train(mode)
        if getattr(self, 'audio_encoder', None) is not None:
            self.audio_encoder.eval()               # 冻结塔保持 eval（无 dropout 漂移）
        return self

    def _set_gradient_checkpointing(self, enable=True, gradient_checkpointing_func=None):
        for sub in (self.thinker, self.talker.decoder):
            sub.gradient_checkpointing = enable
            if gradient_checkpointing_func is not None:
                sub._gradient_checkpointing_func = gradient_checkpointing_func

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        # thinker_model / talker.decoder 均为 PreTrainedModel，各自管理检查点
        self.thinker_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs)
        self.talker.decoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs)

    def set_text_embed_trainable(self, from_row=None):
        self._text_embed_trainable_intended = from_row is not None
        return super().set_text_embed_trainable(from_row)

    def _check_frozen_invariants(self):
        """首个训练步校验两个必须冻结的模块。

        swift 全参模式先 `model.requires_grad_(True)` 再按**前缀**冻结/激活；
        `--trainable_parameters talker`（不带点）会把 `talker_text_embed` 一起激活，
        而它一旦被训练，talker 的文本条件分布就会漂移（见 SI.__init__ 说明），且不报错。
        """
        bad = [n for n, p in self.named_parameters() if p.requires_grad and n.startswith('audio_encoder.')]
        if self.talker_text_embed.weight.requires_grad and not getattr(
                self, '_text_embed_trainable_intended', False):
            bad.append('talker_text_embed.weight')
        if bad:
            raise RuntimeError(
                f'streaming_si 必须冻结的参数被打开了梯度: {bad[:4]}{"..." if len(bad) > 4 else ""}。'
                f'swift 参数请用带点的前缀（如 --trainable_parameters talker.），并保持 --freeze_vit true')
        self._frozen_checked = True

    # ---- 训练前向 ----

    @staticmethod
    def _shifted_views(input_ids, labels, answer_mask, attention_mask):
        """把 swift 约定的未 shift 序列转成 SI 原生约定（逐位等价于源 dataset 的
        `X=ids[:-1] / Y=labels[1:] / answer_mask[:-1]`）。

        answer_mask 必须与 X 同下标：talker 的 text 路按 answer_mask 从 `talker_text_embed(X)`
        gather，错位会取到「答案 token 的前一个 token」，且不报错。
        """
        x = input_ids[:, :-1]
        y = labels[:, 1:] if labels is not None else None
        am = answer_mask[:, :-1] if answer_mask is not None else None
        att = attention_mask[:, :-1] if attention_mask is not None else None
        return x, y, am, att

    def forward(self, input_ids=None, attention_mask=None, labels=None, answer_mask=None,
                audio_inputs=None, audio_lens=None, speech_tokens=None, speech_token_lens=None,
                prompt_speech_tokens=None, prompt_speech_token_lens=None,
                audio_seg_lens=None, audio_feature_offsets=None, text_seg_lens=None,
                speech_seg_lens=None, speech_loss_weights=None, past_key_values=None,
                use_cache=False, **kwargs):
        # kwargs 可能含 S4 collate 的元信息（record_ids / view_kinds ...），forward 不消费
        if past_key_values is not None or labels is None:
            # 推理 / 流式路径：保持 SI 原语义（调用方自己管理 shift）
            return super().forward(
                input_ids, attention_mask=attention_mask, past_key_values=past_key_values,
                use_cache=use_cache, audio_inputs=audio_inputs, audio_lens=audio_lens,
                answer_mask=answer_mask, speech_tokens=speech_tokens,
                speech_token_lens=speech_token_lens, prompt_speech_tokens=prompt_speech_tokens,
                prompt_speech_token_lens=prompt_speech_token_lens,
                audio_seg_lens=audio_seg_lens, audio_feature_offsets=audio_feature_offsets,
                text_seg_lens=text_seg_lens, speech_seg_lens=speech_seg_lens)

        if self.training and not getattr(self, '_frozen_checked', False):
            self._check_frozen_invariants()
        x, y, am, att = self._shifted_views(input_ids, labels, answer_mask, attention_mask)
        text_w = float(self.config.text_loss_weight)
        speech_w = float(self.config.speech_loss_weight)
        res = super().forward(
            x, attention_mask=att, audio_inputs=audio_inputs, audio_lens=audio_lens,
            answer_mask=am if speech_w != 0 else None,
            speech_tokens=speech_tokens, speech_token_lens=speech_token_lens,
            prompt_speech_tokens=prompt_speech_tokens,
            prompt_speech_token_lens=prompt_speech_token_lens,
            skip_text_logits=(text_w == 0),
            audio_seg_lens=audio_seg_lens, audio_feature_offsets=audio_feature_offsets,
            text_seg_lens=text_seg_lens, speech_seg_lens=speech_seg_lens)

        if self.config.s4_decision_weight is not None:
            from .s4_objective import compute_s4_losses
            if not self.config.s4_stream_token_ids:
                raise RuntimeError('S4 目标需要 config.s4_stream_token_ids（由 StreamingSILoader 写入）')
            loss, metrics = compute_s4_losses(
                res, y, self.config.s4_stream_token_ids, self.config.s4_decision_weight,
                text_w, speech_w, speech_loss_weights)
        else:
            text_loss, speech_loss, metrics = compute_si_losses(res, y)
            loss = text_w * text_loss + speech_w * speech_loss + res.aux_loss
            metrics['text_loss'] = text_loss.detach()
            metrics['speech_loss'] = speech_loss.detach()
        # swift Seq2SeqTrainer 仅在 outputs.logits 非 None 时才汇报 metrics；text_w=0（S1）不算
        # lm_head，这里给 0 元素占位，保证分项 loss / speech_acc 仍进入训练日志。
        logits = res.logits if res.logits is not None else loss.new_zeros(0)
        return StreamingSIOutput(loss=loss, logits=logits, speech_logits=res.speech_logits,
                                 speech_target=res.speech_target, metrics=metrics)


def compute_si_losses(res, text_labels):
    """与源 `train_s3_s2st.py` 的 loss 逐式一致：

    * text：`CrossEntropyLoss(reduction='none')` 后对有效位（labels!=-100）求均值；
    * speech：有效位（speech_target!=-100）CE 均值；无有效位时返回可微零。
    """
    zero = res.aux_loss * 0
    metrics = {}
    fused_hidden = getattr(res, 'text_hidden', None)
    if fused_hidden is not None and text_labels is not None:
        # fused text CE：lm_head 投影与 CE 由 liger 按 chunk 融合，整个 logits 不物化。
        # 见 model_si.SI.use_fused_text_ce（含动机、前置条件与已知代价）。
        # 与下面的非 fused 分支在数学上等价（都是对有效位求 CE 均值）：
        # 这里直接用 ignore_index + reduction='mean'，liger 内部按非 ignore 位计数求均值，
        # 与 `(raw * m).sum() / m.sum()` 同式；差异仅在 bf16/fp32 的累加结合律。
        from .model_si import _liger_flce
        flce = _liger_flce()
        if flce is None:
            raise RuntimeError('fused_text_ce 已启用但 liger-kernel 不可用 —— '
                               '应在 SI.use_fused_text_ce 处就被拦下，不该走到这里')
        out = flce(fused_hidden.reshape(-1, fused_hidden.size(-1)), res.lm_head_weight,
                   text_labels.reshape(-1), ignore_index=IGNORE_INDEX, reduction='mean')
        text_loss = out[0] if isinstance(out, tuple) else out
        # text_acc 需要 logits.argmax，fused 路径下没有 logits，故本指标不产出（已在文档/开关注释说明）
    elif res.logits is None or text_labels is None:
        text_loss = zero
    else:
        logits = res.logits.float()
        valid = text_labels != IGNORE_INDEX
        raw = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                              text_labels.reshape(-1).clamp_min(0), reduction='none')
        m = valid.reshape(-1).to(raw.dtype)
        text_loss = (raw * m).sum() / (m.sum() + 1e-9)
        if bool(valid.any()):
            metrics['text_acc'] = (logits.argmax(-1)[valid] == text_labels[valid]).float().mean().detach()

    speech_loss = zero
    if res.speech_logits is not None:
        sl = res.speech_logits.reshape(-1, res.speech_logits.size(-1))
        st = res.speech_target.reshape(-1)
        valid = st != IGNORE_INDEX
        if bool(valid.any()):
            speech_loss = F.cross_entropy(sl[valid].float(), st[valid])
            with torch.no_grad():
                sv = sl[valid]
                metrics['speech_acc'] = (sv.argmax(-1) == st[valid]).float().mean()
                top10 = sv.topk(min(10, sv.size(-1)), dim=-1).indices
                metrics['speech_acc10'] = (top10 == st[valid, None]).any(-1).float().mean()
        else:
            speech_loss = sl.sum() * 0
    return text_loss, speech_loss, metrics

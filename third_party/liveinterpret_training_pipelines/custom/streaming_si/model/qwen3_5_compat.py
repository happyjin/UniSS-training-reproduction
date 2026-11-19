"""Qwen3.5 文本主干的 transformers 4.57 兼容实现（仅训练前向所需部分）。

本仓库钉在 transformers 4.57.3，它没有 `qwen3_5`（>=5.x 才有），但有结构同源的 `qwen3_next`。
Qwen3.5 文本塔 = Qwen3-Next 去掉 MoE（全部 dense MLP），线性注意力的输入投影改为分开的
`in_proj_qkv / in_proj_z / in_proj_b / in_proj_a`（Qwen3-Next 是融合的 `in_proj_qkvz / in_proj_ba`
且需要 `fix_query_key_value_ordering` 重排）。其余组件直接复用 4.57 的 qwen3_next：
门控全注意力（q_proj 输出 query+gate、q/k_norm、部分旋转 RoPE）、(1+w) 的 RMSNorm、
门控 RMSNorm、MLP、torch 版 chunk/recurrent gated delta rule（有 fla / causal-conv1d 时走加速核）。

参数名与官方 `Qwen3_5ForCausalLM` 一一对应（`model.layers.N.linear_attn.in_proj_qkv.weight` ...）。
官方多模态发布（`Qwen3_5ForConditionalGeneration`）的键是 `model.language_model.*`，由
`export_streaming_si.py` 去掉 `language_model.` 前缀后加载；`mtp.*` / `model.visual.*` 不使用。

RoPE：官方为 interleaved mRoPE（mrope_section=[11,11,10]）。纯文本输入时 t/h/w 三路位置相同，
mRoPE 退化为普通 1D RoPE（partial_rotary_factor=0.25 => 旋转维 64），这里按 1D 实现。
等价性由 `test_qwen3_5_compat.py` 与 transformers 5.x 官方实现逐元素对拍保证（含右 padding、跨 64 的 chunk）。

增量缓存（推理用，`use_cache=True` / `past_key_values=`）：状态容器复用 4.57 的
`Qwen3NextDynamicCache`（全注意力层存 KV；线性注意力层存卷积尾部与 delta rule 递归状态）。
与 qwen3_next 不同，这里**每次可以追加任意多个 token**（流式同传每步追加 `<|chunk|>` + 一块音频），
qwen3_next 只在单 token 时才续算、多 token 会把线性注意力状态清零重来。续算方式：
卷积把上次输入的最后 `kernel-1` 帧拼在新输入前面；delta rule 以上次的递归状态为初始状态。
只支持 batch=1 或无 padding 的批（流式推理只追加、从不回退）。等价性见 `test_qwen3_5_compat.py`
（增量逐块前向与整段前向对拍）。
"""
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from transformers import GenerationMixin, PretrainedConfig, PreTrainedModel
from transformers.activations import ACT2FN
from transformers.masking_utils import create_causal_mask
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.models.qwen3_next import modeling_qwen3_next as _qn

QWEN3_5_TEXT_FIELDS = (
    'vocab_size', 'hidden_size', 'intermediate_size', 'num_hidden_layers', 'num_attention_heads',
    'num_key_value_heads', 'head_dim', 'hidden_act', 'max_position_embeddings', 'rms_norm_eps',
    'rope_theta', 'partial_rotary_factor', 'attention_bias', 'attention_dropout', 'layer_types',
    'linear_conv_kernel_dim', 'linear_key_head_dim', 'linear_value_head_dim', 'linear_num_key_heads',
    'linear_num_value_heads', 'tie_word_embeddings', 'initializer_range')


class Qwen3_5TextCompatConfig(PretrainedConfig):
    """Qwen3.5 文本塔配置（字段语义与官方 `Qwen3_5TextConfig` 相同，默认值为 Qwen3.5-4B）。"""

    model_type = 'streaming_si_qwen3_5_text'

    def __init__(self, vocab_size=248320, hidden_size=2560, intermediate_size=9216, num_hidden_layers=32,
                 num_attention_heads=16, num_key_value_heads=4, head_dim=256, hidden_act='silu',
                 max_position_embeddings=262144, rms_norm_eps=1e-6, rope_theta=10000000.0,
                 partial_rotary_factor=0.25, attention_bias=False, attention_dropout=0.0, layer_types=None,
                 linear_conv_kernel_dim=4, linear_key_head_dim=128, linear_value_head_dim=128,
                 linear_num_key_heads=16, linear_num_value_heads=32, tie_word_embeddings=True,
                 initializer_range=0.02, full_attention_interval=4, **kwargs):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.rope_scaling = None
        self.partial_rotary_factor = partial_rotary_factor
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.layer_types = list(layer_types) if layer_types else [
            'full_attention' if (i + 1) % full_attention_interval == 0 else 'linear_attention'
            for i in range(num_hidden_layers)]
        if len(self.layer_types) != num_hidden_layers or not set(self.layer_types) <= {
                'linear_attention', 'full_attention'}:
            raise ValueError(f'layer_types 非法: {self.layer_types}')
        self.linear_conv_kernel_dim = linear_conv_kernel_dim
        self.linear_key_head_dim = linear_key_head_dim
        self.linear_value_head_dim = linear_value_head_dim
        self.linear_num_key_heads = linear_num_key_heads
        self.linear_num_value_heads = linear_num_value_heads
        self.initializer_range = initializer_range
        self.use_cache = False
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)


class Qwen3_5GatedDeltaNet(nn.Module):
    """Qwen3.5 的 Gated DeltaNet（分开的输入投影；无缓存训练前向）。"""

    def __init__(self, config, layer_idx):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_v_heads = config.linear_num_value_heads
        self.num_k_heads = config.linear_num_key_heads
        self.head_k_dim = config.linear_key_head_dim
        self.head_v_dim = config.linear_value_head_dim
        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads
        self.conv_kernel_size = config.linear_conv_kernel_dim
        self.layer_idx = layer_idx
        self.activation = config.hidden_act
        self.act = ACT2FN[config.hidden_act]
        self.conv_dim = self.key_dim * 2 + self.value_dim
        self.conv1d = nn.Conv1d(self.conv_dim, self.conv_dim, bias=False, kernel_size=self.conv_kernel_size,
                                groups=self.conv_dim, padding=self.conv_kernel_size - 1)
        self.in_proj_qkv = nn.Linear(self.hidden_size, self.key_dim * 2 + self.value_dim, bias=False)
        self.in_proj_z = nn.Linear(self.hidden_size, self.value_dim, bias=False)
        self.in_proj_b = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)
        self.in_proj_a = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)
        self.dt_bias = nn.Parameter(torch.ones(self.num_v_heads))
        self.A_log = nn.Parameter(torch.log(torch.empty(self.num_v_heads).uniform_(0, 16)))
        # 始终用 torch 版门控 RMSNorm（fla 的 FusedRMSNormGated 需在 CUDA 上构造，参数名相同）
        self.norm = _qn.Qwen3NextRMSNormGated(self.head_v_dim, eps=config.rms_norm_eps)
        self.out_proj = nn.Linear(self.value_dim, self.hidden_size, bias=False)

    def _conv(self, mixed_qkv, seq_len):
        if _qn.causal_conv1d_fn is not None and mixed_qkv.is_cuda:
            return _qn.causal_conv1d_fn(x=mixed_qkv, weight=self.conv1d.weight.squeeze(1), bias=self.conv1d.bias,
                                        activation=self.activation, seq_idx=None)
        return F.silu(self.conv1d(mixed_qkv)[:, :, :seq_len])

    def _conv_cached(self, mixed_qkv, seq_len, cache):
        """带缓存的因果卷积：前面拼上次输入的最后 kernel-1 帧（首次为零，与训练前向相同）。"""
        prev = cache.conv_states[self.layer_idx]
        if prev is None:
            out = self._conv(mixed_qkv, seq_len)
            full = F.pad(mixed_qkv, (self.conv_kernel_size - 1, 0))
        else:
            full = torch.cat([prev.to(mixed_qkv.dtype), mixed_qkv], dim=-1)
            out = F.silu(F.conv1d(full, self.conv1d.weight, self.conv1d.bias, groups=self.conv_dim))
        cache.conv_states[self.layer_idx] = full[:, :, -(self.conv_kernel_size - 1):].contiguous()
        return out

    def forward(self, hidden_states, attention_mask=None, cache=None):
        hidden_states = _qn.apply_mask_to_padding_states(hidden_states, attention_mask)
        batch_size, seq_len, _ = hidden_states.shape
        mixed_qkv = self.in_proj_qkv(hidden_states).transpose(1, 2)
        mixed_qkv = (self._conv(mixed_qkv, seq_len) if cache is None
                     else self._conv_cached(mixed_qkv, seq_len, cache)).transpose(1, 2)
        z = self.in_proj_z(hidden_states).reshape(batch_size, seq_len, -1, self.head_v_dim)
        b = self.in_proj_b(hidden_states)
        a = self.in_proj_a(hidden_states)
        query, key, value = torch.split(mixed_qkv, [self.key_dim, self.key_dim, self.value_dim], dim=-1)
        query = query.reshape(batch_size, seq_len, -1, self.head_k_dim)
        key = key.reshape(batch_size, seq_len, -1, self.head_k_dim)
        value = value.reshape(batch_size, seq_len, -1, self.head_v_dim)
        beta = b.sigmoid()
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)   # fp16 下 A 可能为 -inf
        if self.num_v_heads // self.num_k_heads > 1:
            query = query.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
            key = key.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
        fast = query.is_cuda and _qn.chunk_gated_delta_rule is not None
        if cache is not None and seq_len == 1:
            rule = _qn.fused_recurrent_gated_delta_rule if (
                query.is_cuda and _qn.fused_recurrent_gated_delta_rule is not None) \
                else _qn.torch_recurrent_gated_delta_rule
        else:
            rule = _qn.chunk_gated_delta_rule if fast else _qn.torch_chunk_gated_delta_rule
        init = None if cache is None else cache.recurrent_states[self.layer_idx]
        core, state = rule(query, key, value, g=g, beta=beta, initial_state=init,
                           output_final_state=cache is not None, use_qk_l2norm_in_kernel=True)
        if cache is not None:
            cache.recurrent_states[self.layer_idx] = state
        core = self.norm(core.reshape(-1, self.head_v_dim), z.reshape(-1, self.head_v_dim))
        return self.out_proj(core.reshape(batch_size, seq_len, -1))


class Qwen3_5CompatDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_type = config.layer_types[layer_idx]
        if self.layer_type == 'linear_attention':
            self.linear_attn = Qwen3_5GatedDeltaNet(config, layer_idx)
        else:
            self.self_attn = _qn.Qwen3NextAttention(config, layer_idx)
        self.mlp = _qn.Qwen3NextMLP(config, intermediate_size=config.intermediate_size)
        self.input_layernorm = _qn.Qwen3NextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = _qn.Qwen3NextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, hidden_states, position_embeddings, attention_mask=None, linear_attn_mask=None,
                past_key_values=None, **kwargs):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        if self.layer_type == 'linear_attention':
            hidden_states = self.linear_attn(hidden_states, attention_mask=linear_attn_mask, cache=past_key_values)
        else:
            hidden_states, _ = self.self_attn(hidden_states=hidden_states, attention_mask=attention_mask,
                                              position_embeddings=position_embeddings,
                                              past_key_values=past_key_values, **kwargs)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = residual + self.mlp(self.post_attention_layernorm(hidden_states))
        return hidden_states


class Qwen3_5CompatPreTrainedModel(PreTrainedModel):
    config_class = Qwen3_5TextCompatConfig
    base_model_prefix = 'model'
    supports_gradient_checkpointing = True
    _no_split_modules = ['Qwen3_5CompatDecoderLayer']
    _supports_sdpa = True
    _supports_flash_attn = True

    def _init_weights(self, module):
        super()._init_weights(module)
        if isinstance(module, Qwen3_5GatedDeltaNet):
            module.dt_bias.data.fill_(1.0)
            module.A_log.data.uniform_(0, 16).log_()
        elif isinstance(module, _qn.Qwen3NextRMSNorm):
            module.weight.data.zero_()       # (1 + weight) 形式，零初始化即单位缩放


class Qwen3_5CompatTextModel(Qwen3_5CompatPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [Qwen3_5CompatDecoderLayer(config, i) for i in range(config.num_hidden_layers)])
        self.norm = _qn.Qwen3NextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = _qn.Qwen3NextRotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def new_cache(self):
        """空的增量缓存（全注意力层 KV + 线性注意力层卷积/递归状态）。"""
        return _qn.Qwen3NextDynamicCache(self.config)

    def forward(self, input_ids=None, attention_mask=None, position_ids=None, past_key_values=None,
                inputs_embeds=None, use_cache=False, output_hidden_states=False, **kwargs):
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError('input_ids / inputs_embeds 必须且只能给一个')
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        if use_cache and past_key_values is None:
            past_key_values = self.new_cache()
        cache = past_key_values
        if cache is not None and attention_mask is not None and not bool(torch.all(attention_mask == 1)):
            raise NotImplementedError('增量缓存不支持带 padding 的 attention_mask（线性注意力状态会被污染）')
        seq_len = inputs_embeds.shape[1]
        past_len = cache.get_seq_length() if cache is not None else 0
        cache_position = torch.arange(past_len, past_len + seq_len, device=inputs_embeds.device)
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)
        causal_mask = create_causal_mask(config=self.config, input_embeds=inputs_embeds,
                                         attention_mask=attention_mask, cache_position=cache_position,
                                         past_key_values=cache, position_ids=position_ids)
        linear_mask = None if attention_mask is None or bool(torch.all(attention_mask == 1)) else attention_mask
        position_embeddings = self.rotary_emb(inputs_embeds, position_ids)
        hidden_states = inputs_embeds
        all_hidden = [hidden_states] if output_hidden_states else None
        for layer in self.layers:
            hidden_states = layer(hidden_states, position_embeddings, attention_mask=causal_mask,
                                  linear_attn_mask=linear_mask, past_key_values=cache,
                                  position_ids=position_ids, cache_position=cache_position)
            if output_hidden_states:
                all_hidden.append(hidden_states)
        hidden_states = self.norm(hidden_states)
        if output_hidden_states:
            all_hidden[-1] = hidden_states      # 与 transformers 4.57 Qwen3Model 相同：末项为 norm 后输出
        return BaseModelOutputWithPast(last_hidden_state=hidden_states, past_key_values=cache,
                                       hidden_states=tuple(all_hidden) if output_hidden_states else None)


class Qwen3_5CompatForCausalLM(Qwen3_5CompatPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ['lm_head.weight']

    def __init__(self, config):
        super().__init__(config)
        self.model = Qwen3_5CompatTextModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, value):
        self.lm_head = value

    def forward(self, input_ids=None, attention_mask=None, position_ids=None, inputs_embeds=None,
                labels=None, output_hidden_states=False, **kwargs) -> CausalLMOutputWithPast:
        out = self.model(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids,
                         inputs_embeds=inputs_embeds, output_hidden_states=output_hidden_states, **kwargs)
        logits = self.lm_head(out.last_hidden_state)
        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size)
        return CausalLMOutputWithPast(loss=loss, logits=logits, hidden_states=out.hidden_states,
                                      past_key_values=out.past_key_values)


def text_config_from_official(raw: dict) -> dict:
    """官方 config.json（`qwen3_5` 多模态或 `qwen3_5_text`）-> Qwen3_5TextCompatConfig 的 kwargs。"""
    tc = raw.get('text_config', raw) if raw.get('model_type') == 'qwen3_5' else raw
    if tc.get('model_type') not in ('qwen3_5_text', None):
        raise ValueError(f'不是 Qwen3.5 文本配置: model_type={tc.get("model_type")!r}')
    if tc.get('attn_output_gate', True) is not True:
        raise ValueError('兼容实现假定 attn_output_gate=True（Qwen3.5 全系如此）')
    if tc.get('mlp_only_layers') or tc.get('num_experts'):
        raise ValueError('兼容实现只支持 dense Qwen3.5（不含 MoE 层）')
    rope = dict(tc.get('rope_parameters') or tc.get('rope_scaling') or {})
    if rope.get('rope_type', 'default') != 'default':
        raise ValueError(f'兼容实现只支持 default RoPE，收到 {rope}')
    out = {k: tc[k] for k in QWEN3_5_TEXT_FIELDS if k in tc}
    out['rope_theta'] = float(rope.get('rope_theta', tc.get('rope_theta', 10000000.0)))
    out['partial_rotary_factor'] = float(rope.get('partial_rotary_factor', tc.get('partial_rotary_factor', 0.25)))
    out['tie_word_embeddings'] = bool(raw.get('tie_word_embeddings', tc.get('tie_word_embeddings', True)))
    return out

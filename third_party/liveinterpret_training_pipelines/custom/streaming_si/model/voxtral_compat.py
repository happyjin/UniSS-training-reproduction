"""Voxtral-Mini-4B-Realtime 音频编码器的 transformers 4.57 兼容实现。

`transformers.models.voxtral_realtime` 只存在于 transformers>=5.2，而本仓库受
ms-swift 4.4.0 约束钉在 4.57.3。SI 只使用其中的 audio tower（因果 Whisper 风格
encoder）与 multi-modal projector，本文件只实现这两者及其前端：

  * 参数命名与官方 checkpoint 的 `audio_tower.*` / `multi_modal_projector.*` 一一对应，
    可直接 `load_state_dict`；
  * 数值与 transformers 5.8.0 的实现逐元素一致（见 `test_voxtral_compat.py`，需在装有
    transformers>=5.2 的环境中运行对拍）；
  * 不实现 3B text decoder、generate、flash/flex attention 分派——SI 全程冻结此模块且
    只做前向，统一走 `F.scaled_dot_product_attention`。

流式接口：`forward(input_features, past_key_values, padding_cache, use_cache,
use_padding_cache)` 与官方同名同义，SI 的 `VoxtralStreamCache` 依赖这些字段。
"""
import json
import math
import os
from dataclasses import dataclass
from functools import cached_property
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PretrainedConfig
from transformers.activations import ACT2FN
from transformers.audio_utils import mel_filter_bank
from transformers.cache_utils import DynamicCache
from transformers.modeling_outputs import ModelOutput


# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------

class VoxtralEncoderConfig(PretrainedConfig):
    """对应官方 `VoxtralRealtimeEncoderConfig`（`config.json` 的 `audio_config`）。"""

    model_type = "streaming_si_voxtral_encoder"
    attribute_map = {
        "d_model": "hidden_size",
        "encoder_layers": "num_hidden_layers",
        "encoder_attention_heads": "num_attention_heads",
        "encoder_ffn_dim": "intermediate_size",
    }

    def __init__(self, hidden_size=1280, intermediate_size=5120, num_hidden_layers=32,
                 num_attention_heads=32, num_key_value_heads=None, head_dim=64,
                 activation_function="gelu", hidden_act="silu", num_mel_bins=128,
                 max_position_embeddings=1500, rms_norm_eps=1e-5, rope_parameters=None,
                 rope_theta=None, sliding_window=750, attention_dropout=0.0,
                 initializer_range=0.02, **kwargs):
        # 官方把 num_key_value_heads 映射到 num_attention_heads（MHA），这里显式处理
        kwargs.pop("model_type", None)
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads or num_attention_heads
        self.head_dim = head_dim if head_dim is not None else hidden_size // num_attention_heads
        self.activation_function = activation_function
        self.hidden_act = hidden_act
        self.num_mel_bins = num_mel_bins
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.sliding_window = sliding_window
        self.attention_dropout = attention_dropout
        self.initializer_range = initializer_range
        rope = dict(rope_parameters or {})
        if rope_theta is not None:
            rope.setdefault("rope_theta", rope_theta)
        rope.setdefault("rope_type", "default")
        rope.setdefault("rope_theta", 10000.0)
        if rope["rope_type"] != "default":
            raise NotImplementedError(
                f"Voxtral 兼容实现只支持 default RoPE，checkpoint 给的是 {rope['rope_type']}")
        self.rope_parameters = rope


@dataclass
class VoxtralProjectorConfig:
    """`VoxtralRealtimeMultiModalProjector` 只需要的四个字段。"""

    audio_hidden_size: int = 1280
    text_hidden_size: int = 3072
    downsample_factor: int = 4
    projector_hidden_act: str = "gelu"


def load_voxtral_configs(path):
    """从 Voxtral 目录的 `config.json` 解析 encoder / projector 配置。

    不经过 AutoConfig：4.57 不认识 `voxtral_realtime` 这个 model_type。
    """
    with open(os.path.join(path, "config.json")) as f:
        raw = json.load(f)
    audio = dict(raw.get("audio_config") or {})
    text = raw.get("text_config") or {}
    enc_cfg = VoxtralEncoderConfig(**audio)
    proj_cfg = VoxtralProjectorConfig(
        audio_hidden_size=enc_cfg.hidden_size,
        # 官方 text_config 缺省 hidden_size 为 3072（_default_text_config_kwargs）
        text_hidden_size=int(text.get("hidden_size", 3072)),
        downsample_factor=int(raw.get("downsample_factor", 4)),
        projector_hidden_act=raw.get("projector_hidden_act", "gelu"),
    )
    return enc_cfg, proj_cfg


# ---------------------------------------------------------------------------
# 流式卷积 padding cache（与官方 VoxtralRealtimeConv1dPaddingCache 同语义）
# ---------------------------------------------------------------------------

class VoxtralConv1dCacheLayer:
    def __init__(self):
        self.cache: Optional[torch.Tensor] = None
        self.left_pad = 0
        self.in_channels = 0

    def update(self, hidden_states, conv_module):
        if self.cache is None:
            self.left_pad = conv_module.left_pad
            self.in_channels = conv_module.in_channels
            self.cache = hidden_states.new_zeros(hidden_states.shape[0], self.in_channels, self.left_pad)
        if self.left_pad > 0:
            shortfall = max(0, self.left_pad - hidden_states.shape[-1])
            if shortfall > 0:
                padding_states = torch.cat([self.cache[:, :, -shortfall:], hidden_states], dim=-1)
            else:
                padding_states = hidden_states[:, :, -self.left_pad:]
        else:
            padding_states = hidden_states.new_empty(hidden_states.shape[0], self.in_channels, 0)
        current = self.cache.clone()
        self.cache.copy_(padding_states)
        return current


class VoxtralConv1dPaddingCache:
    def __init__(self):
        self.layers = {}

    def update(self, hidden_states, cache_key, conv_module):
        if cache_key not in self.layers:
            self.layers[cache_key] = VoxtralConv1dCacheLayer()
        padding_states = self.layers[cache_key].update(hidden_states, conv_module)
        return torch.cat([padding_states, hidden_states], dim=-1)


@dataclass
class VoxtralEncoderOutput(ModelOutput):
    last_hidden_state: Optional[torch.FloatTensor] = None
    past_key_values: Optional[DynamicCache] = None
    padding_cache: Optional[VoxtralConv1dPaddingCache] = None


# ---------------------------------------------------------------------------
# 模块
# ---------------------------------------------------------------------------

class VoxtralRotaryEmbedding(nn.Module):
    def __init__(self, config: VoxtralEncoderConfig):
        super().__init__()
        base = config.rope_parameters["rope_theta"]
        dim = config.head_dim
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x, position_ids):
        inv_freq = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        pos = position_ids[:, None, :].float()
        device_type = x.device.type if x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq.float() @ pos.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos, sin = emb.cos(), emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class VoxtralCausalConv1d(nn.Conv1d):
    def __init__(self, in_channels, out_channels, kernel_size, cache_key, stride=1, dilation=1, bias=True):
        super().__init__(in_channels, out_channels, kernel_size, stride=stride, dilation=dilation, bias=bias)
        self.cache_key = cache_key

    @cached_property
    def left_pad(self):
        return (self.kernel_size[0] - 1) * self.dilation[0] + 1 - self.stride[0]

    def forward(self, x, padding_cache=None):
        if padding_cache is not None:
            x = padding_cache.update(x, self.cache_key, self)
        else:
            x = F.pad(x, (self.left_pad, 0))
        return super().forward(x)


class VoxtralRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


def _rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rotary(q, k, cos, sin):
    cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
    return (q * cos) + (_rotate_half(q) * sin), (k * cos) + (_rotate_half(k) * sin)


class VoxtralAttention(nn.Module):
    def __init__(self, config: VoxtralEncoderConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.head_dim = config.head_dim
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.scaling = self.head_dim ** -0.5
        self.attention_dropout = config.attention_dropout
        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=True)
        # 与 Whisper 原版一致：k 投影无 bias
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=True)

    def forward(self, hidden_states, position_embeddings, attention_mask, past_key_values=None):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        q = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        k = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        v = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        cos, sin = position_embeddings
        q, k = _apply_rotary(q, k, cos, sin)
        if past_key_values is not None:
            k, v = past_key_values.update(k, v, self.layer_idx)
        if self.num_kv_heads != self.num_heads:
            rep = self.num_heads // self.num_kv_heads
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attention_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            scale=self.scaling)
        out = out.transpose(1, 2).reshape(*input_shape, -1).contiguous()
        return self.o_proj(out)


class VoxtralMLP(nn.Module):
    def __init__(self, config: VoxtralEncoderConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=True)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class VoxtralEmbedder(nn.Module):
    def __init__(self, config: VoxtralEncoderConfig):
        super().__init__()
        self.conv1 = VoxtralCausalConv1d(config.num_mel_bins, config.hidden_size, kernel_size=3, cache_key="conv1")
        self.conv2 = VoxtralCausalConv1d(config.hidden_size, config.hidden_size, kernel_size=3, stride=2,
                                         cache_key="conv2")

    def forward(self, input_features, padding_cache=None):
        x = F.gelu(self.conv1(input_features, padding_cache=padding_cache))
        x = F.gelu(self.conv2(x, padding_cache=padding_cache))
        return x.permute(0, 2, 1)


class VoxtralEncoderLayer(nn.Module):
    def __init__(self, config: VoxtralEncoderConfig, layer_idx: int):
        super().__init__()
        self.self_attn = VoxtralAttention(config, layer_idx)
        self.self_attn_layer_norm = VoxtralRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.final_layer_norm = VoxtralRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = VoxtralMLP(config)

    def forward(self, hidden_states, attention_mask, position_embeddings, past_key_values=None):
        residual = hidden_states
        hidden_states = self.self_attn(self.self_attn_layer_norm(hidden_states), position_embeddings,
                                       attention_mask, past_key_values)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.mlp(self.final_layer_norm(hidden_states))
        return residual + hidden_states


def _sliding_causal_mask(q_len, kv_len, past_len, sliding_window, device):
    """bool 掩码 (1,1,q,kv)，True=可见。

    查询 i 的绝对位置为 past_len+i；可见键 j 满足 j<=pos 且 pos-j<sliding_window，
    与官方 `create_sliding_window_causal_mask` 的可见集合一致。
    """
    q_pos = torch.arange(q_len, device=device)[:, None] + past_len
    k_pos = torch.arange(kv_len, device=device)[None, :] + (past_len + q_len - kv_len)
    visible = k_pos <= q_pos
    if sliding_window is not None:
        visible = visible & (q_pos - k_pos < sliding_window)
    return visible[None, None]


class VoxtralEncoder(nn.Module):
    """参数名与官方 `VoxtralRealtimeEncoder` 相同（embedder / layers / norm / rotary_emb）。"""

    def __init__(self, config: VoxtralEncoderConfig):
        super().__init__()
        self.config = config
        self.embedder = VoxtralEmbedder(config)
        self.layers = nn.ModuleList(
            [VoxtralEncoderLayer(config, i) for i in range(config.num_hidden_layers)])
        self.norm = VoxtralRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = VoxtralRotaryEmbedding(config)

    def forward(self, input_features, past_key_values=None, padding_cache=None,
                use_cache=False, use_padding_cache=False, position_ids=None, bounded_cache=True):
        """Encode new frames, retaining only the visible KV tail by default.

        ``bounded_cache=False`` is a regression-reference switch: attention still
        uses the identical sliding mask, but physically retains the full history.
        Absolute RoPE positions are tracked separately from retained KV length.
        Start a new stream with a new cache (and new convolution padding cache).
        """
        if use_padding_cache and padding_cache is None:
            padding_cache = VoxtralConv1dPaddingCache()
        hidden_states = self.embedder(input_features, padding_cache if use_padding_cache else None)
        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()
        past_len = past_key_values.get_seq_length() if use_cache else 0
        seen = getattr(past_key_values, "_voxtral_seen_tokens", past_len) if use_cache else 0
        q_len = hidden_states.shape[1]
        if position_ids is None:
            position_ids = (torch.arange(q_len, device=hidden_states.device) + seen).unsqueeze(0)
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # Mask coordinates are relative to the retained tail; RoPE above uses
        # absolute stream positions. Crop only after ALL layers consumed this
        # mask, so every layer sees the same complete current input chunk.
        kv_len = past_len + q_len
        mask = _sliding_causal_mask(q_len, kv_len, past_len, self.config.sliding_window,
                                    hidden_states.device)
        for layer in self.layers:
            hidden_states = layer(hidden_states, mask, position_embeddings,
                                  past_key_values if use_cache else None)
        hidden_states = self.norm(hidden_states)
        if use_cache:
            past_key_values._voxtral_seen_tokens = seen + q_len
            window = self.config.sliding_window
            if bounded_cache and window is not None:
                if int(window) <= 0:
                    raise ValueError("Voxtral sliding_window must be positive or None")
                keep = int(window) - 1
                for layer in past_key_values.layers:
                    if layer.keys is not None and layer.keys.shape[-2] > keep:
                        # clone, not a view/contiguous(): a view can retain the
                        # entire historical allocation even when its shape shrinks.
                        start = layer.keys.shape[-2] - keep
                        layer.keys = layer.keys[..., start:, :].clone()
                        layer.values = layer.values[..., start:, :].clone()
        return VoxtralEncoderOutput(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            padding_cache=padding_cache if use_padding_cache else None)


class VoxtralMultiModalProjector(nn.Module):
    """参数名与官方 `VoxtralRealtimeMultiModalProjector` 相同（linear_1 / linear_2）。"""

    def __init__(self, config: VoxtralProjectorConfig):
        super().__init__()
        self.linear_1 = nn.Linear(config.audio_hidden_size * config.downsample_factor,
                                  config.text_hidden_size, bias=False)
        self.act = ACT2FN[config.projector_hidden_act]
        self.linear_2 = nn.Linear(config.text_hidden_size, config.text_hidden_size, bias=False)

    def forward(self, audio_features):
        return self.linear_2(self.act(self.linear_1(audio_features)))


# ---------------------------------------------------------------------------
# 特征提取（与官方 VoxtralRealtimeFeatureExtractor 同算法）
# ---------------------------------------------------------------------------

class VoxtralFeatureExtractor:
    """只实现 SI 用到的整段 log-mel 提取与 `VoxtralWaveformStream` 依赖的属性。

    属性：feature_size / sampling_rate / hop_length / n_fft / mel_filters / global_log_mel_max。
    """

    def __init__(self, feature_size=128, sampling_rate=16000, hop_length=160, n_fft=400,
                 win_length=400, global_log_mel_max=1.5, **_):
        self.feature_size = feature_size
        self.sampling_rate = sampling_rate
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.win_length = win_length
        self.global_log_mel_max = global_log_mel_max
        self.mel_filters = mel_filter_bank(
            num_frequency_bins=1 + n_fft // 2, num_mel_filters=feature_size,
            min_frequency=0.0, max_frequency=8000.0, sampling_rate=sampling_rate,
            norm="slaney", mel_scale="slaney")

    CONFIG_NAMES = ("preprocessor_config.json", "processor_config.json", "voxtral_processor_config.json")

    @classmethod
    def from_pretrained(cls, path):
        """与 transformers 5.x `AutoFeatureExtractor` 同优先级：preprocessor_config.json，
        否则 processor_config.json 的 `feature_extractor` 段。

        官方 Voxtral-Mini-4B-Realtime-2602 只有 processor_config.json，其中
        global_log_mel_max=1.5 —— 这是源项目训练实际生效的值。
        """
        for name in cls.CONFIG_NAMES:
            fp = os.path.join(path, name)
            if not os.path.exists(fp):
                continue
            with open(fp) as f:
                cfg = json.load(f)
            cfg = cfg.get("feature_extractor", cfg.get("audio_processor", cfg))
            return cls(**cfg)
        raise FileNotFoundError(f"{path} 下没有 Voxtral 特征提取配置（{cls.CONFIG_NAMES}）")

    def _extract(self, waveform, device="cpu", center=True):
        window = torch.hann_window(self.n_fft, device=device)
        stft = torch.stft(waveform, self.n_fft, self.hop_length, window=window,
                          return_complex=True, center=center)
        magnitudes = stft[..., :-1].abs() ** 2
        mel_filters = torch.from_numpy(self.mel_filters).to(device, torch.float32)
        log_spec = torch.clamp(mel_filters.T @ magnitudes, min=1e-10).log10()
        if self.global_log_mel_max is not None:
            log_spec_max = torch.tensor(self.global_log_mel_max, device=log_spec.device, dtype=log_spec.dtype)
        else:
            log_spec_max = log_spec.max()
        log_spec = torch.maximum(log_spec, log_spec_max - 8.0)
        return ((log_spec + 4.0) / 4.0).detach().cpu()

    def __call__(self, raw_speech, sampling_rate=None, return_tensors="pt", padding=False,
                 device="cpu", center=True, **_):
        if sampling_rate is not None and sampling_rate != self.sampling_rate:
            raise ValueError(f"期望采样率 {self.sampling_rate}，收到 {sampling_rate}")
        if isinstance(raw_speech, np.ndarray):
            raw_speech = torch.from_numpy(np.ascontiguousarray(raw_speech))
        wav = raw_speech.to(torch.float32)
        if wav.dim() == 1:
            wav = wav[None]
        if padding not in (False, None, "do_not_pad") and wav.dim() != 2:
            raise NotImplementedError("兼容实现只支持单条或等长批量输入")
        return {"input_features": self._extract(wav, device, center).to(torch.float32)}


def load_voxtral_encoder_projector(path):
    """构建（随机初始化的）encoder / projector；权重由调用方按 audio_tower.* 前缀加载。"""
    enc_cfg, proj_cfg = load_voxtral_configs(path)
    return VoxtralEncoder(enc_cfg), VoxtralMultiModalProjector(proj_cfg), proj_cfg.downsample_factor

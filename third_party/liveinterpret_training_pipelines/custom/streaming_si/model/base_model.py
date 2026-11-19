"""
Streaming-SI 基础设施层
=======================
本文件是「与流式同传模型本体解耦」的那部分代码：

  * `OmniSIConfig`        —— 模型全部超参（thinker / talker / 音频 / 声码）
  * `RMSNorm`             —— talker 的 GatedBridge / TalkerFusion 依赖
  * `MMAudioProjector`    —— Voxtral adapter 输出 -> thinker hidden 的投影
  * Voxtral 流式前端      —— 增量 mel + 因果 encoder（同传主路径）
  * 流式特殊 token 协议   —— `<|wait|>` / `<|W|>` / `<|chunk|>` / `<|src_end|>` 等
  * `AudioEncoderMixin`   —— 音频编码与 `<|audio_pad|>` 占位符注入

模型本体见 `model_si.py`。

本文件的基础设施（配置默认值、token 协议、音频前端）**两个变体共用**，但本模块里的
`SIConfig` 默认超参是 `streaming_si`（0.6B）那一套；`streaming_si_4b` 由
`qwen3_5_compat.py` / `variants.py` 覆盖。两者规格见 `variants.py`：

    变体              thinker                              talker
    streaming_si      Qwen3-0.6B（28L / 1024d）            0.1B：8L / 1024d / 16-4 头
    streaming_si_4b   Qwen3.5-4B（32L / 2560d，3:1         0.6B：28L / 1024d / 16-8 头
                      Gated DeltaNet : 门控全注意力）

**当前在训练与评测的是 `streaming_si_4b`**（S4 / S5 的全部已发布指标都出自它），
后端为 Voxtral 流式 encoder(frozen) + Fun-CosyVoice3 flow/HiFT（FSQ v3 词表 6565）。
读本文件里写死的 1024 / 8 层等数值时，注意那是 0.6B 变体的默认值，不是 4B 的。

本文件只保留流式同传路径。历史实现中的 SenseVoice 音频后端、SigLIP 视觉塔与
MMVisionProjector、MiniMind 自研基座（Attention / MiniMindBlock /
MiniMindForCausalLM）、mimi 8 码本 talker、SileroVAD 会话管理均已剔除。
"""
import operator
import os
import warnings

import torch
from torch import nn
from transformers import PretrainedConfig
from transformers.models.qwen3 import Qwen3Config

# ============================================================================
#                            音频前端固定量
# ============================================================================
# 一个 adapter 帧对应的 mel 帧数：conv stride 2 × downsample_factor 4 = 8 帧 = 80ms
MEL_FRAMES_PER_ADAPTER_FRAME = 8
MEL_HOP_LENGTH = 160          # 16k / 160 = 100Hz mel
SR = 16000


def mel_frames_for_chunk_ms(chunk_ms):
    """把 chunk 时长(ms) 换成 mel 帧数，并校验是 80ms 整数倍。"""
    if chunk_ms % 80 != 0:
        raise ValueError(f"chunk 必须是 80ms 的整数倍（1 个 adapter 帧），收到 {chunk_ms}ms")
    return int(chunk_ms / 1000 * SR / MEL_HOP_LENGTH)


# ============================================================================
#                              基础算子
# ============================================================================
class RMSNorm(torch.nn.Module):
    """talker 的 GatedBridge / TalkerFusion 用它做归一化。"""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        return (self.weight * self.norm(x.float())).type_as(x)


class MMAudioProjector(nn.Module):
    """Voxtral adapter 输出(3072) -> thinker hidden 的投影。

    维度由构造参数决定，两个变体不同：
      streaming_si     3072 -> 1024（Qwen3-0.6B thinker）
      streaming_si_4b  3072 -> 2560（Qwen3.5-4B thinker）
    调用方传 `config.audio_hidden_size` / `config.hidden_size`，且必须在
    `attach_to_config()` 之后构造（它把 audio_hidden_size 由 512 改为 3072）。
    """

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, x):
        return self.mlp(x)


# ============================================================================
#                      流式同传特殊 token 协议
# ============================================================================
# 词表预算：两个变体都只有几百个空位，**分桶是必需的**，不是省事。
#   streaming_si      Qwen3-0.6B  len(tokenizer)=151669 < embedding 151936 → 267 个空位
#   streaming_si_4b   Qwen3.5-4B  embedding 248320，加完下面 38 个 stream token 与
#                     <|audio_pad|> 后 len(tokenizer)=248077 → 仍只剩 243 个空位
#                     （实测 S4 checkpoint：vocab_size=248320、audio_ids=[248076]）
# 因此 dur/pause **不能按 N 逐个建 token**（覆盖 40ms~20s 要上千个），改为**分桶**：
# 下面共 38 个 = 3（wait/W/chunk）+ 17 dur + 10 pause + 7 tau + 1 src_end。
STREAM_WAIT_TOKEN = "<|wait|>"      # READ 决策：再听一个 chunk，不输出文本
STREAM_WRITE_TOKEN = "<|W|>"        # WRITE 决策：可以开始输出译文
STREAM_CHUNK_TOKEN = "<|chunk|>"    # audio chunk 边界标记（三流交错序列用）
# 源流结束（真实场景来自 VAD / 录音停止）。必须显式喂给模型：否则它只能从
# 「最后一块特别短」这个假线索去猜，而推理时按固定切分根本看不到这个线索。
# 追加在 stream_special_tokens() 返回列表的**末尾**，不改动上面各 token 的 id。
STREAM_SRC_END_TOKEN = "<|src_end|>"

# 目标时长条件（等时对齐 isochrony）：25Hz -> n_token = dur_ms/40
# 桶按 speech token 数取对数间隔，覆盖 40ms ~ 20s
STREAM_DUR_BUCKETS = (1, 2, 3, 5, 8, 12, 18, 25, 35, 50, 70, 100, 140, 200, 280, 400, 500)
# 停顿条件：源侧 VAD 静音段 -> 译文侧插 pause，单位同为 speech token 数
STREAM_PAUSE_BUCKETS = (1, 2, 3, 5, 8, 12, 18, 25, 40, 60)
# 延迟档 τ 条件化（多 τ 混训）单位 ms
STREAM_TAU_BUCKETS = (320, 640, 960, 1280, 1920, 2560, 3840)


def bucketize(value, buckets):
    """把连续值映射到最近的桶（向上取整到第一个 >= value 的桶，超出则取最大桶）。"""
    for b in buckets:
        if value <= b:
            return b
    return buckets[-1]


def stream_special_tokens(dur_buckets=STREAM_DUR_BUCKETS, pause_buckets=STREAM_PAUSE_BUCKETS,
                          tau_buckets=STREAM_TAU_BUCKETS):
    """返回流式同传所需的全部特殊 token 列表。"""
    tokens = [STREAM_WAIT_TOKEN, STREAM_WRITE_TOKEN, STREAM_CHUNK_TOKEN]
    tokens += [f"<|dur:{n}|>" for n in dur_buckets]
    tokens += [f"<|pause:{n}|>" for n in pause_buckets]
    tokens += [f"<|tau:{t}|>" for t in tau_buckets]
    tokens += [STREAM_SRC_END_TOKEN]
    return tokens


def resolve_stream_token_ids(tokenizer):
    """反解流式 token id（供模型/数据集使用）。未注册的项不会出现在返回值里。"""
    vocab = tokenizer.get_vocab()

    def one(tok):
        return vocab.get(tok)

    ids = {
        'wait_token_id': one(STREAM_WAIT_TOKEN),
        'write_token_id': one(STREAM_WRITE_TOKEN),
        'chunk_token_id': one(STREAM_CHUNK_TOKEN),
        'src_end_token_id': one(STREAM_SRC_END_TOKEN),
        'dur_token_ids': {n: one(f"<|dur:{n}|>") for n in STREAM_DUR_BUCKETS},
        'pause_token_ids': {n: one(f"<|pause:{n}|>") for n in STREAM_PAUSE_BUCKETS},
        'tau_token_ids': {t: one(f"<|tau:{t}|>") for t in STREAM_TAU_BUCKETS},
        'dur_buckets': STREAM_DUR_BUCKETS,
        'pause_buckets': STREAM_PAUSE_BUCKETS,
        'tau_buckets': STREAM_TAU_BUCKETS,
    }
    for key in ('dur_token_ids', 'pause_token_ids', 'tau_token_ids'):
        ids[key] = {k: v for k, v in ids[key].items() if v is not None}
    return ids


def resolve_omni_token_ids(tokenizer, audio_special_token="<|audio_pad|>",
                           image_special_token="<|image_pad|>"):
    """从 tokenizer 反解模型依赖的全部特殊 token id（换词表后必须调用）。"""
    encode = lambda text: tokenizer.encode(text, add_special_tokens=False)
    ids = dict(
        audio_ids=encode(audio_special_token),
        image_ids=encode(image_special_token),
        think_end_ids=encode("</think>\n\n"),
        audio_special_token=audio_special_token,
        image_special_token=image_special_token,
    )
    enter_ids = encode("\n")
    if enter_ids:
        ids["enter_token_id"] = enter_ids[0]
    for name in ("pad_token_id", "bos_token_id", "eos_token_id"):
        value = getattr(tokenizer, name, None)
        if value is not None:
            ids[name] = value
    return ids


def build_qwen_tokenizer(thinker_path, audio_special_token="<|audio_pad|>",
                         add_stream_tokens=False, embedding_capacity=None):
    """加载 Qwen tokenizer 并对齐 Streaming-SI 的 token 协议。

    做两件事：
      1. 补一个 <|audio_pad|> 特殊 token（Qwen 自带 <|image_pad|>，但没有 audio 占位符），
         并按需注册流式同传 token。
      2. 把 bos/eos/pad 对齐到 <|im_start|>/<|im_end|>/<|endoftext|>。

    不改 `chat_template`：流式同传的 prompt 是引擎里的固定指令串
    （见 `inference.streaming_engine.S4StreamingEngine._prefill`），推理链路全程只用
    `encode` / `decode`，不经过 `apply_chat_template`。

    Args:
        add_stream_tokens: 是否注册流式同传 token。推理必须传 True。
        embedding_capacity: embedding 行数（Qwen3-0.6B 为 151936）。给定时会校验
            新增 token 不超出空位，避免静默越界。
    """
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(thinker_path)
    new_tokens = [audio_special_token] if audio_special_token not in tokenizer.get_vocab() else []
    if add_stream_tokens:
        vocab = tokenizer.get_vocab()
        new_tokens += [t for t in stream_special_tokens() if t not in vocab]
    if new_tokens:
        if embedding_capacity is not None:
            free = embedding_capacity - len(tokenizer)
            if len(new_tokens) > free:
                raise ValueError(
                    f"新增 {len(new_tokens)} 个特殊 token 超出 embedding 空位 {free} 个"
                    f"（len(tokenizer)={len(tokenizer)}, capacity={embedding_capacity}）。"
                    f"需 resize embedding 或缩减 dur/pause/tau 分桶数量。")
        tokenizer.add_special_tokens({"additional_special_tokens": new_tokens})
    for attr, token in (("bos_token", "<|im_start|>"), ("eos_token", "<|im_end|>"),
                        ("pad_token", "<|endoftext|>")):
        if token in tokenizer.get_vocab():
            setattr(tokenizer, attr, token)
    return tokenizer


# ============================================================================
#                                 配置
# ============================================================================
# Qwen3-0.6B 官方超参（thinker 主干）
QWEN3_0_6B = dict(
    hidden_size=1024,
    num_hidden_layers=28,
    num_attention_heads=16,
    num_key_value_heads=8,
    head_dim=128,           # 注意：与 hidden_size/num_heads 解耦
    intermediate_size=3072,
    vocab_size=151936,
    max_position_embeddings=40960,
    rms_norm_eps=1e-6,
    rope_theta=1e6,
    tie_word_embeddings=True,
    attention_bias=False,   # Qwen3 的 q/k/v_proj 无 bias
)

# 自研轻量 talker 的超参（8 层 Qwen3 风格 decoder）
QWEN3_TALKER_8L = dict(
    talker_hidden_size=1024,
    talker_num_hidden_layers=8,
    talker_num_attention_heads=16,
    talker_num_key_value_heads=4,
    talker_head_dim=128,             # 也可设 64（更省 KV cache，参数少约 20M）
    talker_intermediate_size=3072,
    talker_rope_theta=1e6,
    talker_special_tokens=4,         # sos / eos / task / fill，全部并入 speech_embedding
    load_pretrained_talker=False,    # talker 从零训练，不加载官方 llm.pt
)

# Fun-CosyVoice3-0.5B 的声码侧常量（FSQ v3 词表 + flow/HiFT）
COSYVOICE3_DECODER = dict(
    speech_token_size=6561,          # FSQ 码本大小 3^8
    token_frame_rate=25,             # 25 speech token / 秒
    audio_sample_rate=24000,
    talker_config_yaml="cosyvoice2.yaml",   # 下游 flow + HiFT 的官方 yaml
)


class OmniSIConfig(PretrainedConfig):
    """流式同传模型配置（Qwen3-0.6B thinker + 自研 8 层 talker + CosyVoice3 声码）。

    单一扁平配置类，全部超参在 `__init__` 里一次落定。

    解析结果：thinker 1024d/28L(head_dim 128)、talker 1024d/8L(16/4 heads, hd128,
              ffn3072)、speech vocab 6565(6561+4)、weight_tag=talker8

    `weight_tag` 与 `hidden_size` 共同决定 checkpoint 文件名后缀（`1024_talker8`），
    改动会导致既有权重加载不到。
    """
    model_type = "streaming-si"

    def __init__(self, thinker_path=None, talker_path=None, **kwargs):
        super().__init__(**kwargs)
        # ---------------- thinker 主干（Qwen3-0.6B）----------------
        for key, value in QWEN3_0_6B.items():
            setattr(self, key, kwargs.get(key, value))
        self.hidden_act = kwargs.get("hidden_act", 'silu')
        self.use_moe = kwargs.get("use_moe", False)
        self.dropout = kwargs.get("dropout", 0.0)
        self.flash_attn = kwargs.get("flash_attn", True)
        self.initializer_range = kwargs.get("initializer_range", 0.02)
        self.rope_scaling = kwargs.get("rope_scaling", None)
        self.thinker_path = thinker_path
        # checkpoint 文件名后缀标记，勿改（out/*_1024_talker8.pth）
        self.weight_tag = kwargs.get("weight_tag", "talker8")

        # ------------- 特殊 token（实际以 resolve_omni_token_ids 反解值为准）-------------
        self.audio_ids = kwargs.get("audio_ids", [151669])       # 追加的 <|audio_pad|>
        self.audio_special_token = kwargs.get("audio_special_token", "<|audio_pad|>")
        self.think_end_ids = kwargs.get("think_end_ids", [151668, 271])
        self.enter_token_id = kwargs.get("enter_token_id", 198)
        self.pad_token_id = kwargs.get("pad_token_id", 151643)   # <|endoftext|>
        self.bos_token_id = kwargs.get("bos_token_id", 151644)   # <|im_start|>
        self.eos_token_id = kwargs.get("eos_token_id", 151645)   # <|im_end|>

        # ---------------- 音频编码侧（Voxtral）----------------
        # 默认 512 是历史上 SenseVoice 的维度；voxtral 后端会在构造 audio_proj
        # **之前**通过 `VoxtralAudioBackend.attach_to_config` 改成 3072。
        self.audio_hidden_size = kwargs.get("audio_hidden_size", 512)
        self.audio_backend = kwargs.get("audio_backend", "voxtral")
        self.audio_frame_rate = kwargs.get("audio_frame_rate", 12.5)
        self.spk_emb_size = kwargs.get("spk_emb_size", 192)

        # ---------------- talker 主干（自研 8 层）----------------
        for key, value in QWEN3_TALKER_8L.items():
            setattr(self, key, kwargs.get(key, value))
        for key, value in COSYVOICE3_DECODER.items():
            setattr(self, key, kwargs.get(key, value))
        self.talker_path = talker_path
        # talker 侧特殊 token（索引 speech_embedding），相对顺序与官方 CosyVoice3LM 一致
        speech_token_size = self.speech_token_size
        self.talker_vocab_size = speech_token_size + self.talker_special_tokens
        self.talker_sos_token = speech_token_size + 0
        self.talker_eos_token = speech_token_size + 1
        self.talker_task_token = speech_token_size + 2
        self.talker_fill_token = speech_token_size + 3
        # talker 条件取 thinker 第几层 hidden：<0 表示取最后一层（已过 final norm）
        self.talker_bridge_layer = kwargs.get("talker_bridge_layer", -1)
        # 实验 b：talker 条件取**哪些位置**（见 model_si.build_talker_condition）
        #   answer       只取译文位置（默认，与历史行为逐位一致）
        #   audio        只取音频帧位置（<|audio_pad|> 占位符处）
        #   audio+answer 音频在前、译文在后拼接（序列里音频帧本就在译文之前，故顺序天然如此）
        # 只改 forward 的掩码，不改参数结构，故可像取层一样按阶段切换。
        self.talker_cond_positions = kwargs.get("talker_cond_positions", "answer")
        # 消融：TalkerFusion 末尾 RMSNorm 后加回 log-RMS 幅值通道（零初始化，见 model_si.TalkerFusion）。
        # 会新增参数 talker.fusion.scale_gain；从未开启的 checkpoint 起步时该参数按缺失键零初始化。
        self.talker_fusion_scale = bool(kwargs.get("talker_fusion_scale", False))
        self.min_token_text_ratio = kwargs.get("min_token_text_ratio", 2)
        self.max_token_text_ratio = kwargs.get("max_token_text_ratio", 20)

        # ---------------- 训练期开关（见 build_talker_condition 文档）----------------
        # cond_mode='text_only' 把 thinker hidden 条件整段置零（S1-0 只学「怎么说」）
        self.cond_mode = kwargs.get("cond_mode", "full")
        # True = text loss 走 liger fused_linear_cross_entropy（融合 lm_head 投影 + CE，
        # 不物化 151936 词表的全序列 logits）。对 S2 / S4 这类 text_w>0 的阶段省大量显存与带宽；
        # 代价是拿不到 text_acc 指标。默认 False = 行为零回归。见 model_si.SI.use_fused_text_ce
        self.fused_text_ce = kwargs.get("fused_text_ce", False)
        # True 且 cond_mode='text_only' 且 text_loss_weight=0 时，连 thinker 主干前向都跳过
        # （结果本就被丢弃，见 model_si.SI.can_skip_thinker）。默认 False = 行为零回归。
        self.skip_thinker_when_text_only = kwargs.get("skip_thinker_when_text_only", False)
        # hidden_noise=σ 给 bridge 前的 thinker hidden 加成比例高斯噪声（仅训练态）
        self.hidden_noise = kwargs.get("hidden_noise", 0.0)

    def to_qwen_config(self):
        """thinker 主干（Qwen3-0.6B）的配置。"""
        return Qwen3Config(
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_key_value_heads,
            head_dim=self.head_dim,
            intermediate_size=self.intermediate_size,
            vocab_size=self.vocab_size,
            max_position_embeddings=self.max_position_embeddings,
            rms_norm_eps=self.rms_norm_eps,
            rope_theta=self.rope_theta,
            tie_word_embeddings=self.tie_word_embeddings,
            attention_bias=False,
            attention_dropout=0.0,
            hidden_act=self.hidden_act,
        )

    def to_talker_qwen3_config(self):
        """自研 talker 主干（Qwen3 风格 causal decoder）的配置。

        vocab_size 直接设为 speech token 词表大小，于是 Qwen3Model 自带的 embed_tokens
        就是 speech_embedding，零冗余参数。
        """
        return Qwen3Config(
            hidden_size=self.talker_hidden_size,
            num_hidden_layers=self.talker_num_hidden_layers,
            num_attention_heads=self.talker_num_attention_heads,
            num_key_value_heads=self.talker_num_key_value_heads,
            head_dim=self.talker_head_dim,
            intermediate_size=self.talker_intermediate_size,
            vocab_size=self.talker_vocab_size,
            max_position_embeddings=self.max_position_embeddings,
            rms_norm_eps=self.rms_norm_eps,
            rope_theta=self.talker_rope_theta,
            tie_word_embeddings=False,
            attention_bias=False,
            attention_dropout=0.0,
            hidden_act="silu",
        )


# ============================================================================
#                    Voxtral 流式音频编码后端
# ============================================================================
# Voxtral Realtime encoder 的流式能力（scripts/verify_a1_voxtral_streaming.py 已验证）：
#   * CausalConv1d 卷积 stem（左侧 padding，left_pad/stride = [(2,1),(1,2)]）
#   * causal attention + sliding_window=750（15s，状态有界 -> 支持无限流）
#   * 官方内置 VoxtralRealtimeConv1dPaddingCache（conv 状态）+ DynamicCache（attention KV）
#   * 实测 chunk 拼接 vs 整段：encoder max|diff| 2.2e-06，端到端 4.5e-07
#   * 因果性反证：篡改未来 mel，已产出前缀 max|diff| = 0.000e+00（精确 0）
#
# 帧率链路：
#   16k wav -> 100Hz mel(128) -> conv stride 2 -> 50Hz -> adapter /4 -> 12.5Hz (80ms/帧)
#   而 CosyVoice FSQ speech token 是 25Hz (40ms/token)，恰好 1:2 —— 硬同步分叉点。
#
# chunk 必须是 80ms 的整数倍（8 mel 帧 -> 4 encoder 帧 -> 1 adapter 帧，整除）。
# 40ms 不整除（2 encoder 帧凑不满 adapter 的 4 帧），会产生余数堆积。
class VoxtralStreamCache:
    """Voxtral encoder 的流式状态。

    两套 cache 必须**同时**维护（缺一不可）：
      * padding_cache: CausalConv1d 的左侧 padding 状态（conv stem 跨 chunk 连续性）
      * past_key_values: attention KV（sliding_window=750 自动裁剪，状态有界）

    另外维护 mel 余数：chunk 边界不一定落在 8 帧整数倍上，不足一个 adapter 帧的
    尾部 mel 必须留到下一个 chunk，否则会被静默丢弃（导致音频内容缺失）。
    """

    def __init__(self):
        self.padding_cache = None
        self.past_key_values = None
        self.mel_remainder = None   # (B, 128, r) 待与下个 chunk 拼接的 mel 余数
        self.n_mel_seen = 0         # 累计消费的 mel 帧数（供延迟核算）
        self.n_frames_out = 0       # 累计产出的 adapter 帧数（12.5Hz）

    def reset(self):
        self.__init__()

    @property
    def audio_seconds(self):
        return self.n_mel_seen * MEL_HOP_LENGTH / SR


class VoxtralAudioProcessor:
    """Voxtral mel 前端（16k / n_fft=400 / hop=160 / 128 mel）。

    **流式关键**：官方 `_torch_extract_fbank_features` 在 `global_log_mel_max=None` 时
    用 `log_spec.max()` 归一化 —— 逐 chunk 调用会各自取局部最大值，导致**归一化不一致**，
    进而使 chunk 拼接与整段编码结果偏离。故流式路径必须固定 `global_log_mel_max`。
    """

    def __init__(self, feature_extractor=None, global_log_mel_max=2.0):
        self.fe = feature_extractor
        self.global_log_mel_max = global_log_mel_max
        if self.fe is not None and getattr(self.fe, 'global_log_mel_max', None) is None:
            # 固定全局最大值，保证逐 chunk 与整段归一化一致
            self.fe.global_log_mel_max = global_log_mel_max

    def __call__(self, wav, sr=SR, device='cpu'):
        """wav: 1D np.ndarray/tensor -> (128, T) log-mel tensor"""
        if self.fe is None:
            raise RuntimeError("VoxtralRealtimeFeatureExtractor 未加载")
        if torch.is_tensor(wav):
            wav = wav.detach().cpu().numpy()
        out = self.fe(wav, sampling_rate=sr, return_tensors='pt',
                      padding=False, device=str(device))
        feats = out['input_features']
        return feats[0] if feats.dim() == 3 else feats


class VoxtralWaveformStream:
    """把原始波形前缀转换成与整段训练严格一致的增量 mel。

    Voxtral 前端的 STFT 使用 ``center=True``。逐块独立调用会在每个块的左右边界
    重新做 reflect padding，因此不能直接拼接。这里按全局 frame 编号截取每批新帧
    真正依赖的波形窗口，并以 ``center=False`` 计算；只在整段录音的真实首尾做
    reflect padding。已释放帧之前的波形随即丢弃，因此长音频不会反复重算前缀。
    """

    def __init__(self, processor, sample_rate=SR):
        self.processor = processor
        self.sample_rate = int(sample_rate)
        feature_extractor = getattr(processor, 'fe', None)
        self.feature_extractor = feature_extractor
        self.n_fft = int(getattr(feature_extractor, 'n_fft', 400))
        self.hop_length = int(getattr(feature_extractor, 'hop_length', MEL_HOP_LENGTH))
        self.feature_size = int(getattr(feature_extractor, 'feature_size', 128))
        self.reset()

    @property
    def num_samples(self):
        return int(self._total_samples)

    @property
    def buffered_samples(self):
        return int(self._buffer.size)

    def reset(self):
        import numpy as np

        self._buffer = np.zeros(0, dtype=np.float32)
        self._buffer_start = 0
        self._total_samples = 0
        self._next_frame = 0
        self._finalized = False
        self.emitted_frames = 0

    def _empty_features(self):
        return torch.zeros(self.feature_size, 0, dtype=torch.float32)

    def _extract_frames(self, first_frame, frame_count):
        import torch.nn.functional as F

        half_window = self.n_fft // 2
        absolute_start = first_frame * self.hop_length - half_window
        absolute_end = (
            (first_frame + frame_count - 1) * self.hop_length + half_window
        )
        available_start = max(absolute_start, 0)
        available_end = min(absolute_end, self._total_samples)
        local_start = available_start - self._buffer_start
        local_end = available_end - self._buffer_start
        waveform = torch.from_numpy(self._buffer[local_start:local_end])
        left_pad = max(-absolute_start, 0)
        right_pad = max(absolute_end - self._total_samples, 0)
        if left_pad or right_pad:
            waveform = F.pad(
                waveform[None, None], (left_pad, right_pad), mode='reflect'
            ).reshape(-1)

        fe = self.feature_extractor
        window = torch.hann_window(self.n_fft)
        stft = torch.stft(
            waveform,
            self.n_fft,
            self.hop_length,
            window=window,
            return_complex=True,
            center=False,
        )
        magnitudes = stft.abs().square()
        mel_filters = torch.from_numpy(fe.mel_filters).to(torch.float32)
        mel_spec = mel_filters.T @ magnitudes
        log_spec = mel_spec.clamp_min(1e-10).log10()
        log_spec_max = log_spec.new_tensor(fe.global_log_mel_max)
        log_spec = torch.maximum(log_spec, log_spec_max - 8.0)
        return (log_spec + 4.0) / 4.0

    def feed(self, wav, finalize=False):
        import numpy as np

        if torch.is_tensor(wav):
            wav = wav.detach().cpu().numpy()
        chunk = np.asarray(wav, dtype=np.float32).reshape(-1)
        if self._finalized and chunk.size == 0:
            return self._empty_features()
        if chunk.size:
            self._buffer = np.concatenate([self._buffer, chunk])
            self._total_samples += int(chunk.size)
        if finalize:
            self._finalized = True
        if self._total_samples == 0:
            return self._empty_features()

        if self._finalized:
            take = self._total_samples // self.hop_length
        else:
            stable = (self.num_samples - self.n_fft // 2) // self.hop_length + 1
            take = max(int(stable), 0)
        take = max(int(take), self._next_frame)
        frame_count = take - self._next_frame
        if frame_count == 0:
            return self._empty_features()

        out = self._extract_frames(self._next_frame, frame_count)
        self._next_frame = take
        self.emitted_frames = take

        next_start = max(
            self._next_frame * self.hop_length - self.n_fft // 2,
            0,
        )
        discard = next_start - self._buffer_start
        if discard > 0:
            self._buffer = self._buffer[discard:].copy()
            self._buffer_start = next_start
        return out


class MelReplayStream:
    """重放预录制的 mel chunk 序列，接口与 `VoxtralWaveformStream.feed` 兼容。

    用途（同源多候选）：同一条 session 的 k 个候选，mel 序列由 (wav, chunk 切分)
    决定、与候选无关 —— 第一个候选正常计算时把每 chunk 的 mel 留存，后续候选用
    本类重放，省掉 k-1 次完整的 STFT/mel 前端计算（占缓存命中后推理墙钟的 ~33%，
    是 CPU 饱和的大头）。

    正确性：feed 的调用序列在预跑与重放间一一对应（同一 chunk 迭代器），
    返回值就是当初计算的同一个 tensor（无重算、无精度损失）。
    """

    def __init__(self, mel_chunks):
        self._mel_chunks = list(mel_chunks)
        self._cursor = 0
        self.emitted_frames = sum(m.size(-1) for m in self._mel_chunks)

    def feed(self, wav, finalize=False):
        del wav, finalize  # 输入波形与结束标志都由预跑序列决定
        if self._cursor >= len(self._mel_chunks):
            return torch.zeros(0, 0, dtype=torch.float32) \
                if not self._mel_chunks else self._mel_chunks[-1].new_zeros(
                    self._mel_chunks[-1].size(0), 0)
        mel = self._mel_chunks[self._cursor]
        self._cursor += 1
        return mel


class VoxtralAudioBackend(nn.Module):
    """Voxtral encoder + adapter（全程 frozen），提供整段与流式两种接口。

    输出维度 3072（Voxtral text_config.hidden_size）、帧率 12.5Hz（80ms/帧），
    故必须在构造 audio_proj **之前**调用 `attach_to_config(config)`。
    """

    def __init__(self, encoder, adapter, downsample_factor=4):
        super().__init__()
        self.encoder = encoder
        self.adapter = adapter
        self.downsample_factor = downsample_factor
        for p in self.parameters():
            p.requires_grad = False
        self.eval()

    # ---------- 加载 ----------

    @staticmethod
    def load(path, device='cpu', dtype=None):
        """从本地目录或 HF repo id 加载 encoder + adapter（不加载 3B 的 text decoder）。

        Returns: (VoxtralAudioBackend, VoxtralAudioProcessor) 或 (None, None)
        """
        if path is None:
            return None, None
        # 本仓库钉 transformers 4.57.3（ms-swift 4.4.0），官方 voxtral_realtime 只在 >=5.2，
        # 故用逐元素对拍过的兼容实现（voxtral_compat.py / test_voxtral_compat.py）。
        from .voxtral_compat import VoxtralFeatureExtractor, load_voxtral_encoder_projector
        if not os.path.isdir(path):
            warnings.warn(f"[VoxtralAudioBackend] 路径不存在: {path}")
            return None, None

        try:
            encoder, adapter, downsample_factor = load_voxtral_encoder_projector(path)
            _load_encoder_adapter_weights(path, encoder, adapter)
        except Exception as exc:
            warnings.warn(f"[VoxtralAudioBackend] 加载失败: {exc}")
            return None, None

        backend = VoxtralAudioBackend(encoder, adapter, downsample_factor)
        if dtype is not None:
            backend = backend.to(dtype)
        backend = backend.to(device)

        processor = None
        try:
            processor = VoxtralAudioProcessor(VoxtralFeatureExtractor.from_pretrained(path))
        except Exception as exc:
            warnings.warn(f"[VoxtralAudioBackend] FeatureExtractor 加载失败: {exc}")
        return backend, processor

    @staticmethod
    def from_config_dir(path):
        """只按 Voxtral 目录的 config.json 构建结构（随机权重），供 HF `from_pretrained`
        先建空壳、再由 state_dict 统一灌入 `audio_encoder.*` 权重。"""
        from .voxtral_compat import load_voxtral_encoder_projector
        encoder, adapter, downsample_factor = load_voxtral_encoder_projector(path)
        return VoxtralAudioBackend(encoder, adapter, downsample_factor)

    @property
    def hidden_size(self):
        """adapter 输出维度（= Voxtral text hidden，通常 3072）。"""
        return self.adapter.linear_2.out_features

    @property
    def frame_rate(self):
        return 100.0 / 2 / self.downsample_factor    # conv stride 2 -> 12.5Hz

    # ---------- 前向 ----------

    def _apply_adapter(self, feats):
        """encoder 50Hz 输出 -> 每 downsample_factor 帧拼接 -> adapter -> 12.5Hz"""
        b, t, d = feats.shape
        usable = (t // self.downsample_factor) * self.downsample_factor
        if usable == 0:
            return feats.new_zeros(b, 0, self.hidden_size), feats[:, :0]
        x = feats[:, :usable].reshape(b, usable // self.downsample_factor,
                                      d * self.downsample_factor)
        return self.adapter(x), feats[:, usable:]

    @torch.no_grad()
    def encode_audio(self, mel, mel_lens=None):
        """整段编码（离线路径 / 训练用）。

        Args:
            mel: (B, 128, T) log-mel
            mel_lens: (B,) 每条真实 mel 帧数，用于裁掉 padding 对应的输出帧
        Returns: list[Tensor(n_frames, hidden)]，逐样本按真实长度裁剪
        """
        out = self.encoder(input_features=mel)
        feats = out.last_hidden_state
        adapted, _ = self._apply_adapter(feats)
        if mel_lens is None:
            return [adapted[i] for i in range(adapted.size(0))]
        res = []
        for i in range(adapted.size(0)):
            n = max(1, int(mel_lens[i]) // MEL_FRAMES_PER_ADAPTER_FRAME)
            res.append(adapted[i, :min(n, adapted.size(1))])
        return res

    @torch.no_grad()
    def encode_audio_chunk(self, chunk_mel, cache=None):
        """流式增量编码（同传主路径）。

        Args:
            chunk_mel: (B, 128, t) 本 chunk 的 log-mel；t 不必是 8 的整数倍，
                       不足一个 adapter 帧的余数会自动留到下次（避免静默丢帧）
            cache: VoxtralStreamCache；None 时新建
        Returns: (adapted_frames (B, n, hidden), cache)
        """
        cache = cache or VoxtralStreamCache()
        if chunk_mel is None or chunk_mel.numel() == 0:
            return chunk_mel.new_zeros(1, 0, self.hidden_size) if chunk_mel is not None else None, cache

        # 拼接上次留下的 mel 余数
        if cache.mel_remainder is not None and cache.mel_remainder.size(-1) > 0:
            chunk_mel = torch.cat([cache.mel_remainder, chunk_mel], dim=-1)
            cache.mel_remainder = None

        # 只送入 8 帧整数倍的部分，其余留到下次（保证 adapter 无余数、无丢帧）
        t = chunk_mel.size(-1)
        usable = (t // MEL_FRAMES_PER_ADAPTER_FRAME) * MEL_FRAMES_PER_ADAPTER_FRAME
        if usable < t:
            cache.mel_remainder = chunk_mel[..., usable:].clone()
        if usable == 0:
            return chunk_mel.new_zeros(chunk_mel.size(0), 0, self.hidden_size), cache

        out = self.encoder(
            input_features=chunk_mel[..., :usable],
            past_key_values=cache.past_key_values,
            padding_cache=cache.padding_cache,
            use_cache=True, use_padding_cache=True)
        cache.past_key_values = out.past_key_values
        cache.padding_cache = out.padding_cache
        adapted, _ = self._apply_adapter(out.last_hidden_state)
        cache.n_mel_seen += usable
        cache.n_frames_out += adapted.size(1)
        return adapted, cache

    # ---------- 配置协助 ----------

    def attach_to_config(self, config):
        """把 audio_hidden_size 改成 Voxtral 的输出维度（512 -> 3072）。

        必须在构造 audio_proj **之前**调用，否则投影层输入维度不匹配。
        """
        config.audio_hidden_size = self.hidden_size
        config.audio_frame_rate = self.frame_rate
        config.audio_backend = 'voxtral'
        return config


def _load_encoder_adapter_weights(path, encoder, adapter):
    """只挑 encoder / adapter 的权重加载，跳过 3B text decoder（省显存与时间）。"""
    import glob
    import json

    files = sorted(glob.glob(os.path.join(path, '*.safetensors')))
    if not files:
        files = sorted(glob.glob(os.path.join(path, 'pytorch_model*.bin')))
    if not files:
        raise FileNotFoundError(f"{path} 下未找到权重文件")

    index_path = os.path.join(path, 'model.safetensors.index.json')
    wanted_prefix = ('audio_tower.', 'multi_modal_projector.')
    if os.path.exists(index_path):
        with open(index_path) as f:
            weight_map = json.load(f)['weight_map']
        files = sorted({os.path.join(path, v) for k, v in weight_map.items()
                        if k.startswith(wanted_prefix)})

    enc_sd, ada_sd = {}, {}
    for fp in files:
        sd = _read_state_dict(fp)
        for k, v in sd.items():
            if k.startswith('audio_tower.'):
                enc_sd[k[len('audio_tower.'):]] = v
            elif k.startswith('multi_modal_projector.'):
                ada_sd[k[len('multi_modal_projector.'):]] = v

    if not enc_sd:
        raise RuntimeError(f"{path} 中未找到 audio_tower.* 权重")
    missing, unexpected = encoder.load_state_dict(enc_sd, strict=False)
    if missing:
        warnings.warn(f"[VoxtralAudioBackend] encoder 缺失 {len(missing)} 个权重，如 {missing[:3]}")
    if ada_sd:
        adapter.load_state_dict(ada_sd, strict=False)
    else:
        warnings.warn("[VoxtralAudioBackend] 未找到 multi_modal_projector.* 权重，adapter 保持随机初始化")


def _read_state_dict(fp):
    if fp.endswith('.safetensors'):
        from safetensors.torch import load_file
        return load_file(fp)
    return torch.load(fp, map_location='cpu', weights_only=True)


# ============================================================================
#                   音频编码 + <|audio_pad|> 注入
# ============================================================================
class AudioEncoderMixin:
    """把源音频编码成 thinker hidden，并填进 `<|audio_pad|>` 占位符。

    整理自 `MiniMindOmni` 的音频通路。原实现里这些方法被三个模型类用
    `xxx = MiniMindOmni.xxx` 的方式逐个「借用」，漏借一个的表现是 AttributeError
    且只在流式路径才炸（原注释明确记录过这类潜伏失败）；改为 mixin 后
    借用关系由继承保证。

    使用方需提供 `self.audio_encoder`（VoxtralAudioBackend）、`self.audio_proj`
    （MMAudioProjector）与 `self.config.audio_ids`。
    """

    @staticmethod
    def load_audio_encoder(path, backend='voxtral', device='cpu', dtype=None):
        """加载 Voxtral 流式音频编码器。

        注意：voxtral 输出维度 3072、帧率 12.5Hz，故必须在构造 audio_proj **之前**
        调用 `backend.attach_to_config(config)` 改掉 `audio_hidden_size`，
        否则投影层输入维度不匹配。
        """
        if backend != 'voxtral':
            raise ValueError(
                f"流式同传只支持 voxtral 后端，收到 {backend!r}。"
                f"SenseVoice 的 SANM 是双向注意力，无法增量前向。")
        return VoxtralAudioBackend.load(path, device=device, dtype=dtype)

    @torch.compiler.disable
    def encode_audio_chunk(self, chunk_mel, cache=None):
        """流式增量编码单个 audio chunk。

        Returns: (投影到 thinker hidden 的 (B, n, hidden), cache)
        """
        if self.audio_encoder is None or not hasattr(self.audio_encoder, 'encode_audio_chunk'):
            raise RuntimeError("流式音频编码需要 voxtral 后端；SenseVoice 的 SANM 是双向注意力，"
                               "无法增量前向（见 docs/streaming_simul_design.md §1.1）")
        enc_dtype = next(self.audio_encoder.parameters()).dtype
        feats, cache = self.audio_encoder.encode_audio_chunk(
            chunk_mel.to(dtype=enc_dtype), cache)
        if feats is None or feats.size(1) == 0:
            return feats, cache
        proj_dtype = next(self.audio_proj.parameters()).dtype
        return self.audio_proj(feats.to(proj_dtype)), cache

    @torch.compiler.disable
    def encode_audio_inputs(self, audio_inputs, audio_lens=None):
        """整段编码（训练路径）。输入 (B, 128, T) log-mel，编码器内部已含 adapter 下采样。"""
        if (audio_inputs is None) or (self.audio_encoder is None) or (not audio_inputs.any()):
            return None
        batch_mask = audio_inputs.flatten(1).any(1)
        enc_dtype = next(self.audio_encoder.parameters()).dtype
        valid = audio_inputs[batch_mask].to(dtype=enc_dtype)
        valid_lens = audio_lens[batch_mask] if audio_lens is not None else None
        feats = self.audio_encoder.encode_audio(valid, valid_lens)
        proj_dtype = next(self.audio_proj.parameters()).dtype
        emb_list = [self.audio_proj(f.unsqueeze(0).to(proj_dtype)).squeeze(0) for f in feats]
        if batch_mask.all():
            return emb_list
        out, j = [None] * audio_inputs.size(0), 0
        for i in range(audio_inputs.size(0)):
            if batch_mask[i]:
                out[i] = emb_list[j]
                j += 1
        return out

    @torch.compiler.disable
    def inject_audio_features(self, tokens, h, audio_feats, seqlen=None,
                              audio_seg_lens=None, audio_feature_offsets=None):
        """把音频特征按序填进 input_ids 里的 `<|audio_pad|>` 占位符。

        **支持多段占位符**（chunk-as-turn：一条样本有 K 个 chunk，序列里就有 K 段
        audio_pad）。同一份 `audio_feats[b]` 按**出现顺序跨段连续消费**：第 1 段吃前
        n_0 帧、第 2 段接着吃 n_1 帧 …… 与流式 encoder 逐块产出的顺序完全一致，
        因此训练时可以对整段 mel 调一次 `encode_audio`（因果 encoder，整段 vs 逐块
        实测 max|diff| 2.2e-06），不必真的逐块前向。

        实现用「clone + 一次性 index_put」而非逐段 `torch.cat` 重拼：
        最多 20 段 × batch 32 = 640 次 cat，每次都要复制整条序列，开销远大于
        一次 clone（B=32/L=400/D=1024 bf16 约 26MB）。语义等价、且仍然 out-of-place
        （对 h 原地写会破坏 autograd 对 embed_tokens 输出的依赖）。

        Args:
            audio_seg_lens: 可选，`list[Tensor]`，每条样本各段的**期望帧数**。给定时
                做严格校验并在不符时 **raise** —— 帧数错位是本函数唯一的失效模式，
                而它天然是**静默**的（形状照样对得上，只是模型听到的内容错位/缺尾），
                所以流式训练必须把它传进来。
            audio_feature_offsets: 可选，每条样本在完整 encoder 特征流中的起始帧。
                None 保留单段路径；给定时必须同时给 audio_seg_lens，并严格满足
                offset + suffix marker count == 该样本真实 encoder 特征数。
        """
        # 单段路径（不传 offsets）：允许 min 截断，不引入额外校验、转换或 clone。
        if audio_feature_offsets is None:
            if audio_feats is None or not self.config.audio_ids:
                return h
            marker = self.config.audio_ids[0]
            mask = (tokens == marker)
            if not bool(mask.any()):
                return h
            out = None
            for b in range(h.size(0)):
                af = audio_feats[b] if audio_feats[b] is not None else None
                if af is None:
                    continue
                idx = mask[b].nonzero(as_tuple=True)[0]
                if idx.numel() == 0:
                    continue
                if audio_seg_lens is not None:
                    self._check_audio_segments(idx, af.size(0), audio_seg_lens[b], b)
                n = min(idx.numel(), af.size(0))
                if out is None:
                    out = h.clone()
                out[b, idx[:n]] = af[:n].to(dtype=out.dtype)
            return h if out is None else out

        if audio_seg_lens is None:
            raise ValueError('audio_feature_offsets requires audio_seg_lens')
        batch_size = h.size(0)
        if torch.is_tensor(audio_feature_offsets):
            if audio_feature_offsets.ndim != 1 or audio_feature_offsets.numel() != batch_size:
                raise ValueError(
                    f'audio_feature_offsets batch length must equal {batch_size}')
            if audio_feature_offsets.dtype not in (
                    torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
                raise ValueError('audio_feature_offsets must contain integer values')
            offsets = audio_feature_offsets.detach().cpu().tolist()
        else:
            try:
                raw_offsets = list(audio_feature_offsets)
            except TypeError as exc:
                raise ValueError('audio_feature_offsets must be a batch-length sequence') from exc
            if len(raw_offsets) != batch_size:
                raise ValueError(
                    f'audio_feature_offsets batch length must equal {batch_size}')
            try:
                offsets = [operator.index(value) for value in raw_offsets]
            except TypeError as exc:
                raise ValueError('audio_feature_offsets must contain integer values') from exc
        if any(offset < 0 for offset in offsets):
            raise ValueError('audio_feature_offsets must be nonnegative')
        if len(audio_seg_lens) != batch_size:
            raise ValueError(f'audio_seg_lens batch length must equal {batch_size}')
        if audio_feats is None or len(audio_feats) != batch_size:
            raise RuntimeError('audio feature batch does not match hidden-state batch')
        if not self.config.audio_ids:
            return h
        marker = self.config.audio_ids[0]
        mask = (tokens == marker)
        if not bool(mask.any()):
            return h

        out = None
        for b in range(batch_size):
            idx = mask[b].nonzero(as_tuple=True)[0]
            if idx.numel() == 0:
                continue
            af = audio_feats[b]
            if af is None:
                raise RuntimeError(
                    f'[inject_audio_features] sample {b} has audio markers but no encoder features')
            offset = offsets[b]
            self._check_audio_segments(
                idx, af.size(0), audio_seg_lens[b], b, offset=offset)
            n = idx.numel()
            selected = af[offset:offset + n]
            if selected.size(0) != n:
                raise RuntimeError(
                    f'[inject_audio_features] sample {b} audio feature offset exceeds '
                    f'encoded feature sequence')
            if out is None:
                out = h.clone()
            out[b, idx] = selected.to(device=out.device, dtype=out.dtype)
        return h if out is None else out

    @staticmethod
    def _check_audio_segments(idx, n_feat, expect_lens, b, offset=0):
        """校验 marker 段布局以及 offset + suffix 总长与完整特征流一致。"""
        expect = [int(x) for x in (expect_lens.tolist() if torch.is_tensor(expect_lens)
                                   else expect_lens) if int(x) > 0]
        # idx 是升序的 marker 下标；相邻差 != 1 处即为段边界
        breaks = (idx[1:] - idx[:-1] != 1).nonzero(as_tuple=True)[0]
        bounds = [0] + [int(x) + 1 for x in breaks] + [idx.numel()]
        got = [bounds[i + 1] - bounds[i] for i in range(len(bounds) - 1)]
        if got != expect:
            raise RuntimeError(
                f'[inject_audio_features] 样本 {b} 的 audio_pad 分段与期望不符:\n'
                f'  序列里 {len(got)} 段 {got[:12]}{"..." if len(got) > 12 else ""}\n'
                f'  期望{len(expect)} 段 {expect[:12]}{"..." if len(expect) > 12 else ""}\n'
                f'  多半是 dataset 的逐步帧数算法与 encoder 不一致（1 秒 chunk =12.5 个'
                f' 80ms 帧，必须按**累计** mel 帧整除，不能每步都取 12）。')
        if offset + sum(expect) != n_feat:
            raise RuntimeError(
                f'[inject_audio_features] 样本 {b} 的 audio feature offset {offset} + '
                f'audio_pad 总数 {sum(expect)} != encoder 产出帧数 {n_feat}。'
                f'检查完整 mel、crop offset 与保留段长度是否对齐。')


def load_thinker(config, thinker_path=None, model_cls=None):
    """加载 Qwen3 thinker 主干；路径缺失时按 config 超参随机初始化（便于结构自测）。

    已有权重目录走 AutoModelForCausalLM，因此不同 model_type 都能自动识别。
    """
    if thinker_path and os.path.exists(thinker_path):
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(thinker_path, dtype=torch.float32)
        if model.config.hidden_size != config.hidden_size or \
                model.config.num_hidden_layers != config.num_hidden_layers:
            warnings.warn(
                f"[load_thinker] thinker 结构与 config 不一致: "
                f"({model.config.hidden_size}d/{model.config.num_hidden_layers}L) vs "
                f"({config.hidden_size}d/{config.num_hidden_layers}L)，已按实际权重为准。")
        return model
    warnings.warn(f"[load_thinker] Qwen thinker path not found: {thinker_path}，将随机初始化主干！")
    if model_cls is None:
        from transformers.models.qwen3 import Qwen3ForCausalLM
        model_cls = Qwen3ForCausalLM
    return model_cls(config.to_qwen_config())

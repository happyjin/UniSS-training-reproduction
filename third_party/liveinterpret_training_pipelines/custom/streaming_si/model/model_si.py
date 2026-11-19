"""
Streaming-SI 流式同传模型本体
=============================
    SITalker        —— Qwen3 风格 speech token 解码器（层数/宽度由 talker_* 配置决定）
    SI              —— Thinker + 桥接融合 + Talker + Flow/HiFT 声码

基础设施（配置、Voxtral 音频前端、流式 token 协议）见 `base_model.py`。

架构（本文件两个变体共用，thinker / talker 规格由 config 决定，见 `variants.py`）：
    Voxtral 流式 encoder(frozen) -> thinker -> GatedBridge+Fusion
    -> Talker -> Fun-CosyVoice3 flow/HiFT -> 24kHz PCM

    变体              thinker                     talker        audio_proj
    streaming_si      Qwen3-0.6B  28L / 1024d     8L  / 1024d   3072 -> 1024
    streaming_si_4b   Qwen3.5-4B  32L / 2560d     28L / 1024d   3072 -> 2560

**当前在训练与评测的是 `streaming_si_4b`**（32 层 thinker、28 层 talker）。
下文若出现写死的 1024 / 8 层，均为 0.6B 变体的默认值。

三路输入在 talker 前汇合：
    thinker hidden ──GatedBridge──┐
                                  ├──► TalkerFusion ──► talker 前缀
    文本 token embedding ──Linear─┘
    previous speech token ────────────► 在 talker 内部按 AR 进入

本文件只保留流式同传路径。历史实现中的句级两段式推理
（generate / generate_streaming）、固定 `mix_ratio=[5,15]` 的双流交错
（同传改用数据驱动的真实 WRITE 段）、官方 CosyVoice `llm.pt` 加载
（自研 talker 从零训练）、视觉与 SenseVoice 通路均已剔除。
"""
import inspect
import os
import warnings
from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn
from transformers import GenerationMixin, PreTrainedModel
from transformers.modeling_outputs import BaseModelOutputWithPast, MoeCausalLMOutputWithPast
from transformers.models.qwen3 import Qwen3Model

from .base_model import (AudioEncoderMixin, MMAudioProjector, OmniSIConfig, RMSNorm,
                         load_thinker)


def load_cosy_flow_hift(model_dir, config_yaml='cosyvoice3.yaml', device='cpu',
                        cosyvoice_repo=None):
    """只实例化官方 CosyVoice 的 flow + HiFT（speech token -> 24kHz 波形）。

    `SI.attach_speech_decoder` 与数据验收脚本（`data_prep/verify_fsq_data.py`）共用。
    yaml 里的 `llm` 等训练/前端组件全部覆盖为 None —— 自研 talker 不用官方 llm，
    实例化它会额外加载 0.5B Qwen2 权重。

    `cosyvoice_repo` 会被挂到 sys.path（官方包不是 pip 包，需 `--recursive` 取到
    third_party/Matcha-TTS）。失败直接抛异常，由调用方决定是否降级。
    """
    import sys as _sys
    for p in ([cosyvoice_repo, os.path.join(cosyvoice_repo, 'third_party', 'Matcha-TTS')]
              if cosyvoice_repo else []):
        if not os.path.isdir(p):
            raise RuntimeError(f'CosyVoice 仓库路径不存在: {p}')
        if p not in _sys.path:
            _sys.path.insert(0, p)
    from hyperpyyaml import load_hyperpyyaml
    decode_unused = (
        'llm', 'mel_spec_transform1', 'hifigan',
        'parquet_opener', 'get_tokenizer', 'tokenize', 'filter',
        'resample', 'truncate', 'feat_extractor', 'compute_fbank',
        'compute_f0', 'parse_embedding', 'shuffle', 'sort', 'batch',
        'padding', 'data_pipeline', 'data_pipeline_gan',
    )
    with open(os.path.join(model_dir, config_yaml), 'r') as f:
        configs = load_hyperpyyaml(
            f,
            overrides={
                'qwen_pretrain_path': os.path.join(model_dir, 'CosyVoice-BlankEN'),
                **{name: None for name in decode_unused},
            },
            overrides_must_match=False,
        )
    flow, hift = configs['flow'], configs['hift']
    flow.load_state_dict(torch.load(os.path.join(model_dir, 'flow.pt'), map_location='cpu'), strict=True)
    hift.load_state_dict({k.replace('generator.', ''): v for k, v in torch.load(
        os.path.join(model_dir, 'hift.pt'), map_location='cpu').items()}, strict=True)
    return flow.to(device).eval(), hift.to(device).eval()


@dataclass
class SIOutput(MoeCausalLMOutputWithPast):
    """带 talker 输出的 forward 结果。"""

    speech_logits: Optional[torch.FloatTensor] = None
    speech_target: Optional[torch.LongTensor] = None
    # fused text CE 用：`logits is None` 且本字段非 None 时，text loss 由
    # compute_si_losses 直接从 hidden + lm_head.weight 算（不物化 logits）。
    # 见 SI.forward 的 fused_text_ce 分支与 modeling_streaming_si.compute_si_losses。
    text_hidden: Optional[torch.FloatTensor] = None
    lm_head_weight: Optional[torch.FloatTensor] = None


_LIGER_FLCE = None
_LIGER_FLCE_CHECKED = False


def _liger_flce():
    """liger 的 `fused_linear_cross_entropy`（懒加载 + 缓存 + 失败静默回退）。

    融合 `lm_head` 投影与 CE，按 chunk 计算、不物化整个 logits。
    liger-kernel 未安装（基线镜像就没有）或版本过旧时返回 None，调用方回退到原实现。
    模式与 `custom/qwen_omnispeak/modeling_qwen_omnispeak.py` 的 `_get_liger_*` 一致。
    """
    global _LIGER_FLCE, _LIGER_FLCE_CHECKED
    if not _LIGER_FLCE_CHECKED:
        try:
            from liger_kernel.transformers.functional import liger_fused_linear_cross_entropy
            _LIGER_FLCE = liger_fused_linear_cross_entropy
        except Exception:
            _LIGER_FLCE = None
        _LIGER_FLCE_CHECKED = True
    return _LIGER_FLCE


@dataclass
class DecodedSpeechSegment:
    """一个独立 Flow/HiFT segment 的实际输出与配对声学特征。"""

    pcm_chunks: list[torch.Tensor]
    mel: torch.Tensor
    n_tokens: int
    n_samples: int
    session: object = None


class TalkerDecodeGraph:
    """talker 主干的 StaticCache + 单 token 解码步的 CUDA Graph（推理提速）。

    talker 是 28 层 Qwen3，eager 单 token 前向约 28ms，几乎全是 CPU 发射开销（GPU 实际计算 < 6ms）；
    把单 token 步捕获成 CUDA Graph 后约 5.5ms。多 token 的输入（cond 段、[sos]+prompt）仍走 eager，
    写进同一个 StaticCache。Graph 与 eager StaticCache 的输出逐位相同；与 DynamicCache 只差 bf16 舍入
    （sdpa 在定长 KV 上换了内核）。

    Graph 绑定在 StaticCache 的张量地址上，所以每个 talker 只建一个、跨句复用（`reset()` 原地清零）。
    """

    def __init__(self, talker, max_len=2048):
        from transformers import StaticCache
        weight = talker.speech_embedding.weight
        self.decoder = talker.backbone
        self.max_len = int(max_len)
        self.cache = StaticCache(config=self.decoder.config, max_cache_len=self.max_len)
        self.x = torch.zeros(1, 1, self.decoder.config.hidden_size, device=weight.device, dtype=weight.dtype)
        self.cpos = torch.zeros(1, dtype=torch.long, device=weight.device)
        self.graph, self.out, self.pos = None, None, 0

    def reset(self):
        with torch.inference_mode():           # 缓存张量是在 inference_mode 里懒建的，原地清零也要在里面
            self.cache.reset()
        self.pos = 0

    def _forward(self, x, cache_position):
        return self.decoder(inputs_embeds=x, past_key_values=self.cache, use_cache=True,
                            cache_position=cache_position).last_hidden_state

    def _check(self, n):
        if self.pos + n > self.max_len:
            raise RuntimeError(f'talker 序列超过 StaticCache 上限 {self.max_len}（已有 {self.pos}，追加 {n}）')

    def prefill(self, x):
        """(1, L, D) 的多 token 输入：eager 前向。"""
        self._check(x.size(1))
        h = self._forward(x, torch.arange(self.pos, self.pos + x.size(1), device=x.device))
        self.pos += x.size(1)
        return h

    def step(self, x):
        """(1, 1, D) 的单 token 输入：重放 CUDA Graph。返回的张量下次调用会被覆盖。"""
        self._check(1)
        self.x.copy_(x)
        self.cpos.fill_(self.pos)
        if self.graph is None:
            # 预热（在当前位置写入的 KV 与重放时相同，可重复写）后捕获
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    self._forward(self.x, self.cpos)
            torch.cuda.current_stream().wait_stream(stream)
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                self.out = self._forward(self.x, self.cpos)
        self.graph.replay()
        self.pos += 1
        return self.out


class TalkerStreamState:
    """双流交错生成的跨段状态。

    交错模式下每段只产一小段 token，若把 RAS 的重复检测窗口 `decoded` 放在
    生成器局部变量里，每段都会重置 -> win_size=10 的复读抑制几乎不触发。
    因此把它连同 KV cache 一起外置到本对象，跨段累积。

    `cuda_graph=True`（仅 CUDA）：主干改用 `TalkerDecodeGraph`（StaticCache + 单 token CUDA Graph），
    同一个 talker 上同一时刻只能有一个这样的状态在用（graph 与缓存跨句复用）。
    """

    def __init__(self, talker, prompt_tokens=None, dtype=None, device=None, cuda_graph=False,
                 max_len=2048):
        self.dtype = dtype or talker.speech_embedding.weight.dtype
        self.device = device or talker.speech_embedding.weight.device
        pieces = [talker.sos_embed(self.dtype, self.device)]
        if prompt_tokens is not None and prompt_tokens.numel() > 0:
            pieces.append(talker.speech_embedding(prompt_tokens.to(self.device)).to(self.dtype))
        self.pending = torch.cat(pieces, dim=1)  # 待喂入的前缀（[sos] + prompt）
        self.past_key_values = None
        self.graph = None
        if cuda_graph and torch.device(self.device).type == 'cuda':
            self.graph = talker.decode_graph(max_len)
            self.graph.reset()
        self.decoded = []          # 跨段累积，供 RAS 复读抑制
        self.cond_consumed = 0     # 已消费的 cond token 总数
        self.finished = False
        # ---- fill_token 学习效果统计（S1-1 的首要成败判据）----
        # 背景: stream_generate_s4 的 segment_cap 兜底(段满即交回控制权)会**掩盖**
        # "模型从未学会采 fill_token" 这一失败 —— 流程照样跑通、验证脚本照样 PASS、
        # 音频照样连贯, 但实际退化为 hard-coded 定长交错, 自适应对齐的全部价值静默丢失。
        # 故必须显式计数, 让失败可观测。
        self.n_seg = 0             # 总段数
        self.n_fill = 0            # 采到 fill_token 主动让出的段数
        self.n_segcap = 0          # 段满被动交回的段数（持续 ≈100% 即学废）
        self.seg_lens = []         # 每段实际产出的 token 数（方差=0 即无自适应）
        self.last_stop_reason = None

    @property
    def fill_rate(self):
        """主动让出控制权的比例。应显著 > 0；接近 0 = fill_token 没学会。"""
        return self.n_fill / self.n_seg if self.n_seg else 0.0

    @property
    def segcap_fallback_rate(self):
        """段满兜底触发率。应随训练下降；持续 ≈1.0 即退化为定长交错。"""
        return self.n_segcap / self.n_seg if self.n_seg else 0.0

    def stream_stats(self):
        import statistics
        return {
            'n_seg': self.n_seg, 'n_fill': self.n_fill, 'n_segcap': self.n_segcap,
            'fill_rate': round(self.fill_rate, 4),
            'segcap_fallback_rate': round(self.segcap_fallback_rate, 4),
            'seg_len_mean': round(statistics.fmean(self.seg_lens), 2) if self.seg_lens else 0.0,
            'seg_len_std': round(statistics.pstdev(self.seg_lens), 3) if len(self.seg_lens) > 1 else 0.0,
        }

    def __len__(self):
        return len(self.decoded)


class GatedBridge(nn.Module):
    """Gated Bridge：thinker hidden -> talker 输入空间，SwiGLU 风格门控 + RMSNorm。"""

    def __init__(self, in_dim, out_dim, eps=1e-6):
        super().__init__()
        self.norm_in = RMSNorm(in_dim, eps=eps)
        self.up = nn.Linear(in_dim, out_dim, bias=False)
        self.gate = nn.Linear(in_dim, out_dim, bias=False)
        self.out = nn.Linear(out_dim, out_dim, bias=False)
        self.norm_out = RMSNorm(out_dim, eps=eps)

    def forward(self, x):
        x = self.norm_in(x)
        return self.norm_out(self.out(self.up(x) * torch.sigmoid(self.gate(x))))


class TalkerFusion(nn.Module):
    """融合「thinker hidden（已过桥接）」与「文本 token embedding」两路文本轴信息。

    门控残差：以桥接后的语义为主干，文本 token embedding 作为可门控的补充，
    避免文本 embedding 直接冲淡 thinker 的上下文语义。
    """

    def __init__(self, dim, eps=1e-6, scale_channel=False):
        super().__init__()
        self.gate = nn.Linear(dim * 2, dim, bias=False)
        self.proj = nn.Linear(dim * 2, dim, bias=False)
        self.norm = RMSNorm(dim, eps=eps)
        # 保留幅值的 scale 通道（config.talker_fusion_scale，默认关）：末尾 RMSNorm 会抹掉逐 token
        # 幅值，探针显示 F0 起伏 / 停顿 / 能量起伏在此处显著下降（SUMMARY §10）。开启后把
        # log-RMS(融合前) 乘一个**零初始化**的可学习方向加回输出——从既有 checkpoint 起步时
        # 输出与关闭时逐位相同，只由训练决定用多少幅值信息。
        self.scale_channel = bool(scale_channel)
        if self.scale_channel:
            self.scale_gain = nn.Parameter(torch.zeros(dim))

    def forward(self, cond, text):
        pair = torch.cat([cond, text], dim=-1)
        gate = torch.sigmoid(self.gate(pair))
        x = cond + gate * self.proj(pair)
        y = self.norm(x)
        if self.scale_channel:
            log_rms = 0.5 * torch.log(x.float().pow(2).mean(-1, keepdim=True) + 1e-6)
            y = y + log_rms.to(y.dtype) * self.scale_gain.to(y.dtype)
        return y


class SITalker(nn.Module):
    """Qwen3 风格 Talker（text/hidden -> speech token 的 AR 模型）。

    层数/宽度取自 `talker_*` 配置，**不是固定 8 层**：
    `streaming_si` 为 8L/1024d（0.1b preset），`streaming_si_4b` 为 28L/1024d（0.6b preset，
    与 Qwen3-0.6B 同形，可用其 transformer 层初始化）。见 `variants.py`。

    主干是从零初始化的 `Qwen3Model`，其 `embed_tokens` 的词表已设为 speech vocab
    (6565)，因此**它本身就是 speech_embedding**，零冗余参数。

    模块命名（decoder / llm_decoder / thinker_proj / text_proj / fusion）与
    checkpoint 中的键一一对应，改名会导致既有权重加载不到。
    """

    def __init__(self, config: OmniSIConfig):
        super().__init__()
        self.config = config
        self.decoder = Qwen3Model(config.to_talker_qwen3_config())
        # speech-token head（与 speech_embedding 不 tie）
        self.llm_decoder = nn.Linear(config.talker_hidden_size, config.talker_vocab_size, bias=False)
        self.thinker_proj = GatedBridge(config.hidden_size, config.talker_hidden_size,
                                        eps=config.rms_norm_eps)
        self.text_proj = nn.Linear(config.hidden_size, config.talker_hidden_size, bias=False)
        self.fusion = TalkerFusion(config.talker_hidden_size, eps=config.rms_norm_eps,
                                   scale_channel=bool(getattr(config, 'talker_fusion_scale', False)))
        self.sos_token = config.talker_sos_token
        self.eos_token = config.talker_eos_token
        self.task_token = config.talker_task_token
        self.fill_token = config.talker_fill_token
        self.stop_token_ids = list(range(config.speech_token_size, config.talker_vocab_size))

    # ---------------- 结构别名 ----------------

    @property
    def backbone(self):
        """Qwen3Model（不含 lm_head，talker 用自己的 llm_decoder 输出 speech token）。"""
        return self.decoder

    @property
    def speech_embedding(self):
        """主干自带的 embed_tokens 就是 speech token embedding（词表已设为 speech vocab）。"""
        return self.decoder.embed_tokens

    @property
    def speech_head(self):
        return self.llm_decoder

    def decode_graph(self, max_len=2048):
        """推理用的 `TalkerDecodeGraph`（按缓存长度分别懒建、跨句复用；不进 state_dict）。

        以前只缓存一张图、请求更小的 max_len 时复用已有的大图，缓存长度因此无法缩小。
        现在按 max_len 各存一张：每张图自带一份 StaticCache（28 层 × K/V × 8 头 × 128 维 × max_len，
        2048 时约 235MB），同一时刻只能有一张在用（与原约束相同，由调用方保证）。
        """
        key = int(max_len)
        graphs = self.__dict__.setdefault('_decode_graphs', {})
        g = graphs.get(key)
        if g is None:
            g = TalkerDecodeGraph(self, max_len=key)
            graphs[key] = g
        return g

    # ---------------- 特殊 token embedding ----------------

    def special_embed(self, token_id, dtype=None, device=None):
        emb = self.speech_embedding.weight[token_id].reshape(1, 1, -1)
        return emb.to(dtype=dtype or emb.dtype, device=device or emb.device)

    def sos_embed(self, dtype=None, device=None):
        return self.special_embed(self.sos_token, dtype, device)

    def task_embed(self, dtype=None, device=None):
        return self.special_embed(self.task_token, dtype, device)

    @staticmethod
    def _pad_batch(seqs, targets, cond_embeds, ignore_index):
        """把逐样本序列 right-pad 成 batch，并生成 attention_mask 与 target。"""
        batch_size = len(seqs)
        device = cond_embeds.device
        max_len = max(s.size(0) for s in seqs)
        inputs = cond_embeds.new_zeros(batch_size, max_len, cond_embeds.size(-1))
        attention_mask = torch.zeros(batch_size, max_len, dtype=torch.long, device=device)
        target_tensor = torch.full((batch_size, max_len), ignore_index, dtype=torch.long, device=device)
        for i, (seq, target) in enumerate(zip(seqs, targets)):
            inputs[i, :seq.size(0)] = seq
            attention_mask[i, :seq.size(0)] = 1
            keep = min(len(target), max_len)
            target_tensor[i, :keep] = torch.tensor(target[:keep], dtype=torch.long, device=device)
        return inputs, attention_mask, target_tensor

    def build_inputs(self, cond_embeds, cond_lens, speech_tokens=None, speech_lens=None,
                     prompt_tokens=None, prompt_lens=None, ignore_index=-100):
        """逐样本拼接 talker 序列并 right-pad（前缀式，用于无段信息的退化路径）。

        输入序列: [sos] + cond(thinker hidden 投影) + [task] + prompt_speech + speech
        监督目标:  ignore * (1 + cond_len + prompt_len) + speech + [eos]

        对齐关系与官方 prepare_lm_input_target 一致：紧邻 speech 段前的那个位置
        （无 prompt 时是 task_token，有 prompt 时是 prompt 最后一个 token）预测第 1 个
        speech token。
        """
        batch_size = cond_embeds.size(0)
        dtype, device = cond_embeds.dtype, cond_embeds.device
        sos = self.sos_embed(dtype, device)
        task = self.task_embed(dtype, device)
        seqs, targets = [], []
        for i in range(batch_size):
            cond_len = int(cond_lens[i])
            pieces = [sos[0], cond_embeds[i, :cond_len], task[0]]
            target = [ignore_index] * (1 + cond_len)
            if prompt_tokens is not None and prompt_lens is not None and int(prompt_lens[i]) > 0:
                p_len = int(prompt_lens[i])
                pieces.append(self.speech_embedding(prompt_tokens[i, :p_len]).to(dtype))
                target += [ignore_index] * p_len
            if speech_tokens is not None and speech_lens is not None and int(speech_lens[i]) > 0:
                s_len = int(speech_lens[i])
                codes = speech_tokens[i, :s_len]
                pieces.append(self.speech_embedding(codes).to(dtype))
                target += codes.tolist() + [self.eos_token]
            else:
                target += [ignore_index]  # 无语音监督的样本
            seqs.append(torch.cat(pieces, dim=0))
            targets.append(target)
        return self._pad_batch(seqs, targets, cond_embeds, ignore_index)

    @staticmethod
    def _validate_segs(idx, text_segs, speech_segs, cond_len, speech_len):
        """段长必须与 cond / speech 的真实长度**逐项**对得上。

        错位在这里是唯一的失效模式，而且天然静默：形状照样拼得出来，只是第 i 段
        文本配到了第 j 段语音上，talker loss 会不降但不报错。故一律 raise。
        """
        if len(text_segs) != len(speech_segs):
            raise ValueError(f'[样本 {idx}] text/speech 段数不等: '
                             f'{len(text_segs)} vs {len(speech_segs)}')
        if sum(text_segs) != cond_len:
            raise ValueError(f'[样本 {idx}] Σtext_seg_lens={sum(text_segs)} '
                             f'≠ cond_len={cond_len}（answer_mask 与 text_seg_lens 不同源？）')
        if sum(speech_segs) != speech_len:
            raise ValueError(f'[样本 {idx}] Σspeech_seg_lens={sum(speech_segs)} '
                             f'≠ speech_len={speech_len}')
        if text_segs and min(text_segs) < 1:
            raise ValueError(f'[样本 {idx}] 存在 0 长文本段；每个 WRITE 步至少要有 1 个译文 token')

    def build_seg_interleaved_inputs(self, cond_embeds, cond_lens, speech_tokens=None,
                                     speech_lens=None, prompt_tokens=None, prompt_lens=None,
                                     ignore_index=-100, text_seg_lens=None, speech_seg_lens=None):
        """数据驱动的双流交错（S4 训练主路径）。

        固定 `mix_ratio=[T, S]` 假设文本与语音以恒定比例推进；而同传的真实节奏由
        read-write 策略决定 —— 第 i 个 WRITE 步说了 `text_seg_lens[i]` 个译文 token、
        对应 `speech_seg_lens[i]` 帧语音。按真实段长交错，talker 学到的「读到这里
        该说多长」才与推理时驱动循环给出的节奏一致。

        序列布局（W 段）：

            [sos] + prompt_speech
                  + cond[段0]   + speech[段0]     # 段末预测 fill_token，交还文本轴
                  + ...
                  + cond[段W-1] + speech[段W-1]   # 同样以 fill_token 收尾
                  + [剩余纯文本段] + [task] -> eos

        **每段都以 fill_token 收尾、eos 只出现在 [task] 位**，理由是要对齐
        `stream_generate_s4` 的实际行为：非 finalize 的段 `ignore_eos=True`（只能吐
        fill 让出控制权），只有驱动循环在 thinker 说完 eos 后调 `finalize=True` 才喂
        [task] 并放开 eos。数据驱动交错下每个 speech token 都属于某个 WRITE 步，
        本就没有"除不尽的余数尾巴"，于是"何时结束"完全交给 thinker 决定。

        边界情况 `speech_seg_lens[i] == 0`（该 WRITE 步只出文本、没落到新的 codec 帧）：
        与**后**一段合并。不合并的话这段末位要预测的「下一个 speech token」其实属于
        再后面的段，整条对齐关系会错开一位。末尾若还剩这样的纯文本段，它排在最后一段
        语音之后（时间序如此），直接接 [task]。

        Returns: (inputs, attention_mask, target)，与 `build_inputs` 同构。
        """
        if text_seg_lens is None or speech_seg_lens is None:
            raise ValueError('build_seg_interleaved_inputs 需要 text_seg_lens / speech_seg_lens')
        batch_size = cond_embeds.size(0)
        dtype, device = cond_embeds.dtype, cond_embeds.device
        sos, task = self.sos_embed(dtype, device), self.task_embed(dtype, device)
        emb_all, codes_cpu = None, None
        if speech_tokens is not None and speech_lens is not None:
            emb_all = self.speech_embedding(speech_tokens).to(dtype)   # (B, S, D) 一次算完
            codes_cpu = speech_tokens.tolist()                          # 一次 D2H
        prompt_emb_all = None
        if prompt_tokens is not None and prompt_lens is not None:
            prompt_emb_all = self.speech_embedding(prompt_tokens).to(dtype)
        cond_lens_cpu = cond_lens.tolist()
        speech_lens_cpu = speech_lens.tolist() if speech_lens is not None else [0] * batch_size
        prompt_lens_cpu = prompt_lens.tolist() if prompt_lens is not None else [0] * batch_size

        seqs, targets = [], []
        for i in range(batch_size):
            cond_len = int(cond_lens_cpu[i])
            s_len = int(speech_lens_cpu[i]) if emb_all is not None else 0
            codes = codes_cpu[i] if codes_cpu is not None else None    # python list, 无同步
            t_segs = [int(x) for x in text_seg_lens[i].tolist()]
            s_segs = [int(x) for x in speech_seg_lens[i].tolist()]
            self._validate_segs(i, t_segs, s_segs, cond_len, s_len)

            # 合并 0 语音段：pending 累积到下一个真正有语音的段
            pairs, pending = [], 0
            for t_n, s_n in zip(t_segs, s_segs):
                pending += t_n
                if s_n > 0:
                    pairs.append((pending, s_n))
                    pending = 0

            pieces, target = [sos[0]], [ignore_index]
            if prompt_emb_all is not None and int(prompt_lens_cpu[i]) > 0:
                p_len = int(prompt_lens_cpu[i])
                pieces.append(prompt_emb_all[i, :p_len])
                target += [ignore_index] * p_len

            if not pairs:   # 无语音监督样本：退化为纯 cond 前缀
                pieces.append(cond_embeds[i, :cond_len])
                target += [ignore_index] * cond_len
                pieces.append(task[0]); target.append(ignore_index)
                seqs.append(torch.cat(pieces, dim=0)); targets.append(target)
                continue

            t_pos, s_pos = 0, 0
            for t_n, s_n in pairs:
                seg = codes[s_pos:s_pos + s_n]
                # cond 段：内部 ignore，末位预测该 speech 段首 token
                pieces.append(cond_embeds[i, t_pos:t_pos + t_n])
                target += [ignore_index] * (t_n - 1) + [seg[0]]
                # speech 段：逐位置预测下一个；末位预测 fill_token（让出控制权）
                pieces.append(emb_all[i, s_pos:s_pos + s_n])
                target += seg[1:] + [self.fill_token]
                t_pos += t_n
                s_pos += s_n

            if cond_len > t_pos:            # 末尾"有文本、无语音"的残留段
                pieces.append(cond_embeds[i, t_pos:cond_len])
                target += [ignore_index] * (cond_len - t_pos)
            pieces.append(task[0])          # finalize 位：唯一允许出 eos 的地方
            target.append(self.eos_token)

            seqs.append(torch.cat(pieces, dim=0))
            targets.append(target)

        return self._pad_batch(seqs, targets, cond_embeds, ignore_index)

    # ---------------- 训练前向 ----------------

    def forward(self, cond_embeds, cond_lens, speech_tokens=None, speech_lens=None,
                prompt_tokens=None, prompt_lens=None,
                text_seg_lens=None, speech_seg_lens=None):
        """二选一：数据驱动交错(S4 主路径) > 前缀式(无段信息时的退化路径)。"""
        if text_seg_lens is not None and speech_seg_lens is not None:
            inputs, attention_mask, target = self.build_seg_interleaved_inputs(
                cond_embeds, cond_lens, speech_tokens, speech_lens, prompt_tokens, prompt_lens,
                text_seg_lens=text_seg_lens, speech_seg_lens=speech_seg_lens)
        else:
            inputs, attention_mask, target = self.build_inputs(
                cond_embeds, cond_lens, speech_tokens, speech_lens, prompt_tokens, prompt_lens)
        hidden = self.backbone(inputs_embeds=inputs, attention_mask=attention_mask).last_hidden_state
        return self.llm_decoder(hidden), target

    # ---------------- 流式生成 ----------------

    def stream_generate_s4(self, state, cond_chunk=None, finalize=False,
                           max_speech_tokens=750, top_k=25, top_p=0.8,
                           win_size=10, tau_r=0.1, input_observer=None):
        """按 S4 真实 WRITE 段生成到 fill，不沿用固定 5:15 / text-ratio 上限。

        与 `build_seg_interleaved_inputs` **成对使用**。调用范式（每来一个 WRITE 段
        的文本条件就喂一次，模型自行决定说多少）：

            state = TalkerStreamState(talker, prompt_tokens=...)
            for cond_seg in write_segments:
                for tok in talker.stream_generate_s4(state, cond_seg):
                    ...                                  # 边产边送 flow 解码
            for tok in talker.stream_generate_s4(state, finalize=True):
                ...                                      # 尾部 flush

        Args:
            state: TalkerStreamState，跨段维护 KV cache / decoded / 待消费输入
            cond_chunk: (1, T, D) 本段文本条件（已过桥接融合）；finalize 时可为 None
            finalize: True 表示进入尾部 flush（喂 [task]，放开 eos）
            max_speech_tokens: 单段硬上限。触顶时 `last_stop_reason='segment_cap'`；
                持续触顶说明 fill_token 没学会（见 TalkerStreamState 的统计字段）。
        Yields:
            本段产出的 speech token
        """
        dtype = state.dtype
        pieces = []
        if cond_chunk is not None:
            pieces.append(cond_chunk.to(dtype))
            state.cond_consumed += cond_chunk.size(1)
        if finalize:
            pieces.append(self.task_embed(dtype, state.device))
        if state.pending is not None:
            pieces.insert(0, state.pending)
            state.pending = None
        if not pieces:
            return

        lm_input = torch.cat(pieces, dim=1)
        state.n_seg += 1
        state.last_stop_reason = None
        n_this_seg = 0

        with torch.inference_mode():
            for _ in range(int(max_speech_tokens)):
                if input_observer is not None:
                    input_observer(lm_input)
                if state.graph is not None:
                    hidden = (state.graph.step(lm_input) if lm_input.size(1) == 1
                              else state.graph.prefill(lm_input))
                else:
                    out = self.backbone(
                        inputs_embeds=lm_input,
                        past_key_values=state.past_key_values,
                        use_cache=True,
                    )
                    state.past_key_values = out.past_key_values
                    hidden = out.last_hidden_state
                logp = self.llm_decoder(hidden[:, -1]).log_softmax(dim=-1)
                if finalize:
                    scores = logp[0].clone()
                    eos_score = scores[self.eos_token].clone()
                    scores[self.config.speech_token_size:] = -float('inf')
                    scores[self.eos_token] = eos_score
                    token = self._ras_sampling(
                        scores, state.decoded, top_k, top_p, win_size, tau_r)
                else:
                    token = self._ras_sampling(
                        logp[0], state.decoded, top_k, top_p, win_size, tau_r,
                        ignore_eos=True, allow_tokens=(self.fill_token,))

                if token == self.fill_token:
                    state.n_fill += 1
                    state.last_stop_reason = 'fill'
                    state.seg_lens.append(n_this_seg)
                    return
                if token == self.eos_token:
                    state.finished = True
                    state.last_stop_reason = 'eos'
                    state.seg_lens.append(n_this_seg)
                    return

                state.decoded.append(token)
                n_this_seg += 1
                yield token
                lm_input = self.speech_embedding.weight[token].reshape(1, 1, -1).to(dtype)

        # 最后采出的 speech token 还没进入 KV；留给下一次和 cond/task 一起前向。
        if n_this_seg:
            state.pending = lm_input
        state.n_segcap += 1
        state.last_stop_reason = 'segment_cap'
        state.seg_lens.append(n_this_seg)

    # ---------------- 采样 ----------------

    def _ras_sampling(self, logp, decoded, top_k, top_p, win_size, tau_r, ignore_eos=False,
                      allow_tokens=None):
        """重复感知采样（对齐官方 ras_sampling）。

        `ignore_eos` 时把所有 stop token 屏蔽为 -inf，但 `allow_tokens` 里的例外
        （通常是 fill_token）必须放开 —— 否则 talker 无法主动交回控制权。
        """
        scores = logp.clone()
        if ignore_eos:
            saved = None if not allow_tokens else {
                t: scores[t].clone() for t in allow_tokens}
            scores[self.config.speech_token_size:] = -float('inf')
            if saved:
                for token_id, value in saved.items():
                    scores[token_id] = value
        token = self._nucleus_sampling(scores, top_k, top_p)
        if decoded:
            window = decoded[-win_size:]
            if sum(1 for t in window if t == token) >= win_size * tau_r:
                # 重复率过高 -> 退化为 top_k 随机采样，抑制复读
                topk_val, topk_idx = scores.topk(min(top_k, scores.size(-1)))
                token = int(topk_idx[torch.randint(topk_idx.size(0), (1,))].item())
        return token

    @staticmethod
    def _nucleus_sampling(scores, top_k, top_p):
        probs = scores.softmax(dim=-1)
        k = min(int(top_k), int(probs.size(-1)))
        sorted_probs, sorted_idx = probs.sort(descending=True)
        sorted_probs = sorted_probs[:k]
        sorted_idx = sorted_idx[:k]
        keep = torch.ones_like(sorted_probs, dtype=torch.bool)
        if k > 1:
            keep[1:] = sorted_probs.cumsum(dim=-1)[:-1] < top_p
        masked = sorted_probs * keep
        pick = torch.multinomial(masked / masked.sum(), 1)
        return int(sorted_idx[pick].item())


class SI(AudioEncoderMixin, PreTrainedModel, GenerationMixin):
    """流式同传模型本体：Voxtral encoder + thinker + Talker（规格由 config 决定）。

    `streaming_si` = Qwen3-0.6B thinker + 8L talker；
    `streaming_si_4b` = Qwen3.5-4B thinker（32L/2560d）+ 28L talker（**当前在用**）。

    参数名（`thinker_model` / `talker` / `audio_proj` / `talker_text_embed`）与
    checkpoint 中的键一一对应，改名会导致既有权重加载不到。
    """

    config_class = OmniSIConfig
    _supports_cache_class = True
    talker_cls = SITalker
    # transformers 4.57：_tied_weights_keys 为列表，仅供 save_pretrained 识别共享张量；
    # 真正的 lm_head<->embed_tokens 绑定由 tie_weights 递归到子模块 thinker_model
    # （Qwen3ForCausalLM）自行完成，外层不再重复 tie。
    _tied_weights_keys = ["thinker_model.lm_head.weight"]

    def get_input_embeddings(self):
        thinker = self.__dict__.get('_modules', {}).get('thinker_model')
        return None if thinker is None else thinker.get_input_embeddings()

    def get_output_embeddings(self):
        # 4.57 的 post_init 会立即 tie_weights；此时 thinker_model 尚未挂载，返回 None
        # 让外层 tie_embeddings_and_encoder_decoder 跳过。
        thinker = self.__dict__.get('_modules', {}).get('thinker_model')
        return None if thinker is None else thinker.get_output_embeddings()

    def __init__(self, config: OmniSIConfig = None, audio_encoder_path=None,
                 thinker_path=None, talker_path=None, audio_backend='voxtral'):
        config = config or self.config_class()
        # ---- Voxtral 后端必须在构造 audio_proj **之前** 改掉 audio_hidden_size ----
        # Voxtral adapter 输出 3072 维@12.5Hz，而 config 默认 audio_hidden_size=512。
        # `attach_to_config` 的文档明确要求先于 audio_proj 构造调用，否则投影层
        # 输入维度不匹配（512 vs 3072）—— 这是音频接入的第一处硬约束。
        audio_encoder, audio_processor = None, None
        if audio_encoder_path is not None:
            audio_encoder, audio_processor = self.load_audio_encoder(
                audio_encoder_path, backend=audio_backend)
            if audio_encoder is not None:
                audio_encoder.attach_to_config(config)   # audio_hidden_size 512 -> 3072
            else:
                warnings.warn('[SI] voxtral 后端加载失败，audio 通路不可用')
        super().__init__(config)
        self.audio_proj = MMAudioProjector(config.audio_hidden_size, config.hidden_size)
        self.post_init()  # 只初始化 projector；thinker/talker 随后挂载以保留预训练权重
        self.thinker_model = load_thinker(config, thinker_path or config.thinker_path)
        object.__setattr__(self, 'thinker', self.thinker_model.model)
        object.__setattr__(self, 'lm_head', self.thinker_model.lm_head)
        self.talker = self.talker_cls(config)
        object.__setattr__(self, 'audio_encoder', audio_encoder)
        object.__setattr__(self, 'audio_processor', audio_processor)
        object.__setattr__(self, 'speech_decoder', None)  # flow + HiFT，按需挂载

        # ================= talker 专属文本条件 embedding 副本 =================
        # 为什么必须独立（否则 S1-0 的成果会被 S2 静默作废）：
        #   build_talker_condition 的 text 路只做「token id -> 向量」**查表**，
        #   不经过 thinker 任何 transformer 层（见该方法文档）。原实现直接借用
        #   `thinker.embed_tokens`，而 Qwen3 是 **tied embedding**
        #   （`lm_head` 与 `embed_tokens` 是同一份权重），因此 S2 训练 thinker 时
        #   `embed_tokens` 必然被 `lm_head` 的梯度带着一起变 —— 实测 S2 前后逐 token
        #   余弦仅 **0.3676**（近正交）、范数 0.9251->1.0300。于是 S1-0 在原始 Qwen3
        #   embedding 上学到的 text->speech 映射（text_proj / fusion 的 text 半边 /
        #   decoder 对条件的响应）在合并 S2 thinker 后全部失配。
        #
        #   注：架构上 talker 本就有自己完整的词表体系（speech 6565: decoder.embed_tokens
        #   + llm_decoder），与 thinker 的 151936 文本词表无关；共用 thinker 的查找表
        #   只是"省 155M 参数"的实现耦合，本副本是把这条边界恢复回来。
        #
        # 初始值取构造时的 thinker.embed_tokens（即**原始 Qwen3**）：
        #   init_omni_model 的 load_state_dict 发生在 __init__ **之后**，所以即使随后
        #   加载 S2/合并权重覆盖了 thinker，本副本仍保持原始值 —— 正是所需语义。
        #   若 ckpt 中显式包含 `talker_text_embed.weight`，则由它覆盖（合并脚本会写入）。
        w = self.thinker.embed_tokens.weight
        emb = nn.Embedding(w.size(0), w.size(1))
        emb.weight = nn.Parameter(w.detach().clone(), requires_grad=False)
        self.talker_text_embed = emb
        self._text_embed_hook = None

    # ---------------- 训练期 embedding 控制 ----------------

    def set_text_embed_trainable(self, from_row=None):
        """分段解冻 talker 的文本 embedding 副本。

        * **S1-1 / S3**：`from_row=None`（默认）全冻结 —— 副本的唯一目的就是给 talker
          一个**恒定**的 text 条件分布；训练它等于又回到"移动目标"。
        * **S4**：同样 `from_row=None` 全冻结。~~早先曾建议放开尾部空位行~~，
          **实测证伪**：全放开后跑一次反向，本副本只有 **21/151936** 行拿到非零梯度
          （正是该 batch 的译文 token），
          `<|wait|>`/`<|W|>`/`<|chunk|>`/`<|src_end|>`/`<|audio_pad|>` 五行的
          `|grad|` **精确为 0**。原因见 `build_talker_condition`：text 路是
          `gather_answer_hidden(..., answer_mask)` 之后才用的，而 `answer_mask`
          只标译文 token —— 流式 token 根本进不了 talker。放开只会白占 155M 参数的
          优化器状态（AdamW 约 1.2GB）。
          注：流式 token 的行确实**重复**（基座上 39 行两两余弦 max=1.0000、74 对
          逐元素相同），但那影响的是 **thinker** 侧（tied embedding 让 `<|wait|>` 与
          `<|W|>` 的 logit 初始恒等），该问题由 `reinit_stream_token_rows` 解决，
          与本副本的解冻与否无关。

        实现用 grad hook 做行掩码，因为 `nn.Parameter` 无法按行设置 requires_grad。
        """
        w = self.talker_text_embed.weight
        if self._text_embed_hook is not None:
            self._text_embed_hook.remove()
            self._text_embed_hook = None
        if from_row is None:
            w.requires_grad = False
            return
        w.requires_grad = True

        def _mask_base_rows(grad):
            grad = grad.clone()
            grad[:from_row] = 0   # 已用词表行恒不更新，保住 S1-0 的声学映射
            return grad

        self._text_embed_hook = w.register_hook(_mask_base_rows)

    @torch.no_grad()
    def reinit_stream_token_rows(self, token_ids, seed=1234, scale='match'):
        """给流式 token 的 embedding 行一个可区分的随机初值（**S4 训练前必做一次**）。

        为什么必须做：这些 token 落在 Qwen3 词表的未使用空位上，原始权重里大量重复 ——
        实测 Qwen3-0.6B 基座上 `<|audio_pad|>`/`<|wait|>`/`<|W|>`/`<|chunk|>`/`<|dur:*|>`
        等 39 行两两余弦 mean=0.6234 / max=1.0000，其中 **74 对逐元素完全相同**。
        Qwen3 是 tied embedding，于是 lm_head 对应行也完全相同 —— `<|wait|>` 与 `<|W|>`
        的 logit 在初始时**恒等**，读写决策这一路起步就是对称的（只能靠梯度噪声破缺）。

        另外这些行的范数只有 0.3024~0.3614，而已用词表行中位数约 0.93：即便区分开了，
        决策 token 的 logit 尺度也系统性偏小，决策 CE 前期得先花力气纠正这个尺度。
        故默认 `scale='match'`：随机方向 + 范数对齐到已用行的中位数。

        同时写入 `talker_text_embed`（talker 的文本 embedding 副本），保持
        「副本 == 构造时的 thinker embedding」这一不变量 —— 否则 talker 的 text 路
        看到的仍是那份全部重复的旧值。

        Args:
            token_ids: 要重置的行号（`resolve_stream_token_ids` + audio_pad 的并集）。
            seed: 固定种子 —— DDP 下各 rank 必须得到**同一份**初值。
            scale: 'match' 对齐已用行范数中位数；也可传 float 直接指定范数。
        Returns: dict(统计信息)，供 trainer 打印核对。
        """
        ids = sorted({int(i) for i in token_ids if i is not None})
        if not ids:
            return {'n_rows': 0}
        w = self.thinker.embed_tokens.weight
        base_norms = w[:min(ids)].float().norm(dim=-1)
        target = float(base_norms.median()) if scale == 'match' else float(scale)
        idx = torch.tensor(ids, device=w.device)
        old = w[idx].float().norm(dim=-1)

        g = torch.Generator().manual_seed(int(seed))   # CPU 生成 -> 各 rank 逐位一致
        new = torch.randn(len(ids), w.size(1), generator=g)
        new = new / new.norm(dim=-1, keepdim=True) * target
        new = new.to(device=w.device, dtype=w.dtype)
        w[idx] = new
        if getattr(self, 'talker_text_embed', None) is not None:
            tw = self.talker_text_embed.weight
            tw[idx.to(tw.device)] = new.to(device=tw.device, dtype=tw.dtype)

        e = torch.nn.functional.normalize(new.float(), dim=-1)
        cos = e @ e.T
        off = cos[~torch.eye(len(ids), dtype=torch.bool, device=cos.device)]
        return {'n_rows': len(ids), 'first_row': ids[0], 'target_norm': target,
                'old_norm_mean': float(old.mean()), 'max_pair_cos': float(off.max())}

    # ---------------- Thinker ----------------

    def forward_thinker(self, input_ids=None, attention_mask=None, past_key_values=None,
                        use_cache=False, audio_inputs=None, audio_lens=None, start_pos=0,
                        audio_seg_lens=None, audio_feature_offsets=None, inputs_embeds=None):
        """thinker 前向：文本 embedding + 源音频特征注入。

        `inputs_embeds` 用于流式推理中已预先拼好 embedding 的场景（此时不能再传
        audio_inputs，音频必须已经注入完毕）。
        """
        if inputs_embeds is not None:
            if input_ids is not None:
                raise ValueError('forward_thinker 的 input_ids / inputs_embeds 只能传一个')
            hidden_states = inputs_embeds
        else:
            if input_ids is None:
                raise ValueError('forward_thinker 需要 input_ids 或 inputs_embeds')
            hidden_states = self.thinker.embed_tokens(input_ids)
        seq_length = hidden_states.size(1)
        if audio_inputs is not None and start_pos == 0:
            if input_ids is None:
                raise ValueError('预编码 inputs_embeds 不能再传 audio_inputs')
            audio_features = self.encode_audio_inputs(audio_inputs, audio_lens)
            hidden_states = self.inject_audio_features(
                input_ids, hidden_states, audio_features,
                seqlen=seq_length, audio_seg_lens=audio_seg_lens,
                audio_feature_offsets=audio_feature_offsets)
        # 只有需要中间层 hidden 时才让 thinker 返回全部层，省显存
        need_hidden = int(getattr(self.config, "talker_bridge_layer", -1)) >= 0
        return self.thinker(
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_hidden_states=need_hidden,
        )

    def pick_bridge_hidden(self, thinker_out):
        """取 talker 条件所用的 thinker hidden。<0 表示最后一层（已过 final norm）。"""
        layer = self.config.talker_bridge_layer
        if layer is None or layer < 0:
            return thinker_out.last_hidden_state
        idx = min(layer + 1, len(thinker_out.hidden_states) - 1)
        return thinker_out.hidden_states[idx]

    # ---------------- S1 跳过 thinker 主干（skip_thinker） ----------------

    def can_skip_thinker(self, skip_text_logits=False, use_cache=False, audio_inputs=None):
        """`cond_mode='text_only'` 的阶段（S1）能否完全跳过 thinker 主干前向。

        ## 为什么数学上无损

        `_fuse_talker_condition` 在 `text_only` 下对 cond 路做 `torch.zeros_like(cond)`，
        thinker hidden 的数值**整段被丢弃**；而 `GatedBridge`（thinker_proj）全部由
        `bias=False` 的 Linear + RMSNorm 组成，`thinker_proj(0) == 0`，与被 `zeros_like`
        覆盖的结果逐位相同。故把 thinker 的真实输出换成同形状零张量后，
        talker 看到的 condition **完全不变**（不是近似，是逐位相等）。

        text 路也不受影响：它走 `self.talker_text_embed(input_ids)`（talker 自己的
        embedding 副本，只查表、不过 thinker 的层），与 thinker 前向无关。

        ## 必须同时满足的前置条件（任一不满足即照常跑 thinker）

        * `config.skip_thinker_when_text_only` 显式打开（默认关 = 零回归）
        * `cond_mode == 'text_only'` —— 只有此时 cond 路才被置零。S3 / S3-w / S4 的
          `full` 模式**真的要用** thinker hidden，绝不能跳
        * `skip_text_logits` —— 即 `text_loss_weight == 0`。否则 `lm_head` 要用
          `thinker_out.last_hidden_state` 算文本 CE
        * `not use_cache` —— 推理路径要 `thinker_out.past_key_values`
        * `talker_bridge_layer < 0` —— ≥0 时 `pick_bridge_hidden` 要取
          `hidden_states[idx]` 中间层，零张量替代需要构造整个层列表，不做
        * `audio_inputs is None` —— 有音频输入说明不是纯 TTS 的 S1，音频特征必须
          经 thinker 注入才有意义（S1 的 `encode_s1` 不产出 audio_inputs）
        """
        if not bool(getattr(self.config, 'skip_thinker_when_text_only', False)):
            return False
        if getattr(self.config, 'cond_mode', 'full') != 'text_only':
            return False
        if not skip_text_logits or use_cache or audio_inputs is not None:
            return False
        layer = getattr(self.config, 'talker_bridge_layer', -1)
        return layer is None or layer < 0

    def zero_thinker_out(self, input_ids):
        """与 thinker 真实输出同形状 / dtype / device 的零 hidden（见 can_skip_thinker）。

        dtype 取 thinker embedding 的权重 dtype，与 `forward_thinker` 的输出一致，
        以免 `thinker_proj` 的 matmul 触发 dtype 提升。
        """
        w = self.thinker.embed_tokens.weight
        zeros = torch.zeros((input_ids.size(0), input_ids.size(1), self.config.hidden_size),
                            dtype=w.dtype, device=input_ids.device)
        return BaseModelOutputWithPast(last_hidden_state=zeros)

    # ---------------- fused text CE（S2 / S4 的 lm_head + CE 融合） ----------------

    def use_fused_text_ce(self, use_cache=False):
        """text loss 是否走 liger 的 `fused_linear_cross_entropy`（不物化 logits）。

        ## 动机

        text 词表 151936。`lm_head` 在**全序列**上投影（`logits_to_keep=0` 时
        `slice(-0, None) == slice(0, None)`），再被 `compute_si_losses` 升一次 fp32：
        per_device=32 / L=473 时 bf16 logits 4.3 GiB + fp32 副本 8.6 GiB ≈ 12.9 GiB，
        且 `log_softmax` 要在这个 fp32 大张量上再过一遍内存带宽。
        `fused_linear_cross_entropy` 把投影与 CE 按 chunk 融合，**整个 logits 永不物化**。

        ## 前置条件

        * `config.fused_text_ce` 显式打开（默认关 = 零回归）
        * `not use_cache` —— 推理要真的 logits 做采样
        * `not self.training` 也允许（eval 时同样只需要 loss），但推理路径会带 use_cache
        * liger-kernel 可导入且有 `liger_fused_linear_cross_entropy`

        ## 代价（必须知情）

        * **`text_acc` 指标不可得**：它需要 `logits.argmax(-1)`，而 fused 路径下没有 logits。
          开启后 `compute_si_losses` 不再产出 `text_acc`（日志里该项消失）。
        * 数值上与 `F.cross_entropy(logits.float(), ...)` 不保证逐位相同：liger 在
          Triton kernel 内以 fp32 累加，精度相当但结合律不同。属 bf16 级别差异。
        """
        if not bool(getattr(self.config, 'fused_text_ce', False)):
            return False
        if use_cache:
            return False
        return _liger_flce() is not None

    @staticmethod
    def gather_answer_hidden(hidden, answer_mask):
        """按 answer_mask 抽取每个样本的 answer 段 hidden，并 right-pad 成 (B, L_max, D)。"""
        lens = answer_mask.sum(dim=1).long()
        max_len = int(lens.max().item()) if lens.numel() else 0
        out = hidden.new_zeros(hidden.size(0), max(max_len, 1), hidden.size(-1))
        for i in range(hidden.size(0)):
            if int(lens[i]) > 0:
                out[i, :int(lens[i])] = hidden[i][answer_mask[i].bool()]
        return out, lens.clamp(min=1)

    # ---------------- Talker 条件构造 ----------------

    def _fuse_talker_condition(self, bridge_hidden, text_hidden):
        cond = self.talker.thinker_proj(bridge_hidden)
        if getattr(self.config, 'cond_mode', 'full') == 'text_only':
            cond = torch.zeros_like(cond)
        text = self.talker.text_proj(text_hidden.to(cond.dtype))
        return self.talker.fusion(cond, text)

    def build_stream_talker_condition(self, bridge_hidden, text_ids):
        """把一个 WRITE 段的逐 token hidden/ID 转成与训练相同的 Talker condition。"""
        text_hidden = self.talker_text_embed(text_ids)
        return self._fuse_talker_condition(bridge_hidden, text_hidden)

    def build_talker_condition(self, thinker_out, input_ids, answer_mask):
        """三路输入中的两路文本轴信息在此融合（第三路 previous speech token 在 talker 内部按 AR 进入）。

        Thinker hidden ──GatedBridge──┐
                                      ├──► Fusion ──► talker 前缀
        text token embedding ──Linear─┘

        两个训练期开关（从 config 读，避免改动 forward/generate 的调用链；默认关=零回归）：

        * `cond_mode='text_only'`（**S1-0**）：把 `cond` 路整段置零，只留 text 路。
          用途：S1-0 让 talker 先学「怎么说」（FSQ token 的语言模型），此时**不依赖 thinker
          hidden**，因此可与 S2 并行、且完全不受 thinker hidden 漂移影响。
          S1-1 再切回 'full' 接入真实语义。依据：依赖是单向的（talker->thinker）。
        * `hidden_noise=σ`：给 bridge 前的 thinker hidden 加高斯噪声，σ 按该 batch
          hidden 的 RMS 成比例（而非绝对值，否则不同层/不同尺度下失效）。仅训练态生效。
          用途：吸收 S3/S4/S5 联合训练中的残余小幅漂移。

        注意：**不做 cond-dropout** —— 那会把能力锚点从 cond 迁到 text，
        使架构退化为传统 TTS，且我们推理时永远有 cond、不需要无条件分支。

        Returns: (cond_embeds (B, L, talker_hidden), cond_lens (B,))
        """
        bridge_hidden = self.pick_bridge_hidden(thinker_out)
        sigma = float(getattr(self.config, 'hidden_noise', 0.0) or 0.0)
        if self.training and sigma > 0:
            rms = bridge_hidden.detach().pow(2).mean(dim=-1, keepdim=True).sqrt()
            bridge_hidden = bridge_hidden + torch.randn_like(bridge_hidden) * (sigma * rms)
        cond_mask = self.build_cond_mask(input_ids, answer_mask)
        cond_embeds, cond_lens = self.gather_answer_hidden(bridge_hidden, cond_mask)
        # text 路用 talker 自己的 embedding 副本（见 __init__ 的说明）：
        # 这里只是查表、不过 thinker 的层，故与 thinker 解耦后 cond 路仍完整保留语义。
        # 实验 b：音频位置没有对应文本，text 路在那里置零、只留译文位置——
        # 这样"说什么"的信息不丢，且长度与 thinker 路天然对齐。
        text_all = self.talker_text_embed(input_ids)
        if str(getattr(self.config, 'talker_cond_positions', 'answer')) != 'answer':
            text_all = text_all * answer_mask.to(text_all.dtype).unsqueeze(-1)
        text_embeds, _ = self.gather_answer_hidden(text_all, cond_mask)
        return self._fuse_talker_condition(cond_embeds, text_embeds), cond_lens

    def build_cond_mask(self, input_ids, answer_mask):
        """talker 条件取哪些位置（实验 b 的开关，见 `talker_cond_positions`）。

        answer        : 只取 `answer_mask==1`（默认，与历史行为逐位一致）
        audio         : 只取 `<|audio_pad|>` 占位符位置（已注入音频特征的那些帧）
        audio+answer  : 两者并集；音频帧在序列里本就排在译文之前，故拼接顺序天然是 audio → answer

        注意：S4 里音频帧是 12.5Hz，30s 音频 ≈ 375 帧，比译文 token（几十个）长一个量级，
        cond 序列变长会直接拉高 talker 的显存与耗时——这是实验 b 的已知代价（方案 §4 已记）。
        """
        mode = str(getattr(self.config, 'talker_cond_positions', 'answer') or 'answer')
        if mode == 'answer':
            return answer_mask
        pad_id = (getattr(self.config, 'audio_ids', None) or [None])[0]
        if pad_id is None:
            raise ValueError('talker_cond_positions 需要 config.audio_ids（<|audio_pad|> 的 id）')
        am = answer_mask.to(torch.long)
        audio_mask = (input_ids == int(pad_id)).to(torch.long) * (1 - am)   # 保险：与译文位置不重叠
        if mode == 'audio':
            return audio_mask
        if mode == 'audio+answer':
            return (audio_mask + am).clamp(max=1)
        raise ValueError(f'talker_cond_positions 应为 answer / audio / audio+answer，收到 {mode!r}')

    # ---------------- 训练前向 ----------------

    def forward(self, input_ids, attention_mask=None, past_key_values=None, use_cache=False,
                logits_to_keep=0, audio_inputs=None, audio_lens=None, answer_mask=None,
                speech_tokens=None, speech_token_lens=None, prompt_speech_tokens=None,
                prompt_speech_token_lens=None, skip_text_logits=False, audio_seg_lens=None,
                audio_feature_offsets=None, text_seg_lens=None, speech_seg_lens=None, **args):
        # S1（cond_mode=text_only + text_w=0）下 thinker 的输出会被整段丢弃，
        # 跳过这 4B 的前向；零张量替代后 talker 看到的 condition 逐位不变。
        # 见 can_skip_thinker 的完整条件与等价性论证。
        if self.can_skip_thinker(skip_text_logits=skip_text_logits, use_cache=use_cache,
                                 audio_inputs=audio_inputs):
            thinker_out = self.zero_thinker_out(input_ids)
        else:
            thinker_out = self.forward_thinker(
                input_ids, attention_mask=attention_mask, past_key_values=past_key_values,
                use_cache=use_cache, audio_inputs=audio_inputs, audio_lens=audio_lens,
                start_pos=0 if past_key_values is None else -1,
                audio_seg_lens=audio_seg_lens,
                audio_feature_offsets=audio_feature_offsets)
        # skip_text_logits: 供 `--text_loss_weight 0` 的阶段(S1-0/S1-1, 只训 talker)省掉
        # 整个 lm_head 投影 + 文本 CE —— 它们的结果会被乘 0，纯属浪费。
        # 注意 `logits_to_keep=0` 时 `slice(-0, None)` == `slice(0, None)` = **全序列**，
        # 故 lm_head 是在 (B, L, 1024) x (1024, 151936) 的全序列上算的：
        # BS=32/L~120 => 约 1.2e12 FLOPs + 1.2GB bf16 logits，再叠 CE 的 log_softmax。
        # 仅在 text 权重为 0 时跳过，权重>0 的阶段行为**逐字节不变**。
        # text 路三分支：
        #   1) skip_text_logits（text_w==0，S1）：lm_head 与文本 CE 全跳过
        #   2) fused text CE（text_w>0 且开关打开，S2/S4）：**不执行 lm_head 投影**，
        #      只把 hidden 与 lm_head.weight 传给 loss 侧，由 liger 的
        #      fused_linear_cross_entropy 按 chunk 融合计算，全序列 logits 永不物化。
        #      上面那段注释算过账：logits 本身 1.2GB 级别，compute_si_losses 再 `.float()`
        #      一次又翻倍 —— per_device=32/L=473 时合计约 12.9 GiB，这一支全部省掉。
        #   3) 默认：原样算 logits（行为逐字节不变）
        text_logits = text_hidden = lm_head_weight = None
        if not skip_text_logits:
            slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
            if self.use_fused_text_ce(use_cache=use_cache):
                text_hidden = thinker_out.last_hidden_state[:, slice_indices, :]
                lm_head_weight = self.lm_head.weight
            else:
                text_logits = self.lm_head(thinker_out.last_hidden_state[:, slice_indices, :])

        speech_logits, speech_target = None, None
        if answer_mask is not None:
            cond_embeds, cond_lens = self.build_talker_condition(thinker_out, input_ids, answer_mask)
            speech_logits, speech_target = self.talker(
                cond_embeds, cond_lens, speech_tokens=speech_tokens, speech_lens=speech_token_lens,
                prompt_tokens=prompt_speech_tokens, prompt_lens=prompt_speech_token_lens,
                text_seg_lens=text_seg_lens, speech_seg_lens=speech_seg_lens)

        # dummy gradient：保证 DDP 下未参与计算的分支也有梯度路径
        aux_loss = sum(p.sum() for p in self.audio_proj.parameters()) * 0
        return SIOutput(
            aux_loss=aux_loss, logits=text_logits,
            past_key_values=thinker_out.past_key_values if use_cache else None,
            speech_logits=speech_logits, speech_target=speech_target,
            text_hidden=text_hidden, lm_head_weight=lm_head_weight)

    # ======================================================================
    #                    声码器（Fun-CosyVoice3 flow + HiFT）
    # ======================================================================

    def attach_speech_decoder(self, model_dir, device='cpu', cosyvoice_repo=None, strict=True):
        """挂载官方 CosyVoice 的 flow + HiFT，用于把 speech token 解码成 24kHz 波形。

        依赖官方仓库（https://github.com/FunAudioLLM/CosyVoice，需 `--recursive` 以取到
        third_party/Matcha-TTS）。`cosyvoice_repo` 会被挂到 sys.path —— 官方包不是 pip 包，
        以前由调用方自己挂载，漏挂时只会得到一句 warning。

        Args:
            strict: 默认 **True = 挂载失败直接抛异常**。
                原实现是 `warnings.warn` + `return False`（静默降级为只输出 speech token）。
                这个降级极其危险：下游会"成功运行"并产出**空音频或无意义音频**，
                而 warning 往往被日志淹没。只有在确实允许"没有声码器也继续跑"的场景
                才传 strict=False。
        """
        try:
            flow, hift = load_cosy_flow_hift(
                model_dir, config_yaml=self.config.talker_config_yaml,
                device=device, cosyvoice_repo=cosyvoice_repo)
            object.__setattr__(self, 'speech_decoder', (flow, hift))
            return True
        except Exception as exc:
            if strict:
                raise RuntimeError(
                    f"挂载 CosyVoice flow/hift 失败: {exc}. 需要: (1) git clone --recursive "
                    f"官方仓库并用 cosyvoice_repo 指向它; (2) pip install hyperpyyaml "
                    f"'ruamel.yaml==0.17.40' conformer diffusers; (3) {model_dir} 下有 "
                    f"flow.pt / hift.pt / {self.config.talker_config_yaml}") from exc
            warnings.warn(f"[SI] 未能挂载 CosyVoice flow/hift: {exc}；将只输出 speech token")
            return False

    # ---- 官方流式解码的固定量（cosyvoice/cli/model.py::CosyVoice2Model.__init__）----
    # 这些**不是可调超参**:
    #   STREAM_TOKEN_HOP_LEN 必须等于训练时的 static_chunk_size（cosyvoice2.yaml 的
    #   chunk_size=25, 官方源码注释 "NOTE must matching training static_chunk_size"）,
    #   改了就与 chunk-aware causal 训练时的 mask 布局错位, 音质会劣化而不报错。
    #   HOP 递增(25->50->100)是为了首包低延迟同时减少重复计算（官方注释 "increase
    #   token_hop_len incrementally to avoid duplicate inference"）。
    STREAM_TOKEN_HOP_LEN = 25
    STREAM_TOKEN_MAX_HOP_LEN = 100
    STREAM_SCALE_FACTOR = 2
    STREAM_MEL_CACHE_LEN = 8        # 块间重算的 mel 帧数
    STREAM_SAMPLES_PER_MEL = 480    # 24000Hz / 50Hz mel
    # ODE 欧拉步数。官方硬编码 10（flow.py:409, 不对外暴露）。
    # 实测（scripts/bench_decode_latency.py, H200 独占, Fun-CosyVoice3）:
    #   steps=10 -> 首块 284ms, RTF 0.108, mel 相似度 1.0000（基准）
    #   steps=5  -> 首块 154ms, RTF 0.058, mel 相似度 0.9780, ASR 字符重合仍 100%
    # 即 **1.85x 加速、质量代价极小**, 是流式场景最有效的旋钮
    # （对比: autocast fp16 反而更慢 284->341ms, 与官方对 DiT fp16 的告警一致;
    #   而 CFG 无法关闭省时 —— solve_euler 第 116 行无条件把 batch 复制为 2 再 split）。
    # 默认保持官方 10 以不改变既有行为; 延迟敏感的流式推理建议传 ode_steps=5。
    DEFAULT_ODE_STEPS = 10

    @staticmethod
    def _set_ode_steps(flow, k):
        """覆盖官方硬编码的 n_timesteps。

        `CausalMaskedDiffWithDiT.inference` 以关键字 `n_timesteps=10` 传给 decoder,
        没有对外参数, 故在实例上替换 decoder.forward（实例属性优先于类方法）。
        幂等: 反复调用只会更新步数, 不会叠加包装。
        """
        if k is None:
            return
        dec = flow.decoder
        if not hasattr(dec, '_orig_forward'):
            dec._orig_forward = dec.forward

        def forward(*a, **kw):
            kw['n_timesteps'] = k
            return dec._orig_forward(*a, **kw)
        dec.forward = forward

    @staticmethod
    def _flow_infer(flow, token, token_len, prompt_token, prompt_token_len,
                    prompt_feat, prompt_feat_len, embedding, streaming, finalize):
        """按 flow.inference 的**实际签名**补参，兼容官方两代 API。

        * `CausalMaskedDiffWithXvec`（CosyVoice2/3）: (..., embedding, streaming, finalize)
        * `MaskedDiffWithXvec`（CosyVoice1）: (..., embedding, flow_cache)
        直接按某一版硬写会在换版本时 TypeError。
        """
        kw = dict(token=token, token_len=token_len,
                  prompt_token=prompt_token, prompt_token_len=prompt_token_len,
                  prompt_feat=prompt_feat, prompt_feat_len=prompt_feat_len,
                  embedding=embedding)
        params = inspect.signature(flow.inference).parameters
        if 'streaming' in params:
            kw['streaming'] = streaming
        if 'finalize' in params:
            kw['finalize'] = finalize
        if 'flow_cache' in params:
            kw['flow_cache'] = torch.zeros(1, 80, 0, 2, device=token.device)
        out = flow.inference(**kw)
        return out[0] if isinstance(out, (tuple, list)) else out

    @staticmethod
    def _hift_infer(hift, mel, cache_source=None, finalize=True):
        """按 hift.inference 的实际签名补参。

        * `HiFTGenerator`（CosyVoice2）: (speech_feat, cache_source) -> (wav, source)
        * `CausalHiFTGenerator`（CosyVoice3）: (speech_feat, finalize) -> (wav, source)
          注意两者虽都返回二元组, 但**流式状态管理方式完全不同**, 见
          `_stream_decode_speech_impl` 的 causal_hift 分支。
        """
        params = inspect.signature(hift.inference).parameters
        kw = {'speech_feat': mel}
        if 'cache_source' in params:
            kw['cache_source'] = (cache_source if cache_source is not None
                                  else torch.zeros(1, 1, 0, device=mel.device))
        elif 'finalize' in params:
            kw['finalize'] = finalize
        out = hift.inference(**kw)
        return out if isinstance(out, (tuple, list)) else (out, None)

    def _prep_decode_inputs(self, spk_emb, prompt_speech_tokens, prompt_feat, device,
                            token_mel_ratio):
        """整理 prompt / embedding，并校验 prompt_token 与 prompt_feat 成对。

        flow 内部按 `mel_len1 = prompt_feat.shape[1]` 裁掉 prompt 段的 mel:
            mel_len2 = (prompt_token_len + token_len) * token_mel_ratio - prompt_feat_len
        只有当 `prompt_feat_len == prompt_token_len * token_mel_ratio` 时,
        返回的 mel 长度才恰好等于 `token_len * token_mel_ratio` —— 流式解码按
        `token_offset * token_mel_ratio` 索引 mel, 一旦不成对就会**静默错位**。
        官方 frontend 强制 `speech_feat = speech_feat[:, :2 * token_len]` 正是为此。
        """
        prompt = (torch.as_tensor(prompt_speech_tokens, dtype=torch.long, device=device).reshape(1, -1)
                  if prompt_speech_tokens is not None and len(prompt_speech_tokens)
                  else torch.zeros(1, 0, dtype=torch.long, device=device))
        feat = (prompt_feat.to(device) if prompt_feat is not None
                else torch.zeros(1, 0, 80, device=device))
        if prompt.size(1) * token_mel_ratio != feat.size(1):
            raise ValueError(
                f"prompt_speech_tokens({prompt.size(1)}) 与 prompt_feat({feat.size(1)} 帧)必须成对, "
                f"且帧数须为 token 数的 {token_mel_ratio} 倍（flow 按 prompt_feat 长度裁 mel, "
                f"不成对会让 mel 与 token 的对应关系错位）。只有音色 prompt 的 token 而没有其 mel 时, "
                f"应两者都不传。")
        emb = (torch.as_tensor(spk_emb, dtype=torch.float32, device=device).reshape(1, -1)
               if spk_emb is not None else torch.zeros(1, 192, device=device))
        return prompt, feat, emb

    @torch.inference_mode()
    def decode_speech(self, speech_tokens, spk_emb=None, prompt_speech_tokens=None,
                      prompt_feat=None, streaming=False, ode_steps=None):
        """speech token -> 24kHz 波形（需先 attach_speech_decoder）。一次性离线解码。"""
        if self.speech_decoder is None:
            raise RuntimeError("请先调用 attach_speech_decoder(model_dir) 挂载 flow + HiFT")
        flow, hift = self.speech_decoder
        self._set_ode_steps(flow, ode_steps)
        device = next(flow.parameters()).device
        ratio = int(getattr(flow, 'token_mel_ratio', 2))
        prompt, feat, emb = self._prep_decode_inputs(spk_emb, prompt_speech_tokens, prompt_feat,
                                                     device, ratio)
        tokens = torch.as_tensor(speech_tokens, dtype=torch.long, device=device).reshape(1, -1)
        mel = self._flow_infer(
            flow, tokens, torch.tensor([tokens.size(1)], device=device),
            prompt, torch.tensor([prompt.size(1)], device=device),
            feat, torch.tensor([feat.size(1)], device=device),
            emb, streaming=streaming, finalize=True)
        wav, _ = self._hift_infer(hift, mel, cache_source=None, finalize=True)
        return wav

    @torch.inference_mode()
    def _stream_decode_speech_impl(self, token_iter, spk_emb=None,
                                   prompt_speech_tokens=None, prompt_feat=None,
                                   token_hop_len=None, streaming=True,
                                   ode_steps=None, generated_mel_parts=None,
                                   decode_session=None, out_session=None):
        """流式解码：边消费 speech token 边 yield 24kHz 波形块。

        实现严格对齐官方 `CosyVoice2Model.tts(stream=True)` + `token2wav`。
        **「每块只把该块的 token 送进 flow，再靠 mel crossfade 拼接」是错的**，
        差异在三处，每一处都只影响信号而不报错，因此 mock 契约测试无法发现：

        1. **flow 每次都要看从头到当前的全部 token**，再用 `mel[:, :, token_offset*ratio:]`
           取新增段。flow 的 chunk-aware causal mask 已保证因果性（不会偷看未来），
           所以"看完整历史"不违反流式；反过来只送当块 token 会让每块变成独立解码，
           块间没有声学上下文，音色/韵律在边界处跳变。
        2. **prompt_token / prompt_feat 每块都要传**（不是只传第一块）。因为每块都从
           完整历史重算，prompt 必须始终在场；只在首块传会让后续块丢掉音色条件。
        3. **HiFT 接缝的处理方式取决于哪一代 HiFT**，两代机制完全不同，见下。

        两代 HiFT 的流式状态管理（按 `hift.inference` 的签名自动分支）
        ----------------------------------------------------------
        * **v2 `HiFTGenerator(speech_feat, cache_source)`** —— 非因果，块边界必须人工弥合：
          "上一块 mel 尾 `STREAM_MEL_CACHE_LEN` 帧重算 + 传递激励相位 `cache_source`
          + hamming 交叉淡化 + 裁掉尾部 `source_cache_len` 留给下一块" 四件套缺一不可，
          只传 `cache_source` 仍会有可听咔哒声。
        * **v3 `CausalHiFTGenerator(speech_feat, finalize)`** —— 因果，天然无缝，
          改为"**累积全部历史 mel**，每块对完整 mel 重算，再用 `speech_offset` 只取新增波形"。
          不需要 crossfade / cache_source / 裁尾（`finalize=False` 时 generator 内部
          已自行裁掉尾部不可靠的一段）。
          代价是 **HiFT 计算量随时长 O(n²)**（v2 近似 O(n)）：长时连续输出需按句/段重置
          解码会话把 n 控制住，否则后段延迟持续增长。
          另有一个官方隐含上限：`CausalConditionalCFM.__init__` 里
          `rand_noise = torch.randn([1, 80, 50*300])` -> **单次解码最长 300 秒**。

        另外首块的 hop 要加 `prompt_token_pad`，把 `prompt_len + hop` 对齐到 hop 的整数倍：
        flow 内部把 prompt concat 在我们的 token 前面，chunk mask 按拼接后的绝对位置切分，
        不补齐会让我们的 token 边界与 chunk 边界错位。

        Args:
            token_iter: speech token 的可迭代对象（list 或生成器，支持边生成边解码）
            token_hop_len: 首块 hop，默认 STREAM_TOKEN_HOP_LEN=25（≈1s@25Hz）。
                **不建议修改**，须与训练时的 static_chunk_size 一致。实测降到 15 虽能省
                50ms 端到端延迟，但 ASR 字符重合从 100% 掉到 98%、mel 相似度 0.978->0.972。
            streaming: 传给 flow 的 chunk-causal 开关。末块按官方用 finalize=True 收尾。
            ode_steps: flow matching 欧拉步数，None = 官方默认 10。**流式场景建议 5**。
        Yields:
            每块的 24kHz 波形 tensor (1, n_sample)
        """
        if self.speech_decoder is None:
            raise RuntimeError("请先调用 attach_speech_decoder(model_dir) 挂载 flow + HiFT")
        flow, hift = self.speech_decoder
        self._set_ode_steps(flow, ode_steps)
        device = next(flow.parameters()).device
        ratio = int(getattr(flow, 'token_mel_ratio', 2))
        lookahead = int(getattr(flow, 'pre_lookahead_len', 3))
        if decode_session is None:
            prompt, feat, emb = self._prep_decode_inputs(
                spk_emb, prompt_speech_tokens, prompt_feat, device, ratio)
            all_tokens, state = [], {'offset': 0, 'cache': None}
        else:
            prompt = decode_session['prompt']
            feat = decode_session['feat']
            emb = decode_session['emb']
            all_tokens = list(decode_session['all_tokens'])
            state = decode_session['state']
            if spk_emb is not None:
                emb = torch.as_tensor(spk_emb, dtype=torch.float32, device=device).reshape(1, -1)
        # 按签名而非类名判别代际：类名可能随官方重构变化，而"有 cache_source"与
        # "有 finalize"这两个签名特征直接对应两种不可混用的状态管理策略。
        _hp = inspect.signature(hift.inference).parameters
        causal_hift = ('cache_source' not in _hp) and ('finalize' in _hp)

        hop = int(token_hop_len or self.STREAM_TOKEN_HOP_LEN)
        max_hop = max(hop, self.STREAM_TOKEN_MAX_HOP_LEN)
        mel_cache_len = self.STREAM_MEL_CACHE_LEN
        source_cache_len = mel_cache_len * self.STREAM_SAMPLES_PER_MEL
        window = torch.hamming_window(2 * source_cache_len, periodic=False, device=device)
        prompt_pad = int(-prompt.size(1) % hop)   # 等价 ceil(n/hop)*hop - n

        def token2wav(finalize):
            """解码一块并返回该块波形；v2/v3 两代 HiFT 的状态管理在此分支。"""
            offset = state['offset']
            n = len(all_tokens) if finalize else offset + this_hop + lookahead
            tokens = torch.as_tensor(all_tokens[:n], dtype=torch.long, device=device).reshape(1, -1)
            mel_full = self._flow_infer(
                flow, tokens, torch.tensor([tokens.size(1)], device=device),
                prompt, torch.tensor([prompt.size(1)], device=device),
                feat, torch.tensor([feat.size(1)], device=device),
                emb, streaming=(streaming and not finalize), finalize=finalize)
            mel = mel_full[:, :, offset * ratio:]
            if generated_mel_parts is not None:
                generated_mel_parts.append(mel)
            cache = state['cache']

            if causal_hift:
                # v3: 累积全部历史 mel -> 整体重算 -> speech_offset 切出新增。
                # 不做 crossfade/裁尾: 因果 HiFT 在 finalize=False 时内部已裁掉尾部
                # 不可靠的一段, 且同一时间位置的输出在各块间逐样本一致（官方
                # generator.py __main__ 的自测就是断言这一点）。
                if cache is not None:
                    mel = torch.concat([cache['mel'], mel], dim=2)
                wav, _ = self._hift_infer(hift, mel, cache_source=None, finalize=finalize)
                off = 0 if cache is None else cache['speech_offset']
                out = wav[..., off:]
                state['cache'] = {'mel': mel, 'speech_offset': off + out.size(-1)}
                return out

            # v2: mel 尾帧重算 + 激励相位 cache + hamming 交叉淡化 + 裁尾
            if cache is not None:
                mel = torch.concat([cache['mel'], mel], dim=2)
            wav, source = self._hift_infer(
                hift, mel, cache_source=None if cache is None else cache['source'],
                finalize=finalize)
            if cache is not None:      # 与上一块尾部做 hamming 交叉淡化
                ov = min(source_cache_len, wav.size(-1), cache['speech'].size(-1))
                if ov > 0:
                    w = window if ov == source_cache_len else torch.hamming_window(
                        2 * ov, periodic=False, device=wav.device)
                    wav = wav.clone()
                    wav[..., :ov] = wav[..., :ov] * w[:ov] + cache['speech'][..., -ov:] * w[ov:]
            if not finalize:
                state['cache'] = {
                    'mel': mel[:, :, -mel_cache_len:],
                    'source': (source[:, :, -source_cache_len:] if source is not None
                               else torch.zeros(1, 1, 0, device=device)),
                    'speech': wav[..., -source_cache_len:],
                }
                wav = wav[..., :-source_cache_len]
            return wav

        for tok in token_iter:
            all_tokens.extend(tok if isinstance(tok, (list, tuple)) else [tok])
            while True:
                this_hop = hop + prompt_pad if state['offset'] == 0 else hop
                if len(all_tokens) - state['offset'] < this_hop + lookahead:
                    break
                yield token2wav(finalize=False)
                state['offset'] += this_hop
                hop = min(max_hop, hop * self.STREAM_SCALE_FACTOR)
        if not all_tokens:
            return
        # 末块无条件收尾: 进入上面循环的条件是剩余 >= this_hop + lookahead, 故推进后
        # 必然还剩 >= lookahead(>0) 个 token, 这里的 mel 不会为空。
        this_hop = hop + prompt_pad if state['offset'] == 0 else hop
        yield token2wav(finalize=True)
        state['offset'] = len(all_tokens)
        if out_session is not None:
            out_session.append({
                'all_tokens': all_tokens,
                'state': state,
                'prompt': prompt,
                'feat': feat,
                'emb': emb,
            })

    @torch.inference_mode()
    def stream_decode_speech(self, token_iter, spk_emb=None,
                             prompt_speech_tokens=None, prompt_feat=None,
                             token_hop_len=None, streaming=True,
                             ode_steps=None):
        """边消费 speech token 边 yield PCM；解码细节见共享实现。"""
        yield from self._stream_decode_speech_impl(
            token_iter,
            spk_emb=spk_emb,
            prompt_speech_tokens=prompt_speech_tokens,
            prompt_feat=prompt_feat,
            token_hop_len=token_hop_len,
            streaming=streaming,
            ode_steps=ode_steps,
        )

    @torch.inference_mode()
    def decode_speech_segment(self, speech_tokens, spk_emb=None,
                              prompt_speech_tokens=None, prompt_feat=None,
                              token_hop_len=None, streaming=True,
                              ode_steps=None, decode_session=None,
                              on_pcm_chunk=None):
        """解码一个 WRITE segment，并返回其实际 PCM 与精确配对 Flow mel。

        ``on_pcm_chunk(chunk)``：声码器**每产出一块 PCM 立即回调**，发生在整段
        校验与返回之前。传 None 时行为与 CLI/评测路径一致。
        用途：实时同传边解码边发送——否则要等整段 `list(...)` 收完才对外可见，
        首块 PCM 被整段解码时长（可能数秒）推迟，浏览器播放队列断粮。
        回调内抛异常会中断解码并向上传播。
        """
        tokens = list(speech_tokens)
        flow, _ = self.speech_decoder
        device = next(flow.parameters()).device
        ratio = int(getattr(flow, 'token_mel_ratio', 2))
        mel_parts = []
        out_session = []
        pcm_chunks = []
        for chunk in self._stream_decode_speech_impl(
            tokens,
            spk_emb=spk_emb,
            prompt_speech_tokens=prompt_speech_tokens,
            prompt_feat=prompt_feat,
            token_hop_len=token_hop_len,
            streaming=streaming,
            ode_steps=ode_steps,
            generated_mel_parts=mel_parts,
            decode_session=decode_session,
            out_session=out_session,
        ):
            if on_pcm_chunk is not None:
                on_pcm_chunk(chunk)
            pcm_chunks.append(chunk)
        mel = (torch.cat(mel_parts, dim=2).transpose(1, 2).contiguous()
               if mel_parts else torch.zeros(1, 0, 80, device=device))
        expected_frames = len(tokens) * ratio
        if mel.size(1) != expected_frames:
            raise RuntimeError(
                f"segment Flow mel {mel.size(1)} 帧与 {len(tokens)} speech token "
                f"× {ratio} 不一致")
        n_samples = sum(int(chunk.size(-1)) for chunk in pcm_chunks)
        expected_samples = expected_frames * self.STREAM_SAMPLES_PER_MEL
        if n_samples != expected_samples:
            raise RuntimeError(
                f"segment PCM {n_samples} samples 与 {len(tokens)} speech token "
                f"期望 {expected_samples} 不一致")
        return DecodedSpeechSegment(
            pcm_chunks=pcm_chunks,
            mel=mel,
            n_tokens=len(tokens),
            n_samples=n_samples,
            session=out_session[0] if out_session else None,
        )

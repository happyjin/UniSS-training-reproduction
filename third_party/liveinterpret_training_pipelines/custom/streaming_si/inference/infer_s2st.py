"""离线整句 S2ST 推理 / 评测：源语音 -> 译文 + 目标语音（24kHz WAV）。

适用于 **S3（离线 S2ST joint）** 及其之前阶段的完整模型目录（导出目录、merge 产物、swift 的
S1-S3 checkpoint-*）。S4（chunk-as-turn 流式策略 + LoRA）的推理引擎未迁移，本脚本拒绝 S4 adapter。

    # 单条
    python custom/streaming_si/inference/infer_s2st.py --model_dir <ckpt> --audio src.wav --tgt_lang zh \
        --cosyvoice_repo <CosyVoice 仓库，需 --recursive> --out_dir out/demo
    # 批量评测（jsonl 每行 {"id", "audio", "tgt_lang", 可选 "reference"}）
    python custom/streaming_si/inference/infer_s2st.py --model_dir <ckpt> --input_jsonl eval.jsonl \
        --cosyvoice_repo <CosyVoice> --out_dir out/eval

流程与训练布局逐项对齐（见 si_encode.encode_st / SITalker.build_inputs）：
  1. Thinker：`audio_pad×n + 指令` 自回归生成译文，直到 eos（默认 greedy）；
  2. Talker 条件：对 `prompt + 译文`（不含 eos）整段重跑一次 thinker，按 answer_mask 取译文位置的
     hidden 与 token embedding 融合 —— 与训练时 shift 后的 X / answer_mask 完全相同；
  3. Talker：`[sos] + cond + [task]` 后自回归采 speech token，直到 eos（RAS 采样，top_k 25 / top_p 0.8）；
  4. 声码器：Fun-CosyVoice3 flow + HiFT 把 speech token 解码成 24kHz 波形；说话人向量取自源音频
     （CAM++，`speech_decoder/campplus.onnx`；缺失或 --no_speaker 时用零向量）。
     `--source_prompt`：再把源语音开头最多 `--prompt_sec` 秒编成 CosyVoice token + mel 作为 flow 的
     声学前缀（官方 zero-shot 克隆的用法），让情感 / 发声方式 / 录音质感跟随源语音，见 `source_prompt.py`。

声码器目录：`--cosyvoice_dir` > `<model_dir>/speech_decoder/`（export --cosyvoice_dir 打包）。
CosyVoice 官方包不是 pip 包，需要 `--cosyvoice_repo`（或环境变量 COSYVOICE_REPO）指向
`git clone --recursive` 的仓库，并安装 hyperpyyaml / conformer / diffusers / onnxruntime 等依赖。
`--no_vocoder` 只输出译文与 speech token，不需要以上任何依赖。

评测：jsonl 带 reference 且装了 sacrebleu 时报告文本 BLEU（中文目标用 zh 分词）。ASR-BLEU 需要额外的
ASR 模型，不在本脚本内。
"""
import argparse
import json
import os
import sys
import time
import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # custom/

from streaming_si.model.base_model import VoxtralAudioProcessor  # noqa: E402
from streaming_si.model.model_si import TalkerStreamState  # noqa: E402
from streaming_si.model.modeling_streaming_si import StreamingSIForConditionalGeneration  # noqa: E402
from streaming_si.data.si_encode import MEL_PER_ADAPTER_FRAME, audio_token_id, st_instruction  # noqa: E402
from streaming_si.model.speech_decoder import resolve_speech_decoder  # noqa: E402
from streaming_si.model.voxtral_compat import VoxtralFeatureExtractor  # noqa: E402
from streaming_si.inference.source_prompt import load_wav24  # noqa: E402

SR = 16000
OUT_SR = 24000
ZH_LANGS = ('zh', 'cmn', 'zho', 'yue')


# ---------------------------------------------------------------------------
# 输入
# ---------------------------------------------------------------------------
def load_wav16(path: str, max_sec: Optional[float] = None) -> np.ndarray:
    import soundfile as sf
    wav, sr = sf.read(path, dtype='float32', always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=-1)
    if sr != SR:
        import torchaudio
        wav = torchaudio.functional.resample(torch.from_numpy(np.ascontiguousarray(wav)), sr, SR).numpy()
    if max_sec is not None and wav.size > int(max_sec * SR):
        warnings.warn(f'{path}: {wav.size / SR:.1f}s 超过训练上限 {max_sec}s，截断')
        wav = wav[:int(max_sec * SR)]
    return np.ascontiguousarray(wav, dtype=np.float32)


def build_prompt(tokenizer, mel: torch.Tensor, tgt_lang: str, *, max_length: int = 512,
                 max_new_tokens: int = 256, task: str = 'translate') -> Tuple[List[int], torch.Tensor, bool]:
    """`audio_pad×n + 指令`，与 encode_st 的前缀逐 token 相同；帧数超预算时截音频（与训练同口径）。"""
    n_frames = max(1, mel.size(-1) // MEL_PER_ADAPTER_FRAME)
    instr = tokenizer.encode(st_instruction(tgt_lang, task), add_special_tokens=False)
    budget = max(1, max_length - len(instr) - max_new_tokens)
    truncated = n_frames > budget
    if truncated:
        n_frames = budget
        mel = mel[..., :n_frames * MEL_PER_ADAPTER_FRAME]
    return [audio_token_id(tokenizer)] * n_frames + instr, mel, truncated


# ---------------------------------------------------------------------------
# Thinker：译文生成
# ---------------------------------------------------------------------------
def _pick_token(logits: torch.Tensor, temperature: float, top_p: float) -> int:
    if top_p <= 0 or temperature <= 0:
        return int(logits.argmax())
    probs = (logits.float() / temperature).softmax(-1)
    sorted_p, idx = probs.sort(descending=True)
    keep = (sorted_p.cumsum(-1) - sorted_p) < top_p
    sorted_p = sorted_p * keep
    return int(idx[torch.multinomial(sorted_p / sorted_p.sum(), 1)])


def _prompt_embeds(model, ids: torch.Tensor, mel: torch.Tensor) -> torch.Tensor:
    """文本 embedding + 音频特征注入（与 forward_thinker 的训练路径相同）。"""
    h = model.thinker.embed_tokens(ids)
    feats = model.encode_audio_inputs(mel, torch.tensor([mel.size(-1)], device=mel.device))
    return model.inject_audio_features(ids, h, feats, seqlen=ids.size(1))


@torch.inference_mode()
def generate_text(model, prompt_ids: Sequence[int], mel: torch.Tensor, *, eos_id: int,
                  max_new_tokens: int = 256, temperature: float = 1.0, top_p: float = 0.0,
                  thinker_cache: bool = True, thinker_graph: bool = True) -> Tuple[List[int], str]:
    """返回 (译文 token，不含 eos；停止原因 'eos' | 'max_new_tokens')。

    `thinker_cache=True`（默认）：增量生成。Qwen3 走 HF KV cache；Qwen3.5 走 S5 的 `ThinkerRunner`
    （线性注意力状态 + KV 增量缓存，CUDA 上再叠定长缓存 + CUDA Graph，`thinker_graph` 控制）。
    `thinker_cache=False`：每步整段重算（对照用，结果相同、慢得多）。
    """
    device = model.thinker.embed_tokens.weight.device
    ids = torch.tensor([list(prompt_ids)], dtype=torch.long, device=device)
    emb = _prompt_embeds(model, ids, mel.unsqueeze(0).to(device))
    is_qwen35 = getattr(model.config, 'thinker_arch', 'qwen3') == 'qwen3_5'
    if thinker_cache and is_qwen35:
        from streaming_si.inference.infer_s5_streaming import ThinkerRunner   # 延迟导入：s5 反向依赖本模块
        embeds = [emb]
        run = ThinkerRunner(model, embeds, use_cache=True, graph=thinker_graph)
        gen: List[int] = []
        for _ in range(int(max_new_tokens)):
            logits, _ = run()
            tok = _pick_token(logits, temperature, top_p)
            if tok == eos_id:
                return gen, 'eos'
            gen.append(tok)
            embeds.append(model.thinker.embed_tokens(torch.tensor([[tok]], device=device)))
        return gen, 'max_new_tokens'
    use_cache = thinker_cache and not is_qwen35
    out = model.thinker(inputs_embeds=emb, use_cache=use_cache)
    past = out.past_key_values if use_cache else None
    gen: List[int] = []
    for _ in range(int(max_new_tokens)):
        tok = _pick_token(model.lm_head(out.last_hidden_state[:, -1])[0], temperature, top_p)
        if tok == eos_id:
            return gen, 'eos'
        gen.append(tok)
        step = model.thinker.embed_tokens(torch.tensor([[tok]], device=device))
        if use_cache:
            out = model.thinker(inputs_embeds=step, past_key_values=past, use_cache=True)
            past = out.past_key_values
        else:
            emb = torch.cat([emb, step], dim=1)
            out = model.thinker(inputs_embeds=emb)
    return gen, 'max_new_tokens'


# ---------------------------------------------------------------------------
# Talker：speech token 生成
# ---------------------------------------------------------------------------
@torch.inference_mode()
def talker_condition(model, prompt_ids: Sequence[int], text_ids: Sequence[int], mel: torch.Tensor) -> torch.Tensor:
    """(1, L, D_talker)。X = prompt + 译文、answer_mask 标译文位置 —— 即训练 shift 后的 X / am。"""
    device = model.thinker.embed_tokens.weight.device
    x = torch.tensor([list(prompt_ids) + list(text_ids)], dtype=torch.long, device=device)
    am = torch.tensor([[0] * len(prompt_ids) + [1] * len(text_ids)], dtype=torch.long, device=device)
    mel_b = mel.unsqueeze(0).to(device)
    out = model.forward_thinker(x, audio_inputs=mel_b, audio_lens=torch.tensor([mel.size(-1)], device=device))
    cond, lens = model.build_talker_condition(out, x, am)
    return cond[:, :int(lens[0])]


# talker 静态 KV 缓存的档位。单步耗时随缓存长度近似线性（A10G 实测 B=1：512 → 6.5ms，1024 → 7.9ms，
# 2048 → 10.7ms），而一句话的 [cond + speech] 序列通常只有 100～450，固定 2048 是在扫大半是空的缓存。
# S4 全量 1193 句：序列 > 512 的只有 3 句（0.3%），> 768 为 0。
TALKER_CACHE_LADDER = (512, 1024, 2048)
TALKER_CACHE_DEFAULT = 2048


def _rng_snapshot(device):
    st = {'cpu': torch.get_rng_state()}
    if torch.device(device).type == 'cuda':
        st['cuda'] = torch.cuda.get_rng_state(device)
    return st


def _rng_restore(st, device):
    torch.set_rng_state(st['cpu'])
    if 'cuda' in st:
        torch.cuda.set_rng_state(st['cuda'], device)


def talker_cache_ladder(cache_len):
    """cache_len -> 依次尝试的缓存档位（最后一档兜底）。None = 旧行为（只有 2048）。"""
    if cache_len is None:
        return (TALKER_CACHE_DEFAULT,)
    start = TALKER_CACHE_LADDER[0] if cache_len == 'auto' else int(cache_len)
    ladder = tuple(L for L in TALKER_CACHE_LADDER if L >= start)
    return ladder if ladder else (TALKER_CACHE_DEFAULT,)


def warm_talker_graphs(model, ladder=TALKER_CACHE_LADDER):
    """把各档位的 CUDA Graph 都录好，避免首个落到该档的真实句子承担录图耗时。"""
    talker = model.talker
    dev = talker.speech_embedding.weight.device
    if torch.device(dev).type != 'cuda':
        return
    dt, h = talker.speech_embedding.weight.dtype, talker.backbone.config.hidden_size
    with torch.inference_mode():
        for L in ladder:
            g = talker.decode_graph(L)
            g.reset()
            g.prefill(torch.zeros(1, 4, h, device=dev, dtype=dt))
            g.step(torch.zeros(1, 1, h, device=dev, dtype=dt))
            g.reset()


@torch.inference_mode()
def generate_speech(model, cond: torch.Tensor, *, max_speech_tokens: int = 750,
                    top_k: int = 25, top_p: float = 0.8, cuda_graph: bool = True,
                    cache_len=None) -> Tuple[List[int], str]:
    """`[sos] + cond + [task]` 后自回归到 eos（与 build_inputs 的训练布局一致），返回 (tokens, 停止原因)。

    复用 stream_generate_s4 的 finalize 分支：只放开 speech 码字与 eos（fill 等特殊 token 被屏蔽）。
    `cuda_graph=True`（默认，仅 CUDA 生效）：单 token 步用 StaticCache + CUDA Graph（与 S5 同一实现）。

    `cache_len`：talker 静态 KV 缓存长度。
      * None（默认）：旧行为，固定 2048；最坏序列（cond + max_speech_tokens）装不下时退回 eager。
      * 'auto'：从 512 起步，序列超出该档就丢弃这次生成、恢复随机数状态后升到下一档重来
        （512 → 1024 → 2048），所以结果与「直接用大档位」一致，只多花一次白算。
      * 整数：从该档起步（取 TALKER_CACHE_LADDER 中不小于它的档位）。
    最后一档沿用旧规则：最坏序列装不下时用 eager，不会溢出。
    """
    ladder = talker_cache_ladder(cache_len)
    need = int(cond.size(1)) + int(max_speech_tokens) + 4      # sos + cond + task + speech + eos 余量
    dev = model.talker.speech_embedding.weight.device
    rng = _rng_snapshot(dev) if len(ladder) > 1 else None
    for i, L in enumerate(ladder):
        last = i == len(ladder) - 1
        use_graph = bool(cuda_graph) and (not last or need <= L)
        state = TalkerStreamState(model.talker, cuda_graph=use_graph, max_len=L)
        try:
            tokens = list(model.talker.stream_generate_s4(
                state, cond_chunk=cond, finalize=True, max_speech_tokens=max_speech_tokens,
                top_k=top_k, top_p=top_p))
        except RuntimeError as exc:
            if last or 'StaticCache' not in str(exc):
                raise
            _rng_restore(rng, dev)                   # 升档重来：从同一随机数状态开始
            continue
        return tokens, state.last_stop_reason or 'none'


# ---------------------------------------------------------------------------
# 说话人向量（CAM++，与 CosyVoice 官方 frontend 同前端）
# ---------------------------------------------------------------------------
class CampplusSpeaker:
    def __init__(self, onnx_path: str):
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.log_severity_level = 3
        so.intra_op_num_threads = 4
        self.sess = ort.InferenceSession(onnx_path, sess_options=so, providers=['CPUExecutionProvider'])
        self.input_name = self.sess.get_inputs()[0].name

    def __call__(self, wav16: np.ndarray) -> Optional[np.ndarray]:
        import torchaudio.compliance.kaldi as kaldi
        if wav16.size < SR // 10:
            return None
        feat = kaldi.fbank(torch.from_numpy(wav16).unsqueeze(0), num_mel_bins=80, dither=0, sample_frequency=SR)
        feat = feat - feat.mean(dim=0, keepdim=True)
        emb = self.sess.run(None, {self.input_name: feat.unsqueeze(0).numpy()})[0]
        return np.asarray(emb, dtype=np.float32).reshape(-1)


def add_source_prompt_args(ap):
    """S4（本脚本）与 S5（infer_s5_streaming）共用的源语音 prompt 参数。"""
    ap.add_argument('--no_source_prompt', dest='source_prompt', action='store_false',
                    help='flow 只用 CAM++ 说话人向量，不把源语音作为声学前缀（旧行为）')
    ap.add_argument('--prompt_sec', type=float, default=10.0,
                    help='源语音 prompt 的最长秒数（取开头，见 --prompt_position）')
    ap.add_argument('--prompt_position', default='head', choices=['head', 'tail'],
                    help='prompt 取源语音的开头还是结尾')
    ap.add_argument('--prompt_tokenizer', default=None,
                    help='CosyVoice3 speech_tokenizer_v3.onnx（缺省找 <speech_decoder>/ 与 COSYVOICE_SPEECH_TOKENIZER）')


def build_source_prompt(enabled, decoder_dir, tokenizer_path=None):
    """返回 `SourcePrompt`（关闭时返回 None）。

    找不到 tokenizer 时**硬失败**：源语音 prompt 是音色迁移的主要通路之一，
    早先这里是打一条 WARN 后退回「只用说话人向量」，结果消融实验整轮跑在无 prompt
    状态下而没人察觉（`source_prompt_tokens` 全为 0），得出的 prompt 结论全部失效。
    真的不需要就显式传 --no_source_prompt。
    """
    if not enabled:
        return None
    from streaming_si.inference.source_prompt import SourcePrompt, find_tokenizer
    path = find_tokenizer(decoder_dir, tokenizer_path)
    if path is None:
        raise SystemExit(
            '找不到 speech_tokenizer_v3.onnx：源语音 prompt 无法启用。'
            '请用 --prompt_tokenizer 指定，或设 COSYVOICE_SPEECH_TOKENIZER，'
            '或把它放进 <speech_decoder>/；确认不要 prompt 请显式加 --no_source_prompt')
    print(f'[infer] 源语音 prompt：{path}', file=sys.stderr)
    return SourcePrompt(path)


# ---------------------------------------------------------------------------
# 管线
# ---------------------------------------------------------------------------
class S2STPipeline:
    def __init__(self, model_dir: str, *, device: str = 'cuda:0', dtype: str = 'bfloat16',
                 vocoder: bool = True, cosyvoice_dir: Optional[str] = None,
                 cosyvoice_repo: Optional[str] = None, cosyvoice_config_yaml: Optional[str] = None,
                 speaker: bool = True, max_length: int = 512, max_new_tokens: int = 256,
                 max_speech_tokens: int = 750, max_audio_sec: float = 30.0, temperature: float = 1.0,
                 top_p: float = 0.0, talker_top_k: int = 25, talker_top_p: float = 0.8,
                 ode_steps: Optional[int] = 5, text_only: bool = False, task: str = 'translate',
                 source_prompt: bool = True, prompt_sec: float = 10.0, prompt_position: str = 'head',
                 prompt_tokenizer: Optional[str] = None, thinker_cache: bool = True,
                 thinker_graph: bool = True, talker_graph: bool = True,
                 flow_cfg_rate: Optional[float] = None, talker_cache_len=None):
        from transformers import AutoTokenizer
        if os.path.isfile(os.path.join(model_dir, 's4_metadata.json')):
            raise SystemExit(f'{model_dir} 是 S4 adapter：S4 流式推理引擎未迁移，本脚本只支持 S1-S3 完整模型目录')
        self.device = device
        self.model = StreamingSIForConditionalGeneration.from_pretrained(
            model_dir, dtype=getattr(torch, dtype)).to(device).eval()
        cfg = self.model.config
        if self.model.audio_encoder is None:
            raise SystemExit(f'{model_dir} 未含 Voxtral 音频塔，无法做语音翻译')
        if getattr(cfg, 's4_decision_weight', None) is not None:
            raise SystemExit(f'{model_dir} 是 S4 训练产物（s4_decision_weight 已设置），本脚本不支持')
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        if audio_token_id(self.tokenizer) != int(cfg.audio_ids[0]):
            raise SystemExit(f'tokenizer 的 <|audio_pad|> 与 config.audio_ids={cfg.audio_ids} 不一致')
        self.eos_id = self.tokenizer.eos_token_id
        self.mel = VoxtralAudioProcessor(VoxtralFeatureExtractor.from_pretrained(model_dir))
        self.opts = dict(max_length=max_length, max_new_tokens=max_new_tokens, task=task)
        self.gen = dict(max_new_tokens=max_new_tokens, temperature=temperature, top_p=top_p,
                        thinker_cache=thinker_cache, thinker_graph=thinker_graph)
        self.talk = dict(max_speech_tokens=max_speech_tokens, top_k=talker_top_k, top_p=talker_top_p,
                         cuda_graph=talker_graph, cache_len=talker_cache_len)
        self.max_audio_sec, self.ode_steps = max_audio_sec, ode_steps
        self.text_only = text_only
        self.warnings = []
        if not text_only and float(cfg.speech_loss_weight) == 0:
            self.warnings.append('该 checkpoint 训练时 speech_loss_weight=0（S2），talker 未经训练，语音输出无意义；'
                                 '只要译文请加 --text_only')
        if not text_only and cfg.cond_mode != 'full':
            self.warnings.append(f'cond_mode={cfg.cond_mode}（S1 产物）：talker 只看文本、不看 thinker 语义')
        for w in self.warnings:
            print(f'[infer][WARN] {w}', file=sys.stderr)

        self.vocoder = vocoder and not text_only
        self.speaker = None
        self.decoder_dir = None
        if self.vocoder:
            dec_dir, yaml = resolve_speech_decoder(model_dir, cosyvoice_dir, cosyvoice_config_yaml)
            repo = cosyvoice_repo or os.environ.get('COSYVOICE_REPO')
            if not repo:
                raise SystemExit('声码器需要 --cosyvoice_repo（或 COSYVOICE_REPO）指向 CosyVoice 官方仓库；'
                                 '只要文本和 speech token 请加 --no_vocoder')
            cfg.talker_config_yaml = yaml               # Fun-CosyVoice3 为 cosyvoice3.yaml（config 默认值是 v2 的）
            self.model.attach_speech_decoder(dec_dir, device=device, cosyvoice_repo=repo, strict=True)
            self.decoder_dir = dec_dir
            if flow_cfg_rate is not None:
                # flow 的 classifier-free guidance 强度：dphi = (1+r)*cond - r*uncond
                # （`ConditionalCFM.forward_estimator`）。r 越大越贴合 prompt / 说话人条件，
                # 也越容易过冲。yaml 默认 0.7；对比：Omni 的 VoiceBox 用 cfg_scale=2.0。
                flow = self.model.speech_decoder[0]
                old = getattr(flow.decoder, 'inference_cfg_rate', None)
                if old is None:
                    raise SystemExit('当前 flow.decoder 没有 inference_cfg_rate，无法设置 --flow_cfg_rate')
                flow.decoder.inference_cfg_rate = float(flow_cfg_rate)
                print(f'[infer] flow inference_cfg_rate {old} -> {flow_cfg_rate}', file=sys.stderr)
            onnx = os.path.join(dec_dir, 'campplus.onnx')
            if speaker and os.path.isfile(onnx):
                self.speaker = CampplusSpeaker(onnx)
            elif speaker:
                print(f'[infer][WARN] {dec_dir} 没有 campplus.onnx，用零说话人向量（中性音色）', file=sys.stderr)
        self.prompt = build_source_prompt(source_prompt and self.vocoder, self.decoder_dir, prompt_tokenizer)
        self.prompt_sec, self.prompt_position = prompt_sec, prompt_position
        if talker_cache_len is not None and talker_graph:
            warm_talker_graphs(self.model, talker_cache_ladder(talker_cache_len))

    def translate(self, wav16: np.ndarray, tgt_lang: str, speaker_wav16: Optional[np.ndarray] = None,
                  wav24: Optional[np.ndarray] = None) -> Dict[str, Any]:
        t0 = time.time()
        wav16 = wav16[:int(self.max_audio_sec * SR)] if self.max_audio_sec else wav16
        mel = self.mel(torch.from_numpy(wav16), sr=SR)
        prompt, mel, truncated = build_prompt(self.tokenizer, mel, tgt_lang, **self.opts)
        text_ids, text_stop = generate_text(self.model, prompt, mel, eos_id=self.eos_id, **self.gen)
        res = dict(text=self.tokenizer.decode(text_ids, skip_special_tokens=True).strip(),
                   text_token_ids=text_ids, text_stop=text_stop, audio_truncated=truncated,
                   source_sec=wav16.size / SR, speech_tokens=[], speech_stop=None, wav=None)
        if self.text_only or not text_ids:
            res['status'] = 'ok' if text_ids else 'empty_text'
            res['wall_sec'] = time.time() - t0
            return res
        cond = talker_condition(self.model, prompt, text_ids, mel)
        tokens, speech_stop = generate_speech(self.model, cond, **self.talk)
        res.update(speech_tokens=tokens, speech_stop=speech_stop)
        if self.vocoder and tokens:
            emb = self.speaker(speaker_wav16 if speaker_wav16 is not None else wav16) if self.speaker else None
            p_tok, p_feat = (self.prompt(wav16, max_sec=self.prompt_sec, position=self.prompt_position,
                                         wav24=wav24) if self.prompt is not None else (None, None))
            wav = self.model.decode_speech(tokens, spk_emb=emb, ode_steps=self.ode_steps,
                                           prompt_speech_tokens=p_tok, prompt_feat=p_feat)
            res['wav'] = wav.detach().float().cpu().numpy().reshape(-1)
            res['source_prompt_tokens'] = 0 if p_tok is None else int(len(p_tok))
        res['status'] = 'ok' if tokens else 'empty_speech'
        res['wall_sec'] = time.time() - t0
        return res


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _bleu(hyps: List[str], refs: List[str], tgt_lang: str) -> Optional[float]:
    try:
        import sacrebleu
    except ImportError:
        return None
    tok = 'zh' if (tgt_lang or '').lower() in ZH_LANGS else '13a'
    return float(sacrebleu.corpus_bleu(hyps, [refs], tokenize=tok).score)


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model_dir', required=True, help='S1-S3 完整模型目录（导出目录 / merge 产物 / checkpoint-*）')
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument('--audio', help='单条源音频')
    src.add_argument('--input_jsonl', help='批量：每行 {"id","audio","tgt_lang",可选 "reference"}')
    ap.add_argument('--tgt_lang', default=None, help='单条模式的目标语言（zh / en / cmn / eng ...）')
    ap.add_argument('--speaker_wav', default=None, help='音色参考音频（缺省用源音频）')
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--dtype', default='bfloat16', choices=['float32', 'bfloat16', 'float16'])
    ap.add_argument('--cosyvoice_dir', default=None, help='Fun-CosyVoice3-0.5B 目录（缺省用 <model_dir>/speech_decoder）')
    ap.add_argument('--cosyvoice_repo', default=None, help='CosyVoice 官方仓库（--recursive）；缺省读 COSYVOICE_REPO')
    ap.add_argument('--cosyvoice_config_yaml', default=None)
    ap.add_argument('--no_vocoder', action='store_true', help='不合成波形，只输出译文与 speech token')
    ap.add_argument('--text_only', action='store_true', help='只生成译文（S2 checkpoint 用）')
    ap.add_argument('--no_speaker', action='store_true', help='不提取说话人向量（零向量，中性音色）')
    ap.add_argument('--max_length', type=int, default=512, help='与训练 MAX_LENGTH 一致')
    ap.add_argument('--max_new_tokens', type=int, default=256, help='译文 token 上限（训练 max_target_len=256）')
    ap.add_argument('--max_speech_tokens', type=int, default=750, help='speech token 上限（25Hz，750≈30s）')
    ap.add_argument('--talker_cache_len', default=None,
                    help="talker 静态 KV 缓存长度：缺省=旧行为（固定 2048）；'auto'=从 512 起步、溢出自动升档；"
                         "整数=从该档起步")
    ap.add_argument('--max_audio_sec', type=float, default=30.0,
                    help='源音频上限（训练 SI_MAX_AUDIO_SEC=30；用源项目口径训的模型传 20）')
    ap.add_argument('--temperature', type=float, default=1.0)
    ap.add_argument('--top_p', type=float, default=0.0, help='thinker nucleus；0 = greedy（默认）')
    ap.add_argument('--talker_top_k', type=int, default=25)
    ap.add_argument('--talker_top_p', type=float, default=0.8)
    ap.add_argument('--ode_steps', type=int, default=5, help='flow ODE 步数（源项目评测用 5；官方默认 10）')
    ap.add_argument('--flow_cfg_rate', type=float, default=None,
                    help='覆盖 flow 的 inference_cfg_rate（yaml 默认 0.7）。越大越贴合源语音的音色 / 韵律条件，'
                         '过大会过冲导致音质下降')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--no_thinker_cache', action='store_true',
                    help='thinker 每个 token 整段重算（旧行为，对照用；结果相同、慢）')
    ap.add_argument('--no_thinker_graph', action='store_true', help='thinker 增量前向不用 CUDA Graph')
    ap.add_argument('--no_talker_graph', action='store_true', help='talker 单 token 步不用 CUDA Graph（eager）')
    add_source_prompt_args(ap)
    args = ap.parse_args(argv)
    if args.audio and not args.tgt_lang:
        ap.error('单条模式需要 --tgt_lang')
    return args


def _cache_len_arg(v):
    return None if v in (None, '', 'none') else ('auto' if v == 'auto' else int(v))


def main(argv=None):
    args = parse_args(argv)
    torch.manual_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    pipe = S2STPipeline(
        args.model_dir, device=args.device, dtype=args.dtype, vocoder=not args.no_vocoder,
        cosyvoice_dir=args.cosyvoice_dir, cosyvoice_repo=args.cosyvoice_repo,
        cosyvoice_config_yaml=args.cosyvoice_config_yaml, speaker=not args.no_speaker,
        max_length=args.max_length, max_new_tokens=args.max_new_tokens,
        max_speech_tokens=args.max_speech_tokens, max_audio_sec=args.max_audio_sec,
        temperature=args.temperature, top_p=args.top_p, talker_top_k=args.talker_top_k,
        talker_top_p=args.talker_top_p, ode_steps=args.ode_steps, text_only=args.text_only,
        source_prompt=args.source_prompt, prompt_sec=args.prompt_sec,
        prompt_position=args.prompt_position, prompt_tokenizer=args.prompt_tokenizer,
        thinker_cache=not args.no_thinker_cache, thinker_graph=not args.no_thinker_graph,
        talker_graph=not args.no_talker_graph, flow_cfg_rate=args.flow_cfg_rate,
        talker_cache_len=_cache_len_arg(args.talker_cache_len))
    if args.audio:
        items = [dict(id=os.path.splitext(os.path.basename(args.audio))[0], audio=args.audio, tgt_lang=args.tgt_lang)]
    else:
        with open(args.input_jsonl, encoding='utf-8') as f:
            items = [json.loads(line) for line in f if line.strip()]
    items = items[:args.limit] if args.limit else items
    spk = load_wav16(args.speaker_wav) if args.speaker_wav else None

    import soundfile as sf
    rows, failures = [], []
    for it in items:
        # 每条单独设 seed：talker 是采样生成，只在开头设一次的话，每条的结果取决于排在它前面的条目
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        try:
            wav24 = load_wav24(it['audio'], max_sec=args.max_audio_sec) if pipe.prompt is not None else None
            res = pipe.translate(load_wav16(it['audio']), it['tgt_lang'], speaker_wav16=spk, wav24=wav24)
        except Exception as exc:          # 批量评测：单条失败不中断，记录原因
            failures.append(dict(id=it.get('id'), error=f'{type(exc).__name__}: {exc}'))
            print(f'[infer][FAIL] {it.get("id")}: {exc}', file=sys.stderr)
            if args.audio:
                raise
            continue
        wav = res.pop('wav')
        row = dict(id=it.get('id'), audio=it['audio'], tgt_lang=it['tgt_lang'], **res)
        if it.get('reference') is not None:
            row['reference'] = it['reference']
        if wav is not None:
            row['wav'] = os.path.abspath(os.path.join(args.out_dir, f'{it.get("id")}.wav'))
            sf.write(row['wav'], wav, OUT_SR)
            row['output_sec'] = wav.size / OUT_SR
        row['n_speech_tokens'] = len(row['speech_tokens'])
        rows.append(row)
        print(f'[infer] {row["id"]}: {row["text"]!r} | speech_tokens={row["n_speech_tokens"]} '
              f'stop={row["speech_stop"]} | {row["wall_sec"]:.2f}s', flush=True)

    with open(os.path.join(args.out_dir, 'results.jsonl'), 'w', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    summary = dict(model_dir=os.path.abspath(args.model_dir), decoder_dir=pipe.decoder_dir,
                   n=len(items), completed=len(rows), failed=len(failures), failures=failures,
                   warnings=pipe.warnings,
                   text_stop=_count(r['text_stop'] for r in rows),
                   speech_stop=_count(r['speech_stop'] for r in rows),
                   mean_wall_sec=float(np.mean([r['wall_sec'] for r in rows])) if rows else None)
    refs = [r for r in rows if r.get('reference') is not None]
    if refs:
        langs = {r['tgt_lang'] for r in refs}
        summary['bleu'] = {lang: _bleu([r['text'] for r in refs if r['tgt_lang'] == lang],
                                       [r['reference'] for r in refs if r['tgt_lang'] == lang], lang)
                           for lang in sorted(langs)}
        if any(v is None for v in summary['bleu'].values()):
            summary['bleu_note'] = '未安装 sacrebleu，未计算 BLEU'
    with open(os.path.join(args.out_dir, 'summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps({k: v for k, v in summary.items() if k != 'failures'}, ensure_ascii=False), flush=True)


def _count(values):
    out: Dict[str, int] = {}
    for v in values:
        out[str(v)] = out.get(str(v), 0) + 1
    return out


if __name__ == '__main__':
    main()

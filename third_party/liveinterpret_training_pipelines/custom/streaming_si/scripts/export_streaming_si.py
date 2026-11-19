"""把基座（Thinker + Voxtral）与可选的源项目 .pth 权重导出为 swift 可加载的 HF 目录。

    # streaming_si（Thinker Qwen3-0.6B / Talker 0.1B）
    python custom/streaming_si/scripts/export_streaming_si.py \
        --thinker_dir pretrained/Qwen3-0.6B \
        --voxtral_dir pretrained/Voxtral-Mini-4B-Realtime-2602 \
        [--source_ckpt s1_1a_FINAL_1024_talker8.pth [--source_ckpt ...]] \
        [--add_stream_tokens] [--cosyvoice_dir pretrained/Fun-CosyVoice3-0.5B] --out_dir models/streaming_si_s1_0

`--cosyvoice_dir` 把 Fun-CosyVoice3 的 flow.pt / hift.pt / yaml / campplus.onnx 拷进 `<out_dir>/speech_decoder/`
（只供推理，不进 state_dict；训练中 swift 保存的 S1-S3 checkpoint 会整目录带上，见 speech_decoder.py）。

    # streaming_si_4b（Thinker Qwen3.5-4B / Talker 0.6B；Talker 可选用 Qwen3-0.6B 的 transformer 层初始化）
    python custom/streaming_si/scripts/export_streaming_si.py --variant streaming_si_4b \
        --thinker_dir pretrained/Qwen3.5-4B --voxtral_dir pretrained/Voxtral-Mini-4B-Realtime-2602 \
        [--talker_init_dir pretrained/Qwen3-0.6B] --out_dir models/streaming_si_4b_base

变体（见 variants.py）决定 thinker 架构与 Talker 规格；基座 config.json 的 model_type 必须与之相符
（qwen3 / qwen3_5）。Qwen3.5 官方发布为多模态 `Qwen3_5ForConditionalGeneration`：只取
`model.language_model.*`（去掉 `language_model.` 前缀），`mtp.*` 与 `model.visual.*` 丢弃并打印计数。

与源 `SI.__init__` + `init_omni_model` 的加载语义一致：
  1. thinker 取自基座；
  2. `talker_text_embed` = **基座** thinker embedding 的副本（在灌入 .pth 之前复制，
     之后若 .pth 显式含 `talker_text_embed.weight` 则由其覆盖）；
  3. Voxtral 只取 `audio_tower.*` / `multi_modal_projector.*`；
  4. 按顺序叠加 `--source_ckpt`（后者覆盖前者），与源 checkpoint 键名完全一致。

与源实现的刻意差异（更严格）：shape 不匹配、非 `vision_proj.*` 的多余键一律报错，
而不是跳过——导出是一次性操作，静默跳过会把错误带进后续所有阶段。
"""
import argparse
import json
import os
import shutil
import sys
from collections import Counter

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # custom/

from streaming_si.model.base_model import _read_state_dict, build_qwen_tokenizer, resolve_omni_token_ids  # noqa: E402
from streaming_si.model.modeling_streaming_si import (  # noqa: E402
    StreamingSIConfig, StreamingSIForConditionalGeneration)
from streaming_si.model.qwen3_5_compat import text_config_from_official  # noqa: E402
from streaming_si.model.speech_decoder import bundle_speech_decoder, check_decoder_dir, find_config_yaml  # noqa: E402
from streaming_si.model.variants import TALKER_PRESETS, VARIANTS, thinker_arch_of_dir  # noqa: E402
from streaming_si.model.voxtral_compat import VoxtralFeatureExtractor, load_voxtral_configs  # noqa: E402

VOXTRAL_FE_CONFIG = 'voxtral_processor_config.json'
THINKER_FIELDS = ('hidden_size', 'num_hidden_layers', 'num_attention_heads', 'num_key_value_heads',
                  'head_dim', 'intermediate_size', 'vocab_size', 'max_position_embeddings',
                  'rms_norm_eps', 'rope_theta', 'tie_word_embeddings')


def _read_config(thinker_dir):
    with open(os.path.join(thinker_dir, 'config.json')) as f:
        return json.load(f)


def _thinker_overrides(raw, arch):
    """以基座 config.json 为准（与源 load_thinker「按实际权重为准」一致）。

    qwen3_5 另返回完整文本塔配置 `thinker_text_config`，扁平字段由其派生（两者由 build_thinker 交叉校验）。
    """
    if arch == 'qwen3':
        return {k: raw[k] for k in THINKER_FIELDS if k in raw}
    text = text_config_from_official(raw)
    flat = {k: text[k] for k in THINKER_FIELDS if k in text}
    return dict(flat, thinker_text_config=text)


def _safetensors_files(model_dir):
    index = os.path.join(model_dir, 'model.safetensors.index.json')
    if os.path.exists(index):
        with open(index) as f:
            return sorted(set(json.load(f)['weight_map'].values()))
    return sorted(f for f in os.listdir(model_dir) if f.endswith('.safetensors'))


def _qwen3_5_thinker_state(thinker_dir):
    """官方 Qwen3.5 权重 -> Qwen3_5CompatForCausalLM 键（`model.*` / 可选 `lm_head.weight`）。"""
    out, dropped = {}, Counter()
    for fn in _safetensors_files(thinker_dir):
        for k, v in _read_state_dict(os.path.join(thinker_dir, fn)).items():
            if k.startswith('model.language_model.'):
                out['model.' + k[len('model.language_model.'):]] = v
            elif k.startswith('model.') and not k.startswith('model.visual.'):
                out[k] = v                                   # 纯文本发布（Qwen3_5ForCausalLM）
            elif k == 'lm_head.weight':
                out[k] = v
            else:
                dropped[k.split('.')[0] if not k.startswith('model.') else 'model.visual'] += 1
    if not out:
        raise RuntimeError(f'{thinker_dir} 中没有 Qwen3.5 文本塔权重')
    print(f'[export] Qwen3.5 文本塔 {len(out)} 个张量；丢弃 {dict(dropped)}')
    return out


def _load_thinker(model, thinker_dir, arch):
    thinker = model.thinker_model
    if arch == 'qwen3':
        from transformers import Qwen3ForCausalLM
        base = Qwen3ForCausalLM.from_pretrained(thinker_dir, dtype=torch.float32)
        thinker.load_state_dict(base.state_dict(), strict=True)
        return
    sd = {k: v.float() for k, v in _qwen3_5_thinker_state(thinker_dir).items()}
    if thinker.config.tie_word_embeddings:
        head = sd.pop('lm_head.weight', None)
        if head is not None and not torch.equal(head, sd['model.embed_tokens.weight']):
            raise RuntimeError('config 声明 tie_word_embeddings，但 checkpoint 的 lm_head 与 embed_tokens 不同')
        missing, unexpected = thinker.load_state_dict(sd, strict=False)
        missing = [k for k in missing if k != 'lm_head.weight']
    else:
        missing, unexpected = thinker.load_state_dict(sd, strict=False)
    if missing or unexpected:
        raise RuntimeError(f'Qwen3.5 thinker 键不匹配：缺 {missing[:6]} 多 {unexpected[:6]}')


def _init_talker_from_qwen3(model, init_dir):
    """用 Qwen3（如 Qwen3-0.6B）的 transformer 层 + final norm 初始化 talker.decoder。

    embed_tokens / lm_head 是文本词表，与 speech 词表无关，不加载（保持随机初始化）。
    结构必须与 talker 完全一致（层数、头数、hidden、ffn），否则报错。
    """
    from transformers import Qwen3ForCausalLM
    base = Qwen3ForCausalLM.from_pretrained(init_dir, dtype=torch.float32)
    src = {k[len('model.'):]: v for k, v in base.state_dict().items()
           if k.startswith('model.') and not k.startswith('model.embed_tokens.')}
    dec = model.talker.decoder
    own = {k: v for k, v in dec.state_dict().items() if not k.startswith('embed_tokens.')}
    if set(src) != set(own):
        raise RuntimeError(f'talker 与 {init_dir} 结构不一致：多 {sorted(set(src) - set(own))[:4]} '
                           f'缺 {sorted(set(own) - set(src))[:4]}')
    bad = [k for k in src if src[k].shape != own[k].shape]
    if bad:
        raise RuntimeError(f'talker 与 {init_dir} 形状不一致: {bad[:4]}')
    dec.load_state_dict(src, strict=False)
    print(f'[export] talker.decoder 用 {init_dir} 的 {len(src)} 个张量初始化（embed_tokens 保持随机）')


def _voxtral_state(voxtral_dir):
    files = sorted(f for f in os.listdir(voxtral_dir) if f.endswith('.safetensors'))
    index = os.path.join(voxtral_dir, 'model.safetensors.index.json')
    if os.path.exists(index):
        with open(index) as f:
            wm = json.load(f)['weight_map']
        files = sorted({v for k, v in wm.items() if k.startswith(('audio_tower.', 'multi_modal_projector.'))})
    elif 'model.safetensors' in files:
        files = ['model.safetensors']       # consolidated.safetensors 是 mistral 原生格式，键名不同
    enc, ada = {}, {}
    for fn in files:
        for k, v in _read_state_dict(os.path.join(voxtral_dir, fn)).items():
            if k.startswith('audio_tower.'):
                enc[k[len('audio_tower.'):]] = v
            elif k.startswith('multi_modal_projector.'):
                ada[k[len('multi_modal_projector.'):]] = v
    if not enc or not ada:
        raise RuntimeError(f'{voxtral_dir} 缺 audio_tower.* 或 multi_modal_projector.* 权重（读了 {files}）')
    return enc, ada


def _apply_source_ckpt(model, path):
    sd = torch.load(path, map_location='cpu', weights_only=True)
    own = model.state_dict()
    unexpected = [k for k in sd if k not in own and not k.startswith('vision_proj.')]
    mismatched = [k for k in sd if k in own and sd[k].shape != own[k].shape]
    if unexpected or mismatched:
        raise RuntimeError(f'{path}: 多余键 {unexpected[:6]} / shape 不符 {mismatched[:6]}')
    loadable = {k: v for k, v in sd.items() if k in own}
    model.load_state_dict(loadable, strict=False)
    covered = Counter(k.split('.')[0] for k in loadable)
    total = Counter(k.split('.')[0] for k in own)
    print(f'[export] {os.path.basename(path)}: ' + ', '.join(
        f'{p} {covered.get(p, 0)}/{total[p]}' for p in sorted(total)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--variant', default='streaming_si', choices=sorted(VARIANTS),
                    help='; '.join(f'{k}: {v["desc"]}' for k, v in VARIANTS.items()))
    ap.add_argument('--thinker_dir', required=True)
    ap.add_argument('--voxtral_dir', required=True)
    ap.add_argument('--talker_preset', default=None, choices=sorted(TALKER_PRESETS),
                    help='缺省取变体的默认规格（streaming_si: 0.1b / streaming_si_4b: 0.6b）')
    ap.add_argument('--talker_overrides', default=None,
                    help='JSON，覆盖 talker_* 结构字段（仅用于冒烟夹具缩小尺寸）')
    ap.add_argument('--talker_init_dir', default=None,
                    help='用该 Qwen3 模型的 transformer 层初始化 talker.decoder（需与 talker 同形，如 0.6b + Qwen3-0.6B）')
    ap.add_argument('--source_ckpt', action='append', default=[])
    ap.add_argument('--add_stream_tokens', action='store_true',
                    help='注册 S4 流式特殊 token（S1-S3 不需要；不影响 <|audio_pad|> 的 id）')
    ap.add_argument('--cosyvoice_dir', default=None,
                    help='Fun-CosyVoice3-0.5B 目录；给定时把 flow/HiFT 打包进 <out_dir>/speech_decoder/（仅推理用）')
    ap.add_argument('--cosyvoice_config_yaml', default=None,
                    help='flow/HiFT 的 yaml 名，缺省自动找 cosyvoice3.yaml / cosyvoice2.yaml')
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--dtype', default='bfloat16', choices=['float32', 'bfloat16', 'float16'])
    args = ap.parse_args()

    if args.cosyvoice_dir:                  # 导出前先校验，避免权重写完才发现声码器目录不对
        check_decoder_dir(args.cosyvoice_dir)
        find_config_yaml(args.cosyvoice_dir, args.cosyvoice_config_yaml)
    variant = VARIANTS[args.variant]
    raw_cfg = _read_config(args.thinker_dir)
    arch = thinker_arch_of_dir(raw_cfg)
    if arch != variant['thinker_arch']:
        raise SystemExit(f'--variant {args.variant} 需要 {variant["thinker_arch"]} 基座，'
                         f'{args.thinker_dir} 是 {arch}')
    preset = args.talker_preset or variant['talker_preset']
    talker = dict(TALKER_PRESETS[preset])
    if args.talker_overrides:
        extra = json.loads(args.talker_overrides)
        unknown = set(extra) - set(talker)
        if unknown:
            raise SystemExit(f'--talker_overrides 含未知字段 {sorted(unknown)}')
        talker.update(extra)
    thinker_over = _thinker_overrides(raw_cfg, arch)
    tokenizer = build_qwen_tokenizer(args.thinker_dir, add_stream_tokens=args.add_stream_tokens,
                                     embedding_capacity=thinker_over.get('vocab_size', 151936))
    token_ids = resolve_omni_token_ids(tokenizer)
    enc_cfg, proj_cfg = load_voxtral_configs(args.voxtral_dir)
    config = StreamingSIConfig(
        **thinker_over,
        **talker,
        thinker_arch=arch,
        talker_preset=preset,
        **{k: token_ids[k] for k in ('audio_ids', 'pad_token_id', 'bos_token_id', 'eos_token_id',
                                     'think_end_ids', 'enter_token_id') if k in token_ids},
        voxtral_audio_config={k: v for k, v in enc_cfg.to_dict().items()
                              if k in enc_cfg.__init__.__code__.co_varnames},
        voxtral_projector_config=dict(proj_cfg.__dict__),
    )
    config.architectures = ['StreamingSIForConditionalGeneration']
    if len(config.audio_ids) != 1:
        raise RuntimeError(f'<|audio_pad|> 应为单 token，得到 {config.audio_ids}')

    model = StreamingSIForConditionalGeneration(config).eval()
    _load_thinker(model, args.thinker_dir, arch)
    if args.talker_init_dir:
        _init_talker_from_qwen3(model, args.talker_init_dir)
    with torch.no_grad():
        model.talker_text_embed.weight.copy_(model.thinker.embed_tokens.weight)
    enc_sd, ada_sd = _voxtral_state(args.voxtral_dir)
    model.audio_encoder.encoder.load_state_dict(enc_sd, strict=True)
    model.audio_encoder.adapter.load_state_dict(ada_sd, strict=True)

    for p in args.source_ckpt:
        _apply_source_ckpt(model, p)

    os.makedirs(args.out_dir, exist_ok=True)
    model.to(getattr(torch, args.dtype)).save_pretrained(args.out_dir, safe_serialization=True)
    tokenizer.save_pretrained(args.out_dir)
    fe = VoxtralFeatureExtractor.from_pretrained(args.voxtral_dir)   # 校验配置可解析
    with open(os.path.join(args.out_dir, VOXTRAL_FE_CONFIG), 'w') as f:
        json.dump({k: getattr(fe, k) for k in ('feature_size', 'sampling_rate', 'hop_length', 'n_fft',
                                               'win_length', 'global_log_mel_max')}, f, indent=2)
    with open(os.path.join(args.out_dir, 'streaming_si_export.json'), 'w') as f:
        json.dump(dict(variant=args.variant, thinker_arch=arch, talker_preset=preset,
                       talker_overrides=json.loads(args.talker_overrides) if args.talker_overrides else None,
                       talker_init_dir=os.path.abspath(args.talker_init_dir) if args.talker_init_dir else None,
                       thinker_dir=os.path.abspath(args.thinker_dir),
                       voxtral_dir=os.path.abspath(args.voxtral_dir),
                       source_ckpt=[os.path.abspath(p) for p in args.source_ckpt],
                       add_stream_tokens=args.add_stream_tokens, dtype=args.dtype,
                       cosyvoice_dir=os.path.abspath(args.cosyvoice_dir) if args.cosyvoice_dir else None),
                  f, indent=2)
    if args.cosyvoice_dir:
        dec = bundle_speech_decoder(args.cosyvoice_dir, args.out_dir, args.cosyvoice_config_yaml)
        print(f'[export] 声码器 -> {dec}（{sorted(os.listdir(dec))}）')

    reloaded = StreamingSIForConditionalGeneration.from_pretrained(args.out_dir, dtype=getattr(torch, args.dtype))
    sd_a, sd_b = model.state_dict(), reloaded.state_dict()
    diff = [k for k in sd_a if not torch.equal(sd_a[k], sd_b[k])]
    if diff:
        raise RuntimeError(f'导出后回读不一致: {diff[:6]}')
    n = lambda mod: sum(p.numel() for p in mod.parameters()) / 1e6  # noqa: E731
    print(f'[export] OK -> {args.out_dir}（{args.variant}: thinker {n(model.thinker_model):.1f}M / '
          f'talker {n(model.talker):.1f}M / 合计 {n(model):.1f}M 参数，global_log_mel_max={fe.global_log_mel_max}）')


if __name__ == '__main__':
    main()

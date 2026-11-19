"""SimulS2ST-Omni 冒烟 / 对拍夹具：随机初始化的 tiny Qwen2.5-Omni + 源项目目录布局的迷你语料。

    python custom/simuls2st_omni/make_smoke_fixture.py --meta_dir <发布仓库的 offline/ 所在目录> --out /tmp/simuls2st_smoke

`--meta_dir` 只需要发布仓库 HA-SA-ki/SimulS2ST-Omni 的配置与 tokenizer 小文件（`offline/*.json|txt|jinja`），
不需要任何权重：

    hf download HA-SA-ki/SimulS2ST-Omni --include 'offline/*.json' 'offline/*.txt' 'offline/*.jinja' --local-dir <meta_dir>

产物：

    <out>/model/                         tiny 权重 + 发布版 tokenizer / processor（特殊 token id 与发布版一致）
    <out>/data/emilia/dualcodec/         Stage 1 / Stage 2 TTS（manifest/<LANG>/*.jsonl + codes/<LANG>/*.npy）
    <out>/data/manifests/{asr,s2tt,mt}/  Stage 2 ASR / S2TT / MT（file: 音频引用）
    <out>/data/asr/ ...                  上面 manifest 引用的 wav
    <out>/data/s2st/offline/             Stage 2 离线 S2ST（manifest/<dir>/*.jsonl + codes/）
    <out>/data/s2st/streaming/           Stage 3 流式轨迹（1000ms 网格）
    <out>/data/si_s4/audio/              S2ST 源语音
    <out>/configs/stage{1,2,3}.yaml      指向上述语料的阶段配置（相对 data 根目录）

模型维度只为能在 CPU 上几秒跑完一步；词表、特殊 token、DualCodec talker 词表（16384 + 4）与发布版一致。
"""
import argparse
import json
import math
import os
import shutil

import numpy as np

TOKENIZER_FILES = ('added_tokens.json', 'chat_template.jinja', 'merges.txt', 'preprocessor_config.json',
                   'special_tokens_map.json', 'tokenizer.json', 'tokenizer_config.json', 'vocab.json',
                   'generation_config.json')
N_CODES = 16384
SR = 16000


def tiny_config(meta_offline: str):
    from transformers import Qwen2_5OmniConfig
    raw = json.load(open(os.path.join(meta_offline, 'config.json')))
    th, tk, t2w = raw['thinker_config'], raw['talker_config'], raw['token2wav_config']
    text = th['text_config']
    text.update(hidden_size=64, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                intermediate_size=128, max_window_layers=2,
                rope_scaling={'mrope_section': [2, 3, 3], 'rope_type': 'default', 'type': 'default'})
    text.pop('layer_types', None)
    th['audio_config'].update(d_model=32, encoder_layers=2, encoder_attention_heads=2, encoder_ffn_dim=64,
                              output_dim=64)
    th['vision_config'].update(depth=1, embed_dim=32, hidden_size=32, num_heads=2, intermediate_size=64,
                               out_hidden_size=64, fullatt_block_indexes=[0])
    tk.update(hidden_size=32, embedding_size=64, num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
              intermediate_size=64, head_dim=16, max_window_layers=2,
              rope_scaling={'mrope_section': [4, 4, 0], 'rope_type': 'default', 'type': 'default'})
    tk.pop('layer_types', None)
    t2w['dit_config'].update(dim=32, hidden_size=32, depth=1, num_hidden_layers=1, heads=2, num_attention_heads=2,
                             head_dim=16, emb_dim=16, enc_dim=16, enc_emb_dim=16, enc_lin_neurons=16,
                             enc_attention_channels=8, enc_se_channels=8)
    t2w['bigvgan_config'].update(upsample_initial_channel=32, upsample_rates=[5, 3, 2, 2, 2, 2],
                                 upsample_kernel_sizes=[11, 7, 4, 4, 4, 4])
    for key in ('dtype', 'transformers_version'):
        raw.pop(key, None)
    return Qwen2_5OmniConfig(**raw)


def build_model(meta_offline: str, out_dir: str, seed: int = 0):
    import torch
    from transformers import Qwen2_5OmniForConditionalGeneration
    torch.manual_seed(seed)
    config = tiny_config(meta_offline)
    model = Qwen2_5OmniForConditionalGeneration(config)
    assert model.talker.codec_head.out_features == N_CODES + 4, model.talker.codec_head
    model.save_pretrained(out_dir, safe_serialization=True)
    for name in TOKENIZER_FILES:
        src = os.path.join(meta_offline, name)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(out_dir, name))
    # Qwen2_5OmniForConditionalGeneration.from_pretrained 会强制读取 spk_dict.pt（token2wav 的音色表）；
    # 训练不用它，占位即可（结构与官方一致：name -> {cond, ref_mel, bos_token}）。
    torch.save({'Chelsie': {'cond': torch.zeros(1, 192), 'ref_mel': torch.zeros(1, 10, 80), 'bos_token': 151870}},
               os.path.join(out_dir, 'spk_dict.pt'))
    n = sum(p.numel() for p in model.parameters())
    print(f'[fixture] tiny model: {n / 1e6:.1f}M params -> {out_dir}')


def _sine(sec: float, freq: float) -> np.ndarray:
    t = np.arange(int(sec * SR)) / SR
    return (0.1 * np.sin(2 * math.pi * freq * t)).astype(np.float32)


def _write_wav(path: str, wav: np.ndarray):
    import soundfile as sf
    os.makedirs(os.path.dirname(path), exist_ok=True)
    sf.write(path, wav, SR, subtype='PCM_16')


def _write_jsonl(path: str, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')


EN_TEXTS = ['hello world', 'the quick brown fox jumps', 'speech translation is fun', 'good morning everyone',
            'this is a short test', 'machine learning on audio', 'we train three stages', 'the talker emits codes',
            'simultaneous interpretation', 'one more sentence here', 'tokens and budgets', 'final english line']
ZH_TEXTS = ['你好世界', '今天天气很好', '语音翻译很有趣', '大家早上好', '这是一个短测试', '音频上的机器学习',
            '我们训练三个阶段', '说话模块输出编码', '同声传译', '这里还有一句', '令牌与预算', '最后一句中文']


def build_tts(root: str, rng):
    """emilia/dualcodec：每语种两个 shard，长度跨度足以让 token-budget 打出多种 batch 宽度。"""
    for lang, texts in (('EN', EN_TEXTS), ('ZH', ZH_TEXTS)):
        for shard in range(2):
            codes, rows, cur = [], [], 0
            for i, text in enumerate(texts):
                n = int(rng.integers(8, 60))
                codes.append(rng.integers(0, N_CODES, size=n, dtype=np.uint16))
                rows.append({'id': f'{lang}_{shard}_{i}', 'text': text, 'language': lang.lower(),
                             'codes': f'{lang}/shard-{shard:03d}.npy:{cur}:{cur + n}',
                             'dnsmos': float(rng.uniform(2.9, 3.6)), 'duration': n * 0.08})
                cur += n
            os.makedirs(os.path.join(root, 'codes', lang), exist_ok=True)
            np.save(os.path.join(root, 'codes', lang, f'shard-{shard:03d}.npy'), np.concatenate(codes))
            _write_jsonl(os.path.join(root, 'manifest', lang, f'shard-{shard:03d}.jsonl'), rows)


def build_manifests(data_root: str, rng):
    """ASR / S2TT / MT：file: 引用的 wav（各 corpus 一个根目录）。"""
    def audio_rows(corpus, task, n, src_lang, tgt_lang, with_source=True):
        rows = []
        for i in range(n):
            sec = float(rng.uniform(0.6, 2.4))
            rel = f'{corpus}/{i:03d}.wav'
            _write_wav(os.path.join(data_root, 'audio_corpora', rel), _sine(sec, 200 + 37 * i))
            src = (EN_TEXTS if src_lang == 'en' else ZH_TEXTS)[i % 12]
            tgt = (ZH_TEXTS if src_lang == 'en' else EN_TEXTS)[i % 12] if task == 's2tt_translate' else src
            rows.append({'id': f'{corpus}_{i}', 'task_type': task, 'audio': f'file:{rel}', 'corpus': corpus,
                         'src_lang': src_lang, 'tgt_lang': tgt_lang if task == 's2tt_translate' else src_lang,
                         'source_text': src if with_source else '', 'target_text': tgt, 'duration': round(sec, 3)})
        return rows

    _write_jsonl(os.path.join(data_root, 'manifests/asr_en/shard-000.jsonl'),
                 audio_rows('asr_en', 's2tt_asr', 10, 'en', 'en'))
    _write_jsonl(os.path.join(data_root, 'manifests/s2tt_enzh/shard-000.jsonl'),
                 audio_rows('s2tt_enzh', 's2tt_translate', 12, 'en', 'zh'))
    _write_jsonl(os.path.join(data_root, 'manifests/s2tt_zhen/shard-000.jsonl'),
                 audio_rows('s2tt_zhen', 's2tt_translate', 10, 'zh', 'en'))
    mt = [{'id': f'mt_{i}', 'task_type': 't2t_translate', 'audio': None, 'src_lang': 'en', 'tgt_lang': 'zh',
           'source_text': EN_TEXTS[i % 12], 'target_text': ZH_TEXTS[i % 12]} for i in range(14)]
    _write_jsonl(os.path.join(data_root, 'manifests/mt_enzh/shard-000.jsonl'), mt)


def build_s2st(data_root: str, rng, name: str, n_per_dir: int, streaming: bool):
    """离线 S2ST / 流式轨迹共用一份 manifest 结构：codes 切片 + 1000ms 网格的 WAIT/WRITE 轨迹。"""
    root = os.path.join(data_root, 's2st', name)
    for direction in ('en2zh', 'zh2en'):
        codes, rows, cur = [], [], 0
        for i in range(n_per_dir):
            n_windows = int(rng.integers(2, 6))
            src_sec = n_windows - float(rng.uniform(0.0, 0.6))
            rel = f'{direction}/{name}_{i:03d}.wav'
            _write_wav(os.path.join(data_root, 'si_s4', 'audio', rel), _sine(src_sec, 150 + 23 * i))
            tgt_texts = ZH_TEXTS if direction == 'en2zh' else EN_TEXTS
            words = tgt_texts[i % 12]
            chunks, c0 = [], 0
            for w in range(n_windows):
                if w == 0 or rng.random() < 0.35:   # wait 窗口：不写出
                    continue
                n = int(rng.integers(3, 14))
                chunks.append({'window_idx': w, 'text': f'{words}{w}', 'code_start': c0, 'code_end': c0 + n})
                c0 += n
            if not chunks:
                chunks.append({'window_idx': n_windows - 1, 'text': words, 'code_start': 0, 'code_end': 5})
                c0 = 5
            codes.append(rng.integers(0, N_CODES, size=c0, dtype=np.uint16))
            rows.append({'id': f'{name}_{direction}_{i}', 'direction': direction, 'source_audio': rel,
                         'target_text': ' '.join(c['text'] for c in chunks),
                         'codes': f'{direction}/shard-000.npy:{cur}:{cur + c0}',
                         'trajectory': {'num_windows': n_windows, 'base_window_ms': 1000, 'chunks': chunks}})
            cur += c0
        os.makedirs(os.path.join(root, 'codes', direction), exist_ok=True)
        np.save(os.path.join(root, 'codes', direction, 'shard-000.npy'), np.concatenate(codes))
        _write_jsonl(os.path.join(root, 'manifest', direction, 'shard-000.jsonl'), rows)


STAGE1 = """stage: stage1
data:
  kind: tts
  root: emilia/dualcodec
  langs: [EN, ZH]
  min_dnsmos: 3.0
  min_codes: 8
  max_codes: 375
  eval_samples: 8
sampler:
  token_budget: 192
  max_batch_size: 6
"""

STAGE2 = """stage: stage2
data:
  kind: multitask
  weights:
    s2tt_asr: 0.2
    s2tt_translate: 1.0
    t2t_translate: 0.5
    tts_unpaired: 1.0
    s2s_translate: 1.5
  sample_seed: 42
  sources:
    tts_unpaired:
      root: emilia/dualcodec
      langs: [EN, ZH]
      min_dnsmos: 3.0
      min_codes: 8
      max_codes: 375
      max_hours: 0.02
    s2tt_asr:
      max_hours: 0.01
      manifests: [manifests/asr_en]
      roots: {asr_en: audio_corpora}
    s2tt_translate:
      max_hours: 0.05
      manifests: [manifests/s2tt_enzh, manifests/s2tt_zhen]
      roots: {s2tt_enzh: audio_corpora, s2tt_zhen: audio_corpora}
      asr_prob: 0.1
    t2t_translate:
      max_samples: 12
      manifests: [manifests/mt_enzh]
      roots: {}
    s2s_translate:
      root: s2st/offline
      si_root: si_s4/audio
      max_codes: 375
      max_hours: 0.05
  eval_samples: 8
sampler:
  token_budget: 384
  max_batch_size: 4
"""

STAGE3 = """stage: stage3
data:
  kind: streaming
  root: s2st/streaming
  si_root: si_s4/audio
  directions: [en2zh, zh2en]
  latency_multipliers: [1, 2, 3]
  max_codes: 375
  eval_samples: 4
  seed: 42
sampler:
  token_budget: 512
  max_batch_size: 4
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--meta_dir', required=True, help='含 offline/ 子目录（或直接就是 offline/）的发布仓库元数据目录')
    ap.add_argument('--out', required=True)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--skip_model', action='store_true')
    args = ap.parse_args()
    meta = args.meta_dir
    if os.path.isdir(os.path.join(meta, 'offline')):
        meta = os.path.join(meta, 'offline')
    out = os.path.abspath(args.out)
    if not args.skip_model:
        build_model(meta, os.path.join(out, 'model'), seed=args.seed)
    rng = np.random.default_rng(args.seed)
    data = os.path.join(out, 'data')
    if os.path.exists(data):
        shutil.rmtree(data)
    build_tts(os.path.join(data, 'emilia', 'dualcodec'), rng)
    build_manifests(data, rng)
    build_s2st(data, rng, 'offline', 10, streaming=False)
    build_s2st(data, rng, 'streaming', 14, streaming=True)
    cfg_dir = os.path.join(out, 'configs')
    os.makedirs(cfg_dir, exist_ok=True)
    for name, body in (('stage1', STAGE1), ('stage2', STAGE2), ('stage3', STAGE3)):
        with open(os.path.join(cfg_dir, f'{name}.yaml'), 'w') as f:
            f.write(body)
    print(f'[fixture] data -> {data}; configs -> {cfg_dir}')


if __name__ == '__main__':
    main()

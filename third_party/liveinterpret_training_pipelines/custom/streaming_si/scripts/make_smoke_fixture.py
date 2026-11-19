"""生成 swift sft 冒烟用的 tiny 夹具（随机权重 + 真实 Qwen3 tokenizer + 迷你 S1/P2 数据）。

    python custom/streaming_si/scripts/make_smoke_fixture.py --tokenizer_dir <Qwen3-0.6B> --out /tmp/si_smoke

产出：
    <out>/model/                       导出的 streaming_si 目录（swift --model）
    <out>/s1/fsq-00000.parquet         S1 数据（SI_S1_DATA）
    <out>/p2/p2_fsq_v3/fsq-00000.parquet + <out>/p2/p2_source_wav/train-00000.tar（SI_P2_DATA）
    <out>/s4/{si_s4,val,index}         S4 records（si_s4 布局）+ val 划分 + 分层索引（SI_S4_*）

只补 S4 数据（复用已有 model/）：python ... make_smoke_fixture.py --out /tmp/si_smoke --only s4
"""
import argparse
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))              # custom/streaming_si/scripts
CUSTOM_DIR = os.path.dirname(os.path.dirname(HERE))           # custom/
sys.path.insert(0, CUSTOM_DIR)

from streaming_si.model.voxtral_compat import (VoxtralEncoder, VoxtralEncoderConfig,  # noqa: E402
                                               VoxtralMultiModalProjector, VoxtralProjectorConfig)

TINY_AUDIO = dict(hidden_size=32, intermediate_size=64, num_hidden_layers=1, num_attention_heads=2,
                  num_key_value_heads=2, head_dim=16, num_mel_bins=128, sliding_window=750,
                  max_position_embeddings=1500, rms_norm_eps=1e-5,
                  rope_parameters={'rope_theta': 1e6, 'rope_type': 'default'},
                  model_type='voxtral_realtime_encoder')
PAIRS = [('Hello everyone, welcome.', '大家好，欢迎。', 'eng', 'cmn'),
         ('今天我们讨论流式同传。', 'Today we discuss simultaneous interpretation.', 'cmn', 'eng'),
         ('The weather is nice.', '天气很好。', 'eng', 'cmn'),
         ('我们开始吧。', "Let's get started.", 'cmn', 'eng')]


def _copy_tokenizer(tokenizer_dir, dst):
    for f in os.listdir(tokenizer_dir):
        if f.startswith(('tokenizer', 'vocab', 'merges', 'special_tokens')):
            shutil.copy(os.path.join(tokenizer_dir, f), dst)


def make_bases(tokenizer_dir, out):
    from transformers import Qwen3Config, Qwen3ForCausalLM
    thinker = os.path.join(out, 'base_qwen3')
    torch.manual_seed(1)
    Qwen3ForCausalLM(Qwen3Config(hidden_size=64, num_hidden_layers=2, num_attention_heads=4,
                                 num_key_value_heads=2, head_dim=16, intermediate_size=128,
                                 vocab_size=151936, tie_word_embeddings=True)).save_pretrained(thinker)
    _copy_tokenizer(tokenizer_dir, thinker)
    return thinker, make_voxtral(out)


# streaming_si_4b 夹具：tiny Qwen3.5 thinker（3 线性注意力 + 1 全注意力层，真实 Qwen3.5 词表 248320）
TINY_QWEN3_5 = dict(vocab_size=248320, hidden_size=64, intermediate_size=128, num_hidden_layers=4,
                    num_attention_heads=4, num_key_value_heads=2, head_dim=32, rms_norm_eps=1e-6,
                    linear_conv_kernel_dim=4, linear_key_head_dim=16, linear_value_head_dim=16,
                    linear_num_key_heads=2, linear_num_value_heads=4, tie_word_embeddings=True,
                    layer_types=['linear_attention'] * 3 + ['full_attention'])
# tiny talker（保留 0.6b 预设的「与 Qwen3 同形」性质，以便验证 --talker_init_dir）
TINY_TALKER = dict(talker_hidden_size=128, talker_num_hidden_layers=2, talker_num_attention_heads=4,
                   talker_num_key_value_heads=2, talker_head_dim=32, talker_intermediate_size=256)


def make_bases_4b(qwen35_tokenizer_dir, out):
    """按官方 Qwen3.5 多模态发布的布局写 tiny 基座：`model.language_model.*` + 需丢弃的 `mtp.*` / `model.visual.*`。"""
    from safetensors.torch import save_file
    from transformers import Qwen3Config, Qwen3ForCausalLM
    from streaming_si.model.qwen3_5_compat import Qwen3_5CompatForCausalLM, Qwen3_5TextCompatConfig
    thinker = os.path.join(out, 'base_qwen3_5')
    os.makedirs(thinker, exist_ok=True)
    torch.manual_seed(3)
    m = Qwen3_5CompatForCausalLM(Qwen3_5TextCompatConfig(**TINY_QWEN3_5))
    with torch.no_grad():
        for p in m.parameters():
            p.add_(torch.randn_like(p) * 0.02)
    sd = {'model.language_model.' + k[len('model.'):]: v.contiguous() for k, v in m.state_dict().items()
          if k.startswith('model.')}
    sd['mtp.fc.weight'] = torch.zeros(4, 4)
    sd['model.visual.pos_embed.weight'] = torch.zeros(4, 4)
    save_file(sd, os.path.join(thinker, 'model.safetensors'))
    text = dict(TINY_QWEN3_5, model_type='qwen3_5_text', attn_output_gate=True, mlp_only_layers=[],
                rope_parameters={'rope_type': 'default', 'rope_theta': 10000000, 'partial_rotary_factor': 0.25,
                                 'mrope_interleaved': True, 'mrope_section': [1, 1, 2]})
    json.dump({'architectures': ['Qwen3_5ForConditionalGeneration'], 'model_type': 'qwen3_5',
               'text_config': text, 'tie_word_embeddings': True, 'vision_config': {'model_type': 'qwen3_5'}},
              open(os.path.join(thinker, 'config.json'), 'w'), indent=1)
    _copy_tokenizer(qwen35_tokenizer_dir, thinker)
    talker_base = os.path.join(out, 'base_qwen3_talker')
    torch.manual_seed(4)
    Qwen3ForCausalLM(Qwen3Config(
        hidden_size=TINY_TALKER['talker_hidden_size'], num_hidden_layers=TINY_TALKER['talker_num_hidden_layers'],
        num_attention_heads=TINY_TALKER['talker_num_attention_heads'],
        num_key_value_heads=TINY_TALKER['talker_num_key_value_heads'], head_dim=TINY_TALKER['talker_head_dim'],
        intermediate_size=TINY_TALKER['talker_intermediate_size'], vocab_size=1000,
        tie_word_embeddings=True)).save_pretrained(talker_base)
    return thinker, talker_base


def make_voxtral(out):
    from safetensors.torch import save_file
    vox = os.path.join(out, 'base_voxtral')
    os.makedirs(vox, exist_ok=True)
    enc = VoxtralEncoder(VoxtralEncoderConfig(**TINY_AUDIO))
    proj = VoxtralMultiModalProjector(VoxtralProjectorConfig(32, 48, 4))
    sd = {f'audio_tower.{k}': v.contiguous() for k, v in enc.state_dict().items()}
    sd.update({f'multi_modal_projector.{k}': v.contiguous() for k, v in proj.state_dict().items()})
    save_file(sd, os.path.join(vox, 'model.safetensors'))
    json.dump({'audio_config': TINY_AUDIO, 'text_config': {'hidden_size': 48}, 'downsample_factor': 4,
               'projector_hidden_act': 'gelu', 'model_type': 'voxtral_realtime'},
              open(os.path.join(vox, 'config.json'), 'w'))
    json.dump({'feature_extractor': {'feature_size': 128, 'global_log_mel_max': 1.5, 'hop_length': 160,
                                     'n_fft': 400, 'sampling_rate': 16000, 'win_length': 400}},
              open(os.path.join(vox, 'processor_config.json'), 'w'))
    return vox


def make_p2(out, n_rows=16):
    import pyarrow as pa
    import pyarrow.parquet as pq
    import soundfile as sf
    fsq_dir, wav_dir = os.path.join(out, 'p2', 'p2_fsq_v3'), os.path.join(out, 'p2', 'p2_source_wav')
    os.makedirs(fsq_dir, exist_ok=True)
    os.makedirs(wav_dir, exist_ok=True)
    tar_path = os.path.join(wav_dir, 'train-00000.tar')
    rng = np.random.RandomState(0)
    with tarfile.open(tar_path, 'w') as tf:
        for i in range(n_rows):
            buf = io.BytesIO()
            sf.write(buf, (rng.randn(int((0.8 + 0.1 * i) * 16000)) * 0.1).astype('float32'), 16000, format='FLAC')
            info = tarfile.TarInfo(f'{i}.flac')
            info.size = len(buf.getvalue())
            tf.addfile(info, io.BytesIO(buf.getvalue()))
    with tarfile.open(tar_path) as tf:
        refs = [f'train-00000.tar::{m.offset_data}:{m.size}' for m in tf.getmembers()]
    rows = [PAIRS[i % len(PAIRS)] for i in range(n_rows)]
    pq.write_table(pa.table({
        'id': [str(i) for i in range(n_rows)],
        'source_transcript': [r[0] for r in rows], 'translation': [r[1] for r in rows],
        'src_lang': [r[2] for r in rows], 'tgt_lang': [r[3] for r in rows],
        'duration_ratio': [1.0] * n_rows, 'source_wav': refs,
        'target_wav': [''] * n_rows,
        'target_fsq': [rng.randint(0, 6561, 10 + i).tolist() for i in range(n_rows)],
    }), os.path.join(fsq_dir, 'fsq-00000.parquet'))
    return os.path.join(fsq_dir, 'fsq-00000.parquet')


def make_s1(out, n_rows=16):
    import pyarrow as pa
    import pyarrow.parquet as pq
    d = os.path.join(out, 's1')
    os.makedirs(d, exist_ok=True)
    rng = np.random.RandomState(1)
    texts = [p[1] for p in PAIRS]
    convs = [[{'role': 'user', 'content': f'请朗读：{texts[i % 4]}'},
              {'role': 'assistant', 'content': texts[i % 4]}] for i in range(n_rows)]
    pq.write_table(pa.table({
        'conversations': [json.dumps(c, ensure_ascii=False) for c in convs],
        'answer_speech_tokens': [[rng.randint(0, 6561, 12 + i).tolist()] for i in range(n_rows)],
        'ref_speech_tokens': [[rng.randint(0, 6561, 5).tolist()] for _ in range(n_rows)],
        'duration': [1.0 + 0.1 * i for i in range(n_rows)],
    }), os.path.join(d, 'fsq-00000.parquet'))
    return os.path.join(d, 'fsq-00000.parquet')


S4_TEXT = {  # (源转写, 目标词序列)；目标语 zh 无空格、en 有空格（源 _step_texts 的还原规则）
    'en2zh': [('hello everyone welcome to the show', ['大家', '好', '欢迎', '收看', '节目']),
              ('the weather is nice today', ['今天', '天气', '很', '好']),
              ('we will start the meeting now', ['我们', '现在', '开始', '开会']),
              ('please take your seats', ['请', '大家', '就座'])],
    'zh2en': [('大家好欢迎收看节目', ['Hello', 'everyone,', 'welcome', 'to', 'the', 'show.']),
              ('今天天气很好', ['The', 'weather', 'is', 'nice', 'today.']),
              ('我们现在开始开会', ['We', 'will', 'start', 'the', 'meeting', 'now.']),
              ('请大家就座', ['Please', 'take', 'your', 'seats.'])],
}
S4_TGT_LANG = {'en2zh': 'zh', 'zh2en': 'en'}


def _s4_record(direction, rid, n_steps, text_id, rng, audio_root):
    """一条 si-rw-m1-v0 记录 + 源音频 + 10_word_ready.json（字段集合与生产 records 一致）。"""
    import soundfile as sf
    src_text, words = S4_TEXT[direction][text_id % len(S4_TEXT[direction])]
    dur = n_steps * 640 - int(rng.randint(0, 300))
    lead = int(rng.randint(0, 3))                          # 前导 WAIT 步数
    tail = int(rng.randint(0, 2))                          # 译文结束后的尾部 WAIT
    write_turns = sorted(rng.choice(np.arange(lead, n_steps - tail), size=min(len(words), n_steps - lead - tail),
                                    replace=False).tolist())
    groups = np.array_split(np.arange(len(words)), len(write_turns))
    steps, cursor = [], 0
    for t in range(n_steps):
        is_write = t in write_turns
        k = write_turns.index(t) if is_write else -1
        n_codec = int(rng.randint(2, 7)) if is_write else 0
        steps.append({
            'source_start_ms': t * 640, 'source_end_ms': min((t + 1) * 640, dur),
            'action': 'write' if is_write else 'wait',
            'text_delta': ' '.join(words[j] for j in groups[k]) if is_write else '',
            'source_codec_spans': [[cursor, cursor + n_codec]] if is_write else [],
            'commit_ms': min((t + 1) * 640, dur),
            'source_stream_done': t == n_steps - 1,
            'translation_eos': is_write and t == write_turns[-1],
        })
        cursor += n_codec
    sep = '' if S4_TGT_LANG[direction] == 'zh' else ' '
    rel_dir = f'{direction}/shard000/run/{rid}'
    os.makedirs(os.path.join(audio_root, rel_dir), exist_ok=True)
    sf.write(os.path.join(audio_root, rel_dir, 'src.wav'),
             (rng.randn(dur * 16) * 0.1).astype('float32'), 16000)
    # 约 1/3 样本词序乱序（高 NIR -> 分层采样上调权重）；规则与文本/val 分组无关，避免可 crop 的
    # 高权重样本恰好全落在 val 剔除组里（替补全部失败 -> admission_exhausted）
    perm = rng.permutation(len(words)).tolist() if int(rid[-4:]) % 3 == 1 else list(range(len(words)))
    json.dump({'words': [{'word': w, 'dep_src_word_idxs': [perm[j]], 'raw_ready_ms': 300 + 200 * j}
                         for j, w in enumerate(words)]},
              open(os.path.join(audio_root, rel_dir, '10_word_ready.json'), 'w'), ensure_ascii=False)
    return {
        'schema_version': 'si-rw-m1-v0', 'record_id': rid, 'source_audio': f'{rel_dir}/src.wav',
        'source_duration_ms': dur, 'source_transcript': src_text, 'target_lang': S4_TGT_LANG[direction],
        'target_text': sep.join(words), 'target_codec_codes': rng.randint(0, 6561, cursor).tolist(),
        'trajectory': {'base_chunk_ms': 640, 'codec': {'spec': 'cosy3_fsq_25hz', 'frame_ms': 40.0},
                       'raw_target_num_frames': cursor, 'steps': steps},
    }


def make_s4(out, n_train=24, n_val=6):
    """si_s4 生产布局的迷你版 + val 划分 + 分层索引（调用迁移后的 dataset_maker 脚本）。

    步数 4~21 覆盖 prefix-drop 资格门槛（prefix_context_turns=16 需 >=18 步）；val 的第 0 条
    源文本与训练集重复，用来验证按源文本指纹剔除。
    """
    root = os.path.join(out, 's4')
    shutil.rmtree(root, ignore_errors=True)
    audio = os.path.join(root, 'si_s4', 'audio')
    rng = np.random.RandomState(4)
    maker = os.path.join(CUSTOM_DIR, '..', 'dataset_maker', 'streaming_si')
    train_dirs = []
    for d in ('en2zh', 'zh2en'):
        for split, n in (('train', n_train), ('val', n_val)):
            rec_dir = os.path.join(root, 'si_s4', 'records', d, 'chunk_640ms', split, 'records')
            os.makedirs(rec_dir)
            recs = [_s4_record(d, f'{d}-{split}-{i:04d}', 4 + (i * 5) % 18, (i if split == 'train' else i + 1),
                               rng, audio) for i in range(n)]
            with open(os.path.join(rec_dir, 'part-00000.jsonl'), 'w', encoding='utf-8') as f:
                for r in recs:
                    f.write(json.dumps(r, ensure_ascii=False) + '\n')
            json.dump({'direction': d, 'split': split, 'chunk_ms': 640, 'num_records': n,
                       'parts': [{'file': 'part-00000.jsonl', 'num_records': n}]},
                      open(os.path.join(rec_dir, '_manifest.json'), 'w'))
            if split == 'train':
                train_dirs.append(rec_dir)
                subprocess.run([sys.executable, os.path.join(maker, 'build_record_index.py'), '--records', rec_dir,
                                '--direction', d, '--data_root', audio, '--out', os.path.join(root, 'index'),
                                '--workers', '1'], check=True, stdout=subprocess.DEVNULL)
            else:
                subprocess.run([sys.executable, os.path.join(maker, 'build_s4_val_split.py'), '--records', rec_dir,
                                '--direction', d, '--n_val', '2', '--out', os.path.join(root, 'val'),
                                '--data_root', audio], check=True, stdout=subprocess.DEVNULL)
    print('SI_S4_DATA=' + ','.join(train_dirs))
    print('SI_S4_DATA_ROOT=' + audio)
    print('SI_S4_VAL_DIR=' + os.path.join(root, 'val'))
    print('SI_S4_INDEX_DIR=' + os.path.join(root, 'index'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tokenizer_dir', default=None, help='全量生成时必需（Qwen3 tokenizer 目录）')
    ap.add_argument('--out', required=True)
    ap.add_argument('--only', choices=['s4', 'model_4b'], default=None,
                    help='s4：只（重新）生成 S4 数据；model_4b：只生成 streaming_si_4b 的 tiny 模型 <out>/model_4b'
                         '（需 --qwen35_tokenizer_dir；数据与 0.6B 变体共用）')
    ap.add_argument('--qwen35_tokenizer_dir', default=None, help='Qwen3.5 tokenizer 目录（model_4b 必需）')
    args = ap.parse_args()
    if args.only == 's4':
        make_s4(args.out)
        return
    if args.only == 'model_4b':
        if not args.qwen35_tokenizer_dir:
            ap.error('--only model_4b 需要 --qwen35_tokenizer_dir')
        thinker, talker_base = make_bases_4b(args.qwen35_tokenizer_dir, args.out)
        vox = os.path.join(args.out, 'base_voxtral')
        if not os.path.isdir(vox):
            vox = make_voxtral(args.out)
        subprocess.run([sys.executable, os.path.join(HERE, 'export_streaming_si.py'), '--variant', 'streaming_si_4b',
                        '--thinker_dir', thinker, '--voxtral_dir', vox, '--talker_overrides', json.dumps(TINY_TALKER),
                        '--talker_init_dir', talker_base, '--out_dir', os.path.join(args.out, 'model_4b'),
                        '--dtype', 'float32'], check=True)
        return
    if not args.tokenizer_dir:
        ap.error('--tokenizer_dir 必需')
    shutil.rmtree(args.out, ignore_errors=True)
    os.makedirs(args.out)
    thinker, vox = make_bases(args.tokenizer_dir, args.out)
    subprocess.run([sys.executable, os.path.join(HERE, 'export_streaming_si.py'), '--thinker_dir', thinker,
                    '--voxtral_dir', vox, '--out_dir', os.path.join(args.out, 'model'), '--dtype', 'float32'],
                   check=True)
    print('SI_S1_DATA=' + make_s1(args.out))
    print('SI_P2_DATA=' + make_p2(args.out))
    make_s4(args.out)


if __name__ == '__main__':
    main()

"""export_streaming_si.py 端到端：tiny 基座目录 -> 导出 -> 回读 -> 叠加源 .pth。

    SI_QWEN3_TOKENIZER=/path/to/Qwen3-0.6B python -m pytest custom/streaming_si/tests/test_export.py -q

需要真实 Qwen3 tokenizer（验证 <|audio_pad|> id 与词表容量）；模型权重为随机 tiny。
"""
import json
import os
import shutil
import subprocess
import sys

import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))              # custom/streaming_si/tests
CUSTOM_DIR = os.path.dirname(os.path.dirname(HERE))           # custom/
EXPORT = os.path.join(os.path.dirname(HERE), 'scripts', 'export_streaming_si.py')
sys.path.insert(0, CUSTOM_DIR)

from streaming_si.model.modeling_streaming_si import StreamingSIForConditionalGeneration  # noqa: E402
from streaming_si.model.voxtral_compat import (VoxtralEncoder, VoxtralEncoderConfig,  # noqa: E402
                                               VoxtralMultiModalProjector, VoxtralProjectorConfig)

TOKENIZER = os.environ.get('SI_QWEN3_TOKENIZER')
pytestmark = pytest.mark.skipif(not (TOKENIZER and os.path.isdir(TOKENIZER)),
                                reason='需要 SI_QWEN3_TOKENIZER')

TINY_AUDIO = dict(hidden_size=32, intermediate_size=64, num_hidden_layers=1, num_attention_heads=2,
                  num_key_value_heads=2, head_dim=16, num_mel_bins=128, sliding_window=750,
                  max_position_embeddings=1500, rms_norm_eps=1e-5,
                  rope_parameters={'rope_theta': 1e6, 'rope_type': 'default'},
                  model_type='voxtral_realtime_encoder')
TEXT_HIDDEN = 48


def _make_thinker(d):
    from transformers import Qwen3Config, Qwen3ForCausalLM
    torch.manual_seed(1)
    cfg = Qwen3Config(hidden_size=64, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                      head_dim=16, intermediate_size=128, vocab_size=151936, tie_word_embeddings=True)
    Qwen3ForCausalLM(cfg).save_pretrained(d)
    for f in os.listdir(TOKENIZER):
        if f.startswith(('tokenizer', 'vocab', 'merges', 'special_tokens')):
            shutil.copy(os.path.join(TOKENIZER, f), d)


def _make_voxtral(d):
    from safetensors.torch import save_file
    os.makedirs(d)
    torch.manual_seed(2)
    enc = VoxtralEncoder(VoxtralEncoderConfig(**TINY_AUDIO))
    proj = VoxtralMultiModalProjector(VoxtralProjectorConfig(32, TEXT_HIDDEN, 4))
    sd = {f'audio_tower.{k}': v for k, v in enc.state_dict().items()}
    sd.update({f'multi_modal_projector.{k}': v for k, v in proj.state_dict().items()})
    sd['language_model.dummy'] = torch.zeros(1)             # 应被忽略
    save_file({k: v.contiguous() for k, v in sd.items()}, os.path.join(d, 'model.safetensors'))
    json.dump({'audio_config': TINY_AUDIO, 'text_config': {'hidden_size': TEXT_HIDDEN},
               'downsample_factor': 4, 'projector_hidden_act': 'gelu', 'model_type': 'voxtral_realtime'},
              open(os.path.join(d, 'config.json'), 'w'))
    json.dump({'feature_extractor': {'feature_size': 128, 'global_log_mel_max': 1.5, 'hop_length': 160,
                                     'n_fft': 400, 'sampling_rate': 16000, 'win_length': 400}},
              open(os.path.join(d, 'processor_config.json'), 'w'))
    return enc, proj


def _export(args):
    r = subprocess.run([sys.executable, EXPORT, *args],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-3000:]
    return r.stdout


def test_export_roundtrip_and_source_ckpt(tmp_path):
    thinker, vox, out = tmp_path / 'qwen3', tmp_path / 'vox', tmp_path / 'out'
    _make_thinker(str(thinker))
    enc, proj = _make_voxtral(str(vox))
    log = _export(['--thinker_dir', str(thinker), '--voxtral_dir', str(vox), '--out_dir', str(out),
                   '--dtype', 'float32'])
    assert 'global_log_mel_max=1.5' in log

    m = StreamingSIForConditionalGeneration.from_pretrained(str(out)).eval()
    assert m.config.audio_ids == [151669]
    assert m.config.audio_hidden_size == TEXT_HIDDEN
    assert torch.equal(m.audio_encoder.encoder.layers[0].self_attn.q_proj.weight,
                       enc.layers[0].self_attn.q_proj.weight)
    assert torch.equal(m.talker_text_embed.weight, m.thinker.embed_tokens.weight)   # 副本 = 基座
    assert json.load(open(out / 'voxtral_processor_config.json'))['global_log_mel_max'] == 1.5

    # 叠加一个“源项目 checkpoint”：thinker + talker 被改写，talker_text_embed 缺席时保持基座副本
    sd = {k: v.clone() for k, v in m.state_dict().items()
          if not k.startswith(('audio_encoder.', 'talker_text_embed.'))}
    sd['thinker_model.model.embed_tokens.weight'].add_(1.0)
    sd['thinker_model.lm_head.weight'] = sd['thinker_model.model.embed_tokens.weight']
    sd['talker.llm_decoder.weight'].mul_(0.5)
    sd['vision_proj.0.weight'] = torch.zeros(1)            # 旧 checkpoint 残留键，应被忽略
    torch.save(sd, tmp_path / 'fake_1024_talker8.pth')
    out2 = tmp_path / 'out2'
    _export(['--thinker_dir', str(thinker), '--voxtral_dir', str(vox), '--out_dir', str(out2),
             '--dtype', 'float32', '--source_ckpt', str(tmp_path / 'fake_1024_talker8.pth')])
    m2 = StreamingSIForConditionalGeneration.from_pretrained(str(out2)).eval()
    assert torch.equal(m2.thinker.embed_tokens.weight, sd['thinker_model.model.embed_tokens.weight'])
    assert torch.equal(m2.talker.llm_decoder.weight, sd['talker.llm_decoder.weight'])
    assert torch.equal(m2.talker_text_embed.weight, m.talker_text_embed.weight)     # 仍是基座副本
    assert m2.lm_head.weight.data_ptr() == m2.thinker.embed_tokens.weight.data_ptr()


def test_export_rejects_bad_ckpt(tmp_path):
    thinker, vox = tmp_path / 'qwen3', tmp_path / 'vox'
    _make_thinker(str(thinker))
    _make_voxtral(str(vox))
    torch.save({'talker.llm_decoder.weight': torch.zeros(3, 3)}, tmp_path / 'bad.pth')
    r = subprocess.run([sys.executable, EXPORT,
                        '--thinker_dir', str(thinker), '--voxtral_dir', str(vox),
                        '--out_dir', str(tmp_path / 'o'), '--source_ckpt', str(tmp_path / 'bad.pth')],
                       capture_output=True, text=True)
    assert r.returncode != 0 and 'shape 不符' in r.stderr


def test_export_bundles_speech_decoder(tmp_path):
    """--cosyvoice_dir：flow/HiFT 文件进 speech_decoder/，但不进 state_dict；目录不全时在写权重前报错。"""
    from streaming_si.model.speech_decoder import resolve_speech_decoder
    thinker, vox, out = tmp_path / 'qwen3', tmp_path / 'vox', tmp_path / 'out'
    _make_thinker(str(thinker))
    _make_voxtral(str(vox))
    cosy = tmp_path / 'Fun-CosyVoice3-0.5B'
    cosy.mkdir()
    for name in ('flow.pt', 'hift.pt', 'cosyvoice3.yaml', 'campplus.onnx'):
        (cosy / name).write_bytes(name.encode())
    bad = tmp_path / 'bad_cosy'
    bad.mkdir()
    r = subprocess.run([sys.executable, EXPORT, '--thinker_dir', str(thinker),
                        '--voxtral_dir', str(vox), '--out_dir', str(tmp_path / 'o2'), '--cosyvoice_dir', str(bad)],
                       capture_output=True, text=True)
    assert r.returncode != 0 and 'flow.pt' in r.stderr and not (tmp_path / 'o2').exists()

    _export(['--thinker_dir', str(thinker), '--voxtral_dir', str(vox), '--out_dir', str(out),
             '--dtype', 'float32', '--cosyvoice_dir', str(cosy)])
    dec, yaml = resolve_speech_decoder(str(out))
    assert yaml == 'cosyvoice3.yaml' and (out / 'speech_decoder' / 'flow.pt').read_bytes() == b'flow.pt'
    assert json.load(open(out / 'streaming_si_export.json'))['cosyvoice_dir'] == str(cosy)
    m = StreamingSIForConditionalGeneration.from_pretrained(str(out))
    assert m.speech_decoder is None and not any(k.startswith('speech_decoder') for k in m.state_dict())

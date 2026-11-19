"""Voxtral 兼容实现对拍：transformers>=5.2 官方实现 vs 本仓库 4.57 兼容实现。

两个版本的 transformers 不能共存于一个环境，因此分两步：

  # 1) 在装有 transformers>=5.2 的环境生成参考（随机权重，不需要下载 checkpoint）
  python custom/streaming_si/tests/test_voxtral_compat.py reference /tmp/voxtral_ref.pt

  # 2) 在本仓库环境（transformers 4.57.3）加载同一份权重与输入，逐元素比较
  python custom/streaming_si/tests/test_voxtral_compat.py compare /tmp/voxtral_ref.pt

覆盖：整段编码、分块流式编码（KV cache + 卷积 padding cache，总长超过 sliding_window）、
projector、整段 log-mel。
"""
import os
import sys

import torch

# 足够小以便 CPU 快跑，但 sliding_window 取小值以覆盖“超出窗口”的掩码路径
TINY = dict(hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4,
            head_dim=16, num_mel_bins=128, max_position_embeddings=1500, sliding_window=12,
            rms_norm_eps=1e-5, rope_parameters={"rope_type": "default", "rope_theta": 10000.0})
TEXT_HIDDEN = 96
DOWNSAMPLE = 4
MEL_T = 64            # 50Hz 下 conv stride 2 后 32 帧，远超 sliding_window=12
CHUNKS = [16, 8, 24, 16]
WAV_SAMPLES = 16000


def _inputs():
    g = torch.Generator().manual_seed(0)
    mel = torch.randn(1, 128, MEL_T, generator=g)
    wav = torch.randn(WAV_SAMPLES, generator=g) * 0.1
    return mel, wav


def _run_encoder(encoder, mel, stream_kwargs_fn):
    with torch.no_grad():
        full = encoder(input_features=mel).last_hidden_state
        outs, kv, pad, start = [], None, None, 0
        for n in CHUNKS:
            out = encoder(input_features=mel[..., start:start + n], **stream_kwargs_fn(kv, pad))
            kv, pad = out.past_key_values, out.padding_cache
            outs.append(out.last_hidden_state)
            start += n
    return full, torch.cat(outs, dim=1)


def reference(path):
    import transformers
    from transformers.models.voxtral_realtime.configuration_voxtral_realtime import (
        VoxtralRealtimeConfig, VoxtralRealtimeEncoderConfig)
    from transformers.models.voxtral_realtime.feature_extraction_voxtral_realtime import (
        VoxtralRealtimeFeatureExtractor)
    from transformers.models.voxtral_realtime.modeling_voxtral_realtime import (
        VoxtralRealtimeEncoder, VoxtralRealtimeMultiModalProjector)

    torch.manual_seed(0)
    enc_cfg = VoxtralRealtimeEncoderConfig(**TINY)
    enc_cfg._attn_implementation = "eager"
    encoder = VoxtralRealtimeEncoder(enc_cfg).eval()
    cfg = VoxtralRealtimeConfig(audio_config=enc_cfg.to_dict(),
                                text_config={"hidden_size": TEXT_HIDDEN},
                                downsample_factor=DOWNSAMPLE)
    projector = VoxtralRealtimeMultiModalProjector(cfg).eval()

    mel, wav = _inputs()
    full, stream = _run_encoder(
        encoder, mel,
        lambda kv, pad: dict(past_key_values=kv, padding_cache=pad, use_cache=True, use_padding_cache=True))
    b, t, d = full.shape
    usable = t // DOWNSAMPLE * DOWNSAMPLE
    with torch.no_grad():
        proj = projector(full[:, :usable].reshape(b, usable // DOWNSAMPLE, d * DOWNSAMPLE))

    fe = VoxtralRealtimeFeatureExtractor(global_log_mel_max=2.0)
    log_mel = fe(wav.numpy(), sampling_rate=16000, return_tensors="pt", padding=False)["input_features"]

    torch.save(dict(version=transformers.__version__,
                    encoder=encoder.state_dict(), projector=projector.state_dict(),
                    mel=mel, wav=wav, full=full, stream=stream, proj=proj, log_mel=log_mel), path)
    print(f"reference saved: {path} (transformers {transformers.__version__})")


def compare(path, atol=1e-5):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # custom/
    import transformers
    from streaming_si.model.voxtral_compat import (VoxtralEncoder, VoxtralEncoderConfig, VoxtralFeatureExtractor,
                                                   VoxtralMultiModalProjector, VoxtralProjectorConfig)

    ref = torch.load(path, weights_only=False)
    encoder = VoxtralEncoder(VoxtralEncoderConfig(**TINY)).eval()
    missing, unexpected = encoder.load_state_dict(ref["encoder"], strict=True), None
    projector = VoxtralMultiModalProjector(VoxtralProjectorConfig(
        audio_hidden_size=TINY["hidden_size"], text_hidden_size=TEXT_HIDDEN,
        downsample_factor=DOWNSAMPLE)).eval()
    projector.load_state_dict(ref["projector"], strict=True)

    full, stream = _run_encoder(
        encoder, ref["mel"],
        lambda kv, pad: dict(past_key_values=kv, padding_cache=pad, use_cache=True, use_padding_cache=True))
    b, t, d = full.shape
    usable = t // DOWNSAMPLE * DOWNSAMPLE
    with torch.no_grad():
        proj = projector(full[:, :usable].reshape(b, usable // DOWNSAMPLE, d * DOWNSAMPLE))
    log_mel = VoxtralFeatureExtractor(global_log_mel_max=2.0)(
        ref["wav"].numpy(), sampling_rate=16000)["input_features"]

    results = {}
    for name, got in [("full", full), ("stream", stream), ("proj", proj), ("log_mel", log_mel)]:
        exp = ref[name]
        assert got.shape == exp.shape, f"{name}: shape {tuple(got.shape)} != {tuple(exp.shape)}"
        results[name] = (got - exp).abs().max().item()
    # 流式与整段在各自实现内部也应一致（因果 + 滑窗）
    results["stream_vs_full(self)"] = (stream - full).abs().max().item()

    print(f"reference transformers {ref['version']}  vs  compat on transformers {transformers.__version__}")
    ok = True
    for name, diff in results.items():
        flag = "OK " if diff <= atol else "FAIL"
        ok &= diff <= atol
        print(f"  [{flag}] {name:22s} max|Δ| = {diff:.3e}")
    if not ok:
        raise SystemExit(1)
    print("ALL PASS")


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] not in ("reference", "compare"):
        raise SystemExit(__doc__)
    {"reference": reference, "compare": compare}[sys.argv[1]](sys.argv[2])

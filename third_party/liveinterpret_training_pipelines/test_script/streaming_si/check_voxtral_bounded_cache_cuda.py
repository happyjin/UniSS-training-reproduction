"""Real-checkpoint CUDA regression for physical Voxtral KV tail pruning.

No CPU/random-model fallback. Runs one-shot, unbounded chunked, and bounded
chunked encoding on the same actual-audio mel. Repeats that mel when needed to
exercise at least three sliding windows; output JSON identifies this stress input.
Run separately from full S5 inference to avoid concurrent GPU memory pressure.
"""
import argparse
import glob
import hashlib
import json
from pathlib import Path
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'custom'))
from streaming_si.model.voxtral_compat import VoxtralEncoder, VoxtralEncoderConfig, VoxtralFeatureExtractor


def load_encoder(model_dir, device, dtype):
    from safetensors import safe_open
    directory = Path(model_dir)
    raw = json.loads((directory / 'config.json').read_text())
    config = raw.get('voxtral_audio_config') or raw.get('audio_config')
    if not config:
        raise ValueError('checkpoint must embed voxtral_audio_config or audio_config')
    prefix = 'audio_encoder.encoder.' if raw.get('voxtral_audio_config') else 'audio_tower.'
    state = {}
    for file in sorted(glob.glob(str(directory / '*.safetensors'))):
        with safe_open(file, framework='pt', device='cpu') as handle:
            for key in handle.keys():
                if key.startswith(prefix):
                    state[key[len(prefix):]] = handle.get_tensor(key)
    if not state:
        raise ValueError(f'no actual encoder weights with prefix {prefix}')
    encoder = VoxtralEncoder(VoxtralEncoderConfig(**config))
    encoder.load_state_dict(state, strict=True)
    return encoder.eval().to(device=device, dtype=dtype)


def compare(label, actual, expected, atol, rtol):
    diff = (actual.float() - expected.float()).abs()
    result = dict(comparison=label, max_abs=float(diff.max()), mean_abs=float(diff.mean()),
                  within_tolerance=bool(torch.allclose(actual, expected, atol=atol, rtol=rtol)))
    print(json.dumps(result), flush=True)
    if not result['within_tolerance']:
        raise AssertionError(f'{label} failed: {result}')
    return result


@torch.inference_mode()
def check(args):
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required; this harness never runs model execution on CPU')
    import soundfile as sf
    import torchaudio
    dtype = getattr(torch, args.dtype)
    model = load_encoder(args.model_dir, args.device, dtype)
    wave, rate = sf.read(args.audio, dtype='float32', always_2d=True)
    wav = torch.from_numpy(wave.mean(axis=-1))
    if rate != 16000:
        wav = torchaudio.functional.resample(wav, rate, 16000)
    extractor = VoxtralFeatureExtractor.from_pretrained(args.model_dir)
    mel = extractor(wav, sampling_rate=16000)['input_features']
    # Encoder consumes B,C,T; extractor returns C,T for a mono waveform.
    if mel.ndim == 2:
        mel = mel.unsqueeze(0)
    window = int(model.config.sliding_window)
    total = ((window * args.windows * 2 + 7) // 8) * 8
    repeats = (total + mel.size(-1) - 1) // mel.size(-1)
    mel = mel.repeat(1, 1, repeats)[..., :total].to(args.device, dtype)
    one_shot = model(mel).last_hidden_state
    parts, bounded_parts = [], []
    reference_cache = bounded_cache = reference_conv = bounded_conv = None
    trace = []
    for start in range(0, total, args.chunk_mel):
        chunk = mel[..., start:start + args.chunk_mel]
        ref = model(chunk, past_key_values=reference_cache, padding_cache=reference_conv,
                    use_cache=True, use_padding_cache=True, bounded_cache=False)
        reference_cache, reference_conv = ref.past_key_values, ref.padding_cache
        out = model(chunk, past_key_values=bounded_cache, padding_cache=bounded_conv,
                    use_cache=True, use_padding_cache=True, bounded_cache=True)
        bounded_cache, bounded_conv = out.past_key_values, out.padding_cache
        parts.append(ref.last_hidden_state)
        bounded_parts.append(out.last_hidden_state)
        lengths = [int(layer.keys.shape[-2]) for layer in bounded_cache.layers]
        if max(lengths) > window - 1:
            raise AssertionError(f'physical KV exceeds window: {lengths}')
        expected_seen = min(start + args.chunk_mel, total) // 2
        if bounded_cache._voxtral_seen_tokens != expected_seen:
            raise AssertionError('absolute RoPE position counter did not advance monotonically')
        for name, conv in reference_conv.layers.items():
            torch.testing.assert_close(bounded_conv.layers[name].cache, conv.cache, atol=0, rtol=0)
        # Verify actual allocated tensor storage, not merely a narrow view shape.
        for layer in bounded_cache.layers:
            for tensor in (layer.keys, layer.values):
                if tensor.untyped_storage().nbytes() > tensor.numel() * tensor.element_size():
                    raise AssertionError('KV tail is a view retaining historical allocation')
        trace.append(dict(seen=expected_seen, kv=max(lengths)))
    reference = torch.cat(parts, dim=1)
    bounded = torch.cat(bounded_parts, dim=1)
    comparisons = [compare('bounded_vs_unbounded_incremental', bounded, reference, args.atol, args.rtol),
                   compare('unbounded_incremental_vs_one_shot', reference, one_shot, args.atol, args.rtol),
                   compare('bounded_incremental_vs_one_shot', bounded, one_shot, args.atol, args.rtol)]
    result = dict(status='passed', model_dir=args.model_dir, audio=args.audio,
                  input='real_audio_mel_repeated_for_window_stress', mel_frames=total,
                  windows=args.windows, sliding_window=window, chunk_mel=args.chunk_mel,
                  dtype=args.dtype, gpu=torch.cuda.get_device_name(args.device),
                  encoder_sha256=hashlib.sha256((ROOT / 'custom/streaming_si/model/voxtral_compat.py').read_bytes()).hexdigest(),
                  comparisons=comparisons, cache_trace=trace,
                  peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(args.device))
    print(json.dumps(result), flush=True)
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model_dir', required=True)
    parser.add_argument('--audio', required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--dtype', choices=['float32', 'bfloat16'], default='float32')
    parser.add_argument('--windows', type=int, default=3)
    parser.add_argument('--chunk_mel', type=int, default=64)
    parser.add_argument('--atol', type=float, default=1e-4)
    parser.add_argument('--rtol', type=float, default=1e-4)
    parser.add_argument('--out')
    args = parser.parse_args()
    if args.windows < 3 or args.chunk_mel <= 0 or args.chunk_mel % 8:
        parser.error('require >=3 windows and a positive multiple-of-8 chunk_mel')
    check(args)

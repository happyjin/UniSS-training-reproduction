# Copyright (c) ModelScope Contributors. All rights reserved.
"""Speaker x-vector extractor for Qwen-OmniSpeak voice-clone.

This wraps the *upstream* Qwen3-TTS ECAPA speaker encoder (``Qwen3TTSSpeakerEncoder``)
and its ``mel_spectrogram`` verbatim — we only *reference* the upstream code and
checkpoint, never modify it. The encoder is:

  * loaded lazily as a module-level singleton (like ``get_codec_tokenizer``),
  * NOT a registered submodule of the composite model (so it is invisible to
    ``save_pretrained`` / DeepSpeed sharding / the optimizer), and
  * frozen + eval-only.

The full resample -> mel -> encoder pipeline runs inside the model ``forward``
(the dataloader only loads the raw reference waveform). Given a reference
waveform it returns a single ``[enc_dim]`` x-vector, matching upstream
``Qwen3TTSForConditionalGeneration.extract_speaker_embedding``.
"""
import os
import re
import sys
import threading
from io import BytesIO
from typing import Optional

import numpy as np
import torch


class _MinorWaveformRangeWarningFilter:
    """Hide only small Qwen3-TTS range prints without changing the waveform."""

    _pattern = re.compile(
        r'^\[WARNING\] (?:Min|Max) value of input waveform signal is '
        r'(-?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*$')

    def __init__(self, stream, threshold: float):
        self._stream = stream
        self._threshold = float(threshold)
        self._drop_newline = False
        self._qwen3_minor_range_filter = True

    def write(self, buf):
        match = self._pattern.match(buf)
        if match and abs(float(match.group(1))) <= self._threshold:
            self._drop_newline = True
            return len(buf)
        if self._drop_newline and buf == '\n':
            self._drop_newline = False
            return len(buf)
        self._drop_newline = False
        return self._stream.write(buf)

    def flush(self):
        return self._stream.flush()

    def __getattr__(self, name):
        return getattr(self._stream, name)


def _install_minor_range_warning_filter() -> None:
    raw = os.environ.get('QWEN3_TTS_MINOR_RANGE_WARNING_THRESHOLD')
    if not raw or getattr(sys.stdout, '_qwen3_minor_range_filter', False):
        return
    threshold = float(raw)
    if threshold <= 1.0:
        raise ValueError(
            'QWEN3_TTS_MINOR_RANGE_WARNING_THRESHOLD must be greater than 1.0')
    sys.stdout = _MinorWaveformRangeWarningFilter(sys.stdout, threshold)


_install_minor_range_warning_filter()

try:
    from .codec_tokenizer import _resolve_audio_path
    from .wds_audio import resolve_audio_reference
except ImportError:  # pragma: no cover - imported as a top-level module by ms-swift.
    from codec_tokenizer import _resolve_audio_path  # pyright: ignore[reportMissingImports]
    from wds_audio import resolve_audio_reference  # pyright: ignore[reportMissingImports]


# Default source for the frozen ECAPA speaker-encoder weights. Can be either
#   * a local directory (we scan ``*.safetensors`` incl. ``qwen3_tts_extras``),
#   * a single ``.safetensors`` file (e.g. the export sidecar), or
#   * a HF repo id (best-effort download via ``huggingface_hub``).
# Keys prefixed with ``speaker_encoder.`` are extracted (prefix stripped).
DEFAULT_SPEAKER_ENCODER_ID = 'Qwen/Qwen3-TTS-12Hz-1.7B-Base'
# ECAPA final embedding dim used by Qwen3-TTS-*-Base (config.json
# ``speaker_encoder_config.enc_dim``). Upstream default is 192; Base uses 2048.
DEFAULT_ENC_DIM = 2048
DEFAULT_SAMPLE_RATE = 24000

# Mel parameters copied verbatim from
# ``Qwen3TTSForConditionalGeneration.extract_speaker_embedding``.
_MEL_KWARGS = dict(n_fft=1024, num_mels=128, sampling_rate=24000,
                   hop_size=256, win_size=1024, fmin=0, fmax=12000)


def _import_upstream():
    """Import the upstream Qwen3-TTS speaker encoder / mel / config.

    Prefer the installed ``qwen_tts`` package; fall back to the in-repo copy at
    ``project/Qwen3-TTS`` if the package layout differs.
    """
    try:
        from qwen_tts.core.models.modeling_qwen3_tts import (  # type: ignore
            Qwen3TTSSpeakerEncoder, mel_spectrogram)
        from qwen_tts.core.models.configuration_qwen3_tts import (  # type: ignore
            Qwen3TTSSpeakerEncoderConfig)
        return Qwen3TTSSpeakerEncoder, mel_spectrogram, Qwen3TTSSpeakerEncoderConfig
    except Exception:
        pass
    # Fallback: add the in-repo Qwen3-TTS source tree to sys.path.
    import sys
    here = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(here, os.pardir))
    candidate = os.path.join(repo_root, 'project', 'Qwen3-TTS')
    if os.path.isdir(candidate) and candidate not in sys.path:
        sys.path.insert(0, candidate)
    from qwen_tts.core.models.modeling_qwen3_tts import (  # type: ignore
        Qwen3TTSSpeakerEncoder, mel_spectrogram)
    from qwen_tts.core.models.configuration_qwen3_tts import (  # type: ignore
        Qwen3TTSSpeakerEncoderConfig)
    return Qwen3TTSSpeakerEncoder, mel_spectrogram, Qwen3TTSSpeakerEncoderConfig


def _collect_speaker_encoder_state(source: str) -> dict:
    """Load a ``speaker_encoder.*`` state dict from ``source``.

    Accepts a directory, a single ``.safetensors`` file, or a HF repo id.
    Keys are returned with the ``speaker_encoder.`` prefix stripped so they can
    be loaded straight into ``Qwen3TTSSpeakerEncoder``.
    """
    from safetensors.torch import load_file

    files = []
    resolved = _resolve_audio_path(source) if isinstance(source, str) else source
    if os.path.isfile(resolved) and resolved.endswith('.safetensors'):
        files = [resolved]
    elif os.path.isdir(resolved):
        # Prefer the export sidecar; otherwise scan all shards.
        sidecar = os.path.join(resolved, 'qwen3_tts_extras.safetensors')
        if os.path.isfile(sidecar):
            files = [sidecar]
        else:
            files = sorted(
                os.path.join(resolved, f) for f in os.listdir(resolved)
                if f.endswith('.safetensors'))
    else:
        # Treat as a HF repo id: best-effort download of common shard names.
        try:
            from huggingface_hub import hf_hub_download
        except Exception as e:  # pragma: no cover
            raise FileNotFoundError(
                f'Speaker-encoder source {source!r} is not a local path and '
                f'huggingface_hub is unavailable ({e}).')
        for name in ('qwen3_tts_extras.safetensors', 'model.safetensors'):
            try:
                files.append(hf_hub_download(repo_id=source, filename=name))
                break
            except Exception:
                continue
        if not files:
            raise FileNotFoundError(
                f'Could not locate a speaker-encoder safetensors under {source!r}.')

    state = {}
    for path in files:
        try:
            shard = load_file(path)
        except Exception:
            continue
        for k, v in shard.items():
            if k.startswith('speaker_encoder.'):
                state[k[len('speaker_encoder.'):]] = v
    if not state:
        raise KeyError(
            f'No ``speaker_encoder.*`` weights found under {source!r}. Set '
            f'QWEN3_TTS_SPEAKER_ENCODER to the Qwen3-TTS-Base dir / export '
            f'sidecar / repo id that contains them.')
    return state


def _load_speaker_encoder_config_dict(source: str) -> dict:
    """Read ``speaker_encoder_config`` from the Qwen3-TTS-Base ``config.json``.

    Mirrors how ``Qwen3TTSForConditionalGeneration`` builds its speaker encoder
    (``Qwen3TTSSpeakerEncoder(config.speaker_encoder_config)``): we take the exact
    same sub-config from Base so ``enc_dim`` / ``sample_rate`` are never hardcoded.
    Accepts a local dir / a file (its dir is used) / a HF repo id. Returns ``{}``
    on failure so the caller can fall back to library defaults.
    """
    import json

    cfg_path = None
    resolved = _resolve_audio_path(source) if isinstance(source, str) else source
    if os.path.isdir(resolved):
        cand = os.path.join(resolved, 'config.json')
        cfg_path = cand if os.path.isfile(cand) else None
    elif os.path.isfile(resolved):
        cand = os.path.join(os.path.dirname(resolved), 'config.json')
        cfg_path = cand if os.path.isfile(cand) else None
    else:
        # HF repo id: best-effort download of config.json.
        try:
            from huggingface_hub import hf_hub_download
            cfg_path = hf_hub_download(repo_id=source, filename='config.json')
        except Exception:
            cfg_path = None
    if not cfg_path:
        return {}
    try:
        with open(cfg_path, 'r', encoding='utf-8') as f:
            full = json.load(f)
    except Exception:
        return {}
    return dict(full.get('speaker_encoder_config') or {})


class Qwen3TTSSpeakerEmbedder:
    """Frozen ECAPA speaker embedder (lazy, non-registered, eval-only).

    Built exactly like upstream Qwen3-TTS: instantiate ``Qwen3TTSSpeakerEncoder``
    from Base's ``speaker_encoder_config`` and load the Base ``speaker_encoder.*``
    weights. It is NOT a composite submodule (invisible to save_pretrained /
    DeepSpeed / optimizer) and always frozen + eval.
    """

    def __init__(self, encoder_id: Optional[str] = None, enc_dim: Optional[int] = None):
        self.encoder_id = encoder_id or os.environ.get(
            'QWEN3_TTS_SPEAKER_ENCODER', DEFAULT_SPEAKER_ENCODER_ID)
        # ``enc_dim`` is only a fallback; the real value is read from Base's
        # config.json (like upstream). ``None`` -> use Base config, else default.
        self._enc_dim_override = int(enc_dim) if enc_dim is not None else None
        self._sample_rate = DEFAULT_SAMPLE_RATE
        self._encoder = None
        self._mel_fn = None
        self._lock = threading.Lock()

    def _ensure_loaded(self, device, dtype):
        if self._encoder is not None:
            return
        with self._lock:
            if self._encoder is not None:
                return
            Qwen3TTSSpeakerEncoder, mel_spectrogram, Qwen3TTSSpeakerEncoderConfig = _import_upstream()
            # Build the sub-config exactly from Base's config.json (same as
            # ``Qwen3TTSForConditionalGeneration.__init__``); fall back to the
            # library default + optional enc_dim override when unavailable.
            cfg_dict = _load_speaker_encoder_config_dict(self.encoder_id)
            if self._enc_dim_override is not None:
                cfg_dict.setdefault('enc_dim', self._enc_dim_override)
            cfg = Qwen3TTSSpeakerEncoderConfig(**cfg_dict)
            self._sample_rate = int(getattr(cfg, 'sample_rate', DEFAULT_SAMPLE_RATE))
            # Guard against a stray DeepSpeed zero.Init hook partitioning this
            # auxiliary module (same pattern as the codec tokenizer loader).
            try:
                from deepspeed.runtime.zero.partition_parameters import Init as _ZeroInit
                _disable_zero_init = _ZeroInit(enabled=False)
            except Exception:
                _disable_zero_init = None
            if _disable_zero_init is not None:
                with _disable_zero_init:
                    encoder = Qwen3TTSSpeakerEncoder(cfg)
            else:
                encoder = Qwen3TTSSpeakerEncoder(cfg)
            state = _collect_speaker_encoder_state(self.encoder_id)
            missing, unexpected = encoder.load_state_dict(state, strict=False)
            if len(state) == 0 or (missing and len(missing) == len(list(encoder.state_dict()))):
                raise RuntimeError(
                    f'Speaker-encoder weights failed to load from {self.encoder_id!r} '
                    f'(missing={len(missing)}, unexpected={len(unexpected)}).')
            encoder.eval()
            for p in encoder.parameters():
                p.requires_grad_(False)
            # Keep the encoder in float32 for numerically-stable mel/STFT, then
            # cast the resulting x-vector to the caller dtype in ``extract``.
            self._encoder = encoder.to(device=device, dtype=torch.float32)
            self._mel_fn = mel_spectrogram

    @torch.no_grad()
    def extract(self, waveform, sr: int, device, dtype) -> torch.Tensor:
        """Return a single ``[enc_dim]`` x-vector for one reference waveform.

        ``waveform`` is a mono 1-D array/tensor at sampling rate ``sr``. Resample
        to the encoder's sample rate, compute the mel spectrogram and run the
        ECAPA encoder — the whole pipeline runs here (inside ``forward``), not in
        the dataloader. Mirrors upstream ``extract_speaker_embedding``.
        """
        self._ensure_loaded(device, dtype)
        # -> float32 mono tensor on the encoder device.
        if isinstance(waveform, np.ndarray):
            wav = torch.from_numpy(waveform.astype(np.float32))
        else:
            wav = waveform.detach().to(torch.float32)
        if wav.dim() > 1:
            wav = wav.mean(dim=0) if wav.shape[0] < wav.shape[-1] else wav.reshape(-1)
        wav = wav.to(device)

        if int(sr) != int(self._sample_rate):
            import torchaudio
            wav = torchaudio.functional.resample(wav, orig_freq=int(sr),
                                                 new_freq=int(self._sample_rate))

        mels = self._mel_fn(wav.unsqueeze(0), **_MEL_KWARGS).transpose(1, 2)
        emb = self._encoder(mels.to(device=device, dtype=torch.float32))[0]
        return emb.to(dtype=dtype)


_SPEAKER_EMBEDDER: Optional[Qwen3TTSSpeakerEmbedder] = None
_SINGLETON_LOCK = threading.Lock()


def get_speaker_embedder(encoder_id: Optional[str] = None,
                         enc_dim: Optional[int] = None) -> Qwen3TTSSpeakerEmbedder:
    """Module-level singleton accessor for the frozen speaker embedder.

    ``enc_dim`` is only a fallback; the encoder is normally built from Base's
    ``config.json`` (upstream-identical). Pass ``None`` to fully defer to Base.
    """
    global _SPEAKER_EMBEDDER
    if _SPEAKER_EMBEDDER is None:
        with _SINGLETON_LOCK:
            if _SPEAKER_EMBEDDER is None:
                _SPEAKER_EMBEDDER = Qwen3TTSSpeakerEmbedder(encoder_id=encoder_id, enc_dim=enc_dim)
    return _SPEAKER_EMBEDDER


def load_reference_waveform(audio_path, sr=None):
    """Load a mono reference waveform ``(np.float32[T], sr)`` — dataloader side.

    Accepts two kinds of input:

    * a string path / ``wds://`` URI — decoded here via ``resolve_audio_reference``
      (only file IO; the resample/mel/encoder run later in ``forward``); or
    * an already-decoded 1-D waveform (``np.ndarray`` / ``torch.Tensor``) at
      sample rate ``sr``. Callers MUST pass ``sr`` in this case. This lets the
      S2S path reuse the waveform that ``super()._encode`` already decoded for
      the model audio encoder, avoiding a second ``librosa.load`` of the same
      bytes.

    Only the string path performs file IO; the resample/mel/encoder are always
    deferred to ``forward``.
    """
    # Already-decoded waveform: skip the redundant decode entirely.
    if isinstance(audio_path, (np.ndarray, torch.Tensor)):
        if sr is None:
            raise ValueError(
                'load_reference_waveform received a pre-decoded array but no '
                '`sr`; pass the native sample rate (e.g. 16000 for 16 kHz source '
                'audio) so forward can resample to the speaker encoder rate.')
        wav = audio_path.detach().cpu().float().numpy() if isinstance(audio_path, torch.Tensor) else audio_path
        return wav.astype(np.float32), int(sr)
    resolved = resolve_audio_reference(audio_path)
    if isinstance(resolved, (bytes, bytearray, memoryview)):
        audio_source = BytesIO(bytes(resolved))
    else:
        audio_source = _resolve_audio_path(resolved)
        if not os.path.exists(audio_source):
            raise FileNotFoundError(f'reference audio does not exist: {audio_path}')
    import librosa
    wav, sr = librosa.load(audio_source, sr=None, mono=True)
    return wav.astype(np.float32), int(sr)


__all__ = [
    'Qwen3TTSSpeakerEmbedder',
    'get_speaker_embedder',
    'load_reference_waveform',
    'DEFAULT_ENC_DIM',
    'DEFAULT_SAMPLE_RATE',
]

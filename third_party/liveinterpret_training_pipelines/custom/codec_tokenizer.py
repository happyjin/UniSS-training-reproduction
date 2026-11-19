# Copyright (c) ModelScope Contributors. All rights reserved.
"""Qwen3-TTS codec tokenizer helpers for bridge alignment training."""
import hashlib
import os
import uuid
import warnings
from functools import lru_cache
from typing import Any, Dict, Optional
from urllib.parse import urlparse

import numpy as np
import torch
import torch.nn.functional as F

# Silence benign librosa/audioread warnings:
# 1. PySoundFile fails on some formats -> falls back to audioread (UserWarning)
# 2. librosa's __audioread_load is deprecated (FutureWarning, librosa 0.10.0)
# Both are harmless — audioread is the correct fallback path.
warnings.filterwarnings('ignore', message='PySoundFile failed.*')
warnings.filterwarnings('ignore', category=FutureWarning, module='librosa')
warnings.filterwarnings('ignore', category=UserWarning, module='librosa')

DEFAULT_TOKENIZER_ID = 'Qwen3/Qwen3-TTS-Tokenizer-12Hz'

# Fallbacks only used when the live tokenizer does not expose its rates.
DEFAULT_INPUT_SAMPLE_RATE = 24000
DEFAULT_FRAME_MS = 80.0  # 1920-sample hop @ 24kHz == 12.5Hz codec
# Seconds of digital silence to encode when building the steady-state
# "silence codec" band used to fill read-write gaps on the target timeline.
SILENCE_PROBE_SECONDS = 1.5


def _default_cache_dir() -> str:
    """Return the codec cache directory.

    Resolution order:
        1. ``CONTINUE_PRETRAIN_CODEC_CACHE`` (continue-pretrain pipeline).
        2. ``BRIDGE_ALIGNMENT_CODEC_CACHE`` (bridge-alignment pipeline).
        3. ``${cwd}/.cache/qwen_omnispeak_codec`` fallback.
    """
    for env_var in ('CONTINUE_PRETRAIN_CODEC_CACHE', 'BRIDGE_ALIGNMENT_CODEC_CACHE'):
        value = os.environ.get(env_var)
        if value:
            return os.path.abspath(value)
    return os.path.abspath(os.path.join(os.getcwd(), '.cache', 'qwen_omnispeak_codec'))


def _default_device_map() -> str:
    device = os.environ.get('QWEN3_TTS_TOKENIZER_DEVICE')
    if device:
        return device
    if torch.cuda.is_available():
        local_rank = int(os.environ.get('LOCAL_RANK', '0'))
        return f'cuda:{local_rank}'
    return 'cpu'


def _repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir))


def _s3_to_local_candidates(audio_path: str):
    parsed = urlparse(audio_path)
    if parsed.scheme != 's3' or not parsed.netloc or not parsed.path:
        return []
    rel_path = os.path.join(parsed.netloc, parsed.path.lstrip('/'))
    roots = []
    for env_var in (
            'S3_AUDIO_LOCAL_ROOT',
            'AUDIO_LOCAL_ROOT',
            'BRIDGE_ALIGNMENT_AUDIO_ROOT',
            'CONTINUE_PRETRAIN_AUDIO_ROOT'):
        value = os.environ.get(env_var)
        if value:
            roots.append(os.path.abspath(value))
    datasets_root = os.path.join(_repo_root(), 'datasets')
    for subdir in ('text_to_speech', 'speech2speech_v2'):
        roots.append(os.path.join(datasets_root, subdir, 'audio_local'))
    if os.path.isdir(datasets_root):
        for name in sorted(os.listdir(datasets_root)):
            roots.append(os.path.join(datasets_root, name, 'audio_local'))
    seen = set()
    candidates = []
    for root in roots:
        if root in seen:
            continue
        seen.add(root)
        candidates.append(os.path.join(root, rel_path))
    return candidates


def _resolve_audio_path(audio_path: str) -> str:
    """Resolve an audio reference to a LOCAL filesystem path.

    ``wds://`` URIs resolve to their local tar SHARD path. This serves the
    existence checks / duration probing in dataset.py (``drop_row_if_invalid_audio``
    and ``sf.info``); note it deliberately does NOT serve *encoding* — feeding
    the shard path to the tokenizer made it encode the whole tar as one clip
    (the P1-3 bug). Encoding a wds member goes through
    ``Qwen3TTSCodecTokenizer.encode_codes``, which reads the exact byte range
    and fingerprints the full URI instead.
    """
    if os.path.exists(audio_path):
        return audio_path
    if audio_path.startswith('wds://'):
        try:
            from wds_audio import parse_wds_uri
            ref = parse_wds_uri(audio_path)
            if ref.shard_path.is_file():
                return str(ref.shard_path)
        except Exception:
            pass
    for candidate in _s3_to_local_candidates(audio_path):
        if os.path.exists(candidate):
            return candidate
    return audio_path


def _fingerprint(audio_path: str, tokenizer_id: str) -> str:
    stat = os.stat(audio_path)
    payload = '|'.join([
        os.path.abspath(audio_path),
        str(stat.st_size),
        str(int(stat.st_mtime_ns)),
        tokenizer_id,
    ])
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()


def _find_codes(obj: Any) -> torch.Tensor:
    if isinstance(obj, torch.Tensor):
        return obj
    if isinstance(obj, dict):
        for key in ('codes', 'code', 'audio_codes', 'audio_code', 'codec_codes', 'input_ids', 'tokens'):
            if key in obj:
                return _find_codes(obj[key])
        for value in obj.values():
            try:
                return _find_codes(value)
            except (TypeError, ValueError):
                continue
    if hasattr(obj, 'codes'):
        return _find_codes(getattr(obj, 'codes'))
    if hasattr(obj, 'audio_codes'):
        return _find_codes(getattr(obj, 'audio_codes'))
    if isinstance(obj, (list, tuple)):
        if len(obj) == 1:
            return _find_codes(obj[0])
        return torch.as_tensor(obj)
    raise ValueError(f'Cannot extract codec codes from tokenizer output type: {type(obj)!r}')


def _normalize_codes(codes: Any) -> torch.LongTensor:
    codes = _find_codes(codes).detach().cpu().long()
    while codes.dim() > 2 and 1 in codes.shape:
        codes = codes.squeeze(next(i for i, size in enumerate(codes.shape) if size == 1))
    if codes.dim() == 1:
        codes = codes[:, None]
    if codes.dim() != 2:
        raise ValueError(f'Expected 1D/2D codec codes, got shape={tuple(codes.shape)}')
    # Normalize to [T, C]. Qwen3-TTS tokenizer typically returns [C, T].
    if codes.shape[0] <= 32 and codes.shape[1] > codes.shape[0]:
        codes = codes.transpose(0, 1).contiguous()
    return codes


def _build_training_tensors(
    codes: torch.LongTensor,
    *,
    control_prefix_ids,
    codec_eos_id: int,
    num_residual_groups: int = 15,
    residual_vocab_size: int = 2048,
) -> Dict[str, torch.Tensor]:
    """Build talker training tensors with a leading control-token prefix.

    ``control_prefix_ids`` is the list of talker "mode-switch" codec ids that
    precede the audio, e.g.::

        [codec_think, codec_think_bos, language_id, codec_think_eos, codec_pad, codec_bos]   # with language
        [codec_nothink, codec_think_bos, codec_think_eos, codec_pad, codec_bos]              # auto / unknown

    The LAST id must be ``codec_bos``. Layout (k = len(prefix), T = #frames)::

        codec_input_ids : [ p0, p1, ..., p_{k-1}=BOS, m0, m1, ..., m_{T-1} ]      len k+T
        codec_labels    : [ -100 * (k-1)              , m0, m1, ..., m_{T-1}, EOS] len k+T

    i.e. the control prefix is never supervised; the BOS position predicts the
    first main code, and main-code teacher forcing proceeds as before. Residual
    labels are aligned frame-by-frame with ``codec_labels``.
    """
    if codes.numel() == 0:
        raise ValueError('Empty codec codes are not valid for training.')
    main = codes[:, 0].long()
    prefix = main.new_tensor(list(control_prefix_ids))
    k = int(prefix.shape[0])
    if k < 1:
        raise ValueError('control_prefix_ids must contain at least the codec_bos id.')
    codec_input_ids = torch.cat([prefix, main], dim=0)
    label_prefix = main.new_full((k - 1, ), -100)
    codec_labels = torch.cat([label_prefix, main, main.new_tensor([codec_eos_id])], dim=0)
    result = {
        'codec_input_ids': codec_input_ids,
        'codec_labels': codec_labels,
        'codec_attention_mask': torch.ones_like(codec_input_ids, dtype=torch.long),
    }
    if codes.shape[1] > 1 and num_residual_groups > 0:
        residual = codes[:, 1:1 + num_residual_groups].long()
        if residual.shape[1] < num_residual_groups:
            pad = residual.new_full((residual.shape[0], num_residual_groups - residual.shape[1]), -100)
            residual = torch.cat([residual, pad], dim=1)
        residual = residual.masked_fill((residual < 0) | (residual >= residual_vocab_size), -100)
        res_prefix = residual.new_full((k - 1, num_residual_groups), -100)
        res_eos = residual.new_full((1, num_residual_groups), -100)
        result['codec_residual_labels'] = torch.cat([res_prefix, residual, res_eos], dim=0)
    return result


class Qwen3TTSCodecTokenizer:
    """Lazy Qwen3-TTS tokenizer wrapper with atomic on-disk cache."""

    def __init__(
        self,
        tokenizer_id: Optional[str] = None,
        cache_dir: Optional[str] = None,
        device_map: Optional[str] = None,
    ) -> None:
        self.tokenizer_id = tokenizer_id or os.environ.get('QWEN3_TTS_TOKENIZER', DEFAULT_TOKENIZER_ID)
        self.cache_dir = os.path.abspath(cache_dir or _default_cache_dir())
        self.device_map = device_map or _default_device_map()
        self._tokenizer = None
        self._silence_band = None  # cached steady-state silence codec band [S, C]
        self._frame_ms_cache = None
        os.makedirs(self.cache_dir, exist_ok=True)

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            from qwen_tts import Qwen3TTSTokenizer  # pyright: ignore[reportMissingImports]
            # The auxiliary Qwen3-TTS-Tokenizer is *not* part of the trainable
            # model. When DeepSpeed Zero-3 is active, transformers'
            # ``from_pretrained`` would otherwise:
            #   * reject ``device_map`` outright, and
            #   * partition every weight via the global HfDeepSpeedConfig hook
            #     even when no zero.Init scope is open, leading to
            #     "weight should have at least three dimensions" at encode time.
            # We therefore temporarily detach the global HfDeepSpeedConfig
            # weakref while loading this tokenizer, and restore it afterwards.
            try:
                from transformers.integrations import deepspeed as _ds_int
            except Exception:
                _ds_int = None
            saved_ds_ref = None
            if _ds_int is not None and getattr(_ds_int, '_hf_deepspeed_config_weak_ref', None) is not None:
                saved_ds_ref = _ds_int._hf_deepspeed_config_weak_ref
                _ds_int._hf_deepspeed_config_weak_ref = None
            try:
                try:
                    from deepspeed.runtime.zero.partition_parameters import Init as _ZeroInit
                    _disable_zero_init = _ZeroInit(enabled=False)
                except Exception:
                    _disable_zero_init = None

                def _load(use_device_map: bool):
                    if use_device_map:
                        return Qwen3TTSTokenizer.from_pretrained(
                            self.tokenizer_id, device_map=self.device_map)
                    tk = Qwen3TTSTokenizer.from_pretrained(self.tokenizer_id)
                    if hasattr(tk, 'model') and hasattr(tk.model, 'to'):
                        try:
                            tk.model.to(self.device_map)
                        except Exception:
                            pass
                    return tk

                if _disable_zero_init is not None:
                    with _disable_zero_init:
                        try:
                            self._tokenizer = _load(use_device_map=True)
                        except ValueError as e:
                            if 'DeepSpeed Zero-3' not in str(e):
                                raise
                            self._tokenizer = _load(use_device_map=False)
                else:
                    try:
                        self._tokenizer = _load(use_device_map=True)
                    except ValueError as e:
                        if 'DeepSpeed Zero-3' not in str(e):
                            raise
                        self._tokenizer = _load(use_device_map=False)
            finally:
                if saved_ds_ref is not None and _ds_int is not None:
                    _ds_int._hf_deepspeed_config_weak_ref = saved_ds_ref
        return self._tokenizer

    def _encode_wav_tensor(self, wav: torch.Tensor, sr: int, target_sr: int):
        """Resample a ``[C, T]`` (or ``[T]``) waveform to ``target_sr`` and encode it."""
        if sr != target_sr:
            import torchaudio
            resampler = torchaudio.transforms.Resample(
                orig_freq=sr, new_freq=target_sr
            )
            wav = resampler(wav)
        # Mono: average channels if stereo
        if wav.dim() > 1 and wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        if wav.dim() > 1:
            wav = wav.squeeze(0)  # [T]
        # Newer ``qwen_tts.Qwen3TTSTokenizer._normalize_audio_inputs`` accepts
        # only ``str`` / ``np.ndarray`` (list); a ``torch.Tensor`` triggers a
        # TypeError ("Mixed input types are not supported..."). Convert here so
        # both old and new upstream versions work.
        wav_np = wav.detach().cpu().float().numpy()
        return self.tokenizer.encode(wav_np, sr=target_sr)

    def _encode_with_resample(self, audio_path: str, target_sr: int):
        """Encode audio, resampling to target_sr if needed.

        The Qwen3-TTS-Tokenizer expects 24kHz input. Audio files in the
        dataset may be 16kHz / 44.1kHz / etc. We detect the actual sample
        rate via soundfile and resample with torchaudio.transforms.Resample
        before feeding the waveform to the tokenizer.
        """
        import soundfile as sf
        try:
            info = sf.info(audio_path)
            source_sr = info.samplerate
        except Exception:
            source_sr = target_sr  # assume correct, let tokenizer handle

        if source_sr == target_sr:
            return self.tokenizer.encode(audio_path)

        # Resample: load -> resample -> encode from waveform
        import torchaudio
        wav, sr = torchaudio.load(audio_path)  # [C, T]
        return self._encode_wav_tensor(wav, sr, target_sr)

    def _encode_codes_wds(self, audio_path: str) -> torch.LongTensor:
        """Encode ONE indexed-WebDataset audio member (byte range in a tar).

        Two failure modes were fixed here (P1-3): resolving the URI to the
        shard PATH made the tokenizer try to encode the whole tar as a single
        clip, and the cache fingerprint was based on the shard file, so every
        member of a shard shared one cache entry (cross-sample pollution).
        The member bytes are now read through the validated byte range and
        the FULL URI (including offset/length) is the cache identity.
        """
        import io
        try:
            from wds_audio import parse_wds_uri, read_wds_audio_bytes
        except ImportError:  # pragma: no cover - package layout
            from .wds_audio import parse_wds_uri, read_wds_audio_bytes
        import soundfile as sf

        ref = parse_wds_uri(audio_path)  # validates offset/length/format
        stat = os.stat(ref.shard_path)
        key = hashlib.sha256('|'.join([
            audio_path, str(stat.st_size), str(int(stat.st_mtime_ns)),
            self.tokenizer_id,
        ]).encode('utf-8')).hexdigest()
        cache_path = os.path.join(self.cache_dir, f'{key}.pt')
        if os.path.exists(cache_path):
            cached = torch.load(cache_path, map_location='cpu')
            return _normalize_codes(cached['codes'] if isinstance(cached, dict) and 'codes' in cached else cached)

        payload = read_wds_audio_bytes(audio_path)
        try:
            wav_np, sr = sf.read(io.BytesIO(payload), dtype='float32', always_2d=True)
        except Exception as exc:
            raise ValueError(
                f'cannot decode WDS audio member {audio_path!r} '
                f'(format={ref.audio_format!r}): {exc}') from exc
        wav = torch.from_numpy(wav_np).T  # [C, T]
        codes = _normalize_codes(self._encode_wav_tensor(wav, int(sr), self._input_sample_rate()))

        tmp_path = os.path.join(self.cache_dir, f'.{key}.{os.getpid()}.{uuid.uuid4().hex}.tmp')
        torch.save({
            'audio_path': audio_path,
            'tokenizer_id': self.tokenizer_id,
            'codes': codes,
        }, tmp_path)
        os.replace(tmp_path, cache_path)
        return codes

    def encode_codes(self, audio_path: str) -> torch.LongTensor:
        if audio_path.startswith('wds://'):
            return self._encode_codes_wds(audio_path)
        resolved_audio_path = _resolve_audio_path(audio_path)
        if not os.path.exists(resolved_audio_path):
            candidates = _s3_to_local_candidates(audio_path)
            hint = f' Checked local mirrors: {candidates[:3]}' if candidates else ''
            raise FileNotFoundError(f'target_audio does not exist: {audio_path}.{hint}')
        key = _fingerprint(resolved_audio_path, self.tokenizer_id)
        cache_path = os.path.join(self.cache_dir, f'{key}.pt')
        if os.path.exists(cache_path):
            cached = torch.load(cache_path, map_location='cpu')
            return _normalize_codes(cached['codes'] if isinstance(cached, dict) and 'codes' in cached else cached)
        target_sr = self._input_sample_rate()
        encoded = self._encode_with_resample(resolved_audio_path, target_sr)
        codes = _normalize_codes(encoded)
        tmp_path = os.path.join(self.cache_dir, f'.{key}.{os.getpid()}.{uuid.uuid4().hex}.tmp')
        torch.save({
            'audio_path': os.path.abspath(resolved_audio_path),
            'source_audio_path': audio_path,
            'tokenizer_id': self.tokenizer_id,
            'codes': codes,
        }, tmp_path)
        os.replace(tmp_path, cache_path)
        return codes

    def encode_target_audio(
        self,
        audio_path: str,
        *,
        control_prefix_ids,
        codec_eos_id: int,
        num_residual_groups: int = 15,
        residual_vocab_size: int = 2048,
    ) -> Dict[str, torch.Tensor]:
        codes = self.encode_codes(audio_path)
        return _build_training_tensors(
            codes,
            control_prefix_ids=control_prefix_ids,
            codec_eos_id=codec_eos_id,
            num_residual_groups=num_residual_groups,
            residual_vocab_size=residual_vocab_size,
        )

    def frame_ms(self) -> float:
        """Codec frame period in milliseconds, self-calibrated from the live
        tokenizer (``encode_downsample_rate / input_sample_rate``).

        This avoids hard-coding a frame rate (the docs variously say 12Hz / 13
        frames-per-second / 480x, which are mutually inconsistent). The value is
        read straight from the model that actually produced the codes, so the
        ms<->frame mapping used to lay codes on the real timeline is exact.
        """
        if self._frame_ms_cache is not None:
            return self._frame_ms_cache
        value = None
        try:
            ds = float(self.tokenizer.get_encode_downsample_rate())
            sr = float(self.tokenizer.get_input_sample_rate())
            if ds > 0 and sr > 0:
                value = 1000.0 * ds / sr
        except Exception:
            value = None
        if value is None or value <= 0:
            env = os.environ.get('QWEN3_TTS_FRAME_MS')
            try:
                value = float(env) if env else DEFAULT_FRAME_MS
            except ValueError:
                value = DEFAULT_FRAME_MS
        self._frame_ms_cache = float(value)
        return self._frame_ms_cache

    def _input_sample_rate(self) -> int:
        try:
            sr = int(self.tokenizer.get_input_sample_rate())
            if sr > 0:
                return sr
        except Exception:
            pass
        try:
            return int(os.environ.get('QWEN3_TTS_INPUT_SR', '') or DEFAULT_INPUT_SAMPLE_RATE)
        except ValueError:
            return DEFAULT_INPUT_SAMPLE_RATE

    def _steady_silence_band(self) -> torch.LongTensor:
        """Encode a short digital-silence waveform once and cache the
        steady-state codec frames (onset/offset transients trimmed).

        Tiling this band reconstructs genuine acoustic silence at decode time,
        which is what we want the model to emit while it is *listening but not
        yet translating* (the "write = wait" half of the read-write policy).
        """
        if self._silence_band is not None:
            return self._silence_band
        sr = self._input_sample_rate()
        try:
            seconds = float(os.environ.get('QWEN3_TTS_SILENCE_SECONDS', '') or SILENCE_PROBE_SECONDS)
        except ValueError:
            seconds = SILENCE_PROBE_SECONDS
        n_samples = max(1, int(round(sr * seconds)))
        wav = np.zeros(n_samples, dtype=np.float32)
        codes = _normalize_codes(self.tokenizer.encode(wav, sr=sr))  # [T, C]
        if codes.numel() == 0:
            raise ValueError('Failed to encode a silence probe waveform.')
        total = int(codes.shape[0])
        # Drop a few leading/trailing frames (codec onset/offset transients) and
        # keep the stationary middle band; silence is stationary so any slice of
        # the steady region can be safely tiled to an arbitrary length.
        trim = min(4, max(0, (total - 1) // 2))
        band = codes[trim:total - trim] if total - 2 * trim >= 1 else codes
        self._silence_band = band.contiguous()
        return self._silence_band

    def _silence_frames(self, num_frames: int, target_channels: int) -> torch.LongTensor:
        """Return ``[num_frames, target_channels]`` of codec-silence frames."""
        if num_frames <= 0:
            band = self._steady_silence_band()
            return band.new_zeros((0, target_channels))
        band = self._steady_silence_band()
        reps = (num_frames + band.shape[0] - 1) // band.shape[0]
        tiled = band.repeat(reps, 1)[:num_frames].clone()
        c = int(tiled.shape[1])
        if c < target_channels:
            pad = tiled.new_full((num_frames, target_channels - c), -1)
            tiled = torch.cat([tiled, pad], dim=1)
        elif c > target_channels:
            tiled = tiled[:, :target_channels]
        return tiled.contiguous()

    def encode_target_audios_concat(
        self,
        audio_paths,
        *,
        control_prefix_ids,
        codec_eos_id: int,
        num_residual_groups: int = 15,
        residual_vocab_size: int = 2048,
        chunk_start_ms=None,
        chunk_end_ms=None,
    ) -> Dict[str, Any]:
        """Encode per-chunk target audios and lay them on the *real timeline* to
        build a single simultaneous-interpretation talker stream.

        Unlike a naive seamless concatenation (which collapses the task into
        offline TTS), each chunk's codec frames are placed at the frame index
        implied by its ``target_start_ms`` timestamp, and every inter-chunk gap
        (including the leading ``[0, target_start_ms[0]]`` listen-only window) is
        filled with **silence codec frames**. Those silence frames are fully CE
        supervised, so the model is explicitly taught *when to stay quiet and
        keep listening* and *when to start speaking* -- the write half of the
        read-write policy. The read half is attached separately as a per-frame
        ``codec_to_sem_visible`` schedule (see ``_build_streaming_schedule``).

        Timeline construction (``frame_ms`` self-calibrated from the tokenizer):

            frame 0 ........ start[0] ........ start[0]+T0 ... start[1] .........
            [ silence(listen) ][   chunk_0 codec   ][ silence ][  chunk_1 codec  ]

        where ``start[i] = round(target_start_ms[i] / frame_ms)`` and chunk ``i``
        occupies its own ``T_i`` real codec frames. Overlaps (from rounding) are
        avoided by never letting a chunk start before the previous one ends.

        When timestamps are unavailable / mismatched, it gracefully falls back to
        plain time-contiguous concatenation (no silence), preserving the old
        behaviour.

        Returns the usual ``codec_input_ids/codec_labels/codec_residual_labels/
        codec_attention_mask`` plus:

        * ``frame_ms``: self-calibrated codec frame period (ms).
        * ``num_main_frames``: total number of main frames laid on the timeline
          (silence + speech), i.e. ``T`` in the ``k + T`` codec layout.
        * ``control_prefix_len`` (``k``): number of leading control tokens.
        """
        paths = list(audio_paths or [])
        if not paths:
            raise ValueError('encode_target_audios_concat requires at least one audio path.')
        per_chunk_codes = []
        for path in paths:
            codes = self.encode_codes(path)  # [T_i, C]
            if codes.numel() == 0:
                raise ValueError(f'Empty codec codes for target audio chunk: {path}')
            per_chunk_codes.append(codes)
        # Align channel (codebook-group) dims across chunks. Chunks from the same
        # tokenizer share C, so this is just a safety net.
        max_c = max(c.shape[1] for c in per_chunk_codes)

        def _align_c(codes: torch.LongTensor) -> torch.LongTensor:
            if codes.shape[1] < max_c:
                pad = codes.new_full((codes.shape[0], max_c - codes.shape[1]), -1)
                codes = torch.cat([codes, pad], dim=1)
            return codes

        frame_ms = self.frame_ms()
        starts = list(chunk_start_ms) if chunk_start_ms is not None else None
        use_timeline = (
            starts is not None
            and len(starts) == len(per_chunk_codes)
            and all(s is not None for s in starts)
            and frame_ms > 0
        )

        if use_timeline:
            blocks = []
            cursor = 0  # next free frame index on the timeline
            for i, codes in enumerate(per_chunk_codes):
                codes = _align_c(codes)
                start_frame = int(round(float(starts[i]) / frame_ms))
                # Trust timestamp ordering but never overlap previous speech.
                if start_frame < cursor:
                    start_frame = cursor
                gap = start_frame - cursor
                if gap > 0:
                    blocks.append(self._silence_frames(gap, max_c))  # listen / wait
                blocks.append(codes)  # speak chunk i
                cursor = start_frame + int(codes.shape[0])
            concat_codes = torch.cat(blocks, dim=0)
        else:
            concat_codes = torch.cat([_align_c(c) for c in per_chunk_codes], dim=0)

        tensors = _build_training_tensors(
            concat_codes,
            control_prefix_ids=control_prefix_ids,
            codec_eos_id=codec_eos_id,
            num_residual_groups=num_residual_groups,
            residual_vocab_size=residual_vocab_size,
        )
        tensors['frame_ms'] = float(frame_ms)
        tensors['control_prefix_len'] = int(len(list(control_prefix_ids)))
        tensors['num_main_frames'] = int(concat_codes.shape[0])
        return tensors


@lru_cache(maxsize=4)
def get_codec_tokenizer(tokenizer_id: Optional[str] = None, cache_dir: Optional[str] = None,
                        device_map: Optional[str] = None) -> Qwen3TTSCodecTokenizer:
    return Qwen3TTSCodecTokenizer(tokenizer_id=tokenizer_id, cache_dir=cache_dir, device_map=device_map)


def pad_1d(sequences, padding_value: int, *, dtype=torch.long) -> torch.Tensor:
    tensors = [torch.as_tensor(seq, dtype=dtype) for seq in sequences]
    return torch.nn.utils.rnn.pad_sequence(tensors, batch_first=True, padding_value=padding_value)


def pad_2d_time(sequences, padding_value: int, *, dtype=torch.long) -> torch.Tensor:
    tensors = [torch.as_tensor(seq, dtype=dtype) for seq in sequences]
    max_t = max(t.shape[0] for t in tensors)
    max_g = max(t.shape[1] for t in tensors)
    padded = []
    for tensor in tensors:
        pad_t = max_t - tensor.shape[0]
        pad_g = max_g - tensor.shape[1]
        padded.append(F.pad(tensor, (0, pad_g, 0, pad_t), value=padding_value))
    return torch.stack(padded, dim=0)

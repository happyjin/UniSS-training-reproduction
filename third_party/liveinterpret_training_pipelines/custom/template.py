# Copyright (c) ModelScope Contributors. All rights reserved.
"""Template extensions for Qwen-OmniSpeak training pipelines.

Two template overrides are provided:

* ``QwenOmniSpeakBridgeAlignmentTemplate`` — Stage 1 bridge alignment.
  Every sample must carry ``target_audio`` or precomputed
  ``target_codec_codes`` and the supervision is purely on the codec route.
* ``QwenOmniSpeakContinuePretrainTemplate`` — continue-pretrain template
  used by ``liveinterpret_training_pipelines/pipelines/continue_pretrain``.
  Mixes text-only samples (``speech_to_text_translation`` / ``multi_speaker
  _segmentation``) with codec-supervised samples
  (``speech_to_speech_translation``) inside the same batch.
"""
from typing import Any, Dict, List, Optional

import os

import numpy as np
import torch
import torch.nn.functional as F

from swift.template import Template, TemplateMeta, register_template
from swift.utils import get_logger

try:
    from .codec_tokenizer import (
        _build_training_tensors,
        get_codec_tokenizer,
        pad_1d,
        pad_2d_time,
        _resolve_audio_path,
    )
    from .speaker_encoder import load_reference_waveform
    from .wds_audio import (
        is_lance_uri,
        resolve_audio_reference,
        resolve_audio_references,
        resolve_lance_codec_codes,
        resolve_lance_source_features,
        resolve_lance_speaker_embedding,
    )
    from .dataset import (
        EXTRA_KEYS,
        TASK_S2S_TRANSLATION,
        TASK_SIMUL_POLICY,
    )
except ImportError:  # pragma: no cover - external_plugins imports this as a top-level module.
    # ``qwen_omnispeak/register_qwen_omnispeak.py`` puts its own directory at
    # sys.path[0].  It also contains files named ``speaker_encoder.py`` and
    # ``codec_tokenizer.py``; bare fallback imports would therefore select the
    # model/export copies instead of these data-pipeline siblings.  Import the
    # namespace package explicitly so WDS byte support cannot depend on path
    # ordering.
    import sys as _sys
    from pathlib import Path as _Path

    _repo_root = str(_Path(__file__).resolve().parents[1])
    if _repo_root not in _sys.path:
        _sys.path.append(_repo_root)
    from custom.codec_tokenizer import (
        _build_training_tensors,
        get_codec_tokenizer,
        pad_1d,
        pad_2d_time,
        _resolve_audio_path,
    )
    from custom.speaker_encoder import load_reference_waveform
    from custom.wds_audio import (
        is_lance_uri,
        resolve_audio_reference,
        resolve_audio_references,
        resolve_lance_codec_codes,
        resolve_lance_source_features,
        resolve_lance_speaker_embedding,
    )
    from custom.dataset import (
        EXTRA_KEYS,
        TASK_S2S_TRANSLATION,
        TASK_SIMUL_POLICY,
    )

try:
    from qwen_omnispeak.register_qwen_omnispeak import QWEN_OMNISPEAK_TEMPLATE, QwenOmniSpeakTemplate
except ImportError:  # pragma: no cover
    from register_qwen_omnispeak import (  # pyright: ignore[reportMissingImports]
        QWEN_OMNISPEAK_TEMPLATE, QwenOmniSpeakTemplate)

logger = get_logger()

QWEN_OMNISPEAK_CONTINUE_PRETRAIN_TEMPLATE = 'qwen_omnispeak_continue_pretrain'
QWEN_OMNISPEAK_SIMULTANEOUS_POLICY_TEMPLATE = 'qwen_omnispeak_simultaneous_policy'
QWEN_OMNISPEAK_NORMAL_TRANSLATION_TEMPLATE = 'qwen_omnispeak_normal_translation'
QWEN_OMNISPEAK_BILINGUAL_TRANSLATION_TEMPLATE = 'qwen_omnispeak_bilingual_translation'
QWEN_OMNISPEAK_MULTI_SPEAKER_TRANSLATION_TEMPLATE = 'qwen_omnispeak_multi_speaker_translation'
QWEN_OMNISPEAK_JOINT_STAGE1_TEMPLATE = 'qwen_omnispeak_joint_stage1'

# 源音频原生采样率（数据集约定 16 kHz）。S2S 说话人参考复用 super()._encode
# 已解码的波形时，必须如实上报该 sr，forward 内部会把它 resample 到说话人
# 编码器的 24 kHz。
_SOURCE_AUDIO_SR = 16000

_DROP_EXTRA_KEYS = (set(EXTRA_KEYS) | {
    'target_codec_codes',
    'codec_codes',
    'bridge_task',
}) - {
    # ICL tensors are batched in _data_collator and must reach model.forward().
    'source_codec_codes',
    'source_text_ids',
}


def _codec_config_from(template_self) -> Dict[str, int]:
    config = template_self.config
    talker_config = getattr(config, 'talker_config', None)
    return {
        'codec_bos_id': int(getattr(config, 'codec_bos_id', 2149)),
        'codec_eos_id': int(getattr(config, 'codec_eos_id', 2150)),
        'codec_pad_id': int(getattr(config, 'codec_pad_id', 2148)),
        'codec_think_id': int(getattr(config, 'codec_think_id', 2154)),
        'codec_nothink_id': int(getattr(config, 'codec_nothink_id', 2155)),
        'codec_think_bos_id': int(getattr(config, 'codec_think_bos_id', 2156)),
        'codec_think_eos_id': int(getattr(config, 'codec_think_eos_id', 2157)),
        'codec_language_id': dict(getattr(config, 'codec_language_id', None) or {}),
        'num_residual_groups': int(getattr(talker_config, 'num_code_groups', 15)),
        'residual_vocab_size': int(getattr(talker_config, 'code_predictor_vocab_size', 2048)),
        # Upstream talker text-side special tokens (only consumed on the
        # feature/talker-upstream-omniinput branch when
        # ``config.use_upstream_talker_input=True``). Defaults mirror Qwen3-TTS
        # -12Hz-1.7B-Base ``tokenizer_config.json`` and are safely ignored by
        # every existing template on other branches.
        'tts_pad_token_id': int(getattr(talker_config, 'tts_pad_token_id', 151_671)),
        'tts_bos_token_id': int(getattr(talker_config, 'tts_bos_token_id', 151_672)),
        'tts_eos_token_id': int(getattr(talker_config, 'tts_eos_token_id', 151_673)),
        'use_upstream_talker_input': bool(getattr(config, 'use_upstream_talker_input', False)),
        # Qwen2.5-Omni full-add layout (feature/talker-full-add): the codec
        # control prefix becomes ``[codec_mask × T_prompt (+spk) , codec_pad ,
        # codec_bos]`` so it aligns position-wise with the thinker prompt.
        'codec_mask_id': int(getattr(config, 'codec_mask_id', getattr(config, 'codec_pad_id', 2148))),
        'full_add_layout': _full_add_layout(config),
    }


# Normalise common language spellings / ISO codes to the keys used by the
# talker's ``codec_language_id`` map.
_LANG_ALIASES = {
    'en': 'english', 'eng': 'english', 'english': 'english',
    'zh': 'chinese', 'cn': 'chinese', 'chinese': 'chinese', 'mandarin': 'chinese',
    'de': 'german', 'german': 'german',
    'it': 'italian', 'italian': 'italian',
    'pt': 'portuguese', 'portuguese': 'portuguese',
    'es': 'spanish', 'spanish': 'spanish',
    'ja': 'japanese', 'jp': 'japanese', 'japanese': 'japanese',
    'ko': 'korean', 'korean': 'korean',
    'fr': 'french', 'french': 'french',
    'ru': 'russian', 'russian': 'russian',
}


def _resolve_language_id(target_lang, codec_language_id: Dict[str, int]):
    if not target_lang or not codec_language_id:
        return None
    raw = str(target_lang).strip().lower()
    key = _LANG_ALIASES.get(raw, raw)
    return codec_language_id.get(key)


def _use_speaker_prompt(config) -> bool:
    """Whether to inject a reference-audio speaker x-vector (voice clone).

    Priority: env ``QWEN_OMNISPEAK_USE_SPEAKER_PROMPT`` > config flag. Off by
    default so existing runs are unaffected.
    """
    env = os.environ.get('QWEN_OMNISPEAK_USE_SPEAKER_PROMPT')
    if env is not None:
        return env.lower() in {'1', 'true', 'yes', 'y', 'on'}
    return bool(getattr(config, 'use_speaker_prompt', False))


def _nonempty_vector(value) -> bool:
    """True when ``value`` is a usable embedding / codes payload.

    Numpy / tensor must not go through ``value == ''`` or ``value != []`` —
    those are elementwise and raise on arrays. Empty-string is handled in
    the ``str`` branch only.
    """
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value)
    if isinstance(value, (list, tuple)):
        return len(value) > 0
    if isinstance(value, np.ndarray):
        return value.size > 0
    try:
        if isinstance(value, torch.Tensor):
            return value.numel() > 0
    except Exception:
        pass
    return True


def _build_precomputed_codec_fields(raw_codes, cfg: Dict[str, int], control_prefix):
    """Build training labels from explicitly stored Qwen3 [T, 16] codes."""
    codes = torch.as_tensor(raw_codes, dtype=torch.long)
    expected_groups = cfg['num_residual_groups'] + 1
    if codes.ndim != 2:
        raise ValueError(
            f'target_codec_codes must be 2-D [T, C], got shape={tuple(codes.shape)}')
    if codes.shape[1] == expected_groups:
        pass
    elif codes.shape[0] == expected_groups:
        codes = codes.transpose(0, 1).contiguous()
    else:
        raise ValueError(
            f'target_codec_codes must have {expected_groups} codebooks '
            f'(1 main + {cfg["num_residual_groups"]} residual), got shape={tuple(codes.shape)}')
    return _build_training_tensors(
        codes,
        control_prefix_ids=control_prefix,
        codec_eos_id=cfg['codec_eos_id'],
        num_residual_groups=cfg['num_residual_groups'],
        residual_vocab_size=cfg['residual_vocab_size'],
    )


def _full_add_layout(config) -> bool:
    """Whether to use the Qwen2.5-Omni "full-add" talker input layout.

    Priority: env ``QWEN_OMNISPEAK_FULL_ADD_LAYOUT`` > config flag. Only
    meaningful when ``use_upstream_talker_input`` is also on. Off by default so
    the reply-add layout stays the default on this branch.
    """
    env = os.environ.get('QWEN_OMNISPEAK_FULL_ADD_LAYOUT')
    if env is not None:
        return env.lower() in {'1', 'true', 'yes', 'y', 'on'}
    return bool(getattr(config, 'full_add_layout', False))


def _first_reply_index(labels) -> int:
    """Index of the first supervised (reply) position in a 1-D label list.

    ``labels`` is the thinker text-LM label list (``-100`` on prompt / audio /
    chatml-scaffold positions, real token ids on the assistant reply span).
    Returns ``len(labels)`` when there is no reply position (all ``-100``).
    """
    if not labels:
        return 0
    for i, v in enumerate(labels):
        if int(v) != -100:
            return i
    return len(labels)


def _build_full_add_control_prefix(cfg: Dict[str, Any], t_prompt: int,
                                  target_lang=None, with_speaker: bool = False):
    """Qwen2.5-Omni full-add codec control prefix.

    Layout (position-aligned with the thinker prompt + text_bos):
        ``[ codec_mask × t_prompt , think/lang control tokens , (speaker slot) ,
           codec_pad , codec_bos ]``

    The ``codec_mask`` run mirrors ``Qwen2_5OmniTalker``'s
    ``full_like(input_ids, codec_mask_token)`` prompt prefix; the think/lang
    control tokens (identical to the reply-add / bridge-concat prefix) follow
    so the talker sees the same mode-switch + language conditioning regardless
    of layout. ``codec_pad`` aligns with the ``text_bos`` frame and ``codec_bos``
    with the first reply token. Every prefix position is unsupervised
    (label ``-100``).

    Returns ``(prefix_ids, speaker_slot_index)`` (slot is ``None`` when
    ``with_speaker`` is False).
    """
    prefix = [cfg['codec_mask_id']] * int(max(0, t_prompt))
    # Think/lang control tokens (same as _build_control_prefix).
    lang_id = _resolve_language_id(target_lang, cfg['codec_language_id'])
    if lang_id is not None:
        prefix += [cfg['codec_think_id'], cfg['codec_think_bos_id'],
                   int(lang_id), cfg['codec_think_eos_id']]
    else:
        prefix += [cfg['codec_nothink_id'], cfg['codec_think_bos_id'],
                   cfg['codec_think_eos_id']]
    speaker_slot_index = None
    if with_speaker:
        speaker_slot_index = len(prefix)  # right after codec_think_eos
        prefix = prefix + [cfg['codec_pad_id']]  # placeholder, replaced in forward
    prefix += [cfg['codec_pad_id'], cfg['codec_bos_id']]
    return prefix, speaker_slot_index


def _build_control_prefix(cfg: Dict[str, Any], target_lang, with_speaker: bool = False):
    """Build the talker control-token prefix that precedes the codec frames.

    With a recognised target language we use the "think + language" prefix;
    otherwise we fall back to the "nothink" prefix. The prefix always ends with
    ``codec_pad`` + ``codec_bos``. These tokens are never supervised (label =
    -100).

    When ``with_speaker`` is set we reserve one extra "speaker slot" right after
    ``codec_think_eos`` (mirroring the upstream Qwen3-TTS speaker-conditioning
    position). It is filled with a ``codec_pad`` placeholder id here and its
    embedding is replaced by the reference x-vector inside ``forward``.

    Returns ``(prefix_ids, speaker_slot_index)`` where ``speaker_slot_index`` is
    ``None`` when ``with_speaker`` is False.
    """
    lang_id = _resolve_language_id(target_lang, cfg['codec_language_id'])
    if lang_id is not None:
        prefix = [cfg['codec_think_id'], cfg['codec_think_bos_id'], int(lang_id), cfg['codec_think_eos_id']]
    else:
        prefix = [cfg['codec_nothink_id'], cfg['codec_think_bos_id'], cfg['codec_think_eos_id']]
    speaker_slot_index = None
    if with_speaker:
        speaker_slot_index = len(prefix)  # position right after codec_think_eos
        prefix = prefix + [cfg['codec_pad_id']]  # placeholder, replaced in forward
    prefix += [cfg['codec_pad_id'], cfg['codec_bos_id']]
    return prefix, speaker_slot_index


def _build_codec_control_prefix(
    cfg: Dict[str, Any],
    *,
    labels=None,
    target_lang=None,
    with_speaker: bool = False,
):
    """Build the codec prefix shared by initial encode and Megatron recovery."""
    full_add = (
        cfg.get('full_add_layout', False)
        and cfg.get('use_upstream_talker_input', False)
    )
    if full_add:
        return _build_full_add_control_prefix(
            cfg,
            _first_reply_index(labels),
            target_lang=target_lang,
            with_speaker=with_speaker,
        )
    return _build_control_prefix(
        cfg, target_lang, with_speaker=with_speaker)


def _resolve_speaker_audio_path(inputs, raw_audios=None) -> Optional[str]:
    """Resolve the audio file used to extract the speaker x-vector.

    Unified rule across all branches (matches the design in
    ``docs/ideas/branch_comparison.md``):
      * TTS task             -> ``reference_audios[0]`` (a separate reference
        clip of the intended target voice)
      * S2S translation task -> the *source* audio (so the translated speech
        preserves the source speaker's voice)

    ``raw_audios`` is the source-audio path list captured *before*
    ``super()._encode`` loaded them into waveform arrays; it is the reliable
    source of the s2s path (``inputs.audios`` holds numpy arrays post-load, and
    the raw ``audios`` column is consumed by swift as a media field, so it is no
    longer a plain path in ``extra_kwargs``).

    Falls back gracefully: if the task-preferred field is missing, try the
    other one before returning ``None``. Always returns a path *string* (never
    a loaded array), so callers can safely ``load_reference_waveform`` it.
    """
    ek = getattr(inputs, 'extra_kwargs', {}) or {}
    task = str(ek.get('task') or ek.get('bridge_task') or '').lower()

    def _first_path(val):
        if val is None:
            return None
        if isinstance(val, (list, tuple)):
            if len(val) == 0:
                return None
            item = val[0]
        else:
            item = val
        if isinstance(item, dict):
            p = item.get('path')
            return p if isinstance(p, str) else None
        # Only plain path strings are valid; loaded waveforms (np.ndarray /
        # torch tensors) are NOT paths and must be ignored (they would crash a
        # boolean test and cannot be re-loaded from disk).
        return item if isinstance(item, str) else None

    ref_path = _first_path(ek.get('reference_audios'))
    # s2s source path: prefer the pre-load raw path, then extra_kwargs, then the
    # (possibly already-loaded) inputs.audios (kept last; _first_path guards it).
    src_path = (_first_path(raw_audios)
                or _first_path(ek.get('audios'))
                or _first_path(getattr(inputs, 'audios', None)))

    is_s2s = 's2s' in task or 'speech_to_speech' in task or 'speech2speech' in task
    if is_s2s:
        return src_path or ref_path
    # TTS (or unknown): prefer the reference clip.
    return ref_path or src_path


def _collect(batch: List[Dict[str, Any]], key: str):
    values = []
    for row in batch:
        value = row.pop(key, None)
        if value is not None:
            values.append(value)
    return values


# ---------------------------------------------------------------------------
# Simultaneous read-write schedule helpers (Stage 3 SFT).
# ---------------------------------------------------------------------------


def _as_bool_env(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in {'1', 'true', 'yes', 'y', 'on'}


def _resolve_audio_token_id(config) -> Optional[int]:
    """Find the source-audio placeholder token id from a (possibly nested) config.

    Qwen3-Omni exposes ``audio_token_id`` while Qwen2.5-Omni uses
    ``audio_token_index``; either may live on the model config or one/two levels
    deeper under ``thinker_config``.
    """
    nodes = []
    node = config
    for _ in range(3):
        if node is None:
            break
        nodes.append(node)
        node = getattr(node, 'thinker_config', None)
    for cfg in nodes:
        for attr in ('audio_token_id', 'audio_token_index'):
            value = getattr(cfg, attr, None)
            if value is not None:
                try:
                    return int(value)
                except (TypeError, ValueError):
                    continue
    return None


def _source_duration_ms(inputs, alignment, sampling_rate_default: int = 16000,
                        env_prefix: str = 'SIMULTANEOUS_POLICY') -> Optional[float]:
    """Best-effort source-audio duration in ms.

    Preference order:
      1. Read the actual source wav via ``soundfile`` (covers trailing silence).
      2. Fall back to the last chunk's ``source_end_ms`` from ``alignment``.

    ``env_prefix`` selects the ``<PREFIX>_SOURCE_DURATION_FROM_FILE`` switch so a
    subclassed pipeline (e.g. normal_translation) can toggle it independently.
    """
    if _as_bool_env(f'{env_prefix}_SOURCE_DURATION_FROM_FILE', True):
        audios = getattr(inputs, 'audios', None)
        if not audios:
            audios = inputs.extra_kwargs.get('audios') if hasattr(inputs, 'extra_kwargs') else None
        if audios:
            first = audios[0]
            path = first.get('path') if isinstance(first, dict) else first
            if isinstance(path, str) and path:
                try:
                    import soundfile as sf  # pyright: ignore[reportMissingImports]
                    info = sf.info(_resolve_audio_path(path))
                    if info.frames > 0 and info.samplerate > 0:
                        return 1000.0 * info.frames / info.samplerate
                except Exception:
                    pass
    src_ends = [seg.get('source_end_ms') for seg in (alignment or []) if seg.get('source_end_ms') is not None]
    if src_ends:
        return float(max(src_ends))
    return None


def _build_streaming_schedule(
    *,
    input_ids,
    audio_token_id: int,
    control_prefix_len: int,
    num_main_frames: int,
    frame_ms: float,
    source_duration_ms: Optional[float],
    extra_lag_ms: float = 0.0,
) -> Optional[torch.Tensor]:
    """Derive a monotonic non-decreasing ``codec_to_sem_visible`` schedule on the
    **real timeline**.

    The talker codec stream is laid frame-by-frame on the wall clock (see
    :meth:`Qwen3TTSCodecTokenizer.encode_target_audios_concat`): main frame ``j``
    is emitted at real time ``j * frame_ms``. Simultaneous interpretation is
    full-duplex, so the source audio that has *physically arrived* by that
    instant is ``source[0 : j * frame_ms]``. We therefore let frame ``j`` attend
    to the leading semantic prefix ``a0 + arrived(j * frame_ms)``, where:

      * ``a0`` is the index of the first source-audio token (so the task prompt
        before the audio is always visible), and
      * ``arrived(t) = round(m * t / source_duration_ms)`` is how many of the
        ``m`` source-audio tokens have streamed in by time ``t``.

    This single per-frame rule jointly teaches *read* (more source becomes
    visible as time advances) and *write* (the silence frames in the leading /
    inter-chunk gaps can only see the little source that has arrived, so the
    model learns to wait before speaking). ``extra_lag_ms`` shifts the clock for
    a global wait-k style latency knob.

    Returns ``None`` when the schedule cannot be built (no audio tokens / unknown
    duration), which makes the caller fall back to offline full-context attention.
    """
    if audio_token_id is None or input_ids is None:
        return None
    ids = torch.as_tensor(input_ids).flatten()
    audio_pos = (ids == audio_token_id).nonzero(as_tuple=False).flatten()
    if audio_pos.numel() == 0:
        return None
    a0 = int(audio_pos[0].item())
    m = int(audio_pos.numel())  # number of source-audio tokens
    sem_len = int(ids.numel())

    if source_duration_ms is None or source_duration_ms <= 0:
        return None
    if frame_ms is None or frame_ms <= 0:
        return None

    k = int(control_prefix_len)
    total_main = int(num_main_frames)
    codec_len = k + total_main
    if codec_len <= 0:
        return None

    # Real time of each main frame, then the arrived source-token count.
    frame_idx = torch.arange(total_main, dtype=torch.float64)
    t_ms = frame_idx * float(frame_ms) + float(extra_lag_ms)
    frac = torch.clamp(t_ms / float(source_duration_ms), min=0.0, max=1.0)
    arrived = torch.round(frac * float(m)).long().clamp_(0, m)
    # Cap the visible prefix at the end of the source-audio span; assistant
    # target text (which comes afterwards) is never visible while streaming.
    vis_main = torch.clamp(a0 + arrived, max=min(a0 + m, sem_len))

    vis = torch.empty(codec_len, dtype=torch.long)
    # Control-prefix positions precede the first main frame -> share its time.
    head_vis = int(vis_main[0].item()) if total_main > 0 else min(a0, sem_len)
    if k > 0:
        vis[:k] = head_vis
    if total_main > 0:
        vis[k:] = vis_main
    # Enforce monotonic non-decreasing visibility (interface requirement).
    vis, _ = torch.cummax(vis, dim=0)
    return vis


# ---------------------------------------------------------------------------
# Bridge alignment template (Stage 1) — kept unchanged.
# ---------------------------------------------------------------------------

# Whisper audio-tower hop/nfft. Token expansion uses Lance ``frames``, not
# ``silence * hop``. Placeholders stay length-1; these stay as the hop contract.
_MEL_HOP = 160
_MEL_NFFT = 400


def _to_float_dtype(data, dtype):
    try:
        from swift.utils import to_float_dtype
        return to_float_dtype(data, dtype)
    except Exception:
        if isinstance(data, dict):
            return {
                k: (v.to(dtype=dtype) if isinstance(v, torch.Tensor) and v.is_floating_point() else v)
                for k, v in data.items()
            }
        if isinstance(data, torch.Tensor) and data.is_floating_point():
            return data.to(dtype=dtype)
        return data


def _findall_tokens(token_list, token_id):
    try:
        from swift.utils import findall
        return findall(token_list, token_id)
    except Exception:
        # Minimal fallback matching ms-swift findall for int / list[int] needles.
        if isinstance(token_id, list):
            n = len(token_id)
            return [i for i in range(len(token_list) - n + 1) if token_list[i:i + n] == token_id]
        return [i for i, t in enumerate(token_list) if t == token_id]


def _normalize_precomputed_mel(value) -> Optional[List[dict]]:
    """Return list of mel dicts, or None if this sample is not mel-mode.

    Accepts a ``lance://…/features`` URI, a packed dict from
    ``resolve_lance_source_features``, or a list of either.
    """
    if value is None or value == '' or value == []:
        return None
    if is_lance_uri(value):
        value = resolve_lance_source_features(value)
    if isinstance(value, dict) and 'flat' in value and 'frames' in value:
        return [value]
    if isinstance(value, list) and value:
        out = []
        for item in value:
            if is_lance_uri(item):
                item = resolve_lance_source_features(item)
            if not (isinstance(item, dict) and 'flat' in item and 'frames' in item):
                return None
            out.append(item)
        return out
    return None


def _build_precomputed_source_mel_fields(mel_items: List[dict], torch_dtype) -> Dict[str, Any]:
    """Build ``input_features`` / ``feature_attention_mask`` from packed mel dicts.

    Symmetric to ``_build_precomputed_codec_fields``: consumes preprocess-resolved
    (or encode-time-resolved) Lance mel columns and never runs Whisper.
    """
    feats = []
    masks = []
    frame_lens = []
    for m in mel_items:
        fs = int(m.get('feature_size', 128))
        t = int(m['frames'])
        feat = torch.as_tensor(m['flat'], dtype=torch.float32).reshape(fs, t)
        mask = torch.as_tensor(m['mask'], dtype=torch.long)
        if mask.numel() != t:
            if mask.numel() > t:
                mask = mask[:t]
            else:
                mask = F.pad(mask, (0, t - mask.numel()), value=1)
        feats.append(feat)
        masks.append(mask)
        frame_lens.append(t)

    max_t = max(frame_lens)
    feat_bat = torch.stack(
        [F.pad(f, (0, max_t - f.shape[-1])) for f in feats], dim=0)
    mask_bat = torch.stack(
        [F.pad(m, (0, max_t - m.shape[-1])) for m in masks], dim=0)
    media_inputs = _to_float_dtype({'input_features': feat_bat}, torch_dtype)
    media_inputs['feature_attention_mask'] = mask_bat
    media_inputs['_mel_frame_lens'] = frame_lens  # consumed by encode helper only
    return media_inputs


def _pad_time_last(t: torch.Tensor, max_t: int, value: float) -> torch.Tensor:
    pad = max_t - int(t.shape[-1])
    if pad <= 0:
        return t
    return F.pad(t, (0, pad), value=value)


def _pad_batch_input_features(batch: List[Dict[str, Any]]) -> None:
    """Pad per-row Whisper mels so ms-swift can ``torch.concat`` them.

    ``_build_precomputed_source_mel_fields`` only pads *within* one sample.
    Lance stores native-length mels, so a DataLoader batch has
    ``input_features`` of shape ``[1, 128, T_i]`` with unequal ``T_i``.
    Swift ``_data_collator_mm_data`` does ``torch.concat(input_features)``
    and raises ``Sizes of tensors must match except in dimension 0``
    (B ``sts-cmp200lance-ds-1n-49la``: 192 vs 696). Pad the time dim to the
    batch max; ``feature_attention_mask`` pads with 0 so the audio encoder
    ignores extra frames. Token expansion already uses native ``frame_lens``.
    Does not change SCENE / batch size — collate-only.
    """
    feats = []
    for row in batch:
        feat = row.get('input_features')
        if feat is None:
            continue
        feat = feat if isinstance(feat, torch.Tensor) else torch.as_tensor(feat)
        row['input_features'] = feat
        feats.append(feat)
    if len(feats) < 2:
        return
    max_t = max(int(f.shape[-1]) for f in feats)
    if all(int(f.shape[-1]) == max_t for f in feats):
        return
    for row in batch:
        feat = row.get('input_features')
        if feat is None:
            continue
        row['input_features'] = _pad_time_last(feat, max_t, 0.0)
        mask = row.get('feature_attention_mask')
        if mask is not None:
            mask_t = mask if isinstance(mask, torch.Tensor) else torch.as_tensor(mask)
            row['feature_attention_mask'] = _pad_time_last(mask_t, max_t, 0)
        else:
            real_t = int(feat.shape[-1])
            ones = torch.ones(feat.shape[:-2] + (real_t,), dtype=torch.long)
            row['feature_attention_mask'] = _pad_time_last(ones, max_t, 0)


def _silence_placeholders_for_mel(mel_items: List[dict]) -> List[np.ndarray]:
    """Private encode-time placeholders so ``replace_tag`` still sees an audio slot.

    Not part of the JSONL contract — ``audios`` in the manifest stays ``[]`` for
    mel mode. ``replace_tag`` only checks ``isinstance(..., np.ndarray)`` and
    emits a fixed AUDIO tag; token expansion uses Lance ``frames``, not this
    array's length. Keep a length-1 float32 so we do not allocate
    ``frames * hop`` silence.
    """
    return [np.zeros(1, dtype=np.float32) for _ in mel_items]


def _slice_native_mel_row(feat: torch.Tensor, mask: torch.Tensor, frames: int):
    """Take ``[1, 128, T_i]`` / ``[1, T_i]`` from a possibly padded pack."""
    t = int(frames)
    if feat.dim() == 2:
        feat = feat.unsqueeze(0)
    if mask.dim() == 1:
        mask = mask.unsqueeze(0)
    return feat[:, :, :t].contiguous(), mask[:, :t].contiguous()


def attach_native_mel_tensors_to_rows(rows: List[Dict[str, Any]], torch_dtype) -> None:
    """Build mel tensors once for a DataLoader batch; write native-``T_i`` slices.

    Only touches rows whose ``source_input_features`` is a packed dict
    (``flat`` + ``frames``). Token expansion still reads ``frames``, not the
    padded pack width. In-place; encode can reuse the tensors.
    """
    items = []
    owners = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        value = row.get('source_input_features')
        if not (isinstance(value, dict) and 'flat' in value and 'frames' in value):
            continue
        if isinstance(value.get('input_features'), torch.Tensor):
            continue
        items.append(value)
        owners.append(value)
    if not items:
        return
    packed = _build_precomputed_source_mel_fields(items, torch_dtype)
    feats = packed['input_features']
    masks = packed['feature_attention_mask']
    for i, dest in enumerate(owners):
        t = int(dest['frames'])
        dest['input_features'], dest['feature_attention_mask'] = _slice_native_mel_row(
            feats[i], masks[i], t)


def _media_inputs_from_mel_items(mel_items: List[dict], torch_dtype) -> Dict[str, Any]:
    """Reuse batch-attached tensors when present; otherwise build from packed dicts."""
    if (
        mel_items
        and all(
            isinstance(item.get('input_features'), torch.Tensor)
            and isinstance(item.get('feature_attention_mask'), torch.Tensor)
            for item in mel_items
        )
    ):
        feats = []
        masks = []
        frame_lens = []
        for item in mel_items:
            t = int(item['frames'])
            feat, mask = _slice_native_mel_row(
                item['input_features'], item['feature_attention_mask'], t)
            feats.append(feat.squeeze(0))
            masks.append(mask.squeeze(0))
            frame_lens.append(t)
        if len(feats) == 1:
            feat_bat = feats[0].unsqueeze(0)
            mask_bat = masks[0].unsqueeze(0)
        else:
            max_t = max(frame_lens)
            feat_bat = torch.stack(
                [F.pad(f, (0, max_t - f.shape[-1])) for f in feats], dim=0)
            mask_bat = torch.stack(
                [F.pad(m, (0, max_t - m.shape[-1])) for m in masks], dim=0)
        media_inputs = _to_float_dtype({'input_features': feat_bat}, torch_dtype)
        media_inputs['feature_attention_mask'] = mask_bat
        media_inputs['_mel_frame_lens'] = frame_lens
        return media_inputs
    return _build_precomputed_source_mel_fields(mel_items, torch_dtype)


def _encode_omni_with_precomputed_mel(template, inputs, mel_items: List[dict]) -> Dict[str, Any]:
    """Text encode + inject Whisper mel (skip wav decode / feature_extractor).

    Mirrors ``Qwen2_5OmniTemplate._encode`` audio-token expansion. Media tensors
    come from ``_media_inputs_from_mel_items`` (reuse a DataLoader-batch pack
    when present, otherwise build from packed dicts). Token expansion still
    uses native ``frames``, not any pack pad width.
    """
    inputs.audios = _silence_placeholders_for_mel(mel_items)
    encoded = Template._encode(template, inputs)

    media_inputs = _media_inputs_from_mel_items(
        mel_items, template.model_info.torch_dtype)
    frame_lens = media_inputs.pop('_mel_frame_lens')

    input_ids = encoded['input_ids']
    labels = encoded['labels']
    loss_scale = encoded.get('loss_scale', None)
    config = template.config.thinker_config
    if getattr(template, 'version', None) == 'omni_v3':
        audio_token_id = [config.audio_token_id]
    else:
        audio_token_id = template._tokenize('<|AUDIO|>')
    idx_list = _findall_tokens(input_ids, audio_token_id)
    audio_feature_lengths = torch.tensor(frame_lens, dtype=torch.long)
    audio_lengths = template._get_feat_extract_output_lengths(audio_feature_lengths)
    if idx_list:
        def _get_new_audio_tokens(i):
            return audio_token_id * int(audio_lengths[i])

        input_ids, labels, loss_scale = template._extend_tokens(
            input_ids, labels, loss_scale, idx_list, _get_new_audio_tokens)
    encoded['input_ids'] = input_ids
    encoded['labels'] = labels
    encoded['loss_scale'] = loss_scale
    encoded.update(media_inputs)
    return encoded


class QwenOmniSpeakBridgeAlignmentTemplate(QwenOmniSpeakTemplate):  # pyright: ignore[reportGeneralTypeIssues]
    """Reuse Qwen-OmniSpeak multimodal encoding and append target codec labels."""

    def _codec_config(self):
        return _codec_config_from(self)

    def replace_tag(self, media_type, index, inputs):
        # Upstream Qwen2_5OmniTemplate.replace_tag always calls load_audio().
        # Mel-mode synthesize float32 silence placeholders in
        # ``_silence_placeholders_for_mel``; soundfile rejects ndarray — skip
        # reload and emit the same audio bos/eos tags.
        if media_type == 'audio':
            audio = inputs.audios[index]
            if isinstance(audio, np.ndarray):
                if getattr(self, 'version', None) == 'omni_v3':
                    return ['<|audio_start|><|audio_pad|><|audio_end|>']
                return ['<|audio_bos|><|AUDIO|><|audio_eos|>']
        return super().replace_tag(media_type, index, inputs)

    def _encode(self, inputs) -> Dict[str, Any]:
        # Audio validity (missing / over-long) is filtered at dataset-load time
        # in dataset.py::drop_row_if_invalid_audio, which also covers the val
        # split. A Template._encode filter would not: is_training is False for
        # the val split (and lazy_tokenize only *replaces* errored rows), so an
        # over-long val sample reached evaluate() and crashed the run.
        #
        # Capture the raw source-audio path(s) BEFORE ``super()._encode`` loads
        # them into waveform arrays. The s2s speaker x-vector is extracted from
        # the *source* audio; after loading, ``inputs.audios`` holds numpy arrays
        # (not paths), so we must grab the path string up front.
        _raw_audios = [a for a in (getattr(inputs, 'audios', None) or [])]
        # Self-describing mel: presence of ``source_input_features`` (URI or
        # packed dict) selects the precomputed path — same idea as
        # ``target_codec_codes`` selecting ``_build_precomputed_codec_fields``.
        mel_items = _normalize_precomputed_mel(
            inputs.extra_kwargs.get('source_input_features'))
        _resolved_audios = None

        if mel_items is not None:
            encoded = _encode_omni_with_precomputed_mel(self, inputs, mel_items)
        else:
            # Legacy standalone paths pass through unchanged. Indexed wds://
            # references become complete encoded audio bytes, which the upstream
            # ms-swift loader already knows how to decode and resample.
            # lance:// refs are coalesced into one take (source/target each).
            _resolved_audios = resolve_audio_references(_raw_audios)
            if _raw_audios:
                # Upstream mutates ``inputs.audios`` in place from encoded bytes to
                # numpy waveforms.  Give it a distinct list so our bytes snapshot
                # remains available for the speaker-prompt decoder below.
                inputs.audios = list(_resolved_audios)
            encoded = super()._encode(inputs)
        if not self.is_training:
            return encoded
        target_audio = inputs.extra_kwargs.get('target_audio')
        target_codec_codes = inputs.extra_kwargs.get('target_codec_codes')
        # Normal path: LazyLLMDataset.__getitems__ resolves this together with
        # mel/speaker for the whole DataLoader batch. Keep a per-row fallback
        # for direct template.encode calls and older PyTorch fetchers.
        if is_lance_uri(target_codec_codes):
            target_codec_codes = resolve_lance_codec_codes(target_codec_codes)
        has_codec_codes = _nonempty_vector(target_codec_codes)
        if not target_audio and not has_codec_codes:
            raise ValueError(
                'Bridge alignment training requires `target_audio` or '
                '`target_codec_codes` in each sample.')
        cfg = self._codec_config()
        use_spk = _use_speaker_prompt(self.config)
        full_add = (
            cfg.get('full_add_layout', False)
            and cfg.get('use_upstream_talker_input', False)
        )
        t_prompt = _first_reply_index(encoded.get('labels')) if full_add else 0
        control_prefix, speaker_slot_index = _build_codec_control_prefix(
            cfg,
            labels=encoded.get('labels'),
            target_lang=inputs.extra_kwargs.get('target_lang'),
            with_speaker=use_spk,
        )
        if has_codec_codes:
            codec_fields = _build_precomputed_codec_fields(
                target_codec_codes, cfg, control_prefix)
        else:
            codec_fields = get_codec_tokenizer().encode_target_audio(
                target_audio,
                control_prefix_ids=control_prefix,
                codec_eos_id=cfg['codec_eos_id'],
                num_residual_groups=cfg['num_residual_groups'],
                residual_vocab_size=cfg['residual_vocab_size'],
            )
        encoded.update(codec_fields)
        # ── ICL mode: pass reference (source) codec + text ids to the model ──
        # Enabled by env QWEN_OMNISPEAK_USE_ICL=true (default off → x-vector only).
        # The parquet carries ``source_codec_codes`` ([T,16]) and
        # ``source_text_ids`` ([T_txt]); the model's ``_build_icl_prefix`` turns
        # them into the leading in-context segment (see modeling forward).
        import os as _os_icl
        if _os_icl.environ.get('QWEN_OMNISPEAK_USE_ICL', 'false').lower() in ('1', 'true', 'yes'):
            _src_codec = inputs.extra_kwargs.get('source_codec_codes')
            _src_text_ids = inputs.extra_kwargs.get('source_text_ids')
            if _src_codec is not None:
                encoded['source_codec_codes'] = _src_codec
            if _src_text_ids is not None:
                encoded['source_text_ids'] = _src_text_ids
        if use_spk:
            # Prefer offline ECAPA: Lance ``source_speaker_embedding``, then
            # master ``source_xvector``. Falls back to online extract from wav.
            pre_emb = inputs.extra_kwargs.get('source_speaker_embedding')
            if _nonempty_vector(pre_emb) or is_lance_uri(pre_emb):
                if is_lance_uri(pre_emb):
                    pre_emb = resolve_lance_speaker_embedding(pre_emb)
                arr = None
                if isinstance(pre_emb, np.ndarray):
                    arr = pre_emb.astype(np.float32, copy=False).reshape(-1)
                elif isinstance(pre_emb, (list, tuple)) and len(pre_emb) > 0:
                    arr = np.asarray(pre_emb, dtype=np.float32).reshape(-1)
                else:
                    try:
                        import torch as _torch
                        if isinstance(pre_emb, _torch.Tensor):
                            arr = pre_emb.detach().float().cpu().numpy().reshape(-1)
                    except Exception:
                        arr = None
                if arr is not None and arr.size > 0:
                    if arr.size != 2048:
                        raise ValueError(
                            f'source_speaker_embedding dim {arr.size} != 2048 '
                            '(expected offline ECAPA x-vector)')
                    encoded['speaker_embedding'] = arr
                    encoded['speaker_slot_index'] = int(speaker_slot_index)
            if encoded.get('speaker_embedding') is None:
                # Speaker x-vector source (unified Qwen3-TTS ECAPA extraction):
                #   * TTS task              -> ``reference_audios``
                #   * S2S translation task  -> ``audios`` (source)
                # If the dataset provides a pre-computed ``source_xvector``
                # (offline ECAPA), pass it through and skip loading wav.
                _precomp_xvec = inputs.extra_kwargs.get('source_xvector')
                _has_xvec = (_precomp_xvec is not None
                             and not isinstance(_precomp_xvec, str)
                             and len(_precomp_xvec) > 0)
                if _has_xvec:
                    import numpy as _np_xv
                    xv = _np_xv.asarray(_precomp_xvec, dtype=_np_xv.float32).reshape(-1)
                    encoded['source_xvector'] = xv
                    encoded['speaker_embedding'] = xv
                    encoded['speaker_slot_index'] = int(speaker_slot_index)
                else:
                    if mel_items is not None:
                        raise ValueError(
                            'use_speaker_prompt requires source waveform or '
                            'source_speaker_embedding, but this sample uses '
                            'precomputed mel without a speaker x-vector. '
                            'Materialize source_speaker_embedding or set '
                            'QWEN_OMNISPEAK_USE_SPEAKER_PROMPT=false.')
                    # Fallback: load raw waveform, forward runs ECAPA per-step.
                    spk_path = _resolve_speaker_audio_path(inputs, raw_audios=_raw_audios)
                    if not spk_path:
                        raise ValueError(
                            'use_speaker_prompt is on but the sample has no speaker audio '
                            '(TTS needs reference_audios; s2s needs audios/source). '
                            'Build the dataset with the right field or unset '
                            'QWEN_OMNISPEAK_USE_SPEAKER_PROMPT.')
                    # Reuse already-read WDS bytes when the speaker reference is
                    # the S2S source audio. Mel path has no ``_resolved_audios``.
                    spk_audio = spk_path
                    if _resolved_audios is not None:
                        for raw_audio, resolved_audio in zip(_raw_audios, _resolved_audios):
                            if spk_path == raw_audio:
                                spk_audio = resolved_audio
                                break
                    else:
                        for i, raw_audio in enumerate(_raw_audios):
                            if (spk_path == raw_audio
                                    and isinstance(inputs.audios, (list, tuple))
                                    and i < len(inputs.audios)):
                                spk_audio = inputs.audios[i]
                                break
                    ref_wav, ref_sr = load_reference_waveform(spk_audio)
                    encoded['reference_audio_wav'] = ref_wav
                    encoded['reference_audio_sr'] = int(ref_sr)
                    encoded['speaker_slot_index'] = int(speaker_slot_index)
        # ── Upstream Qwen3-TTS talker input protocol side-fields ─────────
        # Only emitted when the model config asks for it; the forward on other
        # branches ignores these keys entirely (via ``_ignored_key`` filter).
        if cfg.get('use_upstream_talker_input', False):
            target_text = inputs.extra_kwargs.get('target_text')
            if target_text is None:
                # Fallback: use the assistant span from the tokenised inputs.
                # This may over-count trailing chatml tokens but stays safe
                # because the model right-pads with ``tts_pad_embed``.
                target_text_ids = list(encoded.get('input_ids') or [])
            else:
                # Tokenise the target transcription with the *thinker* tokenizer
                # (bit-perfect equivalent to the talker's text_embedding vocab;
                # verified in M1a). ``add_special_tokens=False`` avoids
                # accidentally inserting an ``<|endoftext|>`` at the tail.
                target_text_ids = self.tokenizer.encode(
                    str(target_text), add_special_tokens=False)
            encoded['target_text_ids'] = list(map(int, target_text_ids))
            # ``codec_bos_index`` = position of ``codec_bos_id`` in
            # ``codec_input_ids``. It is at the very end of ``control_prefix``.
            encoded['codec_bos_index'] = int(len(control_prefix) - 1)
            if full_add:
                # Length of the thinker prompt run (= number of leading
                # ``codec_mask`` positions). ``forward`` scatters the thinker
                # prompt signal into ``text_channel[0:full_add_prompt_len]`` and
                # places ``text_bos`` at ``codec_bos_index - 1``.
                encoded['full_add_prompt_len'] = int(t_prompt)
        encoded['run_thinker'] = True
        encoded['bridge_task'] = inputs.extra_kwargs.get('bridge_task') or inputs.extra_kwargs.get('task')
        return encoded

    @staticmethod
    def _collect(batch: List[Dict[str, Any]], key: str):
        return _collect(batch, key)

    def _ensure_codec_fields(self, batch: List[Dict[str, Any]], cfg: Dict[str, int]) -> None:
        """Re-derive codec supervision fields when the dataset pipeline dropped them.

        In the transformers/DeepSpeed path ``_encode`` already attaches
        ``codec_input_ids`` / ``codec_labels`` / ``codec_attention_mask`` /
        ``codec_residual_labels`` to every sample, so this is a no-op.

        In the Megatron path the dataset's pre-tokenisation step only keeps a
        token/whitelist schema and strips these extra keys, while source fields
        survive inside ``_extra_kwargs`` (preserved because
        ``--remove_unused_columns false``).

        The recovery must reproduce the ``_encode`` protocol EXACTLY (P1-3):
        resolve ``lance://`` codec pointers, prefer precomputed
        ``target_codec_codes``, rebuild the control prefix with the same
        speaker-slot and full-add layout ``_encode`` would have used, and
        restore ``speaker_slot_index``.
        """
        use_spk = _use_speaker_prompt(self.config)
        for row in batch:
            if row.get('codec_input_ids') is not None and row.get('codec_labels') is not None:
                continue
            ek = row.get('_extra_kwargs') or {}
            target_codec_codes = ek.get('target_codec_codes')
            target_audio = ek.get('target_audio')
            if is_lance_uri(target_codec_codes):
                target_codec_codes = resolve_lance_codec_codes(target_codec_codes)

            has_codec_codes = _nonempty_vector(target_codec_codes)

            if not has_codec_codes and not target_audio:
                # Cannot recover codec supervision for this row; leave it and
                # let the collator raise a clear error below if it is required.
                continue
            control_prefix, speaker_slot_index = _build_codec_control_prefix(
                cfg,
                labels=row.get('labels'),
                target_lang=ek.get('target_lang'),
                with_speaker=use_spk,
            )
            if has_codec_codes:
                codec_fields = _build_precomputed_codec_fields(
                    target_codec_codes, cfg, control_prefix)
            else:
                codec_fields = get_codec_tokenizer().encode_target_audio(
                    target_audio,
                    control_prefix_ids=control_prefix,
                    codec_eos_id=cfg['codec_eos_id'],
                    num_residual_groups=cfg['num_residual_groups'],
                    residual_vocab_size=cfg['residual_vocab_size'])
            row.update(codec_fields)
            if use_spk and speaker_slot_index is not None:
                row['speaker_slot_index'] = int(speaker_slot_index)
            row['run_thinker'] = True
            row['bridge_task'] = ek.get('bridge_task') or ek.get('task')

    def packing_row(self, row: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Carry codec supervision fields through ``padding_free`` packing.

        WARNING: this path is NOT a supported configuration. Megatron's talker
        codec loss requires ``padding_free=False`` (see ``common_args_megatron.sh``
        and ``mcore_talker_loss._check_parallel_guards``). In padding_free mode
        Megatron concatenates multiple samples into a single sequence, which
        breaks the per-sample 1:1 codec<->semantic alignment the block-causal
        mask relies on (cross-sample leakage), so ``compute_codec_loss`` cannot
        consume a concatenated codec sequence correctly. The guard hard-fails on
        ``padding_free=True`` before this is ever reached.

        The override is kept only as a defensive no-op: if padding_free were ever
        re-enabled, it re-attaches the codec fields so the failure is a clear
        guard error rather than a silent ``Encoded batch is missing``. The
        supported data path is ``padding_free=False``, where ``_data_collator``
        collects codec fields per-sample into ``[B, T]`` / ``[B, T, G]`` tensors.
        """
        packed = super().packing_row(row)
        for key in ('codec_input_ids', 'codec_labels', 'codec_attention_mask', 'codec_residual_labels'):
            vals = [x.get(key) for x in row if x.get(key) is not None]
            if not vals:
                continue
            if isinstance(vals[0], torch.Tensor):
                if vals[0].dim() == 1:
                    packed[key] = torch.cat([v.reshape(-1) for v in vals], dim=0)
                else:  # [T_i, G] per sample -> [sum T, G]
                    packed[key] = torch.cat(vals, dim=0)
            else:
                packed[key] = sum((list(v) for v in vals), start=[])
        return packed

    def _data_collator(self, batch: List[Dict[str, Any]], *, padding_to: Optional[int] = None) -> Dict[str, Any]:
        # Lance native-length mels must share a time dim before swift concat.
        _pad_batch_input_features(batch)
        res = super()._data_collator(batch, padding_to=padding_to)
        cfg = self._codec_config()
        # Supported path: ``padding_free=False`` (set in common_args_megatron.sh).
        # The base collator only keeps a token whitelist and drops the custom
        # codec fields, so we re-collect them per-sample into ``[B, T]`` /
        # ``[B, T, G]`` tensors that ``compute_codec_loss`` consumes. This also
        # re-derives codec fields from ``_extra_kwargs.target_audio`` for any
        # path that lost them (no-op when ``_encode`` already attached them).
        self._ensure_codec_fields(batch, cfg)

        codec_input_ids = self._collect(batch, 'codec_input_ids')
        codec_labels = self._collect(batch, 'codec_labels')
        codec_attention_mask = self._collect(batch, 'codec_attention_mask')
        # ---- Row-aligned residual collection (crash fix) ----
        # ``_collect`` only gathers rows that HAVE the key. If one sample in the
        # batch lacks ``codec_residual_labels`` (e.g. the codec tokenizer / an
        # old cache entry returned a single codebook so
        # ``_build_training_tensors`` never attached it), the residual tensor
        # collates to [B-1, T, G] while the main labels are [B, T]. Downstream
        # ``forward_teacher_forced`` then reshapes B*T residual frames and dies
        # with e.g. "shape '[880, 15]' is invalid for input of size 11550"
        # (880 = 8x110 vs 770 = 7x110), killing all ranks (observed on fj8e,
        # master crash 2026-07-24 10:59 UTC). Rows without residual supervision
        # must still occupy their batch slot as all -100 (ignored by the loss),
        # exactly like ``_build_codec_batch`` does for the CPT template. Each
        # placeholder/row is also length-aligned to that row's ``codec_labels``
        # so per-row T always matches the main stream.
        num_g = cfg['num_residual_groups']
        codec_residual_labels = []
        for i, row in enumerate(batch):
            v = row.pop('codec_residual_labels', None)
            t = int(codec_labels[i].shape[0]) if i < len(codec_labels) else 0
            if v is None:
                codec_residual_labels.append(torch.full((t, num_g), -100, dtype=torch.long))
                continue
            v = torch.as_tensor(v, dtype=torch.long)
            if v.shape[0] < t:  # defensively pad short residual rows
                pad_rows = torch.full((t - v.shape[0], num_g), -100, dtype=torch.long)
                v = torch.cat([v, pad_rows], dim=0)
            elif v.shape[0] > t:  # defensively truncate over-long residual rows
                v = v[:t]
            codec_residual_labels.append(v)
        if not codec_input_ids or not codec_labels:
            raise ValueError('Encoded batch is missing codec_input_ids/codec_labels '
                             'even after recovery; check that `target_codec_codes` '
                             'or `target_audio` is present in `_extra_kwargs`.')

        res['codec_input_ids'] = pad_1d(codec_input_ids, cfg['codec_pad_id'], dtype=torch.long)
        res['codec_labels'] = pad_1d(codec_labels, -100, dtype=torch.long)
        if codec_attention_mask:
            res['codec_attention_mask'] = pad_1d(codec_attention_mask, 0, dtype=torch.long)
        else:
            res['codec_attention_mask'] = (res['codec_input_ids'] != cfg['codec_pad_id']).long()
        try:
            from mfu_logging import accumulate_pad_waste
            _lens = res['codec_attention_mask'].sum(dim=-1).tolist()
            accumulate_pad_waste(_lens)
        except Exception:
            pass
        # Row-aligned residual collection above guarantees one entry per sample,
        # so pad unconditionally (master crash fix).
        res['codec_residual_labels'] = pad_2d_time(codec_residual_labels, -100, dtype=torch.long)

        # Voice-clone: pass per-sample reference waveforms + speaker slot index
        # through to the model forward (resample / mel / encoder all run there).
        # Waveforms are variable length, so keep them as a Python list instead of
        # padding; the forward extracts one x-vector per sample.
        # Precomputed ``speaker_embedding`` (offline ECAPA) skips the online path.
        ref_wavs = self._collect(batch, 'reference_audio_wav')
        spk_embs = self._collect(batch, 'speaker_embedding')
        if spk_embs:
            res['speaker_embedding'] = [
                torch.as_tensor(e, dtype=torch.float32).reshape(-1) for e in spk_embs]
            res['speaker_slot_index'] = torch.tensor(
                self._collect(batch, 'speaker_slot_index'), dtype=torch.long)
        elif ref_wavs:
            # Convert to CPU float32 tensors here (variable length -> keep as a
            # list, not stacked). This avoids relying on how the ms-swift / HF
            # ``_prepare_inputs`` path treats bare numpy arrays; torch tensors are
            # handled uniformly. The forward moves each to the right device.
            res['reference_audio_wav'] = [
                torch.as_tensor(w, dtype=torch.float32) for w in ref_wavs]
            res['reference_audio_sr'] = self._collect(batch, 'reference_audio_sr')
            res['speaker_slot_index'] = torch.tensor(
                self._collect(batch, 'speaker_slot_index'), dtype=torch.long)

        # The standard SFT trainer requires a `labels` key and uses it (after a
        # next-token shift) to compute token accuracy against ``outputs.logits``.
        # The model surfaces the MAIN codec logits as ``outputs.logits`` in this
        # codec-only mode, where prediction at position ``t`` (fed codec_input_ids
        # = [BOS, m0, .., m_{T-1}]) targets the main code ``codec_labels[t]``.
        # swift compares ``preds[:-1]`` with ``labels[1:]``, so we build a
        # right-shifted label stream ``[-100, m0, m1, .., m_{T-1}]`` (EOS dropped,
        # pads kept as -100). After swift's shift this aligns pred(t) <-> m_t and
        # ignores padded / EOS positions, giving a correct main-codec token acc.
        codec_labels_tensor = res['codec_labels']
        acc_labels = codec_labels_tensor.new_full(codec_labels_tensor.shape, -100)
        acc_labels[:, 1:] = codec_labels_tensor[:, :-1]
        # Upstream input protocol (reply-add / full-add) needs the *thinker text*
        # labels (reply = target_text positions) to reach forward so it can
        # isolate the reply span. Overwriting ``labels`` with the codec-aligned
        # stream (as the codec-only bridge path does) would null ``text_labels``
        # in the model forward. So keep the parent's text labels for upstream;
        # the forward still computes codec loss/accuracy internally.
        # Always set labels = codec-aligned labels (identical to the original
        # bridge-concat behaviour). This ensures swift's loss scaling uses codec
        # token counts and returned_logits = main_codec_logits, matching the
        # original pipeline exactly.
        if cfg.get('use_upstream_talker_input', False) and 'labels' in res:
            # Upstream mode: save the thinker text labels in a separate field
            # so the forward can still identify reply positions and compute
            # token_text_acc, without polluting the main ``labels`` stream.
            res['thinker_text_labels'] = res['labels']
        res['labels'] = acc_labels
        res['run_thinker'] = True

        # ── Upstream input protocol pass-throughs ─────────────────────────
        # These keys are only present when the model config has
        # ``use_upstream_talker_input=True`` (see ``_encode``); on other
        # templates / branches ``_collect`` returns [] and we skip padding.
        target_text_ids_list = self._collect(batch, 'target_text_ids')
        if target_text_ids_list:
            target_text_ids_tensors = [torch.as_tensor(x, dtype=torch.long)
                                       for x in target_text_ids_list]
            res['target_text_ids'] = pad_1d(
                target_text_ids_tensors, cfg['tts_pad_token_id'], dtype=torch.long)
        codec_bos_index_list = self._collect(batch, 'codec_bos_index')
        if codec_bos_index_list:
            res['codec_bos_index'] = torch.tensor(
                [int(x) for x in codec_bos_index_list], dtype=torch.long)
        full_add_prompt_len_list = self._collect(batch, 'full_add_prompt_len')
        if full_add_prompt_len_list:
            res['full_add_prompt_len'] = torch.tensor(
                [int(x) for x in full_add_prompt_len_list], dtype=torch.long)

        # ── ICL mode: batch reference (source) codec [T,16] + text ids ──
        src_codec = self._collect(batch, 'source_codec_codes')
        if src_codec:
            codec_list = [torch.as_tensor(x, dtype=torch.long) for x in src_codec]
            res['source_codec_codes'] = pad_2d_time(codec_list, cfg['codec_pad_id'], dtype=torch.long)
            # keep-mask: 1 for real ref frames, 0 for padding (use main code group)
            lens = [t.shape[0] for t in codec_list]
            T = res['source_codec_codes'].shape[1]
            res['source_codec_mask'] = torch.stack([
                torch.cat([torch.ones(L, dtype=torch.bool),
                           torch.zeros(T - L, dtype=torch.bool)]) for L in lens])
            src_text = self._collect(batch, 'source_text_ids')
            if src_text:
                text_list = [torch.as_tensor(x, dtype=torch.long) for x in src_text]
                res['source_text_ids'] = pad_1d(text_list, cfg['codec_pad_id'], dtype=torch.long)
        return res

    def data_collator(self, batch: List[Dict[str, Any]], *, padding_to: Optional[int] = None) -> Dict[str, Any]:
        res = super().data_collator(batch, padding_to=padding_to)
        # `remove_unused_columns=false` is needed so target_audio reaches encode(),
        # but raw JSONL fields must not be forwarded into model.forward().
        for key in _DROP_EXTRA_KEYS:
            res.pop(key, None)
        return res


def register_bridge_alignment_template() -> None:
    register_template(
        TemplateMeta(
            template_type=QWEN_OMNISPEAK_TEMPLATE,
            prefix=[],
            prompt=['<|im_start|>user\n{{QUERY}}<|im_end|>\n<|im_start|>assistant\n'],
            chat_sep=['<|im_end|>\n'],
            suffix=['<|im_end|>'],
            system_prefix=['<|im_start|>system\n{{SYSTEM}}<|im_end|>\n'],
            default_system='You are a helpful assistant.',
            stop_words=['<|endoftext|>'],
            agent_template='hermes',
            template_cls=QwenOmniSpeakBridgeAlignmentTemplate,
        ),
        exist_ok=True,
    )
    logger.info('Registered bridge-alignment template override for qwen_omnispeak.')


# ---------------------------------------------------------------------------
# Continue pretrain template (mixed text-only + codec-supervised batches).
# ---------------------------------------------------------------------------


class QwenOmniSpeakContinuePretrainTemplate(QwenOmniSpeakTemplate):  # pyright: ignore[reportGeneralTypeIssues]
    """Continue-pretrain template.

    Per-sample behaviour:
        * speech_to_text_translation / multi_speaker_segmentation:
            keep the standard text labels emitted by the parent template;
            codec fields are NOT generated.
        * speech_to_speech_translation:
            keep the standard text labels (used for thinker text loss)
            AND attach codec_input_ids / codec_labels / codec_attention_mask
            / codec_residual_labels for talker codec loss.

    Per-batch behaviour:
        * Always pads text fields via the parent collator.
        * Pads codec fields only when at least one sample in the batch
          carries codec supervision; non-codec rows are zero-padded so that
          the model's forward can detect "no codec supervision" via masks.
        * The original ``labels`` produced by the parent collator (text LM
          labels, with -100 for prompt tokens) is preserved unchanged so the
          thinker text CE loss is computed correctly.
    """

    def _codec_config(self):
        return _codec_config_from(self)

    def _encode(self, inputs) -> Dict[str, Any]:
        # Audio validity filtered at load time (dataset.py::drop_row_if_invalid_audio).
        encoded = super()._encode(inputs)
        bridge_task = inputs.extra_kwargs.get('bridge_task') or inputs.extra_kwargs.get('task')
        encoded['bridge_task'] = bridge_task
        # Always run the thinker for continue-pretrain so text loss / talker
        # input both have valid hidden states.
        encoded['run_thinker'] = True

        if not self.is_training:
            return encoded

        # Only s2s translation rows produce codec supervision.
        if bridge_task != TASK_S2S_TRANSLATION:
            return encoded

        target_audio = inputs.extra_kwargs.get('target_audio')
        if not target_audio:
            raise ValueError(
                'speech_to_speech_translation training requires `target_audio` in each sample.')
        cfg = self._codec_config()
        control_prefix, _ = _build_control_prefix(cfg, inputs.extra_kwargs.get('target_lang'))
        codec_fields = get_codec_tokenizer().encode_target_audio(
            target_audio,
            control_prefix_ids=control_prefix,
            codec_eos_id=cfg['codec_eos_id'],
            num_residual_groups=cfg['num_residual_groups'],
            residual_vocab_size=cfg['residual_vocab_size'],
        )
        encoded.update(codec_fields)
        return encoded

    @staticmethod
    def _collect(batch: List[Dict[str, Any]], key: str):
        return _collect(batch, key)

    def _build_codec_batch(self, batch: List[Dict[str, Any]], cfg: Dict[str, int]) -> Dict[str, Any]:
        """Pad codec tensors to a uniform batch.

        Rows without codec supervision are emitted as a single ``[BOS]`` token
        with attention_mask=0 and labels filled with -100, so the model's
        ``forward`` will skip codec loss for those rows.
        """
        bos = cfg['codec_bos_id']
        pad = cfg['codec_pad_id']
        num_g = cfg['num_residual_groups']

        ids_list = []
        labels_list = []
        attn_list = []
        residual_list = []
        has_codec_flags = []
        for row in batch:
            codec_input_ids = row.pop('codec_input_ids', None)
            codec_labels = row.pop('codec_labels', None)
            codec_attention_mask = row.pop('codec_attention_mask', None)
            codec_residual_labels = row.pop('codec_residual_labels', None)
            if codec_input_ids is None or codec_labels is None:
                # Placeholder: a single BOS, masked out so loss is skipped.
                ids_list.append(torch.tensor([bos], dtype=torch.long))
                labels_list.append(torch.tensor([-100], dtype=torch.long))
                attn_list.append(torch.tensor([0], dtype=torch.long))
                residual_list.append(torch.full((1, num_g), -100, dtype=torch.long))
                has_codec_flags.append(False)
                continue
            ids_list.append(torch.as_tensor(codec_input_ids, dtype=torch.long))
            labels_list.append(torch.as_tensor(codec_labels, dtype=torch.long))
            if codec_attention_mask is not None:
                attn_list.append(torch.as_tensor(codec_attention_mask, dtype=torch.long))
            else:
                attn_list.append(torch.ones_like(ids_list[-1], dtype=torch.long))
            if codec_residual_labels is not None:
                residual_list.append(torch.as_tensor(codec_residual_labels, dtype=torch.long))
            else:
                t = ids_list[-1].shape[0]
                residual_list.append(torch.full((t, num_g), -100, dtype=torch.long))
            has_codec_flags.append(True)

        result = {
            'codec_input_ids': pad_1d(ids_list, pad, dtype=torch.long),
            'codec_labels': pad_1d(labels_list, -100, dtype=torch.long),
            'codec_attention_mask': pad_1d(attn_list, 0, dtype=torch.long),
            'codec_residual_labels': pad_2d_time(residual_list, -100, dtype=torch.long),
        }
        result['has_codec_supervision'] = torch.tensor(has_codec_flags, dtype=torch.bool)
        return result

    def _data_collator(self, batch: List[Dict[str, Any]], *, padding_to: Optional[int] = None) -> Dict[str, Any]:
        res = super()._data_collator(batch, padding_to=padding_to)
        cfg = self._codec_config()
        codec_batch = self._build_codec_batch(batch, cfg)
        res.update(codec_batch)
        # Keep the original text labels for thinker text CE loss; do NOT
        # overwrite ``labels`` with codec labels (this is the key difference
        # from the bridge-alignment template).
        res['run_thinker'] = True
        return res

    def data_collator(self, batch: List[Dict[str, Any]], *, padding_to: Optional[int] = None) -> Dict[str, Any]:
        res = super().data_collator(batch, padding_to=padding_to)
        for key in _DROP_EXTRA_KEYS:
            res.pop(key, None)
        return res


def register_continue_pretrain_template() -> None:
    register_template(
        TemplateMeta(
            template_type=QWEN_OMNISPEAK_CONTINUE_PRETRAIN_TEMPLATE,
            prefix=[],
            prompt=['<|im_start|>user\n{{QUERY}}<|im_end|>\n<|im_start|>assistant\n'],
            chat_sep=['<|im_end|>\n'],
            suffix=['<|im_end|>'],
            system_prefix=['<|im_start|>system\n{{SYSTEM}}<|im_end|>\n'],
            default_system='You are a helpful assistant.',
            stop_words=['<|endoftext|>'],
            agent_template='hermes',
            template_cls=QwenOmniSpeakContinuePretrainTemplate,
        ),
        exist_ok=True,
    )
    logger.info('Registered continue-pretrain template for qwen_omnispeak.')


# ---------------------------------------------------------------------------
# Simultaneous (read-write) policy learning template (Stage 3 SFT).
# ---------------------------------------------------------------------------


class QwenOmniSpeakSimultaneousPolicyTemplate(QwenOmniSpeakContinuePretrainTemplate):  # pyright: ignore[reportGeneralTypeIssues]
    """Stage-3 simultaneous-interpretation template.

    Builds the per-chunk concatenated target-codec supervision, plus:
      * the codec stream is the *concatenation* of the per-chunk ``target_audios``
        (so the talker is supervised on the full target utterance), and
      * a monotonic non-decreasing ``codec_to_sem_visible`` read-write schedule
        is derived from the chunk-level ``alignment`` timestamps. Each target
        codec frame may only attend to the source-audio prefix that has "arrived"
        by its chunk's ``source_end_ms`` (plus the prompt tokens), implicitly
        teaching the model *when to speak* via the existing codec CE loss.

    This Stage-3 task does NOT predict target text: the model maps source audio
    directly to the target codec stream. Supervision is therefore **codec-only**
    (like bridge alignment) — there is no thinker text CE loss and no text token
    accuracy. The collator surfaces the main codec stream as the accuracy target.

    Streaming can be disabled (``SIMULTANEOUS_POLICY_STREAMING=0``) to recover the
    offline full-context behaviour for ablation; a global wait-k style offset is
    available via ``SIMULTANEOUS_POLICY_EXTRA_LAG_MS``.
    """

    # Environment-variable prefix for the streaming switches. Subclasses (e.g.
    # the normal_translation template) override this to isolate their own knobs
    # (``<PREFIX>_STREAMING`` / ``<PREFIX>_EXTRA_LAG_MS`` /
    # ``<PREFIX>_SOURCE_DURATION_FROM_FILE``) from the simultaneous-policy
    # pipeline. Default keeps the original ``SIMULTANEOUS_POLICY_*`` behaviour.
    env_prefix: str = 'SIMULTANEOUS_POLICY'

    def _encode(self, inputs) -> Dict[str, Any]:
        # Audio validity filtered at load time (dataset.py::drop_row_if_invalid_audio).
        # Skip the continue-pretrain s2s codec path; build our own concat stream.
        encoded = QwenOmniSpeakTemplate._encode(self, inputs)
        encoded['bridge_task'] = inputs.extra_kwargs.get('bridge_task') or inputs.extra_kwargs.get('task')
        encoded['run_thinker'] = True
        if not self.is_training:
            return encoded

        alignment = inputs.extra_kwargs.get('alignment') or []
        target_audios = inputs.extra_kwargs.get('target_audios') or [
            seg.get('target_audio') for seg in alignment if seg.get('target_audio')
        ]
        target_audios = [p for p in target_audios if p]
        if not target_audios:
            raise ValueError(
                'simultaneous_policy_learning training requires target_audios (or per-chunk '
                'alignment[i].target_audio).')

        cfg = self._codec_config()
        control_prefix, _ = _build_control_prefix(cfg, inputs.extra_kwargs.get('target_lang'))
        # Per-chunk target timestamps drive the real-timeline layout (silence in
        # the listen-only gaps). They are trusted as-is; alignment order matches
        # the target_audios order (chunk_id 0..N-1).
        chunk_start_ms = [seg.get('target_start_ms') for seg in alignment] if alignment else None
        chunk_end_ms = [seg.get('target_end_ms') for seg in alignment] if alignment else None
        codec_fields = get_codec_tokenizer().encode_target_audios_concat(
            target_audios,
            control_prefix_ids=control_prefix,
            codec_eos_id=cfg['codec_eos_id'],
            num_residual_groups=cfg['num_residual_groups'],
            residual_vocab_size=cfg['residual_vocab_size'],
            chunk_start_ms=chunk_start_ms,
            chunk_end_ms=chunk_end_ms,
        )
        # Pop the transient timeline metadata so it never leaks into forward.
        frame_ms = float(codec_fields.pop('frame_ms', 0.0) or 0.0)
        control_prefix_len = codec_fields.pop('control_prefix_len', len(control_prefix))
        num_main_frames = int(codec_fields.pop('num_main_frames', 0) or 0)
        codec_fields.pop('chunk_frame_bounds', None)  # legacy key (no longer produced)
        encoded.update(codec_fields)

        if _as_bool_env(f'{self.env_prefix}_STREAMING', True):
            try:
                extra_lag_ms = float(os.environ.get(f'{self.env_prefix}_EXTRA_LAG_MS', '0') or 0.0)
            except ValueError:
                extra_lag_ms = 0.0
            audio_token_id = _resolve_audio_token_id(self.config)
            schedule = _build_streaming_schedule(
                input_ids=encoded.get('input_ids'),
                audio_token_id=audio_token_id,
                control_prefix_len=control_prefix_len,
                num_main_frames=num_main_frames,
                frame_ms=frame_ms,
                source_duration_ms=_source_duration_ms(inputs, alignment, env_prefix=self.env_prefix),
                extra_lag_ms=extra_lag_ms,
            )
            if schedule is not None:
                encoded['codec_to_sem_visible'] = schedule
            elif audio_token_id is None:
                logger.warning(
                    'Could not resolve audio_token_id; falling back to offline '
                    '(full-context) codec attention for %s.', self.env_prefix.lower())
        return encoded

    def _data_collator(self, batch: List[Dict[str, Any]], *, padding_to: Optional[int] = None) -> Dict[str, Any]:
        # Collect the schedule BEFORE the parent pops codec tensors / pads them.
        raw_vis = [row.get('codec_to_sem_visible') for row in batch]
        res = super()._data_collator(batch, padding_to=padding_to)
        for row in batch:
            row.pop('codec_to_sem_visible', None)

        # --- Codec-only supervision (no text prediction) -------------------
        # This Stage-3 task does not predict target text, so we drop the thinker
        # text-CE path entirely and supervise the main codec stream instead
        # (mirroring the bridge-alignment template). Concretely:
        #   * remove ``has_codec_supervision`` so the model's forward takes the
        #     codec-only branch (text_labels -> None, no thinker text CE / no
        #     ``token_text_acc``) and, under Megatron, ``bridge_mode`` zeroes the
        #     meaningless thinker text loss;
        #   * overwrite ``labels`` with a right-shifted copy of ``codec_labels``
        #     so the trainer's standard next-token shift yields the MAIN codec
        #     token accuracy against ``main_codec_logits``.
        res.pop('has_codec_supervision', None)
        codec_labels = res.get('codec_labels')
        if codec_labels is not None:
            acc_labels = codec_labels.new_full(codec_labels.shape, -100)
            acc_labels[:, 1:] = codec_labels[:, :-1]
            res['labels'] = acc_labels

        if any(v is not None for v in raw_vis):
            width = int(res['codec_input_ids'].shape[1])
            # Rows without a schedule fall back to full visibility (offline),
            # never zero visibility (which would blind that row's codec frames).
            full_vis = 1 << 30
            padded = []
            for v in raw_vis:
                if v is None:
                    padded.append(torch.full((width, ), full_vis, dtype=torch.long))
                    continue
                vis = torch.as_tensor(v, dtype=torch.long).flatten()
                if vis.shape[0] < width:
                    fill = int(vis[-1].item()) if vis.numel() > 0 else 0
                    pad = vis.new_full((width - vis.shape[0], ), fill)
                    vis = torch.cat([vis, pad], dim=0)
                elif vis.shape[0] > width:
                    vis = vis[:width]
                padded.append(vis)
            res['codec_to_sem_visible'] = torch.stack(padded, dim=0)
        return res


def register_simultaneous_policy_template() -> None:
    register_template(
        TemplateMeta(
            template_type=QWEN_OMNISPEAK_SIMULTANEOUS_POLICY_TEMPLATE,
            prefix=[],
            prompt=['<|im_start|>user\n{{QUERY}}<|im_end|>\n<|im_start|>assistant\n'],
            chat_sep=['<|im_end|>\n'],
            suffix=['<|im_end|>'],
            system_prefix=['<|im_start|>system\n{{SYSTEM}}<|im_end|>\n'],
            default_system='You are a helpful assistant.',
            stop_words=['<|endoftext|>'],
            agent_template='hermes',
            template_cls=QwenOmniSpeakSimultaneousPolicyTemplate,
        ),
        exist_ok=True,
    )
    logger.info('Registered simultaneous-policy template for qwen_omnispeak.')


# ---------------------------------------------------------------------------
# normal_translation template (Stage-3 SFT, streaming s2s translation).
# ---------------------------------------------------------------------------


class QwenOmniSpeakNormalTranslationTemplate(QwenOmniSpeakSimultaneousPolicyTemplate):  # pyright: ignore[reportGeneralTypeIssues]
    """Stage-3 normal_translation template (streaming speech-to-speech translation).

    Mechanically **identical** to :class:`QwenOmniSpeakSimultaneousPolicyTemplate`
    (codec-only supervision; real-timeline target-codec stream with silence-filled
    listen gaps; monotonic ``codec_to_sem_visible`` read-write schedule derived
    from the chunk ``alignment`` timestamps). The only difference is the
    environment-variable prefix used for the streaming switches, so the two
    pipelines never cross-talk:

      * ``NORMAL_TRANSLATION_STREAMING`` (default true)
      * ``NORMAL_TRANSLATION_EXTRA_LAG_MS`` (default 0)
      * ``NORMAL_TRANSLATION_SOURCE_DURATION_FROM_FILE`` (default true)
    """

    env_prefix: str = 'NORMAL_TRANSLATION'


def register_normal_translation_template() -> None:
    register_template(
        TemplateMeta(
            template_type=QWEN_OMNISPEAK_NORMAL_TRANSLATION_TEMPLATE,
            prefix=[],
            prompt=['<|im_start|>user\n{{QUERY}}<|im_end|>\n<|im_start|>assistant\n'],
            chat_sep=['<|im_end|>\n'],
            suffix=['<|im_end|>'],
            system_prefix=['<|im_start|>system\n{{SYSTEM}}<|im_end|>\n'],
            default_system='You are a helpful assistant.',
            stop_words=['<|endoftext|>'],
            agent_template='hermes',
            template_cls=QwenOmniSpeakNormalTranslationTemplate,
        ),
        exist_ok=True,
    )
    logger.info('Registered normal-translation template for qwen_omnispeak.')


# ---------------------------------------------------------------------------
# bilingual_translation template (Stage-3 SFT, streaming bilingual s2s).
# ---------------------------------------------------------------------------


class QwenOmniSpeakBilingualTranslationTemplate(QwenOmniSpeakSimultaneousPolicyTemplate):  # pyright: ignore[reportGeneralTypeIssues]
    """Stage-3 bilingual_translation template (streaming bilingual mutual translation).

    Mechanically **identical** to :class:`QwenOmniSpeakSimultaneousPolicyTemplate`
    (codec-only supervision; real-timeline target-codec stream with silence-filled
    listen gaps; monotonic ``codec_to_sem_visible`` read-write schedule derived
    from the chunk ``alignment`` timestamps). The only difference is the
    environment-variable prefix used for the streaming switches, so the three
    Stage-3 pipelines never cross-talk:

      * ``BILINGUAL_TRANSLATION_STREAMING`` (default true)
      * ``BILINGUAL_TRANSLATION_EXTRA_LAG_MS`` (default 0)
      * ``BILINGUAL_TRANSLATION_SOURCE_DURATION_FROM_FILE`` (default true)
    """

    env_prefix: str = 'BILINGUAL_TRANSLATION'


def register_bilingual_translation_template() -> None:
    register_template(
        TemplateMeta(
            template_type=QWEN_OMNISPEAK_BILINGUAL_TRANSLATION_TEMPLATE,
            prefix=[],
            prompt=['<|im_start|>user\n{{QUERY}}<|im_end|>\n<|im_start|>assistant\n'],
            chat_sep=['<|im_end|>\n'],
            suffix=['<|im_end|>'],
            system_prefix=['<|im_start|>system\n{{SYSTEM}}<|im_end|>\n'],
            default_system='You are a helpful assistant.',
            stop_words=['<|endoftext|>'],
            agent_template='hermes',
            template_cls=QwenOmniSpeakBilingualTranslationTemplate,
        ),
        exist_ok=True,
    )
    logger.info('Registered bilingual-translation template for qwen_omnispeak.')


# ---------------------------------------------------------------------------
# multi_speaker_translation template (Stage-3 SFT, streaming multi-speaker s2s).
# ---------------------------------------------------------------------------


class QwenOmniSpeakMultiSpeakerTranslationTemplate(QwenOmniSpeakSimultaneousPolicyTemplate):  # pyright: ignore[reportGeneralTypeIssues]
    """Stage-3 multi_speaker_translation template (streaming multi-speaker s2s translation).

    Mechanically **identical** to :class:`QwenOmniSpeakSimultaneousPolicyTemplate`
    (codec-only supervision; real-timeline target-codec stream with silence-filled
    listen gaps; monotonic ``codec_to_sem_visible`` read-write schedule derived
    from the chunk ``alignment`` timestamps). The only difference is the
    environment-variable prefix used for the streaming switches, so the Stage-3
    pipelines never cross-talk:

      * ``MULTI_SPEAKER_TRANSLATION_STREAMING`` (default true)
      * ``MULTI_SPEAKER_TRANSLATION_EXTRA_LAG_MS`` (default 0)
      * ``MULTI_SPEAKER_TRANSLATION_SOURCE_DURATION_FROM_FILE`` (default true)
    """

    env_prefix: str = 'MULTI_SPEAKER_TRANSLATION'


def register_multi_speaker_translation_template() -> None:
    register_template(
        TemplateMeta(
            template_type=QWEN_OMNISPEAK_MULTI_SPEAKER_TRANSLATION_TEMPLATE,
            prefix=[],
            prompt=['<|im_start|>user\n{{QUERY}}<|im_end|>\n<|im_start|>assistant\n'],
            chat_sep=['<|im_end|>\n'],
            suffix=['<|im_end|>'],
            system_prefix=['<|im_start|>system\n{{SYSTEM}}<|im_end|>\n'],
            default_system='You are a helpful assistant.',
            stop_words=['<|endoftext|>'],
            agent_template='hermes',
            template_cls=QwenOmniSpeakMultiSpeakerTranslationTemplate,
        ),
        exist_ok=True,
    )
    logger.info('Registered multi-speaker-translation template for qwen_omnispeak.')


# ---------------------------------------------------------------------------
# joint-stage1 template (Stage-3 SFT joint translation task).
# ---------------------------------------------------------------------------


class QwenOmniSpeakJointStage1Template(QwenOmniSpeakSimultaneousPolicyTemplate):  # pyright: ignore[reportGeneralTypeIssues]
    """Stage-3 joint-stage1 template for joint streaming translation SFT.

    It reuses the shared codec-only streaming target construction used by the
    normal / bilingual / multi-speaker SFT tasks and isolates runtime switches
    under the ``JOINT_STAGE1_*`` environment-variable prefix.
    """

    env_prefix: str = 'JOINT_STAGE1'


def register_joint_stage1_template() -> None:
    register_template(
        TemplateMeta(
            template_type=QWEN_OMNISPEAK_JOINT_STAGE1_TEMPLATE,
            prefix=[],
            prompt=['<|im_start|>user\n{{QUERY}}<|im_end|>\n<|im_start|>assistant\n'],
            chat_sep=['<|im_end|>\n'],
            suffix=['<|im_end|>'],
            system_prefix=['<|im_start|>system\n{{SYSTEM}}<|im_end|>\n'],
            default_system='You are a helpful assistant.',
            stop_words=['<|endoftext|>'],
            agent_template='hermes',
            template_cls=QwenOmniSpeakJointStage1Template,
        ),
        exist_ok=True,
    )
    logger.info('Registered joint-stage1 template for qwen_omnispeak.')


# Backwards-compatible class aliases for older imports.
Qwen3OmniSpeakBridgeAlignmentTemplate = QwenOmniSpeakBridgeAlignmentTemplate
Qwen3OmniSpeakContinuePretrainTemplate = QwenOmniSpeakContinuePretrainTemplate

__all__ = [
    'QwenOmniSpeakBridgeAlignmentTemplate',
    'QwenOmniSpeakContinuePretrainTemplate',
    'QwenOmniSpeakSimultaneousPolicyTemplate',
    'QwenOmniSpeakNormalTranslationTemplate',
    'QwenOmniSpeakBilingualTranslationTemplate',
    'QwenOmniSpeakMultiSpeakerTranslationTemplate',
    'Qwen3OmniSpeakBridgeAlignmentTemplate',
    'Qwen3OmniSpeakContinuePretrainTemplate',
    'register_bridge_alignment_template',
    'register_continue_pretrain_template',
    'register_simultaneous_policy_template',
    'register_normal_translation_template',
    'register_bilingual_translation_template',
    'register_multi_speaker_translation_template',
    'QWEN_OMNISPEAK_CONTINUE_PRETRAIN_TEMPLATE',
    'QWEN_OMNISPEAK_SIMULTANEOUS_POLICY_TEMPLATE',
    'QWEN_OMNISPEAK_NORMAL_TRANSLATION_TEMPLATE',
    'QWEN_OMNISPEAK_BILINGUAL_TRANSLATION_TEMPLATE',
    'QWEN_OMNISPEAK_MULTI_SPEAKER_TRANSLATION_TEMPLATE',
    'QWEN_OMNISPEAK_JOINT_STAGE1_TEMPLATE',
]

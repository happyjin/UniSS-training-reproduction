# Copyright (c) ModelScope Contributors. All rights reserved.
"""Dataset registration for Qwen-OmniSpeak training pipelines.

This module hosts two distinct preprocessor families that share the same
JSONL field schema:

* ``BridgeAlignmentPreprocessor`` — original bridge-alignment (Stage 1)
  preprocessor. Every row must carry ``target_audio`` or precomputed codec
  codes and is intended for pure speech-generation supervision. It supports
  both speech-input and text-input translation tasks.
* ``ContinuePretrainPreprocessor`` — continue-pretrain preprocessor used
  by ``liveinterpret_training_pipelines/pipelines/continue_pretrain``.
  Supports three task families:

    - ``speech_to_text_translation``: text-only supervision (only thinker
      text outputs are trained).
    - ``multi_speaker_segmentation``: text-only supervision.
    - ``speech_to_speech_translation``: end-to-end thinker + talker
      supervision. ``target_audio`` is required and ``target_text`` is
      appended as the assistant text turn.
"""
import copy
import os
from typing import Any, Dict, List, Optional

from swift.dataset import DatasetMeta, RowPreprocessor, register_dataset
from swift.utils import get_logger

try:
    from .codec_tokenizer import _resolve_audio_path
    from .wds_audio import (
        is_lance_uri,
        resolve_lance_audio_batch,
        resolve_lance_codec_codes,
        resolve_lance_codec_codes_batch,
        resolve_lance_source_features,
        resolve_lance_source_features_batch,
        resolve_lance_speaker_embedding,
        resolve_lance_speaker_embedding_batch,
    )  # Step 2/3: lance 源
    from .lance_lazy_batch_patch import lance_dataloader_batch_resolve_enabled
    from .task_prompts import merge_task_prompt, s2s_task_prompt, set_system_prompt
except ImportError:  # pragma: no cover - imported as a top-level module by ms-swift.
    from codec_tokenizer import _resolve_audio_path  # pyright: ignore[reportMissingImports]
    from wds_audio import (  # pyright: ignore[reportMissingImports]
        is_lance_uri,
        resolve_lance_audio_batch,
        resolve_lance_codec_codes,
        resolve_lance_codec_codes_batch,
        resolve_lance_source_features,
        resolve_lance_source_features_batch,
        resolve_lance_speaker_embedding,
        resolve_lance_speaker_embedding_batch,
    )
    from lance_lazy_batch_patch import lance_dataloader_batch_resolve_enabled
    from task_prompts import (  # pyright: ignore[reportMissingImports]
        merge_task_prompt,
        s2s_task_prompt,
        set_system_prompt,
    )

logger = get_logger()

# Repo root: <repo>/liveinterpret_training_pipelines/custom/dataset.py
#                                  ^this file's parent's parent's parent
_PIPELINES_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, 'pipelines'))

# ---------------------------------------------------------------------------
# Preprocessor code version — makes `datasets` map-cache invalidate on edits
# ---------------------------------------------------------------------------
# WHY THIS EXISTS (root-caused 2026-08-04, job ev6i crashed on it):
#   swift calls ``dataset.map(self.batched_preprocess, ...)``
#   (swift/dataset/preprocessor/core.py:356). `datasets` fingerprints that
#   callable with ``Hasher.hash(bound_method)``, which delegates to dill.
#   For a class living in an *importable module* (which we are — swift loads
#   us via ``--external_plugins``), dill serialises the method **by
#   reference** (module + qualname), NOT by source. Verified empirically:
#   editing ``preprocess`` in an imported module leaves the hash byte-identical
#   (38c9a2d0062d42f5 before and after adding a new output column).
#   Consequence: with ``load_from_cache_file=true`` a code change is silently
#   ignored and a STALE arrow cache is reused. That is exactly how the
#   `lengths` column went missing and the sampler died with
#   ``KeyError: Column lengths not in the dataset``.
#
#   dill *does* serialise ``self.__dict__`` though, so an instance attribute
#   derived from the source content propagates into the fingerprint (verified:
#   103315a80a53edf2 -> ef360dbf9c5a8dcc when the attribute changed). Each
#   preprocessor therefore stamps ``self._preprocess_code_version`` in
#   ``__init__``.
#
#   The digest covers this file plus the modules whose functions are called
#   from inside ``preprocess`` — editing any of them must re-run the map.
#   Cost of a false invalidation is one map pass (~4 min for 19M rows);
#   cost of a missed one is a silently wrong dataset, so we err on re-running.
_PREPROCESS_CODE_DEPS = ('dataset.py', 'task_prompts.py', 'codec_tokenizer.py', 'wds_audio.py')


def _compute_preprocess_code_version() -> str:
    import hashlib
    here = os.path.dirname(os.path.abspath(__file__))
    digest = hashlib.sha256()
    for name in _PREPROCESS_CODE_DEPS:
        digest.update(name.encode('utf-8'))
        try:
            with open(os.path.join(here, name), 'rb') as fp:
                digest.update(fp.read())
        except OSError:
            # A dep we cannot read must not silently collapse to a constant
            # hash shared with other unreadable states.
            digest.update(b'<unreadable>')
    return digest.hexdigest()[:16]


_PREPROCESS_CODE_VERSION = _compute_preprocess_code_version()

# ---------------------------------------------------------------------------
# Shared task identifiers
# ---------------------------------------------------------------------------

# Bridge alignment tasks (Stage 1).
TASK_TTS = 'tts'
TASK_S2S_CONVERSATION = 'speech_to_speech_conversation'
TASK_S2S_TRANSLATION = 'speech_to_speech_translation'
TASK_TEXT_TO_SPEECH_TRANSLATION = 'text_to_speech_translation'
TASK_ALL = 'all'
SUPPORTED_TASKS = {
    TASK_TTS,
    TASK_S2S_CONVERSATION,
    TASK_S2S_TRANSLATION,
    TASK_TEXT_TO_SPEECH_TRANSLATION,
}

# Continue pretrain tasks.
TASK_S2T_TRANSLATION = 'speech_to_text_translation'
TASK_MULTI_SPEAKER_SEG = 'multi_speaker_segmentation'
CONTINUE_PRETRAIN_TASKS = {TASK_S2T_TRANSLATION, TASK_MULTI_SPEAKER_SEG, TASK_S2S_TRANSLATION}

# Stage-3 SFT: simultaneous (read-write) policy learning.
TASK_SIMUL_POLICY = 'simultaneous_policy_learning'
SIMULTANEOUS_POLICY_TASKS = {TASK_SIMUL_POLICY}

# Stage-3 SFT: normal_translation (streaming speech-to-speech translation).
# The shipped data files carry ``task='speech2speech_translation'`` (note: this
# is a *different* string from continue-pretrain's ``speech_to_speech_translation``
# with underscores); we also accept the explicit ``normal_translation`` value.
TASK_NORMAL_TRANSLATION = 'normal_translation'
TASK_S2S_TRANSLATION_COMPACT = 'speech2speech_translation'
NORMAL_TRANSLATION_TASKS = {TASK_NORMAL_TRANSLATION, TASK_S2S_TRANSLATION_COMPACT}

# Stage-3 SFT: bilingual_translation (streaming bilingual mutual translation).
TASK_BILINGUAL_TRANSLATION = 'bilingual_translation'
TASK_BILINGUAL_MUTUAL = 'bilingual_mutual_translation'
BILINGUAL_TRANSLATION_TASKS = {TASK_BILINGUAL_TRANSLATION, TASK_BILINGUAL_MUTUAL}

# Stage-3 SFT: multi_speaker_translation (streaming multi-speaker s2s translation).
TASK_MULTI_SPEAKER_TRANSLATION = 'multi_speaker_translation'
MULTI_SPEAKER_TRANSLATION_TASKS = {TASK_MULTI_SPEAKER_TRANSLATION}

# Stage-3 SFT: joint_stage1 (combined normal + bilingual + multi-speaker translation).
SFT_JOINT_STAGE1_TASKS = (
    NORMAL_TRANSLATION_TASKS
    | BILINGUAL_TRANSLATION_TASKS
    | MULTI_SPEAKER_TRANSLATION_TASKS
)

EXTRA_KEYS = [
    'schema_version',
    'task',
    'bridge_task',
    'task_prompt',
    'target_text',
    'target_audio',
    'target_codec_codes',
    'target_audios',
    'alignment',
    'source_langs',
    'source_lang',
    'target_lang',
    'speaker_profile',
    'reference_audios',
    'speaker_spans',
    'latency_budget_ms',
    # ICL mode (Qwen3-TTS style): reference (source) codec + source text ids.
    'source_codec_codes',
    'source_codec_len',
    'source_text_ids',
    # Precomputed Whisper mel from Lance (lance://<_rowid>/features → dict).
    'source_input_features',
    # Precomputed ECAPA x-vector (lance://<_rowid>/speaker → list[2048]).
    'source_speaker_embedding',
    # Master/zeyang name for the same offline x-vector ([D] float).
    # Kept so mixed manifests (parquet + Lance) share EXTRA_KEYS defaults.
    'source_xvector',
    # Cheap length estimate for --group_by_length (Lance: duration_ms → this).
    'source_duration_ms',
    'target_duration_ms',
]


def default_data_path() -> str:
    return os.path.abspath(
        os.environ.get(
            'BRIDGE_ALIGNMENT_DATA',
            os.path.join(_PIPELINES_ROOT, 'bridge_alignment', 'data_format.jsonl')))


def default_continue_pretrain_data_path() -> str:
    return os.path.abspath(
        os.environ.get(
            'CONTINUE_PRETRAIN_DATA',
            os.path.join(_PIPELINES_ROOT, 'continue_pretrain', 'data_format.jsonl')))


def default_simultaneous_policy_data_path() -> str:
    return os.path.abspath(
        os.environ.get(
            'SIMULTANEOUS_POLICY_DATA',
            os.path.join(_PIPELINES_ROOT, 'sft', 'data_format.jsonl')))


def default_normal_translation_data_path() -> str:
    return os.path.abspath(
        os.environ.get(
            'NORMAL_TRANSLATION_DATA',
            os.path.join(_PIPELINES_ROOT, 'sft', 'normal_translation_data_format.jsonl')))


def default_bilingual_translation_data_path() -> str:
    return os.path.abspath(
        os.environ.get(
            'BILINGUAL_TRANSLATION_DATA',
            os.path.join(_PIPELINES_ROOT, 'sft', 'bilingual_translation_data_format.jsonl')))


def default_multi_speaker_translation_data_path() -> str:
    return os.path.abspath(
        os.environ.get(
            'MULTI_SPEAKER_TRANSLATION_DATA',
            os.path.join(_PIPELINES_ROOT, 'sft', 'multi_speaker_translation_data_format.jsonl')))


def default_sft_joint_stage1_data_path() -> str:
    return os.path.abspath(
        os.environ.get(
            'JOINT_STAGE1_DATA',
            os.path.join(_PIPELINES_ROOT, 'sft', 'joint_stage1_data_format.jsonl')))


def _as_bool_env(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in {'1', 'true', 'yes', 'y', 'on'}


def _merge_task_prompt(messages, task_prompt):
    """Backward-compatible alias for the shared idempotent prompt merger."""
    return merge_task_prompt(messages, task_prompt)


def _default_extra_field(key: str) -> Any:
    """Return type-stable defaults for optional columns in mixed-task JSONL."""
    if key == 'speaker_profile':
        return {'speaker_id': None, 'gender': None, 'style': None, 'pitch': None}
    if key in {'reference_audios', 'speaker_spans', 'target_audios', 'alignment',
               # list-typed columns: keep empty list so pyarrow doesn't see a
               # mix of list and '' (str) in the same column when some rows are
               # missing the value (e.g. source_xvector None for short clips).
               'source_codec_codes', 'source_text_ids',
               'source_speaker_embedding', 'source_xvector'}:
        return []
    if key == 'latency_budget_ms':
        return 0
    return ''


def _copy_extra_fields(src: Dict[str, Any], dst: Dict[str, Any]) -> None:
    for key in EXTRA_KEYS:
        if key in dst:
            continue
        if key == 'target_codec_codes':
            # DataLoader batch 模式保留指针，在 __getitems__ 中与 mel/speaker
            # 联合 take；关闭该模式时沿用 map/逐行解引用。
            v = src.get('target_codec_codes')
            if is_lance_uri(v):
                if lance_dataloader_batch_resolve_enabled():
                    dst['target_codec_codes'] = v
                else:
                    dst['target_codec_codes'] = resolve_lance_codec_codes(v)
            else:
                dst['target_codec_codes'] = v if v is not None else _default_extra_field(key)
            continue
        if key == 'source_input_features':
            v = src.get('source_input_features')
            # Keep lance:// pointer lazy unless LANCE_PREFETCH_FEATURES=1
            # (mel arrays are large; resolving in map blows HF cache / runtime).
            if is_lance_uri(v):
                raw = os.environ.get('LANCE_PREFETCH_FEATURES', '0').strip().lower()
                if (not lance_dataloader_batch_resolve_enabled()
                        and raw in {'1', 'true', 'yes', 'y', 'on'}):
                    dst['source_input_features'] = resolve_lance_source_features(v)
                else:
                    dst['source_input_features'] = v
            elif isinstance(v, dict) and 'flat' in v:
                dst['source_input_features'] = v
            else:
                dst['source_input_features'] = v if v is not None else _default_extra_field(key)
            continue
        if key == 'source_speaker_embedding':
            v = src.get('source_speaker_embedding')
            if is_lance_uri(v):
                # 2048 floats — resolve eagerly (tiny vs mel). Prefer map-stage
                # batch prefetch when available; this is the per-row fallback.
                if lance_dataloader_batch_resolve_enabled():
                    dst['source_speaker_embedding'] = v
                else:
                    dst['source_speaker_embedding'] = resolve_lance_speaker_embedding(v)
            elif isinstance(v, (list, tuple)) and v:
                dst['source_speaker_embedding'] = list(v)
            else:
                try:
                    import numpy as _np
                    if isinstance(v, _np.ndarray) and v.size > 0:
                        dst['source_speaker_embedding'] = (
                            v.astype(_np.float32, copy=False).reshape(-1))
                        continue
                except Exception:
                    pass
                dst['source_speaker_embedding'] = (
                    v if v is not None else _default_extra_field(key))
            continue
        if key in src and src[key] is not None:
            dst[key] = copy.deepcopy(src[key])
        else:
            dst[key] = _default_extra_field(key)


def _prefetch_lance_codec_codes_in_batch(rows: Any) -> None:
    """Batch-resolve ``lance://…/codes`` in a map batch (in-place).

    Accepts HF columnar ``{col: [v0, v1, …]}`` or ``list[dict]``.
    Values are identical to calling ``resolve_lance_codec_codes`` per row;
    only the Lance ``take_by_stable_rowid`` is coalesced. No-op when the batch
    has no lance:// codec refs (legacy wds / path rows unchanged).
    """
    if rows is None:
        return
    # Columnar HF batch
    if isinstance(rows, dict) and 'target_codec_codes' in rows:
        col = rows.get('target_codec_codes')
        if isinstance(col, list) and any(is_lance_uri(v) for v in col):
            rows['target_codec_codes'] = resolve_lance_codec_codes_batch(col)
        return
    # list[dict]
    if isinstance(rows, list) and rows and isinstance(rows[0], dict):
        uris = [r.get('target_codec_codes') for r in rows]
        if any(is_lance_uri(u) for u in uris):
            resolved = resolve_lance_codec_codes_batch(uris)
            for r, codes in zip(rows, resolved):
                r['target_codec_codes'] = codes


def _lance_prefetch_audio_enabled() -> bool:
    """Whether map-stage should materialise ``lance://`` audios into bytes.

    Default on (mirrors codes batch prefetch). Set ``LANCE_PREFETCH_AUDIO=0``
    to keep pointer rows light and resolve at template encode time instead
    (lower HF-cache size, more per-sample Lance takes when lazy).

    Mel-mode rows use ``audios=[]`` + ``source_input_features``; there is nothing
    useful to prefetch as wav bytes. Force off when the launch env still sets
    ``LANCE_USE_PRECOMPUTED_MEL=1`` (table has no ``source_audio_bytes``).
    """
    raw_mel = os.environ.get('LANCE_USE_PRECOMPUTED_MEL', '0').strip().lower()
    if raw_mel in {'1', 'true', 'yes', 'y', 'on'}:
        return False
    raw = os.environ.get('LANCE_PREFETCH_AUDIO', '1').strip().lower()
    return raw not in {'0', 'false', 'off', 'no'}


def _row_has_precomputed_mel(row: Dict[str, Any]) -> bool:
    """True when the sample carries self-describing source mel (URI or packed)."""
    v = row.get('source_input_features')
    if v is None or v == '' or v == []:
        return False
    if is_lance_uri(v):
        return True
    if isinstance(v, dict) and 'flat' in v and 'frames' in v:
        return True
    if isinstance(v, list) and v:
        return True
    return False


def _prefetch_lance_source_features_in_batch(rows: Any) -> None:
    """Batch-resolve ``lance://…/features`` in ``source_input_features`` (in-place).

    Gated by ``LANCE_PREFETCH_FEATURES`` (default **off**). Mel arrays are large;
    prefetching them into the HF map cache for full-scale datasets is usually
    worse than resolving at template encode time (mirrors lazy audio).
    """
    if rows is None:
        return
    raw = os.environ.get('LANCE_PREFETCH_FEATURES', '0').strip().lower()
    if raw not in {'1', 'true', 'yes', 'y', 'on'}:
        return
    if isinstance(rows, dict) and 'source_input_features' in rows:
        col = rows.get('source_input_features')
        if isinstance(col, list) and any(is_lance_uri(v) for v in col):
            rows['source_input_features'] = resolve_lance_source_features_batch(col)
        return
    if isinstance(rows, list) and rows and isinstance(rows[0], dict):
        uris = [r.get('source_input_features') for r in rows]
        if any(is_lance_uri(u) for u in uris):
            resolved = resolve_lance_source_features_batch(uris)
            for r, feat in zip(rows, resolved):
                r['source_input_features'] = feat


def _prefetch_lance_speaker_embedding_in_batch(rows: Any) -> None:
    """Batch-resolve ``lance://…/speaker`` in ``source_speaker_embedding`` (in-place).

    Gated by ``LANCE_PREFETCH_SPEAKER`` (default **on**). A 2048-d float vector is
    tiny vs mel; coalescing map-batch takes avoids N per-row Lance round-trips.
    """
    if rows is None:
        return
    raw = os.environ.get('LANCE_PREFETCH_SPEAKER', '1').strip().lower()
    if raw in {'0', 'false', 'off', 'no'}:
        return
    if isinstance(rows, dict) and 'source_speaker_embedding' in rows:
        col = rows.get('source_speaker_embedding')
        if isinstance(col, list) and any(is_lance_uri(v) for v in col):
            rows['source_speaker_embedding'] = resolve_lance_speaker_embedding_batch(col)
        return
    if isinstance(rows, list) and rows and isinstance(rows[0], dict):
        uris = [r.get('source_speaker_embedding') for r in rows]
        if any(is_lance_uri(u) for u in uris):
            resolved = resolve_lance_speaker_embedding_batch(uris)
            for r, emb in zip(rows, resolved):
                r['source_speaker_embedding'] = emb


def _flatten_audio_lists(audio_lists: List[Any]):
    """Flatten ``[[uri|bytes|…], …]`` → (flat_uris_or_vals, rebuild_spec).

    ``rebuild_spec`` is a list of per-row lengths so the flat resolve result
    can be sliced back into the original nested shape.
    """
    flat: List[Any] = []
    lengths: List[int] = []
    for cell in audio_lists:
        if cell is None:
            lengths.append(0)
            continue
        if isinstance(cell, (list, tuple)):
            items = list(cell)
        else:
            # Rare: a bare string/bytes in the audios column.
            items = [cell]
        lengths.append(len(items))
        flat.extend(items)
    return flat, lengths


def _unflatten_audio_lists(flat: List[Any], lengths: List[int]) -> List[List[Any]]:
    out: List[List[Any]] = []
    idx = 0
    for n in lengths:
        out.append(flat[idx:idx + n])
        idx += n
    return out


def _prefetch_lance_audios_in_batch(rows: Any) -> None:
    """Batch-resolve ``lance://…/source|target`` in ``audios`` (in-place).

    Same coalescing idea as :func:`_prefetch_lance_codec_codes_in_batch`, but
    ``audios`` is a *list per row*. All lance URIs across the map batch are
    gathered into one (or two, by side) ``take_by_stable_rowid`` call(s).

    Gated by ``LANCE_PREFETCH_AUDIO`` (default on). No-op for batches with no
    lance:// audio refs.
    """
    if rows is None or not _lance_prefetch_audio_enabled():
        return

    def _resolve_nested(audio_lists: List[Any]) -> Optional[List[List[Any]]]:
        flat, lengths = _flatten_audio_lists(audio_lists)
        if not any(is_lance_uri(v) for v in flat):
            return None
        # Only coalesce the lance entries; leave wds/path/bytes untouched in flat.
        lance_pos = [i for i, v in enumerate(flat) if is_lance_uri(v)]
        resolved = resolve_lance_audio_batch([flat[i] for i in lance_pos])
        for i, payload in zip(lance_pos, resolved):
            flat[i] = payload
        return _unflatten_audio_lists(flat, lengths)

    if isinstance(rows, dict) and 'audios' in rows:
        col = rows.get('audios')
        if isinstance(col, list) and col:
            nested = _resolve_nested(col)
            if nested is not None:
                rows['audios'] = nested
        return

    if isinstance(rows, list) and rows and isinstance(rows[0], dict):
        audio_lists = [r.get('audios') for r in rows]
        nested = _resolve_nested(audio_lists)
        if nested is not None:
            for r, audios in zip(rows, nested):
                r['audios'] = audios


class _LanceBatchResolveMixin:
    """Coalesce Lance payload takes inside ``datasets.map`` when requested.

    ms-swift calls ``dataset.map(self.batched_preprocess, …)``. Prefetching
    here does **not** change which bytes/codes a row gets — only how many Lance
    round-trips the map stage makes. With DataLoader batch resolution enabled,
    features/speaker/codes stay as pointers and only raw audio uses this stage.
    Sampler / loss / labels stay untouched.
    """

    def batched_preprocess(self, rows, **kwargs):
        if not lance_dataloader_batch_resolve_enabled():
            _prefetch_lance_codec_codes_in_batch(rows)
            _prefetch_lance_source_features_in_batch(rows)
            _prefetch_lance_speaker_embedding_in_batch(rows)
        _prefetch_lance_audios_in_batch(rows)
        return super().batched_preprocess(rows, **kwargs)  # type: ignore[misc]


# Back-compat alias (older name only covered codes).
_LanceBatchCodesMixin = _LanceBatchResolveMixin


# ---------------------------------------------------------------------------
# Bridge alignment preprocessor (Stage 1) — kept unchanged.
# ---------------------------------------------------------------------------


class BridgeAlignmentPreprocessor(_LanceBatchCodesMixin, RowPreprocessor):
    """Filter bridge alignment JSONL rows and keep target-audio supervision fields."""

    def __init__(self, task_filter: str = TASK_ALL, **kwargs) -> None:
        super().__init__(**kwargs)
        # Participates in the datasets map-cache fingerprint; see
        # _PREPROCESS_CODE_VERSION for why this is required.
        self._preprocess_code_version = _PREPROCESS_CODE_VERSION
        if task_filter != TASK_ALL and task_filter not in SUPPORTED_TASKS:
            raise ValueError(f'Unsupported task_filter={task_filter!r}; expected one of {sorted(SUPPORTED_TASKS)} or all')
        self.task_filter = task_filter
        self.append_target_text = _as_bool_env('BRIDGE_ALIGNMENT_APPEND_TARGET_TEXT', True)
        # New WDS manifests intentionally stay immutable.  This opt-in lets a
        # dedicated training entry canonicalize their legacy short prompt at
        # mapping time without changing existing standalone-data runs.
        self.normalize_s2s_prompt = _as_bool_env(
            'BRIDGE_ALIGNMENT_NORMALIZE_S2S_PROMPT', False)

    def preprocess(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        task = row.get('task')
        if task not in SUPPORTED_TASKS:
            return None
        if self.task_filter != TASK_ALL and task != self.task_filter:
            return None
        # Mel vs raw is self-describing via ``source_input_features`` on the row
        # (see custom/lance_manifest.py). Do not rewrite audios from env here.
        if drop_row_if_invalid_audio(row):
            return None
        target_audio = row.get('target_audio')
        target_codec_codes = row.get('target_codec_codes')
        if not target_audio and not target_codec_codes:
            raise ValueError(
                'Bridge alignment row requires target_audio or target_codec_codes.')

        messages = copy.deepcopy(row.get('messages') or [])
        if task == TASK_S2S_TRANSLATION and self.normalize_s2s_prompt:
            messages = set_system_prompt(
                messages, s2s_task_prompt(row.get('target_lang')))
        else:
            messages = _merge_task_prompt(messages, row.get('task_prompt'))

        target_text = row.get('target_text')
        if self.append_target_text and target_text:
            if not messages or messages[-1].get('role') != 'assistant':
                messages.append({'role': 'assistant', 'content': str(target_text)})

        if not messages:
            raise ValueError('Bridge alignment row requires messages or target_text.')

        result = {
            'messages': messages,
            'audios': copy.deepcopy(row.get('audios') or []),
            'chat_template_kwargs': copy.deepcopy(row.get('chat_template_kwargs') or {}),
            'bridge_task': task,
        }
        _copy_extra_fields(row, result)
        _attach_lengths(row, result)
        if task == TASK_S2S_TRANSLATION and self.normalize_s2s_prompt:
            # Keep the canonical prompt in one place.  Otherwise the legacy
            # short compatibility column survives into the Arrow map cache.
            result['task_prompt'] = ''
        return result


def register_bridge_alignment_datasets(dataset_path: Optional[str] = None) -> None:
    dataset_path = os.path.abspath(dataset_path or default_data_path())
    for name, task_filter in [
        ('bridge_alignment_tts', TASK_TTS),
        ('bridge_alignment_speech_to_speech_conversation', TASK_S2S_CONVERSATION),
        ('bridge_alignment_speech_to_speech_translation', TASK_S2S_TRANSLATION),
        ('bridge_alignment_text_to_speech_translation', TASK_TEXT_TO_SPEECH_TRANSLATION),
        ('bridge_alignment_all', TASK_ALL),
    ]:
        register_dataset(
            DatasetMeta(
                # ``dataset_name`` (string) is what newer ms-swift uses as the
                # primary lookup key; ``ms_dataset_id`` is also set so the
                # legacy tuple-key path still works.
                dataset_name=name,
                ms_dataset_id=name,
                dataset_path=dataset_path,
                preprocess_func=BridgeAlignmentPreprocessor(task_filter=task_filter),
                tags=['audio', 'speech-generation', 'qwen3-omnispeak', 'bridge-alignment'],
            ),
            exist_ok=True,
        )


# ---------------------------------------------------------------------------
# Continue pretrain preprocessor.
# ---------------------------------------------------------------------------


class ContinuePretrainPreprocessor(_LanceBatchCodesMixin, RowPreprocessor):
    """Continue-pretrain preprocessor.

    Supports speech-to-text translation, multi-speaker segmentation, and
    speech-to-speech translation in a single JSONL. ``task_filter`` selects
    a subset of rows. For ``speech_to_speech_translation`` rows we require
    ``target_audio`` to enable codec supervision; for the other two tasks
    only the assistant text turn is required.
    """

    def __init__(self, task_filter: str = TASK_ALL, **kwargs) -> None:
        super().__init__(**kwargs)
        # Participates in the datasets map-cache fingerprint; see
        # _PREPROCESS_CODE_VERSION for why this is required.
        self._preprocess_code_version = _PREPROCESS_CODE_VERSION
        if task_filter != TASK_ALL and task_filter not in CONTINUE_PRETRAIN_TASKS:
            raise ValueError(
                f'Unsupported task_filter={task_filter!r}; '
                f'expected one of {sorted(CONTINUE_PRETRAIN_TASKS)} or "all".')
        self.task_filter = task_filter
        self.append_target_text = _as_bool_env('CONTINUE_PRETRAIN_APPEND_TARGET_TEXT', True)
        self.strict = _as_bool_env('CONTINUE_PRETRAIN_STRICT', False)

    def _is_text_task(self, task: str) -> bool:
        return task in (TASK_S2T_TRANSLATION, TASK_MULTI_SPEAKER_SEG)

    def preprocess(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        task = row.get('task')
        if task not in CONTINUE_PRETRAIN_TASKS:
            return None
        if self.task_filter != TASK_ALL and task != self.task_filter:
            return None
        if drop_row_if_invalid_audio(row):
            return None

        messages = copy.deepcopy(row.get('messages') or [])
        messages = _merge_task_prompt(messages, row.get('task_prompt'))

        if task == TASK_S2S_TRANSLATION:
            target_audio = row.get('target_audio')
            target_codec_codes = row.get('target_codec_codes')
            if not target_audio and not target_codec_codes:
                if self.strict:
                    raise ValueError(
                        'speech_to_speech_translation row requires target_audio or target_codec_codes.')
                return None
            target_text = row.get('target_text')
            if self.append_target_text and target_text:
                if not messages or messages[-1].get('role') != 'assistant':
                    messages.append({'role': 'assistant', 'content': str(target_text)})
        else:
            # Pure text-supervision tasks must already include an assistant turn.
            if not messages or messages[-1].get('role') != 'assistant':
                if self.strict:
                    raise ValueError(
                        f'{task} row requires an assistant message; got messages={messages!r}.')
                return None

        if not messages:
            return None

        result = {
            'messages': messages,
            'audios': copy.deepcopy(row.get('audios') or []),
            'chat_template_kwargs': copy.deepcopy(row.get('chat_template_kwargs') or {}),
            'bridge_task': task,
        }
        _copy_extra_fields(row, result)
        _attach_lengths(row, result)
        return result


def register_continue_pretrain_datasets(dataset_path: Optional[str] = None) -> None:
    dataset_path = os.path.abspath(dataset_path or default_continue_pretrain_data_path())
    for name, task_filter in [
        ('continue_pretrain_speech_to_text_translation', TASK_S2T_TRANSLATION),
        ('continue_pretrain_multi_speaker_segmentation', TASK_MULTI_SPEAKER_SEG),
        ('continue_pretrain_speech_to_speech_translation', TASK_S2S_TRANSLATION),
        ('continue_pretrain_all', TASK_ALL),
    ]:
        register_dataset(
            DatasetMeta(
                dataset_name=name,
                ms_dataset_id=name,
                dataset_path=dataset_path,
                preprocess_func=ContinuePretrainPreprocessor(task_filter=task_filter),
                tags=['audio', 'speech-generation', 'qwen3-omnispeak', 'continue-pretrain'],
            ),
            exist_ok=True,
        )


# ---------------------------------------------------------------------------
# Simultaneous (read-write) policy learning preprocessor (Stage 3 SFT).
# ---------------------------------------------------------------------------


class SimultaneousPolicyPreprocessor(_LanceBatchCodesMixin, RowPreprocessor):
    """Preprocessor for the ``simultaneous_policy_learning`` task.

    Each row carries a chunk-level ``alignment`` (source/target timestamps per
    chunk) plus the per-chunk ``target_audios`` (chunk_0000.wav...). We keep the
    user ``<audio>`` turn only and forward ``alignment`` / ``target_audios`` so
    the template can (a) concatenate the per-chunk target codec stream and (b)
    derive a monotonic ``codec_to_sem_visible`` read-write schedule from the
    timestamps.

    The model does NOT predict target text in this Stage-3 task, so the assistant
    target-text turn is intentionally not appended (text supervision is removed:
    no thinker text CE / text token accuracy). Supervision is codec-only.

    All Stage-3 SFT subclasses share the same timeline validation here: source
    and target chunks must both be strictly non-overlapping, and the target
    timeline duration must satisfy the common source-duration tail bound.
    """

    # The set of ``task`` strings this preprocessor accepts. Subclasses override
    # it to support a different task family while reusing all of the logic below.
    task_set = SIMULTANEOUS_POLICY_TASKS
    # Environment-variable prefix used for switches (``<PREFIX>_STRICT`` ...).
    env_prefix = 'SIMULTANEOUS_POLICY'

    def __init__(self, task_filter: str = TASK_ALL, **kwargs) -> None:
        super().__init__(**kwargs)
        # Participates in the datasets map-cache fingerprint; see
        # _PREPROCESS_CODE_VERSION for why this is required.
        self._preprocess_code_version = _PREPROCESS_CODE_VERSION
        if task_filter != TASK_ALL and task_filter not in self.task_set:
            raise ValueError(
                f'Unsupported task_filter={task_filter!r}; '
                f'expected one of {sorted(self.task_set)} or "all".')
        self.task_filter = task_filter
        self.strict = _as_bool_env(f'{self.env_prefix}_STRICT', False)

    # Slack added on top of the source duration in the total-duration bound (ms).
    TAIL_SLACK_MS = 3000.0

    def _collect_target_audios(self, row: Dict[str, Any]):
        target_audios = row.get('target_audios')
        if target_audios:
            return list(target_audios)
        # Fall back to per-chunk target_audio entries inside alignment.
        alignment = row.get('alignment') or []
        derived = [seg.get('target_audio') for seg in alignment if seg.get('target_audio')]
        return derived

    def _reject(self, reason: str) -> bool:
        if self.strict:
            raise ValueError(f'{self.env_prefix.lower()} sample rejected: {reason}')
        logger.debug('Discarding %s sample: %s', self.env_prefix.lower(), reason)
        return False

    def _timeline_source_duration_ms(
        self,
        row: Dict[str, Any],
        alignment_sorted: List[Dict[str, Any]],
    ) -> Optional[float]:
        if _as_bool_env(f'{self.env_prefix}_SOURCE_DURATION_FROM_FILE', True):
            audios = row.get('audios') or []
            if audios:
                first = audios[0]
                path = first.get('path') if isinstance(first, dict) else first
                duration = _read_audio_duration_ms(path)
                if duration is not None:
                    return duration
        src_ends = self._timeline_values(alignment_sorted, 'source_end_ms')
        if src_ends:
            return float(max(src_ends))
        return None

    def _timeline_values(self, alignment_sorted: List[Dict[str, Any]], field: str) -> Optional[List[float]]:
        values = []
        for seg in alignment_sorted:
            value = seg.get(field)
            if value is None:
                return None
            try:
                values.append(float(value))
            except (TypeError, ValueError):
                return None
        return values

    def _passes_timeline_validation(self, alignment_sorted: List[Dict[str, Any]], row: Dict[str, Any]) -> bool:
        # No alignment -> let the existing required-field check decide.
        if not alignment_sorted:
            return True

        source_starts = self._timeline_values(alignment_sorted, 'source_start_ms')
        source_ends = self._timeline_values(alignment_sorted, 'source_end_ms')
        target_starts = self._timeline_values(alignment_sorted, 'target_start_ms')
        target_ends = self._timeline_values(alignment_sorted, 'target_end_ms')
        if source_starts is None or source_ends is None:
            return self._reject('missing/invalid source_start_ms/source_end_ms in alignment.')
        if target_starts is None or target_ends is None:
            return self._reject('missing/invalid target_start_ms/target_end_ms in alignment.')

        # Source and target chunk timelines must both be strictly non-overlapping.
        for i in range(1, len(alignment_sorted)):
            if not (source_ends[i - 1] < source_starts[i]):
                return self._reject(
                    f'source chunk {i - 1} end ({source_ends[i - 1]}ms) >= '
                    f'source chunk {i} start ({source_starts[i]}ms).')
            if not (target_ends[i - 1] < target_starts[i]):
                return self._reject(
                    f'target chunk {i - 1} end ({target_ends[i - 1]}ms) >= '
                    f'target chunk {i} start ({target_starts[i]}ms).')

        # Total target timeline duration must stay within the source + tail bound.
        total_target = float(max(target_ends))
        source_duration_ms = self._timeline_source_duration_ms(row, alignment_sorted)
        if source_duration_ms is None:
            return self._reject('could not determine source audio duration.')
        last_src_start = source_starts[-1]
        last_src_end = source_ends[-1]
        bound = source_duration_ms + 0.5 * (last_src_end - last_src_start) + self.TAIL_SLACK_MS
        if not (total_target < bound):
            return self._reject(
                f'total target duration ({total_target}ms) >= bound ({bound:.1f}ms).')
        return True

    def preprocess(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        task = row.get('task')
        if task not in self.task_set:
            return None
        if self.task_filter != TASK_ALL and task != self.task_filter:
            return None
        if drop_row_if_invalid_audio(row):
            return None

        alignment = row.get('alignment') or []
        target_audios = self._collect_target_audios(row)
        if not alignment or not target_audios:
            if self.strict:
                raise ValueError(
                    f'{task} row requires alignment and target_audios '
                    f'(got alignment={len(alignment)} chunks, target_audios={len(target_audios)}).')
            return None

        alignment_sorted = sorted(alignment, key=lambda seg: seg.get('chunk_id', 0))
        if not self._passes_timeline_validation(alignment_sorted, row):
            return None

        messages = copy.deepcopy(row.get('messages') or [])
        messages = _merge_task_prompt(messages, row.get('task_prompt'))

        # NOTE: the assistant target-text turn is intentionally NOT appended.
        # This Stage-3 task does not predict text (codec-only supervision), so
        # there is no thinker text CE / text token accuracy.
        if not messages:
            if self.strict:
                raise ValueError(f'{task} row requires messages.')
            return None

        result = {
            'messages': messages,
            'audios': copy.deepcopy(row.get('audios') or []),
            'chat_template_kwargs': copy.deepcopy(row.get('chat_template_kwargs') or {}),
            'bridge_task': task,
        }
        # Normalise target_audios so the column is always populated even when the
        # row only carried per-chunk alignment[i].target_audio.
        row = dict(row)
        row['target_audios'] = target_audios
        _copy_extra_fields(row, result)
        _attach_lengths(row, result)
        return result


def register_simultaneous_policy_datasets(dataset_path: Optional[str] = None) -> None:
    dataset_path = os.path.abspath(dataset_path or default_simultaneous_policy_data_path())
    for name, task_filter in [
        ('simultaneous_policy_learning', TASK_SIMUL_POLICY),
        ('simultaneous_policy_all', TASK_ALL),
    ]:
        register_dataset(
            DatasetMeta(
                dataset_name=name,
                ms_dataset_id=name,
                dataset_path=dataset_path,
                preprocess_func=SimultaneousPolicyPreprocessor(task_filter=task_filter),
                tags=['audio', 'speech-generation', 'qwen3-omnispeak',
                      'simultaneous-interpretation', 'sft'],
            ),
            exist_ok=True,
        )


# ---------------------------------------------------------------------------
# normal_translation preprocessor (Stage-3 SFT, streaming s2s translation).
# ---------------------------------------------------------------------------


def _read_audio_duration_ms(path: str) -> Optional[float]:
    """Best-effort wav duration in ms via ``soundfile`` (covers trailing silence)."""
    if not isinstance(path, str) or not path:
        return None
    try:
        import soundfile as sf  # pyright: ignore[reportMissingImports]
        info = sf.info(_resolve_audio_path(path))
        if info.frames > 0 and info.samplerate > 0:
            return 1000.0 * info.frames / info.samplerate
    except Exception:
        return None
    return None


# ---------------------------------------------------------------------------
# Audio-duration filter: drop whole samples whose ANY audio exceeds the limit.
#
# This runs in the dataset *preprocessor* (dataset load time, BEFORE the
# dataloader). swift's RowPreprocessor.batched_preprocess converts a
# ``return None`` from ``preprocess`` into an empty row, which is then removed
# from the HF dataset entirely. That is the only place a sample can be truly
# DROPPED: swift always wraps the dataset in ``LazyLLMDataset`` (see
# ``SwiftSft._post_process_datasets``), whose ``__getitem__`` only *replaces* an
# errored sample with a random one and never removes it. So filtering inside
# template._encode cannot drop a sample; the drop must happen here, at load
# time.
# ---------------------------------------------------------------------------
def _max_audio_seconds() -> float:
    """Max allowed duration (seconds) for ANY single audio in a training sample.

    Tunable via ``OMNISPEAK_MAX_AUDIO_SECONDS`` (default 120). A sample whose
    target OR source audio exceeds this is dropped at load time. We deliberately
    do NOT filter inside the dataloader: once a full step is under way, returning
    ``None`` from the dataloader cannot cleanly bail out a step and only produces
    more errors.
    """
    return float(os.environ.get('OMNISPEAK_MAX_AUDIO_SECONDS', '120'))


# ---------------------------------------------------------------------------
# Pre-baked ``lengths`` column for ``--group_by_length``.
#
# WHY THIS EXISTS
# ms-swift only auto-enables ``lazy_tokenize`` for multimodal models when
# ``group_by_length`` is OFF (``base_args.py::_init_lazy_tokenize``). With
# ``group_by_length=true`` and ``lazy_tokenize`` left unset it silently flips to
# eager mode and runs ``AddLengthPreprocessor`` over the WHOLE dataset — a full
# audio-decode + codec-encode pass whose result is then THROWN AWAY, keeping
# only one integer per row. On the 19M-row unist manifest that pass exhausted
# the 1500 GiB container cgroup and the job was OOMKilled before step 1.
#
# The sampler only ever needs the number, never the encoded row:
#   ``MegatronPretrainingRandomSampler.__init__`` does ``dataset['lengths']``
#   and ``LazyLLMDataset.__getitem__`` forwards *string* keys straight to the
#   underlying HF dataset. So a ``lengths`` column baked here at map time is
#   consumed with ZERO encode calls, and ``lazy_tokenize=true`` stays on.
#
# ACCURACY: ``get_length_grouped_indices`` only SORTS by this value, so a
# monotonically-correlated estimate is sufficient — it does not have to match
# the true token count exactly.
# ---------------------------------------------------------------------------
def _audio_tokens_from_ms(duration_ms: float) -> int:
    """Thinker-side audio token count for a given audio duration.

    Mirrors the Qwen2.5-Omni audio tower exactly:
      mel frames = samples / hop_length(160)      -> 100 frames/s
      conv1 (stride 2): (n - 1) // 2 + 1
      conv2 (stride 2): (n - 2) // 2 + 1
    i.e. 25 tokens/s (40 ms per token). Verified against
    ``Qwen2_5OmniAudioEncoder._get_feat_extract_output_lengths``:
    1120ms -> 28, 3740ms -> 93, 5580ms -> 139.
    """
    mel = int(duration_ms) * 16000 // 1000 // 160
    if mel <= 2:
        return 0
    conv1 = (mel - 1) // 2 + 1
    return max((conv1 - 2) // 2 + 1, 0)


def _estimate_row_lengths(row: Dict[str, Any]) -> int:
    """Cheap (zero-IO) estimate of the thinker-side sequence length of a row.

    Reads only pre-baked manifest scalars, so it costs nothing at map time and
    works for ``wds://`` rows whose audio lives inside a tar shard.

    Composition: a fixed chat-template/system-prompt base + source audio tokens
    + target text tokens. The base term is constant across rows so it does not
    affect bucketing at all; it exists only to keep the number close to the real
    ``len(input_ids)`` for log readability. ``OMNISPEAK_LENGTH_BASE_TOKENS``
    tunes it.

    Never returns ``None``: the column must stay a clean integer column or
    ``get_length_grouped_indices`` would fail on a null. Rows with no usable
    duration fall back to the base value and simply land in one bucket
    (degrading to random order for those rows, which is harmless).
    """
    base = int(os.environ.get('OMNISPEAK_LENGTH_BASE_TOKENS', '48'))

    duration_ms = 0.0
    src = row.get('source_duration_ms')
    if not isinstance(src, (int, float)) or src <= 0:
        # Shape A/B Lance pointer rows before rename, or raw table dumps.
        src = row.get('duration_ms')
    if isinstance(src, (int, float)) and src > 0:
        duration_ms = float(src)
    else:
        # Fallback for manifests without an explicit source duration: the
        # target duration is strongly correlated with the source one.
        tgt = row.get('target_duration_ms')
        if isinstance(tgt, (int, float)) and tgt > 0:
            duration_ms = float(tgt)

    text = row.get('target_text')
    text_tokens = len(text) if isinstance(text, str) else 0

    return base + _audio_tokens_from_ms(duration_ms) + text_tokens


def _attach_lengths(row: Dict[str, Any], result: Dict[str, Any]) -> None:
    """Bake the ``lengths`` column used by ``--group_by_length`` into a mapped row."""
    result['lengths'] = _estimate_row_lengths(row)


def _row_audio_paths(row: Dict[str, Any]) -> List[str]:
    """All candidate audio paths in a raw JSONL row: target_audio, target_audios,
    per-chunk alignment target_audio, source audios, and reference audios."""
    paths: List[str] = []

    def _add(p):
        if isinstance(p, str) and p:
            paths.append(p)

    _add(row.get('target_audio'))
    for p in (row.get('target_audios') or []):
        _add(p)
    for seg in (row.get('alignment') or []):
        if isinstance(seg, dict):
            _add(seg.get('target_audio'))
    for a in (row.get('audios') or []):
        _add(a.get('path') if isinstance(a, dict) else a)
    for a in (row.get('reference_audios') or []):
        _add(a.get('path') if isinstance(a, dict) else a)
    return paths


def _row_exceeds_audio_limit(row: Dict[str, Any], max_seconds: float) -> bool:
    """True if ANY audio (target or source) in the raw row is longer than max_seconds.

    Fast path (zero IO, WDS-safe): read the pre-baked duration fields
    ``source_duration_ms`` / ``target_duration_ms`` directly from the manifest
    row. ``wds://`` rows resolve to a *tar shard* path, so ``soundfile`` cannot
    sniff the member duration and the old file-sniff path returned ``None`` ->
    the sample was KEPT, silently disabling this filter for the main WDS data.
    Reading the manifest fields makes the guard effective without touching the
    audio tar at all. Falls back to sniffing the actual audio file when the
    fields are absent (legacy standalone paths).

    NOTE: once this fast path is enabled, over-long WDS samples are finally
    *dropped* (previously kept). That is the intended gate behaviour; watch the
    ``[AUDIO_DUR][DROP]`` counter on first rollout.
    """
    # Fast path: pre-baked manifest duration fields (ms). Member-level precise,
    # works for wds:// rows whose tar members cannot be sniffed.
    for field in ('source_duration_ms', 'target_duration_ms'):
        v = row.get(field)
        if isinstance(v, (int, float)) and v / 1000.0 > max_seconds:
            return True
    # Fallback: sniff the actual audio file (non-WDS / legacy paths, or rows
    # missing the duration fields). If a duration cannot be read we KEEP the
    # sample rather than risk dropping good data.
    for p in _row_audio_paths(row):
        if is_lance_uri(p):
            # lance:// 无法 soundfile sniff；source_duration_ms 快路径已覆盖超长判断。
            continue
        dur_ms = _read_audio_duration_ms(p)
        if dur_ms is not None and dur_ms / 1000.0 > max_seconds:
            return True
    return False


def drop_row_if_invalid_audio(row: Dict[str, Any]) -> bool:
    """Single audio-validity gate for every preprocessor (missing + over-long).

    Runs at dataset-load time so a rejected row is truly removed from the HF
    dataset (``preprocess`` returning ``None`` -> dropped by
    ``RowPreprocessor.batched_preprocess``), which — unlike a
    ``Template._encode`` filter — also applies to the val split carved out by
    ``--split_dataset_ratio``. Returns True when the caller should drop the row.

    Mel-mode rows may have empty ``audios`` and carry source via
    ``source_input_features``; those are valid.
    """
    paths = list(_row_audio_paths(row))
    if not paths and _row_has_precomputed_mel(row):
        # Features-only source: no wav path to validate.
        max_seconds = _max_audio_seconds()
        if _row_exceeds_audio_limit(row, max_seconds):
            print(
                f'[AUDIO_DUR][DROP] sample dropped: any audio > {max_seconds:g}s '
                f"(task={row.get('task')}, target_audio={row.get('target_audio')})",
                flush=True,
            )
            return True
        return False
    for p in paths:
        # lance:// 不是文件路径，存在性由 lance 自身保证；跳过 os.path.exists
        # 检查，否则整行会被误判为缺失而 DROP（已知陷阱：lance对接训练架构设计.md §7）。
        if is_lance_uri(p):
            continue
        if not os.path.exists(_resolve_audio_path(p)):
            print(f'[AUDIO_MISSING][DROP] sample dropped: audio not found: {p}', flush=True)
            return True
    max_seconds = _max_audio_seconds()
    if _row_exceeds_audio_limit(row, max_seconds):
        print(
            f'[AUDIO_DUR][DROP] sample dropped: any audio > {max_seconds:g}s '
            f"(task={row.get('task')}, target_audio={row.get('target_audio')})",
            flush=True,
        )
        return True
    return False


class NormalTranslationPreprocessor(SimultaneousPolicyPreprocessor):
    """Preprocessor for the ``normal_translation`` task.

    Uses the shared Stage-3 SFT timeline validation from
    :class:`SimultaneousPolicyPreprocessor`: both source and target chunks must
    be strictly non-overlapping, and ``max(target_end_ms)`` must be smaller than
    ``source_duration_ms + 0.5 * last_source_chunk_duration + 3000`` ms.
    """

    task_set = NORMAL_TRANSLATION_TASKS
    env_prefix = 'NORMAL_TRANSLATION'

    def _reject(self, reason: str) -> bool:
        if self.strict:
            raise ValueError(f'normal_translation sample rejected: {reason}')
        logger.debug('Discarding normal_translation sample: %s', reason)
        return False


def register_normal_translation_datasets(dataset_path: Optional[str] = None) -> None:
    dataset_path = os.path.abspath(dataset_path or default_normal_translation_data_path())
    for name, task_filter in [
        ('normal_translation', TASK_ALL),
    ]:
        register_dataset(
            DatasetMeta(
                dataset_name=name,
                ms_dataset_id=name,
                dataset_path=dataset_path,
                preprocess_func=NormalTranslationPreprocessor(task_filter=task_filter),
                tags=['audio', 'speech-generation', 'qwen3-omnispeak',
                      'normal-translation', 'simultaneous-interpretation', 'sft'],
            ),
            exist_ok=True,
        )


# ---------------------------------------------------------------------------
# bilingual_translation preprocessor (Stage-3 SFT, streaming bilingual s2s).
# ---------------------------------------------------------------------------


class BilingualTranslationPreprocessor(NormalTranslationPreprocessor):
    """Preprocessor for the ``bilingual_translation`` task.

    Uses the same shared Stage-3 SFT timeline validation as the other SFT tasks.
    """

    task_set = BILINGUAL_TRANSLATION_TASKS
    env_prefix = 'BILINGUAL_TRANSLATION'

    def _reject(self, reason: str) -> bool:
        if self.strict:
            raise ValueError(f'bilingual_translation sample rejected: {reason}')
        logger.debug('Discarding bilingual_translation sample: %s', reason)
        return False


def register_bilingual_translation_datasets(dataset_path: Optional[str] = None) -> None:
    dataset_path = os.path.abspath(dataset_path or default_bilingual_translation_data_path())
    for name, task_filter in [
        ('bilingual_translation', TASK_ALL),
    ]:
        register_dataset(
            DatasetMeta(
                dataset_name=name,
                ms_dataset_id=name,
                dataset_path=dataset_path,
                preprocess_func=BilingualTranslationPreprocessor(task_filter=task_filter),
                tags=['audio', 'speech-generation', 'qwen3-omnispeak',
                      'bilingual-translation', 'simultaneous-interpretation', 'sft'],
            ),
            exist_ok=True,
        )


# ---------------------------------------------------------------------------
# multi_speaker_translation preprocessor (Stage-3 SFT, streaming multi-speaker s2s).
# ---------------------------------------------------------------------------


class MultiSpeakerTranslationPreprocessor(NormalTranslationPreprocessor):
    """Preprocessor for the ``multi_speaker_translation`` task.

    Uses the same shared Stage-3 SFT timeline validation as the other SFT tasks:
    source chunks and target chunks must both be strictly non-overlapping, and
    the target timeline duration must satisfy the common tail bound.
    """

    task_set = MULTI_SPEAKER_TRANSLATION_TASKS
    env_prefix = 'MULTI_SPEAKER_TRANSLATION'

    def _reject(self, reason: str) -> bool:
        if self.strict:
            raise ValueError(f'multi_speaker_translation sample rejected: {reason}')
        logger.debug('Discarding multi_speaker_translation sample: %s', reason)
        return False


def register_multi_speaker_translation_datasets(dataset_path: Optional[str] = None) -> None:
    dataset_path = os.path.abspath(dataset_path or default_multi_speaker_translation_data_path())
    for name, task_filter in [
        ('multi_speaker_translation', TASK_ALL),
    ]:
        register_dataset(
            DatasetMeta(
                dataset_name=name,
                ms_dataset_id=name,
                dataset_path=dataset_path,
                preprocess_func=MultiSpeakerTranslationPreprocessor(task_filter=task_filter),
                tags=['audio', 'speech-generation', 'qwen3-omnispeak',
                      'multi-speaker-translation', 'simultaneous-interpretation', 'sft'],
            ),
            exist_ok=True,
        )


# ---------------------------------------------------------------------------
# sft_joint_stage1 preprocessor (Stage-3 SFT joint translation task).
# ---------------------------------------------------------------------------


class SFTJointStage1Preprocessor(NormalTranslationPreprocessor):
    """Joint SFT preprocessor for normal / bilingual / multi-speaker translation.

    Rows are accepted by their JSONL ``task`` field and then processed with the
    same Stage-3 streaming codec-only logic used by the individual SFT tasks.
    """

    task_set = SFT_JOINT_STAGE1_TASKS
    env_prefix = 'JOINT_STAGE1'

    def _reject(self, reason: str) -> bool:
        if self.strict:
            raise ValueError(f'sft_joint_stage1 sample rejected: {reason}')
        logger.debug('Discarding sft_joint_stage1 sample: %s', reason)
        return False


def register_sft_joint_stage1_datasets(dataset_path: Optional[str] = None) -> None:
    dataset_path = os.path.abspath(dataset_path or default_sft_joint_stage1_data_path())
    for name, task_filter in [
        ('sft_joint_stage1', TASK_ALL),
    ]:
        register_dataset(
            DatasetMeta(
                dataset_name=name,
                ms_dataset_id=name,
                dataset_path=dataset_path,
                preprocess_func=SFTJointStage1Preprocessor(task_filter=task_filter),
                tags=['audio', 'speech-generation', 'qwen3-omnispeak',
                      'joint-stage1', 'normal-translation', 'bilingual-translation',
                      'multi-speaker-translation', 'simultaneous-interpretation', 'sft'],
            ),
            exist_ok=True,
        )


__all__ = [
    'BridgeAlignmentPreprocessor',
    'ContinuePretrainPreprocessor',
    'SimultaneousPolicyPreprocessor',
    'NormalTranslationPreprocessor',
    'BilingualTranslationPreprocessor',
    'MultiSpeakerTranslationPreprocessor',
    'SFTJointStage1Preprocessor',
    'register_bridge_alignment_datasets',
    'register_continue_pretrain_datasets',
    'register_simultaneous_policy_datasets',
    'register_normal_translation_datasets',
    'register_bilingual_translation_datasets',
    'register_multi_speaker_translation_datasets',
    'register_sft_joint_stage1_datasets',
    'default_data_path',
    'default_continue_pretrain_data_path',
    'default_simultaneous_policy_data_path',
    'default_normal_translation_data_path',
    'default_bilingual_translation_data_path',
    'default_multi_speaker_translation_data_path',
    'default_sft_joint_stage1_data_path',
    'TASK_TTS',
    'TASK_S2S_CONVERSATION',
    'TASK_S2S_TRANSLATION',
    'TASK_TEXT_TO_SPEECH_TRANSLATION',
    'TASK_S2T_TRANSLATION',
    'TASK_MULTI_SPEAKER_SEG',
    'TASK_SIMUL_POLICY',
    'TASK_NORMAL_TRANSLATION',
    'TASK_S2S_TRANSLATION_COMPACT',
    'NORMAL_TRANSLATION_TASKS',
    'TASK_BILINGUAL_TRANSLATION',
    'TASK_BILINGUAL_MUTUAL',
    'BILINGUAL_TRANSLATION_TASKS',
    'TASK_MULTI_SPEAKER_TRANSLATION',
    'MULTI_SPEAKER_TRANSLATION_TASKS',
    'SFT_JOINT_STAGE1_TASKS',
    'TASK_ALL',
    'EXTRA_KEYS',
]

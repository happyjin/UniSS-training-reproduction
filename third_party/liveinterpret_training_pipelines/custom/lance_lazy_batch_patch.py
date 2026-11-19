# Copyright (c) ModelScope Contributors. All rights reserved.
"""Batch-resolve Lance payloads before ms-swift lazy tokenization.

PyTorch's map-style DataLoader calls ``dataset.__getitems__(indices)`` when the
method exists.  ms-swift's ``LazyLLMDataset`` only implements ``__getitem__``,
so without this patch every sample enters ``template.encode`` independently
and lazy mel pointers each issue ``take_by_stable_rowid([one])``.

This patch adds the optional batched protocol.  One DataLoader batch is fetched
from the lightweight HF dataset, its Lance pointers are resolved by one
coalesced table take, and the resulting rows are then encoded individually as
before.  The model-facing collator and batch semantics are unchanged.

Set ``LANCE_DATALOADER_BATCH_RESOLVE=0`` to disable the optimization.  When the
variable is unset, it is enabled only if ``LANCE_TRAIN_SOURCE_URI`` is set.
"""
from __future__ import annotations

import os
from typing import Any, Dict, Iterable, List

try:
    from .wds_audio import resolve_lance_training_rows_batch
except ImportError:  # pragma: no cover - loaded as top-level external plugin.
    from wds_audio import resolve_lance_training_rows_batch

try:
    from swift.utils import get_logger
    logger = get_logger()
except Exception:  # pragma: no cover - unit tests do not install ms-swift.
    import logging
    logger = logging.getLogger(__name__)


_PATCH_MARKER = '_qwen_omnispeak_lance_batch_resolve'
_BATCH_FETCH_WARNING_EMITTED = False


def lance_dataloader_batch_resolve_enabled() -> bool:
    raw = os.environ.get('LANCE_DATALOADER_BATCH_RESOLVE')
    if raw is None or not raw.strip():
        return bool(os.environ.get('LANCE_TRAIN_SOURCE_URI'))
    return raw.strip().lower() in {'1', 'true', 'yes', 'y', 'on'}


def _columnar_to_rows(batch: Any, size: int) -> List[Dict[str, Any]]:
    """Convert HF ``dataset[list_of_indices]`` output to row dictionaries."""
    if isinstance(batch, list):
        if len(batch) != size or not all(isinstance(row, dict) for row in batch):
            raise TypeError('batched dataset access returned an invalid row list')
        return batch
    if not isinstance(batch, dict):
        raise TypeError(
            f'batched dataset access must return dict/list, got {type(batch)!r}')
    rows: List[Dict[str, Any]] = [dict() for _ in range(size)]
    for key, values in batch.items():
        try:
            if len(values) != size:
                raise ValueError(
                    f'column {key!r} has {len(values)} rows, expected {size}')
        except TypeError as exc:
            raise TypeError(f'column {key!r} is not batched') from exc
        for i in range(size):
            rows[i][key] = values[i]
    return rows


def _lazy_torch_dtype(lazy_dataset: Any):
    template = getattr(lazy_dataset, 'template', None)
    info = getattr(template, 'model_info', None)
    dtype = getattr(info, 'torch_dtype', None)
    if dtype is not None:
        return dtype
    raw = os.environ.get('TORCH_DTYPE', 'bfloat16')
    try:
        import torch
    except ImportError:  # pragma: no cover - unit tests without torch
        return raw
    return getattr(torch, raw, torch.bfloat16)


def _attach_native_mel_tensors(rows: List[Dict[str, Any]], torch_dtype) -> None:
    try:
        from .template import attach_native_mel_tensors_to_rows
    except ImportError:  # pragma: no cover
        from template import attach_native_mel_tensors_to_rows
    attach_native_mel_tensors_to_rows(rows, torch_dtype)


def _fetch_rows(dataset: Any, indices: List[int]) -> List[Dict[str, Any]]:
    """Use Arrow/HF vectorized indexing, with a generic Dataset fallback."""
    try:
        return _columnar_to_rows(dataset[indices], len(indices))
    except (TypeError, KeyError, IndexError, NotImplementedError):
        return [dataset[index] for index in indices]


def _install_on_class(lazy_dataset_cls: type) -> bool:
    """Install the patch on one LazyLLMDataset-compatible class."""
    if getattr(lazy_dataset_cls, _PATCH_MARKER, False):
        return False
    original_getitem = lazy_dataset_cls.__getitem__

    def __getitems__(self, indices: Iterable[int]):
        indices = list(indices)
        if not indices:
            return []
        if not lance_dataloader_batch_resolve_enabled():
            return [original_getitem(self, index) for index in indices]

        global _BATCH_FETCH_WARNING_EMITTED
        try:
            rows = _fetch_rows(self.dataset, indices)
            # This is the sole payload read for unresolved features, speaker and
            # codes in the DataLoader batch. It is a no-op for old caches whose
            # fields are already materialized.
            resolve_lance_training_rows_batch(rows)
        except Exception:
            # Preserve upstream behavior for unsupported Dataset wrappers and
            # malformed rows. Calling the original method retains strict mode,
            # n_try_fetch, random replacement and traceback-limit semantics.
            if not _BATCH_FETCH_WARNING_EMITTED:
                logger.warning(
                    '[lance-batch] batch resolve failed; falling back to '
                    'LazyLLMDataset.__getitem__ for this batch.',
                    exc_info=True)
                _BATCH_FETCH_WARNING_EMITTED = True
            return [original_getitem(self, index) for index in indices]

        try:
            _attach_native_mel_tensors(rows, _lazy_torch_dtype(self))
        except Exception:
            logger.debug(
                '[lance-batch] native mel attach skipped; encode will build per row.',
                exc_info=True)

        encoded = []
        for index, row in zip(indices, rows):
            try:
                encoded.append(self.encode_func(row, return_length=True))
            except Exception:
                # Let the exact upstream implementation handle errors and pick
                # replacement samples. This path is exceptional, so repeating
                # the first encode is preferable to duplicating version-specific
                # retry internals here.
                encoded.append(original_getitem(self, index))
        return encoded

    lazy_dataset_cls.__getitems__ = __getitems__
    setattr(lazy_dataset_cls, _PATCH_MARKER, True)
    setattr(lazy_dataset_cls, '_lance_original_getitem', original_getitem)
    return True


def install_lance_lazy_batch_patch() -> bool:
    """Patch ms-swift's LazyLLMDataset; return True when newly installed."""
    try:
        from swift.dataset.utils import LazyLLMDataset
    except Exception as exc:  # pragma: no cover - guarded in plugin import.
        logger.warning(f'Skip Lance lazy batch patch, import failed: {exc}')
        return False
    installed = _install_on_class(LazyLLMDataset)
    if installed:
        logger.info(
            'Installed Lance DataLoader batch resolver '
            '(LazyLLMDataset.__getitems__).')
    return installed


__all__ = [
    'install_lance_lazy_batch_patch',
    'lance_dataloader_batch_resolve_enabled',
]

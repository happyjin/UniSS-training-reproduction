# Copyright (c) ModelScope Contributors. All rights reserved.
"""Periodic real-autoregressive validation for Qwen-OmniSpeak S2S training.

The ordinary Swift validation loop remains responsible for teacher-forced
text/main-codec/residual-codec token accuracy.  This callback complements it
with a small, fixed validation subset that is decoded autoregressively every
``AR_EVAL_STEPS`` optimizer steps.

Only the main codec stream is rolled out.  Residual codec groups do not feed
back into the next main-code prediction, so skipping them preserves EOS and
length behaviour while making an in-training diagnostic practical.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import random
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch

from swift.callbacks import TrainerCallback
from swift.utils import get_logger


logger = get_logger()

CALLBACK_NAME = 'qwen_omnispeak_ar_eval'

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


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {'1', 'true', 'yes', 'y', 'on'}


def _codec_config(model_config: Any) -> Dict[str, Any]:
    talker_config = getattr(model_config, 'talker_config', None)
    return {
        'codec_bos_id': int(getattr(model_config, 'codec_bos_id', 2149)),
        'codec_eos_id': int(getattr(model_config, 'codec_eos_id', 2150)),
        'codec_pad_id': int(getattr(model_config, 'codec_pad_id', 2148)),
        'codec_mask_id': int(getattr(model_config, 'codec_mask_id', 2148)),
        'codec_think_id': int(getattr(model_config, 'codec_think_id', 2154)),
        'codec_nothink_id': int(getattr(model_config, 'codec_nothink_id', 2155)),
        'codec_think_bos_id': int(getattr(model_config, 'codec_think_bos_id', 2156)),
        'codec_think_eos_id': int(getattr(model_config, 'codec_think_eos_id', 2157)),
        'codec_language_id': dict(getattr(model_config, 'codec_language_id', None) or {}),
        'num_residual_groups': int(getattr(talker_config, 'num_code_groups', 15)),
    }


def _raw_dataset(dataset: Any) -> Any:
    """Peel Swift's lazy encoding wrapper without touching other datasets."""
    seen = set()
    while hasattr(dataset, 'dataset') and id(dataset) not in seen:
        seen.add(id(dataset))
        candidate = getattr(dataset, 'dataset')
        if candidate is dataset:
            break
        dataset = candidate
    return dataset


def _codec_frame_count(row: Dict[str, Any]) -> int:
    codes = row.get('target_codec_codes')
    if codes is None:
        return 0
    try:
        if len(codes) == 0:
            return 0
        # Expected manifest layout is [T, 16].  Keep compatibility with [16, T].
        if len(codes) == 16 and hasattr(codes[0], '__len__') and len(codes[0]) != 16:
            return int(len(codes[0]))
        return int(len(codes))
    except (TypeError, ValueError):
        return 0


def _length_bucket(frames: int) -> int:
    if frames <= 80:
        return 0
    if frames <= 160:
        return 1
    return 2


def select_ar_eval_indices(
    dataset: Any,
    count: int,
    *,
    seed: int = 42,
    candidate_count: int = 1024,
    max_target_frames: Optional[int] = None,
) -> List[int]:
    """Choose a deterministic direction/length-stratified validation subset."""
    size = len(dataset)
    if size <= 0 or count <= 0:
        return []
    rng = random.Random(seed)
    candidates = rng.sample(range(size), min(size, max(count * 8, candidate_count)))
    buckets: Dict[Tuple[str, int], List[int]] = {}
    fallback: List[int] = []
    for index in candidates:
        row = dataset[index]
        frames = _codec_frame_count(row)
        if frames <= 0 or (max_target_frames is not None and frames > max_target_frames):
            continue
        direction = f"{row.get('source_lang', '?')}->{row.get('target_lang', '?')}"
        buckets.setdefault((direction, _length_bucket(frames)), []).append(index)
        fallback.append(index)

    selected: List[int] = []
    keys = sorted(buckets)
    cursor = 0
    while keys and len(selected) < min(count, size):
        key = keys[cursor % len(keys)]
        values = buckets[key]
        if values:
            index = values.pop(0)
            if index not in selected:
                selected.append(index)
        keys = [k for k in keys if buckets[k]]
        cursor += 1
    for index in fallback:
        if len(selected) >= min(count, size):
            break
        if index not in selected:
            selected.append(index)
    if len(selected) < min(count, size):
        candidate_set = set(candidates)
        for index in range(size):
            if index in candidate_set:
                continue
            row = dataset[index]
            frames = _codec_frame_count(row)
            if frames <= 0 or (max_target_frames is not None and frames > max_target_frames):
                continue
            selected.append(index)
            if len(selected) >= min(count, size):
                break
    return selected


def _median(values: Sequence[float]) -> float:
    return float(statistics.median(values)) if values else float('nan')


def summarize_ar_records(
    records: Sequence[Dict[str, Any]],
    *,
    pathological_frames: int = 350,
) -> Dict[str, float]:
    """Aggregate sample records into scalar metrics suitable for Trainer.log."""
    valid = [record for record in records if not record.get('error')]
    errors = len(records) - len(valid)
    summary: Dict[str, float] = {
        'ar_eval/sample_count': float(len(valid)),
        'ar_eval/error_count': float(errors),
    }
    if not valid:
        return summary

    generated = [float(record['generated_frames']) for record in valid]
    eos = [float(bool(record['eos_triggered'])) for record in valid]
    timely = []
    early = []
    pathological = []
    text_exact = []
    boundary = []
    for record in valid:
        gt = max(1, int(record['target_codec_frames']))
        frames = int(record['generated_frames'])
        triggered = bool(record['eos_triggered'])
        threshold = max(gt + 12, int(math.ceil(1.5 * gt)))
        timely.append(float(triggered and frames <= threshold))
        early.append(float(triggered and frames < 0.75 * gt))
        pathological.append(float(frames >= pathological_frames))
        text_exact.append(float(bool(record.get('text_exact_match'))))
        point = record.get('gt_boundary')
        if point is not None:
            boundary.append(point)

    summary.update({
        'ar_eval/eos_trigger_rate': sum(eos) / len(eos),
        'ar_eval/timely_eos_rate': sum(timely) / len(timely),
        'ar_eval/early_eos_rate': sum(early) / len(early),
        'ar_eval/pathological_long_rate': sum(pathological) / len(pathological),
        'ar_eval/generated_frames_mean': sum(generated) / len(generated),
        'ar_eval/generated_frames_median': _median(generated),
        'ar_eval/text_exact_match': sum(text_exact) / len(text_exact),
        'ar_eval/alive_at_gt_boundary_rate': len(boundary) / len(valid),
    })
    if boundary:
        summary.update({
            'ar_eval/gt_boundary_eos_top1_rate': sum(
                float(bool(point['top1_is_eos'])) for point in boundary) / len(boundary),
            'ar_eval/gt_boundary_eos_probability_median': _median([
                float(point['eos_probability']) for point in boundary]),
            'ar_eval/gt_boundary_eos_rank_median': _median([
                float(point['eos_rank']) for point in boundary]),
        })
    return summary


def _move_to_device(value: Any, device: torch.device) -> Any:
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {key: _move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_move_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_move_to_device(item, device) for item in value)
    return value


def _get_tokenizer(processor: Any) -> Any:
    return getattr(processor, 'tokenizer', processor)


def _decode_text(processor: Any, token_ids: Sequence[int]) -> str:
    tokenizer = _get_tokenizer(processor)
    return tokenizer.decode(list(token_ids), skip_special_tokens=True)


def _text_stop_ids(processor: Any) -> set[int]:
    tokenizer = _get_tokenizer(processor)
    ids = set()
    for attr in ('eos_token_id', 'pad_token_id'):
        value = getattr(tokenizer, attr, None)
        if isinstance(value, int):
            ids.add(value)
    return ids


def _val_messages(row: Dict[str, Any], audio_path: str) -> List[Dict[str, Any]]:
    """Restore the mapped training prompt and remove its GT assistant turn."""
    messages = []
    for raw_message in row.get('messages') or []:
        message = json.loads(raw_message) if isinstance(raw_message, str) else dict(raw_message)
        if message.get('role') == 'assistant':
            continue
        content = message.get('content', '')
        if isinstance(content, str):
            blocks = []
            if '<audio>' in content:
                blocks.append({'type': 'audio', 'audio': audio_path})
                remaining = content.replace('<audio>', '').strip()
                if remaining:
                    blocks.append({'type': 'text', 'text': remaining})
            else:
                blocks.append({'type': 'text', 'text': content})
            message['content'] = blocks
        elif isinstance(content, list):
            normalized = []
            for block in content:
                if isinstance(block, dict) and block.get('type') == 'audio':
                    block = {**block, 'audio': audio_path}
                normalized.append(block)
            message['content'] = normalized
        messages.append(message)
    return messages


def _prepare_prompt_inputs(
    processor: Any,
    messages: List[Dict[str, Any]],
    device: torch.device,
) -> Dict[str, Any]:
    if hasattr(processor, 'apply_chat_template'):
        text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    else:
        text = _get_tokenizer(processor).apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False)

    try:
        from qwen_omni_utils import process_mm_info
        try:
            mm = process_mm_info(messages, use_audio_in_video=False)
        except TypeError:
            mm = process_mm_info(messages)
    except Exception as exc:
        raise RuntimeError(f'Unable to prepare AR audio input: {exc}') from exc

    if isinstance(mm, dict):
        media = mm
    elif isinstance(mm, (list, tuple)):
        names = ('audio', 'image', 'video')
        media = {names[i]: mm[i] for i in range(min(len(mm), len(names))) if mm[i] is not None}
    else:
        media = {}
    audio = media.get('audio') or media.get('audios')
    if audio is None:
        raise RuntimeError('AR validation processor received no decoded source audio.')

    base = {'text': [text], 'return_tensors': 'pt', 'padding': True}
    candidates = []
    for audio_key in ('audio', 'audios'):
        kwargs = dict(base)
        kwargs[audio_key] = audio
        candidates.append(kwargs)
    last_error: Optional[Exception] = None
    for kwargs in candidates:
        try:
            return _move_to_device(dict(processor(**kwargs)), device)
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f'Failed to encode AR validation prompt: {last_error}')


def _split_thinker_inputs(inputs: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    input_ids = inputs['input_ids']
    attention_mask = inputs.get('attention_mask')
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)
    kwargs = {key: value for key, value in inputs.items()
              if key not in {'input_ids', 'attention_mask'}}
    return input_ids, attention_mask, kwargs


def _generate_text_ids(
    model: Any,
    processor: Any,
    prompt_inputs: Dict[str, Any],
    max_new_tokens: int,
) -> List[int]:
    input_ids, attention_mask, thinker_kwargs = _split_thinker_inputs(prompt_inputs)
    current_ids = input_ids.clone()
    current_mask = attention_mask.clone()
    stop_ids = _text_stop_ids(processor)
    generated: List[int] = []
    with torch.inference_mode():
        for _ in range(max_new_tokens):
            output = model.model.run_thinker(
                input_ids=current_ids,
                attention_mask=current_mask,
                output_text_embeds=False,
                **thinker_kwargs,
            )
            logits = output.get('logits')
            if logits is None:
                raise RuntimeError('Thinker returned no logits during AR validation.')
            next_id = logits[:, -1, :].argmax(dim=-1)
            token = int(next_id.item())
            if token in stop_ids:
                break
            generated.append(token)
            current_ids = torch.cat([current_ids, next_id[:, None]], dim=1)
            current_mask = torch.cat(
                [current_mask, torch.ones_like(next_id[:, None])], dim=1)
    return generated


def _eos_stats(logits: torch.Tensor, eos_id: int) -> Tuple[float, int, int]:
    scores = logits.float()
    eos_score = scores[eos_id]
    probability = float(torch.exp(eos_score - torch.logsumexp(scores, dim=-1)).item())
    rank = int((scores > eos_score).sum().item()) + 1
    return probability, rank, int(scores.argmax().item())


def _build_codec_prefix(
    model_config: Any,
    target_lang: str,
    prompt_length: int,
    *,
    use_speaker: bool,
) -> Tuple[List[int], Optional[int]]:
    cfg = _codec_config(model_config)
    raw = str(target_lang or '').strip().lower()
    lang_key = _LANG_ALIASES.get(raw, raw)
    lang_id = cfg['codec_language_id'].get(lang_key)
    if lang_id is None:
        think_part = [cfg['codec_nothink_id'], cfg['codec_think_bos_id'], cfg['codec_think_eos_id']]
    else:
        think_part = [
            cfg['codec_think_id'], cfg['codec_think_bos_id'],
            int(lang_id), cfg['codec_think_eos_id'],
        ]
    full_add = _env_bool('QWEN_OMNISPEAK_FULL_ADD_LAYOUT', False)
    prefix = ([cfg['codec_mask_id']] * prompt_length + think_part) if full_add else list(think_part)
    speaker_slot = None
    if use_speaker:
        speaker_slot = len(prefix)
        prefix.append(cfg['codec_pad_id'])
    prefix.extend([cfg['codec_pad_id'], cfg['codec_bos_id']])
    return prefix, speaker_slot


def _rollout_main_codec(
    model: Any,
    model_config: Any,
    prompt_inputs: Dict[str, Any],
    text_ids: Sequence[int],
    target_lang: str,
    target_frames: int,
    reference_wav: Any,
    reference_sr: int,
    max_frames: int,
) -> Dict[str, Any]:
    input_ids, attention_mask, thinker_kwargs = _split_thinker_inputs(prompt_inputs)
    reply = torch.tensor([list(text_ids)], dtype=torch.long, device=input_ids.device)
    full_ids = torch.cat([input_ids, reply], dim=1)
    full_mask = torch.cat([attention_mask, torch.ones_like(reply)], dim=1)
    text_labels = torch.full_like(full_ids, -100)
    if reply.numel():
        text_labels[:, input_ids.size(1):] = reply

    with torch.inference_mode():
        thinker_out = model.model.run_thinker(
            input_ids=full_ids,
            attention_mask=full_mask,
            output_text_embeds=True,
            **thinker_kwargs,
        )

    cfg = _codec_config(model_config)
    eos_id = cfg['codec_eos_id']
    prefix, speaker_slot = _build_codec_prefix(
        model_config, target_lang, int(input_ids.size(1)), use_speaker=True)
    codec_ids = torch.tensor([prefix], dtype=torch.long, device=input_ids.device)
    # Extract the frozen speaker vector once. Passing the waveform to every AR
    # forward would rerun ECAPA for every generated frame; ``source_xvector``
    # is the model's mathematically identical fast path.
    model_module = sys.modules.get(model.__class__.__module__)
    get_speaker_embedder = getattr(model_module, 'get_speaker_embedder', None)
    if get_speaker_embedder is None:
        raise RuntimeError('Model speaker embedder is unavailable during AR validation.')
    speaker_embedder = get_speaker_embedder(
        enc_dim=int(getattr(model_config, 'speaker_embedding_dim', 2048)))
    source_xvector = speaker_embedder.extract(
        reference_wav,
        reference_sr,
        device=input_ids.device,
        dtype=thinker_out['hidden_states'].dtype,
    ).unsqueeze(0)
    boundary = None
    eos_triggered = False
    max_eos_probability = 0.0
    best_eos_rank = math.inf
    started = time.perf_counter()

    with torch.inference_mode():
        for step in range(max_frames + 1):
            output = model(
                thinker_text_embeds=thinker_out['text_embeds'],
                thinker_hidden_states=thinker_out['hidden_states'],
                thinker_attention_mask=full_mask,
                thinker_text_labels=text_labels,
                codec_input_ids=codec_ids,
                codec_attention_mask=torch.ones_like(codec_ids),
                source_xvector=source_xvector,
                speaker_slot_index=torch.tensor(
                    [speaker_slot], dtype=torch.long, device=input_ids.device),
                return_dict=True,
            )
            logits = output.main_codec_logits[0, -1]
            eos_probability, eos_rank, top1 = _eos_stats(logits, eos_id)
            max_eos_probability = max(max_eos_probability, eos_probability)
            best_eos_rank = min(best_eos_rank, eos_rank)
            if step == target_frames:
                boundary = {
                    'eos_probability': eos_probability,
                    'eos_rank': eos_rank,
                    'top1_is_eos': top1 == eos_id,
                }
            if top1 == eos_id:
                eos_triggered = True
                generated_frames = step
                break
            if step >= max_frames:
                generated_frames = max_frames
                break
            codec_ids = torch.cat([
                codec_ids,
                torch.tensor([[top1]], dtype=torch.long, device=codec_ids.device),
            ], dim=1)

    return {
        'eos_triggered': eos_triggered,
        'generated_frames': generated_frames,
        'gt_boundary': boundary,
        'max_eos_probability': max_eos_probability,
        'best_eos_rank': int(best_eos_rank),
        'runtime_seconds': time.perf_counter() - started,
    }


def _first_audio(row: Dict[str, Any]) -> str:
    value = row.get('audios') or row.get('audio')
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError:
            return value
        value = decoded
    if isinstance(value, (list, tuple)) and value:
        return str(value[0])
    raise ValueError('Validation row has no source audio.')


def _materialize_audio(audio: str, cache_dir: Path) -> Tuple[str, Any, int]:
    # Resolve/load the waveform through the same code path as the training
    # template.  The processor currently wants a path, so WDS bytes are also
    # placed in a small, deterministic per-run cache.
    try:
        from .speaker_encoder import load_reference_waveform
        from .wds_audio import parse_wds_uri, resolve_audio_reference, is_wds_uri
    except ImportError:  # pragma: no cover - external plugin top-level import.
        from custom.speaker_encoder import load_reference_waveform
        from custom.wds_audio import parse_wds_uri, resolve_audio_reference, is_wds_uri

    reference_wav, reference_sr = load_reference_waveform(audio)
    if not is_wds_uri(audio):
        return audio, reference_wav, reference_sr

    payload = resolve_audio_reference(audio)
    reference = parse_wds_uri(audio)
    suffix = f".{reference.audio_format or 'wav'}"
    digest = hashlib.sha256(audio.encode('utf-8')).hexdigest()[:20]
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f'{digest}{suffix}'
    if not path.is_file() or path.stat().st_size != len(payload):
        temporary = path.with_suffix(path.suffix + f'.tmp-{os.getpid()}')
        temporary.write_bytes(bytes(payload))
        os.replace(temporary, path)
    return str(path), reference_wav, reference_sr


class QwenOmniSpeakAREvalCallback(TrainerCallback):
    """Run distributed real-AR S2S validation at a configurable interval."""

    def __init__(self, args, trainer):
        super().__init__(args, trainer)
        self.interval = int(os.environ.get('AR_EVAL_STEPS', '1000'))
        self.num_samples = int(os.environ.get('AR_EVAL_NUM_SAMPLES', '32'))
        self.max_text_tokens = int(os.environ.get('AR_EVAL_MAX_TEXT_TOKENS', '128'))
        self.max_codec_frames = int(os.environ.get('AR_EVAL_MAX_CODEC_FRAMES', '384'))
        self.pathological_frames = int(os.environ.get('AR_EVAL_PATHOLOGICAL_FRAMES', '350'))
        self.seed = int(os.environ.get('AR_EVAL_SEED', '42'))
        self.strict = _env_bool('AR_EVAL_STRICT', True)
        self._last_step = -1
        self._indices: Optional[List[int]] = None

    @staticmethod
    def _dist_info() -> Tuple[int, int]:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return torch.distributed.get_rank(), torch.distributed.get_world_size()
        return 0, 1

    def _run_one(
        self,
        model: Any,
        processor: Any,
        row: Dict[str, Any],
        index: int,
        device: torch.device,
        cache_dir: Path,
    ) -> Dict[str, Any]:
        audio = _first_audio(row)
        audio_path, reference_wav, reference_sr = _materialize_audio(audio, cache_dir)
        prompt_inputs = _prepare_prompt_inputs(
            processor, _val_messages(row, audio_path), device)
        text_ids = _generate_text_ids(
            model, processor, prompt_inputs, self.max_text_tokens)
        predicted_text = _decode_text(processor, text_ids).strip()
        target_text = str(row.get('target_text') or '').strip()
        target_frames = _codec_frame_count(row)
        if target_frames <= 0:
            raise ValueError('Validation row has no target codec frames.')
        rollout = _rollout_main_codec(
            model=model,
            model_config=model.config,
            prompt_inputs=prompt_inputs,
            text_ids=text_ids,
            target_lang=str(row.get('target_lang') or ''),
            target_frames=target_frames,
            reference_wav=reference_wav,
            reference_sr=reference_sr,
            max_frames=self.max_codec_frames,
        )
        return {
            'val_index': index,
            'direction': f"{row.get('source_lang', '?')}->{row.get('target_lang', '?')}",
            'target_text': target_text,
            'predicted_text': predicted_text,
            'text_exact_match': predicted_text == target_text,
            'target_codec_frames': target_frames,
            **rollout,
        }

    def on_step_end(self, args, state, control, **kwargs):
        step = int(state.global_step)
        if self.interval <= 0 or step <= 0 or step % self.interval or step == self._last_step:
            return control
        self._last_step = step
        rank, world_size = self._dist_info()
        dataset = _raw_dataset(self.trainer.eval_dataset)
        if dataset is None or len(dataset) == 0:
            message = 'AR validation is enabled but trainer.eval_dataset is empty.'
            if self.strict:
                raise RuntimeError(message)
            logger.warning(message)
            return control

        if self._indices is None:
            self._indices = select_ar_eval_indices(
                dataset,
                self.num_samples,
                seed=self.seed,
                # The rollout evaluates EOS at ``step == max_codec_frames``
                # before applying its hard stop, so a target exactly at the
                # configured legal maximum is valid and should be selectable.
                max_target_frames=max(1, self.max_codec_frames),
            )
        if not self._indices:
            message = (
                'AR validation found no row with target codec frames below '
                f'AR_EVAL_MAX_CODEC_FRAMES={self.max_codec_frames}.')
            if self.strict:
                raise RuntimeError(message)
            logger.warning(message)
            return control
        local_indices = self._indices[rank::world_size]
        output_root = Path(args.output_dir) / 'ar_eval' / f'step_{step:08d}'
        cache_dir = Path(args.output_dir) / 'ar_eval' / 'source_cache'

        model = self.trainer.accelerator.unwrap_model(self.trainer.model)
        processor = self.trainer.template.processor
        device = next(model.parameters()).device
        was_training = model.training
        local_records: List[Dict[str, Any]] = []
        logger.info(
            f'[AR eval] step={step} rank={rank}/{world_size} '
            f'local_samples={len(local_indices)} total_samples={len(self._indices)}')
        model.eval()
        try:
            # Swift's training template installs a pre-forward media hook that
            # expects raw input_ids.  The Talker rollout intentionally calls
            # the model with precomputed Thinker states, so use Swift's own
            # generation context to remove that hook temporarily and restore
            # it before the next training step.
            with self.trainer.template.generate_context():
                for index in local_indices:
                    try:
                        record = self._run_one(
                            model, processor, dataset[index], index, device, cache_dir)
                    except Exception as exc:
                        logger.exception(f'[AR eval] val_index={index} failed: {exc}')
                        record = {'val_index': index, 'error': f'{type(exc).__name__}: {exc}'}
                    local_records.append(record)
        finally:
            if was_training:
                model.train()

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            gathered: List[Optional[List[Dict[str, Any]]]] = [None] * world_size
            torch.distributed.all_gather_object(gathered, local_records)
            records = [record for part in gathered if part for record in part]
        else:
            records = local_records

        summary = summarize_ar_records(
            records, pathological_frames=self.pathological_frames)
        summary['ar_eval/step'] = float(step)
        # Keep Trainer state/log history identical on all ranks. Integrations
        # such as W&B already restrict actual writes to the world-process zero.
        self.trainer.log(dict(summary))

        if rank == 0:
            output_root.mkdir(parents=True, exist_ok=True)
            with (output_root / 'samples.jsonl').open('w', encoding='utf-8') as handle:
                for record in sorted(records, key=lambda item: int(item.get('val_index', -1))):
                    handle.write(json.dumps(record, ensure_ascii=False) + '\n')
            (output_root / 'summary.json').write_text(
                json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
            logger.info(f'[AR eval] step={step} metrics={summary} output={output_root}')

        error_count = int(summary['ar_eval/error_count'])
        if error_count and self.strict:
            raise RuntimeError(
                f'AR validation failed for {error_count}/{len(records)} samples at step {step}; '
                f'see {output_root}/samples.jsonl')
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return control


def register_ar_eval_callback() -> None:
    """Register the callback in ms-swift's external callback mapping."""
    from swift.callbacks import callbacks_map
    callbacks_map[CALLBACK_NAME] = QwenOmniSpeakAREvalCallback


__all__ = [
    'CALLBACK_NAME',
    'QwenOmniSpeakAREvalCallback',
    'register_ar_eval_callback',
    'select_ar_eval_indices',
    'summarize_ar_records',
]

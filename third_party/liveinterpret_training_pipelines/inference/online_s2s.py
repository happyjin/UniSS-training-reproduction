"""End-to-end inference-only token-level Thinker→Talker orchestration."""

from __future__ import annotations

import os
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch

from inference.codec_buffer import CodecFrameBuffer
from inference.code_predictor_runtime import CodePredictorRunner
from inference.common import (
    AudioCallback,
    AudioGenerationResult,
    InferenceConfig,
    InferenceSample,
    LatencyMetrics,
    TextGenerationResult,
    _codec_config,
    build_control_prefix,
    codec_sample_rate,
    decode_codec,
    decode_text,
    eos_token_ids,
    get_codec_tokenizer,
    input_audio_duration_ms,
    is_non_silent,
    new_optimization_metrics,
    sample_next_token,
    split_thinker_kwargs,
    thinker_sampling_params,
    timed_phase,
)
from inference.online_pipeline import (
    IncrementalThinkerSession,
    OnlineTalkerSession,
    ResidualFrameBatcher,
    ThinkerTalkerScheduler,
)
from inference.streaming_decoder import CompiledStreamingDecoder


TextCallback = Callable[[str, float], None]


def build_online_control_prefix(
    model_config: Any,
    *,
    target_lang: Optional[str],
    prompt_len: int,
    full_add: bool,
    use_speaker: bool,
) -> Tuple[List[int], Optional[int]]:
    """Build the same codec prefix layout as the existing offline path."""
    codec_cfg = _codec_config(model_config)
    prefix = build_control_prefix(model_config, target_lang)
    if full_add:
        codec_mask_id = int(
            getattr(model_config, "codec_mask_id", codec_cfg["codec_pad_id"])
        )
        prefix = [codec_mask_id] * int(prompt_len) + prefix
    speaker_slot = None
    if use_speaker:
        speaker_slot = len(prefix) - 2
        prefix = (
            prefix[:speaker_slot]
            + [codec_cfg["codec_pad_id"]]
            + prefix[speaker_slot:]
        )
    return prefix, speaker_slot


def _source_xvector(
    model_config: Any,
    sample: InferenceSample,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Optional[torch.Tensor]:
    if not sample.audio:
        return None
    from speaker_encoder import get_speaker_embedder, load_reference_waveform

    waveform, sample_rate = load_reference_waveform(sample.audio)
    embedder = get_speaker_embedder(
        enc_dim=int(getattr(model_config, "speaker_embedding_dim", 2048))
    )
    vector = embedder.extract(
        waveform, sample_rate, device=device, dtype=dtype
    )
    return vector.unsqueeze(0)


def _predictor_runner(model: Any, config: InferenceConfig) -> CodePredictorRunner:
    key = (
        bool(config.use_code_predictor_kv_cache),
        bool(config.compile_code_predictor),
        config.code_predictor_compile_mode,
    )
    runners = getattr(model, "_code_predictor_runners", None)
    if runners is None:
        runners = {}
        setattr(model, "_code_predictor_runners", runners)
    if key not in runners:
        runners[key] = CodePredictorRunner(
            model.talker,
            use_kv_cache=config.use_code_predictor_kv_cache,
            compile_model=config.compile_code_predictor,
            compile_mode=config.code_predictor_compile_mode,
        )
    return runners[key]


@torch.inference_mode()
def generate_online_s2s(
    model: Any,
    processor: Any,
    model_config: Any,
    inputs: Dict[str, Any],
    sample: InferenceSample,
    config: InferenceConfig,
    *,
    stream: bool = False,
    callback: Optional[AudioCallback] = None,
    text_callback: Optional[TextCallback] = None,
    use_speaker: bool = False,
) -> Tuple[TextGenerationResult, AudioGenerationResult]:
    """Generate translation text and speech through one watermark scheduler."""
    if not config.online_thinker_talker:
        raise ValueError("generate_online_s2s requires online_thinker_talker=True")
    config.validate_static_optimizations()
    input_ids, attention_mask, thinker_kwargs = split_thinker_kwargs(inputs)
    device = input_ids.device
    started_at = time.perf_counter()
    target_lang = sample.target_lang or config.target_lang
    codec_cfg = _codec_config(model_config)
    full_add = os.environ.get(
        "QWEN_OMNISPEAK_FULL_ADD_LAYOUT", ""
    ).lower() in {"1", "true", "yes", "on"}
    prefix, speaker_slot = build_online_control_prefix(
        model_config,
        target_lang=target_lang,
        prompt_len=input_ids.size(1),
        full_add=full_add,
        use_speaker=use_speaker,
    )
    control_prefix_ids = torch.tensor(
        [prefix], dtype=torch.long, device=device
    )
    thinker_temperature, thinker_top_k, thinker_top_p = thinker_sampling_params(
        config
    )

    def _sample_text(logits: torch.Tensor) -> torch.Tensor:
        return sample_next_token(
            logits,
            temperature=thinker_temperature,
            top_k=thinker_top_k,
            top_p=thinker_top_p,
            repetition_penalty=1.0,
        )

    hidden_selection = getattr(
        model_config,
        "hidden_state_selection",
        getattr(model_config, "accept_hidden_layer", 24),
    )
    last_text_snapshot = ""

    def _on_text_tokens(token_ids: tuple[int, ...]) -> None:
        nonlocal last_text_snapshot
        if text_callback is None:
            return
        text = decode_text(processor, token_ids)
        if not text or text == last_text_snapshot or text.endswith("\ufffd"):
            return
        last_text_snapshot = text
        text_callback(text, (time.perf_counter() - started_at) * 1000.0)

    thinker_session = IncrementalThinkerSession(
        thinker=model.thinker,
        input_ids=input_ids,
        attention_mask=attention_mask,
        thinker_kwargs=thinker_kwargs,
        stop_ids=set(eos_token_ids(processor)),
        max_new_tokens=config.max_new_tokens,
        hidden_selection=hidden_selection,
        sample_fn=_sample_text,
        token_callback=_on_text_tokens,
        profiling_mode=config.latency_profiling,
    )
    thinker_session.prefill()
    if thinker_session.prompt_hidden is None:
        raise RuntimeError("Thinker prefill did not return prompt hidden states")

    source_xvector = (
        _source_xvector(
            model_config,
            sample,
            device=device,
            dtype=thinker_session.prompt_hidden.dtype,
        )
        if use_speaker
        else None
    )
    talker_session = OnlineTalkerSession(
        model=model,
        control_prefix_ids=control_prefix_ids,
        prompt_hidden=thinker_session.prompt_hidden,
        prompt_embeddings=thinker_session.prompt_embeddings,
        full_add_prompt_len=input_ids.size(1) if full_add else None,
        speaker_slot_index=speaker_slot,
        source_xvector=source_xvector,
        compile_model=config.compile_talker,
        profiling_mode=config.latency_profiling,
    )
    talker_session.prefill_metadata()

    def _sample_main(
        logits: torch.Tensor, previous: Optional[torch.Tensor]
    ) -> torch.Tensor:
        return sample_next_token(
            logits[:, -1, :],
            temperature=config.temperature,
            top_k=config.top_k,
            top_p=config.top_p,
            repetition_penalty=config.repetition_penalty,
            prev_tokens=previous,
        )

    scheduler = ThinkerTalkerScheduler(
        thinker_session=thinker_session,
        talker_session=talker_session,
        start_tokens=config.online_talker_start_tokens,
        max_codec_frames=config.max_codec_frames,
        codec_eos_id=codec_cfg["codec_eos_id"],
        sample_main_fn=_sample_main,
        thinker_prefilled=True,
        talker_metadata_prefilled=True,
    )
    predictor_runner = _predictor_runner(model, config)
    residual_batcher = ResidualFrameBatcher(
        predictor_runner, batch_size=config.predictor_frame_batch
    )
    codec_buffer = CodecFrameBuffer(
        config.max_codec_frames,
        codec_cfg["num_residual_groups"],
        device=device if config.use_gpu_codec_buffer else torch.device("cpu"),
        use_gpu=config.use_gpu_codec_buffer,
    )
    codec_tokenizer = get_codec_tokenizer(
        tokenizer_id=config.codec_tokenizer,
        device_map=config.codec_tokenizer_device,
    )
    audio_metrics = LatencyMetrics(
        sample_id=sample.sample_id,
        task=sample.task,
        output="audio",
        input_audio_duration_ms=input_audio_duration_ms(sample.audio),
        extra=new_optimization_metrics(),
    )
    stream_emit_every = max(1, int(config.stream_emit_every))
    stream_window = max(stream_emit_every, int(config.stream_decode_window))
    stream_frames_emitted = 0
    first_audio_at: Optional[float] = None
    first_main_at: Optional[float] = None
    max_predictor_backlog = 0

    compiled_decoder = None
    if config.compile_streaming_decoder:
        decoder_cache = getattr(
            codec_tokenizer, "_compiled_streaming_decoders", None
        )
        if decoder_cache is None:
            decoder_cache = {}
            setattr(codec_tokenizer, "_compiled_streaming_decoders", decoder_cache)
        if stream_window not in decoder_cache:
            compiled_decoder = CompiledStreamingDecoder(
                codec_tokenizer, window_frames=stream_window
            )
            audio_metrics.extra["compile_warmup_ms"] += compiled_decoder.warmup()
            decoder_cache[stream_window] = compiled_decoder
        else:
            compiled_decoder = decoder_cache[stream_window]

    def _decode_window(codes: torch.Tensor):
        with timed_phase(
            audio_metrics.extra,
            "stream_decoder_ms",
            device,
            mode=config.latency_profiling,
        ):
            if compiled_decoder is not None:
                return compiled_decoder.decode(codes)
            return decode_codec(
                codec_tokenizer,
                codes.detach().cpu().contiguous().numpy(),
            )

    def _emit_if_ready() -> None:
        nonlocal stream_frames_emitted, first_audio_at
        if not (stream and callback and config.streaming):
            return
        if len(codec_buffer) - stream_frames_emitted < stream_emit_every:
            return
        window_start = max(0, len(codec_buffer) - stream_window)
        window_codes = codec_buffer.window(window_start)
        waveform, sample_rate = _decode_window(window_codes)
        new_frames = len(codec_buffer) - stream_frames_emitted
        samples_per_frame = len(waveform) / max(1, window_codes.size(0))
        new_samples = int(round(new_frames * samples_per_frame))
        chunk = waveform[-new_samples:] if new_samples > 0 else waveform
        stream_frames_emitted = len(codec_buffer)
        if chunk.size:
            now = time.perf_counter()
            if first_audio_at is None and is_non_silent(
                chunk, config.silence_threshold
            ):
                first_audio_at = now
                audio_metrics.first_audio_latency_ms = (
                    now - started_at
                ) * 1000.0
            callback(chunk, sample_rate, (now - started_at) * 1000.0)

    def _append_ready(ready) -> None:
        for frame in ready:
            codec_buffer.append(frame.main_code, frame.residual_codes)
            _emit_if_ready()

    previous_predictor_warmup = predictor_runner.compile_warmup_ms
    for scheduled in scheduler.run():
        if first_main_at is None:
            first_main_at = time.perf_counter()
        with timed_phase(
            audio_metrics.extra,
            "code_predictor_ms",
            device,
            mode=config.latency_profiling,
        ):
            ready = residual_batcher.add(
                scheduled.main_code, scheduled.codec_hidden
            )
        max_predictor_backlog = max(
            max_predictor_backlog, residual_batcher.pending
        )
        _append_ready(ready)
    with timed_phase(
        audio_metrics.extra,
        "code_predictor_ms",
        device,
        mode=config.latency_profiling,
    ):
        _append_ready(residual_batcher.flush())

    predictor_warmup = (
        predictor_runner.compile_warmup_ms - previous_predictor_warmup
    )
    if predictor_warmup > 0:
        audio_metrics.extra["compile_warmup_ms"] += predictor_warmup
        audio_metrics.extra["code_predictor_ms"] = max(
            0.0,
            audio_metrics.extra["code_predictor_ms"] - predictor_warmup,
        )
    audio_metrics.extra["compile_warmup_ms"] += talker_session.compile_warmup_ms
    audio_metrics.extra["thinker_generate_ms"] = thinker_session.total_forward_ms
    audio_metrics.extra["talker_prefill_ms"] = max(
        0.0, talker_session.prefill_ms - talker_session.compile_warmup_ms
    )
    audio_metrics.extra["talker_step_ms"] = talker_session.step_ms

    with timed_phase(
        audio_metrics.extra,
        "codec_transfer_ms",
        device,
        mode=config.latency_profiling,
    ):
        codec_codes = codec_buffer.to_numpy()
    if len(codec_buffer):
        final_waveform, final_sample_rate = decode_codec(
            codec_tokenizer, codec_codes
        )
    else:
        final_waveform = np.zeros(0, dtype=np.float32)
        final_sample_rate = codec_sample_rate(codec_tokenizer)

    if stream and callback and len(codec_buffer) > stream_frames_emitted:
        remaining = len(codec_buffer) - stream_frames_emitted
        context = min(
            stream_frames_emitted, max(0, stream_window - remaining)
        )
        window_start = stream_frames_emitted - context
        waveform, sample_rate = _decode_window(codec_buffer.window(window_start))
        samples_per_frame = len(waveform) / max(1, remaining + context)
        skip = int(round(context * samples_per_frame))
        chunk = waveform[skip:]
        if chunk.size:
            now = time.perf_counter()
            if first_audio_at is None and is_non_silent(
                chunk, config.silence_threshold
            ):
                first_audio_at = now
                audio_metrics.first_audio_latency_ms = (
                    now - started_at
                ) * 1000.0
            callback(chunk, sample_rate, (now - started_at) * 1000.0)

    finished_at = time.perf_counter()
    text_finished_at = thinker_session.completed_at or finished_at
    text_metrics = LatencyMetrics(
        sample_id=sample.sample_id,
        task=sample.task,
        output="text",
        total_latency_ms=(text_finished_at - started_at) * 1000.0,
        first_token_latency_ms=(
            (thinker_session.first_sample_at - started_at) * 1000.0
            if thinker_session.first_sample_at is not None
            else None
        ),
        input_audio_duration_ms=audio_metrics.input_audio_duration_ms,
        generated_text_tokens=len(thinker_session.generated_ids),
        extra=new_optimization_metrics(),
    )
    text_metrics.extra["thinker_generate_ms"] = thinker_session.total_forward_ms
    first_condition_ms = (
        (thinker_session.first_commit_at - started_at) * 1000.0
        if thinker_session.first_commit_at is not None
        else None
    )
    audio_metrics.total_latency_ms = (finished_at - started_at) * 1000.0
    audio_metrics.generated_codec_frames = len(codec_buffer)
    audio_metrics.output_audio_duration_ms = (
        1000.0 * len(final_waveform) / final_sample_rate
        if final_sample_rate > 0
        else None
    )
    if audio_metrics.output_audio_duration_ms:
        audio_metrics.rtf = (
            audio_metrics.total_latency_ms / audio_metrics.output_audio_duration_ms
        )
    audio_metrics.extra.update(
        {
            "first_text_sample_ms": text_metrics.first_token_latency_ms,
            "first_talker_condition_ms": first_condition_ms,
            "first_main_codec_ms": (
                (first_main_at - started_at) * 1000.0
                if first_main_at is not None
                else None
            ),
            "max_text_condition_backlog": scheduler.max_condition_backlog,
            "max_predictor_backlog": max_predictor_backlog,
        }
    )
    text_result = TextGenerationResult(
        decode_text(processor, thinker_session.generated_ids),
        list(thinker_session.generated_ids),
        text_metrics,
    )
    audio_result = AudioGenerationResult(
        final_waveform, final_sample_rate, codec_codes, audio_metrics
    )
    return text_result, audio_result


__all__ = ["TextCallback", "build_online_control_prefix", "generate_online_s2s"]

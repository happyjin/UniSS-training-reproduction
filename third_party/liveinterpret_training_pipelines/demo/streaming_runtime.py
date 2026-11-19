"""Model/runtime adapter used by the single-GPU Flask streaming demo."""

from __future__ import annotations

import os
import queue
import threading
import time
from pathlib import Path
from typing import Any, Callable, Iterator

import numpy as np
import torch

from custom.task_prompts import s2s_task_prompt
from inference.common import (
    InferenceConfig,
    InferenceSample,
    build_e2e_latency_metrics,
    input_audio_duration_ms,
    load_model_and_processor,
    prepare_messages_inputs,
)
from inference.online_s2s import generate_online_s2s


REPO = Path(__file__).resolve().parents[1]
_END = object()


def resolve_model_asset(name: str, *, repo: Path = REPO) -> Path:
    """Resolve untracked model assets from a checkout or its main worktree."""
    candidates = [repo / "models" / name]
    if repo.parent.name == ".worktrees":
        candidates.append(repo.parent.parent / "models" / name)
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    return candidates[0]


DEFAULT_CODEC_TOKENIZER = resolve_model_asset("Qwen3-TTS-Tokenizer-12Hz")
DEFAULT_SPEAKER_ENCODER = resolve_model_asset("Qwen3-TTS-12Hz-1.7B-Base")


def _messages(prompt: str, audio_path: str) -> list[dict[str, Any]]:
    return [
        {"role": "system", "content": [{"type": "text", "text": prompt}]},
        {"role": "user", "content": [{"type": "audio", "audio": audio_path}]},
    ]


class StreamingDemoRuntime:
    """Own one loaded model and serialize requests onto one CUDA device."""

    def __init__(
        self,
        config: InferenceConfig,
        *,
        talker_layout: str,
        speaker_prompt: bool,
        speaker_encoder: str,
    ) -> None:
        self.config = config
        self.talker_layout = talker_layout
        self.speaker_prompt = bool(speaker_prompt)
        self.speaker_encoder = speaker_encoder
        self.ready = False
        self._runtime = None
        self._request_lock = threading.Lock()

    @classmethod
    def from_args(cls, args) -> "StreamingDemoRuntime":
        codec_tokenizer = args.codec_tokenizer or str(DEFAULT_CODEC_TOKENIZER)
        codec_device = args.codec_tokenizer_device or args.device
        speaker_encoder = args.speaker_encoder or str(DEFAULT_SPEAKER_ENCODER)
        config = InferenceConfig(
            model=args.checkpoint,
            model_type=args.model_type,
            device=args.device,
            dtype=args.dtype,
            attn_impl=args.attn_impl,
            max_new_tokens=args.max_new_tokens,
            max_codec_frames=args.max_codec_frames,
            temperature=0.9,
            top_k=50,
            top_p=1.0,
            repetition_penalty=1.05,
            thinker_temperature=0.0,
            thinker_top_k=0,
            thinker_top_p=1.0,
            codec_tokenizer=codec_tokenizer,
            codec_tokenizer_device=codec_device,
            streaming=True,
            stream_emit_every=4,
            stream_decode_window=80,
            use_kv_cache=True,
            use_code_predictor_kv_cache=True,
            compile_code_predictor=True,
            code_predictor_compile_mode="default",
            use_thinker_kv_cache=True,
            use_gpu_codec_buffer=True,
            compile_streaming_decoder=True,
            online_thinker_talker=True,
            online_talker_start_tokens=1,
            compile_talker=True,
            predictor_frame_batch=4,
            latency_profiling="off",
        )
        config.validate_static_optimizations()
        return cls(
            config,
            talker_layout=args.talker_layout,
            speaker_prompt=args.speaker_prompt,
            speaker_encoder=speaker_encoder,
        )

    def load(self) -> None:
        os.environ["QWEN3_TTS_TOKENIZER"] = str(self.config.codec_tokenizer)
        os.environ["QWEN_OMNISPEAK_FULL_ADD_LAYOUT"] = (
            "true" if self.talker_layout == "full_add" else "false"
        )
        os.environ["QWEN_OMNISPEAK_USE_SPEAKER_PROMPT"] = (
            "true" if self.speaker_prompt else "false"
        )
        if self.speaker_prompt:
            os.environ["QWEN3_TTS_SPEAKER_ENCODER"] = self.speaker_encoder
        self._runtime = load_model_and_processor(self.config)
        self.ready = True

    def iter_events(self, audio_path: str, target_lang: str) -> Iterator[dict[str, Any]]:
        if not self.ready:
            raise RuntimeError("model is not ready")
        events: queue.Queue[Any] = queue.Queue()

        def emit(event: dict[str, Any]) -> None:
            events.put(event)

        def worker() -> None:
            try:
                with self._request_lock:
                    self._run_request(audio_path, target_lang, emit)
            except BaseException as exc:
                events.put(exc)
            finally:
                events.put(_END)

        thread = threading.Thread(target=worker, name="streaming-s2s", daemon=True)
        thread.start()
        try:
            while True:
                event = events.get()
                if event is _END:
                    break
                if isinstance(event, BaseException):
                    raise event
                yield event
        finally:
            thread.join()

    def _run_request(
        self,
        audio_path: str,
        target_lang: str,
        emit: Callable[[dict[str, Any]], None],
    ) -> None:
        if self._runtime is None:
            raise RuntimeError("model is not loaded")
        model, processor, model_config, device, _ = self._runtime
        prompt = s2s_task_prompt(target_lang)
        sample = InferenceSample(
            sample_id=f"demo-{time.time_ns()}",
            prompt=prompt,
            audio=audio_path,
            task="s2s",
            output="audio",
            target_lang=target_lang,
        )
        inputs = prepare_messages_inputs(
            processor,
            _messages(prompt, audio_path),
            device,
            require_audio=True,
        )
        source_ms = input_audio_duration_ms(audio_path)
        emit({"type": "started", "input_audio_duration_ms": source_ms})

        first_text_ms: float | None = None
        first_pcm_ms: float | None = None

        def on_text(text: str, elapsed_ms: float) -> None:
            nonlocal first_text_ms
            if first_text_ms is None:
                first_text_ms = elapsed_ms
            emit(
                {
                    "type": "translation",
                    "text": text,
                    "server_elapsed_ms": elapsed_ms,
                }
            )

        def on_audio(chunk: np.ndarray, sample_rate: int, elapsed_ms: float) -> None:
            nonlocal first_pcm_ms
            if first_pcm_ms is None:
                first_pcm_ms = elapsed_ms
            emit(
                {
                    "type": "pcm",
                    "audio": np.asarray(chunk, dtype=np.float32).copy(),
                    "sample_rate": int(sample_rate),
                    "server_elapsed_ms": elapsed_ms,
                }
            )

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started_at = time.perf_counter()
        text_result, audio_result = generate_online_s2s(
            model,
            processor,
            model_config,
            inputs,
            sample,
            self.config,
            stream=True,
            callback=on_audio,
            text_callback=on_text,
            use_speaker=self.speaker_prompt,
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        complete_ms = (time.perf_counter() - started_at) * 1000.0
        metrics = build_e2e_latency_metrics(
            complete_wav_ms=complete_ms,
            first_text_ms=first_text_ms,
            first_pcm_ms=first_pcm_ms,
            compile_warmup_ms=float(
                audio_result.metrics.extra.get("compile_warmup_ms", 0.0)
            ),
        )
        metrics.update(
            {
                "input_audio_duration_ms": source_ms,
                "output_audio_duration_ms": audio_result.metrics.output_audio_duration_ms,
                "rtf": audio_result.metrics.rtf,
                "generated_text_tokens": text_result.metrics.generated_text_tokens,
                "generated_codec_frames": audio_result.metrics.generated_codec_frames,
            }
        )
        emit({"type": "metrics", "metrics": metrics})
        emit(
            {
                "type": "done",
                "translation": text_result.text,
                "sample_rate": int(audio_result.sample_rate),
            }
        )

    def warmup(self, audio_path: str, target_lang: str) -> None:
        path = Path(audio_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"Warmup audio does not exist: {path}")
        print(f"Warming streaming runtime with {path} ...", flush=True)
        started_at = time.perf_counter()
        for _ in self.iter_events(str(path), target_lang):
            pass
        print(
            f"Streaming runtime warmup complete in {time.perf_counter() - started_at:.2f}s",
            flush=True,
        )


__all__ = ["StreamingDemoRuntime", "resolve_model_asset"]

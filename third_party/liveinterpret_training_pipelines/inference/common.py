"""Shared runtime helpers for qwen_omnispeak inference scripts.

The helpers intentionally keep the implementation self-contained and use the
project's custom model package directly.  They provide pragmatic test-time
inference paths for text generation and codec/audio generation; the audio path
uses incremental re-computation because the custom model currently does not
expose a cache-aware generation API.
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
import tempfile
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch

try:
    import soundfile as sf
except ImportError as exc:  # pragma: no cover - dependency declared by root requirements.
    raise RuntimeError("soundfile is required for inference audio IO. Install project requirements.txt.") from exc

REPO_ROOT = Path(__file__).resolve().parents[1]
CUSTOM_DIR = REPO_ROOT / "custom"
QWEN_OMNISPEAK_DIR = CUSTOM_DIR / "qwen_omnispeak"


def ensure_project_imports() -> None:
    """Make local custom model modules importable from standalone scripts."""
    for path in (str(REPO_ROOT), str(CUSTOM_DIR), str(QWEN_OMNISPEAK_DIR)):
        if path not in sys.path:
            sys.path.insert(0, path)


ensure_project_imports()

from codec_tokenizer import DEFAULT_INPUT_SAMPLE_RATE, get_codec_tokenizer  # noqa: E402
from configuration_qwen_omnispeak import QwenOmniSpeakConfig  # noqa: E402
from modeling_qwen_omnispeak import QwenOmniSpeakForConditionalGeneration  # noqa: E402
from inference.codec_buffer import CodecFrameBuffer  # noqa: E402
from inference.code_predictor_runtime import CodePredictorRunner  # noqa: E402
from inference.streaming_decoder import CompiledStreamingDecoder  # noqa: E402


TextCallback = Callable[[str, int], None]
AudioCallback = Callable[[np.ndarray, int, float], None]

OPTIMIZATION_METRIC_KEYS = (
    "thinker_generate_ms",
    "thinker_finalize_ms",
    "talker_prefill_ms",
    "talker_step_ms",
    "code_predictor_ms",
    "codec_transfer_ms",
    "stream_decoder_ms",
    "compile_warmup_ms",
)


def new_optimization_metrics() -> Dict[str, float]:
    """Return the stable phase-metric schema used by every inference run."""
    return {key: 0.0 for key in OPTIMIZATION_METRIC_KEYS}


def build_e2e_latency_metrics(
    *,
    complete_wav_ms: float,
    first_text_ms: Optional[float],
    first_pcm_ms: Optional[float],
    compile_warmup_ms: float,
) -> Dict[str, Optional[float]]:
    """Build the shared input-ready latency schema for offline and online S2S."""
    return {
        "input_ready_to_first_text_ms": first_text_ms,
        "input_ready_to_first_pcm_ms": (
            first_pcm_ms if first_pcm_ms is not None else complete_wav_ms
        ),
        "input_ready_to_complete_wav_ms": complete_wav_ms,
        "compile_warmup_ms": compile_warmup_ms,
    }


@contextmanager
def timed_phase(
    metrics: Dict[str, Any],
    key: str,
    device: torch.device,
    *,
    mode: str = "sync",
) -> Iterator[None]:
    """Accumulate phase time with an explicit synchronization policy.

    ``sync`` preserves the legacy exact wall-time measurement. ``off``
    disables phase accounting entirely so end-to-end wall time can be measured
    without serializing hot loops.
    """
    if key not in OPTIMIZATION_METRIC_KEYS:
        raise KeyError(f"Unknown inference phase metric: {key}")
    if mode not in {"sync", "off"}:
        raise ValueError("mode must be 'sync' or 'off'")
    if mode == "off":
        yield
        return
    device = torch.device(device)
    if mode == "sync" and device.type == "cuda":
        torch.cuda.synchronize(device)
    started_at = time.perf_counter()
    try:
        yield
    finally:
        if mode == "sync" and device.type == "cuda":
            torch.cuda.synchronize(device)
        metrics[key] += (time.perf_counter() - started_at) * 1000.0


@dataclass
class InferenceConfig:
    model: str
    model_type: Optional[str] = None
    device: str = "auto"
    dtype: str = "auto"
    attn_impl: Optional[str] = None
    output_dir: str = "inference_outputs"
    max_new_tokens: int = 128
    max_codec_frames: int = 256
    codec_chunk_frames: int = 4
    temperature: float = 0.0
    top_k: int = 0
    top_p: float = 1.0
    repetition_penalty: float = 1.0
    # Optional Thinker-only overrides. ``None`` preserves the historical
    # shared temperature/top-k behavior; Thinker top-p was historically 1.0.
    thinker_temperature: Optional[float] = None
    thinker_top_k: Optional[int] = None
    thinker_top_p: Optional[float] = None
    target_lang: Optional[str] = None
    silence_threshold: float = 1e-4
    streaming_extra_lag_ms: float = 0.0
    use_streaming_schedule: bool = False
    codec_tokenizer: Optional[str] = None
    codec_tokenizer_device: Optional[str] = None
    # Streaming decode (borrowed from Qwen3-TTS-streaming): only decode the most
    # recent `stream_decode_window` codec frames each time and emit the tail
    # `stream_emit_every` frames' worth of new samples, instead of re-decoding
    # the whole sequence from scratch (O(N^2)). Off by default so the existing
    # non-streaming / full-decode paths are byte-for-byte unchanged.
    streaming: bool = False
    stream_emit_every: int = 4
    stream_decode_window: int = 80
    stream_overlap_samples: int = 0
    # Inference-only KV-cache for the talker backbone (uses the model's
    # make_cached_talker_session path; O(1) attention per frame instead of the
    # O(T^2) full recompute). Off by default; only the upstream reply_add/
    # full_add path is supported.
    use_kv_cache: bool = False
    # Independently switchable acceleration paths. They remain disabled by
    # default so existing callers keep their current numerical behavior.
    use_code_predictor_kv_cache: bool = False
    compile_code_predictor: bool = False
    code_predictor_compile_mode: str = "default"
    use_thinker_kv_cache: bool = False
    use_gpu_codec_buffer: bool = False
    compile_streaming_decoder: bool = False
    online_thinker_talker: bool = False
    online_talker_start_tokens: int = 1
    compile_talker: bool = False
    predictor_frame_batch: int = 1
    latency_profiling: str = "sync"

    def validate_static_optimizations(self) -> None:
        """Reject invalid optimization combinations before loading weights."""
        if self.code_predictor_compile_mode not in {"default", "reduce-overhead"}:
            raise ValueError(
                "code_predictor_compile_mode must be 'default' or "
                "'reduce-overhead'"
            )
        if self.compile_streaming_decoder and not self.streaming:
            raise ValueError("compile_streaming_decoder requires streaming=True")
        if self.online_talker_start_tokens not in {1, 2, 4}:
            raise ValueError("online_talker_start_tokens must be one of 1, 2, or 4")
        if self.predictor_frame_batch not in {1, 2, 4}:
            raise ValueError("predictor_frame_batch must be one of 1, 2, or 4")
        if self.latency_profiling not in {"sync", "off"}:
            raise ValueError("latency_profiling must be 'sync' or 'off'")


def thinker_sampling_params(config: InferenceConfig) -> Tuple[float, int, float]:
    """Resolve Thinker decoding without changing the Talker sampling fields."""
    return (
        config.temperature
        if config.thinker_temperature is None
        else config.thinker_temperature,
        config.top_k if config.thinker_top_k is None else config.thinker_top_k,
        1.0 if config.thinker_top_p is None else config.thinker_top_p,
    )


@dataclass
class InferenceSample:
    sample_id: str
    prompt: str
    audio: Optional[str] = None
    task: str = "auto"
    output: str = "text"
    target_lang: Optional[str] = None


@dataclass
class LatencyMetrics:
    sample_id: str
    task: str
    output: str
    total_latency_ms: float = 0.0
    first_token_latency_ms: Optional[float] = None
    first_audio_latency_ms: Optional[float] = None
    input_audio_duration_ms: Optional[float] = None
    output_audio_duration_ms: Optional[float] = None
    rtf: Optional[float] = None
    generated_text_tokens: int = 0
    generated_codec_frames: int = 0
    output_text_path: Optional[str] = None
    output_audio_path: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TextGenerationResult:
    text: str
    token_ids: List[int]
    metrics: LatencyMetrics


@dataclass
class AudioGenerationResult:
    audio: np.ndarray
    sample_rate: int
    codec_codes: np.ndarray
    metrics: LatencyMetrics


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def resolve_dtype(dtype: str, device: torch.device) -> torch.dtype:
    if dtype == "auto":
        return torch.float16 if device.type == "cuda" else torch.float32
    mapping = {
        "fp16": torch.float16,
        "float16": torch.float16,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    if dtype not in mapping:
        raise ValueError(f"Unsupported dtype: {dtype}")
    return mapping[dtype]


def move_to_device(obj: Any, device: torch.device) -> Any:
    if torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: move_to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, list):
        return [move_to_device(v, device) for v in obj]
    if isinstance(obj, tuple):
        return tuple(move_to_device(v, device) for v in obj)
    return obj


def _hoist_token_ids_for_processor(config: Any) -> None:
    keys = (
        "audio_token_id", "audio_start_token_id", "audio_end_token_id",
        "image_token_id", "video_token_id", "vision_start_token_id", "vision_end_token_id",
        "audio_token_index", "image_token_index", "video_token_index",
    )
    outer = getattr(config, "thinker_config", None)
    if outer is None:
        return
    sources = [getattr(outer, "thinker_config", None), getattr(outer, "talker_config", None)]
    sources = [src for src in sources if src is not None]
    for key in keys:
        if getattr(outer, key, None) is not None:
            continue
        for src in sources:
            value = getattr(src, key, None)
            if value is not None:
                try:
                    setattr(outer, key, value)
                except Exception:
                    pass
                break


def load_model_and_processor(config: InferenceConfig):
    """Load custom qwen_omnispeak model and matching Omni processor/tokenizer."""
    config.validate_static_optimizations()
    model_dir = str(Path(config.model).expanduser())
    device = resolve_device(config.device)
    dtype = resolve_dtype(config.dtype, device)

    if config.model_type:
        # Match the training path: register the Qwen-OmniSpeak variants, then
        # let ms-swift select the loader from the explicit model_type.  This is
        # important for checkpoints whose config.json only says
        # ``model_type=qwen_omnispeak`` and therefore cannot distinguish A/B/...
        # or the Qwen2.5 3B/7B and Qwen3 thinker families by itself.
        import importlib

        importlib.import_module("qwen_omnispeak.register_qwen_omnispeak")
        from swift.model import get_model_processor

        args_path = Path(model_dir) / "args.json"
        if args_path.is_file():
            try:
                saved_args = json.loads(args_path.read_text(encoding="utf-8"))
                saved_model_type = saved_args.get("model_type")
            except (OSError, json.JSONDecodeError):
                saved_model_type = None
            if saved_model_type and saved_model_type != config.model_type:
                raise ValueError(
                    "Checkpoint/model-type mismatch: "
                    f"checkpoint records {saved_model_type!r}, but inference was given "
                    f"{config.model_type!r}."
                )

        model, processor = get_model_processor(
            model_dir,
            model_type=config.model_type,
            torch_dtype=dtype,
            device_map=str(device),
            attn_impl=config.attn_impl,
            task_type="causal_lm",
            download_model=False,
        )
        hf_config = model.config
        _hoist_token_ids_for_processor(hf_config)
        model.eval()
        return model, processor, hf_config, device, dtype

    hf_config = QwenOmniSpeakConfig.from_pretrained(model_dir)
    _hoist_token_ids_for_processor(hf_config)
    model = QwenOmniSpeakForConditionalGeneration.from_pretrained(
        model_dir,
        torch_dtype=dtype,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model.to(device)
    model.eval()

    processor = load_processor(model_dir, hf_config)
    return model, processor, hf_config, device, dtype


def load_processor(model_dir: str, config: Any):
    from transformers import AutoTokenizer

    family = getattr(config, "thinker_family", "qwen3_omni_moe")
    if family == "qwen2_5_omni":
        try:
            from transformers import Qwen2_5OmniProcessor
            return Qwen2_5OmniProcessor.from_pretrained(model_dir, trust_remote_code=True)
        except Exception:
            return AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)

    try:
        from transformers import Qwen3OmniMoeProcessor
        return Qwen3OmniMoeProcessor.from_pretrained(model_dir, trust_remote_code=True)
    except Exception:
        return AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)


def get_tokenizer(processor: Any):
    return getattr(processor, "tokenizer", processor)


def decode_text(processor: Any, token_ids: Sequence[int], *, skip_special_tokens: bool = True) -> str:
    tokenizer = get_tokenizer(processor)
    if hasattr(tokenizer, "decode"):
        return tokenizer.decode(list(token_ids), skip_special_tokens=skip_special_tokens)
    if hasattr(processor, "decode"):
        return processor.decode(list(token_ids), skip_special_tokens=skip_special_tokens)
    return "".join(str(x) for x in token_ids)


def eos_token_ids(processor: Any) -> List[int]:
    tokenizer = get_tokenizer(processor)
    ids: List[int] = []
    for attr in ("eos_token_id", "pad_token_id"):
        value = getattr(tokenizer, attr, None)
        if isinstance(value, int) and value not in ids:
            ids.append(value)
    return ids


def build_messages(prompt: str, audio_path: Optional[str] = None) -> List[Dict[str, Any]]:
    content: List[Dict[str, Any]] = []
    if audio_path:
        content.append({"type": "audio", "audio": str(audio_path)})
    content.append({"type": "text", "text": prompt})
    return [{"role": "system", "content": [
        {"type": "text", "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}
    ]}, {"role": "user", "content": content}]


def _processor_call_candidates(processor: Any, text: str, mm: Dict[str, Any], *, require_audio: bool = False) -> List[Dict[str, Any]]:
    base = {"text": [text], "return_tensors": "pt", "padding": True}
    candidates: List[Dict[str, Any]] = []
    audio = mm.get("audio") or mm.get("audios")
    image = mm.get("image") or mm.get("images")
    video = mm.get("video") or mm.get("videos")
    if audio is not None or image is not None or video is not None:
        kwargs = dict(base)
        if audio is not None:
            kwargs["audio"] = audio
        if image is not None:
            kwargs["images"] = image
        if video is not None:
            kwargs["videos"] = video
        candidates.append(kwargs)
        kwargs2 = dict(base)
        if audio is not None:
            kwargs2["audios"] = audio
        if image is not None:
            kwargs2["image"] = image
        if video is not None:
            kwargs2["video"] = video
        candidates.append(kwargs2)
    candidates.append(base)
    return candidates


def _extract_mm_inputs(messages: List[Dict[str, Any]], *, require_audio: bool = False) -> Dict[str, Any]:
    try:
        from qwen_omni_utils import process_mm_info
    except Exception:
        return {}
    try:
        result = process_mm_info(messages, use_audio_in_video=False)
    except TypeError:
        result = process_mm_info(messages)
    if isinstance(result, dict):
        return result
    if isinstance(result, (list, tuple)):
        names = ("audio", "image", "video")
        return {names[i]: result[i] for i in range(min(len(result), len(names))) if result[i] is not None}
    return {}


def prepare_messages_inputs(
    processor: Any,
    messages: List[Dict[str, Any]],
    device: torch.device,
    *,
    require_audio: bool = False,
) -> Dict[str, Any]:
    """Encode an already-constructed conversation with optional media.

    This is used by training-aligned inference entrypoints that must preserve
    the dataset's exact system/user topology instead of ``build_messages``'s
    generic Qwen system prompt.
    """
    if hasattr(processor, "apply_chat_template"):
        text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    else:
        tokenizer = get_tokenizer(processor)
        text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

    mm = _extract_mm_inputs(messages, require_audio=require_audio) if require_audio else {}
    last_error: Optional[Exception] = None
    for kwargs in _processor_call_candidates(processor, text, mm, require_audio=require_audio):
        try:
            encoded = processor(**kwargs)
            return move_to_device(dict(encoded), device)
        except Exception as exc:
            last_error = exc
    if require_audio:
        raise RuntimeError(f"Failed to build audio model inputs from processor; refusing text-only fallback: {last_error}")
    raise RuntimeError(f"Failed to build model inputs from processor: {last_error}")


def prepare_inputs(processor: Any, prompt: str, audio_path: Optional[str], device: torch.device) -> Dict[str, Any]:
    messages = build_messages(prompt, audio_path)
    return prepare_messages_inputs(
        processor, messages, device, require_audio=audio_path is not None)


def input_audio_duration_ms(audio_path: Optional[str]) -> Optional[float]:
    if not audio_path:
        return None
    info = sf.info(audio_path)
    if info.samplerate <= 0:
        return None
    return 1000.0 * info.frames / info.samplerate


def sample_next_token(
    logits: torch.Tensor,
    temperature: float = 0.0,
    top_k: int = 0,
    top_p: float = 1.0,
    repetition_penalty: float = 1.0,
    prev_tokens: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Sample the next token id.

    Supports greedy (temperature<=0), temperature + top_k + top_p sampling, and
    an optional repetition_penalty applied to previously generated tokens
    (`prev_tokens`, shape [B, T]), mirroring HF/Qwen3-TTS generate semantics.
    """
    logits = logits.clone()
    # Repetition penalty (applied before temperature, HF convention).
    if repetition_penalty and repetition_penalty != 1.0 and prev_tokens is not None and prev_tokens.numel() > 0:
        for b in range(logits.size(0)):
            uniq = torch.unique(prev_tokens[b])
            score = logits[b, uniq]
            # HF rule: divide positive logits, multiply negative logits.
            score = torch.where(score > 0, score / repetition_penalty, score * repetition_penalty)
            logits[b, uniq] = score

    if temperature <= 0:
        return torch.argmax(logits, dim=-1)
    logits = logits / max(temperature, 1e-6)
    if top_k and top_k > 0 and top_k < logits.size(-1):
        values, indices = torch.topk(logits, top_k, dim=-1)
        probs = torch.softmax(values, dim=-1)
        picked = torch.multinomial(probs, num_samples=1).squeeze(-1)
        return indices.gather(-1, picked.unsqueeze(-1)).squeeze(-1)
    if top_p and 0.0 < top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
        sorted_probs = torch.softmax(sorted_logits, dim=-1)
        cumprobs = torch.cumsum(sorted_probs, dim=-1)
        # Mask tokens whose cumulative prob exceeds top_p (keep at least 1).
        remove = cumprobs - sorted_probs > top_p
        sorted_logits[remove] = float("-inf")
        probs = torch.softmax(sorted_logits, dim=-1)
        picked = torch.multinomial(probs, num_samples=1).squeeze(-1)
        return sorted_idx.gather(-1, picked.unsqueeze(-1)).squeeze(-1)
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)


def split_thinker_kwargs(inputs: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    if "input_ids" not in inputs:
        raise ValueError("processor output is missing input_ids")
    input_ids = inputs["input_ids"]
    attention_mask = inputs.get("attention_mask")
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)
    kwargs = {k: v for k, v in inputs.items() if k not in {"input_ids", "attention_mask"}}
    return input_ids, attention_mask, kwargs


class _FirstTokenTracker:
    """Minimal HF streamer for latency and the existing per-token callback."""

    def __init__(
        self,
        *,
        processor: Any,
        metrics: LatencyMetrics,
        started_at: float,
        stop_ids: set[int],
        callback: Optional[TextCallback],
    ) -> None:
        self.processor = processor
        self.metrics = metrics
        self.started_at = started_at
        self.stop_ids = stop_ids
        self.callback = callback
        self._expect_prompt = True

    def put(self, value: Any) -> None:
        if self._expect_prompt:
            self._expect_prompt = False
            return
        if torch.is_tensor(value):
            token_ids = value.detach().reshape(-1).tolist()
        elif isinstance(value, (list, tuple)):
            token_ids = list(value)
        else:
            token_ids = [value]
        if self.metrics.first_token_latency_ms is None and token_ids:
            self.metrics.first_token_latency_ms = (
                time.perf_counter() - self.started_at
            ) * 1000.0
        if self.callback is None:
            return
        for raw_token in token_ids:
            token = int(raw_token)
            if token not in self.stop_ids:
                self.callback(decode_text(self.processor, [token]), token)

    def end(self) -> None:
        """HF streamer completion hook."""


def _generate_text_with_thinker_cache(
    model: Any,
    processor: Any,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    thinker_kwargs: Dict[str, Any],
    sample: InferenceSample,
    config: InferenceConfig,
    *,
    stream: bool,
    callback: Optional[TextCallback],
) -> TextGenerationResult:
    """Generate through the Thinker's native HF KV-cache implementation."""
    thinker_temperature, thinker_top_k, thinker_top_p = thinker_sampling_params(
        config
    )
    stop_ids = set(eos_token_ids(processor))
    metrics = LatencyMetrics(
        sample_id=sample.sample_id,
        task=sample.task,
        output="text",
        input_audio_duration_ms=input_audio_duration_ms(sample.audio),
        extra=new_optimization_metrics(),
    )
    started_at = time.perf_counter()
    tracker = _FirstTokenTracker(
        processor=processor,
        metrics=metrics,
        started_at=started_at,
        stop_ids=stop_ids,
        callback=callback if stream else None,
    )
    tokenizer = get_tokenizer(processor)
    generation_kwargs: Dict[str, Any] = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        **thinker_kwargs,
        "max_new_tokens": config.max_new_tokens,
        "do_sample": thinker_temperature > 0,
        "top_p": thinker_top_p,
        "repetition_penalty": 1.0,
        "use_cache": True,
        "return_dict_in_generate": True,
        "output_hidden_states": False,
        "streamer": tracker,
    }
    if stop_ids:
        generation_kwargs["eos_token_id"] = sorted(stop_ids)
    pad_id = getattr(tokenizer, "pad_token_id", None)
    if pad_id is None:
        pad_id = getattr(tokenizer, "eos_token_id", None)
    if pad_id is not None:
        generation_kwargs["pad_token_id"] = int(pad_id)
    if thinker_temperature > 0:
        generation_kwargs["temperature"] = thinker_temperature
        generation_kwargs["top_k"] = thinker_top_k

    with torch.inference_mode(), timed_phase(
        metrics.extra, "thinker_generate_ms", input_ids.device,
        mode=config.latency_profiling,
    ):
        result = model.model.thinker.generate(**generation_kwargs)
    sequences = result.sequences if hasattr(result, "sequences") else result
    suffix = sequences[0, input_ids.size(1):].detach().cpu().tolist()
    generated: List[int] = []
    for raw_token in suffix:
        token = int(raw_token)
        if token in stop_ids:
            break
        generated.append(token)
    metrics.total_latency_ms = (time.perf_counter() - started_at) * 1000.0
    metrics.generated_text_tokens = len(generated)
    return TextGenerationResult(
        decode_text(processor, generated), generated, metrics
    )


def generate_text(
    model: Any,
    processor: Any,
    inputs: Dict[str, Any],
    sample: InferenceSample,
    config: InferenceConfig,
    *,
    stream: bool = False,
    callback: Optional[TextCallback] = None,
) -> TextGenerationResult:
    input_ids, attention_mask, thinker_kwargs = split_thinker_kwargs(inputs)
    thinker_temperature, thinker_top_k, thinker_top_p = thinker_sampling_params(
        config
    )
    if config.use_thinker_kv_cache:
        return _generate_text_with_thinker_cache(
            model,
            processor,
            input_ids,
            attention_mask,
            thinker_kwargs,
            sample,
            config,
            stream=stream,
            callback=callback,
        )
    cur_ids = input_ids.clone()
    cur_mask = attention_mask.clone()
    stop_ids = set(eos_token_ids(processor))
    generated: List[int] = []
    metrics = LatencyMetrics(
        sample_id=sample.sample_id,
        task=sample.task,
        output="text",
        input_audio_duration_ms=input_audio_duration_ms(sample.audio),
        extra=new_optimization_metrics(),
    )
    start = time.perf_counter()
    first_token_at: Optional[float] = None

    with torch.inference_mode():
        for _ in range(config.max_new_tokens):
            with timed_phase(
                metrics.extra, "thinker_generate_ms", input_ids.device,
                mode=config.latency_profiling,
            ):
                out = model.model.run_thinker(
                    input_ids=cur_ids,
                    attention_mask=cur_mask,
                    output_text_embeds=False,
                    **thinker_kwargs,
                )
            logits = out.get("logits")
            if logits is None:
                raise RuntimeError("Thinker did not return logits; text generation is unavailable.")
            next_id = sample_next_token(
                logits[:, -1, :],
                thinker_temperature,
                thinker_top_k,
                thinker_top_p,
            )
            token = int(next_id.item())
            if first_token_at is None:
                first_token_at = time.perf_counter()
                metrics.first_token_latency_ms = (first_token_at - start) * 1000.0
            if token in stop_ids:
                break
            generated.append(token)
            if stream and callback is not None:
                callback(decode_text(processor, [token]), token)
            cur_ids = torch.cat([cur_ids, next_id[:, None]], dim=1)
            cur_mask = torch.cat([cur_mask, torch.ones_like(next_id[:, None])], dim=1)

    metrics.total_latency_ms = (time.perf_counter() - start) * 1000.0
    metrics.generated_text_tokens = len(generated)
    return TextGenerationResult(decode_text(processor, generated), generated, metrics)


def _codec_config(model_config: Any) -> Dict[str, Any]:
    talker_config = getattr(model_config, "talker_config", None)
    return {
        "codec_bos_id": int(getattr(model_config, "codec_bos_id", 2149)),
        "codec_eos_id": int(getattr(model_config, "codec_eos_id", 2150)),
        "codec_pad_id": int(getattr(model_config, "codec_pad_id", 2148)),
        "codec_think_id": int(getattr(model_config, "codec_think_id", 2154)),
        "codec_nothink_id": int(getattr(model_config, "codec_nothink_id", 2155)),
        "codec_think_bos_id": int(getattr(model_config, "codec_think_bos_id", 2156)),
        "codec_think_eos_id": int(getattr(model_config, "codec_think_eos_id", 2157)),
        "codec_language_id": dict(getattr(model_config, "codec_language_id", None) or {}),
        "num_residual_groups": int(getattr(talker_config, "num_code_groups", 15)),
        "residual_vocab_size": int(getattr(talker_config, "code_predictor_vocab_size", 2048)),
    }


_LANG_ALIASES = {
    "en": "english", "eng": "english", "english": "english",
    "zh": "chinese", "cn": "chinese", "chinese": "chinese", "mandarin": "chinese",
    "de": "german", "german": "german",
    "it": "italian", "italian": "italian",
    "pt": "portuguese", "portuguese": "portuguese",
    "es": "spanish", "spanish": "spanish",
    "ja": "japanese", "jp": "japanese", "japanese": "japanese",
    "ko": "korean", "korean": "korean",
    "fr": "french", "french": "french",
    "ru": "russian", "russian": "russian",
}


def build_control_prefix(model_config: Any, target_lang: Optional[str]) -> List[int]:
    cfg = _codec_config(model_config)
    lang_id = None
    if target_lang and cfg["codec_language_id"]:
        key = _LANG_ALIASES.get(target_lang.strip().lower(), target_lang.strip().lower())
        lang_id = cfg["codec_language_id"].get(key)
    if lang_id is not None:
        prefix = [cfg["codec_think_id"], cfg["codec_think_bos_id"], int(lang_id), cfg["codec_think_eos_id"]]
    else:
        prefix = [cfg["codec_nothink_id"], cfg["codec_think_bos_id"], cfg["codec_think_eos_id"]]
    prefix += [cfg["codec_pad_id"], cfg["codec_bos_id"]]
    return prefix


def resolve_audio_token_id(config: Any) -> Optional[int]:
    node = config
    for _ in range(4):
        if node is None:
            break
        for attr in ("audio_token_id", "audio_token_index"):
            value = getattr(node, attr, None)
            if value is not None:
                return int(value)
        node = getattr(node, "thinker_config", None)
    return None


def build_streaming_visible_schedule(
    input_ids: torch.Tensor,
    config: Any,
    codec_len: int,
    control_prefix_len: int,
    frame_ms: float,
    source_duration_ms: Optional[float],
    extra_lag_ms: float = 0.0,
) -> Optional[torch.Tensor]:
    audio_token_id = resolve_audio_token_id(config)
    if audio_token_id is None or source_duration_ms is None or source_duration_ms <= 0 or frame_ms <= 0:
        return None
    ids = input_ids.flatten()
    audio_pos = (ids == audio_token_id).nonzero(as_tuple=False).flatten()
    if audio_pos.numel() == 0:
        return None
    a0 = int(audio_pos[0].item())
    m = int(audio_pos.numel())
    sem_len = int(ids.numel())
    main_len = max(0, codec_len - control_prefix_len)
    if codec_len <= 0:
        return None
    vis = torch.empty(codec_len, dtype=torch.long, device=input_ids.device)
    if main_len == 0:
        vis[:] = min(a0, sem_len)
        return vis
    frame_idx = torch.arange(main_len, dtype=torch.float64, device=input_ids.device)
    t_ms = frame_idx * float(frame_ms) + float(extra_lag_ms)
    frac = torch.clamp(t_ms / float(source_duration_ms), min=0.0, max=1.0)
    arrived = torch.round(frac * float(m)).long().clamp_(0, m)
    vis_main = torch.clamp(a0 + arrived, max=min(a0 + m, sem_len))
    vis[:control_prefix_len] = int(vis_main[0].item())
    vis[control_prefix_len:] = vis_main
    vis, _ = torch.cummax(vis, dim=0)
    return vis


def _normalize_decode_output(obj: Any) -> Tuple[np.ndarray, Optional[int]]:
    sr = None
    if isinstance(obj, dict):
        for key in ("sampling_rate", "sample_rate", "sr"):
            if key in obj:
                sr = int(obj[key])
                break
        for key in ("wav", "waveform", "audio", "samples", "array"):
            if key in obj:
                obj = obj[key]
                break
    elif isinstance(obj, (list, tuple)) and len(obj) == 2:
        first, second = obj
        if isinstance(second, (int, float)):
            obj, sr = first, int(second)
        elif isinstance(first, (int, float)):
            obj, sr = second, int(first)
    if torch.is_tensor(obj):
        arr = obj.detach().cpu().float().numpy()
    else:
        arr = np.asarray(obj, dtype=np.float32)
    arr = np.squeeze(arr)
    if arr.ndim > 1:
        # Prefer mono for test scripts; keep the longest dimension as time.
        if arr.shape[0] <= 8 and arr.shape[1] > arr.shape[0]:
            arr = arr.mean(axis=0)
        else:
            arr = arr.mean(axis=-1)
    return arr.astype(np.float32, copy=False), sr


def codec_sample_rate(codec_tokenizer: Any) -> int:
    tokenizer = codec_tokenizer.tokenizer
    for method in ("get_output_sample_rate", "get_sample_rate", "get_input_sample_rate"):
        if hasattr(tokenizer, method):
            try:
                sr = int(getattr(tokenizer, method)())
                if sr > 0:
                    return sr
            except Exception:
                pass
    return DEFAULT_INPUT_SAMPLE_RATE


def decode_codec(codec_tokenizer: Any, codes: np.ndarray) -> Tuple[np.ndarray, int]:
    """Decode `[T, C]` codec codes to mono waveform.

    Qwen3-TTS tokenizer V2 decode expects ``audio_codes`` as a list of
    ``[T, num_quantizers]`` tensors (matching the encode output format).
    ``num_quantizers`` is 16 (1 main + 15 residual).
    """
    tokenizer = codec_tokenizer.tokenizer
    codes_np = np.asarray(codes, dtype=np.int64)
    if codes_np.ndim == 1:
        codes_np = codes_np[:, None]
    # codes_np is [T, C]. Pad/truncate C to 16 (num_quantizers).
    target_groups = 16
    if codes_np.shape[1] < target_groups:
        padding = np.zeros((codes_np.shape[0], target_groups - codes_np.shape[1]), dtype=np.int64)
        codes_np = np.concatenate([codes_np, padding], axis=1)
    elif codes_np.shape[1] > target_groups:
        codes_np = codes_np[:, :target_groups]

    # Sanitize: the talker may generate out-of-distribution ids (e.g. control
    # tokens codec_pad/bos/eos/think with id >= 2148, or ids >= codebook size)
    # when the codec_head is unfrozen and under-trained. Replace control tokens
    # with 0 and clamp all codes to the acoustic codebook range [0, 2047] so
    # _codebook.decode does not raise IndexError. This is an inference-side
    # safety net; the root cause is model generation quality, not decoding.
    codes_np = np.where(codes_np >= 2148, 0, codes_np)
    codes_np = np.clip(codes_np, 0, 2047)

    # Build the expected input: list with a single [T, C] tensor.
    codes_tensor = torch.as_tensor(codes_np).long().to(
        getattr(tokenizer, 'device', 'cpu')
        if hasattr(tokenizer, 'device') and isinstance(getattr(tokenizer, 'device'), torch.device) else 'cpu')
    enc_out = type('EncodedOutput', (), {'audio_codes': [codes_tensor]})()

    try:
        wavs, sr = tokenizer.decode(enc_out)
        wav = wavs[0] if isinstance(wavs, list) else wavs
        wav = np.asarray(wav, dtype=np.float32).flatten()
        return wav, int(sr or codec_sample_rate(codec_tokenizer))
    except Exception as e:
        raise RuntimeError(f"codec decode failed (codes shape={codes_np.shape}): {e}")


def is_non_silent(wav: np.ndarray, threshold: float) -> bool:
    if wav.size == 0:
        return False
    peak = float(np.max(np.abs(wav)))
    rms = float(np.sqrt(np.mean(np.square(wav.astype(np.float32)))))
    return peak >= threshold or rms >= threshold


def _crossfade(prev_tail: np.ndarray, new_head: np.ndarray) -> np.ndarray:
    """Linear crossfade between the end of the previous chunk and the start of
    the new chunk, to smooth window boundaries. Borrowed from
    Qwen3-TTS-streaming ``_crossfade``."""
    n = min(len(prev_tail), len(new_head))
    if n <= 0:
        return new_head
    w = np.linspace(0.0, 1.0, n, dtype=np.float32)
    return prev_tail[:n] * (1.0 - w) + new_head[:n] * w


def generate_audio(
    model: Any,
    processor: Any,
    model_config: Any,
    inputs: Dict[str, Any],
    sample: InferenceSample,
    config: InferenceConfig,
    *,
    stream: bool = False,
    realtime_schedule: bool = False,
    callback: Optional[AudioCallback] = None,
    text_token_ids: Optional[List[int]] = None,
    use_speaker: bool = False,
) -> AudioGenerationResult:
    input_ids, attention_mask, thinker_kwargs = split_thinker_kwargs(inputs)
    codec_tokenizer = get_codec_tokenizer(
        tokenizer_id=config.codec_tokenizer,
        device_map=config.codec_tokenizer_device,
    )
    target_lang = sample.target_lang or config.target_lang
    codec_cfg = _codec_config(model_config)
    eos_id = codec_cfg["codec_eos_id"]
    # control_prefix: full_add 布局用 [codec_mask×t_prompt, think/lang, (spk), pad, bos]，
    # reply_add 用 [think/lang, (spk), pad, bos]，与训练 template 一致。
    full_add = os.environ.get("QWEN_OMNISPEAK_FULL_ADD_LAYOUT", "").lower() in ("1", "true", "yes", "on")
    codec_mask_id = int(getattr(model_config, "codec_mask_id", codec_cfg["codec_pad_id"]))
    _lang_key = _LANG_ALIASES.get((target_lang or "").strip().lower(), (target_lang or "").strip().lower())
    lang_id = codec_cfg["codec_language_id"].get(_lang_key)
    if lang_id is not None:
        think_part = [codec_cfg["codec_think_id"], codec_cfg["codec_think_bos_id"],
                      int(lang_id), codec_cfg["codec_think_eos_id"]]
    else:
        think_part = [codec_cfg["codec_nothink_id"], codec_cfg["codec_think_bos_id"],
                      codec_cfg["codec_think_eos_id"]]
    speaker_slot_index = None
    if full_add:
        t_prompt = int(input_ids.size(1))
        prefix = [codec_mask_id] * t_prompt + think_part
    else:
        prefix = list(think_part)
    if use_speaker:
        speaker_slot_index = len(prefix)
        prefix = prefix + [codec_cfg["codec_pad_id"]]  # speaker slot placeholder
    prefix = prefix + [codec_cfg["codec_pad_id"], codec_cfg["codec_bos_id"]]
    codec_ids = torch.tensor([prefix], dtype=torch.long, device=input_ids.device)
    # speaker: 源音频波形 (s2s 保源说话人音色)
    reference_audio_wav = None
    reference_audio_sr = None
    if use_speaker and sample.audio:
        from speaker_encoder import load_reference_waveform
        _rw, _rs = load_reference_waveform(sample.audio)
        reference_audio_wav = [_rw]
        reference_audio_sr = [_rs]
    codec_buffer = CodecFrameBuffer(
        config.max_codec_frames,
        codec_cfg["num_residual_groups"],
        device=(
            input_ids.device
            if config.use_gpu_codec_buffer
            else torch.device("cpu")
        ),
        use_gpu=config.use_gpu_codec_buffer,
    )
    metrics = LatencyMetrics(
        sample_id=sample.sample_id,
        task=sample.task,
        output="audio",
        input_audio_duration_ms=input_audio_duration_ms(sample.audio),
        extra=new_optimization_metrics(),
    )
    frame_ms = codec_tokenizer.frame_ms()
    source_duration = metrics.input_audio_duration_ms
    # Windowed-streaming state (only used when config.streaming is True).
    stream_on = bool(getattr(config, "streaming", False))
    stream_emit_every = max(1, int(getattr(config, "stream_emit_every", 4)))
    stream_window = max(stream_emit_every, int(getattr(config, "stream_decode_window", 80)))
    stream_overlap = max(0, int(getattr(config, "stream_overlap_samples", 0)))
    compiled_stream_decoder = None
    if config.compile_streaming_decoder:
        decoder_cache = getattr(
            codec_tokenizer, "_compiled_streaming_decoders", None
        )
        if decoder_cache is None:
            decoder_cache = {}
            setattr(
                codec_tokenizer, "_compiled_streaming_decoders", decoder_cache
            )
        if stream_window not in decoder_cache:
            compiled_stream_decoder = CompiledStreamingDecoder(
                codec_tokenizer, window_frames=stream_window
            )
            metrics.extra["compile_warmup_ms"] += (
                compiled_stream_decoder.warmup()
            )
            decoder_cache[stream_window] = compiled_stream_decoder
        else:
            compiled_stream_decoder = decoder_cache[stream_window]
    start = time.perf_counter()
    first_audio_at: Optional[float] = None
    emitted_samples = 0
    final_wav = np.zeros(0, dtype=np.float32)
    final_sr = codec_sample_rate(codec_tokenizer)
    stream_frames_since_emit = 0
    stream_frames_emitted = 0
    stream_tail: Optional[np.ndarray] = None
    runner_key = (
        bool(config.use_code_predictor_kv_cache),
        bool(config.compile_code_predictor),
        config.code_predictor_compile_mode,
    )
    runner_cache = getattr(model, "_code_predictor_runners", None)
    if runner_cache is None:
        runner_cache = {}
        setattr(model, "_code_predictor_runners", runner_cache)
    if runner_key not in runner_cache:
        runner_cache[runner_key] = CodePredictorRunner(
            model.talker,
            use_kv_cache=config.use_code_predictor_kv_cache,
            compile_model=config.compile_code_predictor,
            compile_mode=config.code_predictor_compile_mode,
        )
    predictor_runner = runner_cache[runner_key]

    def _buffer_numpy() -> np.ndarray:
        with timed_phase(
            metrics.extra, "codec_transfer_ms", input_ids.device,
            mode=config.latency_profiling,
        ):
            return codec_buffer.to_numpy()

    def _decode_stream_window(
        window_codes: torch.Tensor,
    ) -> Tuple[np.ndarray, int]:
        if compiled_stream_decoder is not None:
            with timed_phase(
                metrics.extra, "stream_decoder_ms", input_ids.device,
                mode=config.latency_profiling,
            ):
                return compiled_stream_decoder.decode(window_codes)
        with timed_phase(
            metrics.extra, "codec_transfer_ms", input_ids.device,
            mode=config.latency_profiling,
        ):
            host_codes = (
                window_codes.detach().cpu().contiguous().numpy()
            )
        with timed_phase(
            metrics.extra, "stream_decoder_ms", input_ids.device,
            mode=config.latency_profiling,
        ):
            return decode_codec(codec_tokenizer, host_codes)

    # Append the autoregressively-generated translation text as the reply span
    # so thinker_hidden covers [source audio prompt + translation reply], and
    # thinker_text_labels marks the reply positions. This makes the talker see
    # the SAME full_add prompt+reply layout as training. Without it the forward
    # falls back to text_labels=None and full_add degrades (no reply段).
    thinker_text_labels: Optional[torch.Tensor] = None
    if text_token_ids:
        reply_ids = torch.tensor([text_token_ids], dtype=torch.long, device=input_ids.device)
        thinker_in_ids = torch.cat([input_ids, reply_ids], dim=1)
        thinker_in_mask = torch.cat([attention_mask, torch.ones_like(reply_ids)], dim=1)
        thinker_text_labels = torch.full_like(thinker_in_ids, -100)
        thinker_text_labels[:, input_ids.size(1):] = reply_ids
    else:
        thinker_in_ids = input_ids
        thinker_in_mask = attention_mask

    with torch.inference_mode():
        with timed_phase(
            metrics.extra, "thinker_finalize_ms", input_ids.device,
            mode=config.latency_profiling,
        ):
            thinker_out = model.model.run_thinker(
                input_ids=thinker_in_ids,
                attention_mask=thinker_in_mask,
                output_text_embeds=True,
                **thinker_kwargs,
            )
        thinker_hidden = thinker_out["hidden_states"]
        thinker_text_embeds = thinker_out["text_embeds"]

        # ── Optional inference-only KV-cache path (upstream reply_add/full_add).
        # Uses the model's cached talker session so each frame recomputes O(1)
        # attention instead of the O(T^2) full-sequence forward below.
        use_kv = bool(getattr(config, "use_kv_cache", False))
        kv_session = None
        kv_last_hidden = None
        if use_kv:
            source_xvector = None
            spk_slot_tensor = None
            if use_speaker and speaker_slot_index is not None and reference_audio_wav is not None:
                try:
                    from speaker_encoder import get_speaker_embedder
                    embedder = get_speaker_embedder(
                        enc_dim=int(getattr(model_config, "speaker_embedding_dim", 2048)))
                    wav0 = reference_audio_wav[0]
                    sr0 = int(reference_audio_sr[0]) if reference_audio_sr is not None else 24000
                    xvec = embedder.extract(wav0, sr0, device=input_ids.device,
                                            dtype=thinker_hidden.dtype)
                    source_xvector = xvec.unsqueeze(0)
                    spk_slot_tensor = torch.tensor([speaker_slot_index], dtype=torch.long,
                                                   device=input_ids.device)
                except Exception as exc:  # fall back to full path on any speaker failure
                    raise RuntimeError(
                        f"KV-cache path failed to precompute source x-vector: {exc}")
            kv_session = model.make_cached_talker_session(
                thinker_text_embeds=thinker_text_embeds,
                thinker_hidden_states=thinker_hidden,
                thinker_attention_mask=thinker_in_mask,
                thinker_text_labels=thinker_text_labels,
                control_prefix_ids=codec_ids,
                speaker_slot_index=spk_slot_tensor,
                source_xvector=source_xvector,
                max_frames=config.max_codec_frames,
            )

        for _ in range(config.max_codec_frames):
            if use_kv:
                if kv_session is None:
                    raise RuntimeError("KV-cache session was not initialized")
                if len(codec_buffer) == 0:
                    with timed_phase(
                        metrics.extra, "talker_prefill_ms", input_ids.device,
                        mode=config.latency_profiling,
                    ):
                        logits_full, kv_last_hidden = kv_session.prefill()
                else:
                    with timed_phase(
                        metrics.extra, "talker_step_ms", input_ids.device,
                        mode=config.latency_profiling,
                    ):
                        logits_full, kv_last_hidden = kv_session.step(
                            codec_buffer.last_main()
                        )
                logits = logits_full
            else:
                codec_attention_mask = torch.ones_like(codec_ids, dtype=torch.long)
                schedule = None
                if realtime_schedule:
                    schedule = build_streaming_visible_schedule(
                        input_ids=input_ids,
                        config=model_config,
                        codec_len=codec_ids.size(1),
                        control_prefix_len=len(prefix),
                        frame_ms=frame_ms,
                        source_duration_ms=source_duration,
                        extra_lag_ms=config.streaming_extra_lag_ms,
                    )
                    if schedule is not None:
                        schedule = schedule.unsqueeze(0)
                talker_metric = (
                    "talker_prefill_ms"
                    if len(codec_buffer) == 0
                    else "talker_step_ms"
                )
                with timed_phase(
                    metrics.extra, talker_metric, input_ids.device,
                    mode=config.latency_profiling,
                ):
                    out = model(
                        thinker_text_embeds=thinker_text_embeds,
                        thinker_hidden_states=thinker_hidden,
                        thinker_attention_mask=thinker_in_mask,
                        codec_input_ids=codec_ids,
                        codec_attention_mask=codec_attention_mask,
                        codec_to_sem_visible=schedule,
                        thinker_text_labels=thinker_text_labels,
                        reference_audio_wav=reference_audio_wav,
                        reference_audio_sr=reference_audio_sr,
                        speaker_slot_index=(torch.tensor([speaker_slot_index], dtype=torch.long, device=codec_ids.device) if speaker_slot_index is not None else None),
                        return_dict=True,
                    )
                logits = out.main_codec_logits
            if logits is None:
                raise RuntimeError("Model did not return main_codec_logits; audio generation is unavailable.")
            prev_main = (
                codec_buffer.main_history().to(logits.device)
                if (
                    config.repetition_penalty
                    and config.repetition_penalty != 1.0
                    and len(codec_buffer) > 0
                )
                else None
            )
            next_main = sample_next_token(
                logits[:, -1, :],
                temperature=config.temperature,
                top_k=config.top_k,
                top_p=config.top_p,
                repetition_penalty=config.repetition_penalty,
                prev_tokens=prev_main,
            )
            main_id = int(next_main.item())
            if main_id == eos_id:
                break

            if use_kv:
                codec_hidden = kv_last_hidden
                if codec_hidden is None or codec_hidden.size(1) == 0:
                    raise RuntimeError("KV-cache session returned empty codec_hidden")
            else:
                last_hidden = out.talker_hidden_states
                if last_hidden is None or last_hidden.size(1) == 0:
                    # Fallback: use the logits directly to get a hidden via the
                    # codec head's weight (not ideal but works for residual pred).
                    raise RuntimeError(f"talker_hidden_states is empty/None. main_codec_logits shape={logits.shape}")
                # Upstream path: sem_block is text_channel, talker_hidden_states
                # includes the full codec sequence. Take the last frame.
                codec_hidden = last_hidden[:, -1:, :]
            previous_warmup_ms = predictor_runner.compile_warmup_ms
            with timed_phase(
                metrics.extra, "code_predictor_ms", input_ids.device,
                mode=config.latency_profiling,
            ):
                residual = predictor_runner.predict(codec_hidden, next_main)
            warmup_delta_ms = (
                predictor_runner.compile_warmup_ms - previous_warmup_ms
            )
            if warmup_delta_ms > 0:
                metrics.extra["compile_warmup_ms"] += warmup_delta_ms
                metrics.extra["code_predictor_ms"] = max(
                    0.0,
                    metrics.extra["code_predictor_ms"] - warmup_delta_ms,
                )

            codec_buffer.append(next_main, residual)
            codec_ids = torch.cat([codec_ids, next_main[:, None]], dim=1)

            should_emit = stream and (
                len(codec_buffer) % max(1, config.codec_chunk_frames) == 0
            )
            if stream_on and stream and callback is not None:
                # Windowed streaming: decode only the most recent `stream_window`
                # frames and emit the tail worth of new samples. O(1) per chunk
                # instead of re-decoding the whole sequence.
                stream_frames_since_emit += 1
                if stream_frames_since_emit >= stream_emit_every:
                    stream_frames_since_emit = 0
                    win_start = max(0, len(codec_buffer) - stream_window)
                    win_codes_tensor = codec_buffer.window(win_start)
                    wav, sr = _decode_stream_window(win_codes_tensor)
                    final_sr = sr
                    # New frames since last emit -> tail samples of this window.
                    new_frames = len(codec_buffer) - stream_frames_emitted
                    spf = len(wav) / max(1, win_codes_tensor.shape[0])
                    n_new = int(round(new_frames * spf))
                    chunk = wav[-n_new:] if n_new > 0 else wav
                    if stream_tail is not None and stream_overlap > 0:
                        ov = min(stream_overlap, len(stream_tail), len(chunk))
                        if ov > 0:
                            head = _crossfade(stream_tail[-ov:], chunk[:ov])
                            chunk = np.concatenate([head, chunk[ov:]], axis=0)
                    stream_tail = chunk.copy()
                    stream_frames_emitted = len(codec_buffer)
                    if chunk.size:
                        now = time.perf_counter()
                        if first_audio_at is None and is_non_silent(chunk, config.silence_threshold):
                            first_audio_at = now
                            metrics.first_audio_latency_ms = (first_audio_at - start) * 1000.0
                        callback(chunk, sr, (now - start) * 1000.0)
            elif should_emit and callback is not None:
                codes_np = _buffer_numpy()
                with timed_phase(
                    metrics.extra, "stream_decoder_ms", input_ids.device,
                    mode=config.latency_profiling,
                ):
                    wav, sr = decode_codec(codec_tokenizer, codes_np)
                final_wav, final_sr = wav, sr
                delta = wav[emitted_samples:]
                emitted_samples = len(wav)
                if delta.size:
                    now = time.perf_counter()
                    if first_audio_at is None and is_non_silent(delta, config.silence_threshold):
                        first_audio_at = now
                        metrics.first_audio_latency_ms = (first_audio_at - start) * 1000.0
                    callback(delta, sr, (now - start) * 1000.0)

    if len(codec_buffer) > 0:
        codes_np = _buffer_numpy()
        final_wav, final_sr = decode_codec(codec_tokenizer, codes_np)
        if stream_on and stream and callback is not None:
            # Flush: emit the remaining frames not yet streamed. Decode a window
            # that includes some already-emitted context for boundary quality,
            # then skip the context samples.
            remaining = len(codec_buffer) - stream_frames_emitted
            if remaining > 0:
                context = min(stream_frames_emitted, max(0, stream_window - remaining))
                start_idx = stream_frames_emitted - context
                win_codes_tensor = codec_buffer.window(start_idx)
                wav, sr = _decode_stream_window(win_codes_tensor)
                final_sr = sr
                spf = len(wav) / max(1, win_codes_tensor.shape[0])
                skip = int(round(context * spf))
                chunk = wav[skip:] if skip > 0 else wav
                if stream_tail is not None and stream_overlap > 0 and chunk.size:
                    ov = min(stream_overlap, len(stream_tail), len(chunk))
                    if ov > 0:
                        head = _crossfade(stream_tail[-ov:], chunk[:ov])
                        chunk = np.concatenate([head, chunk[ov:]], axis=0)
                if chunk.size:
                    now = time.perf_counter()
                    if first_audio_at is None and is_non_silent(chunk, config.silence_threshold):
                        first_audio_at = now
                        metrics.first_audio_latency_ms = (first_audio_at - start) * 1000.0
                    callback(chunk, final_sr, (now - start) * 1000.0)
        elif stream and callback is not None and emitted_samples < len(final_wav):
            delta = final_wav[emitted_samples:]
            if delta.size:
                now = time.perf_counter()
                if first_audio_at is None and is_non_silent(delta, config.silence_threshold):
                    first_audio_at = now
                    metrics.first_audio_latency_ms = (first_audio_at - start) * 1000.0
                callback(delta, final_sr, (now - start) * 1000.0)
    else:
        codes_np = np.zeros((0, 1), dtype=np.int64)

    metrics.total_latency_ms = (time.perf_counter() - start) * 1000.0
    metrics.generated_codec_frames = len(codec_buffer)
    metrics.output_audio_duration_ms = 1000.0 * len(final_wav) / final_sr if final_sr > 0 else None
    if metrics.output_audio_duration_ms and metrics.total_latency_ms:
        metrics.rtf = metrics.total_latency_ms / metrics.output_audio_duration_ms
    return AudioGenerationResult(final_wav, final_sr, codes_np, metrics)


def save_text(path: Path, text: str) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return str(path)


def save_audio(path: Path, wav: np.ndarray, sample_rate: int) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), wav, sample_rate)
    return str(path)


def append_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _validate_output(output: str) -> str:
    if output not in {"text", "audio"}:
        raise ValueError(f"Unsupported output modality: {output!r}; expected 'text' or 'audio'.")
    return output


def _resolve_sample_audio(audio: Optional[str], base_dir: Path) -> Optional[str]:
    if not audio:
        return None
    path = Path(audio).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return str(path)


def read_jsonl_samples(path: str, *, default_task: str, default_output: str) -> Iterator[InferenceSample]:
    jsonl_path = Path(path).expanduser()
    base_dir = jsonl_path.parent
    with jsonl_path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if not line.strip():
                continue
            row = json.loads(line)
            output = _validate_output(str(row.get("output") or default_output))
            yield InferenceSample(
                sample_id=str(row.get("sample_id") or row.get("id") or idx),
                prompt=str(row.get("prompt") or row.get("query") or row.get("text") or ""),
                audio=_resolve_sample_audio(row.get("audio") or row.get("audio_path") or row.get("source_audio"), base_dir),
                task=str(row.get("task") or default_task),
                output=output,
                target_lang=row.get("target_lang"),
            )


def single_sample(sample_id: str, prompt: str, audio: Optional[str], task: str, output: str,
                  target_lang: Optional[str]) -> InferenceSample:
    return InferenceSample(sample_id=sample_id, prompt=prompt, audio=audio, task=task, output=_validate_output(output),
                           target_lang=target_lang)


def write_temp_wav(samples: np.ndarray, sample_rate: int, directory: str, name: str = "chunk.wav") -> str:
    path = Path(directory) / name
    sf.write(str(path), samples, sample_rate)
    return str(path)


def iter_audio_chunks(audio_path: str, chunk_ms: int) -> Iterator[Tuple[np.ndarray, int, float]]:
    data, sr = sf.read(audio_path, dtype="float32", always_2d=False)
    if data.ndim > 1:
        data = data.mean(axis=1)
    chunk = max(1, int(round(sr * chunk_ms / 1000.0)))
    offset = 0
    while offset < len(data):
        next_offset = min(len(data), offset + chunk)
        yield data[offset:next_offset], sr, 1000.0 * next_offset / sr
        offset = next_offset


def require_sounddevice():
    try:
        import sounddevice as sd
        return sd
    except ImportError as exc:
        raise RuntimeError(
            "Realtime microphone/playback requires optional dependency `sounddevice`. "
            "Install with: pip install -r inference/requirements.txt"
        ) from exc


def play_audio(wav: np.ndarray, sample_rate: int, *, blocking: bool = True) -> None:
    sd = require_sounddevice()
    sd.play(wav, sample_rate, blocking=blocking)


def make_output_paths(output_dir: str, sample_id: str) -> Tuple[Path, Path]:
    safe_id = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in sample_id)
    base = Path(output_dir)
    return base / "texts" / f"{safe_id}.txt", base / "audios" / f"{safe_id}.wav"


def print_metric_summary(metrics: LatencyMetrics) -> None:
    print(json.dumps(metrics.to_json(), ensure_ascii=False, indent=2))

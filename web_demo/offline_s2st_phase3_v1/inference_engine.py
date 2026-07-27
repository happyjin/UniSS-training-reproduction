"""Phase3 Quality-only offline speech-to-speech inference engine."""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from evaluation.uniss_outputs import parse_with_tokenizer
from training import constants_uniss as c
from training.generate_unist_eval_audio import truncate_at_eos
from uniss import UniSSTokenizer, process_input

from .audio_io import (
    SAMPLE_RATE,
    cleanup_expired,
    create_request_directory,
    normalize_uploaded_audio,
    split_on_silence,
    stitch_audio,
    write_json,
)
from .config import DemoConfig

ProgressCallback = Callable[[float, str], None]


DIRECTION_TO_TARGET = {
    "中文 → 英文": "<|eng|>",
    "英文 → 中文": "<|cmn|>",
}


def target_language_tag(direction: str) -> str:
    try:
        return DIRECTION_TO_TARGET[direction]
    except KeyError as exc:
        raise ValueError(f"Unsupported translation direction: {direction!r}") from exc


def maximum_identical_run(values: Sequence[int]) -> int:
    best = current = 0
    previous = object()
    for value in values:
        if value == previous:
            current += 1
        else:
            previous = value
            current = 1
        best = max(best, current)
    return best


def quality_output_warnings(parsed: Mapping[str, object]) -> list[str]:
    warnings = []
    if not str(parsed.get("generated_transcription") or "").strip():
        warnings.append("模型没有生成完整的源语音 transcription")
    if not str(parsed.get("generated_translation") or "").strip():
        warnings.append("模型没有生成完整的目标 translation")
    if not parsed.get("has_semantic_start") or not parsed.get("has_semantic_end"):
        warnings.append("模型的 BiCodec semantic 边界不完整")
    if not parsed.get("has_eos"):
        warnings.append("模型输出没有 EOS")
    if not parsed.get("semantic_values"):
        warnings.append("模型没有生成可解码的 semantic token")
    return warnings


@dataclass
class ChunkResult:
    index: int
    input_path: str
    output_path: str | None
    transcription: str
    translation: str
    semantic_token_count: int
    maximum_identical_semantic_run: int
    prompt_tokens: int
    generated_tokens: int
    tokenize_seconds: float
    generation_seconds: float
    decode_seconds: float
    warnings: list[str] = field(default_factory=list)


@dataclass
class InferenceResult:
    request_dir: str
    input_audio_path: str
    output_audio_path: str
    result_json_path: str
    direction: str
    model_label: str
    mode: str
    transcription: str
    translation: str
    input_duration_seconds: float
    output_duration_seconds: float
    total_seconds: float
    warnings: list[str]
    chunks: list[ChunkResult]

    def to_dict(self) -> dict[str, object]:
        value = asdict(self)
        value["chunks"] = [asdict(chunk) for chunk in self.chunks]
        return value


class Phase3QualityEngine:
    """One-model, one-lock inference service for the frozen Phase3 export."""

    def __init__(self, config: DemoConfig):
        config.validate()
        self.config = config
        self.device = torch.device(
            config.device if torch.cuda.is_available() else "cpu"
        )
        self.model = None
        self.tokenizer = None
        self.speech_tokenizer: UniSSTokenizer | None = None
        self.export_manifest: dict[str, object] = {}
        self.lock = threading.Lock()

    @property
    def loaded(self) -> bool:
        return (
            self.model is not None
            and self.tokenizer is not None
            and self.speech_tokenizer is not None
        )

    def load(self, progress: ProgressCallback | None = None) -> None:
        if self.loaded:
            return
        from transformers import AutoModelForCausalLM, AutoTokenizer

        notify = progress or (lambda _fraction, _message: None)
        notify(0.02, "加载 Phase3 tokenizer")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_path,
            local_files_only=True,
            trust_remote_code=False,
        )
        notify(0.08, "加载 Phase3 Qwen 权重")
        dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_path,
            local_files_only=True,
            trust_remote_code=False,
            torch_dtype=dtype,
        ).to(self.device)
        self.model.eval()
        notify(0.16, "加载 GLM tokenizer 和 BiCodec")
        self.speech_tokenizer = UniSSTokenizer.from_pretrained(
            self.config.speech_tokenizer_path,
            device=self.device,
        )
        self.export_manifest = json.loads(
            (self.config.model_path / "export_manifest.json").read_text(
                encoding="utf-8"
            )
        )
        notify(0.2, "Phase3 Quality 模型加载完成")

    def _infer_chunk(
        self,
        chunk_path: Path,
        *,
        direction: str,
        chunk_index: int,
        output_path: Path,
    ) -> ChunkResult:
        assert (
            self.model is not None
            and self.tokenizer is not None
            and self.speech_tokenizer is not None
        )
        target_tag = target_language_tag(direction)
        started = time.perf_counter()
        linguistic_tokens, bicodec_tokens = self.speech_tokenizer.tokenize(chunk_path)
        tokenize_seconds = time.perf_counter() - started
        prompt = process_input(
            linguistic_tokens,
            bicodec_tokens,
            self.config.task_name,
            target_tag,
            speed=1.0,
        )
        prompt_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        model_vocab_size = int(self.model.config.vocab_size)
        suppressed_dummy_ids = list(range(c.VOCAB_SIZE, model_vocab_size))
        generator = torch.Generator(device=self.device)
        generator.manual_seed(self.config.seed + chunk_index)
        generation_kwargs: dict[str, object] = {
            "max_new_tokens": self.config.max_new_tokens,
            "do_sample": self.config.temperature > 0,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "repetition_penalty": self.config.repetition_penalty,
            "pad_token_id": c.TOKEN_PAD,
            "eos_token_id": c.TOKEN_EOS,
            "suppress_tokens": suppressed_dummy_ids,
            "generator": generator,
        }
        started = time.perf_counter()
        with torch.inference_mode():
            generated = self.model.generate(prompt_ids, **generation_kwargs)
        generation_seconds = time.perf_counter() - started
        generated_tail = truncate_at_eos(generated[0, prompt_ids.shape[1] :].tolist())
        parsed = parse_with_tokenizer(
            generated_tail, mode=self.config.mode, tokenizer=self.tokenizer
        )
        semantic_values = [int(value) for value in parsed.get("semantic_values") or []]
        warnings = quality_output_warnings(parsed)
        max_run = maximum_identical_run(semantic_values)
        if max_run >= 100:
            warnings.append(f"semantic token出现长重复，最大连续长度={max_run}")
        audio_path: str | None = None
        decode_seconds = 0.0
        if semantic_values:
            global_values = [int(value) for value in bicodec_tokens[:32]]
            decode_tokens = torch.tensor(
                [*global_values, *semantic_values],
                dtype=torch.long,
                device=self.device,
            )
            started = time.perf_counter()
            with torch.inference_mode():
                waveform = self.speech_tokenizer.decode(decode_tokens)
            decode_seconds = time.perf_counter() - started
            values = np.asarray(waveform, dtype=np.float32).reshape(-1)
            if values.size:
                sf.write(output_path, values, SAMPLE_RATE, subtype="PCM_16")
                audio_path = str(output_path.resolve())
            else:
                warnings.append("BiCodec解码得到空 waveform")
        return ChunkResult(
            index=chunk_index,
            input_path=str(chunk_path.resolve()),
            output_path=audio_path,
            transcription=str(parsed.get("generated_transcription") or "").strip(),
            translation=str(parsed.get("generated_translation") or "").strip(),
            semantic_token_count=len(semantic_values),
            maximum_identical_semantic_run=max_run,
            prompt_tokens=int(prompt_ids.shape[1]),
            generated_tokens=len(generated_tail),
            tokenize_seconds=tokenize_seconds,
            generation_seconds=generation_seconds,
            decode_seconds=decode_seconds,
            warnings=warnings,
        )

    def translate(
        self,
        input_audio: str | Path,
        *,
        direction: str,
        use_silence_chunking: bool = True,
        progress: ProgressCallback | None = None,
    ) -> InferenceResult:
        notify = progress or (lambda _fraction, _message: None)
        cleanup_expired(self.config.output_root, self.config.output_ttl_hours)
        request_dir = create_request_directory(self.config.output_root)
        normalized_path = request_dir / "input_16k.wav"
        metadata = normalize_uploaded_audio(
            input_audio,
            normalized_path,
            max_upload_bytes=self.config.max_upload_bytes,
            min_audio_seconds=self.config.min_audio_seconds,
            max_audio_seconds=self.config.max_audio_seconds,
        )
        waveform, _sample_rate = sf.read(normalized_path, dtype="float32")
        chunks = (
            split_on_silence(
                waveform,
                sample_rate=SAMPLE_RATE,
                max_chunk_seconds=self.config.max_chunk_seconds,
            )
            if use_silence_chunking
            else [np.asarray(waveform, dtype=np.float32)]
        )
        if not chunks:
            raise ValueError("No speech/audio chunks were produced")
        chunks_dir = request_dir / "chunks"
        generated_dir = request_dir / "generated"
        chunks_dir.mkdir()
        generated_dir.mkdir()
        with self.lock:
            self.load(progress=notify)
            started = time.perf_counter()
            results: list[ChunkResult] = []
            for index, chunk in enumerate(chunks):
                fraction = 0.2 + 0.7 * index / max(1, len(chunks))
                notify(fraction, f"处理第 {index + 1}/{len(chunks)} 个音频段")
                chunk_path = chunks_dir / f"chunk_{index:03d}.wav"
                generated_path = generated_dir / f"chunk_{index:03d}.wav"
                sf.write(chunk_path, chunk, SAMPLE_RATE, subtype="PCM_16")
                results.append(
                    self._infer_chunk(
                        chunk_path,
                        direction=direction,
                        chunk_index=index,
                        output_path=generated_path,
                    )
                )
            generated_waves = [
                sf.read(result.output_path, dtype="float32")[0]
                for result in results
                if result.output_path
            ]
            if not generated_waves:
                raise RuntimeError(
                    "Phase3 did not produce any playable translated speech"
                )
            combined = stitch_audio(
                generated_waves,
                sample_rate=SAMPLE_RATE,
                silence_seconds=self.config.chunk_silence_seconds,
            )
            output_path = request_dir / "output_translation.wav"
            sf.write(output_path, combined, SAMPLE_RATE, subtype="PCM_16")
            total_seconds = time.perf_counter() - started
        transcription = " ".join(
            result.transcription for result in results if result.transcription
        ).strip()
        translation = " ".join(
            result.translation for result in results if result.translation
        ).strip()
        warnings = [warning for result in results for warning in result.warnings]
        output_duration = combined.size / SAMPLE_RATE
        result = InferenceResult(
            request_dir=str(request_dir.resolve()),
            input_audio_path=str(normalized_path.resolve()),
            output_audio_path=str(output_path.resolve()),
            result_json_path=str((request_dir / "result.json").resolve()),
            direction=direction,
            model_label=self.config.model_label,
            mode="Quality",
            transcription=transcription,
            translation=translation,
            input_duration_seconds=float(metadata["duration_seconds"]),
            output_duration_seconds=float(output_duration),
            total_seconds=total_seconds,
            warnings=warnings,
            chunks=results,
        )
        payload = result.to_dict()
        payload["input_metadata"] = metadata
        payload["checkpoint_export_manifest"] = self.export_manifest
        write_json(result.result_json_path, payload)
        notify(1.0, "Phase3 Quality 翻译完成")
        return result

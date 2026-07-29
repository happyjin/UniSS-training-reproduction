"""Transcribe generated speech with the paper's language-specific ASR models."""

from __future__ import annotations

import argparse
import json
import math
from itertools import islice
from pathlib import Path
from typing import Iterable, Iterator, Mapping, Sequence

from evaluation.io_utils import iter_jsonl, write_json
from evaluation.sharding import load_keys, select_shard
from training.constants_uniss import normalize_language
from training.generate_unist_eval_audio import write_jsonl_row


WHISPER_ASR_PROTOCOL = "whisper-large-v3-attention-mask-v2"
WHISPER_MAX_WORDS_PER_SECOND = 12.0
WHISPER_MIN_LENGTH_GUARD_WORDS = 64


def chunks(values: Iterable[Mapping[str, object]], size: int) -> Iterator[list[Mapping[str, object]]]:
    source = iter(values)
    while batch := list(islice(source, size)):
        yield batch


def target_asr_backend(language: str) -> str:
    return "whisper-large-v3" if normalize_language(language) == "eng" else "paraformer-zh"


def audio_duration_sort_key(row: Mapping[str, object]) -> float:
    """Sort ASR inputs by duration to minimize variable-length padding."""

    try:
        duration = float(row.get("audio_duration_seconds", math.inf))
    except (TypeError, ValueError):
        return math.inf
    return duration if duration > 0 and math.isfinite(duration) else math.inf


def whisper_duration_bucket(row: Mapping[str, object], *, max_duration_seconds: float) -> str:
    """Keep Whisper pipeline batches on one preprocessing schema.

    Transformers adds ``num_frames`` to short Whisper inputs but omits it for
    inputs longer than the feature extractor window.  Mixing both schemas in a
    single pipeline batch makes its collator fail before inference.
    """

    duration = audio_duration_sort_key(row)
    if not math.isfinite(duration):
        return "unknown"
    return "long" if duration > max_duration_seconds else "short"


def whisper_call_options(bucket_name: str) -> dict[str, object]:
    options: dict[str, object] = {
        "generate_kwargs": {"language": "english", "task": "transcribe"}
    }
    if bucket_name != "short":
        options["return_timestamps"] = True
    return options


def configure_whisper_attention_mask(recognizer: object) -> None:
    """Make batched Whisper ignore padded audio frames during generation.

    Whisper uses the same token for padding and EOS, so its generation code
    cannot infer an attention mask reliably.  Without an explicit mask,
    batched short utterances can continue decoding over padded frames and
    repeat a phrase until the decoder limit even when batch=1 is correct.
    """

    feature_extractor = getattr(recognizer, "feature_extractor", None)
    if feature_extractor is None:
        raise TypeError("Whisper recognizer does not expose a feature_extractor")
    feature_extractor.return_attention_mask = True


def whisper_transcript_is_suspicious(
    row: Mapping[str, object],
    text: str,
    *,
    max_words_per_second: float = WHISPER_MAX_WORDS_PER_SECOND,
    minimum_words: int = WHISPER_MIN_LENGTH_GUARD_WORDS,
) -> bool:
    """Detect decoder-limit hallucinations before they contaminate BLEU."""

    duration = audio_duration_sort_key(row)
    if not math.isfinite(duration):
        return False
    maximum_words = max(minimum_words, math.ceil(duration * max_words_per_second))
    return len(text.split()) > maximum_words


def audio_path(row: Mapping[str, object], *, results_path: Path) -> Path:
    path = Path(str(row["audio_path"]))
    return path if path.is_absolute() else results_path.parent / path


def load_audio_array(path: Path, *, expected_sample_rate: int):
    import soundfile as sf

    audio, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    if sample_rate != expected_sample_rate:
        raise ValueError(f"Expected {expected_sample_rate} Hz audio, got {sample_rate} Hz: {path}")
    return audio.mean(axis=1)


def transcribe_whisper(rows, *, results_path: Path, model_name: str, device: str, batch_size: int):
    import torch
    from transformers import pipeline

    device_index = int(device.split(":", 1)[1]) if device.startswith("cuda:") else -1
    dtype = torch.float16 if device_index >= 0 else torch.float32
    recognizer = pipeline(
        "automatic-speech-recognition",
        model=model_name,
        torch_dtype=dtype,
        device=device_index,
        model_kwargs={"local_files_only": False},
    )
    configure_whisper_attention_mask(recognizer)
    sampling_rate = recognizer.feature_extractor.sampling_rate
    max_duration_seconds = recognizer.feature_extractor.n_samples / sampling_rate
    buckets = {"short": [], "long": [], "unknown": []}
    for row in rows:
        buckets[whisper_duration_bucket(row, max_duration_seconds=max_duration_seconds)].append(row)
    for bucket_name in ("short", "long", "unknown"):
        bucket_rows = buckets[bucket_name]
        # The pipeline collator can pad normal <=30s Whisper features, but its
        # long-form feature tensors have independently sized time axes and fail
        # when more than one is collated.  Preserve batching for the common
        # short case and process only the long/unknown tail one at a time.
        effective_batch_size = batch_size if bucket_name == "short" else 1
        for batch in chunks(bucket_rows, effective_batch_size):
            paths = [str(audio_path(row, results_path=results_path)) for row in batch]
            inputs = [
                load_audio_array(Path(path), expected_sample_rate=sampling_rate)
                for path in paths
            ]
            outputs = recognizer(
                inputs,
                batch_size=effective_batch_size,
                **whisper_call_options(bucket_name),
            )
            if isinstance(outputs, dict):
                outputs = [outputs]
            for row, output in zip(batch, outputs):
                text = str(output.get("text", "")).strip()
                if whisper_transcript_is_suspicious(row, text):
                    raise RuntimeError(
                        "Whisper transcript length guard triggered for "
                        f"id={row.get('id')} mode={row.get('mode')} "
                        f"duration={row.get('audio_duration_seconds')} "
                        f"words={len(text.split())}; refusing to write a likely "
                        "padding hallucination"
                    )
                yield row, text


def transcribe_paraformer(rows, *, results_path: Path, model_name: str, device: str, batch_size: int):
    from funasr import AutoModel

    recognizer = AutoModel(model=model_name, device=device, disable_update=True)
    for batch in chunks(rows, batch_size):
        paths = [str(audio_path(row, results_path=results_path)) for row in batch]
        outputs = recognizer.generate(input=paths, batch_size_s=300)
        if isinstance(outputs, dict):
            outputs = [outputs]
        if len(outputs) != len(batch):
            raise RuntimeError(f"Paraformer returned {len(outputs)} outputs for {len(batch)} inputs")
        for row, output in zip(batch, outputs):
            yield row, str(output.get("text", "")).strip()


def run_asr(args: argparse.Namespace) -> dict[str, int]:
    input_path = Path(args.input)
    output_path = Path(args.output)
    if output_path.exists() and not args.resume:
        raise FileExistsError(f"Refusing to overwrite ASR output: {output_path}")
    completed: set[tuple[str, str]] = load_keys(args.completed_input)
    if args.resume and output_path.exists():
        completed.update((str(row["id"]), str(row["mode"])) for row in iter_jsonl(output_path))

    requested_target_languages = {
        normalize_language(language)
        for language in (getattr(args, "target_language", None) or [])
    }
    rows = [
        row
        for row in select_shard(
            iter_jsonl(input_path),
            num_shards=args.num_shards,
            shard_index=args.shard_index,
        )
        if (str(row["id"]), str(row["mode"])) not in completed
        and row.get("audio_path")
        and not row.get("error")
        and (
            not requested_target_languages
            or normalize_language(str(row["tgt_lang"])) in requested_target_languages
        )
    ]
    by_backend = {
        "whisper-large-v3": [row for row in rows if target_asr_backend(str(row["tgt_lang"])) == "whisper-large-v3"],
        "paraformer-zh": [row for row in rows if target_asr_backend(str(row["tgt_lang"])) == "paraformer-zh"],
    }
    for backend_rows in by_backend.values():
        backend_rows.sort(key=audio_duration_sort_key)
    counts = {"transcribed": 0, "empty": 0, "skipped_existing": len(completed)}
    if by_backend["whisper-large-v3"]:
        for row, text in transcribe_whisper(
            by_backend["whisper-large-v3"],
            results_path=input_path,
            model_name=args.whisper_model,
            device=args.device,
            batch_size=args.batch_size,
        ):
            write_jsonl_row(
                output_path,
                {
                    **row,
                    "asr_text": text,
                    "asr_model": args.whisper_model,
                    "asr_protocol": WHISPER_ASR_PROTOCOL,
                    "asr_attention_mask": True,
                    "asr_batch_size": args.batch_size,
                },
            )
            counts["transcribed"] += 1
            counts["empty"] += int(not text)
            write_json(output_path.with_suffix(".summary.json"), counts)
    if by_backend["paraformer-zh"]:
        for row, text in transcribe_paraformer(
            by_backend["paraformer-zh"],
            results_path=input_path,
            model_name=args.paraformer_model,
            device=args.device,
            batch_size=args.batch_size,
        ):
            write_jsonl_row(output_path, {**row, "asr_text": text, "asr_model": args.paraformer_model})
            counts["transcribed"] += 1
            counts["empty"] += int(not text)
            write_json(output_path.with_suffix(".summary.json"), counts)
    return counts


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--whisper-model", default="openai/whisper-large-v3")
    parser.add_argument(
        "--paraformer-model",
        default="iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--target-language",
        action="append",
        choices=("eng", "cmn"),
        help="Only transcribe rows with this normalized target language; repeat for multiple languages.",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--completed-input", action="append", default=[])
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    print(json.dumps(run_asr(parse_args(argv)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

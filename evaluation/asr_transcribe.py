"""Transcribe generated speech with the paper's language-specific ASR models."""

from __future__ import annotations

import argparse
import json
from itertools import islice
from pathlib import Path
from typing import Iterable, Iterator, Mapping, Sequence

from evaluation.io_utils import iter_jsonl, write_json
from training.constants_uniss import normalize_language
from training.generate_unist_eval_audio import write_jsonl_row


def chunks(values: Iterable[Mapping[str, object]], size: int) -> Iterator[list[Mapping[str, object]]]:
    source = iter(values)
    while batch := list(islice(source, size)):
        yield batch


def target_asr_backend(language: str) -> str:
    return "whisper-large-v3" if normalize_language(language) == "eng" else "paraformer-zh"


def audio_path(row: Mapping[str, object], *, results_path: Path) -> Path:
    path = Path(str(row["audio_path"]))
    return path if path.is_absolute() else results_path.parent / path


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
    for batch in chunks(rows, batch_size):
        paths = [str(audio_path(row, results_path=results_path)) for row in batch]
        outputs = recognizer(
            paths,
            batch_size=batch_size,
            generate_kwargs={"language": "english", "task": "transcribe"},
        )
        if isinstance(outputs, dict):
            outputs = [outputs]
        for row, output in zip(batch, outputs):
            yield row, str(output.get("text", "")).strip()


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
    completed: set[tuple[str, str]] = set()
    if args.resume and output_path.exists():
        completed = {(str(row["id"]), str(row["mode"])) for row in iter_jsonl(output_path)}

    rows = [
        row
        for row in iter_jsonl(input_path)
        if (str(row["id"]), str(row["mode"])) not in completed and row.get("audio_path") and not row.get("error")
    ]
    by_backend = {
        "whisper-large-v3": [row for row in rows if target_asr_backend(str(row["tgt_lang"])) == "whisper-large-v3"],
        "paraformer-zh": [row for row in rows if target_asr_backend(str(row["tgt_lang"])) == "paraformer-zh"],
    }
    counts = {"transcribed": 0, "empty": 0, "skipped_existing": len(completed)}
    if by_backend["whisper-large-v3"]:
        for row, text in transcribe_whisper(
            by_backend["whisper-large-v3"],
            results_path=input_path,
            model_name=args.whisper_model,
            device=args.device,
            batch_size=args.batch_size,
        ):
            write_jsonl_row(output_path, {**row, "asr_text": text, "asr_model": args.whisper_model})
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
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    print(json.dumps(run_asr(parse_args(argv)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

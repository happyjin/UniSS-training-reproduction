"""Generate UniSS validation/test audio from tokenized UniST parquet rows.

This evaluator is intentionally HF-checkpoint based. Megatron training
checkpoints must first be exported to a Hugging Face checkpoint directory, then
this script can synthesize fixed validation/test samples for listening and
downstream metric computation.
"""

from __future__ import annotations

import argparse
import glob
import json
import re
import sys
import time
from pathlib import Path
from typing import Iterable, Iterator, Mapping, Sequence

import pyarrow.parquet as pq
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training import constants_uniss as c
from training import sample_builders as builders
from training.prepare_unist_s2st import normalize_unist_record
from evaluation.io_utils import iter_jsonl, write_json
from evaluation.uniss_outputs import parse_with_tokenizer


EVAL_MODES = ("quality", "performance", "direct_s2st", "tts")


def expand_input_paths(patterns: Sequence[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        matched = [Path(path) for path in glob.glob(pattern)]
        if matched:
            paths.extend(sorted(matched))
        else:
            path = Path(pattern)
            if path.exists():
                paths.append(path)
    unique_paths = sorted(dict.fromkeys(paths))
    if not unique_paths:
        raise FileNotFoundError(f"No parquet files matched: {patterns}")
    return unique_paths


def normalized_limit(limit_records: int | None) -> int | None:
    if limit_records is None or limit_records <= 0:
        return None
    return limit_records


def iter_unist_records(paths: Sequence[Path], limit_records: int | None = None) -> Iterator[dict[str, object]]:
    limit_records = normalized_limit(limit_records)
    emitted = 0
    for path in paths:
        table = pq.read_table(path)
        for raw in table.to_pylist():
            yield normalize_unist_record(raw)
            emitted += 1
            if limit_records is not None and emitted >= limit_records:
                return


def resolve_manifest_parquet_path(value: object, *, manifest_path: Path) -> Path:
    path = Path(str(value))
    if path.is_absolute():
        return path
    repo_candidate = REPO_ROOT / path
    if repo_candidate.exists():
        return repo_candidate
    return manifest_path.parent / path


def iter_manifest_records(manifest_path: Path, limit_records: int | None = None) -> Iterator[dict[str, object]]:
    limit_records = normalized_limit(limit_records)
    tables: dict[Path, object] = {}
    emitted = 0
    for entry in iter_jsonl(manifest_path):
        parquet_path = resolve_manifest_parquet_path(entry["parquet_path"], manifest_path=manifest_path)
        table = tables.get(parquet_path)
        if table is None:
            if not parquet_path.exists():
                raise FileNotFoundError(f"Manifest parquet does not exist: {parquet_path}")
            table = pq.read_table(parquet_path)
            tables[parquet_path] = table
        row_index = int(entry["row_index"])
        if row_index < 0 or row_index >= table.num_rows:  # type: ignore[union-attr]
            raise IndexError(f"Manifest row_index {row_index} is invalid for {parquet_path}")
        raw = table.slice(row_index, 1).to_pylist()[0]  # type: ignore[union-attr]
        if str(raw.get("id")) != str(entry.get("id")):
            raise ValueError(
                f"Manifest id mismatch at {parquet_path}:{row_index}: "
                f"expected {entry.get('id')!r}, found {raw.get('id')!r}"
            )
        yield normalize_unist_record(raw)
        emitted += 1
        if limit_records is not None and emitted >= limit_records:
            return


def load_hf_text_encoder(tokenizer) -> builders.TextEncoder:
    def encode(text: str) -> list[int]:
        return tokenizer.encode(text, add_special_tokens=False)

    return encode


def build_eval_sample(
    record: Mapping[str, object],
    *,
    mode: str,
    text_encoder: builders.TextEncoder,
) -> builders.TrainingSample:
    if mode == "quality":
        return builders.build_quality_sample(
            source_glm=record["source_glm"],  # type: ignore[arg-type]
            bicodec_global=record["bicodec_global"],  # type: ignore[arg-type]
            src_lang=str(record["src_lang"]),
            tgt_lang=str(record["tgt_lang"]),
            transcription=str(record["transcription"]),
            translation=str(record["translation"]),
            target_bicodec=record["target_bicodec"],  # type: ignore[arg-type]
            text_encoder=text_encoder,
            source_id=str(record.get("id", "")) or None,
        )
    if mode == "performance":
        return builders.build_performance_sample(
            source_glm=record["source_glm"],  # type: ignore[arg-type]
            bicodec_global=record["bicodec_global"],  # type: ignore[arg-type]
            tgt_lang=str(record["tgt_lang"]),
            translation=str(record["translation"]),
            target_bicodec=record["target_bicodec"],  # type: ignore[arg-type]
            text_encoder=text_encoder,
            source_id=str(record.get("id", "")) or None,
        )
    if mode == "direct_s2st":
        return builders.build_direct_s2st_sample(
            source_glm=record["source_glm"],  # type: ignore[arg-type]
            bicodec_global=record["bicodec_global"],  # type: ignore[arg-type]
            tgt_lang=str(record["tgt_lang"]),
            target_bicodec=record["target_bicodec"],  # type: ignore[arg-type]
            source_id=str(record.get("id", "")) or None,
        )
    if mode == "tts":
        return builders.build_tts_sample(
            bicodec_global=record["bicodec_global"],  # type: ignore[arg-type]
            src_lang=str(record["src_lang"]),
            transcription=str(record["transcription"]),
            source_bicodec=record["source_bicodec"],  # type: ignore[arg-type]
            text_encoder=text_encoder,
            source_id=str(record.get("id", "")) or None,
        )
    raise ValueError(f"Unsupported eval mode {mode!r}")


def reference_bicodec_values(record: Mapping[str, object], mode: str) -> Sequence[int]:
    if mode == "tts":
        return record["source_bicodec"]  # type: ignore[return-value]
    return record["target_bicodec"]  # type: ignore[return-value]


def truncate_at_eos(token_ids: Sequence[int], eos_token_id: int = c.TOKEN_EOS) -> list[int]:
    output: list[int] = []
    for token_id in token_ids:
        output.append(int(token_id))
        if int(token_id) == eos_token_id:
            break
    return output


def extract_bicodec_semantic_values(token_ids: Iterable[int]) -> list[int]:
    values: list[int] = []
    for token_id in token_ids:
        if c.BICODEC_SEMANTIC_OFFSET <= int(token_id) <= c.BICODEC_SEMANTIC_SPAN.last_id:
            values.append(c.BICODEC_SEMANTIC_SPAN.value_for(int(token_id)))
    return values


def clean_generated_text(text: str) -> str:
    return re.sub(r"<\|.*?\|>", "", text).strip()


def safe_sample_name(index: int, sample_id: object, mode: str) -> str:
    sample_text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(sample_id))[:80]
    return f"{index:05d}_{mode}_{sample_text}"


def maybe_decode_audio(
    *,
    speech_tokenizer,
    global_values: Sequence[int],
    semantic_values: Sequence[int],
    output_path: Path,
    device: torch.device,
) -> tuple[str | None, str | None]:
    if not semantic_values:
        return None, "no_semantic_tokens"
    tokens = torch.tensor([*global_values, *semantic_values], dtype=torch.long, device=device)
    try:
        audio = speech_tokenizer.decode(tokens)
        speech_tokenizer.save_audio(audio, output_path, sample_rate=16000)
    except Exception as exc:  # pragma: no cover - depends on local codec assets/GPU
        return None, f"decode_error:{type(exc).__name__}:{exc}"
    return str(output_path), None


def write_jsonl_row(path: Path, row: Mapping[str, object]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
        handle.write("\n")


def audio_duration_seconds(path: str | None) -> float | None:
    if path is None:
        return None
    try:
        import soundfile as sf

        info = sf.info(path)
        return float(info.frames / info.samplerate)
    except Exception:
        return None


def ensure_safe_output_dir(output_dir: Path, *, overwrite: bool) -> None:
    if not output_dir.exists():
        return
    if overwrite:
        return
    protected_names = ("results.jsonl", "summary.json", "run_config.json", "wav", "source_wav", "reference_wav")
    if any((output_dir / name).exists() for name in protected_names):
        raise FileExistsError(
            f"Refusing to reuse non-empty evaluation output directory without --overwrite: {output_dir}"
        )


def generate_audio(args: argparse.Namespace) -> dict[str, int]:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers import set_seed
    from uniss import UniSSTokenizer

    set_seed(args.seed)
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    ensure_safe_output_dir(output_dir, overwrite=args.overwrite)
    wav_dir = output_dir / "wav"
    source_dir = output_dir / "source_wav"
    ref_dir = output_dir / "reference_wav"
    output_dir.mkdir(parents=True, exist_ok=True)
    wav_dir.mkdir(parents=True, exist_ok=True)
    if args.save_source_audio:
        source_dir.mkdir(parents=True, exist_ok=True)
    if args.save_reference_audio:
        ref_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = output_dir / "results.jsonl"
    if metadata_path.exists() and args.overwrite:
        metadata_path.unlink()

    run_config = {
        "model": str(Path(args.model).resolve()),
        "speech_tokenizer": str(Path(args.speech_tokenizer).resolve()),
        "input": list(args.input or []),
        "manifest": str(Path(args.manifest).resolve()) if args.manifest else None,
        "modes": list(args.mode),
        "limit_records": args.limit_records,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "repetition_penalty": args.repetition_penalty,
        "seed": args.seed,
        "dtype": args.dtype,
        "device": args.device,
        "save_source_audio": args.save_source_audio,
        "save_reference_audio": args.save_reference_audio,
        "skip_audio_decode": args.skip_audio_decode,
    }
    write_json(output_dir / "run_config.json", run_config)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        local_files_only=args.local_files_only,
        trust_remote_code=False,
    )
    text_encoder = load_hf_text_encoder(tokenizer)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        local_files_only=args.local_files_only,
        trust_remote_code=False,
        torch_dtype=torch.bfloat16 if args.dtype == "bfloat16" else torch.float16 if args.dtype == "float16" else "auto",
    )
    model.to(device)
    model.eval()

    speech_tokenizer = None
    if not args.skip_audio_decode:
        speech_tokenizer = UniSSTokenizer.from_pretrained(args.speech_tokenizer, device=device)

    if args.manifest:
        records = iter_manifest_records(Path(args.manifest), limit_records=args.limit_records)
    else:
        paths = expand_input_paths(args.input)
        records = iter_unist_records(paths, limit_records=args.limit_records)
    counts = {
        "total": 0,
        "generated_audio": 0,
        "source_audio": 0,
        "reference_audio": 0,
        "failed": 0,
        "no_semantic_tokens": 0,
    }

    for record_index, record in enumerate(records):
        for mode in args.mode:
            sample = build_eval_sample(record, mode=mode, text_encoder=text_encoder)
            prompt_ids = torch.tensor([sample.prompt_ids], dtype=torch.long, device=device)
            generate_kwargs = {
                "max_new_tokens": args.max_new_tokens,
                "do_sample": args.temperature > 0,
                "repetition_penalty": args.repetition_penalty,
                "pad_token_id": c.TOKEN_PAD,
                "eos_token_id": c.TOKEN_EOS,
            }
            if args.temperature > 0:
                generate_kwargs["temperature"] = args.temperature
                generate_kwargs["top_p"] = args.top_p
                generate_kwargs["top_k"] = 0 if args.top_k < 0 else args.top_k
            started = time.perf_counter()
            with torch.inference_mode():
                generated = model.generate(prompt_ids, **generate_kwargs)
            generation_seconds = time.perf_counter() - started
            generated_tail = truncate_at_eos(generated[0, prompt_ids.shape[1] :].tolist())
            parsed = parse_with_tokenizer(generated_tail, mode=mode, tokenizer=tokenizer)
            semantic_values = parsed["semantic_values"]
            generated_text = tokenizer.decode(generated_tail, skip_special_tokens=False)
            name = safe_sample_name(record_index, record.get("id"), mode)

            audio_path = None
            error = None
            if speech_tokenizer is not None:
                audio_path, error = maybe_decode_audio(
                    speech_tokenizer=speech_tokenizer,
                    global_values=record["bicodec_global"],  # type: ignore[arg-type]
                    semantic_values=semantic_values,
                    output_path=wav_dir / f"{name}.wav",
                    device=device,
                )

            reference_audio_path = None
            source_audio_path = None
            source_error = None
            reference_error = None
            if args.save_source_audio and speech_tokenizer is not None:
                source_audio_path, source_error = maybe_decode_audio(
                    speech_tokenizer=speech_tokenizer,
                    global_values=record["bicodec_global"],  # type: ignore[arg-type]
                    semantic_values=record["source_bicodec"],  # type: ignore[arg-type]
                    output_path=source_dir / f"{name}.wav",
                    device=device,
                )
                if source_error is None:
                    counts["source_audio"] += 1

            if args.save_reference_audio and speech_tokenizer is not None:
                reference_audio_path, reference_error = maybe_decode_audio(
                    speech_tokenizer=speech_tokenizer,
                    global_values=record["bicodec_global"],  # type: ignore[arg-type]
                    semantic_values=reference_bicodec_values(record, mode),
                    output_path=ref_dir / f"{name}.wav",
                    device=device,
                )
                if reference_error is None:
                    counts["reference_audio"] += 1

            row = {
                "index": record_index,
                "id": record.get("id"),
                "mode": mode,
                "src_lang": record.get("src_lang"),
                "tgt_lang": record.get("tgt_lang"),
                "dataset_name": record.get("dataset_name"),
                "transcription_ref": record.get("transcription"),
                "translation_ref": record.get("translation"),
                "generated_text_raw": generated_text,
                "generated_text_clean": clean_generated_text(generated_text),
                "generated_transcription": parsed["generated_transcription"],
                "generated_translation": parsed["generated_translation"],
                "generated_token_ids": generated_tail if args.save_generated_token_ids else None,
                "semantic_token_count": len(semantic_values),
                "has_semantic_start": parsed["has_semantic_start"],
                "has_semantic_end": parsed["has_semantic_end"],
                "has_eos": parsed["has_eos"],
                "generation_seconds": generation_seconds,
                "audio_path": audio_path,
                "audio_duration_seconds": audio_duration_seconds(audio_path),
                "source_audio_path": source_audio_path,
                "source_audio_duration_seconds": audio_duration_seconds(source_audio_path),
                "source_audio_error": source_error,
                "reference_audio_path": reference_audio_path,
                "reference_audio_duration_seconds": audio_duration_seconds(reference_audio_path),
                "reference_audio_error": reference_error,
                "error": error,
                "checkpoint": str(args.model),
                "seed": args.seed,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "top_k": args.top_k,
                "repetition_penalty": args.repetition_penalty,
            }
            write_jsonl_row(metadata_path, row)
            counts["total"] += 1
            if audio_path:
                counts["generated_audio"] += 1
            if error:
                counts["failed"] += 1
            if not semantic_values:
                counts["no_semantic_tokens"] += 1

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(counts, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return counts


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", nargs="+", help="UniST parquet files or glob patterns")
    source.add_argument("--manifest", help="Fixed manifest containing parquet_path, row_index, and id")
    parser.add_argument("--model", required=True, help="HF checkpoint path to evaluate")
    parser.add_argument("--speech-tokenizer", default="pretrained_models/UniSS")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--mode", nargs="+", choices=EVAL_MODES, default=["quality", "performance"])
    parser.add_argument("--limit-records", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=1500)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=-1, help="-1 disables top-k filtering, matching the paper")
    parser.add_argument("--repetition-penalty", type=float, default=1.1)
    parser.add_argument("--seed", type=int, default=20260726)
    parser.add_argument("--dtype", choices=["auto", "bfloat16", "float16"], default="bfloat16")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--skip-audio-decode", action="store_true")
    parser.add_argument("--save-source-audio", action="store_true")
    parser.add_argument("--save-reference-audio", action="store_true")
    parser.add_argument(
        "--save-generated-token-ids",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    counts = generate_audio(args)
    print(json.dumps(counts, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

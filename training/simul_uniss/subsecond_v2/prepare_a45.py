"""Prepare formal Stage-A A4/A5 timestamp and reconstructed-teacher data.

Each worker owns one input shard and one GPU.  It reuses the v1 reconstructed
source audio, decodes the missing target audio, runs batched Qwen3 forced
alignment for the known source/target text, and re-encodes source audio with
the frozen WhisperVQ teacher.  The released UniST GLM and reconstructed-audio
teacher GLM are both retained and audited; they are never silently conflated.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
import time
from array import array
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import soundfile as sf
import torch

from training.simul_uniss.jsonl_index import load_index, write_index
from training.simul_uniss.subsecond_v1.stage_a import DecoderOnlyBiCodec, _write_flac
from training.simul_uniss.subsecond_v1.validate_stage_b import _edit_distance
from training.simul_uniss.subsecond_v2.formal_supervision import (
    alignment_coverage,
    normalize_language,
    normalize_words,
)
from uniss.speech_tokenizer.glm4.glm4_tokenizer import Glm4Tokenizer


SCHEMA = "simul_uniss_subsecond_stage_a_a45_part_v2"


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _glm_end_times(duration_ms: int, count: int) -> list[int]:
    if count <= 0:
        return []
    return [max(1, min(duration_ms, math.ceil(duration_ms * (index + 1) / count))) for index in range(count)]


def _to_words(result: Iterable[Any]) -> list[dict[str, object]]:
    return [
        {
            "text": str(getattr(value, "text", "")),
            "start_ms": int(round(float(getattr(value, "start_time", 0.0)) * 1000)),
            "end_ms": int(round(float(getattr(value, "end_time", 0.0)) * 1000)),
            "confidence": None,
        }
        for value in result
    ]


def build_a45_record(
    item: Mapping[str, object],
    *,
    target_audio: str,
    target_duration_ms: int,
    source_alignment: Iterable[Mapping[str, object]],
    target_alignment: Iterable[Mapping[str, object]],
    teacher_source_glm: Sequence[int],
    minimum_alignment_coverage: float,
) -> dict[str, object]:
    """Create one audited A4/A5 record from backend outputs."""

    source_duration_ms = int(item["source_duration_ms"])
    source_words = normalize_words(source_alignment, duration_ms=source_duration_ms)
    target_words = normalize_words(target_alignment, duration_ms=target_duration_ms)
    source_language = normalize_language(str(item["src_lang"]))
    target_language = normalize_language(str(item["tgt_lang"]))
    source_coverage = alignment_coverage(source_words, str(item["transcription"]), source_language)
    target_coverage = alignment_coverage(target_words, str(item["translation"]), target_language)
    released = [int(value) for value in item["source_glm"]]  # type: ignore[index]
    teacher = [int(value) for value in teacher_source_glm]
    if not teacher:
        raise ValueError("WhisperVQ reconstructed-audio teacher returned no GLM tokens")
    distance = _edit_distance(teacher, released)
    compatibility = max(0.0, 1.0 - distance / max(1, len(released)))
    quality_flags: list[str] = []
    if source_coverage < minimum_alignment_coverage:
        quality_flags.append("low_source_forced_alignment_coverage")
    if target_coverage < minimum_alignment_coverage:
        quality_flags.append("low_target_forced_alignment_coverage")
    if compatibility < 0.90:
        quality_flags.append("released_vs_reconstructed_teacher_domain_mismatch")
    value = dict(item)
    value.update(
        {
            "schema_version": SCHEMA,
            "stage_a_scope": "formal_a4_a5_and_reconstructed_teacher_v2",
            "source_alignment_kind": "qwen3_forced_aligner_word_time_v1",
            "target_alignment_kind": "qwen3_forced_aligner_word_time_v1",
            "source_words": source_words,
            "target_words": target_words,
            "source_alignment_coverage": source_coverage,
            "target_alignment_coverage": target_coverage,
            "target_audio": target_audio,
            "target_duration_ms": target_duration_ms,
            "teacher_source_glm": teacher,
            "teacher_source_glm_end_ms": _glm_end_times(source_duration_ms, len(teacher)),
            "released_source_glm": released,
            "released_vs_reconstructed_teacher_edit_distance": distance,
            "released_vs_reconstructed_teacher_agreement": compatibility,
            "stage_b_supervision_field": "teacher_source_glm",
            "formal_a45_pass": source_coverage >= minimum_alignment_coverage
            and target_coverage >= minimum_alignment_coverage,
            "quality_flags": sorted(set([*item.get("quality_flags", []), *quality_flags])),
        }
    )
    return value


class QwenForcedAlignerBackend:
    def __init__(self, model_path: str, device: str, batch_size: int) -> None:
        from qwen_asr import Qwen3ForcedAligner

        self.model = Qwen3ForcedAligner.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
            device_map=device,
        )
        self.batch_size = batch_size

    @torch.inference_mode()
    def align(
        self, audio: Sequence[str], text: Sequence[str], language: Sequence[str]
    ) -> list[list[dict[str, object]]]:
        output: list[list[dict[str, object]]] = []
        for start in range(0, len(audio), self.batch_size):
            results = self.model.align(
                audio=list(audio[start : start + self.batch_size]),
                text=list(text[start : start + self.batch_size]),
                language=list(language[start : start + self.batch_size]),
            )
            output.extend([_to_words(value) for value in results])
        return output


def _read_items(path: Path, start: int, limit: int | None) -> list[dict[str, object]]:
    offsets = load_index(path)
    if offsets is None:
        raise ValueError(f"missing JSONL offset index for {path}")
    stop = len(offsets) if limit is None else min(len(offsets), start + limit)
    values: list[dict[str, object]] = []
    with path.open("rb") as handle:
        for index in range(start, stop):
            handle.seek(offsets[index])
            value = json.loads(handle.readline())
            value["formal_input_index"] = index
            values.append(value)
    return values


def prepare(args: argparse.Namespace) -> dict[str, object]:
    input_manifest = Path(args.input_manifest).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "a45_manifest.jsonl"
    marker_path = output_dir / "STAGE_A_A45_COMPLETE.json"
    if marker_path.is_file() and output.is_file():
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        print(json.dumps({"status": "already_complete", **marker}, sort_keys=True))
        return marker
    items = _read_items(input_manifest, args.start_index, args.limit_records)
    if not items:
        raise ValueError("selected A4/A5 input slice is empty")

    device = torch.device(args.device)
    decoder = DecoderOnlyBiCodec(Path(args.bicodec_checkpoint), device)
    aligner = QwenForcedAlignerBackend(args.forced_aligner_model, args.device, args.alignment_batch_size)
    teacher = Glm4Tokenizer(args.whispervq_model, device=args.device)
    target_dir = output_dir / "target_audio"
    target_dir.mkdir(parents=True, exist_ok=True)
    counts: Counter[str] = Counter()
    offsets = array("Q")
    temporary = output_dir / f".a45_manifest.jsonl.tmp.{os.getpid()}"
    byte_offset = 0
    started = time.time()
    try:
        with temporary.open("wb") as handle:
            for batch_start in range(0, len(items), args.worker_batch_size):
                batch = items[batch_start : batch_start + args.worker_batch_size]
                target_paths: list[str] = []
                target_durations: list[int] = []
                for item in batch:
                    target_path = target_dir / f"{int(item['formal_input_index']):07d}.flac"
                    if target_path.is_file():
                        info = sf.info(target_path)
                        duration = round(1000 * info.frames / info.samplerate)
                    else:
                        waveform = decoder.decode(
                            [int(value) for value in item["bicodec_global"]],  # type: ignore[index]
                            [int(value) for value in item["target_bicodec"]],  # type: ignore[index]
                        )
                        duration = round(1000 * len(waveform) / args.sample_rate)
                        _write_flac(target_path, waveform, args.sample_rate)
                    target_paths.append(str(target_path))
                    target_durations.append(duration)

                source_paths = [str(value["source_audio"]) for value in batch]
                combined_audio = [*source_paths, *target_paths]
                combined_text = [
                    *[str(value["transcription"]) for value in batch],
                    *[str(value["translation"]) for value in batch],
                ]
                combined_language = [
                    *["Chinese" if normalize_language(str(value["src_lang"])) == "zh" else "English" for value in batch],
                    *["Chinese" if normalize_language(str(value["tgt_lang"])) == "zh" else "English" for value in batch],
                ]
                aligned = aligner.align(combined_audio, combined_text, combined_language)
                source_alignments = aligned[: len(batch)]
                target_alignments = aligned[len(batch) :]
                teacher_tokens = teacher.bacth_tokenize(source_paths)
                for item, target_path, target_duration, source_words, target_words, glm in zip(
                    batch,
                    target_paths,
                    target_durations,
                    source_alignments,
                    target_alignments,
                    teacher_tokens,
                ):
                    try:
                        value = build_a45_record(
                            item,
                            target_audio=target_path,
                            target_duration_ms=target_duration,
                            source_alignment=source_words,
                            target_alignment=target_words,
                            teacher_source_glm=glm,
                            minimum_alignment_coverage=args.minimum_alignment_coverage,
                        )
                        counts["records"] += 1
                        counts["formal_pass"] += int(bool(value["formal_a45_pass"]))
                        counts[f"direction:{value['src_lang']}->{value['tgt_lang']}"] += 1
                        counts["released_teacher_agreement_sum_ppm"] += round(
                            float(value["released_vs_reconstructed_teacher_agreement"]) * 1_000_000
                        )
                    except Exception as error:
                        counts["rejected"] += 1
                        value = {
                            "schema_version": SCHEMA,
                            "id": item.get("id"),
                            "formal_input_index": item["formal_input_index"],
                            "formal_a45_pass": False,
                            "formal_a45_error": f"{type(error).__name__}: {error}",
                        }
                    encoded = (json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
                    offsets.append(byte_offset)
                    handle.write(encoded)
                    byte_offset += len(encoded)
                if args.progress_interval and counts["records"] % args.progress_interval < len(batch):
                    elapsed = max(time.time() - started, 1e-6)
                    print(
                        json.dumps(
                            {
                                "processed": counts["records"] + counts["rejected"],
                                "formal_pass": counts["formal_pass"],
                                "records_per_second": (counts["records"] + counts["rejected"]) / elapsed,
                            }
                        ),
                        flush=True,
                    )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    index = write_index(output, offsets)
    records = max(1, counts["records"])
    marker = {
        "schema_version": SCHEMA,
        "status": "complete",
        "scope": "formal_stage_a_a4_a5_and_reconstructed_teacher_v2",
        "input_manifest": str(input_manifest),
        "output_manifest": str(output),
        "index": index,
        "start_index": args.start_index,
        "limit_records": args.limit_records,
        "counts": dict(counts),
        "formal_pass_rate": counts["formal_pass"] / records,
        "mean_released_vs_reconstructed_teacher_agreement": counts[
            "released_teacher_agreement_sum_ppm"
        ]
        / records
        / 1_000_000,
        "forced_aligner_model": str(Path(args.forced_aligner_model).resolve()),
        "whispervq_model": str(Path(args.whispervq_model).resolve()),
        "bicodec_checkpoint": str(Path(args.bicodec_checkpoint).resolve()),
        "elapsed_seconds": time.time() - started,
    }
    _atomic_json(marker_path, marker)
    print(json.dumps(marker, sort_keys=True))
    return marker


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--forced-aligner-model", required=True)
    parser.add_argument("--whispervq-model", required=True)
    parser.add_argument("--bicodec-checkpoint", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit-records", type=int)
    parser.add_argument("--worker-batch-size", type=int, default=64)
    parser.add_argument("--alignment-batch-size", type=int, default=128)
    parser.add_argument("--minimum-alignment-coverage", type=float, default=0.85)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--progress-interval", type=int, default=1000)
    return parser.parse_args()


def main() -> None:
    prepare(parse_args())


if __name__ == "__main__":
    main()


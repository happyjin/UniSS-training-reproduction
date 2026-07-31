"""Prepare formal Stage-A A6--A8 bilingual support and Micro-WRITE data."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from array import array
from collections import Counter
from pathlib import Path

from training.simul_uniss.jsonl_index import load_index, write_index
from training.simul_uniss.subsecond_v2.formal_supervision import (
    FORMAL_ALIGNMENT_KIND,
    build_micro_write_supervision,
    build_support_alignment,
    merge_alignment_segments,
    normalize_language,
)
from training.simul_uniss.subsecond_v2.neural_word_aligner import BatchedNeuralWordAligner


SCHEMA = "simul_uniss_subsecond_stage_a_a68_part_v2"


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


def _read(path: Path) -> list[dict[str, object]]:
    offsets = load_index(path)
    if offsets is None:
        raise ValueError(f"missing JSONL index for {path}")
    values: list[dict[str, object]] = []
    with path.open("rb") as handle:
        for offset in offsets:
            handle.seek(offset)
            value = json.loads(handle.readline())
            if bool(value.get("formal_a45_pass")):
                values.append(value)
    return values


def _lexical_words(
    words: list[dict[str, object]], text: str, language: str
) -> list[dict[str, object]]:
    if normalize_language(language) != "zh":
        return words
    import jieba

    return merge_alignment_segments(words, list(jieba.cut(text, cut_all=False)))


def prepare(args: argparse.Namespace) -> dict[str, object]:
    input_manifest = Path(args.input_manifest).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "formal_manifest.jsonl"
    marker_path = output_dir / "STAGE_A_A68_COMPLETE.json"
    if marker_path.is_file() and output.is_file():
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        print(json.dumps({"status": "already_complete", **marker}, sort_keys=True))
        return marker
    values = _read(input_manifest)
    if not values:
        raise ValueError("no A4/A5-passing records were found")
    aligner = BatchedNeuralWordAligner(
        args.word_aligner_model,
        device=args.device,
        batch_size=args.batch_size,
        mutual_threshold=args.mutual_threshold,
        union_threshold=args.union_threshold,
    )
    temporary = output_dir / f".formal_manifest.jsonl.tmp.{os.getpid()}"
    offsets = array("Q")
    byte_offset = 0
    counts: Counter[str] = Counter()
    started = time.time()
    try:
        with temporary.open("wb") as handle:
            for start in range(0, len(values), args.batch_size):
                batch = values[start : start + args.batch_size]
                lexical_pairs = [
                    (
                        _lexical_words(
                            value["source_words"],  # type: ignore[arg-type]
                            str(value["transcription"]),
                            str(value["src_lang"]),
                        ),
                        _lexical_words(
                            value["target_words"],  # type: ignore[arg-type]
                            str(value["translation"]),
                            str(value["tgt_lang"]),
                        ),
                    )
                    for value in batch
                ]
                source_tokens = [[str(word["text"]) for word in pair[0]] for pair in lexical_pairs]
                target_tokens = [[str(word["text"]) for word in pair[1]] for pair in lexical_pairs]
                batch_links = aligner.align_batch(source_tokens, target_tokens)
                for value, (source_words, target_words), links in zip(batch, lexical_pairs, batch_links):
                    try:
                        support = build_support_alignment(
                            source_words,
                            target_words,
                            links,
                            minimum_confidence=args.minimum_link_confidence,
                        )
                        micro = build_micro_write_supervision(
                            target_words,
                            support,
                            language=str(value["tgt_lang"]),
                            target_duration_ms=int(value["target_duration_ms"]),
                            target_semantic_count=len(value["target_bicodec"]),  # type: ignore[arg-type]
                            minimum_semantic=args.minimum_semantic,
                            maximum_semantic=args.maximum_semantic,
                            hard_maximum_semantic=args.hard_maximum_semantic,
                            tick_ms=args.tick_ms,
                            safety_margin_ms=args.safety_margin_ms,
                        )
                        linked_targets = {int(link["target_index"]) for link in links}
                        link_coverage = len(linked_targets) / max(1, len(target_words))
                        uncertain_events = sum(bool(event["uncertain_alignment"]) for event in micro)
                        oversize_events = sum(bool(event["oversize_word"]) for event in micro)
                        formal_pass = (
                            link_coverage >= args.minimum_target_link_coverage
                            and uncertain_events == 0
                            and oversize_events == 0
                        )
                        record = dict(value)
                        record.update(
                            {
                                "schema_version": SCHEMA,
                                "stage_a_scope": "formal_a4_a8_v2",
                                "alignment_kind": FORMAL_ALIGNMENT_KIND,
                                "source_words": source_words,
                                "target_words": target_words,
                                "bilingual_links": links,
                                "target_support": support,
                                "micro_write_events": micro,
                                "target_link_coverage": link_coverage,
                                "uncertain_micro_write_events": uncertain_events,
                                "oversize_micro_write_events": oversize_events,
                                "formal_a68_pass": formal_pass,
                            }
                        )
                        counts["records"] += 1
                        counts["formal_pass"] += int(formal_pass)
                        counts["micro_write_events"] += len(micro)
                        counts["uncertain_micro_write_events"] += uncertain_events
                        counts["oversize_micro_write_events"] += oversize_events
                        counts["target_links"] += len(links)
                    except Exception as error:
                        counts["rejected"] += 1
                        record = {
                            "schema_version": SCHEMA,
                            "id": value.get("id"),
                            "formal_a68_pass": False,
                            "formal_a68_error": f"{type(error).__name__}: {error}",
                        }
                    encoded = (json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n").encode(
                        "utf-8"
                    )
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
    marker = {
        "schema_version": SCHEMA,
        "status": "complete",
        "scope": "formal_stage_a_a4_a8_v2",
        "input_manifest": str(input_manifest),
        "output_manifest": str(output),
        "index": index,
        "word_aligner_model": args.word_aligner_model,
        "counts": dict(counts),
        "formal_pass_rate": counts["formal_pass"] / max(1, counts["records"]),
        "elapsed_seconds": time.time() - started,
    }
    _atomic_json(marker_path, marker)
    print(json.dumps(marker, sort_keys=True))
    return marker


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--word-aligner-model", default="bert-base-multilingual-cased")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--mutual-threshold", type=float, default=0.35)
    parser.add_argument("--union-threshold", type=float, default=0.55)
    parser.add_argument("--minimum-link-confidence", type=float, default=0.35)
    parser.add_argument("--minimum-target-link-coverage", type=float, default=0.45)
    parser.add_argument("--minimum-semantic", type=int, default=8)
    parser.add_argument("--maximum-semantic", type=int, default=16)
    parser.add_argument("--hard-maximum-semantic", type=int, default=24)
    parser.add_argument("--tick-ms", type=int, default=160)
    parser.add_argument("--safety-margin-ms", type=int, default=80)
    parser.add_argument("--progress-interval", type=int, default=1000)
    return parser.parse_args()


def main() -> None:
    prepare(parse_args())


if __name__ == "__main__":
    main()

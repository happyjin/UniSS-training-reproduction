"""Compute symmetric AutoPCP multilingual-v2 scores for generated speech."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from itertools import islice
from pathlib import Path
from statistics import mean, pstdev
from typing import Iterable, Iterator, Mapping, Sequence

import torch

from evaluation.io_utils import iter_jsonl, write_json
from evaluation.sharding import load_keys, select_shard
from training.generate_unist_eval_audio import write_jsonl_row


def chunks(values: Iterable[Mapping[str, object]], size: int) -> Iterator[list[Mapping[str, object]]]:
    source = iter(values)
    while batch := list(islice(source, size)):
        yield batch


def aggregate_scores(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    groups: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for row in rows:
        groups[(str(row["mode"]), str(row["src_lang"]), str(row["tgt_lang"]))].append(float(row["autopcp_score"]))
    return {
        "groups": {
            f"{mode}:{src}->{tgt}": {
                "sample_count": len(values),
                "mean": mean(values),
                "std": pstdev(values),
                "min": min(values),
                "max": max(values),
            }
            for (mode, src, tgt), values in sorted(groups.items())
        },
        "scored_count": len(rows),
    }


def resolve_path(value: object, *, input_path: Path) -> str:
    path = Path(str(value))
    return str(path if path.is_absolute() else input_path.parent / path)


def run_autopcp(args: argparse.Namespace) -> dict[str, object]:
    from stopes.eval.auto_pcp.audio_comparator import Comparator, encode_audios, get_model_pred
    from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    per_sample_path = output_dir / "per_sample_autopcp.jsonl"
    existing_rows: list[dict[str, object]] = []
    completed: set[tuple[str, str]] = load_keys(args.completed_input)
    if per_sample_path.exists():
        if not args.resume:
            raise FileExistsError(f"Refusing to overwrite AutoPCP output: {per_sample_path}")
        existing_rows = list(iter_jsonl(per_sample_path))
        completed.update((str(row["id"]), str(row["mode"])) for row in existing_rows)

    pending = [
        row
        for row in select_shard(
            iter_jsonl(input_path),
            num_shards=args.num_shards,
            shard_index=args.shard_index,
        )
        if (str(row["id"]), str(row["mode"])) not in completed
        and row.get("source_audio_path")
        and row.get("audio_path")
        and not row.get("error")
    ]
    use_cuda = args.device.startswith("cuda") and torch.cuda.is_available()
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(args.encoder_model, local_files_only=True)
    encoder = Wav2Vec2Model.from_pretrained(args.encoder_model, local_files_only=True)
    comparator = Comparator.load(args.comparator_path, use_gpu=use_cuda)
    if use_cuda:
        encoder.to(args.device)
    scored = list(existing_rows)
    for batch in chunks(pending, args.chunk_size):
        source_paths = [resolve_path(row["source_audio_path"], input_path=input_path) for row in batch]
        target_paths = [resolve_path(row["audio_path"], input_path=input_path) for row in batch]
        source_embeddings = encode_audios(
            source_paths,
            fex=feature_extractor,
            model=encoder,
            batch_size=args.batch_size,
            pick_layer=args.pick_layer,
            num_process=args.num_process,
            progress=args.show_progress,
        )[:, 0]
        target_embeddings = encode_audios(
            target_paths,
            fex=feature_extractor,
            model=encoder,
            batch_size=args.batch_size,
            pick_layer=args.pick_layer,
            num_process=args.num_process,
            progress=args.show_progress,
        )[:, 0]
        forward = get_model_pred(
            comparator,
            src=source_embeddings,
            mt=target_embeddings,
            use_gpu=use_cuda,
            batch_size=args.batch_size,
        )[:, 0]
        if args.symmetrize:
            backward = get_model_pred(
                comparator,
                src=target_embeddings,
                mt=source_embeddings,
                use_gpu=use_cuda,
                batch_size=args.batch_size,
            )[:, 0]
            predictions = ((forward + backward) / 2).detach().cpu().tolist()
        else:
            predictions = forward.detach().cpu().tolist()
        for row, score in zip(batch, predictions):
            scored_row = {
                "id": row.get("id"),
                "mode": row.get("mode"),
                "src_lang": row.get("src_lang"),
                "tgt_lang": row.get("tgt_lang"),
                "source_audio_path": row.get("source_audio_path"),
                "audio_path": row.get("audio_path"),
                "autopcp_score": float(score),
                "encoder_model": args.encoder_model,
                "comparator_path": str(Path(args.comparator_path).resolve()),
                "pick_layer": args.pick_layer,
                "symmetrize": args.symmetrize,
            }
            write_jsonl_row(per_sample_path, scored_row)
            scored.append(scored_row)
        write_json(output_dir / "autopcp.json", aggregate_scores(scored))
    return aggregate_scores(scored)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--comparator-path", required=True)
    parser.add_argument("--encoder-model", default="facebook/wav2vec2-large-xlsr-53")
    parser.add_argument("--pick-layer", type=int, default=9)
    parser.add_argument("--symmetrize", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--chunk-size", type=int, default=1024)
    # stopes commit a4e75e8 resets torch.multiprocessing's global start method
    # without force after each parallel read, which raises RuntimeError in this
    # mixed ASR/UTMOS/AutoPCP process stack. A single reader is deterministic
    # and avoids mutating global multiprocessing state; GPU encoding remains
    # batched independently through --batch-size.
    parser.add_argument("--num-process", type=int, default=1)
    parser.add_argument("--show-progress", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--completed-input", action="append", default=[])
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    print(json.dumps(run_autopcp(parse_args(argv)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

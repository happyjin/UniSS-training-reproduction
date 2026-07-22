"""Reconstruct source/target audio from UniST BiCodec tokens for bootstrap alignment."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pyarrow.parquet as pq
import soundfile as sf
import torch


def safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return cleaned or "sample"


def decode_tokens(tokenizer, global_tokens: list[int], semantic_tokens: list[int]):
    global_tensor = torch.tensor([global_tokens], dtype=torch.long, device=tokenizer.device)
    semantic_tensor = torch.tensor([semantic_tokens], dtype=torch.long, device=tokenizer.device)
    return tokenizer.detokenize(global_tensor, semantic_tensor)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bicodec-model-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit-records", type=int, default=1)
    parser.add_argument("--side", choices=["source", "target", "both"], default="both")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from uniss.speech_tokenizer.bicodec.bicodec_tokenizer import BiCodecTokenizer

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "audio_manifest.jsonl"
    tokenizer = BiCodecTokenizer(Path(args.bicodec_model_dir), device=torch.device(args.device))
    count = 0
    with manifest_path.open("w", encoding="utf-8") as manifest:
        for input_value in args.input:
            parquet = pq.ParquetFile(input_value)
            columns = ["id", "source_bicodec", "target_bicodec", "bicodec_global", "src_lang", "tgt_lang"]
            for batch in parquet.iter_batches(columns=columns, batch_size=16):
                for row in batch.to_pylist():
                    item: dict[str, object] = {
                        "id": str(row["id"]),
                        "src_lang": str(row["src_lang"]),
                        "tgt_lang": str(row["tgt_lang"]),
                        "source_parquet": str(Path(input_value).resolve()),
                    }
                    name = f"{count:06d}_{safe_name(str(row['id']))}"
                    if args.side in {"source", "both"}:
                        path = output_dir / f"{name}_source.flac"
                        if args.overwrite or not path.exists():
                            waveform = decode_tokens(tokenizer, row["bicodec_global"], row["source_bicodec"])
                            sf.write(path, waveform, 16000)
                        item["source_audio"] = str(path.resolve())
                    if args.side in {"target", "both"}:
                        path = output_dir / f"{name}_target.flac"
                        if args.overwrite or not path.exists():
                            waveform = decode_tokens(tokenizer, row["bicodec_global"], row["target_bicodec"])
                            sf.write(path, waveform, 16000)
                        item["target_audio"] = str(path.resolve())
                    manifest.write(json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n")
                    count += 1
                    print(json.dumps({"reconstructed": count, "id": row["id"]}), flush=True)
                    if args.limit_records is not None and count >= args.limit_records:
                        print(json.dumps({"manifest": str(manifest_path), "records": count}, sort_keys=True))
                        return
    print(json.dumps({"manifest": str(manifest_path), "records": count}, sort_keys=True))


if __name__ == "__main__":
    main()

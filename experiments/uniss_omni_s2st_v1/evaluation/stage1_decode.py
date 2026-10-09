"""Decode Stage 1's generated codes to audio, in the project's own layout.

Writes ``results.jsonl`` and ``wav/`` exactly as the CVSS-T evaluation does,
so ``run_objective_metrics.sh`` can score this run against every published
figure without a second, slightly different audio path.

Both the generated codes and the gold codes are decoded. The gold decode is
the ceiling: it is what this vocoder can do with perfect codes for the same
utterance and the same 32 speaker tokens, so a Stage 1 result is only
readable against it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from training.generate_unist_eval_audio import maybe_decode_audio


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--generated", nargs="+", required=True)
    ap.add_argument("--speech-tokenizer", default="pretrained_models/UniSS")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--decode-gold", action="store_true")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    from uniss import UniSSTokenizer

    rows = []
    for path in args.generated:
        rows.extend(
            json.loads(line)
            for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    rows.sort(key=lambda row: row["index"])
    rows = rows[args.shard :: args.num_shards]

    out_dir = Path(args.output)
    wav_dir = out_dir / "wav"
    wav_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    tokenizer = UniSSTokenizer.from_pretrained(args.speech_tokenizer, device=device)

    results_path = out_dir / f"results_{args.shard}.jsonl"
    failures = 0
    with results_path.open("w", encoding="utf-8") as handle:
        for position, row in enumerate(rows):
            name = f"{row['index']:06d}_{row['id'].replace('/', '_')}_{row['tgt_lang']}"
            audio_path, error = maybe_decode_audio(
                speech_tokenizer=tokenizer,
                global_values=row["global_values"],
                semantic_values=row["semantic_values"],
                output_path=wav_dir / f"{name}.wav",
                device=device,
            )
            record = {
                **{k: v for k, v in row.items() if k not in ("semantic_values", "global_values")},
                "audio_path": audio_path,
                "error": error,
                "generated_text_raw": "",
                "generated_text_clean": "",
            }
            if error:
                failures += 1
            if args.decode_gold:
                gold_path, gold_error = maybe_decode_audio(
                    speech_tokenizer=tokenizer,
                    global_values=row["global_values"],
                    semantic_values=row.get("gold_semantic_values") or [],
                    output_path=wav_dir / f"{name}_gold.wav",
                    device=device,
                )
                record["gold_audio_path"] = gold_path
                record["gold_error"] = gold_error
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            if (position + 1) % 25 == 0:
                print(f"  {position + 1}/{len(rows)}", flush=True)

    print(f"{len(rows)} decoded, {failures} failed -> {results_path}")


if __name__ == "__main__":
    main()

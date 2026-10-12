"""Phase B: speak someone else's translations with our Talker.

Phase A measured what a 30B Thinker translates. This measures what that
is worth once spoken, which is the number the project is actually judged
on -- and it needs no training, because the cascade path feeds the Talker
a TTS prompt built from text. The Talker does not care who wrote the text.

That decoupling is what makes the backbone question cheap to answer. It
was measured, not assumed: the same stage 2 Talker weights score 19.64
ASR-BLEU reading S2ST reply states and 24.41 reading a TTS prompt.

Two processes rather than one, because Qwen3-Omni needs transformers 5.x
and our Talker runs on 4.53. Phase A writes text in the new environment;
this reads it in the old one. Nothing here imports Qwen3.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from experiments.uniss_omni_s2st_v1.evaluation.stage1_generate import generate_codes
from experiments.uniss_omni_s2st_v1.modeling.talker_surgery import (
    read_codec_layout,
    retarget_talker_codebook,
    valid_output_mask,
)
from experiments.uniss_omni_s2st_v1.runtime.omni_processor import (
    load_audio_text_processor,
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="pretrained_models/Qwen2.5-Omni-3B")
    ap.add_argument("--checkpoint", required=True, help="a Talker export")
    ap.add_argument("--translations", required=True, help="Phase A merged.json")
    ap.add_argument("--dev-parquet", required=True)
    ap.add_argument("--speech-tokenizer", default="pretrained_models/UniSS")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--max-steps", type=int, default=700)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--repetition-penalty", type=float, default=1.3)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    import pyarrow.parquet as pq
    from transformers import Qwen2_5OmniForConditionalGeneration
    from uniss import UniSSTokenizer

    hypothesis = {
        r["id"]: r["hypothesis"]
        for r in json.loads(Path(args.translations).read_text(encoding="utf-8"))
    }
    table = pq.read_table(
        args.dev_parquet,
        columns=[
            "id", "translation", "transcription", "src_lang", "tgt_lang",
            "bicodec_global", "target_bicodec", "source_audio_path",
            "reference_audio_path", "synthetic_source", "synthetic_reference",
            "source_audio_duration_seconds", "reference_audio_duration_seconds",
        ],
    )
    rows = [r for r in table.to_pylist() if r["id"] in hypothesis]
    missing = len(hypothesis) - len(rows)
    if missing:
        raise SystemExit(f"{missing} translations have no parquet row; the join is wrong")
    rows.sort(key=lambda r: (r["id"], r["tgt_lang"]))
    if args.limit:
        rows = rows[: args.limit]
    rows = rows[args.shard :: args.num_shards]
    print(f"{len(rows)} utterances", flush=True)

    processor = load_audio_text_processor(args.model)
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        args.model, torch_dtype=torch.bfloat16
    ).to(args.device)
    if hasattr(model, "token2wav"):
        del model.token2wav
    layout = read_codec_layout(model.talker.config)
    retarget_talker_codebook(model.talker, head_scale=0.1)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.talker.load_state_dict(state["talker"], strict=False)
    global_embed = None
    if state.get("global_embed") is not None:
        weight = state["global_embed"]
        global_embed = torch.nn.Embedding(weight.shape[0], weight.shape[1])
        with torch.no_grad():
            global_embed.weight.copy_(weight)
        global_embed = global_embed.to(device=model.device, dtype=weight.dtype)
    model.eval()
    output_mask = valid_output_mask(layout, device=model.device)
    vocoder = UniSSTokenizer.from_pretrained(
        args.speech_tokenizer, device=torch.device(args.device)
    )

    out_dir = Path(args.output)
    (out_dir / "wav").mkdir(parents=True, exist_ok=True)
    records = []
    for position, row in enumerate(rows):
        text = hypothesis[row["id"]].strip()
        codes = generate_codes(
            model, processor, text, row["tgt_lang"],
            layout=layout, output_mask=output_mask, device=model.device,
            max_steps=args.max_steps, temperature=args.temperature,
            repetition_penalty=args.repetition_penalty,
            global_embed=global_embed,
            bicodec_global=(
                None if global_embed is None
                else torch.as_tensor(row["bicodec_global"], dtype=torch.long)
            ),
        ) if text else []
        index = args.shard + position * args.num_shards
        name = f"{index:05d}_{row['src_lang']}2{row['tgt_lang']}"
        audio_path = None
        if codes:
            tokens = torch.tensor(
                [*row["bicodec_global"], *codes], dtype=torch.long, device=model.device
            )
            vocoder.save_audio(
                vocoder.decode(tokens), out_dir / "wav" / f"{name}.wav",
                sample_rate=16000,
            )
            audio_path = str(out_dir / "wav" / f"{name}.wav")
        records.append({
            "index": index, "name": name, "id": row["id"], "mode": "tts",
            "src_lang": row["src_lang"], "tgt_lang": row["tgt_lang"],
            "dataset_name": "CVSS-T",
            "transcription_ref": row["transcription"],
            "translation_ref": row["translation"],
            "generated_text_raw": text, "generated_text_clean": text,
            "source_audio_path": row["source_audio_path"],
            "reference_audio_path": row["reference_audio_path"],
            "source_audio_duration_seconds": row["source_audio_duration_seconds"],
            "reference_audio_duration_seconds": row["reference_audio_duration_seconds"],
            "synthetic_source": bool(row["synthetic_source"]),
            "synthetic_reference": bool(row["synthetic_reference"]),
            "reference_text": row["translation"], "translated_text": text,
            "codes": len(codes), "gold_codes": len(row["target_bicodec"]),
            "audio_path": audio_path, "error": None,
        })
        if position % 50 == 0:
            print(f"  {position}/{len(rows)}  {text[:44]}", flush=True)

    with (out_dir / f"spoken_{args.shard}.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"-> {out_dir}")


if __name__ == "__main__":
    main()

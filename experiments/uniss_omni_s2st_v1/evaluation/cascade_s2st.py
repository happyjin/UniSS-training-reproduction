"""Speech in, translated speech out, by cascading what exists today.

Stage 1 trains a TTS task: the target text is given and the Talker speaks
it. That measures pronunciation, not translation. This runs the whole
path instead --

    source audio -> Thinker (S2TT) -> translated text
                 -> Talker (+ the source speaker's 32 global tokens)
                 -> BiCodec semantic codes -> vocoder -> speech

-- so the output is actually translated audio, and ASR-BLEU on it is the
end-to-end number rather than a TTS one.

The speaker tokens are the *source* utterance's, which is what Stage 1
was trained on (``bicodec_global`` is zh_global for cmn->eng and en_global
for eng->cmn). The translated speech therefore carries the original
speaker's voice, which is the property this project's AutoPCP result
rests on.

This is a cascade, not the joint model Stage 2 builds. Nothing here is
streaming.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import soundfile as sf
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

INSTRUCTION = {
    "eng": "Listen to the Chinese speech and translate it into English. "
           "Reply with the English translation only.",
    "cmn": "Listen to the English speech and translate it into Chinese. "
           "Reply with the Chinese translation only.",
}


@torch.no_grad()
def translate(model, processor, audio_path: str, target_lang: str, max_new_tokens: int):
    """Thinker only: source speech to target text."""
    waveform, _ = sf.read(audio_path, dtype="float32")
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": audio_path},
                {"type": "text", "text": INSTRUCTION[target_lang]},
            ],
        }
    ]
    text = processor.apply_chat_template(
        [conversation], add_generation_prompt=True, tokenize=False
    )
    inputs = processor(
        text=text, audio=[waveform], sampling_rate=16_000, return_tensors="pt"
    )
    inputs = {
        k: (v.to(model.device) if hasattr(v, "to") else v) for k, v in inputs.items()
    }
    prompt_length = inputs["input_ids"].shape[1]
    out = model.generate(
        **inputs, thinker_max_new_tokens=max_new_tokens,
        thinker_do_sample=False, return_audio=False,
    )
    return processor.tokenizer.decode(
        out[0][prompt_length:], skip_special_tokens=True
    ).strip()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="pretrained_models/Qwen2.5-Omni-3B")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--dev-parquet", nargs="+", required=True)
    ap.add_argument("--subset", required=True)
    ap.add_argument("--speech-tokenizer", default="pretrained_models/UniSS")
    ap.add_argument("--limit", type=int, default=24)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--max-steps", type=int, default=700)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--repetition-penalty", type=float, default=1.3)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    import pyarrow.parquet as pq
    from transformers import Qwen2_5OmniForConditionalGeneration
    from uniss import UniSSTokenizer

    wanted = {
        json.loads(line)["id"]
        for line in Path(args.subset).read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    rows = []
    for path in args.dev_parquet:
        table = pq.read_table(
            path,
            columns=[
                "id", "translation", "transcription", "src_lang", "tgt_lang",
                "bicodec_global", "target_bicodec", "source_audio_path",
            ],
        )
        rows.extend(r for r in table.to_pylist() if r["id"] in wanted)
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
        hypothesis = translate(
            model, processor, row["source_audio_path"], row["tgt_lang"],
            args.max_new_tokens,
        )
        codes = generate_codes(
            model, processor, hypothesis, row["tgt_lang"],
            layout=layout, output_mask=output_mask, device=model.device,
            max_steps=args.max_steps, temperature=args.temperature,
            repetition_penalty=args.repetition_penalty,
            global_embed=global_embed,
            bicodec_global=(
                None if global_embed is None
                else torch.as_tensor(row["bicodec_global"], dtype=torch.long)
            ),
        )
        index = args.shard + position * args.num_shards
        name = f"{index:03d}_{row['src_lang']}2{row['tgt_lang']}"
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
        records.append(
            {
                "index": index, "name": name, "id": row["id"],
                "src_lang": row["src_lang"], "tgt_lang": row["tgt_lang"],
                "source_audio_path": row["source_audio_path"],
                "reference_text": row["translation"],
                "translated_text": hypothesis,
                "codes": len(codes), "gold_codes": len(row["target_bicodec"]),
                "audio_path": audio_path,
            }
        )
        print(f"  {name}  译文: {hypothesis[:48]}", flush=True)

    with (out_dir / f"cascade_{args.shard}.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"-> {out_dir}")


if __name__ == "__main__":
    main()

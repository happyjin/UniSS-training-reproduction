"""Stage 2 end to end: source audio in, translated speech out, one model.

Stage 1's evaluator cascades -- the Thinker translates to text, the text is
re-templated as a TTS prompt, and the Talker reads it back. Stage 2 trains
the two together on the S2ST task, so the Talker consumes the hidden states
of the reply the Thinker actually produced. Re-encoding the text would
throw those away and measure a cascade the model was no longer trained as.

The reply is generated first and then run through the Thinker once more, so
the hidden states come from a forward over prompt + reply exactly as in
training. That second forward is the price of using generated text where
training had gold text; it costs one prompt-length pass per utterance.

Quality mode, so the reply is ``transcript\\ntranslation``. The transcript
is the model planning through what was said; the translation is the last
line, and that is what Text-BLEU scores and what the speech should say.

Nothing here is streaming. This is the offline number, comparable to the
Stage 1 Table 1 run.
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
from experiments.uniss_omni_s2st_v1.training import stage2_tasks
from experiments.uniss_streaming_p2st_traj_v1.training import lora

LANG_KEY = {"eng": "en", "cmn": "zh", "en": "en", "zh": "zh"}


def split_reply(reply: str) -> tuple[str, str]:
    """Quality mode's reply is transcript then translation, one per line.

    Training's target is exactly ``f"{transcript}\n{translation}"`` with a
    single-line transcript, so the translation is the second line -- not
    the last. The two agree on a well-formed reply and disagree when the
    model runs on, which it does: a reply cut off mid-translation came
    back as three lines, and taking the last one scored a fragment of a
    retry instead of the translation.
    """
    lines = [line.strip() for line in reply.splitlines() if line.strip()]
    if not lines:
        return "", ""
    if len(lines) == 1:
        return "", lines[0]
    return lines[0], lines[1]


def translation_start(tokenizer, ids: list[int]) -> int:
    """Index in ``ids`` where the translation begins.

    Quality mode's reply is transcript, newline, translation. The newline
    may be its own token or merged into a neighbour, so the boundary is
    found by decoding prefixes rather than by searching for a token id.
    Returns 0 when there is no newline -- a reply with no transcript is
    all translation.
    """
    for i in range(1, len(ids) + 1):
        if "\n" in tokenizer.decode(ids[:i]):
            return i
    return 0


@torch.no_grad()
def speak(model, processor, audio_path, src_lang, tgt_lang, *, max_new_tokens,
          talker_text: str = "full"):
    """Generate the reply, then the hidden states the Talker reads.

    ``talker_text`` picks which of those states the Talker gets.

    "full" is what Stage 2 trained on: the whole reply, transcript first.
    build_text_stream lays one state per codec step from step 0, so the
    first ~16 steps -- the opening third of a second of every utterance --
    are conditioned on the source-language transcript while the Talker is
    supposed to be speaking the translation.

    "translation" starts the stream at the translation instead, which is
    the text the codes actually correspond to. It is a diagnostic: the
    model was not trained this way, so it measures whether that onset
    mismatch is what costs Stage 2 its intelligibility.
    """
    waveform, _ = sf.read(audio_path, dtype="float32")
    sample = stage2_tasks.build(
        "s2st",
        {
            "audio": audio_path,
            "source_lang": src_lang,
            "target_lang": tgt_lang,
            "source_transcript": "",
            "target_text": "",
            "codes": None,
            "globals": None,
        },
    )
    text = processor.apply_chat_template(
        [stage2_tasks.conversation(sample)], add_generation_prompt=True, tokenize=False
    )
    inputs = processor(
        text=text, audio=[waveform], sampling_rate=16_000, return_tensors="pt"
    )
    inputs = {k: (v.to(model.device) if hasattr(v, "to") else v) for k, v in inputs.items()}
    prompt_length = inputs["input_ids"].shape[1]
    generated = model.generate(
        **inputs, thinker_max_new_tokens=max_new_tokens,
        thinker_do_sample=False, return_audio=False,
    )
    reply_ids = generated[0][prompt_length:]
    tokenizer = processor.tokenizer
    reply = tokenizer.decode(reply_ids, skip_special_tokens=True).strip()

    # The Talker was trained on the states of the reply tokens alone, so the
    # trailing <|im_end|> has to go: it is in the generated ids but was never
    # in the gold reply.
    keep = [int(t) for t in reply_ids if int(t) not in tokenizer.all_special_ids]
    if not keep:
        return reply, None
    ids = torch.cat(
        [inputs["input_ids"][0], torch.tensor(keep, device=model.device)]
    ).unsqueeze(0)
    forward = {"input_ids": ids, "output_hidden_states": True, "return_dict": True}
    if "input_features" in inputs:
        forward["input_features"] = inputs["input_features"]
        forward["feature_attention_mask"] = inputs["feature_attention_mask"]
    out = model.thinker(**forward)
    hidden = out.hidden_states[-1][:, -len(keep) :, :]
    embeds = model.thinker.get_input_embeddings()(
        torch.tensor([keep], device=model.device)
    )
    if talker_text == "translation":
        start = translation_start(tokenizer, keep)
        if start and start < len(keep):
            hidden, embeds = hidden[:, start:, :], embeds[:, start:, :]
    return reply, (hidden, embeds)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="pretrained_models/Qwen2.5-Omni-3B")
    ap.add_argument("--checkpoint", required=True, help="export_stage2_from_megatron output")
    ap.add_argument("--dev-parquet", nargs="+", required=True)
    ap.add_argument("--subset", default=None)
    ap.add_argument("--speech-tokenizer", default="pretrained_models/UniSS")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--max-steps", type=int, default=700)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--repetition-penalty", type=float, default=1.3)
    ap.add_argument("--lora-rank", type=int, default=32)
    ap.add_argument("--lora-alpha", type=float, default=64.0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--talker-text", choices=("full", "translation"), default="full")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    import pyarrow.parquet as pq
    from transformers import Qwen2_5OmniForConditionalGeneration
    from uniss import UniSSTokenizer

    wanted = None
    if args.subset:
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
                "reference_audio_path", "synthetic_source", "synthetic_reference",
                "source_audio_duration_seconds", "reference_audio_duration_seconds",
            ],
        )
        rows.extend(r for r in table.to_pylist() if wanted is None or r["id"] in wanted)
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
    if "thinker_lora" not in state:
        raise SystemExit(
            f"{args.checkpoint} has no thinker_lora; that is a Stage 1 export."
            " Use cascade_s2st.py for those."
        )
    layers = lora.inject(
        model.thinker, rank=args.lora_rank, alpha=args.lora_alpha,
        targets=stage2_tasks.THINKER_TARGETS,
    )
    lora.load_state_dict(layers, state["thinker_lora"])
    # inject() freezes everything it walked, which here includes the Thinker
    # it is wrapping. Harmless under no_grad, but the adapters have to be in
    # the model's dtype or the matmul raises.
    for layer in layers:
        layer.to(device=model.device, dtype=torch.bfloat16)
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
        reply, states = speak(
            model, processor, row["source_audio_path"],
            row["src_lang"], row["tgt_lang"], max_new_tokens=args.max_new_tokens,
            talker_text=args.talker_text,
        )
        transcript, translation = split_reply(reply)
        codes = []
        if states is not None:
            codes = generate_codes(
                model, processor, translation, row["tgt_lang"],
                layout=layout, output_mask=output_mask, device=model.device,
                max_steps=args.max_steps, temperature=args.temperature,
                repetition_penalty=args.repetition_penalty,
                global_embed=global_embed,
                bicodec_global=(
                    None if global_embed is None
                    else torch.as_tensor(row["bicodec_global"], dtype=torch.long)
                ),
                reply_states=states,
            )
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
        records.append(
            {
                "index": index, "name": name, "id": row["id"],
                "mode": "tts",
                "src_lang": row["src_lang"], "tgt_lang": row["tgt_lang"],
                "dataset_name": "CVSS-T",
                "transcription_ref": row["transcription"],
                "translation_ref": row["translation"],
                "generated_text_raw": reply,
                "generated_text_clean": translation,
                "generated_transcript": transcript,
                "source_audio_path": row["source_audio_path"],
                "reference_audio_path": row["reference_audio_path"],
                "source_audio_duration_seconds": row["source_audio_duration_seconds"],
                "reference_audio_duration_seconds": row["reference_audio_duration_seconds"],
                "synthetic_source": bool(row["synthetic_source"]),
                "synthetic_reference": bool(row["synthetic_reference"]),
                "reference_text": row["translation"],
                "translated_text": translation,
                "codes": len(codes), "gold_codes": len(row["target_bicodec"]),
                "audio_path": audio_path, "error": None,
            }
        )
        if position % 25 == 0:
            print(f"  {name}  译文: {translation[:48]}", flush=True)

    with (out_dir / f"stage2_{args.shard}.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"-> {out_dir}")


if __name__ == "__main__":
    main()

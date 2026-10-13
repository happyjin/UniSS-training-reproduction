"""Phase D end to end: source audio in, translated speech out, one model.

Every Phase D number so far is TTS mode -- gold text in, so it measures
the Talker's fidelity with the translation error removed. That was the
right gauge while the question was whether the Talker could read Qwen3 at
all. It is not the deliverable.

This is the deliverable. Qwen3's Thinker hears the source and writes the
translation; the states it wrote it from go straight to the Talker, with
no text string in between. That matters beyond tidiness: Phase B's 27.68
and 34.14 came from a cascade that has to hold the whole translation
before it can speak, and cannot stream. This path can.

Both halves run in one process here, which is only possible because
transformers 5.19 carries Qwen2.5-Omni as well -- Phase B needed two
environments and a file between them.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import soundfile as sf
import torch

from experiments.uniss_omni_s2st_v1.evaluation.stage1_generate import generate_codes
from experiments.uniss_qwen3_omni_v1.modeling.qwen3_assembly import build_assembly

INSTRUCTION = {
    "eng": "Listen to the Chinese speech and translate it into English. "
           "Reply with the English translation only.",
    "cmn": "Listen to the English speech and translate it into Chinese. "
           "Reply with the Chinese translation only.",
}


@torch.no_grad()
def translate_and_state(a, audio_path, tgt_lang, max_new_tokens):
    """The translation, and the states the Thinker wrote it from."""
    waveform, _ = sf.read(audio_path, dtype="float32")
    conversation = [{"role": "user", "content": [
        {"type": "audio", "audio": audio_path},
        {"type": "text", "text": INSTRUCTION[tgt_lang]},
    ]}]
    text = a.processor.apply_chat_template(
        [conversation], add_generation_prompt=True, tokenize=False)
    inputs = a.processor(text=text, audio=[waveform], sampling_rate=16_000,
                         return_tensors="pt")
    inputs = {k: (v.to(a.device) if hasattr(v, "to") else v) for k, v in inputs.items()}
    prompt_length = inputs["input_ids"].shape[1]
    tok = a.tokenizer
    # eos_token_id explicitly, because thinker_config.text_config has it as
    # None: the top-level Qwen3OmniMoe generate supplies one, and calling
    # the Thinker's own generate goes around that. Without it nothing ever
    # stops and every reply runs to max_new_tokens -- measured at 135
    # tokens against Phase A's 20.7, which cost 17 of the 32 seconds an
    # utterance took and then made the Talker speak the overrun until it
    # hit its own 700-step ceiling. Same shape as the stage 1 bug where
    # max_new_tokens was silently overridden and a 26.65 became 13.91.
    out = a.thinker.generate(
        **inputs, max_new_tokens=max_new_tokens, do_sample=False,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id,
    )
    reply_ids = out[0][prompt_length:]
    hypothesis = tok.decode(reply_ids, skip_special_tokens=True).strip()
    # The Talker was trained on the states of content tokens; the trailing
    # turn marker was never in a gold reply.
    keep = [int(t) for t in reply_ids if int(t) not in tok.all_special_ids]
    if not keep:
        return hypothesis, None
    ids = torch.cat([inputs["input_ids"][0],
                     torch.tensor(keep, device=a.device)]).unsqueeze(0)
    forward = {"input_ids": ids, "output_hidden_states": True, "return_dict": True}
    for k in ("input_features", "feature_attention_mask"):
        if k in inputs:
            forward[k] = inputs[k]
    states = a.thinker(**forward)
    hidden = states.hidden_states[-1][:, -len(keep):, :]
    embeds = a.thinker.get_input_embeddings()(torch.tensor([keep], device=a.device))
    return hypothesis, (hidden, embeds)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--talker-checkpoint", required=True)
    ap.add_argument("--phase-d-checkpoint", required=True)
    ap.add_argument("--dev-parquet", required=True)
    ap.add_argument("--subset", default=None)
    ap.add_argument("--speech-tokenizer", default="pretrained_models/UniSS")
    ap.add_argument("--limit", type=int, default=0)
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
    from uniss import UniSSTokenizer

    wanted = None
    if args.subset:
        wanted = {json.loads(l)["id"] for l in
                  Path(args.subset).read_text(encoding="utf-8").splitlines() if l.strip()}
    rows = pq.read_table(args.dev_parquet, columns=[
        "id", "translation", "transcription", "src_lang", "tgt_lang",
        "bicodec_global", "target_bicodec", "source_audio_path",
        "reference_audio_path", "synthetic_source", "synthetic_reference",
        "source_audio_duration_seconds", "reference_audio_duration_seconds",
    ]).to_pylist()
    if wanted is not None:
        rows = [r for r in rows if r["id"] in wanted]
    rows.sort(key=lambda r: (r["id"], r["tgt_lang"]))
    if args.limit:
        rows = rows[: args.limit]
    rows = rows[args.shard :: args.num_shards]
    print(f"{len(rows)} utterances", flush=True)

    a = build_assembly(talker_checkpoint=args.talker_checkpoint, device=args.device)
    state = torch.load(args.phase_d_checkpoint, map_location="cpu", weights_only=False)
    a.talker.load_state_dict(state["talker"], strict=False)
    with torch.no_grad():
        a.global_embed.weight.copy_(state["global_embed"].to(a.global_embed.weight.dtype))
        a.markers.eos.copy_(state["markers"]["eos"])
        a.markers.pad.copy_(state["markers"]["pad"])
    a.talker.eval()

    class _Shim:
        def __init__(self, assembly):
            self.thinker = assembly.thinker
            self.talker = assembly.talker
    model = _Shim(a)
    vocoder = UniSSTokenizer.from_pretrained(
        args.speech_tokenizer, device=torch.device(args.device))
    out_dir = Path(args.output)
    (out_dir / "wav").mkdir(parents=True, exist_ok=True)

    records = []
    for position, row in enumerate(rows):
        hypothesis, states = translate_and_state(
            a, row["source_audio_path"], row["tgt_lang"], args.max_new_tokens)
        codes = []
        if states is not None:
            codes = generate_codes(
                model, a.processor, hypothesis, row["tgt_lang"],
                layout=a.layout, output_mask=a.output_mask, device=a.device,
                max_steps=args.max_steps, temperature=args.temperature,
                repetition_penalty=args.repetition_penalty,
                global_embed=a.global_embed,
                bicodec_global=torch.as_tensor(row["bicodec_global"], dtype=torch.long),
                reply_states=states,
                markers=a.markers.embeds(states[0].dtype, a.device))
        index = args.shard + position * args.num_shards
        name = f"{index:05d}_{row['src_lang']}2{row['tgt_lang']}"
        audio_path = None
        if codes:
            tokens = torch.tensor([*row["bicodec_global"], *codes],
                                  dtype=torch.long, device=a.device)
            vocoder.save_audio(vocoder.decode(tokens),
                               out_dir / "wav" / f"{name}.wav", sample_rate=16000)
            audio_path = str(out_dir / "wav" / f"{name}.wav")
        records.append({
            "index": index, "name": name, "id": row["id"], "mode": "tts",
            "src_lang": row["src_lang"], "tgt_lang": row["tgt_lang"],
            "dataset_name": "CVSS-T",
            "transcription_ref": row["transcription"],
            "translation_ref": row["translation"],
            "generated_text_raw": hypothesis, "generated_text_clean": hypothesis,
            "translated_text": hypothesis, "reference_text": row["translation"],
            "source_audio_path": row["source_audio_path"],
            "reference_audio_path": row["reference_audio_path"],
            "source_audio_duration_seconds": row["source_audio_duration_seconds"],
            "reference_audio_duration_seconds": row["reference_audio_duration_seconds"],
            "synthetic_source": bool(row["synthetic_source"]),
            "synthetic_reference": bool(row["synthetic_reference"]),
            "codes": len(codes), "gold_codes": len(row["target_bicodec"]),
            "audio_path": audio_path, "error": None,
        })
        if position % 50 == 0:
            print(f"  {position}/{len(rows)}  {hypothesis[:44]}", flush=True)
    with (out_dir / f"e2e_{args.shard}.jsonl").open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"-> {out_dir}")


if __name__ == "__main__":
    main()

"""End to end for the decoder-only route: audio in, translated speech out.

The two-stream probe cannot be reused. There is no talker to load and no
hidden states to hand across a boundary -- the thinker writes the reply
and then keeps going, emitting codes from its own sequence through a head
of its own. So the loop here is two generate passes over one model:

    1. greedy text until <|im_end|>          -> the translation
    2. append the 32 speaker slots, then     -> the BiCodec codes
       sample codes one at a time, feeding
       each back through audio_embed

Both passes share a KV cache, so the second does not re-read the audio.
That is the whole point of the route: one model, one sequence, and the
codes conditioned on everything the model heard rather than on a summary
handed over a projection.

Scored exactly as the two-stream probe is -- same 64 utterances per
direction, same vocoder, same ASR, same evaluation.text_metrics -- so the
numbers sit in the same table as phase D through H.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import soundfile as sf
import torch

from experiments.uniss_qwen3_omni_v1.training.deconly_batch import (
    BICODEC_GLOBAL_SIZE,
    CODE_EOS,
    CODE_VOCAB,
    THINKER_HIDDEN,
)

INSTRUCTION = {
    "eng": "Listen to the Chinese speech and translate it into English. "
           "Reply with the English translation only.",
    "cmn": "Listen to the English speech and translate it into Chinese. "
           "Reply with the Chinese translation only.",
}


def load_pieces(path, thinker, device):
    """Rebuild the three modules and the adapter the export carries."""
    import torch.nn as nn

    from experiments.uniss_omni_s2st_v1.training.stage2_tasks import THINKER_TARGETS
    from experiments.uniss_streaming_p2st_traj_v1.training import lora

    state = torch.load(path, map_location="cpu", weights_only=False)
    dtype = thinker.get_input_embeddings().weight.dtype
    audio_embed = nn.Embedding(CODE_VOCAB, THINKER_HIDDEN, dtype=dtype)
    global_embed = nn.Embedding(BICODEC_GLOBAL_SIZE, THINKER_HIDDEN, dtype=dtype)
    audio_head = nn.Linear(THINKER_HIDDEN, CODE_VOCAB, dtype=dtype)
    audio_embed.load_state_dict(state["audio_embed"])
    global_embed.load_state_dict(state["global_embed"])
    audio_head.load_state_dict(state["audio_head"])

    names = {k.rsplit(".", 1)[0] for k in state["thinker_lora"]}
    layers = lora.inject(
        thinker, rank=32, alpha=64.0, targets=THINKER_TARGETS,
        layer_filter=lambda n: any(x.startswith(n + ".") for x in names))
    lora.load_state_dict(layers, state["thinker_lora"])
    thinker.requires_grad_(False)
    print(f"deconly loaded: step {state.get('step')}, {len(layers)} LoRA layers",
          flush=True)
    return (audio_embed.to(device).eval(), global_embed.to(device).eval(),
            audio_head.to(device).eval())


@torch.no_grad()
def translate_and_speak(thinker, processor, pieces, audio_path, tgt_lang,
                        speaker, *, max_new_tokens, max_codes, device):
    audio_embed, global_embed, audio_head = pieces
    tok = processor.tokenizer
    waveform, _ = sf.read(audio_path, dtype="float32")
    conversation = [{"role": "user", "content": [
        {"type": "audio", "audio": audio_path},
        {"type": "text", "text": INSTRUCTION[tgt_lang]},
    ]}]
    text = processor.apply_chat_template(
        [conversation], add_generation_prompt=True, tokenize=False)
    inputs = processor(text=text, audio=[waveform], sampling_rate=16_000,
                       return_tensors="pt")
    inputs = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in inputs.items()}

    # eos_token_id explicitly: thinker_config.text_config carries None, and
    # the thinker's own generate goes around the wrapper that supplies one.
    out = thinker.generate(
        **inputs, max_new_tokens=max_new_tokens, do_sample=False,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id,
        return_dict_in_generate=True)
    sequences = out.sequences
    reply_ids = sequences[0][inputs["input_ids"].shape[1]:]
    hypothesis = tok.decode(reply_ids, skip_special_tokens=True).strip()
    if not hypothesis:
        return hypothesis, []

    # Re-forward the whole prompt plus reply once, to start the code loop
    # from a cache that saw the audio. Cheaper than it looks: it is the
    # only full pass the code side needs.
    embeds = thinker.get_input_embeddings()(sequences)
    forward = {"input_ids": sequences, "inputs_embeds": embeds,
               "use_cache": True, "return_dict": True,
               "output_hidden_states": True}
    for k in ("input_features", "feature_attention_mask"):
        if k in inputs:
            forward[k] = inputs[k]
    state = thinker(**forward)
    past = state.past_key_values

    # The 32 speaker slots go in before any code is asked for: they carry
    # the voice and the model has to see them to condition on it.
    spk = global_embed(torch.as_tensor([speaker], device=device, dtype=torch.long))
    state = thinker(inputs_embeds=spk, past_key_values=past, use_cache=True,
                    return_dict=True, output_hidden_states=True)
    past = state.past_key_values
    hidden = state.hidden_states[-1][:, -1:, :]

    codes = []
    for _ in range(max_codes):
        nxt = int(audio_head(hidden).argmax(dim=-1).flatten()[0])
        if nxt == CODE_EOS:
            break
        codes.append(nxt)
        step = audio_embed(torch.tensor([[nxt]], device=device))
        state = thinker(inputs_embeds=step, past_key_values=past, use_cache=True,
                        return_dict=True, output_hidden_states=True)
        past = state.past_key_values
        hidden = state.hidden_states[-1][:, -1:, :]
    return hypothesis, codes


def main() -> None:
    import pyarrow.parquet as pq

    from transformers import (
        Qwen3OmniMoeForConditionalGeneration,
        Qwen3OmniMoeProcessor,
    )
    from uniss import UniSSTokenizer

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--deconly-checkpoint", required=True)
    ap.add_argument("--qwen3-dir",
                    default="pretrained_models/Qwen3-Omni-30B-A3B-Instruct")
    ap.add_argument("--dev-parquet", required=True)
    ap.add_argument("--speech-tokenizer", default="pretrained_models/UniSS")
    ap.add_argument("--limit", type=int, default=64)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--max-codes", type=int, default=700)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    device = torch.device(args.device)
    # Not build_assembly: it loads Qwen2.5-Omni-3B to harvest a talker,
    # and this route has none. Eight shards would each read 7 GB from
    # disk for a module never called.
    omni3 = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        args.qwen3_dir, dtype=torch.bfloat16, attn_implementation="sdpa")
    for attr in ("talker", "code2wav", "token2wav"):
        if hasattr(omni3, attr):
            delattr(omni3, attr)
    thinker = omni3.thinker.to(device).eval()
    del omni3
    torch.cuda.empty_cache()
    processor = Qwen3OmniMoeProcessor.from_pretrained(args.qwen3_dir)
    pieces = load_pieces(args.deconly_checkpoint, thinker, device)
    vocoder = UniSSTokenizer.from_pretrained(
        args.speech_tokenizer, device=device)

    rows = pq.read_table(args.dev_parquet, columns=[
        "id", "translation", "src_lang", "tgt_lang", "bicodec_global",
        "source_audio_path",
    ]).to_pylist()
    rows.sort(key=lambda r: (r["id"], r["tgt_lang"]))
    if args.limit:
        rows = rows[: args.limit]
    rows = rows[args.shard :: args.num_shards]
    print(f"{len(rows)} utterances", flush=True)

    out_dir = Path(args.output)
    (out_dir / "wav").mkdir(parents=True, exist_ok=True)
    records = []
    for position, row in enumerate(rows):
        hypothesis, codes = translate_and_speak(
            thinker, processor, pieces, row["source_audio_path"],
            row["tgt_lang"], list(row["bicodec_global"]),
            max_new_tokens=args.max_new_tokens, max_codes=args.max_codes,
            device=device)
        index = args.shard + position * args.num_shards
        name = f"{index:05d}_{row['src_lang']}2{row['tgt_lang']}"
        audio_path = None
        if codes:
            tokens = torch.tensor([*row["bicodec_global"], *codes],
                                  dtype=torch.long, device=device)
            vocoder.save_audio(vocoder.decode(tokens),
                               out_dir / "wav" / f"{name}.wav", sample_rate=16000)
            audio_path = str(out_dir / "wav" / f"{name}.wav")
        records.append({
            "index": index, "name": name, "id": row["id"], "mode": "tts",
            "src_lang": row["src_lang"], "tgt_lang": row["tgt_lang"],
            "dataset_name": "CVSS-T", "audio_path": audio_path,
            "codes": len(codes), "translated_text": hypothesis,
            "generated_text_raw": hypothesis, "generated_text_clean": hypothesis,
            "translation_ref": row["translation"], "reference_text": row["translation"],
            "error": None if codes else "no codes",
        })
        if (position + 1) % 8 == 0:
            print(f"  {position+1}/{len(rows)}", flush=True)

    path = out_dir / f"deconly_{args.shard}.jsonl"
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n"
                            for r in records), encoding="utf-8")
    print(f"-> {path}", flush=True)


if __name__ == "__main__":
    main()

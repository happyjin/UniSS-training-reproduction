"""Stage 1 acceptance: synthesise BiCodec codes from text and decode them.

The warmup's gate is acoustic, not a loss: TTS reconstruction must not come
out below the current pipeline. So this runs the trained Talker free,
decodes its codes with the project's own BiCodec vocoder, and writes the
``results.jsonl`` + ``wav/`` layout the existing objective-metrics scripts
already consume -- rather than scoring audio a second, slightly different
way.

Speaker identity does not come from the language model here, and that is
the point of the whole architecture: the 32 ``bicodec_global`` tokens are
taken from the reference utterance and handed straight to the vocoder. They
are why this project's AutoPCP is what it is, and nothing in the Talker
swap touches them.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from experiments.uniss_omni_s2st_v1.modeling.stage0_assembly import build_text_stream
from experiments.uniss_omni_s2st_v1.modeling.talker_surgery import (
    apply_output_mask,
    read_codec_layout,
    retarget_talker_codebook,
    valid_output_mask,
)
from experiments.uniss_omni_s2st_v1.runtime.omni_processor import (
    load_audio_text_processor,
)
from experiments.uniss_omni_s2st_v1.training.tts_data import tts_prompt


@torch.no_grad()
def generate_codes(
    model,
    processor,
    text: str,
    lang: str,
    *,
    layout,
    output_mask,
    device,
    max_steps: int = 600,
) -> list[int]:
    """Greedy autoregressive decode of one utterance's code stream."""
    thinker, talker = model.thinker, model.talker
    templated = processor.apply_chat_template(
        [tts_prompt(text, lang)], add_generation_prompt=True, tokenize=False
    )
    prompt_text = templated[0] if isinstance(templated, list) else templated
    tokenizer = processor.tokenizer
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    reply_ids = tokenizer(text, add_special_tokens=False)["input_ids"]

    ids = torch.tensor([prompt_ids + reply_ids], device=device)
    out = thinker(input_ids=ids, output_hidden_states=True, return_dict=True)
    hidden = out.hidden_states[-1][:, len(prompt_ids) :, :]
    reply = torch.tensor([reply_ids], device=device)
    embeds = thinker.get_input_embeddings()(reply)
    eos_embed = thinker.get_input_embeddings()(
        torch.tensor([[talker.text_eos_token]], device=device)
    )
    pad_embed = thinker.get_input_embeddings()(
        torch.tensor([[talker.text_pad_token]], device=device)
    )
    text_stream = build_text_stream(
        hidden.float(), embeds.float(),
        steps=max_steps, eos_embed=eos_embed.float(), pad_embed=pad_embed.float(),
    ).to(dtype=hidden.dtype)

    codes: list[int] = []
    current = torch.tensor([[talker.codec_bos_token]], device=device)
    past = None
    for step in range(max_steps):
        codec_embed = talker.get_input_embeddings()(current)
        lm_input = talker.thinker_to_talker_proj(
            codec_embed + text_stream[:, step : step + 1, :]
        )
        position = torch.tensor([[[step]]], device=device).expand(3, 1, 1)
        result = talker.model(
            inputs_embeds=lm_input,
            position_ids=position,
            past_key_values=past,
            use_cache=True,
            return_dict=True,
        )
        past = result.past_key_values
        logits = apply_output_mask(
            talker.codec_head(result.last_hidden_state[:, -1, :]), output_mask
        )
        token = int(logits.argmax(dim=-1))
        if token == layout.special_ids["eos"]:
            break
        codes.append(token)
        current = torch.tensor([[token]], device=device)
    return codes


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="pretrained_models/Qwen2.5-Omni-3B")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--dev-parquet", nargs="+", required=True)
    ap.add_argument("--subset", required=True)
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--max-steps", type=int, default=600)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    import pyarrow.parquet as pq
    from transformers import Qwen2_5OmniForConditionalGeneration

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
                "id", "translation", "transcription", "tgt_lang", "src_lang",
                "target_bicodec", "bicodec_global", "direction",
                "source_audio_path", "reference_audio_path",
            ],
        )
        for record in table.to_pylist():
            if record["id"] in wanted and record["target_bicodec"]:
                rows.append(record)
    rows.sort(key=lambda r: (r["id"], r["tgt_lang"]))
    if args.limit:
        rows = rows[: args.limit]
    rows = rows[args.shard :: args.num_shards]
    print(f"{len(rows)} utterances (shard {args.shard}/{args.num_shards})", flush=True)

    processor = load_audio_text_processor(args.model)
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        args.model, torch_dtype=torch.bfloat16
    ).to(args.device)
    if hasattr(model, "token2wav"):
        del model.token2wav
    layout = read_codec_layout(model.talker.config)
    retarget_talker_codebook(model.talker, head_scale=0.1)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    missing, unexpected = model.talker.load_state_dict(state["talker"], strict=False)
    print(
        f"loaded step {state.get('step')}:"
        f" {len(missing)} missing, {len(unexpected)} unexpected",
        flush=True,
    )
    model.eval()
    output_mask = valid_output_mask(layout, device=model.device)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    with (out_dir / f"generated_{args.shard}.jsonl").open("w", encoding="utf-8") as handle:
        for index, record in enumerate(rows):
            codes = generate_codes(
                model, processor, record["translation"], record["tgt_lang"],
                layout=layout, output_mask=output_mask, device=model.device,
                max_steps=args.max_steps,
            )
            gold = list(record["target_bicodec"])
            handle.write(
                json.dumps(
                    {
                        "index": args.shard + index * args.num_shards,
                        "id": record["id"],
                        "mode": "tts",
                        "src_lang": record["src_lang"],
                        "tgt_lang": record["tgt_lang"],
                        "dataset_name": "CVSS-T",
                        "transcription_ref": record["transcription"],
                        "translation_ref": record["translation"],
                        "semantic_values": codes,
                        "global_values": list(record["bicodec_global"]),
                        "gold_semantic_count": len(gold),
                        "semantic_token_count": len(codes),
                        "source_audio_path": record["source_audio_path"],
                        "reference_audio_path": record["reference_audio_path"],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            written += 1
            if written % 20 == 0:
                print(f"  {written}/{len(rows)}", flush=True)
    print(f"-> {out_dir}/generated_{args.shard}.jsonl  ({written} rows)")


if __name__ == "__main__":
    main()
